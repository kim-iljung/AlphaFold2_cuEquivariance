import torch

from cuequivariance_torch import triangle_attention


class Linear(torch.nn.Linear):
    """Linear layer wrapper that preserves precision for bfloat16 inputs."""

    def __init__(self, in_dim, out_dim, bias=True, init="default"):
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dtype is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
                return torch.nn.functional.linear(input, self.weight.to(dtype=input.dtype), bias)

        return torch.nn.functional.linear(input, self.weight, self.bias)


class LayerNorm(torch.nn.Module):
    """LayerNorm that keeps numerics stable for bfloat16 activations."""

    def __init__(self, c_in: int, eps: float = 1e-5):
        super().__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(c_in))
        self.bias = torch.nn.Parameter(torch.zeros(c_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                return torch.nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=x.dtype),
                    self.bias.to(dtype=x.dtype),
                    self.eps,
                )

        return torch.nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )


class MSARowAttentionWithPairBias(torch.nn.Module):
    """Row-wise gated MSA attention with pair bias."""

    def __init__(self, c_m: int = 256, c_z: int = 128, c_h: int = 4, n_head: int = 8):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = LayerNorm(c_m)
        self.layer_norm_b = LayerNorm(c_z)
        self.proj_q = Linear(c_m, c_h * n_head, bias=False)
        self.proj_k = Linear(c_m, c_h * n_head, bias=False)
        self.proj_v = Linear(c_m, c_h * n_head, bias=False)
        self.proj_b = Linear(c_z, n_head, bias=False)
        self.proj_g = Linear(c_m, c_h * n_head)
        self.proj_o = Linear(c_h * n_head, c_m)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply row-wise pair-biased attention to an MSA embedding."""
        is_batched = m.dim() == 4
        if not is_batched:
            m = m.unsqueeze(0)
            z = z.unsqueeze(0)
            if msa_mask is not None:
                msa_mask = msa_mask.unsqueeze(0)

        m_norm = self.layer_norm(m)
        z_norm = self.layer_norm_b(z)

        q = self.proj_q(m_norm)
        k = self.proj_k(m_norm)
        v = self.proj_v(m_norm)
        b = self.proj_b(z_norm)
        gate = self.sigmoid(self.proj_g(m))

        B, s, i, _ = m.shape
        h, c_h = self.n_head, self.c_h

        q = q.view(B, s, i, h, c_h)
        k = k.view(B, s, i, h, c_h)
        v = v.view(B, s, i, h, c_h)

        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        b = b.permute(0, 3, 1, 2).unsqueeze(1)

        attn_mask = None
        if msa_mask is not None:
            mask_2d = msa_mask.unsqueeze(-1) * msa_mask.unsqueeze(-2)
            attn_mask = mask_2d.bool()

        o = triangle_attention(
            q,
            k,
            v,
            bias=b,
            mask=attn_mask,
        )

        o = o.permute(0, 1, 3, 2, 4)
        o = o.reshape(B, s, i, h * c_h)

        o = o * gate
        o = self.proj_o(o)

        if not is_batched:
            o = o.squeeze(0)

        return o


class MSAColumnGlobalAttention(torch.nn.Module):
    """Column-wise global MSA attention accelerated by the fused kernel."""

    def __init__(self, c: int = 8, c_h: int = 1, n_head: int = 8):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = LayerNorm(c)
        self.proj_q = Linear(c, c_h * n_head, bias=False)
        self.proj_k = Linear(c, c_h, bias=False)
        self.proj_v = Linear(c, c_h, bias=False)
        self.proj_g = Linear(c, c_h * n_head)
        self.proj_o = Linear(c_h * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m: torch.Tensor, msa_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply column-wise gated attention to an MSA embedding."""
        is_batched = m.dim() == 4
        if not is_batched:
            m = m.unsqueeze(0)
            if msa_mask is not None:
                msa_mask = msa_mask.unsqueeze(0)

        if msa_mask is None:
            msa_mask = torch.ones(m.shape[:-1], dtype=m.dtype, device=m.device)

        m = m.transpose(-2, -3)
        msa_mask = msa_mask.transpose(-1, -2)

        m_norm = self.layer_norm(m)

        weights = msa_mask.unsqueeze(-1)
        denom = torch.sum(msa_mask, dim=-1, keepdim=True) + 1e-5
        q_global = torch.sum(m_norm * weights, dim=-2) / denom

        q = self.proj_q(q_global)
        k = self.proj_k(m_norm)
        v = self.proj_v(m_norm)
        gate = self.sigmoid(self.proj_g(m_norm))

        B, i, s, _ = m.shape
        h, c_h = self.n_head, self.c_h

        q = q.view(B, i, h, c_h).unsqueeze(-2).repeat(1, 1, 1, s, 1)
        k = k.unsqueeze(-3).repeat(1, 1, h, 1, 1)
        v = v.unsqueeze(-3).repeat(1, 1, h, 1, 1)
        gate = gate.view(B, i, s, h, c_h)

        mask_keys = msa_mask.bool()
        attn_mask = mask_keys.unsqueeze(-2) & mask_keys.unsqueeze(-1)
        attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, h, -1, -1)

        q = q * mask_keys.unsqueeze(-1).unsqueeze(-1)

        o = triangle_attention(
            q,
            k,
            v,
            mask=attn_mask,
        )

        o = o.permute(0, 1, 3, 2, 4)
        o = o * gate * mask_keys.unsqueeze(-1).unsqueeze(-1)
        o = o.reshape(B, i, s, h * c_h)
        o = self.proj_o(o)
        o = o.transpose(-2, -3)

        if not is_batched:
            o = o.squeeze(0)

        return o


class MSAColumnAttention(torch.nn.Module):
    """Column-wise gated MSA attention implemented with the fused kernel."""

    def __init__(self, c: int = 32, c_h: int = 4, n_head: int = 8):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = LayerNorm(c)

        self.proj_q = Linear(c, c_h * n_head, bias=False)
        self.proj_k = Linear(c, c_h * n_head, bias=False)
        self.proj_v = Linear(c, c_h * n_head, bias=False)
        self.proj_g = Linear(c, c_h * n_head)
        self.proj_o = Linear(c_h * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m: torch.Tensor, msa_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply column-wise gated attention to an MSA embedding."""
        is_batched = m.dim() == 4
        if not is_batched:
            m = m.unsqueeze(0)
            if msa_mask is not None:
                msa_mask = msa_mask.unsqueeze(0)

        m_norm = self.layer_norm(m)

        q = self.proj_q(m_norm)
        k = self.proj_k(m_norm)
        v = self.proj_v(m_norm)
        gate = self.sigmoid(self.proj_g(m_norm))

        B, s, i, _ = m.shape
        h, c_h = self.n_head, self.c_h

        q = q.view(B, s, i, h, c_h)
        k = k.view(B, s, i, h, c_h)
        v = v.view(B, s, i, h, c_h)
        gate = gate.view(B, s, i, h, c_h)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        attn_mask = None
        if msa_mask is not None:
            mask_T = msa_mask.transpose(1, 2)
            mask_2d = mask_T.unsqueeze(-1) * mask_T.unsqueeze(-2)
            attn_mask = mask_2d.bool()
            attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, h, -1, -1)

        o = triangle_attention(
            q,
            k,
            v,
            mask=attn_mask,
        )

        o = o.permute(0, 1, 3, 2, 4)
        o = o.transpose(1, 2)

        o = o.reshape(B, s, i, h * c_h)
        gate = gate.reshape(B, s, i, h * c_h)

        o = o * gate
        o = self.proj_o(o)

        if not is_batched:
            o = o.squeeze(0)

        return o
