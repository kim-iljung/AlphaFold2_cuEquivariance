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
    """Column-wise global attention that emits a single query per column."""
    def __init__(self, c=8, c_h=1, n_head=8):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)
        self.proj_q = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.proj_k = torch.nn.Linear(c, c_h, bias=False)
        self.proj_v = torch.nn.Linear(c, c_h, bias=False)
        self.proj_g = torch.nn.Linear(c, c_h * n_head)
        self.proj_o = torch.nn.Linear(c_h * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m, mask=None):
        """Compute column-wise attention over an MSA embedding."""
        is_batched = (m.dim() == 4)
        if not is_batched:
            m = m.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        B, s, i, c = m.shape

        if mask is None:
            mask = torch.ones((B, s, i), dtype=m.dtype, device=m.device)

        m_col = m.transpose(-2, -3)
        mask_col = mask.transpose(-1, -2)

        m_norm = self.layer_norm(m_col)

        denom = mask_col.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        q_global = (m_norm * mask_col.unsqueeze(-1)).sum(dim=-2) / denom

        q = self.proj_q(q_global)
        k = self.proj_k(m_norm)
        v = self.proj_v(m_norm)
        g = self.sigmoid(self.proj_g(m_norm))

        H, C_h = self.n_head, self.c_h

        q = q.view(B, i, H, 1, C_h)

        k = k.unsqueeze(-3).expand(B, i, H, s, C_h)
        v = v.unsqueeze(-3).expand(B, i, H, s, C_h)

        bias = torch.zeros((B, 1, H, 1, s), dtype=torch.float32, device=m.device)

        attn_mask = mask_col.to(dtype=torch.bool).unsqueeze(-2).unsqueeze(-2)

        o = triangle_attention(
            q=q, k=k, v=v,
            bias=bias,
            mask=attn_mask,
            scale=None,
        )                                                # (B, i, H, 1, C_h)

        o = o.squeeze(-2).unsqueeze(-3)                 # (B, i, 1, H, C_h)
        g = g.view(B, i, s, H, C_h)                     # (B, i, s, H, C_h)
        o = (o * g).reshape(B, i, s, H * C_h)           # (B, i, s, H*C_h)

        o = self.proj_o(o)                              # (B, i, s, c)
        o = o.transpose(-2, -3)                         # (B, s, i, c)

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
        gate = gate.transpose(1, 2)

        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        gate = gate.permute(0, 1, 3, 2, 4)

        mask_values = torch.ones((B, i, s), dtype=q.dtype, device=m.device)
        attn_mask = None
        if msa_mask is not None:
            mask_keys = msa_mask.transpose(1, 2).bool()
            mask_values = mask_keys.to(dtype=q.dtype)
            attn_mask = mask_keys.unsqueeze(-2) & mask_keys.unsqueeze(-1)
            attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, h, -1, -1)

        q = q * mask_values.unsqueeze(2).unsqueeze(-1)
        k = k * mask_values.unsqueeze(2).unsqueeze(-1)
        v = v * mask_values.unsqueeze(2).unsqueeze(-1)
        gate = gate * mask_values.unsqueeze(2).unsqueeze(-1)

        bias = torch.zeros(B, 1, h, i, s, dtype=q.dtype, device=q.device)

        o = triangle_attention(
            q,
            k,
            v,
            bias=bias,
            mask=attn_mask,
        )

        o = o * gate
        o = o * mask_values.unsqueeze(2).unsqueeze(-1)

        o = o.permute(0, 1, 3, 2, 4)
        o = o.reshape(B, i, s, h * c_h)
        o = self.proj_o(o)
        o = o.transpose(1, 2)

        if not is_batched:
            o = o.squeeze(0)

        return o
