import torch
import time
import math
import opt_einsum as oe

from torch.utils.checkpoint import checkpoint
from cuequivariance_torch import triangle_attention
from module.chunked_attention import _attention_chunked_trainable, permute_final_dims

class Linear(torch.nn.Linear):
    """Linear layer wrapper that preserves precision for bfloat16 inputs.

    Accepts tensors of shape ``(..., in_dim)`` and returns
    ``(..., out_dim)`` while ensuring the computation is executed in full
    precision on bfloat16 activations.
    """

    def __init__(self, in_dim, out_dim, bias=True,init="default"):
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        d = input.dtype
        if d is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                bias = self.bias.to(dtype=d) if self.bias is not None else None
                return torch.nn.functional.linear(input, self.weight.to(dtype=d), bias)

        return torch.nn.functional.linear(input, self.weight, self.bias)


class LayerNorm(torch.nn.Module):
    """LayerNorm that keeps numerics stable for bfloat16 activations.

    Expects input with trailing dimension ``c_in`` and returns a tensor with
    identical shape after applying affine normalization.
    """

    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()
        
        self.c_in = (c_in,)
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.ones(c_in))
        self.bias = torch.nn.Parameter(torch.zeros(c_in))

    def forward(self, x): 
        d = x.dtype
        if d is torch.bfloat16:
            with torch.cuda.amp.autocast(enabled=False):
                out = torch.nn.functional.layer_norm(
                    x, 
                    self.c_in, 
                    self.weight.to(dtype=d), 
                    self.bias.to(dtype=d), 
                    self.eps
                )
        else:
            out = torch.nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out

def softmax_no_cast(t, dim=-1):
    d = t.dtype
    if d is torch.bfloat16:
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


class MSARowAttentionWithPairBias(torch.nn.Module):
    """Row-wise gated MSA attention with pair bias.

    Args:
        c_m: MSA embedding channel dimension.
        c_z: Pair embedding channel dimension.
        c_h: Per-head channel dimension.
        n_head: Number of attention heads.

    The module consumes an MSA representation ``m`` of shape ``(B, s, i, c)``
    (or ``(s, i, c)`` without batching), a pair embedding ``z`` with shape
    ``(B, i, j, c)`` and an optional mask ``msa_mask`` shaped ``(B, s, i)``. The
    returned tensor matches the leading dimensions of ``m``.
    """

    def __init__(self, c_m=256, c_z=128, c_h=4, n_head=8):
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

    def forward(self, m, z, msa_mask=None):
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
            q, k, v,
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

class MSARowAttentionWithPairBias_ckpt(torch.nn.Module):
    """Checkpointed variant of row-wise pair-biased MSA attention.

    Shares the same input and output contracts as
    :class:`MSARowAttentionWithPairBias` but uses activation checkpointing to
    reduce memory usage during training.
    """
    def __init__(self, c_m=256, c_z=128, c_h=4, n_head=8):
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

    def get_qkv(self, m, z):
        m = self.layer_norm(m)

        q, k, v = self.proj_q(m), self.proj_k(m), self.proj_v(m)
        gate = self.sigmoid(self.proj_g(m))
        b = self.proj_b(self.layer_norm_b(z)) # (B, i, j, h)

        q_shape = q.shape[:-1]
        h, c_h = self.n_head, self.c_h

        q = q.reshape(*q_shape, h, c_h) # (B, s, i, h, c)
        k = k.reshape(*q_shape, h, c_h)
        v = v.reshape(*q_shape, h, c_h)

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)    
        q = q / math.sqrt(c_h)

        b = permute_final_dims(b, (2, 0, 1)).unsqueeze(-4)

        gate = gate.reshape(*q_shape, h, c_h)

        return q, k, v, b, gate


    def get_a(self, q, k, b, msa_mask):
        h, c_h = self.n_head, self.c_h
        a = oe.contract("... s i h c, ... s j h c -> ... s i j h", q, k) / (c_h**0.5) # (B, s, i, j, h)
        if msa_mask != None:
            a = a + (1e9 * (msa_mask-1))[..., :, None, :, None]
        a = a + b
        
        return a
    
    def get_sm(self, a):
        return softmax_no_cast(a, dim=-2)

    def get_o(self, a, v, gate):
        h, c_h = self.n_head, self.c_h
        o = gate * oe.contract("... s i j h, ... s j h c -> ... s i h c", a, v) # (B, s, i, h, c)
        o = o.reshape(*o.shape[:-2], h * c_h) # (B, s, i, h * c)
        o = self.proj_o(o)

        return o
    
    def wrap(self, o, gate):
        h, c_h = self.n_head, self.c_h
        o = gate * o
        o = o.reshape(*o.shape[:-2], h * c_h)
        o = self.proj_o(o)
        return o


    def forward(self, m, z, msa_mask=None):
        """
        Algorithm 7: MSA row-wise gated self-attention with pair bias

        m: (B, s, i, c)
        z: (B, i, j, c)
        msa_mask: (B, s, i)

        return: (B, s, i, c)
        """

        if torch.is_grad_enabled():
            q, k, v, b, gate = checkpoint(self.get_qkv, m, z)
        else:
            q, k, v, b, gate = self.get_qkv(m, z)

        if msa_mask == None:
            msa_mask = m.new_ones(m.shape[:-1])
        
        if torch.is_grad_enabled():
            o = _attention_chunked_trainable(
                    query=q, 
                    key=k, 
                    value=v, 
                    biases=[(1e9 * (msa_mask-1))[..., :, None, :, None], b], 
                    chunk_size=4, 
                    chunk_dim=-4,
                    checkpoint_flag=True,
                )
        else:
            o = _attention_chunked_trainable(
                    query=q, 
                    key=k, 
                    value=v, 
                    biases=[(1e9 * (msa_mask-1))[..., :, None, :, None], b], 
                    chunk_size=4, 
                    chunk_dim=-4,
                    checkpoint_flag=False,
                )
        
        if torch.is_grad_enabled():
            o = checkpoint(self.wrap, o, gate)
        else:
            o = self.wrap(o, gate)
            del q, k, v, b, gate

        return o # (B, s, i, c)



class MSAColumnGlobalAttention(torch.nn.Module):
    """Global column-wise MSA attention used during inference.

    Operates on an MSA embedding ``m`` of shape ``(B, s, i, c)`` (or without a
    batch dimension) alongside an optional mask ``mask`` shaped ``(B, s, i)``.
    Returns an updated MSA embedding with the same leading dimensions.
    """
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
        """
        Algorithm 19: MSA global column-wise gated self-attention

        m: (B, s, i, c)

        return: (B, s, i, c)
        """

        if mask is None:
            mask = torch.ones(m.shape[:-1], dtype=m.dtype, device=m.device).detach()

        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        m = self.layer_norm(m)
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + 1e-5
        )

        q, k, v = self.proj_q(q), self.proj_k(m), self.proj_v(m)
        g = self.sigmoid(self.proj_g(m))

        h, c, c_h = self.n_head, self.c, self.c_h
        q = q * (c_h ** (-0.5))

        q_shape = q.shape[:-1]
        q = q.view(*q_shape, h, c_h)
        g = g.view(*g.shape[:-1], h, c_h)

        bias = (1e9 * (mask - 1))[..., :, None, :]

        a = torch.matmul(q, k.transpose(-1, -2))
        a = a + bias
        a = torch.nn.functional.softmax(a, dim=-1)
        o = torch.matmul(a, v)
        o = o.unsqueeze(-3) * g
        o = o.view(*o.shape[:-2], h*c_h)
        o = self.proj_o(o)
    

        o = o.transpose(-2, -3)

        return o # (B, s, i, c)
    

class MSAColumnGlobalAttention_ckpt(torch.nn.Module):
    """Checkpointed global column-wise MSA attention.

    Shares the same inputs and outputs as
    :class:`MSAColumnGlobalAttention` while relying on activation
    checkpointing to reduce peak memory usage.
    """
    def __init__(self, c=8, c_h=1, n_head=8):
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


    def get_qkv(self, m, mask):
        m = self.layer_norm(m)
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + 1e-5
        )

        q, k, v = self.proj_q(q), self.proj_k(m), self.proj_v(m)
        g = self.sigmoid(self.proj_g(m))

        h, c, c_h = self.n_head, self.c, self.c_h
        q = q * (c_h ** (-0.5))

        q_shape = q.shape[:-1]
        q = q.view(*q_shape, h, c_h)
        g = g.view(*g.shape[:-1], h, c_h)

        bias = (1e9 * (mask - 1))[..., :, None, :]

        return q, k, v, bias, g
    
    def attention(self, q, k, v, bias, g):
        h, c, c_h = self.n_head, self.c, self.c_h
        a = torch.matmul(q, k.transpose(-1, -2))
        a = a + bias
        a = softmax_no_cast(a, dim=-1)
        o = torch.matmul(a, v)
        o = o.unsqueeze(-3) * g
        o = o.view(*o.shape[:-2], h*c_h)
        o = self.proj_o(o)

        return o


    def forward(self, m, mask=None):
        """
        Algorithm 19: MSA global column-wise gated self-attention

        m: (B, s, i, c)

        return: (B, s, i, c)
        """

        if mask is None:
            mask = torch.ones(m.shape[:-1], dtype=m.dtype, device=m.device).detach()

        m = m.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

        if torch.is_grad_enabled():
            q, k, v, bias, g = checkpoint(self.get_qkv, m, mask)
        else:
            q, k, v, bias, g = self.get_qkv(m, mask)
            tmps = (q, k, v, bias, g)
            del q, k, v, bias, g
        
        if torch.is_grad_enabled():
            o = checkpoint(self.attention, q, k, v, bias, g)
        else:
            o = self.attention(*tmps)
            del tmps
    
        o = o.transpose(-2, -3)

        return o # (B, s, i, c)


class MSAColumnAttention(torch.nn.Module):
    """Column-wise gated MSA attention implemented with the fused kernel.

    Consumes an MSA embedding ``m`` with shape ``(B, s, i, c)`` (or without a
    batch dimension) and an optional mask ``msa_mask`` shaped ``(B, s, i)``. The
    output tensor matches the leading dimensions of ``m``.
    """
    def __init__(self, c=32, c_h=4, n_head=8):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = LayerNorm(c)

        self.proj_q = Linear(c, c_h* n_head, bias=False)
        self.proj_k = Linear(c, c_h* n_head, bias=False)
        self.proj_v = Linear(c, c_h* n_head, bias=False)
        self.proj_g = Linear(c, c_h * n_head)
        self.proj_o = Linear(c_h * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, m, msa_mask=None):
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

        zero_bias = torch.zeros((B, i, s, s), device=q.device, dtype=q.dtype)

        o = triangle_attention(
            q, k, v,
            bias=zero_bias,
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

class MSAColumnAttention_ckpt(torch.nn.Module):
    """Checkpointed column-wise gated MSA attention.

    Shares the same input and output signatures as
    :class:`MSAColumnAttention` while trading computation for lower memory via
    checkpointing.
    """
    def __init__(self, c=32, c_h=4, n_head=8):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = LayerNorm(c)

        self.proj_q = Linear(c, c_h* n_head, bias=False)
        self.proj_k = Linear(c, c_h* n_head, bias=False)
        self.proj_v = Linear(c, c_h* n_head, bias=False)
        self.proj_g = Linear(c, c_h * n_head)
        self.proj_o = Linear(c_h * n_head, c)

        self.sigmoid = torch.nn.Sigmoid()


    def get_qkv(self, m):
        m = self.layer_norm(m)

        q, k, v = self.proj_q(m), self.proj_k(m), self.proj_v(m)
        gate = self.sigmoid(self.proj_g(m))

        q_shape = q.shape[:-1]
        h, c_h = self.n_head, self.c_h

        q = q.reshape(*q_shape, h, c_h) # (B, s, i, h, c)
        k = k.reshape(*q_shape, h, c_h)
        v = v.reshape(*q_shape, h, c_h)

        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)    
        q = q / math.sqrt(c_h)

        gate = gate.reshape(*q_shape, h, c_h)
        return q, k, v, gate
    
    def get_a(self, q, k, msa_mask):
        h, c, c_h = self.n_head, self.c, self.c_h
        a = oe.contract("... s i h c, ... t i h c -> ... i h s t", q, k) / (c_h**0.5)# (B, s, t, i, h)
        if msa_mask != None:
            a = a + (1e9 * (msa_mask-1)) [..., :, None, None, :]
        a = softmax_no_cast(a, dim=-1)
        return a
    
    def get_o(self, a, v, gate):
        h, c, c_h = self.n_head, self.c, self.c_h
        v = v.transpose(-4, -3)
        tmp = oe.contract("... i h d s, ... i s h c -> ... i h d c", a, v)
        tmp = tmp.transpose(-2, -3)
        tmp = tmp.transpose(-4, -3)
        o = gate * tmp # (B, N_s, N_r, h, c)
        o = o.view(*o.shape[:-2], h * c_h)
        o = self.proj_o(o)
        return o

    def wrap(self, o, gate):
        h, c_h = self.n_head, self.c_h
        o = gate * o
        o = o.reshape(*o.shape[:-2], h * c_h)
        o = self.proj_o(o)
        o = o.transpose(-2,-3)
        return o

    def forward(self, m, msa_mask=None):
        """
        Algorithm 8: MSA column-wise gated self-attention

        m: (B, s, i, c)

        return: (B, s, i, c)
        """
        m = m.transpose(-2, -3)
        if msa_mask == None:
            msa_mask = m.new_ones(m.shape[:-1])
        msa_mask = msa_mask.transpose(-1, -2)

        if torch.is_grad_enabled():
            q, k, v, gate = checkpoint(self.get_qkv, m)
        else:
            q, k, v, gate = self.get_qkv(m)

        
        if torch.is_grad_enabled():
            o = _attention_chunked_trainable(
                    query=q, 
                    key=k, 
                    value=v, 
                    biases=[(1e9 * (msa_mask-1))[..., :, None, None, :]], 
                    chunk_size=4, 
                    chunk_dim=-4,
                    checkpoint_flag=True,
                )
        else:
            o = _attention_chunked_trainable(
                    query=q, 
                    key=k, 
                    value=v, 
                    biases=[(1e9 * (msa_mask-1))[..., :, None, None, :]], 
                    chunk_size=4, 
                    chunk_dim=-4,
                    checkpoint_flag=False,
                )
        
        if torch.is_grad_enabled():
            o = checkpoint(self.wrap, o, gate)
        else:
            o = self.wrap(o, gate)
            del q, k, v, gate
        msa_mask = msa_mask.transpose(-1, -2)

        return o  # (B, s, i, c)