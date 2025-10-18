import contextlib
import inspect

import torch
import opt_einsum as oe

from cuequivariance_torch import triangle_multiplicative_update

class TriangleMultiplicationOutgoing(torch.nn.Module):
    """Triangular multiplicative update operating on outgoing edges.

    Args:
        c: Channel dimension of the pair representation.

    The module accepts a pair tensor ``z`` shaped ``(B, i, j, c)`` (or without a
    batch dimension) and optionally a binary ``pair_mask`` of shape ``(B, i, j)``.
    It returns an updated tensor with identical leading dimensions while the
    fused kernel handles normalization, gating, and projections internally.
    """

    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """Apply the outgoing triangular multiplicative update.

        Args:
            z: Pair embedding with shape ``(B, i, j, c)`` or ``(i, j, c)``.
            pair_mask: Optional mask with shape ``(B, i, j)`` or ``(i, j)``.

        Returns:
            Tensor with the same leading dimensions as ``z``.
        """

        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        pair_mask = pair_mask.to(dtype=z.dtype)

        # Fuse projections and gates to match the fused kernel interface.
        p_in_weight_fused = torch.cat([self.proj_a.weight, self.proj_b.weight], dim=0)
        p_in_bias_fused = torch.cat([self.proj_a.bias, self.proj_b.bias], dim=0)
        g_in_weight_fused = torch.cat([self.gate_a.weight, self.gate_b.weight], dim=0)
        g_in_bias_fused = torch.cat([self.gate_a.bias, self.gate_b.bias], dim=0)

        output = triangle_multiplicative_update(
            x=z,
            direction="outgoing",
            mask=pair_mask,
            norm_in_weight=self.layer_norm.weight,
            norm_in_bias=self.layer_norm.bias,
            p_in_weight=p_in_weight_fused,
            g_in_weight=g_in_weight_fused,
            norm_out_weight=self.layer_norm_out.weight,
            norm_out_bias=self.layer_norm_out.bias,
            p_out_weight=self.proj_o.weight,
            g_out_weight=self.gate.weight,
        )

        if not is_batched:
            output = output.squeeze(0)

        return output


class TriangleMultiplicationIncoming(torch.nn.Module):
    """Triangular multiplicative update operating on incoming edges.

    Mirrors :class:`TriangleMultiplicationOutgoing` but propagates information
    from incoming neighbours. Input shapes and return values follow the same
    conventions.
    """

    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.layer_norm = torch.nn.LayerNorm(c)
        self.layer_norm_out = torch.nn.LayerNorm(c)

        self.proj_a = torch.nn.Linear(c, c)
        self.gate_a = torch.nn.Linear(c, c)
        self.proj_b = torch.nn.Linear(c, c)
        self.gate_b = torch.nn.Linear(c, c)
        self.gate = torch.nn.Linear(c, c)
        self.proj_o = torch.nn.Linear(c, c)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, z, pair_mask=None):
        """Apply the incoming triangular multiplicative update."""
        
        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        pair_mask = pair_mask.to(dtype=z.dtype)

        p_in_weight_fused = torch.cat([self.proj_a.weight, self.proj_b.weight], dim=0)
        p_in_bias_fused = torch.cat([self.proj_a.bias, self.proj_b.bias], dim=0)
        g_in_weight_fused = torch.cat([self.gate_a.weight, self.gate_b.weight], dim=0)
        g_in_bias_fused = torch.cat([self.gate_a.bias, self.gate_b.bias], dim=0)

        output = triangle_multiplicative_update(
            x=z,
            direction="incoming",
            mask=pair_mask,
            norm_in_weight=self.layer_norm.weight,
            norm_in_bias=self.layer_norm.bias,
            p_in_weight=p_in_weight_fused,
            g_in_weight=g_in_weight_fused,
            norm_out_weight=self.layer_norm_out.weight,
            norm_out_bias=self.layer_norm_out.bias,
            p_out_weight=self.proj_o.weight,
            g_out_weight=self.gate.weight,
        )

        if not is_batched:
            output = output.squeeze(0)

        return output