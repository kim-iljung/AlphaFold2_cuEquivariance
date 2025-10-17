import contextlib
import torch
import opt_einsum as oe

try:  # Prefer the fused cuEquivariance kernel when available.
    from cuequivariance_torch import triangle_multiplicative_update  # type: ignore
except ImportError:  # pragma: no cover - kernel package missing at import time.
    triangle_multiplicative_update = None  # type: ignore[assignment]

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

        output = None
        if triangle_multiplicative_update is not None:
            with contextlib.suppress(RuntimeError):
                output = triangle_multiplicative_update(
                    x=z.contiguous(),
                    direction="outgoing",
                    mask=pair_mask.contiguous(),
                    norm_in_weight=self.layer_norm.weight,
                    norm_in_bias=self.layer_norm.bias,
                    p_in_weight=p_in_weight_fused,
                    p_in_bias=p_in_bias_fused,
                    g_in_weight=g_in_weight_fused,
                    g_in_bias=g_in_bias_fused,
                    norm_out_weight=self.layer_norm_out.weight,
                    norm_out_bias=self.layer_norm_out.bias,
                    p_out_weight=self.proj_o.weight,
                    p_out_bias=self.proj_o.bias,
                    g_out_weight=self.gate.weight,
                    g_out_bias=self.gate.bias,
                )

        if output is None:
            output = self._triangle_multiplicative_update(
                z,
                pair_mask,
                direction="outgoing",
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

        output = None
        if triangle_multiplicative_update is not None:
            with contextlib.suppress(RuntimeError):
                output = triangle_multiplicative_update(
                    x=z.contiguous(),
                    direction="incoming",
                    mask=pair_mask.contiguous(),
                    norm_in_weight=self.layer_norm.weight,
                    norm_in_bias=self.layer_norm.bias,
                    p_in_weight=p_in_weight_fused,
                    p_in_bias=p_in_bias_fused,
                    g_in_weight=g_in_weight_fused,
                    g_in_bias=g_in_bias_fused,
                    norm_out_weight=self.layer_norm_out.weight,
                    norm_out_bias=self.layer_norm_out.bias,
                    p_out_weight=self.proj_o.weight,
                    p_out_bias=self.proj_o.bias,
                    g_out_weight=self.gate.weight,
                    g_out_bias=self.gate.bias,
                )

        if output is None:
            output = self._triangle_multiplicative_update(
                z,
                pair_mask,
                direction="incoming",
            )

        if not is_batched:
            output = output.squeeze(0)

        return output

    def _triangle_multiplicative_update(self, z, pair_mask, direction):
        """Torch fallback mirroring the triangular multiplicative update."""

        mask = pair_mask.unsqueeze(-1)

        z_norm = self.layer_norm(z)
        gate = torch.sigmoid(self.gate(z_norm))

        proj_a = self.proj_a(z_norm) * torch.sigmoid(self.gate_a(z_norm))
        proj_b = self.proj_b(z_norm) * torch.sigmoid(self.gate_b(z_norm))

        proj_a = proj_a * mask
        proj_b = proj_b * mask

        if direction == "outgoing":
            # Contract outgoing edges j <- k -> i.
            contracted = oe.contract(
                "...ikc,...jkc->...ijc", proj_a, proj_b, optimize="optimal"
            )
        else:
            contracted = oe.contract(
                "...kjc,...kic->...ijc", proj_a, proj_b, optimize="optimal"
            )

        contracted = contracted * pair_mask.unsqueeze(-1)
        contracted = self.layer_norm_out(contracted)
        contracted = self.proj_o(contracted)
        contracted = contracted * gate

        return contracted
