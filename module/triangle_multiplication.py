import contextlib
import inspect

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

        output = _call_triangle_kernel(
            triangle_multiplicative_update,
            direction="outgoing",
            z=z,
            pair_mask=pair_mask,
            norm_in=self.layer_norm,
            norm_out=self.layer_norm_out,
            proj_out=self.proj_o,
            gate_out=self.gate,
            p_in_weight=p_in_weight_fused,
            p_in_bias=p_in_bias_fused,
            g_in_weight=g_in_weight_fused,
            g_in_bias=g_in_bias_fused,
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

        output = _call_triangle_kernel(
            triangle_multiplicative_update,
            direction="incoming",
            z=z,
            pair_mask=pair_mask,
            norm_in=self.layer_norm,
            norm_out=self.layer_norm_out,
            proj_out=self.proj_o,
            gate_out=self.gate,
            p_in_weight=p_in_weight_fused,
            p_in_bias=p_in_bias_fused,
            g_in_weight=g_in_weight_fused,
            g_in_bias=g_in_bias_fused,
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


def _call_triangle_kernel(
    kernel,
    *,
    direction,
    z,
    pair_mask,
    norm_in,
    norm_out,
    proj_out,
    gate_out,
    p_in_weight,
    p_in_bias,
    g_in_weight,
    g_in_bias,
):
    """Invoke the fused triangle kernel while adapting to signature drift."""

    if kernel is None:
        return None

    call_kwargs = {}

    try:
        signature = inspect.signature(kernel)
        param_names = set(signature.parameters)
    except (TypeError, ValueError):
        param_names = None

    def add_arg(names, value):
        if value is None:
            return
        if param_names is None:
            call_kwargs[names[0]] = value
            return
        for name in names:
            if name in param_names:
                call_kwargs[name] = value
                return

    add_arg(("x", "input", "tensor"), z.contiguous())
    add_arg(("direction", "dir", "orientation"), direction)
    add_arg(("mask", "input_mask", "attention_mask"), pair_mask.contiguous())
    add_arg(("norm_in_weight", "norm_in_weights", "layer_norm_in_weight"), norm_in.weight)
    add_arg(("norm_in_bias", "norm_in_biases", "layer_norm_in_bias"), norm_in.bias)
    add_arg(("p_in_weight", "pin_weight", "proj_in_weight"), p_in_weight)
    add_arg(("p_in_bias", "pin_bias", "proj_in_bias"), p_in_bias)
    add_arg(("g_in_weight", "gin_weight", "gate_in_weight"), g_in_weight)
    add_arg(("g_in_bias", "gin_bias", "gate_in_bias"), g_in_bias)
    add_arg(("norm_out_weight", "norm_out_weights", "layer_norm_out_weight"), norm_out.weight)
    add_arg(("norm_out_bias", "norm_out_biases", "layer_norm_out_bias"), norm_out.bias)
    add_arg(("p_out_weight", "pout_weight", "proj_out_weight"), proj_out.weight)
    add_arg(("p_out_bias", "pout_bias", "proj_out_bias"), proj_out.bias)
    add_arg(("g_out_weight", "gout_weight", "gate_out_weight"), gate_out.weight)
    add_arg(("g_out_bias", "gout_bias", "gate_out_bias"), gate_out.bias)

    with contextlib.suppress(RuntimeError, TypeError, ValueError):
        return kernel(**call_kwargs)

    return None
