import torch
import opt_einsum as oe
from cuequivariance_torch import triangle_attention

class TriangleAttentionStartingNode(torch.nn.Module):
    """Gated attention over residue pairs referenced by a starting node.

    The module consumes a pair representation ``z`` with shape ``(B, i, j, c)``
    and an optional pair mask of shape ``(B, i, j)``. It emits an updated pair
    representation of identical shape. Internally the tensor is reshaped into
    ``n_head`` attention heads with per-head channel dimension ``c_h`` and the
    fused CUDA kernel provided by ``cuequivariance_torch`` performs the
    attention in the ``(i, j)`` plane.
    """
    def __init__(self, c=32, c_h=16, n_head=4):
        super().__init__()
        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.key = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.value = torch.nn.Linear(c, c_h * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c_h * n_head)
        self.output = torch.nn.Linear(c_h * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z, pair_mask=None):
        """Apply starting-node triangular attention to ``z``.

        Args:
            z: Pair embedding with shape ``(B, i, j, c)`` or ``(i, j, c)``.
            pair_mask: Optional mask with shape ``(B, i, j)`` or ``(i, j)``
                used to zero out invalid residue interactions.

        Returns:
            Pair embedding with the same leading dimensions as ``z``.
        """

        # Ensure a batch dimension is present.
        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        z_norm = self.layer_norm(z)
        q = self.query(z_norm)
        k = self.key(z_norm)
        v = self.value(z_norm)
        b = self.bias(z_norm)
        g = self.sigmoid(self.gate(z_norm))

        B, i_dim, j_dim, _ = z.shape
        h, c_h = self.n_head, self.c_h

        q, k, v, g = [x.view(B, i_dim, j_dim, h, c_h) for x in (q, k, v, g)]

        # Move the head axis in front of the sequence axis for the kernel.
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # Reshape the bias to broadcast over the attention matrix.
        b = b.permute(0, 3, 1, 2)

        attn_mask = None
        if pair_mask is not None:
            mask_2d = pair_mask.unsqueeze(-1) * pair_mask.unsqueeze(-2)
            attn_mask = mask_2d.bool()

        o = triangle_attention(
            q, k, v,
            bias=b,
            mask=attn_mask,
        )

        o = o.permute(0, 1, 3, 2, 4)

        o = o.reshape(B, i_dim, j_dim, h * c_h)
        g = g.reshape(B, i_dim, j_dim, h * c_h)

        o = g * o
        o = self.output(o)

        if not is_batched:
            o = o.squeeze(0)

        return o


class TriangleAttentionEndingNode(torch.nn.Module):
    """Gated attention around residue pairs referenced by an ending node.

    The module accepts and returns tensors shaped ``(B, i, j, c)`` while
    internally transposing the pair axes to swap the starting and ending nodes.
    An optional pair mask of shape ``(B, i, j)`` disables invalid residue
    interactions.
    """

    def __init__(self, c=32, c_h=16, n_head=4):
        super().__init__()
        self.c = c
        self.c_h = c_h
        self.n_head = n_head

        self.layer_norm = torch.nn.LayerNorm(c)

        self.query = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.key = torch.nn.Linear(c, c_h * n_head, bias=False)
        self.value = torch.nn.Linear(c, c_h * n_head, bias=False)

        self.bias = torch.nn.Linear(c, n_head, bias=False)
        self.gate = torch.nn.Linear(c, c_h * n_head)
        self.output = torch.nn.Linear(c_h * n_head, c)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, z, pair_mask=None):
        """Apply ending-node triangular attention to ``z``.

        Args:
            z: Pair embedding with shape ``(B, i, j, c)`` or ``(i, j, c)``.
            pair_mask: Optional mask with shape ``(B, i, j)`` or ``(i, j)``.

        Returns:
            Pair embedding with the same leading dimensions as ``z``.
        """

        z = z.transpose(-2, -3)
        is_batched = z.dim() == 4
        if not is_batched:
            z = z.unsqueeze(0)
            if pair_mask is not None:
                pair_mask = pair_mask.unsqueeze(0)

        z_norm = self.layer_norm(z)
        q = self.query(z_norm)
        k = self.key(z_norm)
        v = self.value(z_norm)
        b = self.bias(z_norm)
        g = self.sigmoid(self.gate(z_norm))

        B, i_dim, j_dim, _ = z.shape
        h, c_h = self.n_head, self.c_h

        q, k, v, g = [x.view(B, i_dim, j_dim, h, c_h) for x in (q, k, v, g)]

        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        b = b.permute(0, 3, 1, 2)

        attn_mask = None
        if pair_mask is not None:
            mask_2d = pair_mask.unsqueeze(-1) * pair_mask.unsqueeze(-2)
            attn_mask = mask_2d.bool()

        o = triangle_attention(
            q, k, v,
            bias=b,
            mask=attn_mask,
        )

        o = o.permute(0, 1, 3, 2, 4)

        o = o.reshape(B, i_dim, j_dim, h * c_h)
        g = g.reshape(B, i_dim, j_dim, h * c_h)

        o = g * o
        o = self.output(o).transpose(-2, -3)

        if not is_batched:
            o = o.squeeze(0)

        return o