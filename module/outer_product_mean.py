import torch
import opt_einsum as oe

class OuterProductMean(torch.nn.Module):
    """Computes the normalized outer product features used to update pair representations."""
    def __init__(self, c_m=256, c_z=128, c_h=32, eps=1e-3):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.eps = eps

        self.layer_norm = torch.nn.LayerNorm(c_m)
        self.linear_a = torch.nn.Linear(c_m, c_h)
        self.linear_b = torch.nn.Linear(c_m, c_h)
        self.linear_out = torch.nn.Linear(c_h * c_h, c_z)

    def forward(self, m, msa_mask=None):
        if msa_mask is None:
            msa_mask = m.new_ones(m.shape[:-1])
        msa_mask = msa_mask.unsqueeze(-1)
        
        m = self.layer_norm(m)

        a = self.linear_a(m) * msa_mask
        b = self.linear_b(m) * msa_mask

        a = a.transpose(-2, -3)
        b = b.transpose(-2, -3)
        outer = oe.contract("...bac,...dae->...bdce", a, b)

        outer = outer.reshape(outer.shape[:-2] + (-1,))

        outer = self.linear_out(outer)

        norm = oe.contract("...abc,...adc->...bdc", msa_mask, msa_mask)
        norm = norm + self.eps
        outer = outer / norm

        return outer

