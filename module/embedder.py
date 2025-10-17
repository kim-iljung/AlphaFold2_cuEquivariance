import torch

class InputEmbedder(torch.nn.Module):
    def __init__(self, tf_dim, msa_dim, c_z=128, c_m=256, relpos_k=32):
        super().__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim
        self.c_z = c_z
        self.c_m = c_m
        self.relpos_k = relpos_k

        self.linear1 = torch.nn.Linear(tf_dim, c_z)
        self.linear2 = torch.nn.Linear(tf_dim, c_z)

        self.linear3 = torch.nn.Linear(msa_dim, c_m)
        self.linear4 = torch.nn.Linear(tf_dim, c_m)

        self.relpos_linear = torch.nn.Linear(2 * relpos_k + 1, c_z)

    def one_hot(self, x, v_bins):
        """
        Algorithm 5: One-hot encoding with nearest bin

        x: (B, N, N) -> (B, N, N, 1)
        v_bins: (1, 1, 2 * relpos_k + 1)

        return: (B, N, N, 2 * relpos_k + 1)
        """
        x = x.unsqueeze(-1)
        p = torch.argmin(torch.abs(x - v_bins), dim=-1)

        p = torch.nn.functional.one_hot(p, num_classes=v_bins.shape[-1]).to(x.dtype)

        return p

    
    def relpos(self, ri):
        """
        Algorithm 4: Relative Position Encoding

        ri: (B, i), residue_index
        d: (B, i, j)
        v_bins: (2 * relpos_k + 1) -> (1, 1, 2 * relpos_k + 1)

        return: (B, i, j, c_z)
        """
        d = ri.unsqueeze(-1) - ri.unsqueeze(-2)
        v_bins = torch.arange(-self.relpos_k, self.relpos_k + 1, device=d.device, dtype=d.dtype, requires_grad=False).view(1, 1, -1)
        p = self.relpos_linear(self.one_hot(d, v_bins))

        return p

    
    def forward(self, tf, ri, msa):
        """
        Algorithm 3: Embeddings for initial representations
        tf: (B, N_res, tf_dim)
        ri: (B, N_res), residue_index
        msa: (B, N_clust, N_res, msa_dim)

        a: (B, N_res, c_z)
        b: (B, N_res, c_z)

        z: (B, N_res, c_z, c_z)

        return: m: (B, N_clust, N_res, c_m), z: (B, N_res, c_z, c_z)
        """
        a, b = self.linear1(tf), self.linear2(tf)
        z = a[..., None, :] + b[..., None, :, :]
        z += self.relpos(ri.float())

        m = self.linear3(msa) + self.linear4(tf)[..., None, :, :]

        return m, z


class RecyclingEmbedder(torch.nn.Module):
    def __init__(self, c_z, c_m):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z

        self.min_bin = 3.25
        self.max_bin = 20.75
        self.no_bins = 15

        self.linear = torch.nn.Linear(self.no_bins, c_z)
        self.layernorm_m = torch.nn.LayerNorm(c_m)
        self.layernorm_z = torch.nn.LayerNorm(c_z)

    def one_hot(self, x, v_bins):
        """
        Algorithm 5: One-hot encoding with nearest bin

        x: (B, N_res, N_res) -> (B, N_res, N_res, 1)
        v_bins: (1, 1, 2 * relpos_k + 1)

        return: (B, N, N, 2 * relpos_k + 1)
        """
        x = x.unsqueeze(-1)
        p = torch.argmin(torch.abs(x - v_bins), dim=-1)

        p = torch.nn.functional.one_hot(p, num_classes=v_bins.shape[-1]).float()

        return p

    def forward(self, m, z, x):
        """
        Algorithm 32: Embedding of Evoformer and Structure module outputs for recycling

        m: (B, N_res, c_m)
        z: (B, N_res, N_res, c_z)
        x: (B, N_res, 3) -> predicted C_beta coordinates

        return: [
            m: (B, i, c)
            z: (B, i, j, c)
        ]
        """

        m, z = self.layernorm_m(m), self.layernorm_z(z)

        v_bins = torch.linspace(self.min_bin, self.max_bin, self.no_bins, dtype=x.dtype, device=x.device, requires_grad=False)
        squared_bins = v_bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([1e9])], dim=-1
        )
        d = torch.sum(x[..., None, :] - x[..., None, :, :], dim=-1, keepdims=True) # (B, i, j, n_bins)
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)
        d = self.linear(d) # (B, i, j, c)

        z += d

        return m, z

class ExtraMSAEmbedder(torch.nn.Module):
    def __init__(self, c_in=25, c_out=64):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = torch.nn.Linear(c_in, c_out)

    def forward(self, a):
        """
        Algorithm 2, line 15: Embedding of MSA extra features

        a: (B, s, i, c)

        return: (B, s, i, c)
        """
        a = self.linear(a)

        return a