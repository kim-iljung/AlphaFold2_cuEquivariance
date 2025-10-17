import torch

class PairTransition(torch.nn.Module):
    def __init__(self, c=128, n=4):
        super().__init__()

        self.c = c
        self.n = n

        self.layer_norm = torch.nn.LayerNorm(c)
        self.proj_in = torch.nn.Linear(c, c * n)
        self.proj_out = torch.nn.Linear(c * n, c)
        self.relu = torch.nn.ReLU()

    def forward(self, z, pair_mask=None):
        """
        Algorithm 15: Transition layer in the pair stack

        z: (B, i, j, c)

        return: (B, i, j, c)
        """
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])

        pair_mask = pair_mask.unsqueeze(-1)

        z = self.layer_norm(z)
        z = self.proj_in(z)
        z = self.relu(z)
        z = self.proj_out(z) * pair_mask

        return z

class MSATransition(torch.nn.Module):
    def __init__(self, c=32, n=4):
        super().__init__()

        self.c = c
        self.n = n

        self.layer_norm = torch.nn.LayerNorm(c)
        self.proj_in = torch.nn.Linear(c, c * n)
        self.proj_out = torch.nn.Linear(c * n, c)
        self.relu = torch.nn.ReLU()

    def forward(self, m, msa_mask=None):
        """
        Algorithm 9: Transition layer in the MSA Stack
        
        m: (B, s, i, c)

        return: (B, s, i, c)
        """
        if msa_mask == None:
            msa_mask = m.new_ones(m.shape[:-1])
        msa_mask = msa_mask.unsqueeze(-1)
        m = self.layer_norm(m)
        m = self.proj_in(m)
        m = self.relu(m)
        m = self.proj_out(m) * msa_mask
        

        return m
