import torch


class DropoutRowwise(torch.nn.Module):
    def __init__(self, p=0.25):
        super().__init__()

        self.p = p
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        """
        Dropout for row-wise inputs

        x: (B, N_res, N_res, c_z)

        return: (B, N_res, N_res, c_z)
        """
        if not self.training:
            return x


        shape = list(x.shape)
        shape[-3] = 1 # Row-wise
        mask = x.new_ones(shape, dtype=x.dtype)
        mask = self.dropout(mask)

        return x * mask


class DropoutColumnwise(torch.nn.Module):
    def __init__(self, p=0.25):
        super().__init__()

        self.p = p
        self.dropout = torch.nn.Dropout(p)

    def forward(self, x):
        """
        Dropout for column-wise inputs

        x: (B, N_res, N_res, c_z)

        return: (B, N_res, N_res, c_z)
        """
        if not self.training:
            return x


        shape = list(x.shape)
        shape[-2] = 1
        mask = x.new_ones(shape, dtype=x.dtype)
        mask = self.dropout(mask)

        return x * mask
