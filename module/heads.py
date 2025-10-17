import torch

class PerResidueLDDTCaPredictor(torch.nn.Module):
    def __init__(self, no_bins, c):
        super().__init__()

        self.no_bins = no_bins
        self.c = c

        self.layer_norm = torch.nn.LayerNorm(384)

        self.linear_1 = torch.nn.Linear(384, 128)
        self.linear_2 = torch.nn.Linear(128, 128)
        self.linear_3 = torch.nn.Linear(128, 50)

        self.relu = torch.nn.ReLU()
    
    def forward(self, s):
        """
        Algorithm 29: Predict model confidence pLDDT

        s: (B, i, c)
        """
        s = self.layer_norm(s)

        return self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(s)))))


class DistogramHead(torch.nn.Module):
    def __init__(self, c, no_bins):
        super().__init__()

        self.no_bins = no_bins
        self.c = c

        self.linear = torch.nn.Linear(c, no_bins)
    
    def forward(self, z):
        """
        Section 1.9.8: Distogram prediction

        z: (B, i, j, c)

        return: (B, i, j, no_bins) -> distogram probability distribution
        """

        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)

        return logits # (B, i, j, no_bins)


class TMScoreHead(torch.nn.Module):
    def __init__(self, c, no_bins):
        super().__init__()

        self.no_bins = no_bins
        self.c = c

        self.linear = torch.nn.Linear(c, no_bins)

    def forward(self, z):
        """
        Section 1.9.7: TM-score prediction

        z: (B, i, j, c)

        return: (B, i, j, no_bins) -> TM-score prediction
        """

        logits = self.linear(z)
        return logits


class MaskedMSAHead(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.linear = torch.nn.Linear(c, 23)
    
    def forward(self, m):
        """
        Section 1.9.9: Masked MSA prediction

        m: (B, s, i, c) MSA Embedding

        return: (B, s, i, c) -> MSA embedding
        """

        logits = self.linear(m)
        return logits


class ExperimentallyResolvedHead(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c
        
        self.linear = torch.nn.Linear(c, 37)

    def forward(self, s):
        """
        Section 1.9.10: "Experimentally resolved" prediction

        s: (B, i, c)

        return: (B, i, c) logits
        """

        logits = self.linear(s)
        return logits


def compute_plddt(logits):
    no_bins = logits.shape[-1]
    bin_width = 1.0 / no_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )

    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def compute_tm(logits, max_bin=31, no_bins=64):
    residue_weights = logits.new_ones(logits.shape[-2])
    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, bin_centers[-1:]], dim=0)

    clipped_n = max(torch.sum(residue_weights), 19)
    d_0 = 1.24 * (clipped_n - 15) ** (1 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)
    f_d = 1.0 / (1 + (bin_centers / d_0) ** 2.0)

    predicted_tm_term = torch.sum(probs * f_d, dim=-1)
    normed_residue_mask = residue_weights / (1e-8 + residue_weights.sum())
    per_alignment = torch.sum(normed_residue_mask * predicted_tm_term, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]

    return per_alignment[tuple(argmax)]


class AuxiliaryHeads(torch.nn.Module):
    def __init__(self, c, no_bins):
        super().__init__()

        self.c = c
        self.no_bins = no_bins

        self.plddt = PerResidueLDDTCaPredictor(64, 128)
        self.distogram = DistogramHead(128, 64)
        self.masked_msa = MaskedMSAHead(256)
        # self.tmscore = TMScoreHead(c, no_bins)
        self.experimentally_resolved = ExperimentallyResolvedHead(384)

    def forward(self, outputs):
        plddt_logits = self.plddt(outputs["single"])
        distogram_logits = self.distogram(outputs["pair"])
        masked_msa_logits = self.masked_msa(outputs["msa_feat"])
        experimentally_resolved_logits = self.experimentally_resolved(outputs["single"])
        # tm_logits = self.tmscore(outputs["pair"])

        aux_out = {
            "lddt_logits": plddt_logits,
            "plddt": compute_plddt(plddt_logits),
            "distogram_logits": distogram_logits,
            "masked_msa_logits": masked_msa_logits,
            "experimentally_resolved_logits": experimentally_resolved_logits,
            # "tm_logits": tm_logits,
            # "predicted_tm_score": compute_tm(tm_logits),
        }
        


        return aux_out