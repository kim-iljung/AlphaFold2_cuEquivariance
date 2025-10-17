import torch

from module.util import DropoutRowwise, DropoutColumnwise
from module.msa import MSAColumnGlobalAttention, MSARowAttentionWithPairBias
from module.transition import PairTransition, MSATransition
from module.outer_product_mean import OuterProductMean
from module.triangle_attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode
from module.triangle_multiplication import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming

class ExtraMSAStackBlock(torch.nn.Module):
    def __init__(self, c_m=256, c_z=128, c_h_o=32, c_h_m=8, c_h_p=32, n=4, n_m=8, n_head=4, p_msa=0.15, p_pair=0.25):
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_h_o = c_h_o
        self.c_h_m = c_h_m
        self.c_h_p = c_h_p
        self.n = n
        self.n_m = n_m
        self.n_head = n_head
        self.p_msa = p_msa
        self.p_pair = p_pair

        self.dropout_row_msa = DropoutRowwise(p_msa)
        self.dropout_row_pair = DropoutRowwise(p_pair)
        self.dropout_col = DropoutColumnwise(p_pair)

        # MSA stack
        self.msa_row_attn = MSARowAttentionWithPairBias(c_m, c_z, c_h_m, n_m)
        self.msa_col_attn = MSAColumnGlobalAttention(c_m, c_h_m, n_m)
        self.msa_transition = MSATransition(c_m, n)

        # Communication
        self.outer_product_mean = OuterProductMean(c_m=64, c_z=c_z, c_h=c_h_o)

        # Pair stack
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z)
        self.tri_attn_start = TriangleAttentionStartingNode(c_z, c_h_p, n_head)
        self.tri_attn_end = TriangleAttentionEndingNode(c_z, c_h_p, n_head)
        self.pair_transition = PairTransition(c_z, n)
    
    def pair_stack_block(self, z, pair_mask):
        z = z + self.dropout_row_pair(self.tri_mul_out(z, pair_mask))
        z = z + self.dropout_row_pair(self.tri_mul_in(z, pair_mask))
        z = z + self.dropout_row_pair(self.tri_attn_start(z, pair_mask))
        z = z + self.dropout_col(self.tri_attn_end(z, pair_mask))
        z = z + self.pair_transition(z, pair_mask)
        return z

    def forward(self, e, z, msa_mask, pair_mask):
        """
        Algorithm 18: Extra MSA Stack (Block)

        e: (B, s, i, c)
        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        # MSA Stack
        e = e + self.dropout_row_msa(self.msa_row_attn(e, z, msa_mask))
        e = e + self.msa_col_attn(e, msa_mask)
        e = e + self.msa_transition(e, msa_mask)

        z = z + self.outer_product_mean(e, msa_mask)

        # Pair Stack

        z = self.pair_stack_block(z, pair_mask)

        return e, z
    

class ExtraMSAStack(torch.nn.Module):
    def __init__(self, n_block, c_m=64, c_z=128, c_h_o=32, c_h_m=8, c_h_p= 32, n=4, n_m=8, n_head=4, p_msa=0.15, p_pair=0.25):
        super().__init__()

        self.n_block = n_block
        self.c_m = c_m
        self.c_z = c_z
        self.c_h_o = c_h_o
        self.c_h_m = c_h_m
        self.c_h_p = c_h_p
        self.n = n
        self.n_m = n_m
        self.n_head = n_head
        self.p_msa = p_msa
        self.p_pair = p_pair

        self.blocks = torch.nn.ModuleList([
            ExtraMSAStackBlock(c_m, c_z, c_h_o, c_h_m, c_h_p, n, n_m, n_head, p_msa, p_pair)
                for _ in range(n_block)
        ])
    
    def forward(self, e, z, msa_mask, pair_mask):
        for block in self.blocks:
            e, z = block(e, z, msa_mask, pair_mask)

        return z
