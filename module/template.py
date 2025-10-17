import torch
import opt_einsum as oe
from module.util import DropoutRowwise, DropoutColumnwise
from module.transition import PairTransition
from module.triangle_attention import TriangleAttentionStartingNode, TriangleAttentionEndingNode
from module.triangle_multiplication import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming

class TemplateAngleEmbedder(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        
        self.linear1 = torch.nn.Linear(c_in, c_out)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(c_out, c_out)

    def forward(self, f):
        """
        Algorithm 2, line 7: Embedding of template angles

        f: (B, s, i, c)

        return: (B, s, i, c)
        """
        f = self.linear1(f)
        f = self.relu(f)
        f = self.linear2(f)

        return f


class TemplatePairEmbedder(torch.nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = torch.nn.Linear(c_in, c_out)

    def forward(self, f):
        """
        Algorithm 2, line 9: Embedding of template pairs

        f: (B, s, i, c)

        return: (B, s, i, c)
        """

        return self.linear(f)


class TemplatePairStackBlock(torch.nn.Module):
    def __init__(self, c=64, c_h=16, n=4, n_head=4, p=0.25):
        super().__init__()

        self.c = c
        self.c_h = c_h
        self.n = n
        self.n_head = n_head
        self.p = p

        self.dropout_row = DropoutRowwise(p)
        self.dropout_col = DropoutColumnwise(p)

        self.tri_mult_out = TriangleMultiplicationOutgoing(c)
        self.tri_mult_in = TriangleMultiplicationIncoming(c)
        self.tri_attn_start = TriangleAttentionStartingNode(c, c_h, n_head)
        self.tri_attn_end = TriangleAttentionEndingNode(c, c_h, n_head)
        self.pair_transition = PairTransition(c, n)
    
    def forward(self, t, pair_mask):
        """
        Algorithm 16: Template Pair Stack (Block)

        t: (B, i, j, c)

        return: (B, i, j, c)
        """
        single_templates = [
            t_.unsqueeze(-4) for t_ in torch.unbind(t, dim=-4)
        ]
        single_templates_masks = [
            m_.unsqueeze(-3) for m_ in torch.unbind(pair_mask, dim=-3)
        ]
        for i in range(len(single_templates)):
            single = single_templates[i]
            if len(single_templates_masks) == 1:
                single_mask = single_templates_masks[0]
            else:
                single_mask = single_templates_masks[i]
            
            single_ = self.dropout_row(self.tri_mult_out(single, single_mask))
            single = single + single_
            single_ = self.dropout_row(self.tri_mult_in(single, single_mask))
            single = single + single_
            single_ = self.dropout_row(self.tri_attn_start(single, single_mask))
            single = single + single_
            single_ = self.dropout_col(self.tri_attn_end(single, single_mask))
            single = single + single_
            single_ = self.pair_transition(single, single_mask)
            single = single + single_
            single_templates[i] = single
        
        templates = torch.cat(single_templates, dim=-4)

        return templates


class TemplatePairStack(torch.nn.Module):
    def __init__(self, n_block, c=64, c_h=16, n=4, n_head=4, p=0.25):
        super().__init__()

        self.n_block = n_block
        self.c = c
        self.c_h = c_h
        self.n = n
        self.n_head = n_head
        self.p = p
        

        self.blocks = torch.nn.ModuleList([
                TemplatePairStackBlock(c, c_h, n, n_head, p)
                for _ in range(n_block)
        ])
        self.layer_norm = torch.nn.LayerNorm(c)
    
    def forward(self, t, pair_mask):
        for block in self.blocks:
            t = block(t, pair_mask)
        t = self.layer_norm(t)
        
        return t


class TemplatePointwiseAttention(torch.nn.Module):
    def __init__(self, c_t=128, c_z=64, c_h=16, n_head=4):
        super().__init__()

        self.c_t = c_t
        self.c_z = c_z
        self.c_h = c_h
        self.n_head = n_head

        self.proj_q = torch.nn.Linear(c_t, c_h * n_head, bias=False)
        self.proj_k = torch.nn.Linear(c_z, c_h * n_head, bias=False)
        self.proj_v = torch.nn.Linear(c_z, c_h * n_head, bias=False)

        self.proj_o = torch.nn.Linear(c_z, c_t)
    
    def forward(self, t, z, template_mask=None):
        """
        Algorithm 17: Template Pointwise Attention

        t: (B, s, i, j, c). s: N_templ, i: N_res, j: N_res
        z: (B, i, j, c)

        return: (B, i, j, c)
        """

        if template_mask is None:
            template_mask = t.new_ones(t.shape[:-3])


        q = self.proj_q(z) # (B, i, j, c * h)
        k = self.proj_k(t) # (B, s, i, j, c * h)
        v = self.proj_v(t)

        # B, s, i, j, _ = k.shape
        q_shape = q.shape[:-1]
        k_shape = k.shape[:-1]
        h, c_z, c_h = self.n_head, self.c_z, self.c_h

        q = q.view(*q_shape, h, c_h) # (B, i, j, h, c)
        k = k.view(*k_shape, h, c_h) # (B, s, i, j, h, c)
        v = v.view(*k_shape, h, c_h)



        a = oe.contract("... i j h c, ... s i j h c -> ... s i j h", q, k) * (c_h ** -0.5) + 1e9 * (template_mask[..., None, None, None, :] - 1)
        a = torch.nn.functional.softmax(a, dim=-4) # (B, s, i, j, h)

        o = oe.contract("... s i j h, ... s i j h c -> ... i j h c", a, v) # (B, i, j, h, c)
        o = o.view(*q_shape, h * c_h)
        o = self.proj_o(o)

        return o # (B, i, j, c)

