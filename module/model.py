import torch
import gc
# import GPUtil
# import matplotlib.pyplot as plt

from module.rigid_utils import Rigid
import module.residue_constants as rc

from module.feats import atom14_to_atom37, build_extra_msa_feat

from module.tensor_utils import tensor_tree_map

from module.embedder import InputEmbedder, RecyclingEmbedder, ExtraMSAEmbedder
from module.template import TemplateAngleEmbedder, TemplatePairEmbedder, TemplatePairStack,TemplatePointwiseAttention
from module.evoformer import Evoformer
from module.structure_module import StructureModule
from module.heads import AuxiliaryHeads
from module.extra_msa import ExtraMSAStack

class Alphafold2(torch.nn.Module):
    def __init__(self, n_block=48, c=384, n_head=8, p=0.25, no_bins=64):
        super().__init__()

        self.n_block = n_block
        self.c = c
        self.n_head = n_head
        self.p = p
        self.no_bins = no_bins

        self.input_embedder = InputEmbedder(22,49) # 22 49 128 256
        self.recycling_embedder = RecyclingEmbedder(128,256) # 128, 256

        self.template_angle_embedder = TemplateAngleEmbedder(57, 256) # 57, 256
        self.template_pair_embedder = TemplatePairEmbedder(88, 64) # 88 64
        self.template_pair_stack = TemplatePairStack(2, 64, 16, 2, 4, 0.25)
        self.template_pointwise_att = TemplatePointwiseAttention()

        self.extra_msa_embedder = ExtraMSAEmbedder(25, 64)
        self.extra_msa_stack = ExtraMSAStack(n_block=4, c_m=64)

        self.evoformer = Evoformer()
        self.structure_module = StructureModule()
        self.aux_heads = AuxiliaryHeads(c, no_bins)


    def build_template_angle_feat(self, batch):
        template_aatype = batch["template_aatype"]
        torsion_angles_sin_cos = batch["template_torsion_angles_sin_cos"]
        alt_torsion_angles_sin_cos = batch["template_alt_torsion_angles_sin_cos"]
        torsion_angles_mask = batch["template_torsion_angles_mask"]

        ta_shape = template_aatype.shape

        return torch.cat([
            torch.nn.functional.one_hot(template_aatype.to(torch.int64), 22),
            torsion_angles_sin_cos.reshape(*ta_shape, 7 * 2),
            alt_torsion_angles_sin_cos.reshape(*ta_shape, 7 * 2),
            torsion_angles_mask
        ], dim=-1)

    def build_template_pair_feat(
        self,
        batch, 
        min_bin, max_bin, no_bins, 
        use_unit_vector=False, 
        eps=1e-20, inf=1e8
    ):
        template_mask = batch["template_pseudo_beta_mask"]
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

        # Compute distogram (this seems to differ slightly from Alg. 5)
        tpb = batch["template_pseudo_beta"]
        dgram = torch.sum(
            (tpb[..., None, :] - tpb[..., None, :, :]) ** 2, dim=-1, keepdim=True
        )
        lower = torch.linspace(min_bin, max_bin, no_bins, device=tpb.device) ** 2
        upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
        dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

        to_concat = [dgram, template_mask_2d[..., None]]

        aatype_one_hot = torch.nn.functional.one_hot(
            batch["template_aatype"],
            rc.restype_num + 2,
        )

        n_res = batch["template_aatype"].shape[-1]
        to_concat.append(
            aatype_one_hot[..., None, :, :].expand(
                *aatype_one_hot.shape[:-2], n_res, -1, -1
            )
        )
        to_concat.append(
            aatype_one_hot[..., None, :].expand(
                *aatype_one_hot.shape[:-2], -1, n_res, -1
            )
        )

        n, ca, c = [rc.atom_order[a] for a in ["N", "CA", "C"]]
        rigids = Rigid.make_transform_from_reference(
            n_xyz=batch["template_all_atom_positions"][..., n, :],
            ca_xyz=batch["template_all_atom_positions"][..., ca, :],
            c_xyz=batch["template_all_atom_positions"][..., c, :],
            eps=eps,
        )
        points = rigids.get_trans()[..., None, :, :]
        rigid_vec = rigids[..., None].invert_apply(points)

        inv_distance_scalar = torch.rsqrt(eps + torch.sum(rigid_vec ** 2, dim=-1))

        t_aa_masks = batch["template_all_atom_mask"]
        template_mask = (
            t_aa_masks[..., n] * t_aa_masks[..., ca] * t_aa_masks[..., c]
        )
        template_mask_2d = template_mask[..., None] * template_mask[..., None, :]

        inv_distance_scalar = inv_distance_scalar * template_mask_2d
        unit_vector = rigid_vec * inv_distance_scalar[..., None]
        
        if(not use_unit_vector):
            unit_vector = unit_vector * 0.
        
        to_concat.extend(torch.unbind(unit_vector[..., None, :], dim=-1))
        to_concat.append(template_mask_2d[..., None])

        act = torch.cat(to_concat, dim=-1)
        act = act * template_mask_2d[..., None]

        return act
    
    def embed_templates(self, batch, z, pair_mask, templ_dim):
        n_templ = batch["template_aatype"].shape[templ_dim]        


        t_embeds = []
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx).squeeze(templ_dim),
                batch,)
            t = self.build_template_pair_feat(single_template_feats, 3.25, 50.75, 39, 1e5, 1e-6)
            t = self.template_pair_embedder(t)
            t_embeds.append(t)
        t = torch.stack(t_embeds, dim=templ_dim)
        t = self.template_pair_stack(t, pair_mask.unsqueeze(-3))
        t = self.template_pointwise_att(t, z, batch["template_mask"])

        t_mask = torch.sum(batch["template_mask"], dim=-1) > 0
        t_mask = t_mask.reshape(
            *t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape)))
        )
        t = t * t_mask
            

        a = self.build_template_angle_feat(batch)
        a = self.template_angle_embedder(a)

        return t, a
    
    

    def pseudo_beta_fn(self, aatype, x):
        is_gly = aatype == rc.restype_order["G"]
        ca_idx = rc.atom_order["CA"]
        cb_idx = rc.atom_order["CB"]
        pseudo_beta = torch.where(
            is_gly[..., None].expand(*((-1,) * len(is_gly.shape)), 3),
            x[..., ca_idx, :],
            x[..., cb_idx, :],
        )

        return pseudo_beta

    
    # def atom14_to_atom37(atom14, batch):
    
    

    def iteration(self, batch, prev_m, prev_z, prev_x):
        outputs = {}

        # default openfold input infomation
        batch_dims = batch["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = batch["target_feat"].shape[-2]
        n_seq = batch["msa_feat"].shape[-3]
        device = batch["target_feat"].device

        seq_mask = batch["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = batch["msa_mask"]

        # input embedder
        # if torch.is_grad_enabled():
        #     batch["target_feat"], batch["residue_index"], batch["msa_feat"]
        m, z = self.input_embedder(batch["target_feat"], batch["residue_index"], batch["msa_feat"])
        
        m_, z_ = self.recycling_embedder(
            prev_m,
            prev_z,
            self.pseudo_beta_fn(batch["aatype"], prev_x),
        )
        m[..., 0, :, :] = m[..., 0, :, :] + m_
        z = z + z_
        if not torch.is_grad_enabled():
            del m_, z_, prev_m, prev_z, prev_x


        template_batch = {k: v for k, v in batch.items() if k.startswith("template_")}
        t, a = self.embed_templates(template_batch, z, pair_mask, no_batch_dims)
        z = z + t
        m = torch.cat([m, a], dim=-3)
        if not torch.is_grad_enabled():
            del t, a
        msa_mask = torch.cat([batch["msa_mask"], batch["template_torsion_angles_mask"][..., 2]], dim=-2)
        

        extra_msa_feat = build_extra_msa_feat(batch).to(dtype=z.dtype)
        a = self.extra_msa_embedder(extra_msa_feat)
        z = self.extra_msa_stack(a, z, batch["extra_msa_mask"], pair_mask)
        if not torch.is_grad_enabled():
            del a

        m, z, s = self.evoformer(m, z, msa_mask, pair_mask)

        outputs["msa_feat"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s
        outputs["sm"] = self.structure_module(s.to(torch.float32), z.to(torch.float32), batch["aatype"], batch["seq_mask"].to(torch.float32))
        outputs["final_atom_positions"] = atom14_to_atom37(outputs["sm"]["positions"][-1], batch)
        outputs["final_atom_mask"] = batch["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]
        if not torch.is_grad_enabled():
            del m, z, s, batch

        return outputs, outputs["msa_feat"][..., 0, :, :], outputs["pair"], outputs["final_atom_positions"]


    def forward(self, batch):
        """
        Algorithm 2: Alphafold2 inference

        batch: {
            "aatype": (B, i) -> amino acid indices
            "residue_index": (B, i) -> residue indices
            "target_feat": (B, i, c) -> target features

            "template_aatype": (B, t, i) -> template amino acid indices
            "template_all_atom_positions": (B, t, i, 37, 3) -> template all atom positions
            "template_pseudo_beta": (B, t, i, 3) -> position of template carbon beta atoms
        }
        """
        t_shape = batch["target_feat"].shape[:-2]

        m = torch.zeros([*t_shape, 256], device=batch["target_feat"].device, dtype=batch["target_feat"].dtype)
        z = torch.zeros([*t_shape, t_shape[-1], 128], device=batch["target_feat"].device, dtype=batch["target_feat"].dtype)
        x = torch.zeros([*t_shape[:-2], 37, 3], device=batch["target_feat"].device, dtype=batch["target_feat"].dtype)
        # print("starting iteration...")
        prevs = [m, z, x]

        num_iters = batch["aatype"].shape[-1]
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            if cycle_no != num_iters - 1:
                with torch.no_grad():
                    outputs, m, z, x = self.iteration(feats, *prevs)
                    del prevs
                    prevs = [m, z, x]
                    del outputs, m, z, x
                    torch.clear_autocast_cache()
                    torch.cuda.empty_cache()
                    gc.collect()
            else: # last iteration
                prevs[0].requires_grad_(), prevs[1].requires_grad_(), prevs[2].requires_grad_()
                
                outputs, m, z, x = self.iteration(feats, *prevs)
                # GPUtil.showUtilization()
                # print(cycle_no)

            
            # print(f"{cycle_no + 1}th iteration done.")

        outputs.update(self.aux_heads(outputs))
        

        return outputs