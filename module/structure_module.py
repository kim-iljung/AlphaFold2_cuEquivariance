import torch

from module.rigid_utils import Rotation, Rigid

from module.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from module.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)

from module.tensor_utils import dict_multimap

class BackboneUpdate(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c
        self.linear = torch.nn.Linear(c, 6)

    
    def forward(self, s):
        """
        Algorithm 23: Backbone update

        s: (B, i, c)

        b, c, d: (B, i, 1)
        t: (B, i, 3)

        return: 
        """
        s = self.linear(s)

        return s

class StructureModuleTransitionLayer(torch.nn.Module):
    def __init__(self, c):
        super().__init__()

        self.c = c

        self.linear_1 = torch.nn.Linear(c, c)
        self.linear_2 = torch.nn.Linear(c, c)
        self.linear_3 = torch.nn.Linear(c, c)

        self.relu = torch.nn.ReLU()
    
    def forward(self, s):
        s = s + self.linear_3(self.relu(self.linear_2(self.relu(self.linear_1(s)))))

        return s

class StructureModuleTransition(torch.nn.Module):
    def __init__(self, c, n_layers, p=0.1):
        super().__init__()

        self.c = c
        self.n_layers = n_layers

        self.layers = torch.nn.ModuleList([
                StructureModuleTransitionLayer(c)
                for _ in range(n_layers)
        ])
        self.dropout = torch.nn.Dropout(p)
        self.layer_norm = torch.nn.LayerNorm(c)
    
    def forward(self, s):
        for layer in self.layers:
            s = layer(s)

        return self.layer_norm(self.dropout(s))


class AngleResnetBlock(torch.nn.Module):
    def __init__(self, c=128):
        super().__init__()

        self.c = c

        self.linear_1 = torch.nn.Linear(c, c)
        self.linear_2 = torch.nn.Linear(c, c)

        self.relu = torch.nn.ReLU()

    def forward(self, a):
        a += self.linear_2(self.relu(self.linear_1(self.relu(a))))

        return a


class AngleResnet(torch.nn.Module):
    def __init__(self, c_in=384, c_h=128, n_layer=8, n_angle=7, eps=1e5):
        super().__init__()


        self.n_layer = n_layer
        self.n_angle = n_angle

        self.linear_in = torch.nn.Linear(c_in, c_h)
        self.linear_initial = torch.nn.Linear(c_in, c_h)

        self.layers = torch.nn.ModuleList([
                AngleResnetBlock(c_h)
                for _ in range(n_layer)
        ])
        self.layer_norm = torch.nn.LayerNorm(c_h)
        self.linear_out = torch.nn.Linear(c_h, n_angle * 2)

        self.relu = torch.nn.ReLU()
    
    def forward(self, s, s_initial):
        """
        s, s_initial: (B, i, c)

        return: [
            unnomarlized_a: (B, i, 7, 2)
            a: (B, i, 7, 2)
        ]
        """
        # B, i, c = s.shape
        s_shape = s.shape
        a = self.linear_in(self.relu(s)) + self.linear_initial(self.relu(s_initial))

        for layer in self.layers:
            a = layer(a)

        a = self.linear_out(self.relu(a)) # (B, i, 7, 2)
        a = a.view(*s_shape[:-1], self.n_angle, 2)

        unnomarlized_a = a.clone()

        norm_eps = torch.sqrt(
            torch.clamp(
                torch.sum(a ** 2, dim=-1, keepdim=True),
                min=1e-12,
            )
        )

        a = a / norm_eps
        # a = a / torch.norm(a, dim=-1, keepdim=True)

        return unnomarlized_a, a



class StructureModule(torch.nn.Module):
    def __init__(self, c_s=384, c_z=128, c_ipa=16, c_res=128, n_h_i=12, n_qk_points=4, n_v_points=8, p=0.1, n_blocks=8, n_transition_layers=1, n_resnet_blocks=2, n_angles=7,trans_scale_factor=10, eps=1e-12, inf=1e5):
        super().__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.n_blocks = n_blocks
        self.n_transition_layers = n_transition_layers
        self.n_resnet_blocks = n_resnet_blocks

        self.layernorm_s = torch.nn.LayerNorm(c_s)
        self.layernorm_z = torch.nn.LayerNorm(c_z)

        self.linear_in = torch.nn.Linear(c_s, c_s)
        self.ipa = InvariantPointAttention(c_s=c_s, c_z=c_z, c_ipa=c_ipa, n_head=n_h_i, n_qk_points=n_qk_points, n_v_points=n_v_points, inf=inf, eps=eps)
        self.dropout = torch.nn.Dropout(p)
        self.layernorm_ipa = torch.nn.LayerNorm(c_s)

        self.transition = StructureModuleTransition(c_s, n_transition_layers, p)
        self.backbone_update = BackboneUpdate(c_s)
        self.angle_resnet = AngleResnet(c_s, c_res, n_resnet_blocks, n_angles, eps)


    def forward(self, s_initial, z, aatype, seq_mask=None):
        """
        s_initial: (B, i, c)
        z: (B, i, j, c)
        aatype: (B, i) -> amino acid indices

        return: [
            s: (B, i, c),
            T: Rigid, (B, i) -> transformation object
            a: (B, i, 7, 2)
        ]
        """
        if seq_mask is None:
            seq_mask = s.new_ones(s.shape[:-1])

        s_initial = self.layernorm_s(s_initial)
        z = self.layernorm_z(z)

        s = self.linear_in(s_initial)
        rigids = Rigid.identity(
            s.shape[:-1], 
            s.dtype, 
            s.device, 
            self.training,
            fmt="quat",
        )
        
        outputs = []
        for _ in range(self.n_blocks):
            s = s + self.ipa(s, z, rigids, seq_mask)
            s = self.layernorm_ipa(self.dropout(s))
            s = self.transition(s)

            rigids = rigids.compose_q_update_vec(self.backbone_update(s))

            backb_to_global = Rigid(
                Rotation(
                    rot_mats=rigids.get_rots().get_rot_mats(), 
                    quats=None
                ),
                rigids.get_trans(),
            )

            backb_to_global = backb_to_global.scale_translation(
                10
            )

            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            all_frames_to_global = self.torsion_angles_to_frames(
                backb_to_global, angles, aatype
            )

            pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
                all_frames_to_global, aatype
            )

            scaled_rigids = rigids.scale_translation(10)

            preds = {
                "frames": scaled_rigids.to_tensor_7(),
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
                "states": s,
            }

            outputs.append(preds)
            rigids = rigids.stop_rot_gradient()

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        
        return outputs
    
    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        self._init_residue_constants(alpha.dtype, alpha.device)
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        self._init_residue_constants(r.dtype, r.device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )


class InvariantPointAttention(torch.nn.Module):
    def __init__(self, c_s=384, c_z=128, c_ipa=16, n_head=12, n_qk_points=4, n_v_points=8, inf=1e5, eps=1e-12):
        super().__init__()

        self.c_ipa = c_ipa
        self.n_head = n_head
        self.n_qk_points = n_qk_points
        self.n_v_points = n_v_points

        self.query = torch.nn.Linear(c_s, c_ipa * n_head)
        self.kv = torch.nn.Linear(c_s, c_ipa * n_head * 2)
        self.bias = torch.nn.Linear(c_z, n_head)

        self.query_points = torch.nn.Linear(c_s, 3 * n_qk_points * n_head)
        self.kv_points = torch.nn.Linear(c_s, 3 * (n_qk_points+n_v_points) * n_head)

        self.w_c = torch.tensor((2 / (9 * self.n_qk_points)) ** 0.5, requires_grad=False)
        self.w_L = torch.tensor((1 / 3) ** 0.5, requires_grad=False)
        self.gamma = torch.nn.Parameter(torch.zeros(n_head))

        self.linear_out = torch.nn.Linear((c_z + c_ipa + n_v_points * 4)*n_head, c_s)
        self.softplus = torch.nn.Softplus()
    

    def forward(self, s, z, T: Rigid, seq_mask):
        """
        Algorithm 22: Invariant point attention (IPA)

        s: (B, i, c)
        z: (B, i, j, c)
        T: Rigid, (B, i) -> transformation object
        
        return: (B, i, c)
        """

        # B, i, _ = s.shape
        s_shape = s.shape[:-1]
        h = self.n_head
        c = self.c_ipa
        q, kv = self.query(s), self.kv(s) # (B, i, c * h)
        kv = kv.view(kv.shape[:-1] + (h, -1))
        k, v = torch.split(kv, c, dim=-1) # Fix c_h!
        q, k, v = q.view(*s_shape, h, c), k.view(*s_shape, h, c), v.view(*s_shape, h, c)

        q_points, kv_points = self.query_points(s), self.kv_points(s)
        kv_points = torch.split(kv_points, kv_points.shape[-1] // 3, dim=-1)
        kv_points = torch.stack(kv_points, dim=-1)
        kv_points = T[..., None].apply(kv_points)

        kv_points = kv_points.view(kv_points.shape[:-2] + (h, -1, 3))
        k_points, v_points = torch.split(
            kv_points, [self.n_qk_points, self.n_v_points], dim=-2
        )
        
        # (B, i, h * q_points, 3), (B, i, h * k_points, 3), (B, i, h * v_points, 3)

        # (B, i, h, q_points, 3)
        q_points = torch.split(q_points, q_points.shape[-1] // 3, dim=-1)
        q_points = torch.stack(q_points, dim=-1)
        q_points = T[..., None].apply(q_points).view(*s_shape, h, self.n_qk_points, 3)

        if q.dim() == 3:
            b = self.bias(z).permute(2, 0, 1) # (i, j, h)
        else:
            b = self.bias(z).permute(0, 3, 1, 2) # (B, i, j, h) # fix here
        
        if q.dim() == 3:
            a = q.permute(1, 0, 2) @ k.permute(1, 2, 0) * ((3*c) ** -0.5) + (3**-0.5)*b # (h, i, j)
        else:
            a = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1) * ((3*c) ** -0.5) + (3**-0.5)*b # (B, h, i, j)

        point_attention = torch.sum((q_points.unsqueeze(-4) - k_points.unsqueeze(-5)) ** 2, dim=-1) # (B, i, j, h, q_points)
        if a.dim() == 3:
            head_weights = self.softplus(self.gamma).view(1, 1, h) # (1, 1, h, 1)
        else:
            head_weights = self.softplus(self.gamma).view(1, 1, 1, h) # (1, 1, 1, h, 1)
        point_attention = torch.sum(point_attention, dim=-1) * (-0.5) # (B, i, j, h)
        point_attention *= head_weights * self.w_c * 3 ** (-0.5)
        
        if a.dim() == 3:
            a += point_attention.permute(2, 0, 1) # (h, i, j)
        else:
            a += point_attention.permute(0, 3, 1, 2) # (B, h, i, j)
        # a *= self.w_L
        mask_bias = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)
        mask_bias = 1e5 * (mask_bias-1)[..., None, :, :]
        a = a + mask_bias
        a = torch.nn.functional.softmax(a, dim=-1) # (B, h, i, j)

        o = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)
        o = o.reshape(*s_shape, h * c) # (B, i, h * c)


        if v_points.dim() == 4:
            v_points = v_points.permute(1, 3, 0, 2) # (h, 3, i, v_points)
        else:
            v_points = v_points.permute(0, 2, 4, 1, 3) # (B, h, 3, i, v_points)
        o_points = torch.sum(a[..., None, :, :, None] * v_points[..., None, :, :], dim=-2)
        # (B, h, 3, i, v_points)

        if o_points.dim() == 4:
            o_points = o_points.permute(2, 0, 3, 1) # (i, h, v_points, 3)
        else:
            o_points = o_points.permute(0, 3, 1, 4, 2) # (B, i, h, v_points, 3)
        o_points = T[..., None, None].invert_apply(o_points)

        o_points_norm = torch.norm(o_points, dim=-1, keepdim=False) # (B, i, h, v_points)
        o_points_norm = torch.clamp(o_points_norm, min=1e-6).view(*s_shape, h * self.n_v_points) # (B, i, h * v_points)
        o_points = o_points.reshape(*s_shape, h * self.n_v_points, 3) # (B, i, h * v_points, 3)

        o_pair = a.transpose(-2, -3) @ z # (B, i, h, j) @ (B, i, j, c) -> (B, i, h, c)
        o_pair = o_pair.view(*s_shape, h * self.n_v_points * c) # (B, i, h * h * c)

        s = self.linear_out(torch.cat([o, *torch.unbind(o_points, dim=-1), o_points_norm, o_pair], dim=-1)) # (B, i, c)

        return s