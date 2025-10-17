"""Feature utility functions for the standalone AlphaFold2 model."""

import torch
import torch.nn as nn

from module.rigid_utils import Rigid
from module.tensor_utils import batched_gather


def atom14_to_atom37(atom14: torch.Tensor, batch: dict) -> torch.Tensor:
    """Maps atom14 positions to atom37 positions using batch metadata."""
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def build_extra_msa_feat(batch: dict) -> torch.Tensor:
    """Builds extra MSA features following the OpenFold reference implementation."""
    msa_1hot = nn.functional.one_hot(batch["extra_msa"], 23)
    msa_feat = [
        msa_1hot,
        batch["extra_has_deletion"].unsqueeze(-1),
        batch["extra_deletion_value"].unsqueeze(-1),
    ]
    return torch.cat(msa_feat, dim=-1)


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
):
    """Converts predicted torsion angles to rigid frames."""
    rigid_type = type(r)

    default_4x4 = default_frames[aatype, ...]
    default_r = rigid_type.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    all_rots = alpha.new_zeros(default_r.shape + (4, 4))
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:3] = alpha

    all_rots = rigid_type.from_tensor_4x4(all_rots)
    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = rigid_type.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor:
    """Converts rigid frames to atom14 coordinates using literature positions."""
    default_4x4 = default_frames[aatype, ...]
    group_mask = group_idx[aatype, ...]

    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    t_atoms_to_global = r[..., None, :] * group_mask
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
