"""Escanade-guided weight surgery (no training) for Track 3.

The escanade (run_erase_diff.py on I2R_f6_r4_ce18_fl085) localizes instance memory in
head.weight/head.bias and blocks 7-11 mlp.fc2.bias / attn.proj.bias, whose deltas from
orig are quasi rank-1 (top-1 SVD energy 37-63%). Surgery transplants a dosed rank-1
piece of that erase movement into a moderate in-window candidate, optionally combined
with a (1 - beta) scaling of the 45 active head rows. The 55 absent head rows/bias
entries and every patch_embed tensor stay bit-identical to the candidate input.
"""

from dataclasses import dataclass

import torch

LATE_BLOCKS = range(7, 12)


@dataclass(frozen=True)
class SurgeryConfig:
    """One no-train intervention: erase-dose alpha and/or active head-row shrink beta."""

    name: str
    rank1_alpha: float = 0.0
    head_beta: float = 0.0


def target_keys() -> list[str]:
    """Escanade instance-memory carriers: head + late-block fc2/proj biases."""
    keys = ["head.weight", "head.bias"]
    for block in LATE_BLOCKS:
        keys.append(f"blocks.{block}.mlp.fc2.bias")
        keys.append(f"blocks.{block}.attn.proj.bias")
    return keys


def top_rank1(delta: torch.Tensor) -> torch.Tensor:
    """Dominant rank-1 component of a delta: the vector for 1D, top SVD pair for 2D."""
    if delta.dim() == 1:
        return delta
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    return (s[0] * torch.outer(u[:, 0], vh[0, :])).to(delta.dtype)


def extract_erase_directions(
    orig_sd: dict[str, torch.Tensor],
    extreme_sd: dict[str, torch.Tensor],
    active_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Rank-1 erase directions from (extreme - orig); absent head entries zeroed first."""
    absent = ~active_mask
    missing = [key for key in target_keys() if key not in orig_sd or key not in extreme_sd]
    if missing:
        raise KeyError(f"erase-direction extraction lacks targets: {missing}")
    directions = {}
    for key in target_keys():
        delta = extreme_sd[key].float() - orig_sd[key].float()
        if key.startswith("head."):
            delta[absent] = 0.0
        directions[key] = top_rank1(delta)
    return directions


def dampen_rank1(
    sd: dict[str, torch.Tensor],
    directions: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Dampen instance memory: add the alpha-dosed rank-1 erase movement to each target."""
    if alpha <= 0.0:
        return sd
    out = dict(sd)
    for key, direction in directions.items():
        out[key] = sd[key] + alpha * direction.to(sd[key].dtype)
    return out


def scale_active_head_rows(
    sd: dict[str, torch.Tensor],
    active_mask: torch.Tensor,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Shrink the active head rows by (1 - beta); absent rows are never touched."""
    if beta <= 0.0:
        return sd
    weight = sd["head.weight"].clone()
    weight[active_mask] = weight[active_mask] * (1.0 - beta)
    return {**sd, "head.weight": weight}


def restore_absent_head(
    sd: dict[str, torch.Tensor],
    ref_sd: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Copy the 55 absent head rows/bias entries back from ref_sd (rank-1 leaks there)."""
    out = dict(sd)
    for key in ("head.weight", "head.bias"):
        tensor = out[key].clone()
        tensor[absent_mask] = ref_sd[key][absent_mask]
        out[key] = tensor
    return out


def guard_violations(
    sd: dict[str, torch.Tensor],
    ref_sd: dict[str, torch.Tensor],
    active_mask: torch.Tensor,
) -> list[str]:
    """Bit-integrity guards: absent head rows/bias and every patch_embed tensor."""
    absent = ~active_mask
    bad = []
    for key in ("head.weight", "head.bias"):
        if not torch.equal(sd[key][absent], ref_sd[key][absent]):
            bad.append(f"{key}[absent]")
    for key, ref in ref_sd.items():
        if key.startswith("patch_embed") and not torch.equal(sd[key], ref):
            bad.append(key)
    return bad


def run_head_surgery(
    candidate_sd: dict[str, torch.Tensor],
    directions: dict[str, torch.Tensor],
    cfg: SurgeryConfig,
    active_mask: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Apply one SurgeryConfig; returns (new state dict, guard violations vs candidate)."""
    if cfg.rank1_alpha < 0.0 or cfg.head_beta < 0.0:
        raise ValueError(f"negative surgery dose: {cfg}")
    if cfg.rank1_alpha == 0.0 and cfg.head_beta == 0.0:
        raise ValueError(f"degenerate surgery config (nothing to do): {cfg}")
    missing = [key for key in directions if key not in candidate_sd]
    if missing:
        raise KeyError(f"candidate state dict lacks surgery targets: {missing}")
    if candidate_sd["head.weight"].shape[0] != len(active_mask):
        raise ValueError("active_mask length does not match head rows")
    absent = ~active_mask
    sd = dampen_rank1(candidate_sd, directions, cfg.rank1_alpha)
    sd = scale_active_head_rows(sd, active_mask, cfg.head_beta)
    sd = restore_absent_head(sd, candidate_sd, absent)
    return sd, guard_violations(sd, candidate_sd, active_mask)


def build_surgery_grid() -> list[SurgeryConfig]:
    """8 escanade-guided configs: 2 rank-1 doses, 2 head scalings, 4 combinations."""
    cfgs = [SurgeryConfig(f"HS_a{a:.1f}", rank1_alpha=a) for a in (0.3, 0.5)]
    cfgs += [SurgeryConfig(f"HS_b{b:.2f}", head_beta=b) for b in (0.05, 0.10)]
    cfgs += [
        SurgeryConfig(f"HS_a{a:.1f}_b{b:.2f}", rank1_alpha=a, head_beta=b)
        for a in (0.3, 0.5)
        for b in (0.05, 0.10)
    ]
    return cfgs
