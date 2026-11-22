"""CPU synthetic unit tests for escanade-guided weight surgery (methods/head_surgery.py)."""

import pytest
import torch

from methods.head_surgery import (
    SurgeryConfig,
    build_surgery_grid,
    dampen_rank1,
    extract_erase_directions,
    guard_violations,
    restore_absent_head,
    run_head_surgery,
    scale_active_head_rows,
    target_keys,
    top_rank1,
)

ACTIVE = torch.tensor([True, True, False, False])
ABSENT = ~ACTIVE


def _mini_sd(fill: float = 1.0) -> dict[str, torch.Tensor]:
    """4-class / dim-3 state dict with every surgery target plus guard tensors."""
    return {
        "head.weight": torch.full((4, 3), fill),
        "head.bias": torch.full((4,), fill * 0.1),
        **{f"blocks.{b}.{p}.bias": torch.zeros(3) for b in (7, 8, 9, 10, 11) for p in ("mlp.fc2", "attn.proj")},
        "patch_embed.proj.weight": torch.eye(3),
        "pos_embed": torch.full((1, 2, 3), fill),
    }


def _directions() -> dict[str, torch.Tensor]:
    orig = _mini_sd(0.0)
    extreme = _mini_sd(0.0)
    extreme["head.bias"][:2] = torch.tensor([0.5, 0.0])
    extreme["blocks.7.mlp.fc2.bias"] = torch.tensor([0.2, 0.0, 0.0])
    extreme["blocks.9.attn.proj.bias"] = torch.tensor([0.0, 0.4, 0.0])
    # simulates a leaked absent row in the extreme checkpoint
    extreme["head.weight"][:2] = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    extreme["head.weight"][3] = 5.0
    return extract_erase_directions(orig, extreme, ACTIVE)


def test_target_keys_cover_head_and_late_blocks_only():
    keys = target_keys()
    assert keys[:2] == ["head.weight", "head.bias"]
    assert "blocks.7.mlp.fc2.bias" in keys and "blocks.11.attn.proj.bias" in keys
    assert "blocks.6.mlp.fc2.bias" not in keys and "blocks.12.mlp.fc2.bias" not in keys
    assert not any(k.endswith(".weight") and k != "head.weight" for k in keys)
    assert len(keys) == 12


def test_top_rank1_exact_rank1_matrix_and_1d_passthrough():
    u = torch.tensor([1.0, 0.0, 1.0])
    v = torch.tensor([2.0, 1.0, 0.0, 3.0])
    delta = 2.5 * torch.outer(u, v)
    assert torch.allclose(top_rank1(delta), delta, atol=1e-5)
    vec = torch.tensor([0.3, -1.0, 2.0])
    assert torch.equal(top_rank1(vec), vec)


def test_extract_erase_directions_zeroes_absent_head_entries():
    directions = _directions()
    assert torch.all(directions["head.weight"][ABSENT] == 0.0)
    assert torch.all(directions["head.bias"][ABSENT] == 0.0)
    assert torch.allclose(directions["head.bias"][:2], torch.tensor([0.5, 0.0]))
    assert torch.allclose(directions["blocks.7.mlp.fc2.bias"], torch.tensor([0.2, 0.0, 0.0]))


def test_extract_erase_directions_raises_on_missing_target():
    with pytest.raises(KeyError):
        extract_erase_directions({}, {}, ACTIVE)


def test_dampen_rank1_doses_only_targets_and_zero_alpha_is_noop():
    sd = _mini_sd(1.0)
    directions = _directions()
    assert dampen_rank1(sd, directions, 0.0) is sd
    out = dampen_rank1(sd, directions, 0.5)
    assert torch.allclose(out["head.bias"][:2], sd["head.bias"][:2] + 0.5 * directions["head.bias"][:2])
    assert torch.allclose(out["blocks.7.mlp.fc2.bias"], torch.tensor([0.1, 0.0, 0.0]))
    assert out["pos_embed"] is sd["pos_embed"] and out["patch_embed.proj.weight"] is sd["patch_embed.proj.weight"]


def test_scale_active_head_rows_shrinks_active_only():
    sd = _mini_sd(2.0)
    assert scale_active_head_rows(sd, ACTIVE, 0.0) is sd
    out = scale_active_head_rows(sd, ACTIVE, 0.10)
    assert torch.allclose(out["head.weight"][ACTIVE], torch.full((2, 3), 1.8))
    assert torch.equal(out["head.weight"][ABSENT], sd["head.weight"][ABSENT])
    assert out["head.bias"] is sd["head.bias"]


def test_restore_absent_head_is_bitwise():
    sd = _mini_sd(1.0)
    sd["head.weight"][3] = 99.0  # simulated rank-1 leak into an absent row
    sd["head.bias"][2:] = 7.0
    ref = _mini_sd(1.0)
    out = restore_absent_head(sd, ref, ABSENT)
    assert torch.equal(out["head.weight"][ABSENT], ref["head.weight"][ABSENT])
    assert torch.equal(out["head.bias"][ABSENT], ref["head.bias"][ABSENT])


def test_run_head_surgery_end_to_end_guards_hold():
    candidate = _mini_sd(1.0)
    directions = _directions()
    cfg = SurgeryConfig("HS_test", rank1_alpha=0.5, head_beta=0.10)
    sd, violations = run_head_surgery(candidate, directions, cfg, ACTIVE)
    assert violations == []
    # absent rows / bias and patch_embed bit-identical to the candidate input
    assert torch.equal(sd["head.weight"][ABSENT], candidate["head.weight"][ABSENT])
    assert torch.equal(sd["head.bias"][ABSENT], candidate["head.bias"][ABSENT])
    assert torch.equal(sd["patch_embed.proj.weight"], candidate["patch_embed.proj.weight"])
    # compose order: dampen first, then scale active rows
    expected_active = (candidate["head.weight"] + 0.5 * directions["head.weight"])[:2] * 0.9
    assert torch.allclose(sd["head.weight"][ACTIVE], expected_active, atol=1e-6)
    assert sd["pos_embed"] is candidate["pos_embed"]


def test_run_head_surgery_rejects_degenerate_and_negative_configs():
    sd, directions = _mini_sd(1.0), _directions()
    with pytest.raises(ValueError):
        run_head_surgery(sd, directions, SurgeryConfig("noop"), ACTIVE)
    with pytest.raises(ValueError):
        run_head_surgery(sd, directions, SurgeryConfig("neg", rank1_alpha=-0.1), ACTIVE)


def test_run_head_surgery_raises_on_missing_candidate_target():
    sd, directions = _mini_sd(1.0), _directions()
    del sd["blocks.9.attn.proj.bias"]
    with pytest.raises(KeyError):
        run_head_surgery(sd, directions, SurgeryConfig("HS_x", rank1_alpha=0.3), ACTIVE)


def test_guard_violations_flag_absent_and_patch_tampering():
    ref = _mini_sd(1.0)
    clean, _ = run_head_surgery(ref, _directions(), SurgeryConfig("HS_g", rank1_alpha=0.3), ACTIVE)
    assert guard_violations(clean, ref, ACTIVE) == []
    tampered = {**clean, "patch_embed.proj.weight": clean["patch_embed.proj.weight"] + 1.0}
    assert "patch_embed.proj.weight" in guard_violations(tampered, ref, ACTIVE)
    bad_head = {**clean, "head.weight": clean["head.weight"].clone()}
    bad_head["head.weight"][3] += 0.5
    assert "head.weight[absent]" in guard_violations(bad_head, ref, ACTIVE)


def test_build_surgery_grid_yields_8_unique_configs():
    cfgs = build_surgery_grid()
    names = [c.name for c in cfgs]
    assert len(cfgs) == 8 and len(set(names)) == 8
    alphas = sorted({c.rank1_alpha for c in cfgs})
    betas = sorted({c.head_beta for c in cfgs})
    assert alphas == [0.0, 0.3, 0.5] and betas == [0.0, 0.05, 0.1]
    combos = [c for c in cfgs if c.rank1_alpha > 0 and c.head_beta > 0]
    assert len(combos) == 4
    assert "HS_a0.3" in names and "HS_b0.05" in names and "HS_a0.5_b0.10" in names
