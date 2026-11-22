"""CPU synthetic unit tests for diagnostic scripts (run_erase_diff.py, run_per_half_diagnostic.py)."""

import pytest
import torch

from run_erase_diff import (
    ABSENT_ROW_TOL,
    analyze_head,
    finalize_ranked,
    split_group,
    tensor_delta_norms,
)


def test_split_group_covers_all_module_kinds():
    assert split_group("pos_embed") == ("pos_embed", "")
    assert split_group("cls_token") == ("cls_token", "")
    assert split_group("head.weight") == ("head.weight", "")
    assert split_group("patch_embed.proj.weight") == ("patch_embed", "proj")
    assert split_group("blocks.3.attn.proj.weight") == ("blocks.3", "attn.proj")
    assert split_group("blocks.11.mlp.fc2.weight") == ("blocks.11", "mlp.fc2")
    assert split_group("blocks.0.ls_1.gamma") == ("blocks.0", "ls_1")
    assert split_group("blocks.0.norm1.weight") == ("blocks.0", "norm1")


def test_tensor_delta_norms_relative_and_infinite():
    orig = torch.ones(4)
    delta, norms = tensor_delta_norms(orig, orig + torch.tensor([0.0, 0.0, 0.0, 3.0]))
    assert torch.allclose(delta, torch.tensor([0.0, 0.0, 0.0, 3.0]))
    assert norms["l2"] == pytest.approx(3.0)
    assert norms["rel"] == pytest.approx(3.0 / 2.0)
    assert norms["lInf"] == pytest.approx(3.0)
    _, zero = tensor_delta_norms(torch.zeros(3), torch.zeros(3))
    assert zero["rel"] == 0.0


def test_finalize_ranked_sorts_by_relative_norm_desc():
    sums = {"small": [1.0, 100.0], "big": [4.0, 4.0], "zero": [0.0, 5.0]}
    ranked = finalize_ranked(sums)
    assert [e["group"] for e in ranked] == ["big", "small", "zero"]
    assert ranked[0]["rel"] == pytest.approx(1.0)


def test_analyze_head_flags_absent_row_leak():
    active = torch.tensor([True, False, False])
    delta_w = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1e-9, 0.0]])
    delta_b = torch.zeros(3)
    report = analyze_head(delta_w, delta_b, active)
    assert report["num_active"] == 1 and report["num_absent"] == 2
    assert report["absent_integrity_ok"] is True
    assert report["active_row_l2_max"] == pytest.approx(1.0)
    leak_w = delta_w.clone()
    leak_w[2, 0] = ABSENT_ROW_TOL * 10
    assert analyze_head(leak_w, delta_b, active)["absent_integrity_ok"] is False
