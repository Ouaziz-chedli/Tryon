"""CPU synthetic unit tests for Phase 3 two-view unlearning (phase3_views + grid + gap fields)."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import MethodConfig
from methods.phase3_views import run_phase3_two_view, view_modes
from run_phase3_views import _build_phase3_specs, _gap_fields
from task_template import preprocess


def test_view_modes_complementary_and_mono():
    assert view_modes("aug", False) == ("moderate", "none")
    assert view_modes("clean", False) == ("none", "moderate")
    assert view_modes("clean", True) == ("none", "none")
    assert view_modes("aug", True) == ("moderate", "moderate")
    with pytest.raises(ValueError):
        view_modes("bogus", False)


def test_build_phase3_specs_grid_shape_and_recipe():
    specs = _build_phase3_specs()
    assert len(specs) == 10
    two_view = [(cfg, v) for cfg, v, mono in specs if not mono]
    mono = [(cfg, v) for cfg, v, m in specs if m]
    assert len(two_view) == 8 and len(mono) == 2
    assert {v for _, v in two_view} == {"clean", "aug"}
    assert {cfg.seed for cfg, _, _ in two_view} == {42, 101}
    assert {cfg.target_forget_ce for cfg, _, _ in two_view} == {0.85, 0.95}
    cfg0, view0, mono0 = specs[0]
    assert mono0 is False and view0 in ("clean", "aug")
    # winning recipe #18 hyperparameters on every config
    for cfg, _, _ in specs:
        assert (cfg.lr, cfg.npo_beta, cfg.retain_kl_weight) == (1e-4, 2.0, 2.8)
        assert (cfg.salun_keep_ratio, cfg.steps, cfg.freeze_until_block) == (0.20, 210, 6)
        assert cfg.head_lr_mult == 1.0 and cfg.loss_kind == "npo"
    for cfg, _, _ in mono:
        assert cfg.target_forget_ce == 0.9 and cfg.seed == 42
        assert cfg.target_ce_upper == pytest.approx(1.35)


def test_gap_fields_from_synthetic_halves():
    halves = {
        "exact": {"ce_mean": 1.00, "ce_std": 1.0, "ce_p50": 0.5, "frac_gt_95": 0.1, "calibrated_proxy_forget_only": 0.3},
        "altered": {"ce_mean": 1.06, "ce_std": 1.1, "ce_p50": 0.8, "frac_gt_95": 0.2, "calibrated_proxy_forget_only": 0.9},
    }
    fields = _gap_fields(halves)
    assert fields["exact_altered_gap"] == pytest.approx(0.06)
    assert fields["gap_below_010"] is True
    assert fields["exact_half"]["ce_mean"] == 1.00 and fields["altered_half"]["ce_mean"] == 1.06
    # negative gap (exact half ahead) still flags via the absolute value
    halves["altered"]["ce_mean"] = 0.80
    assert _gap_fields(halves)["gap_below_010"] is False
    empty = _gap_fields(None)
    assert empty["exact_altered_gap"] is None and empty["gap_below_010"] is None


class TinyHead(torch.nn.Module):
    """Minimal ViT stand-in: trainable body + head so SalUn/absent-row plumbing runs."""

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.body = torch.nn.Linear(3 * 224 * 224, 16)
        self.head = torch.nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.body(x.flatten(1))))


def _tiny_cfg() -> MethodConfig:
    return MethodConfig(
        name="p3tiny", strategy="salun_npo", steps=2, batch_size=3, lr=1e-3,
        npo_beta=2.0, retain_kl_weight=2.8, l2_weight=2e-3, salun_keep_ratio=0.5,
        target_forget_ce=9.0, seed=0,
    )


def _tiny_inputs():
    torch.manual_seed(7)
    f_x = torch.randint(0, 256, (3, 224, 224, 3), dtype=torch.uint8)
    p_x = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8)
    orig = TinyHead().eval()
    with torch.no_grad():
        orig_f = orig(preprocess(f_x)).float().cpu()
        orig_p = orig(preprocess(p_x)).float().cpu()
    absent = torch.tensor([False, True, False, False])
    return orig, f_x, orig_f, p_x, orig_p, absent


def test_run_phase3_two_view_updates_and_keeps_absent_rows():
    orig, f_x, orig_f, p_x, orig_p, absent = _tiny_inputs()
    cfg = _tiny_cfg()
    student = run_phase3_two_view(orig, f_x, orig_f, p_x, orig_p, absent, cfg, forget_view="aug")
    assert not student.training
    assert torch.allclose(student.head.weight[1], orig.head.weight[1])
    assert torch.allclose(student.head.bias[1], orig.head.bias[1])
    moved = any(
        not torch.allclose(p, dict(orig.named_parameters())[n])
        for n, p in student.named_parameters()
    )
    assert moved, "two-view training must move at least one trainable parameter"


def test_run_phase3_mono_view_runs_and_preserves_absent_rows():
    orig, f_x, orig_f, p_x, orig_p, absent = _tiny_inputs()
    student = run_phase3_two_view(orig, f_x, orig_f, p_x, orig_p, absent, _tiny_cfg(), forget_view="clean", mono=True)
    assert torch.allclose(student.head.weight[1], orig.head.weight[1])
    assert torch.allclose(student.head.bias[1], orig.head.bias[1])
