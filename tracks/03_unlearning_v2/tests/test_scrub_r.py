"""Unit tests for true SCRUB+R with Patch-Dropout difficulty weighting."""

import torch

from methods.scrub_r import (
    ScrubRConfig,
    _select_rewind_snapshot,
    compute_pd_difficulty_ratio,
    scrub_max_loss,
)
from run_scrub_r import round1_candidates


def test_compute_pd_difficulty_ratio_ordering_and_bounds() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    pd_logits = torch.zeros(4, 5)
    pd_logits[torch.arange(4), labels] = torch.tensor([2.5, 1.8, 1.0, 0.2])
    ce_pd, ratio = compute_pd_difficulty_ratio(pd_logits, labels)
    assert ce_pd.shape == (4,) and ratio.shape == (4,)
    assert ratio[0] < ratio[1] < ratio[2] < ratio[3]
    assert float(ratio.min()) >= 0.25 and float(ratio.max()) <= 2.5


def test_scrub_max_loss_npo_and_kl_ascent_finite_and_gated() -> None:
    torch.manual_seed(0)
    orig = torch.zeros(4, 6)
    labels = torch.tensor([0, 1, 2, 3])
    orig[torch.arange(4), labels] = 6.0
    diff_ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
    for kind in ("npo", "kl_ascent"):
        sub = (orig + 0.25 * torch.randn_like(orig)).detach().requires_grad_(True)
        cfg = ScrubRConfig(name="test", max_loss_kind=kind, target_forget_ce=1.5)
        loss = scrub_max_loss(sub, orig, labels, diff_ratio, cfg)
        loss.backward()
        assert torch.isfinite(loss) and sub.grad is not None and sub.grad.abs().sum() > 0


def test_scrub_max_loss_zero_when_above_gate() -> None:
    orig = torch.zeros(4, 6)
    labels = torch.tensor([0, 1, 2, 3])
    sub = torch.zeros(4, 6, requires_grad=True)  # CE = log(6) ~= 1.79 > 0.5
    cfg = ScrubRConfig(name="test", target_forget_ce=0.5, pd_gate_scale=0.0)
    loss = scrub_max_loss(sub, orig, labels, torch.ones(4), cfg)
    assert float(loss.item()) == 0.0


def test_select_rewind_snapshot_picks_best_or_last() -> None:
    snaps = [
        {"epoch": 0, "p50": 0.05, "rewind_dist": 0.1},
        {"epoch": 1, "p50": 1.02, "rewind_dist": 0.25},
        {"epoch": 2, "p50": 1.40, "rewind_dist": 0.90},
    ]
    assert _select_rewind_snapshot(snaps, use_rewind=True)["epoch"] == 1
    assert _select_rewind_snapshot(snaps, use_rewind=False)["epoch"] == 2


def test_round1_candidates_unique_and_valid() -> None:
    cands = round1_candidates()
    names = [c.name for c in cands]
    assert len(cands) == 5 and len(set(names)) == 5
    assert any(c.max_loss_kind == "kl_ascent" for c in cands)
    assert any(not c.use_rewind for c in cands)


if __name__ == "__main__":
    test_compute_pd_difficulty_ratio_ordering_and_bounds()
    test_scrub_max_loss_npo_and_kl_ascent_finite_and_gated()
    test_scrub_max_loss_zero_when_above_gate()
    test_select_rewind_snapshot_picks_best_or_last()
    test_round1_candidates_unique_and_valid()
    print("SCRUB_R_TESTS_OK")
