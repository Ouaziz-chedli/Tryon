"""Unit tests for SCRUB+R Round 2 PD-depth grid (run where torch is available)."""

from dataclasses import asdict

import math

import torch
import torch.nn.functional as F
from augment import augment_training_batch

from methods.scrub_r import ScrubRConfig, scrub_max_loss
from methods.common import npo_loss
from methods.patch_drop_targets import calibrate_per_image_targets
from methods.scrub_r_rest import (
    GOLD_BELL_MEAN,
    GOLD_BELL_STD,
    GateQConfig,
    PD_AFFINE_AUTO,
    REWIND_MAX_DELTA_L2,
    REWIND_MAX_PROXY_KL,
    ScrubRBudgetedConfig,
    ScrubRPdTargetConfig,
    ScrubRRestConfig,
    _apply_gate_transforms,
    _gold_lognormal_params,
    _select_rewind_snapshot_budgeted,
    affine_calibrated_targets,
    gold_bell_quantile,
    npo_per_image_gate_loss,
    pd_target_mimic_loss,
    quantile_calibrated_targets,
    rest_kl_loss,
    scrub_max_rest_loss,
)
from run_scrub_r2 import (
    CHAMPION_CKPT,
    GOLD_MU,
    GOLD_SIGMA,
    PD_SEED,
    _gold_affine,
    _in_window,
    _spearman_corr,
    round2_candidates,
    round3_candidates,
    round4_candidates,
    round5_candidates,
    round6_candidates,
    round7_candidates,
    round8_candidates,
    round9_candidates,
    round10_candidates,
    round11_candidates,
    round12_candidates,
    round13_candidates,
    round14_candidates,
    round15_candidates,
    round16_candidates,
    round17_candidates,
)


def test_round2_candidates_grid_and_control() -> None:
    cands = round2_candidates()
    assert len(cands) == 7 and len({c.name for c in cands}) == 7
    grid = cands[:6]
    assert {(c.target_forget_ce, c.pd_alpha) for c in grid} == {
        (ce, a) for ce in (2.0, 2.4, 2.8) for a in (0.6, 1.0)
    }
    assert all(c.max_loss_kind == "npo" and c.use_rewind for c in cands)
    assert all(c.target_mu_sig == 3.84 and c.target_p50 == 1.00 for c in cands)
    assert all(c.max_epochs >= 8 and c.max_epochs < c.total_epochs for c in cands)
    ctrl = cands[-1]
    mid = next(c for c in grid if c.target_forget_ce == 2.4 and c.pd_alpha == 1.0)
    assert ctrl.pd_alpha == 0.0 and ctrl.pd_gate_scale == 0.0
    assert ctrl.target_forget_ce == mid.target_forget_ce == 2.4
    assert ctrl.max_epochs == mid.max_epochs and ctrl.total_epochs == mid.total_epochs


def test_high_target_keeps_gate_open_below_ce() -> None:
    labels = torch.tensor([0, 1, 2, 3])
    orig = torch.zeros(4, 6)
    orig[torch.arange(4), labels] = 6.0
    sub = torch.zeros(4, 6, requires_grad=True)  # CE = log(6) ~= 1.79
    high = ScrubRConfig(name="t", target_forget_ce=2.8)
    assert float(scrub_max_loss(sub, orig, labels, torch.ones(4), high)) > 0.0
    low = ScrubRConfig(name="t", target_forget_ce=0.5)
    assert float(scrub_max_loss(sub, orig, labels, torch.ones(4), low)) == 0.0


def test_pd_weights_increase_with_difficulty() -> None:
    labels = torch.tensor([0, 1])
    orig = torch.zeros(2, 5)
    orig[torch.arange(2), labels] = 6.0
    torch.manual_seed(0)
    sub = (orig + 0.25 * torch.randn_like(orig)).detach().requires_grad_(True)
    cfg = ScrubRConfig(name="t", target_forget_ce=2.8)
    easy = scrub_max_loss(sub, orig, labels, torch.ones(2), cfg)
    hard = scrub_max_loss(sub, orig, labels, torch.tensor([2.5, 2.5]), cfg)
    assert float(hard) > float(easy) > 0.0


def test_control_gate_ignores_diff_ratio() -> None:
    labels = torch.tensor([0, 1])
    orig = torch.zeros(2, 5)
    orig[torch.arange(2), labels] = 6.0
    torch.manual_seed(0)
    sub = (orig + 0.25 * torch.randn_like(orig)).detach().requires_grad_(True)
    ctrl = ScrubRConfig(name="t", target_forget_ce=2.4, pd_alpha=0.0, pd_gate_scale=0.0)
    easy = scrub_max_loss(sub, orig, labels, torch.tensor([0.25, 0.25]), ctrl)
    hard = scrub_max_loss(sub, orig, labels, torch.tensor([2.5, 2.5]), ctrl)
    assert float(easy) == float(hard) and float(easy) > 0.0


def test_in_window_bounds() -> None:
    ok = {
        "mu_sigma": 3.90, "ce_std": 2.25, "ce_p50": 1.00,
        "proxy_kl_t2": 0.050, "delta_l2_total": 4.50, "frac_gt_95": 0.20,
    }
    assert _in_window(ok)
    for key, bad in (
        ("mu_sigma", 3.50), ("mu_sigma", 4.45), ("ce_std", 2.00),
        ("ce_p50", 0.70), ("ce_p50", 1.25), ("proxy_kl_t2", 0.057),
        ("delta_l2_total", 4.75), ("frac_gt_95", 0.31),
    ):
        assert not _in_window(ok | {key: bad}), key


def test_spearman_corr_sign_and_pd_seed() -> None:
    a = torch.arange(5).float()
    assert _spearman_corr(a, a) == 1.0
    assert _spearman_corr(a, -a) == -1.0
    assert PD_SEED == 42


def test_round3_candidates_deep_amplitude() -> None:
    cands = round3_candidates()
    assert [c.name for c in cands] == ["SR8_balanced", "SR8_push_mu", "SR8_high_pd"]
    for c in cands:
        assert c.total_epochs == 16 and c.target_mu_sig == 4.20 and c.use_rewind
        assert c.retain_kl_weight == 2.4 and c.l2_weight == 1.6e-3 and c.seed == 42
        assert c.freeze_until_block == 6 and c.salun_keep_ratio == 0.20
        assert c.aug_mode == "moderate" and c.max_loss_kind == "npo"
    balanced, push_mu, high_pd = cands
    assert (balanced.target_forget_ce, balanced.pd_alpha, balanced.npo_beta, balanced.lr_max) == (
        2.6, 0.8, 1.4, 1.5e-4,
    )
    assert (balanced.max_epochs, balanced.max_steps_per_epoch) == (11, 22)
    assert (push_mu.target_forget_ce, push_mu.pd_alpha, push_mu.npo_beta, push_mu.lr_max) == (
        2.8, 0.6, 1.3, 1.7e-4,
    )
    assert (push_mu.max_epochs, push_mu.max_steps_per_epoch) == (12, 24)
    assert (high_pd.target_forget_ce, high_pd.pd_alpha, high_pd.npo_beta, high_pd.lr_max) == (
        2.6, 1.0, 1.35, 1.6e-4,
    )
    assert (high_pd.max_epochs, high_pd.max_steps_per_epoch) == (12, 22)
    assert GOLD_MU == 1.70 and GOLD_SIGMA == 2.25


def test_round4_candidates_rest_and_pd_target() -> None:
    cands = round4_candidates()
    assert [c.name for c in cands] == [
        "SR9_rest02", "SR9_rest035", "SR9_rest05", "SR10_pdt05", "SR10_pdt10",
    ]
    assert [c.rest_kl_weight for c in cands[:3]] == [0.20, 0.35, 0.50]
    assert [c.pd_target_weight for c in cands[3:]] == [0.50, 1.00]
    balanced = round3_candidates()[0]
    for c in cands:
        assert c.target_forget_ce == balanced.target_forget_ce == 2.6
        assert (c.max_epochs, c.max_steps_per_epoch, c.total_epochs) == (11, 22, 16)
        assert c.pd_alpha == balanced.pd_alpha and c.npo_beta == balanced.npo_beta
        assert c.lr_max == balanced.lr_max and c.target_mu_sig == 4.20
        assert isinstance(c, (ScrubRRestConfig, ScrubRPdTargetConfig))
    assert all(isinstance(c, ScrubRRestConfig) for c in cands[:3])
    for c in cands[3:]:
        assert isinstance(c, ScrubRPdTargetConfig)
        assert c.pd_target_mean_ce == 1.70


def test_rest_kl_loss_zero_when_identical_and_ignores_label_logit() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1])
    orig = torch.randn(2, 5)
    same = orig.clone().requires_grad_(True)
    assert float(rest_kl_loss(same, orig, labels, 0.35)) == 0.0
    shifted = orig.clone()
    shifted[0, labels[0]] += 5.0  # only the label logit moves -> rest distribution unchanged
    assert float(rest_kl_loss(shifted, orig, labels, 0.35)) == 0.0
    moved = (orig + 0.5 * torch.randn_like(orig)).detach().requires_grad_(True)
    loss = rest_kl_loss(moved, orig, labels, 0.35)
    assert float(loss) > 0.0
    loss.backward()
    assert moved.grad is not None and torch.isfinite(moved.grad).all()
    assert float(moved.grad.abs().sum()) > 0.0


def test_scrub_max_rest_loss_is_npo_plus_rest() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2, 3])
    orig = torch.randn(4, 6)
    cfg = ScrubRRestConfig(name="t", target_forget_ce=0.5, rest_kl_weight=0.35)
    sub = (orig + 0.25 * torch.randn_like(orig)).detach().requires_grad_(True)
    total = scrub_max_rest_loss(sub, orig, labels, torch.ones(4), cfg)
    npo = scrub_max_loss(sub, orig, labels, torch.ones(4), cfg)
    rest = rest_kl_loss(sub, orig, labels, cfg.rest_kl_weight)
    assert torch.isfinite(total)
    assert abs(float(total - (npo + rest))) < 1e-6
    assert float(total) > float(npo)  # gate-closed NPO still leaves the rest pin active


def test_pd_target_mimic_loss_zero_at_target_and_scales_with_weight() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1])
    orig = torch.randn(2, 5)
    lse_rest = torch.logsumexp(orig.scatter(1, labels.unsqueeze(1), -1e9), dim=-1, keepdim=True)
    p_tgt = torch.tensor([[0.30], [0.45]])
    z_tgt = orig.scatter(1, labels.unsqueeze(1), lse_rest + torch.log(p_tgt / (1.0 - p_tgt)))
    ce_tgt = -torch.log(p_tgt).squeeze(1)
    cfg_w1 = ScrubRPdTargetConfig(name="t", pd_target_weight=1.0)
    cfg_w05 = ScrubRPdTargetConfig(name="t", pd_target_weight=0.5)
    at_target = z_tgt.clone().requires_grad_(True)
    assert float(pd_target_mimic_loss(at_target, z_tgt, ce_tgt, labels, cfg_w1)) < 1e-4  # exact-zero is fp-unreachable on real torch (reconstructed z_tgt via scatter/lse)
    moved = (z_tgt + 0.4 * torch.randn_like(z_tgt)).detach().requires_grad_(True)
    loss_w1 = pd_target_mimic_loss(moved, z_tgt, ce_tgt, labels, cfg_w1)
    loss_w05 = pd_target_mimic_loss(moved, z_tgt, ce_tgt, labels, cfg_w05)
    assert float(loss_w1) > 0.0 and abs(float(loss_w05) - 0.5 * float(loss_w1)) < 1e-6
    loss_w1.backward()
    assert moved.grad is not None and torch.isfinite(moved.grad).all()


def test_calibrated_pd_targets_pin_rest_logits_and_mean_ce() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2, 3])
    orig = torch.randn(4, 6)
    pd_logits = orig + 0.3 * torch.randn_like(orig)
    z_tgt, ce_tgt = calibrate_per_image_targets(pd_logits, orig, labels, target_mean_ce=1.70)
    keep = torch.ones_like(orig, dtype=torch.bool)
    keep.scatter_(1, labels.unsqueeze(1), False)
    assert torch.equal(z_tgt[keep], orig[keep])  # the 99 rest logits are orig verbatim
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    expected = (ce_pd * (1.70 / float(ce_pd.mean()))).clamp(0.05, 8.0)
    assert torch.allclose(ce_tgt, expected)
    assert abs(float(ce_tgt.mean()) - 1.70) < 0.02  # no clamp binds at this scale


def test_budgeted_rewind_selection_filters_pool() -> None:
    snaps = [
        {"epoch": 0, "p50": 0.005, "rewind_dist": 5.00, "proxy_kl": 0.020, "delta_l2": 1.50},
        {"epoch": 1, "p50": 0.050, "rewind_dist": 3.40, "proxy_kl": 0.050, "delta_l2": 3.60},
        {"epoch": 2, "p50": 0.009, "rewind_dist": 0.90, "proxy_kl": REWIND_MAX_PROXY_KL, "delta_l2": 4.60},
        {"epoch": 3, "p50": 0.180, "rewind_dist": 1.10, "proxy_kl": 0.210, "delta_l2": 6.50},
        {"epoch": 4, "p50": 0.200, "rewind_dist": 0.90, "proxy_kl": 0.040, "delta_l2": 5.00},
    ]
    best = _select_rewind_snapshot_budgeted(snaps, use_rewind=True)
    assert best["epoch"] == 2  # deep epochs 3 (pkl) and 4 (dT) excluded; dist picks 2 over 1/0
    assert _select_rewind_snapshot_budgeted(snaps, use_rewind=False)["epoch"] == 4


def test_budgeted_rewind_falls_back_when_no_snapshot_in_budget() -> None:
    out_of_budget = [
        {"epoch": 0, "p50": 0.010, "rewind_dist": 5.00, "proxy_kl": 0.200, "delta_l2": 6.00},
        {"epoch": 1, "p50": 0.300, "rewind_dist": 1.00, "proxy_kl": 0.250, "delta_l2": 7.00},
    ]
    fallback = _select_rewind_snapshot_budgeted(out_of_budget, use_rewind=True)
    assert fallback["epoch"] == 1  # upstream p50 >= 0.15 pool behavior
    assert _select_rewind_snapshot_budgeted(out_of_budget, use_rewind=False)["epoch"] == 1
    boundary = [{"epoch": 5, "p50": 0.400, "rewind_dist": 0.50,
                 "proxy_kl": REWIND_MAX_PROXY_KL, "delta_l2": REWIND_MAX_DELTA_L2}]
    assert _select_rewind_snapshot_budgeted(boundary, use_rewind=True)["epoch"] == 5  # <= inclusive


def test_round5_candidates_flat_gate_braked() -> None:
    cands = round5_candidates()
    assert [c.name for c in cands] == [
        "SR11_a08_kl30", "SR11_a10_kl30", "SR11_a08_kl35", "SR11_a10_kl35_sal15",
    ]
    balanced = round3_candidates()[0]
    for c in cands:
        assert isinstance(c, ScrubRBudgetedConfig)  # routes to the budgeted rewind
        assert c.pd_gate_scale == 0.0 and c.target_forget_ce == 2.6
        assert c.lr_max == 1.5e-4 and c.npo_beta == 1.4  # never push_mu's 1.7e-4
        assert (c.max_epochs, c.max_steps_per_epoch, c.total_epochs) == (11, 22, 16)
        assert c.target_mu_sig == 4.20 and c.use_rewind and c.seed == 42
        assert c.retain_kl_weight in (3.0, 3.5) and c.retain_kl_weight > balanced.retain_kl_weight
    assert [c.pd_alpha for c in cands] == [0.8, 1.0, 0.8, 1.0]
    assert [c.retain_kl_weight for c in cands] == [3.0, 3.0, 3.5, 3.5]
    assert cands[3].salun_keep_ratio == 0.15 and all(c.salun_keep_ratio == 0.20 for c in cands[:3])


def test_round6_candidates_pushing_ascent_budgeted() -> None:
    cands = round6_candidates()
    assert [c.name for c in cands] == ["SR12_kl24_a08", "SR12_kl26_a08", "SR12_kl24_a10"]
    for c in cands:
        assert isinstance(c, ScrubRBudgetedConfig)  # hard pkl/dT rewind filter ON
        assert c.pd_gate_scale == 0.0 and c.target_forget_ce == 2.6
        assert c.lr_max == 1.5e-4 and c.npo_beta == 1.4
        assert (c.max_epochs, c.max_steps_per_epoch, c.total_epochs) == (11, 22, 16)
        assert c.target_mu_sig == 4.20 and c.use_rewind and c.seed == 42
        assert c.salun_keep_ratio == 0.20  # mask back to the SR8 recipe
    assert [c.retain_kl_weight for c in cands] == [2.4, 2.6, 2.4]  # the pushing dosage
    assert [c.pd_alpha for c in cands] == [0.8, 0.8, 1.0]
    assert all(c.retain_kl_weight < 3.0 for c in cands)  # never round-5's suffocating brake


def test_flat_gate_closes_at_target_while_pd_weights_stay() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2, 3])
    orig = torch.zeros(4, 6)
    orig[torch.arange(4), labels] = 6.0
    sub = torch.zeros(4, 6, requires_grad=True)  # CE = log(6) ~= 1.79
    flat = ScrubRBudgetedConfig(name="t", target_forget_ce=0.8, pd_gate_scale=0.0, pd_alpha=0.8)
    # flat gate: closed at 0.8 even for the hardest image; upstream PD gate would open
    assert float(scrub_max_loss(sub, orig, labels, torch.tensor([2.5, 2.5, 2.5, 2.5]), flat)) == 0.0
    deep = ScrubRBudgetedConfig(name="t", target_forget_ce=2.6, pd_gate_scale=0.0, pd_alpha=0.8)
    easy = scrub_max_loss(sub, orig, labels, torch.tensor([0.25, 0.25, 0.25, 0.25]), deep)
    hard = scrub_max_loss(sub, orig, labels, torch.tensor([2.5, 2.5, 2.5, 2.5]), deep)
    assert float(hard) > float(easy) > 0.0  # gate flat at 2.6, PD weighting still live


def test_round7_candidates_gold_mimic_and_hybrid() -> None:
    cands = round7_candidates()
    assert [c.name for c in cands] == [
        "SR13_pdt15_kl22", "SR13_pdt20_kl24", "SR13_npo_rest35_pd10",
    ]
    low, high, hybrid = cands
    assert isinstance(low, ScrubRPdTargetConfig) and isinstance(high, ScrubRPdTargetConfig)
    assert (low.pd_target_weight, low.pd_target_mean_ce, low.retain_kl_weight) == (1.5, 1.75, 2.2)
    assert (low.max_epochs, low.max_steps_per_epoch, low.total_epochs) == (12, 22, 16)
    assert (high.pd_target_weight, high.pd_target_mean_ce) == (2.0, 1.80)
    assert (high.max_epochs, high.max_steps_per_epoch, high.total_epochs) == (12, 24, 17)
    assert high.retain_kl_weight == 2.4 and high.lr_max == 1.6e-4
    assert isinstance(hybrid, ScrubRRestConfig)  # routes to run_scrub_r_rest, no wrapper
    assert (hybrid.pd_gate_scale, hybrid.pd_alpha, hybrid.rest_kl_weight) == (0.0, 1.0, 0.35)
    assert (hybrid.max_epochs, hybrid.max_steps_per_epoch, hybrid.total_epochs) == (11, 22, 16)
    assert hybrid.lr_max == 1.5e-4 and hybrid.retain_kl_weight == 2.4 and hybrid.npo_beta == 1.4
    for c in cands:
        assert c.target_mu_sig == 4.20 and c.target_p50 == 1.00 and c.use_rewind


def test_hybrid_flat_gate_keeps_rest_pin_alive() -> None:
    labels = torch.tensor([0, 1])
    orig = torch.zeros(2, 5)
    orig[torch.arange(2), labels] = 6.0
    sub = torch.tensor(
        [[0.0, 0.5, 0.0, -0.7, 0.0], [0.3, -0.5, 0.0, 0.0, -0.4]], requires_grad=True
    )  # label logits low (CE 1.6-2.0 > 0.8), rest logits moved off orig's uniform
    hybrid = ScrubRRestConfig(name="t", target_forget_ce=0.8, pd_gate_scale=0.0, rest_kl_weight=0.35)
    ratio = torch.tensor([2.5, 2.5])
    total = scrub_max_rest_loss(sub, orig, labels, ratio, hybrid)
    npo = scrub_max_loss(sub, orig, labels, ratio, hybrid)
    rest = rest_kl_loss(sub, orig, labels, hybrid.rest_kl_weight)
    assert float(npo) == 0.0  # flat gate closed even for the hardest image
    assert float(total) == float(rest) and float(rest) > 0.0  # the 99-logit pin stays live


def test_round8_candidates_champion_refinement() -> None:
    cands = round8_candidates()
    assert [c.name for c in cands] == ["SR14_champ_ref05", "SR14_champ_ref10", "SR14_champ_ref03_long"]
    for c in cands:
        assert isinstance(c, ScrubRPdTargetConfig)
        assert c.init_from == CHAMPION_CKPT and c.pd_target_affine == PD_AFFINE_AUTO
        assert c.rest_kl_weight == 0.35 and c.retain_kl_weight == 2.8 and c.retain_ce_weight == 0.15
        assert c.lr_max == 5.0e-5 and c.lr_min == 5.0e-5  # 10x softer, single gentle phase
        assert c.max_steps_per_epoch == 22 and c.max_epochs == c.total_epochs  # every epoch ascends
        assert c.use_rewind and c.target_mu_sig == 4.20 and c.seed == 42
        assert c.rewind_delta_l2_max is None  # round-8: PKL-only budget (champion at dT 4.65)
    assert [c.pd_target_weight for c in cands] == [0.5, 1.0, 0.3]
    assert [c.total_epochs for c in cands] == [5, 5, 8]
    # retro-compat: the SR13 pdt configs keep the fresh-run dT budget
    assert all(c.rewind_delta_l2_max == REWIND_MAX_DELTA_L2 for c in round7_candidates()[:2])


def test_pkl_only_budget_ignores_delta_l2() -> None:
    snaps = [
        {"epoch": 0, "p50": 0.40, "rewind_dist": 3.20, "proxy_kl": 0.050, "delta_l2": 4.65},
        {"epoch": 1, "p50": 0.42, "rewind_dist": 1.10, "proxy_kl": 0.055, "delta_l2": 5.20},
        {"epoch": 2, "p50": 0.45, "rewind_dist": 0.90, "proxy_kl": 0.060, "delta_l2": 5.40},
    ]
    # round-8 mode (delta_l2_max=None): the drifted-dT epoch 1 is selectable, pkl rules
    assert _select_rewind_snapshot_budgeted(snaps, True, delta_l2_max=None)["epoch"] == 1
    # default fresh-run mode: epoch 1 (dT 5.20 > 4.70) and 2 (pkl) stay excluded
    assert _select_rewind_snapshot_budgeted(snaps, True)["epoch"] == 0


def test_gold_affine_maps_pd_distribution_onto_gold() -> None:
    ces = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])  # the too-easy PD-CEs
    labels = torch.arange(5)
    pd_logits = torch.zeros(5, 6)
    pd_logits[torch.arange(5), labels] = torch.log(5.0 / (ces.exp() - 1.0))
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    assert torch.allclose(ce_pd, ces, atol=1e-5)  # fixture: exact CE control
    a, b = _gold_affine(pd_logits, labels)
    assert a > 0.0  # order preserved (rho 0.72 intact)
    mapped = a * ces + b
    assert abs(float(mapped.mean()) - GOLD_MU) < 1e-3
    assert abs(float(mapped.std(correction=0)) - GOLD_SIGMA) < 1e-3
    assert bool((mapped.argsort() == ces.argsort()).all())


def test_affine_calibrated_targets_pin_rest_and_stay_self_consistent() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2, 3])
    orig = torch.randn(4, 6)
    pd_logits = orig + 0.3 * torch.randn_like(orig)
    z_tgt, ce_tgt = affine_calibrated_targets(pd_logits, orig, labels, (2.0, 0.5))
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    assert torch.allclose(ce_tgt, (2.0 * ce_pd + 0.5).clamp(0.05, 8.0), atol=1e-5)
    keep = torch.ones_like(orig, dtype=torch.bool)
    keep.scatter_(1, labels.unsqueeze(1), False)
    assert torch.equal(z_tgt[keep], orig[keep])  # the 99 rest logits are orig verbatim
    ce_at_target = F.cross_entropy(z_tgt, labels, reduction="none")
    assert torch.allclose(ce_at_target, ce_tgt, atol=1e-4)  # targets reachable (no clamp binds)


def test_round9_candidates_quantile_map_keeps_champion_recipe() -> None:
    cands = round9_candidates()
    assert [c.name for c in cands] == ["SR15_qmap_w05", "SR15_qmap_w10", "SR15_qmap_w05_long"]
    for c in cands:
        assert isinstance(c, ScrubRPdTargetConfig)
        assert c.pd_target_quantile and c.pd_target_affine is None  # rank map, not affine
        assert c.init_from == CHAMPION_CKPT and c.rest_kl_weight == 0.35
        assert c.retain_kl_weight == 2.8 and c.retain_ce_weight == 0.15
        assert c.lr_max == 5.0e-5 and c.lr_min == 5.0e-5
        assert c.rewind_delta_l2_max is None  # pkl-only budget kept from round-8
        assert c.max_steps_per_epoch == 22 and c.max_epochs == c.total_epochs
    assert [c.pd_target_weight for c in cands] == [0.5, 1.0, 0.5]
    assert [c.total_epochs for c in cands] == [6, 6, 10]
    # retro-compat: SR14 keeps the affine path, SR13 the plain mean-scaled one
    assert all(not c.pd_target_quantile for c in round8_candidates())
    assert all(not c.pd_target_quantile for c in round7_candidates()[:2])


def test_gold_bell_quantile_lognormal_moments_and_monotonicity() -> None:
    mu_log, sigma_log = _gold_lognormal_params()
    bell = torch.distributions.LogNormal(mu_log, sigma_log)
    assert abs(float(bell.mean) - GOLD_BELL_MEAN) < 1e-5
    assert abs(float(bell.stddev) - GOLD_BELL_STD) < 1e-5
    ranks = torch.linspace(0.01, 0.99, 99)
    targets = gold_bell_quantile(ranks)
    assert bool((targets[1:] > targets[:-1]).all())  # strictly monotone in rank
    assert float(targets.min()) > 0.05  # no clamp ever binds
    median = float(gold_bell_quantile(torch.tensor([0.5])))
    assert abs(median - math.exp(mu_log)) < 1e-6  # lognormal median = exp(mu_log)


def test_quantile_calibrated_targets_rank_map_and_pin() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2, 3])
    orig = torch.randn(4, 6)
    pd_logits = orig + 0.3 * torch.randn_like(orig)
    z_tgt, ce_tgt = quantile_calibrated_targets(pd_logits, orig, labels)
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    assert bool((ce_tgt[ce_pd.argsort()].diff() > 0).all())  # target order == pd order
    ranks = (ce_pd.argsort().argsort().float() + 0.5) / 4.0
    assert torch.allclose(ce_tgt, gold_bell_quantile(ranks), atol=1e-5)
    keep = torch.ones_like(orig, dtype=torch.bool)
    keep.scatter_(1, labels.unsqueeze(1), False)
    assert torch.equal(z_tgt[keep], orig[keep])  # the 99 rest logits are orig verbatim
    ce_at_target = F.cross_entropy(z_tgt, labels, reduction="none")
    assert torch.allclose(ce_at_target, ce_tgt, atol=1e-4)  # targets reachable


def test_round10_candidates_drop_rest_pin_pull() -> None:
    cands = round10_candidates()
    assert [c.name for c in cands] == [
        "SR16_qmap_norest_w05", "SR16_qmap_norest_w10", "SR16_qmap_softrest_w05",
    ]
    assert [c.rest_kl_weight for c in cands] == [0.0, 0.0, 0.05]
    assert [c.pd_target_weight for c in cands] == [0.5, 1.0, 0.5]
    sr15 = round9_candidates()[0]
    for c in cands:
        assert isinstance(c, ScrubRPdTargetConfig)
        assert c.pd_target_quantile and c.pd_target_affine is None  # same quantile map
        assert c.init_from == CHAMPION_CKPT and c.rewind_delta_l2_max is None
        assert c.retain_kl_weight == 2.8 and c.retain_ce_weight == 0.15
        assert c.lr_max == 5.0e-5 and c.lr_min == 5.0e-5
        assert (c.max_epochs, c.total_epochs, c.max_steps_per_epoch) == (6, 6, 22)
    # only the rest-pin weight differs from SR15 -- everything else is verbatim
    assert sr15.rest_kl_weight == 0.35 and all(c.rest_kl_weight < sr15.rest_kl_weight for c in cands)


def test_round11_candidates_soften_train_proxy_retain() -> None:
    cands = round11_candidates()
    assert [c.name for c in cands] == [
        "SR17_rkl05_qmap_w05", "SR17_rkl10_qmap_w05", "SR17_rkl05_qmap_w10",
    ]
    assert [c.retain_kl_weight for c in cands] == [0.5, 1.0, 0.5]
    assert [c.pd_target_weight for c in cands] == [0.5, 0.5, 1.0]
    sr16 = round10_candidates()[0]
    for c in cands:
        assert isinstance(c, ScrubRPdTargetConfig)
        assert c.rest_kl_weight == 0.0 and c.retain_ce_weight == 0.05
        assert c.pd_target_quantile and c.pd_target_affine is None
        assert c.init_from == CHAMPION_CKPT and c.rewind_delta_l2_max is None
        assert c.lr_max == 5.0e-5 and c.lr_min == 5.0e-5
        assert (c.max_epochs, c.total_epochs, c.max_steps_per_epoch) == (6, 6, 22)
        assert c.retain_kl_weight < sr16.retain_kl_weight  # the train-proxy pull loosened
    # everything else verbatim vs SR16-norest (only retain knobs and w move)
    skip = ("name", "retain_kl_weight", "retain_ce_weight", "pd_target_weight")
    sr16_d = {k: v for k, v in asdict(sr16).items() if k not in skip}
    assert all({k: v for k, v in asdict(c).items() if k not in skip} == sr16_d for c in cands)


def test_round12_candidates_pure_ce_match() -> None:
    cands = round12_candidates()
    assert [c.name for c in cands] == ["SR18_cematch_w05", "SR18_cematch_w10", "SR18_cematch_w20"]
    assert [c.pd_target_weight for c in cands] == [0.5, 1.0, 2.0]
    sr17 = round11_candidates()[0]
    for c in cands:
        assert isinstance(c, ScrubRPdTargetConfig)
        assert c.mimic_ce_only  # the ONLY delta vs SR17-rkl05: the disguised rest-pin is gone
        assert c.retain_kl_weight == 0.5 and c.retain_ce_weight == 0.05
        assert c.rest_kl_weight == 0.0 and c.pd_target_quantile and c.pd_target_affine is None
        assert c.init_from == CHAMPION_CKPT and c.rewind_delta_l2_max is None
        assert c.lr_max == 5.0e-5 and c.lr_min == 5.0e-5
        assert (c.max_epochs, c.total_epochs, c.max_steps_per_epoch) == (6, 6, 22)
    # retro-compat: every earlier pdt family keeps the kl_ce loss verbatim
    assert all(not c.mimic_ce_only for c in round11_candidates() + round10_candidates())
    assert all(not c.mimic_ce_only for c in round9_candidates() + round8_candidates())


def test_mimic_ce_only_loss_ignores_target_logits() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1])
    sub = torch.randn(2, 5)
    target_ce = torch.tensor([1.0, 2.0])
    z_a, z_b = torch.randn(2, 5), torch.randn(2, 5)  # wildly different rest geometry
    ce_cfg = ScrubRPdTargetConfig(name="t", pd_target_weight=0.5, mimic_ce_only=True)
    la = pd_target_mimic_loss(sub, z_a, target_ce, labels, ce_cfg)
    lb = pd_target_mimic_loss(sub, z_b, target_ce, labels, ce_cfg)
    assert float(la - lb) == 0.0  # the KL pull toward orig-shaped targets is gone
    ce_i = F.cross_entropy(sub.float(), labels, reduction="none")
    assert abs(float(la) - float(0.5 * F.smooth_l1_loss(ce_i, target_ce))) < 1e-6
    # retro-compat: the default kl_ce loss still contains the distributional term
    kl_cfg = ScrubRPdTargetConfig(name="t", pd_target_weight=0.5, mimic_ce_only=False)
    l_kl_a = pd_target_mimic_loss(sub, z_a, target_ce, labels, kl_cfg)
    l_kl_b = pd_target_mimic_loss(sub, z_b, target_ce, labels, kl_cfg)
    assert float(l_kl_a - l_kl_b) != 0.0  # kl path depends on the rest geometry
    assert float(l_kl_a - la) != 0.0  # and differs from the pure CE match


def test_round13_candidates_exact_views() -> None:
    cands = round13_candidates()
    assert [c.name for c in cands] == ["SR19_exact_w05", "SR19_exact_w10", "SR19_exact_w20"]
    assert [c.pd_target_weight for c in cands] == [0.5, 1.0, 2.0]
    sr18 = round12_candidates()[0]
    for c in cands:
        assert isinstance(c, ScrubRPdTargetConfig)
        assert c.aug_mode == "identity"  # the ONLY delta vs SR18: exact views, matched targets
        assert c.mimic_ce_only and c.retain_kl_weight == 0.5 and c.retain_ce_weight == 0.05
        assert c.rest_kl_weight == 0.0 and c.pd_target_quantile and c.pd_target_affine is None
        assert c.init_from == CHAMPION_CKPT and c.rewind_delta_l2_max is None
        assert c.lr_max == 5.0e-5 and c.lr_min == 5.0e-5
        assert (c.max_epochs, c.total_epochs, c.max_steps_per_epoch) == (6, 6, 22)
    # retro-compat: every earlier family keeps its original aug_mode
    assert sr18.aug_mode == "moderate"
    assert all(c.aug_mode != "identity" for c in round12_candidates() + round11_candidates())


def test_augment_identity_is_verbatim_and_none_still_flips() -> None:
    torch.manual_seed(0)
    images = torch.randint(0, 256, (64, 32, 32, 3), dtype=torch.uint8)
    assert torch.equal(augment_training_batch(images, mode="identity"), images)  # bit-exact
    out = augment_training_batch(images, mode="none")  # hflip 50%: some rows flip, not all
    flipped_rows = (out != images).any(dim=(1, 2, 3))
    assert 0 < int(flipped_rows.sum()) < 64  # "none" is NOT identity


def test_round14_candidates_champion_trajectory_gates() -> None:
    cands = round14_candidates()
    assert [c.name for c in cands] == [
        "SR20_flatref_s101", "SR20_gateq_b20", "SR20_gateqoff_b20", "SR20_gateq_b14",
    ]
    for c in cands:
        assert isinstance(c, GateQConfig)
        assert (c.steps, c.batch_size, c.lr) == (210, 64, 1.0e-4)  # the verbatim grid recipe
        assert c.retain_kl_weight == 2.8 and c.salun_keep_ratio == 0.20 and c.l2_weight == 2.0e-3
        assert c.aug_mode == "moderate" and c.seed == 101 and c.proxy_seed == 42
        assert c.strategy == "salun_npo" and c.loss_kind == "npo"
    flat, gateq, off, b14 = cands
    assert not flat.per_image_gate and flat.target_forget_ce == 1.00  # champion control
    assert all(c.per_image_gate for c in (gateq, off, b14))
    assert gateq.gate_offset == 0.0 and off.gate_offset == -0.9
    assert gateq.npo_beta == 2.0 and b14.npo_beta == 1.4
    # everything else verbatim vs the flat control
    skip = ("name", "per_image_gate", "gate_offset", "npo_beta")
    flat_d = {k: v for k, v in asdict(flat).items() if k not in skip}
    assert all({k: v for k, v in asdict(c).items() if k not in skip} == flat_d for c in (gateq, off, b14))


def test_npo_per_image_gate_constant_equals_npo_loss() -> None:
    torch.manual_seed(0)
    labels = torch.tensor([0, 1, 2, 3])
    orig = torch.randn(4, 6)
    orig[torch.arange(4), labels] = 6.0
    sub = orig + 0.25 * torch.randn_like(orig)
    # Row 0 hand-built so its CE lands between the two gates deterministically:
    # CE_0 = log(1 + 5*exp(-1.5)) ~ 0.75 -- above the 0.5 per-image gate (mask OFF),
    # below the flat 1.00 gate (mask ON), so the per-image mask actually bites.
    sub[0, :] = 0.0
    sub[0, labels[0]] = 1.5
    sub = sub.requires_grad_(True)
    flat_gate = torch.full((4,), 1.00)
    mine = npo_per_image_gate_loss(sub, orig, labels, beta=2.0, temperature=2.0, max_ce_i=flat_gate)
    reference = npo_loss(sub, orig, labels, beta=2.0, temperature=2.0, max_ce=1.00)
    assert abs(float(mine - reference)) < 1e-7  # constant gate is bitwise npo_loss
    # per-image gates: only images below THEIR gate keep pushing
    per_image = torch.tensor([0.5, 2.0, 0.5, 2.0])
    ce_i = F.cross_entropy(sub.detach().float(), labels, reduction="none")
    gated = npo_per_image_gate_loss(sub, orig, labels, beta=2.0, temperature=2.0, max_ce_i=per_image)
    per_sample = (2.0 / 2.0) * F.softplus(2.0 * (
        F.log_softmax(sub.detach().float() / 2.0, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
        - F.log_softmax(orig.float() / 2.0, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    ))
    expected = (per_sample * (ce_i < per_image).float()).mean()
    assert abs(float(gated - expected)) < 1e-7
    assert float(gated) != float(mine)  # the per-image gate actually changes the loss


def test_round15_candidates_surgical_placement() -> None:
    cands = round15_candidates()
    assert [c.name for c in cands] == [
        "SR21_shiftm07", "SR21_spread115", "SR21_shiftm27_spread115", "SR21_gateq_s202",
    ]
    shift, spread, combined, replicate = cands
    record = round14_candidates()[1]  # SR20_gateq_b20, the #29 record run
    for c in cands:
        assert isinstance(c, GateQConfig) and c.per_image_gate
        assert (c.steps, c.batch_size, c.lr, c.npo_beta) == (210, 64, 1.0e-4, 2.0)
        assert c.retain_kl_weight == 2.8 and c.salun_keep_ratio == 0.20
        assert c.aug_mode == "moderate" and c.proxy_seed == 42
    assert shift.gate_offset == -0.07 and shift.gate_high_mult == 1.0
    assert spread.gate_offset == 0.0 and spread.gate_high_mult == 1.15
    # combined shift is -0.27 (mean-compensated): the 1.15 mult moves the gate mean +0.220
    # (heavy lognormal tail), so -0.07 would land mu ~1.92 under the measured +0.05 rule
    assert combined.gate_offset == -0.27 and combined.gate_high_mult == 1.15
    assert replicate.seed == 202 and replicate.gate_offset == 0.0 and replicate.gate_high_mult == 1.0
    # everything else verbatim vs the record run (only gate knobs / seed move)
    skip = ("name", "gate_offset", "gate_high_mult", "seed")
    record_d = {k: v for k, v in asdict(record).items() if k not in skip}
    assert all({k: v for k, v in asdict(c).items() if k not in skip} == record_d for c in cands)


def test_gate_transforms_offset_then_upper_half_spread() -> None:
    gates = gold_bell_quantile((torch.arange(8, dtype=torch.float32) + 0.5) / 8)
    identity = _apply_gate_transforms(
        gates, GateQConfig(name="t", strategy="salun_npo", loss_kind="npo", per_image_gate=True)
    )
    assert torch.equal(identity, gates)  # offset 0, mult 1.0: bitwise no-op (round-14 compat)
    shift = _apply_gate_transforms(
        gates, GateQConfig(name="t", strategy="salun_npo", loss_kind="npo", gate_offset=-0.07, per_image_gate=True)
    )
    assert torch.allclose(shift, (gates - 0.07).clamp_min(0.05))
    spread = _apply_gate_transforms(
        gates, GateQConfig(name="t", strategy="salun_npo", loss_kind="npo", gate_high_mult=1.15, per_image_gate=True)
    )
    med = gates.median()
    assert torch.allclose(spread, torch.where(gates > med, gates * 1.15, gates))
    assert float(spread.min()) == float(gates.min())  # the easy half untouched
    assert bool((spread[1:] > spread[:-1]).all())  # rank order (rho) preserved
    combined = _apply_gate_transforms(
        gates,
        GateQConfig(name="t", strategy="salun_npo", loss_kind="npo",
                    gate_offset=-0.07, gate_high_mult=1.15, per_image_gate=True),
    )
    shifted = (gates - 0.07).clamp_min(0.05)
    expect = torch.where(shifted > shifted.median(), shifted * 1.15, shifted)
    assert torch.allclose(combined, expect)  # offset first, then the post-offset upper half


def test_round16_candidates_seed_sweep() -> None:
    cands = round16_candidates()
    seeds = (303, 404, 505, 606, 707, 808, 909, 111)
    assert [c.name for c in cands] == [f"SR22_sp115_s{s}" for s in seeds]
    assert sorted(c.seed for c in cands) == sorted(seeds)
    record = round15_candidates()[1]  # SR21_spread115, the best-predicted recipe
    for c in cands:
        assert isinstance(c, GateQConfig) and c.per_image_gate
        assert c.gate_high_mult == 1.15 and c.gate_offset == 0.0
        assert (c.steps, c.batch_size, c.lr, c.npo_beta) == (210, 64, 1.0e-4, 2.0)
        assert c.retain_kl_weight == 2.8 and c.salun_keep_ratio == 0.20
        assert c.aug_mode == "moderate" and c.proxy_seed == 42  # proxy NEVER varies with seed
    # only the seed moves vs the record recipe -- everything else verbatim
    skip = ("name", "seed")
    record_d = {k: v for k, v in asdict(record).items() if k not in skip}
    assert all({k: v for k, v in asdict(c).items() if k not in skip} == record_d for c in cands)


def test_round17_candidates_more_lottery_tickets() -> None:
    cands = round17_candidates()
    seeds = (121, 131, 141, 151, 161, 171, 181, 191)
    assert [c.name for c in cands] == [f"SR23_sp115_s{s}" for s in seeds]
    record = round15_candidates()[1]  # SR21_spread115, the record recipe
    skip = ("name", "seed")
    record_d = {k: v for k, v in asdict(record).items() if k not in skip}
    assert all(
        {k: v for k, v in asdict(c).items() if k not in skip} == record_d
        and c.gate_high_mult == 1.15 and c.proxy_seed == 42
        for c in cands
    )

if __name__ == "__main__":
    test_round2_candidates_grid_and_control()
    test_high_target_keeps_gate_open_below_ce()
    test_pd_weights_increase_with_difficulty()
    test_control_gate_ignores_diff_ratio()
    test_in_window_bounds()
    test_spearman_corr_sign_and_pd_seed()
    test_round3_candidates_deep_amplitude()
    test_round4_candidates_rest_and_pd_target()
    test_rest_kl_loss_zero_when_identical_and_ignores_label_logit()
    test_scrub_max_rest_loss_is_npo_plus_rest()
    test_pd_target_mimic_loss_zero_at_target_and_scales_with_weight()
    test_calibrated_pd_targets_pin_rest_logits_and_mean_ce()
    test_budgeted_rewind_selection_filters_pool()
    test_budgeted_rewind_falls_back_when_no_snapshot_in_budget()
    test_round5_candidates_flat_gate_braked()
    test_round6_candidates_pushing_ascent_budgeted()
    test_flat_gate_closes_at_target_while_pd_weights_stay()
    test_round7_candidates_gold_mimic_and_hybrid()
    test_hybrid_flat_gate_keeps_rest_pin_alive()
    test_round8_candidates_champion_refinement()
    test_gold_affine_maps_pd_distribution_onto_gold()
    test_affine_calibrated_targets_pin_rest_and_stay_self_consistent()
    test_pkl_only_budget_ignores_delta_l2()
    test_round9_candidates_quantile_map_keeps_champion_recipe()
    test_gold_bell_quantile_lognormal_moments_and_monotonicity()
    test_quantile_calibrated_targets_rank_map_and_pin()
    test_round10_candidates_drop_rest_pin_pull()
    test_round11_candidates_soften_train_proxy_retain()
    test_round12_candidates_pure_ce_match()
    test_mimic_ce_only_loss_ignores_target_logits()
    test_round13_candidates_exact_views()
    test_augment_identity_is_verbatim_and_none_still_flips()
    test_round14_candidates_champion_trajectory_gates()
    test_npo_per_image_gate_constant_equals_npo_loss()
    test_round15_candidates_surgical_placement()
    test_gate_transforms_offset_then_upper_half_spread()
    test_round16_candidates_seed_sweep()
    test_round17_candidates_more_lottery_tickets()
    print("SCRUB_R2_TESTS_OK")
