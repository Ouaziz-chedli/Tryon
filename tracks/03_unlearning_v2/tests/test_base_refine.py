"""CPU synthetic unit tests for run_base_refine.py."""

import torch

from run_base_refine import (
    MID_LIMITS,
    _tight_adjust_controller,
    _tight_snapshot_score,
    blend_state_dicts,
    build_round1_configs,
    build_round2_configs,
    build_round2_soup_specs,
    build_round3_configs,
    build_round3_soup_specs,
    build_soup_specs,
    hits_target,
)
from selective_benchmark import SelectiveReport


def _dummy_report(**overrides: float) -> SelectiveReport:
    """Build a target-hitting SelectiveReport with optional field overrides."""
    defaults: dict[str, object] = {
        "name": "AT_test",
        "weighted_selective_score": 0.58,
        "fold_score_std": 0.02,
        "weighted_forget_ce_raw": 1.70,
        "weighted_forget_ce_alt": 2.40,
        "weighted_intra_retain_acc": 0.85,
        "weighted_intra_gap_50_50": 0.90,
        "weighted_inter_retain_acc": 0.92,
        "weighted_inter_retain_kl": 0.035,
        "full_ce_mean": 1.70,
        "full_ce_std": 2.00,
        "full_ce_p50": 0.85,
        "full_frac_gt_95": 0.18,
        "full_proxy_kl_t2": 0.032,
        "full_absent_kl_t2": 0.015,
        "full_delta_l2": 4.10,
        "folds": [],
    }
    defaults.update(overrides)
    return SelectiveReport(**defaults)


def _assert_raises(fn: object, *args: object) -> None:
    try:
        fn(*args)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def test_blend_state_dicts_convex_and_guards() -> None:
    sd_a = {"w": torch.tensor([1.0, 3.0], dtype=torch.float16)}
    sd_b = {"w": torch.tensor([5.0, 7.0], dtype=torch.float32)}
    out = blend_state_dicts([(0.25, sd_a), (0.75, sd_b)])
    assert out["w"].dtype == torch.float32
    assert torch.allclose(out["w"], torch.tensor([4.0, 6.0]))
    _assert_raises(blend_state_dicts, [])
    _assert_raises(blend_state_dicts, [(0.5, sd_a), (0.4, sd_b)])


def test_build_round1_configs_and_soups() -> None:
    cfgs = build_round1_configs()
    assert len(cfgs) == 8
    cfg_names = {c.name for c in cfgs}
    assert "ANCHOR_ce0.50_b2.0_f6" in cfg_names and "ANCHOR_ce1.00_b1.8_f6" in cfg_names
    soups = build_soup_specs()
    assert len(soups) == 6
    for _, spec in soups:
        assert abs(sum(w for w, _ in spec) - 1.0) <= 1e-5
        for _, src in spec:
            assert src in cfg_names


def test_build_round2_configs_and_soups() -> None:
    r2_cfgs = build_round2_configs()
    assert len(r2_cfgs) == 8
    runners = [rn for rn, _ in r2_cfgs]
    assert runners.count("tight") == 6 and runners.count("adaptive") == 2
    valid_sources = {c.name for _, c in r2_cfgs} | {"ORIG"}
    soups = build_round2_soup_specs()
    assert len(soups) == 6
    for _, spec in soups:
        assert abs(sum(w for w, _ in spec) - 1.0) <= 1e-5
        for _, src in spec:
            assert src in valid_sources


def test_build_round3_configs_and_soups() -> None:
    r3_cfgs = build_round3_configs()
    assert len(r3_cfgs) == 6
    runners = [rn for rn, _ in r3_cfgs]
    assert runners.count("tight_mid") == 4 and runners.count("adaptive") == 1 and runners.count("tight") == 1
    seeds = {c.name: c.seed for _, c in r3_cfgs}
    assert seeds["AT3_s101_ce0.90_b1.8_kl28"] == 101
    assert seeds["AT3_s202_ce0.95_b1.8_kl26"] == 202
    assert seeds["AT3_s303_ce0.85_b1.9_kl30"] == 303
    assert seeds["AT3_s42_ce0.90_b1.8_kl28"] == 42
    valid_sources = set(seeds) | {"ORIG"}
    soups = build_round3_soup_specs()
    assert len(soups) == 8
    for _, spec in soups:
        assert abs(sum(w for w, _ in spec) - 1.0) <= 1e-5
        for _, src in spec:
            assert src in valid_sources


def test_tight_controller_and_snapshot_score() -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.AdamW([p], lr=1e-4)
    kl_w, l2_w = _tight_adjust_controller(opt, 3.2, 4e-3, pkl=0.035, d_l2=3.5, p50=0.6)
    assert kl_w > 3.2 and l2_w > 4e-3 and opt.param_groups[0]["lr"] < 1e-4
    opt_mid = torch.optim.AdamW([p], lr=1e-4)
    kl_m, _ = _tight_adjust_controller(opt_mid, 2.8, 2.8e-3, pkl=0.035, d_l2=3.5, p50=0.6, limits=MID_LIMITS)
    assert kl_m == 2.8 and opt_mid.param_groups[0]["lr"] == 1e-4
    s_clean = _tight_snapshot_score(1.5, 1.8, 0.85, 0.15, pkl=0.028, d_l2=4.0, target_p50=0.85)
    s_drift = _tight_snapshot_score(1.5, 1.8, 0.85, 0.15, pkl=0.045, d_l2=4.4, target_p50=0.85)
    assert s_clean > s_drift


def test_hits_target_gate() -> None:
    assert hits_target(_dummy_report())
    assert not hits_target(_dummy_report(full_ce_mean=1.50))
    assert not hits_target(_dummy_report(full_ce_mean=1.80, full_ce_std=2.20))
    assert not hits_target(_dummy_report(full_ce_p50=0.60))
    assert not hits_target(_dummy_report(full_frac_gt_95=0.25))
    assert not hits_target(_dummy_report(full_proxy_kl_t2=0.040))
    assert not hits_target(_dummy_report(full_delta_l2=4.5))
    assert not hits_target(_dummy_report(weighted_selective_score=0.55))


if __name__ == "__main__":
    test_blend_state_dicts_convex_and_guards()
    test_build_round1_configs_and_soups()
    test_build_round2_configs_and_soups()
    test_build_round3_configs_and_soups()
    test_tight_controller_and_snapshot_score()
    test_hits_target_gate()
    print("BASE_REFINE_TESTS_OK")
