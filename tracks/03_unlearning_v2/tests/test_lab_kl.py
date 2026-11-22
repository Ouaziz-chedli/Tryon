"""Unit tests for the P4.1 lab-KL builder and local judge (torch required: run on the cluster)."""

import importlib
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

LAB_KL_DIR = Path(__file__).resolve().parents[1] / "lab_kl"
if str(LAB_KL_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_KL_DIR))


def load_module(name):
    return importlib.import_module(name)


class TinyHeadModel(torch.nn.Module):
    """Smallest module exposing a .head so head-row pinning and head lr groups are exercisable."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.head(x)


def test_server_scores_table_has_11_known_pairs():
    judge = load_module("run_lab_kl_judge")
    assert len(judge.SERVER_SCORES) == 11
    assert len(set(judge.SERVER_SCORES.values())) == 11
    assert all(0.0 < v < 1.0 for v in judge.SERVER_SCORES.values())


def test_spearman_monotone_inverse_ties_and_constant():
    judge = load_module("run_lab_kl_judge")
    assert judge.spearman([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]) == pytest.approx(1.0)
    assert judge.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert judge.spearman([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.8660, abs=1e-3)
    assert math.isnan(judge.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_ranking_recovery_on_synthetic_lab():
    judge = load_module("run_lab_kl_judge")
    results = {
        s: {"final_score": v, "forget_score": v * 0.9, "prediction_score": v,
            "mean_std_score": v, "retain_score": v, "kl_gold_sub_t2": 1.0 - v, "error": None}
        for s, v in judge.SERVER_SCORES.items()
    }
    rhos = judge.compute_ranking_rhos(results)
    name, rho = judge.best_rank_signal(rhos)
    assert rho >= 0.99
    assert name in judge.RANK_METRICS


def test_ranking_requires_three_known_stems():
    judge = load_module("run_lab_kl_judge")
    rhos = judge.compute_ranking_rhos({"unknown_stem": {"final_score": 0.5, "error": None}})
    assert "error" in rhos


def test_pin_absent_head_rows_restores_original_prototypes():
    build = load_module("build_lab_kl")
    student, orig = TinyHeadModel(), TinyHeadModel()
    with torch.no_grad():
        student.head.weight.add_(1.0)
    mask = torch.tensor([True, False, True])
    build.pin_absent_head_rows(student, orig, mask)
    assert torch.equal(student.head.weight[0], orig.head.weight[0])
    assert torch.equal(student.head.weight[2], orig.head.weight[2])
    assert not torch.equal(student.head.weight[1], orig.head.weight[1])


def test_make_optimizer_drops_empty_group_and_sets_head_lr():
    build = load_module("build_lab_kl")
    opt = build.make_optimizer(TinyHeadModel(), lr=1e-4, head_lr_mult=10.0)
    assert len(opt.param_groups) == 1  # body group (empty on a head-only module) is dropped
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_fine_tune_skips_when_already_saturated():
    build = load_module("build_lab_kl")
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    x = torch.randn(16, 4)
    labels = model(x).argmax(dim=-1)  # CE ~ 0 by construction
    steps, ce_mean = build.fine_tune_to_ce(
        model, x, labels, target_ce=0.01, max_steps=50, lr=0.05,
        head_lr_mult=1.0, batch_size=8, probe_every=5)
    assert steps == 0
    assert ce_mean < 0.01


def test_fine_tune_saturates_and_pins_absent_rows():
    build = load_module("build_lab_kl")
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    w_true = torch.randn(4, 3)
    labels = (x @ w_true).argmax(dim=-1)  # representable target: a linear model can fit it exactly
    student, orig = TinyHeadModel(), TinyHeadModel()
    steps, ce_mean = build.fine_tune_to_ce(
        student, x, labels, target_ce=0.01, max_steps=300, lr=0.05,
        head_lr_mult=1.0, batch_size=8, probe_every=10,
        absent_anchor=(orig, torch.tensor([True, False, True])))
    assert steps > 0
    assert ce_mean < 0.01
    # the 55-absent-class analogue: pinned rows never move from the original prototype
    assert torch.equal(student.head.weight[0], orig.head.weight[0])
    assert torch.equal(student.head.weight[2], orig.head.weight[2])


def test_build_control_proxy_is_deterministic_and_disjoint_by_seed():
    judge = load_module("run_lab_kl_judge")
    images = torch.randint(0, 255, (4, 224, 224, 3), dtype=torch.uint8)
    proxy_a = judge.build_control_proxy(images, 999)
    proxy_b = judge.build_control_proxy(images, 999)
    proxy_c = judge.build_control_proxy(images, 777)
    assert proxy_a.shape == (4, 224, 224, 3)
    assert torch.equal(proxy_a, proxy_b)  # deterministic within a seed
    assert not torch.equal(proxy_a, proxy_c)  # control (999) and val (777) proxies differ
