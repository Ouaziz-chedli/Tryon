"""CPU unit tests for Track G exact server scoring formulas."""

from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.server_metrics import compute_mean_std_score, compute_prediction_score, compute_retain_metrics


def test_mean_std_and_prediction_perfect_match() -> None:
    gold_l = torch.tensor([0.5, 1.0, 1.5])
    orig_l = torch.tensor([0.0, 0.0, 0.0])
    assert compute_mean_std_score(gold_l, gold_l, orig_l) == 1.0
    assert compute_mean_std_score(orig_l, gold_l, orig_l) == 0.0

    torch.manual_seed(0)
    gold_z = torch.randn(8, 100)
    orig_z = torch.randn(8, 100) * 3.0
    assert compute_prediction_score(gold_z, gold_z, orig_z, temperature=2.0) == 1.0
    assert compute_prediction_score(orig_z, gold_z, orig_z, temperature=2.0) == 0.0


def test_retain_metrics_identity_is_one() -> None:
    logits = torch.randn(10, 100)
    labels = logits.argmax(dim=-1)
    acc_s, kl_s, r_s = compute_retain_metrics(logits, logits, labels, temperature=2.0)
    assert acc_s == 1.0 and kl_s == 1.0 and r_s == 1.0
