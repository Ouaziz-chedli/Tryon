"""Exact Track 3 server evaluation formulas for the Ground-Truth Lab (Track G)."""

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

from eval_harness import compute_kl_at_temperature
from task_template import predict


@dataclass(frozen=True)
class LabScoreReport:
    """Exact Track 3 scoring breakdown against a true gold model."""

    val_acc: float
    utility_gate_passed: bool
    mean_std_score: float
    prediction_score: float
    forget_score: float
    retain_acc_score: float
    retain_kl_similarity: float
    retain_score: float
    final_score: float


def compute_mean_std_score(sub_losses: torch.Tensor, gold_losses: torch.Tensor) -> float:
    """Score proximity of (mean, std) of per-sample CE losses to gold."""
    d_mean = abs(float(sub_losses.mean().item()) - float(gold_losses.mean().item()))
    d_std = abs(float(sub_losses.std().item()) - float(gold_losses.std().item()))
    return round(math.exp(-(d_mean + d_std)), 4)


def compute_prediction_score(
    sub_logits: torch.Tensor,
    gold_logits: torch.Tensor,
    temperature: float = 2.0,
) -> float:
    """Score image-by-image softmax KL(p_gold || p_sub) at T=2."""
    kl = float(compute_kl_at_temperature(gold_logits, sub_logits, temperature).mean().item())
    return round(1.0 / (1.0 + kl), 4)


def compute_retain_metrics(
    orig_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
) -> tuple[float, float, float]:
    """Return (retain_acc_score, retain_kl_similarity, retain_score) on control set."""
    y = labels.to(orig_logits.device)
    acc_orig = float((orig_logits.argmax(dim=-1) == y).float().mean().item())
    acc_sub = float((sub_logits.argmax(dim=-1) == y).float().mean().item())
    acc_score = max(0.0, min(1.0, acc_sub / max(acc_orig, 1e-6)))
    kl = float(compute_kl_at_temperature(orig_logits, sub_logits, temperature).mean().item())
    kl_sim = 1.0 / (1.0 + kl)
    return round(acc_score, 4), round(kl_sim, 4), round(acc_score * kl_sim, 4)


def evaluate_against_gold(
    orig_model: torch.nn.Module,
    gold_model: torch.nn.Module,
    sub_model: torch.nn.Module,
    forget_eval_imgs: torch.Tensor,
    forget_eval_labels: torch.Tensor,
    control_imgs: torch.Tensor,
    control_labels: torch.Tensor,
    val_imgs: torch.Tensor,
    val_labels: torch.Tensor,
    temperature: float = 2.0,
) -> LabScoreReport:
    """Compute full Track 3 ground-truth score against gold_model."""
    val_logits = predict(sub_model, val_imgs, batch_size=128)
    val_acc = float((val_logits.argmax(dim=-1) == val_labels.to(val_logits.device)).float().mean().item())
    gate = val_acc >= 0.70

    sub_f = predict(sub_model, forget_eval_imgs, batch_size=128)
    gold_f = predict(gold_model, forget_eval_imgs, batch_size=128)
    y_f = forget_eval_labels.to(sub_f.device)
    ms_score = compute_mean_std_score(
        F.cross_entropy(sub_f.float(), y_f, reduction="none"),
        F.cross_entropy(gold_f.float(), y_f, reduction="none"),
    )
    pred_score = compute_prediction_score(sub_f, gold_f, temperature)
    forget_score = round(ms_score * pred_score, 4)

    orig_c = predict(orig_model, control_imgs, batch_size=128)
    sub_c = predict(sub_model, control_imgs, batch_size=128)
    r_acc, r_kl, retain_score = compute_retain_metrics(orig_c, sub_c, control_labels, temperature)
    final = round(forget_score * retain_score, 4) if gate else 0.0
    return LabScoreReport(
        val_acc=round(val_acc, 4),
        utility_gate_passed=gate,
        mean_std_score=ms_score,
        prediction_score=pred_score,
        forget_score=forget_score,
        retain_acc_score=r_acc,
        retain_kl_similarity=r_kl,
        retain_score=retain_score,
        final_score=final,
    )
