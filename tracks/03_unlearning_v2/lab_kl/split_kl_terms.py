"""KL chain-rule split on the 11 server-scored pairs (Antigravity protocol, 10h50).

Runs INSIDE the lab-KL judge flow once gold_lab exists: reuses gold_f, sub_f,
forget_labels computed in run_lab_kl_judge.judge_all. One pass, no retraining.

Decision rules:
  share_rest >= 0.65  -> code the orig_rest distillation (99 non-target logits)
                          into scrub_max_loss (SCRUB+R2 Round 4).
  share_bin  >= 0.50  -> per-image pd_alpha / target_forget_ce tuning instead.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def split_kl_terms(gold_f: torch.Tensor, sub_f: torch.Tensor, forget_labels: torch.Tensor) -> dict:
    """Per-image KL(T=2) decomposed as kl_bin(q_y||p_y) + kl_rest (chain rule)."""
    q = F.softmax(gold_f.float() / 2.0, dim=-1).clamp_min(1e-12)
    p = F.softmax(sub_f.float() / 2.0, dim=-1).clamp_min(1e-12)
    y = forget_labels.unsqueeze(1)
    qy = q.gather(1, y).squeeze(1)
    py = p.gather(1, y).squeeze(1)
    kl_total = (q * (q / p).log()).sum(dim=-1)
    kl_bin = qy * (qy / py).log() + (1 - qy) * (
        (1 - qy).clamp_min(1e-12) / (1 - py).clamp_min(1e-12)
    ).log()
    kl_rest = kl_total - kl_bin  # == (1 - qy) * KL(q_rest || p_rest)
    return {
        "kl_total_mean": float(kl_total.mean()),
        "kl_bin_mean": float(kl_bin.mean()),
        "kl_rest_mean": float(kl_rest.mean()),
        "share_rest": float(kl_rest.mean() / kl_total.mean().clamp_min(1e-12)),
        "share_bin": float(kl_bin.mean() / kl_total.mean().clamp_min(1e-12)),
    }
