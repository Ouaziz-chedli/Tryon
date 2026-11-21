"""Shared helpers for Track 3 unlearning strategies (freezing, losses, anchors)."""

import copy
import torch
import torch.nn.functional as F

from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
from task_template import DEVICE, preprocess


def prepare_student(orig_model: torch.nn.Module, freeze_until_block: int = 6) -> torch.nn.Module:
    """Clone orig_model and freeze patch_embed, pos/cls, and blocks < freeze_until_block."""
    student = copy.deepcopy(orig_model).to(DEVICE).requires_grad_(True)
    for name, param in student.named_parameters():
        if name.startswith("patch_embed") or name in ("cls_token", "pos_embed"):
            param.requires_grad_(False)
        elif name.startswith("blocks."):
            block_idx = int(name.split(".")[1])
            if block_idx < freeze_until_block:
                param.requires_grad_(False)
    return student


def mask_absent_head_grads(student: torch.nn.Module, absent_mask: torch.Tensor) -> None:
    """Zero out gradients on head rows belonging to the 55 absent classes."""
    mask_dev = absent_mask.to(DEVICE)
    for param in (student.head.weight, student.head.bias):
        if param.grad is not None:
            param.grad[mask_dev] = 0.0


def npo_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Negative Preference Optimization (COLM 2024) at temperature to avoid saturation."""
    sub_lp = F.log_softmax(sub_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    return (2.0 / beta) * F.softplus(beta * (sub_lp - orig_lp)).mean()


@torch.no_grad()
def should_stop_early(
    student: torch.nn.Module,
    raw_batch: torch.Tensor,
    pseudo_labels: torch.Tensor,
    target_ce: float,
) -> bool:
    """Check target CE on clean (unaugmented) forget samples."""
    student.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        logits = student(preprocess(raw_batch))
    ce = float(F.cross_entropy(logits.float(), pseudo_labels.to(DEVICE)).item())
    return ce >= target_ce


def l2_anchor_loss(student: torch.nn.Module, orig_params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of squared parameter drift on trainable parameters."""
    loss = torch.tensor(0.0, device=DEVICE)
    for name, param in student.named_parameters():
        if param.requires_grad:
            loss = loss + F.mse_loss(param, orig_params[name], reduction="sum")
    return loss


def retain_kl_loss(
    student: torch.nn.Module,
    proxy_batch: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute KL(p_orig || p_sub) at temperature on a retain proxy batch."""
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        sub_logits = student(preprocess(proxy_batch))
    return compute_kl_at_temperature(orig_proxy_logits.to(DEVICE), sub_logits, temperature).mean()
