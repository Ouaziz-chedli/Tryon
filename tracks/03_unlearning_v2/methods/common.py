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
    max_ce: float | None = None,
) -> torch.Tensor:
    """Negative Preference Optimization (COLM 2024), optionally gated per-sample by max_ce."""
    sub_lp = F.log_softmax(sub_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    per_sample = (2.0 / beta) * F.softplus(beta * (sub_lp - orig_lp))
    if max_ce is not None:
        with torch.no_grad():
            ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
        per_sample = per_sample * (ce_i < max_ce).to(per_sample.dtype)
    return per_sample.mean()


@torch.no_grad()
def should_stop_early(
    student: torch.nn.Module,
    raw_batch: torch.Tensor,
    pseudo_labels: torch.Tensor,
    target_ce: float,
) -> bool:
    """Check median CE on clean forget samples so outliers do not trigger premature stop."""
    student.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        logits = student(preprocess(raw_batch))
    ce_med = float(F.cross_entropy(logits.float(), pseudo_labels.to(DEVICE), reduction="none").median().item())
    return ce_med >= target_ce


def l2_anchor_loss(student: torch.nn.Module, orig_params: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of squared parameter drift on trainable parameters present in orig_params."""
    loss = torch.tensor(0.0, device=DEVICE)
    for name, param in student.named_parameters():
        if param.requires_grad and name in orig_params:
            loss = loss + F.mse_loss(param, orig_params[name], reduction="sum")
    return loss


def two_sided_npo_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
    temperature: float = 2.0,
    target_ce: float = 0.45,
    upper_ce: float = 0.65,
    pullback_weight: float = 0.5,
) -> torch.Tensor:
    """Push ce_i < target_ce up via NPO and gently pull ce_i > upper_ce back toward orig."""
    sub_lp = F.log_softmax(sub_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    push_i = (2.0 / beta) * F.softplus(beta * (sub_lp - orig_lp))
    pull_i = compute_kl_at_temperature(orig_logits, sub_logits, temperature)
    with torch.no_grad():
        ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
    push_mask = (ce_i < target_ce).to(push_i.dtype)
    pull_mask = (ce_i > upper_ce).to(pull_i.dtype)
    return (push_i * push_mask + pullback_weight * pull_i * pull_mask).mean()


def gold_mimic_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    target_prob: float = 0.60,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Match a per-sample Gold-Mimic target: p*=target_prob on top-1, (1-p*) on secondary classes at T."""
    with torch.no_grad():
        sec = F.softmax(orig_logits.float() / temperature, dim=-1)
        sec.scatter_(1, labels.unsqueeze(1), 0.0)
        sec = sec / sec.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        q_star = sec * (1.0 - target_prob)
        q_star.scatter_(1, labels.unsqueeze(1), target_prob)
    return F.kl_div(F.log_softmax(sub_logits.float(), dim=-1), q_star, reduction="batchmean")


def compute_forget_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    cfg: MethodConfig,
) -> torch.Tensor:
    """Dispatch forget objective according to cfg.loss_kind."""
    if cfg.loss_kind == "gold_mimic":
        return gold_mimic_loss(sub_logits, orig_logits, labels, cfg.gold_target_prob, cfg.temperature)
    if cfg.loss_kind == "two_sided_npo":
        return two_sided_npo_loss(
            sub_logits, orig_logits, labels, cfg.npo_beta, cfg.temperature,
            cfg.target_forget_ce, cfg.target_ce_upper, cfg.pullback_weight,
        )
    return npo_loss(sub_logits, orig_logits, labels, cfg.npo_beta, cfg.temperature, max_ce=cfg.target_forget_ce)


@torch.no_grad()
def get_batch_orig_logits(
    orig_model: torch.nn.Module | None,
    batch_images: torch.Tensor,
    cached_logits: torch.Tensor,
    aug_mode: str,
) -> torch.Tensor:
    """Compare student(aug) against orig(aug) when aug_mode != 'none', else use cached_logits."""
    if aug_mode == "none" or orig_model is None:
        return cached_logits.to(DEVICE)
    orig_model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        return orig_model(preprocess(batch_images))


def build_optimizer(student: torch.nn.Module, lr: float, head_lr_mult: float = 1.0) -> torch.optim.AdamW:
    """Build AdamW with optional differentiated learning rate for head vs backbone."""
    head_params = [p for n, p in student.named_parameters() if p.requires_grad and n.startswith("head.")]
    body_params = [p for n, p in student.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [{"params": body_params, "lr": lr}, {"params": head_params, "lr": lr * head_lr_mult}]
    return torch.optim.AdamW([g for g in groups if g["params"]])


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
