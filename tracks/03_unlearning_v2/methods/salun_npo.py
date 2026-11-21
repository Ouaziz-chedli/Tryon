"""Strategy A: Tempered SalUn (T=2) + NPO + Absent-Class Freeze."""

import torch
import torch.nn.functional as F

from augment import augment_training_batch
from contracts import MethodConfig
from methods.common import (
    build_optimizer,
    compute_forget_loss,
    get_batch_orig_logits,
    l2_anchor_loss,
    mask_absent_head_grads,
    prepare_student,
    restore_absent_head_rows,
    retain_kl_loss,
    should_stop_early,
)
from methods.ssd import run_ssd
from task_template import DEVICE, preprocess


def compute_tempered_saliency(
    model: torch.nn.Module,
    forget_images: torch.Tensor,
    pseudo_labels: torch.Tensor,
    keep_ratio: float,
    temperature: float = 2.0,
    num_batches: int = 32,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Compute layer-wise SalUn binary masks using tempered logits on shuffled indices."""
    model.eval()
    model.zero_grad(set_to_none=True)
    dev = next(model.parameters()).device
    accum = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters() if p.requires_grad}
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(forget_images), generator=gen)[: min(len(forget_images), num_batches * batch_size)]
    for start in range(0, len(perm), batch_size):
        idx = perm[start : start + batch_size]
        x = preprocess(forget_images[idx]).to(dev)
        y = pseudo_labels[idx].to(dev)
        loss = F.cross_entropy(model(x) / temperature, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                accum[n] += p.grad.detach().abs().float()
        model.zero_grad(set_to_none=True)
    return {
        k: (v >= torch.quantile(v.reshape(-1), 1.0 - keep_ratio)).to(v.dtype)
        for k, v in accum.items()
    }


def _salun_step(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    pseudo: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    student.train()
    f_idx = torch.randint(0, len(forget_images), (cfg.batch_size,))
    p_idx = torch.randint(0, len(proxy_images), (cfg.batch_size,))
    f_batch = augment_training_batch(forget_images[f_idx], mode=cfg.aug_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        sub_f = student(preprocess(f_batch))
    orig_batch = get_batch_orig_logits(orig_model, f_batch, orig_forget_logits[f_idx], cfg.aug_mode)
    loss_f = compute_forget_loss(sub_f, orig_batch, pseudo[f_idx].to(DEVICE), cfg)
    loss_r = retain_kl_loss(student, proxy_images[p_idx], orig_proxy_logits[p_idx], cfg.temperature)
    total = loss_f + cfg.retain_kl_weight * loss_r + cfg.l2_weight * l2_anchor_loss(student, orig_params)
    opt.zero_grad(set_to_none=True)
    total.backward()
    for n, p in student.named_parameters():
        if p.grad is not None and n in masks:
            p.grad.mul_(masks[n])
    mask_absent_head_grads(student, absent_mask)
    opt.step()
    restore_absent_head_rows(student, orig_model, absent_mask)
    return forget_images[f_idx], pseudo[f_idx]


def run_salun_npo(
    orig_model: torch.nn.Module,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
) -> torch.nn.Module:
    """Run SalUn + forget loss + Retain KL + Absent-Class Freeze."""
    torch.manual_seed(cfg.seed)
    base = (
        run_ssd(orig_model, forget_images, orig_forget_logits, proxy_images, orig_proxy_logits, absent_mask, cfg)
        if cfg.use_ssd_init
        else orig_model
    )
    student = prepare_student(base, cfg.freeze_until_block)
    pseudo = orig_forget_logits.argmax(dim=-1)
    masks = compute_tempered_saliency(student, forget_images, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    opt = build_optimizer(student, cfg.lr, cfg.head_lr_mult)

    for step in range(1, cfg.steps + 1):
        f_raw, f_lbl = _salun_step(
            student, orig_model, opt, forget_images, orig_forget_logits,
            pseudo, proxy_images, orig_proxy_logits, orig_params, masks, absent_mask, cfg,
        )
        if step % 10 == 0 and should_stop_early(student, f_raw, f_lbl, cfg.target_forget_ce):
            break
    return student.eval()
