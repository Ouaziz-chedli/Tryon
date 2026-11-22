"""Phase 3 two-view asymmetric unlearning (rupture plan 3.1/3.2).

Each sampled batch gets a clean view (identity/hflip) and a moderate-aug view.
The bounded NPO forget loss is applied on `forget_view` while the retain KL
anchor is applied on the complementary view; mono controls collapse both onto
the same single view to isolate the two-view effect.
"""

import torch

from augment import augment_training_batch
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
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
from methods.salun_npo import compute_tempered_saliency
from task_template import DEVICE, preprocess

VIEW_MODES = {"clean": "none", "aug": "moderate"}


def view_modes(forget_view: str, mono: bool) -> tuple[str, str]:
    """Return (forget_mode, anchor_mode); two-view arms are complementary, mono collapses."""
    if forget_view not in VIEW_MODES:
        raise ValueError(f"forget_view must be one of {sorted(VIEW_MODES)}, got {forget_view!r}")
    forget_mode = VIEW_MODES[forget_view]
    if mono:
        return forget_mode, forget_mode
    anchor_view = "clean" if forget_view == "aug" else "aug"
    return forget_mode, VIEW_MODES[anchor_view]


def _anchor_kl(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    proxy_batch: torch.Tensor,
    cached_logits: torch.Tensor,
    anchor_mode: str,
    temperature: float,
) -> torch.Tensor:
    """KL(p_orig || p_sub) on the anchor view; cached raw logits only match mode 'none'."""
    if anchor_mode == "none":
        return retain_kl_loss(student, proxy_batch, cached_logits, temperature)
    orig_logits = get_batch_orig_logits(orig_model, proxy_batch, cached_logits, anchor_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        sub_logits = student(preprocess(proxy_batch))
    return compute_kl_at_temperature(orig_logits.float(), sub_logits, temperature).mean()


def _two_view_step(
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
    forget_mode: str,
    anchor_mode: str,
    forget_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One optimizer step: NPO on the forget view + KL anchor on the anchor view + L2."""
    student.train()
    f_idx = torch.randint(0, len(forget_images), (cfg.batch_size,))
    p_idx = torch.randint(0, len(proxy_images), (cfg.batch_size,))
    f_view = augment_training_batch(forget_images[f_idx], mode=forget_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        sub_f = student(preprocess(f_view))
    orig_f = get_batch_orig_logits(orig_model, f_view, orig_forget_logits[f_idx], forget_mode)
    loss_f = compute_forget_loss(sub_f, orig_f, pseudo[f_idx].to(DEVICE), cfg)
    p_view = augment_training_batch(proxy_images[p_idx], mode=anchor_mode)
    loss_r = _anchor_kl(student, orig_model, p_view, orig_proxy_logits[p_idx], anchor_mode, cfg.temperature)
    total = forget_weight * loss_f + cfg.retain_kl_weight * loss_r + cfg.l2_weight * l2_anchor_loss(student, orig_params)
    opt.zero_grad(set_to_none=True)
    total.backward()
    for n, p in student.named_parameters():
        if p.grad is not None and n in masks:
            p.grad.mul_(masks[n])
    mask_absent_head_grads(student, absent_mask)
    opt.step()
    restore_absent_head_rows(student, orig_model, absent_mask)
    return forget_images[f_idx], pseudo[f_idx]


def run_phase3_two_view(
    orig_model: torch.nn.Module,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
    forget_view: str = "aug",
    mono: bool = False,
    forget_weight: float = 1.0,
) -> torch.nn.Module:
    """SalUn mask + bounded NPO on the forget view + KL anchor on the complementary view."""
    torch.manual_seed(cfg.seed)
    student = prepare_student(orig_model, cfg.freeze_until_block)
    pseudo = orig_forget_logits.argmax(dim=-1)
    masks = compute_tempered_saliency(student, forget_images, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    opt = build_optimizer(student, cfg.lr, cfg.head_lr_mult)
    forget_mode, anchor_mode = view_modes(forget_view, mono)
    for step in range(1, cfg.steps + 1):
        f_raw, f_lbl = _two_view_step(
            student, orig_model, opt, forget_images, orig_forget_logits,
            pseudo, proxy_images, orig_proxy_logits, orig_params, masks, absent_mask,
            cfg, forget_mode, anchor_mode, forget_weight,
        )
        if step % 10 == 0 and should_stop_early(student, f_raw, f_lbl, cfg.target_forget_ce):
            break
    return student.eval()
