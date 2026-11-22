"""Per-image Patch-Dropout Virtual Gold targets to optimize Track 3 prediction_score."""

import torch
import torch.nn.functional as F

from augment import augment_training_batch
from auto_tune_4gpu import _build_train_proxy
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
from methods.common import (
    build_optimizer,
    get_batch_orig_logits,
    l2_anchor_loss,
    mask_absent_head_grads,
    npo_loss,
    prepare_student,
    restore_absent_head_rows,
    retain_kl_loss,
)
from methods.salun_npo import compute_tempered_saliency
from task_template import DEVICE, NUM_CLASSES, predict, preprocess


def mask_random_patches(
    images_uint8: torch.Tensor,
    drop_prob: float = 0.45,
    patch_size: int = 16,
    seed: int = 42,
) -> torch.Tensor:
    """Zero/mean-fill `drop_prob` fraction of 16x16 ViT patches per image."""
    if drop_prob <= 0.0:
        return images_uint8.clone()
    n, h, w, _ = images_uint8.shape
    gh, gw = h // patch_size, w // patch_size
    gen = torch.Generator(device=images_uint8.device).manual_seed(seed)
    keep = (torch.rand((n, gh, gw, 1), generator=gen, device=images_uint8.device) >= drop_prob).to(torch.uint8)
    keep_px = keep.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return images_uint8 * keep_px + 128 * (1 - keep_px)


@torch.no_grad()
def estimate_patchdrop_gold_logits(
    orig_model: torch.nn.Module,
    images_uint8: torch.Tensor,
    drop_prob: float = 0.45,
    num_views: int = 4,
    batch_size: int = 128,
    seed: int = 42,
) -> torch.Tensor:
    """Estimate per-image Virtual Gold logits via multi-view Patch-Dropout on orig_model."""
    orig_model.eval()
    acc = torch.zeros((len(images_uint8), NUM_CLASSES), dtype=torch.float32)
    for v in range(num_views):
        masked = mask_random_patches(images_uint8, drop_prob=drop_prob, seed=seed + v)
        if v % 2 == 1:
            masked = torch.flip(masked, dims=[2])
        acc += predict(orig_model, masked, batch_size=batch_size).float() / float(num_views)
    return acc


@torch.no_grad()
def calibrate_per_image_targets(
    pd_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mean_ce: float = 1.75,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale Patch-Dropout CE per image to match server sweet-spot mean while keeping 99 non-target logits."""
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    scale = float(target_mean_ce) / max(float(ce_pd.mean().item()), 1e-3)
    ce_tgt = (ce_pd * scale).clamp(0.05, 8.0)
    p_tgt = torch.exp(-ce_tgt).clamp(1e-4, 0.95).unsqueeze(1)
    z_orig = orig_logits.float()
    lse_rest = torch.logsumexp(z_orig.scatter(1, labels.unsqueeze(1), -1e9), dim=-1, keepdim=True)
    z_tgt = z_orig.scatter(1, labels.unsqueeze(1), lse_rest + torch.log(p_tgt / (1.0 - p_tgt)))
    return z_tgt, ce_tgt


def per_sample_calibrated_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_ce: torch.Tensor,
    labels: torch.Tensor,
    kl_weight: float = 1.0,
    npo_weight: float = 0.5,
    beta: float = 1.8,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Align per-sample T=2 distribution to Patch-Dropout Gold targets + sample-gated NPO."""
    kl_i = compute_kl_at_temperature(target_logits, sub_logits.float(), temperature)
    sub_lp = F.log_softmax(sub_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    npo_i = (2.0 / beta) * F.softplus(beta * (sub_lp - orig_lp))
    with torch.no_grad():
        ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
        gate = (ce_i < target_ce).to(npo_i.dtype)
    return kl_weight * kl_i.mean() + npo_weight * (npo_i * gate).mean()


def _pd_step(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    tgt_logits: torch.Tensor,
    tgt_ce: torch.Tensor,
    pseudo: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
    kl_weight: float,
    npo_weight: float,
) -> None:
    student.train()
    f_idx = torch.randint(0, len(f_x), (cfg.batch_size,))
    p_idx = torch.randint(0, len(p_x), (cfg.batch_size,))
    f_batch = augment_training_batch(f_x[f_idx], mode=cfg.aug_mode)
    dev = next(student.parameters()).device
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        sub_f = student(preprocess(f_batch).to(dev))
    orig_b = get_batch_orig_logits(orig_model, f_batch, orig_f[f_idx], cfg.aug_mode)
    loss_f = per_sample_calibrated_loss(
        sub_f, orig_b, tgt_logits[f_idx].to(dev), tgt_ce[f_idx].to(dev),
        pseudo[f_idx].to(dev), kl_weight, npo_weight, cfg.npo_beta, cfg.temperature,
    )
    loss_r = retain_kl_loss(student, p_x[p_idx], orig_p[p_idx], cfg.temperature)
    total = loss_f + cfg.retain_kl_weight * loss_r + cfg.l2_weight * l2_anchor_loss(student, orig_params)
    opt.zero_grad(set_to_none=True)
    total.backward()
    for n, p in student.named_parameters():
        if p.grad is not None and n in masks:
            p.grad.mul_(masks[n])
    mask_absent_head_grads(student, absent_mask)
    opt.step()
    restore_absent_head_rows(student, orig_model, absent_mask)


def run_patchdrop_unlearn(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
    pd_logits: torch.Tensor | None = None,
    kl_weight: float = 1.0,
    npo_weight: float = 0.5,
    target_mean_ce: float = 1.75,
) -> torch.nn.Module:
    """Run clean static-SalUn (f6) unlearning guided per-image by Patch-Dropout Gold targets."""
    torch.manual_seed(cfg.seed)
    pseudo = orig_f.argmax(dim=-1)
    if pd_logits is None:
        pd_logits = estimate_patchdrop_gold_logits(orig_model, f_x, drop_prob=0.45, num_views=4, seed=cfg.seed)
    tgt_logits, tgt_ce = calibrate_per_image_targets(pd_logits, orig_f, pseudo, target_mean_ce=target_mean_ce)
    student = prepare_student(orig_model, cfg.freeze_until_block)
    masks = compute_tempered_saliency(student, f_x, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    opt = build_optimizer(student, cfg.lr, cfg.head_lr_mult)
    for _ in range(cfg.steps):
        _pd_step(
            student, orig_model, opt, f_x, orig_f, tgt_logits, tgt_ce,
            pseudo, p_x, orig_p, orig_params, masks, absent_mask, cfg, kl_weight, npo_weight,
        )
    return student.eval()
