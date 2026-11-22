"""True SCRUB+R (NeurIPS 2023) with Patch-Dropout difficulty weighting and Rewind."""

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from augment import augment_training_batch
from eval_harness import compute_kl_at_temperature
from methods.common import (
    build_optimizer,
    get_batch_orig_logits,
    l2_anchor_loss,
    mask_absent_head_grads,
    prepare_student,
    restore_absent_head_rows,
    retain_kl_loss,
)
from methods.patch_drop_targets import estimate_patchdrop_gold_logits
from methods.salun_npo import compute_tempered_saliency
from task_template import DEVICE, predict, preprocess


@dataclass(frozen=True)
class ScrubRConfig:
    """Hyperparameters for alternating SCRUB+R with Patch-Dropout difficulty weighting."""

    name: str
    total_epochs: int = 10
    max_epochs: int = 6
    max_steps_per_epoch: int = 18
    min_steps_per_epoch: int = 24
    batch_size_f: int = 32
    batch_size_r: int = 64
    lr_max: float = 1.1e-4
    lr_min: float = 8.0e-5
    lr_decay_tail: float = 0.25
    head_lr_mult: float = 1.0
    temperature: float = 2.0
    max_loss_kind: str = "npo"
    npo_beta: float = 1.8
    target_forget_ce: float = 1.50
    pd_alpha: float = 0.75
    pd_gate_scale: float = 1.0
    retain_kl_weight: float = 2.4
    retain_ce_weight: float = 0.15
    l2_weight: float = 1.6e-3
    salun_keep_ratio: float = 0.20
    freeze_until_block: int = 6
    aug_mode: str = "moderate"
    target_mu_sig: float = 3.80
    target_p50: float = 1.00
    max_proxy_kl: float = 0.050
    use_rewind: bool = True
    seed: int = 42


@torch.no_grad()
def compute_pd_difficulty_ratio(
    pd_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-image Patch-Dropout CE and normalized difficulty ratio in [0.25, 2.5]."""
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    mean_ce = ce_pd.mean().clamp_min(1e-4)
    ratio = (ce_pd / mean_ce).clamp(0.25, 2.5)
    return ce_pd, ratio


def scrub_max_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    diff_ratio: torch.Tensor,
    cfg: ScrubRConfig,
) -> torch.Tensor:
    """Compute difficulty-weighted & gated forget ascent loss (NPO or T^2 KL ascent)."""
    with torch.no_grad():
        ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
        max_ce_i = cfg.target_forget_ce * diff_ratio if cfg.pd_gate_scale > 0.0 else cfg.target_forget_ce
        gate = (ce_i < max_ce_i).to(sub_logits.dtype)
        weights = diff_ratio.pow(cfg.pd_alpha) if cfg.pd_alpha > 0.0 else 1.0
    sub_lp = F.log_softmax(sub_logits.float() / cfg.temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / cfg.temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    npo_i = (2.0 / cfg.npo_beta) * F.softplus(cfg.npo_beta * (sub_lp - orig_lp))
    if cfg.max_loss_kind == "kl_ascent":
        kl_i = compute_kl_at_temperature(orig_logits, sub_logits, cfg.temperature)
        # 0.05*npo_i breaks the exact KL(orig||orig) zero-gradient saddle point at step 0.
        loss_i = -(cfg.temperature ** 2) * kl_i + 0.05 * npo_i
        return (loss_i * gate * weights).mean()
    return (npo_i * gate * weights).mean()


def _apply_masked_step(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    loss: torch.Tensor,
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
) -> None:
    opt.zero_grad(set_to_none=True)
    loss.backward()
    for name, param in student.named_parameters():
        if param.grad is not None and name in masks:
            param.grad.mul_(masks[name])
    mask_absent_head_grads(student, absent_mask)
    opt.step()
    restore_absent_head_rows(student, orig_model, absent_mask)


def _do_max_epoch(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt_max: torch.optim.Optimizer,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    pseudo: torch.Tensor,
    diff_ratio: torch.Tensor,
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    cfg: ScrubRConfig,
) -> None:
    student.train()
    dev = next(student.parameters()).device
    for _ in range(cfg.max_steps_per_epoch):
        idx = torch.randint(0, len(f_x), (cfg.batch_size_f,))
        f_batch = augment_training_batch(f_x[idx], mode=cfg.aug_mode)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            sub_f = student(preprocess(f_batch).to(dev))
        orig_b = get_batch_orig_logits(orig_model, f_batch, orig_f[idx], cfg.aug_mode)
        loss = scrub_max_loss(sub_f, orig_b, pseudo[idx].to(dev), diff_ratio[idx].to(dev), cfg)
        _apply_masked_step(student, orig_model, opt_max, loss, masks, absent_mask)


def _do_min_epoch(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt_min: torch.optim.Optimizer,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    orig_p_lbl: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    cfg: ScrubRConfig,
) -> None:
    student.train()
    dev = next(student.parameters()).device
    for _ in range(cfg.min_steps_per_epoch):
        idx = torch.randint(0, len(p_x), (cfg.batch_size_r,))
        loss_kl = retain_kl_loss(student, p_x[idx], orig_p[idx], cfg.temperature)
        loss = cfg.retain_kl_weight * loss_kl + cfg.l2_weight * l2_anchor_loss(student, orig_params)
        if cfg.retain_ce_weight > 0.0:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                sub_p = student(preprocess(p_x[idx]).to(dev))
            loss = loss + cfg.retain_ce_weight * F.cross_entropy(sub_p.float(), orig_p_lbl[idx].to(dev))
        _apply_masked_step(student, orig_model, opt_min, loss, masks, absent_mask)


@torch.no_grad()
def _eval_epoch_snapshot(
    epoch: int,
    student: torch.nn.Module,
    f_x: torch.Tensor,
    pseudo: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    cfg: ScrubRConfig,
) -> dict:
    sub_f = predict(student, f_x, batch_size=128)
    ce_i = F.cross_entropy(sub_f.float(), pseudo, reduction="none")
    ce_mean, ce_std = float(ce_i.mean()), float(ce_i.std(correction=0))
    mu_sig = ce_mean + ce_std
    p50 = float(ce_i.median())
    f95 = float((torch.softmax(sub_f.float(), dim=-1).max(dim=-1).values > 0.95).float().mean())
    sub_p = predict(student, p_x, batch_size=128)
    pkl = float(compute_kl_at_temperature(orig_p, sub_p, cfg.temperature).mean())
    d_l2 = float(sum(float((p.detach() - orig_params[n]).pow(2).sum()) for n, p in student.named_parameters() if n in orig_params) ** 0.5)
    dist = abs(mu_sig - cfg.target_mu_sig) + 0.8 * abs(p50 - cfg.target_p50)
    dist += 15.0 * max(0.0, pkl - cfg.max_proxy_kl) + 0.5 * max(0.0, f95 - 0.25)
    return {
        "epoch": epoch, "ce_mean": round(ce_mean, 4), "ce_std": round(ce_std, 4),
        "mu_plus_sig": round(mu_sig, 4), "p50": round(p50, 4), "frac_gt_95": round(f95, 4),
        "proxy_kl": round(pkl, 4), "delta_l2": round(d_l2, 4), "rewind_dist": round(dist, 4),
    }


def _select_rewind_snapshot(snapshots: list[dict], use_rewind: bool) -> dict:
    """Pick epoch snapshot closest to target sweet spot, or final snapshot if rewind is off."""
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    if not use_rewind:
        return snapshots[-1]
    active = [s for s in snapshots if s["p50"] >= 0.15]
    pool = active if active else snapshots
    return min(pool, key=lambda s: s["rewind_dist"])


def _scale_optimizer_lr(opt: torch.optim.Optimizer, factor: float) -> None:
    for group in opt.param_groups:
        group["lr"] *= factor


def run_scrub_r(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: ScrubRConfig,
    pd_logits: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, list[dict]]:
    """Run alternating SCRUB+R (Max-Step + Min-Step + Tail + Rewind) with Patch-Dropout weights."""
    torch.manual_seed(cfg.seed)
    pseudo = orig_f.argmax(dim=-1)
    if pd_logits is None:
        pd_logits = estimate_patchdrop_gold_logits(orig_model, f_x, drop_prob=0.45, num_views=4, seed=cfg.seed)
    _, diff_ratio = compute_pd_difficulty_ratio(pd_logits, pseudo)
    student = prepare_student(orig_model, cfg.freeze_until_block)
    masks = compute_tempered_saliency(student, f_x, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    opt_max = build_optimizer(student, cfg.lr_max, cfg.head_lr_mult)
    opt_min = build_optimizer(student, cfg.lr_min, cfg.head_lr_mult)
    orig_p_lbl = orig_p.argmax(dim=-1)

    snapshots: list[dict] = []
    for epoch in range(cfg.total_epochs):
        if epoch == cfg.max_epochs:
            _scale_optimizer_lr(opt_min, cfg.lr_decay_tail)
        if epoch < cfg.max_epochs:
            _do_max_epoch(student, orig_model, opt_max, f_x, orig_f, pseudo, diff_ratio, masks, absent_mask, cfg)
        _do_min_epoch(student, orig_model, opt_min, p_x, orig_p, orig_p_lbl, orig_params, masks, absent_mask, cfg)
        snap = _eval_epoch_snapshot(epoch, student, f_x, pseudo, p_x, orig_p, orig_params, cfg)
        snap["weights"] = {n: p.detach().cpu().clone() for n, p in student.named_parameters() if n in masks}
        snapshots.append(snap)

    best = _select_rewind_snapshot(snapshots, cfg.use_rewind)
    with torch.no_grad():
        for name, param in student.named_parameters():
            if name in best["weights"]:
                param.copy_(best["weights"][name].to(param.device))
    history = [{k: v for k, v in s.items() if k != "weights"} | {"selected": s["epoch"] == best["epoch"]} for s in snapshots]
    return student.eval(), history
