"""Track I: Erase-Then-Heal (Extreme Unlearning followed by Bottlenecked Structural LoRA Re-binding)."""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from augment import augment_training_batch
from auto_tune_4gpu import _probe_step, _sample_probes
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
from methods.common import (
    get_batch_orig_logits,
    mask_absent_head_grads,
    npo_loss,
    prepare_student,
    restore_absent_head_rows,
    retain_kl_loss,
)
from methods.lora_unlearn import LoRALinear
from methods.salun_npo import compute_tempered_saliency
from methods.track_h import (
    _h3_l2_anchor_loss,
    build_track_h_optimizer,
    clamp_layerscale_gammas,
    project_head_grad_orthogonal,
)
from task_template import NUM_CLASSES, preprocess


@dataclass(frozen=True)
class TrackIConfig:
    """Hyperparameters for Piste I (Phase 1 Extreme Erasure + Phase 2 Structural LoRA Healing)."""

    base: MethodConfig
    alpha_proto: float = 0.80
    stuck_boost: float = 1.8
    stuck_ce_thresh: float = 0.25
    gamma_lr_mult: float = 1.0
    gamma_clamp_rel: float = 0.5
    phase1_target_f95: float = 0.06
    phase1_max_ce: float = 3.4
    phase1_max_dtheta: float = 1e9
    phase1_probe_every: int = 15
    saliency_refresh_frac: float = 0.0
    heal_steps: int = 50
    heal_lr: float = 1.5e-4
    heal_rank: int = 4
    heal_freeze_until_block: int = 6
    heal_ce_floor: float = 0.95
    heal_ce_ceiling: float = 1.60
    w_proxy_kl: float = 3.5
    w_dark_liaison: float = 1.0
    w_ceiling_heal: float = 0.35
    w_floor_guard: float = 1.6
    heal_l2_weight: float = 1.5e-3
    dark_temp: float = 3.0


def compute_extreme_forget_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    class_ce_ema: torch.Tensor,
    cfg: TrackIConfig,
) -> torch.Tensor:
    """Phase 1 NPO with stuck-sample boost (ce_i < stuck_ce_thresh) and class velocity governor."""
    temp, beta, tgt = cfg.base.temperature, cfg.base.npo_beta, max(cfg.base.target_forget_ce, 1e-4)
    sub_lp = F.log_softmax(sub_logits.float() / temp, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / temp, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    npo_i = (2.0 / beta) * F.softplus(beta * (sub_lp - orig_lp))
    with torch.no_grad():
        ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
        counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
        sums = torch.zeros(NUM_CLASSES, device=ce_i.device).scatter_add_(0, labels, ce_i)
        present = counts > 0
        class_ce_ema[present] = 0.85 * class_ce_ema[present] + 0.15 * (sums[present] / counts[present])
        stuck = (ce_i < cfg.stuck_ce_thresh).float()
        inst_w = torch.clamp(1.0 + cfg.stuck_boost * stuck - 0.35 * (ce_i / tgt), 0.25, 3.5)
        inst_w = inst_w * (ce_i < tgt * 1.15).float()
        cls_w = torch.clamp((tgt - class_ce_ema[labels]) / tgt, 0.15, 1.60)
        weights = (inst_w * cls_w) / (inst_w * cls_w).mean().clamp_min(1e-4)
    return (npo_i * weights).mean()


def _apply_phase1_grads(
    student: nn.Module,
    orig_model: nn.Module,
    opt: torch.optim.Optimizer,
    total: torch.Tensor,
    masks: dict[str, torch.Tensor],
    orig_params: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    cfg: TrackIConfig,
) -> None:
    opt.zero_grad(set_to_none=True)
    total.backward()
    for n, p in student.named_parameters():
        if p.grad is not None and n in masks:
            p.grad.mul_(masks[n])
    project_head_grad_orthogonal(student, orig_model.head.weight.detach(), cfg.alpha_proto)
    mask_absent_head_grads(student, absent_mask)
    opt.step()
    clamp_layerscale_gammas(student, orig_params, cfg.gamma_clamp_rel)
    restore_absent_head_rows(student, orig_model, absent_mask)


def _phase1_step(
    student: nn.Module,
    orig_model: nn.Module,
    opt: torch.optim.Optimizer,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    pseudo: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    class_ce_ema: torch.Tensor,
    cfg: TrackIConfig,
) -> None:
    student.train()
    dev = next(student.parameters()).device
    f_idx = torch.randint(0, len(f_x), (cfg.base.batch_size,))
    p_idx = torch.randint(0, len(p_x), (cfg.base.batch_size,))
    f_batch = augment_training_batch(f_x[f_idx], mode=cfg.base.aug_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        sub_f = student(preprocess(f_batch).to(dev))
    orig_batch = get_batch_orig_logits(orig_model, f_batch, orig_f[f_idx], cfg.base.aug_mode)
    loss_f = compute_extreme_forget_loss(sub_f, orig_batch.to(dev), pseudo[f_idx].to(dev), class_ce_ema, cfg)
    loss_r = retain_kl_loss(student, p_x[p_idx], orig_p[p_idx], cfg.base.temperature)
    total = loss_f + cfg.base.retain_kl_weight * loss_r + cfg.base.l2_weight * _h3_l2_anchor_loss(student, orig_params)
    _apply_phase1_grads(student, orig_model, opt, total, masks, orig_params, absent_mask, cfg)


def _run_phase1_erasure(
    orig_model: nn.Module, f_x: torch.Tensor, orig_f: torch.Tensor,
    p_x: torch.Tensor, orig_p: torch.Tensor, absent_mask: torch.Tensor, cfg: TrackIConfig,
) -> nn.Module:
    """Phase 1: Push extreme unlearning until stubborn >95% traces collapse."""
    student = prepare_student(orig_model, cfg.base.freeze_until_block)
    pseudo = orig_f.argmax(dim=-1)
    masks = compute_tempered_saliency(student, f_x, pseudo, cfg.base.salun_keep_ratio, cfg.base.temperature, seed=cfg.base.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    opt = build_track_h_optimizer(student, cfg.base.lr, cfg.base.head_lr_mult, cfg.gamma_lr_mult)
    class_ce_ema = torch.zeros(NUM_CLASSES, device=next(student.parameters()).device)
    probes = _sample_probes(f_x, pseudo, p_x, orig_p, seed=cfg.base.seed)
    best_key, best_sd = (-1e9, -1e9), None
    for step in range(1, cfg.base.steps + 1):
        _phase1_step(student, orig_model, opt, f_x, orig_f, pseudo, p_x, orig_p, orig_params, masks, absent_mask, class_ce_ema, cfg)
        if cfg.saliency_refresh_frac > 0.0 and step == int(cfg.base.steps * cfg.saliency_refresh_frac):
            masks = compute_tempered_saliency(student, f_x, pseudo, cfg.base.salun_keep_ratio, cfg.base.temperature, seed=cfg.base.seed + step)
        if step % cfg.phase1_probe_every == 0:
            ce_mean, _s, p50, f95, pkl, d_l2 = _probe_step(student, *probes, orig_params)
            key = (1.0 if p50 >= cfg.base.target_forget_ce * 0.85 else 0.0, -f95 - 0.5 * pkl)
            if ce_mean <= cfg.phase1_max_ce and d_l2 <= cfg.phase1_max_dtheta and key > best_key:
                best_key, best_sd = key, {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            if (f95 <= cfg.phase1_target_f95 and p50 >= cfg.base.target_forget_ce) or ce_mean > cfg.phase1_max_ce or d_l2 > cfg.phase1_max_dtheta:
                break
    if best_sd is not None:
        student.load_state_dict(best_sd)
    restore_absent_head_rows(student, orig_model, absent_mask)
    return student


def inject_frozen_head_lora(model: nn.Module, freeze_until_block: int, rank: int) -> None:
    """Freeze all base weights including head, and inject FP32 LoRA adapters on upper blocks."""
    model.requires_grad_(False)
    for idx in range(freeze_until_block, len(model.blocks)):
        blk = model.blocks[idx]
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank)
        blk.attn.proj = LoRALinear(blk.attn.proj, rank)
        blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, rank)
        blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, rank)


def merge_track_i_lora(model: nn.Module, freeze_until_block: int) -> None:
    """Merge all LoRA adapters back into standard FP32 nn.Linear layers."""
    for idx in range(freeze_until_block, len(model.blocks)):
        blk = model.blocks[idx]
        blk.attn.qkv = blk.attn.qkv.merge_to_linear()
        blk.attn.proj = blk.attn.proj.merge_to_linear()
        blk.mlp.fc1 = blk.mlp.fc1.merge_to_linear()
        blk.mlp.fc2 = blk.mlp.fc2.merge_to_linear()


def compute_dark_liaison_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 3.0,
) -> torch.Tensor:
    """Align KL over the 99 non-target classes by masking out top-1 class y_i in both logits."""
    idx = labels.to(sub_logits.device).unsqueeze(1)
    orig_masked = orig_logits.float().to(sub_logits.device).scatter(1, idx, -1e9)
    sub_masked = sub_logits.float().scatter(1, idx, -1e9)
    return compute_kl_at_temperature(orig_masked, sub_masked, temperature).mean()


def compute_ceiling_heal_loss(
    sub_logits: torch.Tensor,
    labels: torch.Tensor,
    ce_ceiling: float,
) -> torch.Tensor:
    """Pull over-damaged samples (ce_i > ce_ceiling) back toward ce_ceiling; zero grad when ce_i <= ce_ceiling."""
    ce_i = F.cross_entropy(sub_logits.float(), labels.to(sub_logits.device), reduction="none")
    excess = (ce_i - ce_ceiling).clamp_min(0.0)
    return F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=0.5, reduction="mean")


def compute_floor_guard_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    ce_floor: float,
) -> torch.Tensor:
    """Prevent re-memorization by pushing ce_i < ce_floor back up; zero grad when ce_i >= ce_floor."""
    y = labels.to(sub_logits.device)
    ce_i = F.cross_entropy(sub_logits.float(), y, reduction="none")
    deficit = (ce_floor - ce_i).clamp_min(0.0)
    l_hinge = F.smooth_l1_loss(deficit, torch.zeros_like(deficit), beta=0.3, reduction="mean")
    l_npo = npo_loss(sub_logits, orig_logits.to(sub_logits.device), y, beta=1.2, temperature=3.5, max_ce=ce_floor)
    return l_npo + 0.6 * l_hinge


def _collect_orig_lora_weights(orig_model: nn.Module, freeze_until_block: int, dev: torch.device) -> dict[str, torch.Tensor]:
    """Snapshot original weights for the LoRA-adapted attention/MLP submodules."""
    out: dict[str, torch.Tensor] = {}
    for idx in range(freeze_until_block, len(orig_model.blocks)):
        blk = orig_model.blocks[idx]
        for mod, attr in (("attn", "qkv"), ("attn", "proj"), ("mlp", "fc1"), ("mlp", "fc2")):
            out[f"blocks.{idx}.{mod}.{attr}"] = getattr(getattr(blk, mod), attr).weight.detach().clone().to(dev)
    return out


def _collect_lora_modules(student: nn.Module, freeze_until_block: int) -> dict[str, LoRALinear]:
    """Collect injected LoRALinear modules by their block/submodule key."""
    out: dict[str, LoRALinear] = {}
    for idx in range(freeze_until_block, len(student.blocks)):
        blk = student.blocks[idx]
        for mod, attr in (("attn", "qkv"), ("attn", "proj"), ("mlp", "fc1"), ("mlp", "fc2")):
            out[f"blocks.{idx}.{mod}.{attr}"] = getattr(getattr(blk, mod), attr)
    return out


def compute_merged_lora_l2_loss(lora_modules: dict[str, LoRALinear], orig_lora_w: dict[str, torch.Tensor]) -> torch.Tensor:
    """Anchor merged weight W_base + scale * (B @ A) directly to orig_model weights."""
    first_w = next(iter(orig_lora_w.values()))
    total = torch.tensor(0.0, device=first_w.device)
    for key, mod in lora_modules.items():
        merged_w = mod.base.weight.float() + mod.scale * (mod.lora_b @ mod.lora_a)
        total = total + F.mse_loss(merged_w, orig_lora_w[key].float(), reduction="sum")
    return total


def _phase2_step(
    student: nn.Module, orig_model: nn.Module, opt: torch.optim.Optimizer,
    f_x: torch.Tensor, orig_f: torch.Tensor, pseudo: torch.Tensor,
    p_x: torch.Tensor, orig_p: torch.Tensor,
    lora_modules: dict[str, LoRALinear], orig_lora_w: dict[str, torch.Tensor], cfg: TrackIConfig,
) -> None:
    student.train()
    dev = next(student.parameters()).device
    f_idx = torch.randint(0, len(f_x), (cfg.base.batch_size,))
    p_idx = torch.randint(0, len(p_x), (cfg.base.batch_size,))
    f_batch = augment_training_batch(f_x[f_idx], mode=cfg.base.aug_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        sub_f = student(preprocess(f_batch).to(dev))
    orig_batch = get_batch_orig_logits(orig_model, f_batch, orig_f[f_idx], cfg.base.aug_mode)
    y_batch = pseudo[f_idx].to(dev)
    l_proxy = retain_kl_loss(student, p_x[p_idx], orig_p[p_idx], 2.0)
    l_dark = compute_dark_liaison_loss(sub_f, orig_batch, y_batch, cfg.dark_temp)
    l_ceil = compute_ceiling_heal_loss(sub_f, y_batch, cfg.heal_ce_ceiling)
    l_floor = compute_floor_guard_loss(sub_f, orig_batch, y_batch, cfg.heal_ce_floor)
    l_reg = cfg.heal_l2_weight * compute_merged_lora_l2_loss(lora_modules, orig_lora_w)
    total = cfg.w_proxy_kl * l_proxy + cfg.w_dark_liaison * l_dark + cfg.w_ceiling_heal * l_ceil + cfg.w_floor_guard * l_floor + l_reg
    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
    opt.step()


def _load_lora_snapshot(student: nn.Module, best_lora: dict[str, torch.Tensor] | None) -> None:
    if best_lora is None:
        return
    with torch.no_grad():
        for n, p in student.named_parameters():
            if n in best_lora:
                p.data.copy_(best_lora[n].to(p.device))


def _run_phase2_healing(
    student: nn.Module, orig_model: nn.Module, f_x: torch.Tensor, orig_f: torch.Tensor,
    p_x: torch.Tensor, orig_p: torch.Tensor, absent_mask: torch.Tensor, cfg: TrackIConfig,
) -> nn.Module:
    """Phase 2: Freeze theta_extreme & head, train bottlenecked LoRA with merged-weight L2 anchor."""
    dev = next(student.parameters()).device
    orig_lora_w = _collect_orig_lora_weights(orig_model, cfg.heal_freeze_until_block, dev)
    inject_frozen_head_lora(student, cfg.heal_freeze_until_block, cfg.heal_rank)
    lora_modules = _collect_lora_modules(student, cfg.heal_freeze_until_block)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.heal_lr, weight_decay=0.0)
    pseudo = orig_f.argmax(dim=-1)
    probes = _sample_probes(f_x, pseudo, p_x, orig_p, seed=cfg.base.seed + 7)
    best_score, best_lora, fb_score, fb_lora = -1e9, None, -1e9, None
    for step in range(1, cfg.heal_steps + 1):
        _phase2_step(student, orig_model, opt, f_x, orig_f, pseudo, p_x, orig_p, lora_modules, orig_lora_w, cfg)
        if step % 5 != 0:
            continue
        _cm, _cs, p50, f95, pkl, _dl = _probe_step(student, *probes, {})
        snap = -pkl - 0.30 * f95 - 0.15 * abs(p50 - 0.90)
        if f95 <= 0.14 and 0.55 <= p50 <= 1.75 and snap > best_score:
            best_score, best_lora = snap, {n: p.detach().cpu().clone() for n, p in student.named_parameters() if p.requires_grad}
        if f95 <= 0.22 and p50 >= 0.45 and snap > fb_score:
            fb_score, fb_lora = snap, {n: p.detach().cpu().clone() for n, p in student.named_parameters() if p.requires_grad}
        if f95 > 0.25 and (best_lora is not None or fb_lora is not None):
            break
    _load_lora_snapshot(student, best_lora if best_lora is not None else fb_lora)
    merge_track_i_lora(student, cfg.heal_freeze_until_block)
    restore_absent_head_rows(student, orig_model, absent_mask)
    return student.eval()


def run_track_i(
    orig_model: nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: TrackIConfig,
    return_phase1: bool = False,
) -> nn.Module | tuple[nn.Module, nn.Module]:
    """Run Piste I (Erase-Then-Heal): Phase 1 extreme erasure + optional Phase 2 structural LoRA healing."""
    torch.manual_seed(cfg.base.seed)
    phase1_model = _run_phase1_erasure(orig_model, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
    if cfg.heal_steps <= 0:
        phase1_eval = phase1_model.eval()
        return (phase1_eval, phase1_eval) if return_phase1 else phase1_eval
    if return_phase1:
        p1_copy = {k: v.detach().cpu().clone() for k, v in phase1_model.state_dict().items()}
    healed = _run_phase2_healing(phase1_model, orig_model, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
    if not return_phase1:
        return healed
    p1_shell = prepare_student(orig_model, cfg.base.freeze_until_block)
    p1_shell.load_state_dict({k: v.to(next(p1_shell.parameters()).device) for k, v in p1_copy.items()})
    return healed, p1_shell.eval()
