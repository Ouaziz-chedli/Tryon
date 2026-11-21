"""Closed-loop multi-strategy auto-tuner targeting the Track 3 sweet spot on 4 GPUs."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from augment import augment_training_batch
from contracts import EvalReport, MethodConfig
from eval_harness import compute_kl_at_temperature, evaluate_candidate, make_patchmix_proxy
from methods.abliteration import run_abliteration
from methods.bad_teacher import run_bad_teacher
from methods.common import (
    build_optimizer,
    compute_forget_loss,
    get_batch_orig_logits,
    l2_anchor_loss,
    mask_absent_head_grads,
    prepare_student,
    retain_kl_loss,
)
from methods.lora_unlearn import run_lora_npo
from methods.salun_npo import compute_tempered_saliency
from task_template import IMAGE_SIZE, NUM_CLASSES, load_model, predict, preprocess


def score_sweet_spot(ev: EvalReport) -> float:
    """Rank candidates by homogeneous forgetting while penalizing retain/weight drift."""
    fr, rp = ev.forget_raw, ev.retain_proxy
    if not ev.checkpoint or not ev.checkpoint.valid:
        return -1e6
    pen = 0.0
    if rp.proxy_kl_t2 > 0.038:
        pen += 25.0 * (rp.proxy_kl_t2 - 0.038)
    if rp.absent_classes_kl_t2 > 0.025:
        pen += 20.0 * (rp.absent_classes_kl_t2 - 0.025)
    if rp.delta_l2_total > 3.5:
        pen += 0.5 * (rp.delta_l2_total - 3.5)
    if fr.frac_gt_95 > 0.45:
        pen += 3.0 * (fr.frac_gt_95 - 0.45)
    cv = fr.ce_std / max(fr.ce_mean, 0.1)
    p50_gain = min(fr.ce_p50, 0.70) - 0.5 * max(0.0, fr.ce_p50 - 0.85)
    return 2.0 * p50_gain + (1.0 - fr.frac_gt_95) - 0.25 * cv - 5.0 * rp.proxy_kl_t2 - pen


def _build_train_proxy(forget_images: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Build 224x224 multi-grid PatchMix proxy (4x4 + 7x7) without resolution shortcuts."""
    half = len(forget_images) // 2
    if half == 0:
        return make_patchmix_proxy(forget_images, seed=seed, grid=4)
    return torch.cat([
        make_patchmix_proxy(forget_images[:half], seed=seed, grid=4),
        make_patchmix_proxy(forget_images[half:], seed=seed + 1, grid=7),
    ], dim=0)


@torch.no_grad()
def _probe_step(
    student: torch.nn.Module, f_probe: torch.Tensor, pseudo_probe: torch.Tensor,
    p_probe: torch.Tensor, orig_p_probe: torch.Tensor, orig_params: dict[str, torch.Tensor],
) -> tuple[float, float, float, float, float, float]:
    """Measure (ce_mean, ce_std, p50_ce, frac_gt_95, proxy_kl_t2, delta_l2) on probes."""
    sub_f = predict(student, f_probe, batch_size=128)
    ce_i = torch.nn.functional.cross_entropy(sub_f, pseudo_probe, reduction="none")
    frac95 = float((torch.softmax(sub_f, dim=-1).max(dim=-1).values > 0.95).float().mean())
    sub_p = predict(student, p_probe, batch_size=128)
    pkl = float(compute_kl_at_temperature(orig_p_probe, sub_p, temperature=2.0).mean().item())
    d_l2 = float(sum(float((p.detach() - orig_params[n]).pow(2).sum()) for n, p in student.named_parameters() if n in orig_params) ** 0.5)
    return float(ce_i.mean()), float(ce_i.std()), float(ce_i.median()), frac95, pkl, d_l2


def _snapshot_score(ce_mean: float, ce_std: float, p50: float, f95: float, pkl: float, d_l2: float) -> float:
    """Score intermediate snapshot: peak p50 in [0.35, 0.55], compress ce_std and drift."""
    p50_term = min(p50, 0.55) - 1.5 * max(0.0, p50 - 0.65)
    drift_pen = 15.0 * max(0.0, pkl - 0.032) + 0.8 * max(0.0, d_l2 - 3.2)
    return p50_term - 0.35 * ce_std - 0.8 * f95 - drift_pen


def _adjust_controller(opt: torch.optim.Optimizer, kl_w: float, l2_w: float, pkl: float, d_l2: float, p50: float) -> tuple[float, float]:
    """Closed-loop PI-style adjustment of retain KL, L2 anchor, and learning rate."""
    if pkl > 0.032 or d_l2 > 3.2:
        kl_w = min(8.0, kl_w * 1.35)
        l2_w = min(0.05, l2_w * 1.5)
        for g in opt.param_groups:
            g["lr"] = max(2e-5, g["lr"] * 0.85)
    elif pkl < 0.020 and d_l2 < 2.5 and p50 < 0.15:
        kl_w = max(1.0, kl_w * 0.85)
    return kl_w, l2_w


def _stratified_indices(labels: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Select up to `count` shuffled indices stratified across all classes in `labels`."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(labels), generator=gen)
    buckets = [perm[labels[perm] == c] for c in labels.unique(sorted=True)]
    picked: list[torch.Tensor] = []
    max_len = max((len(b) for b in buckets), default=0)
    for i in range(max_len):
        for b in buckets:
            if i >= len(b):
                continue
            picked.append(b[i])
            if len(picked) == count:
                return torch.stack(picked)
    return torch.stack(picked) if picked else perm[:0]


def _sample_probes(
    f_x: torch.Tensor, pseudo: torch.Tensor, p_x: torch.Tensor, orig_p: torch.Tensor,
    n_probe: int = 256, seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample shuffled stratified forget probe and shuffled retain probe."""
    f_idx = _stratified_indices(pseudo, min(n_probe, len(f_x)), seed)
    p_idx = torch.randperm(len(p_x), generator=torch.Generator().manual_seed(seed + 1))[: min(n_probe, len(p_x))]
    return f_x[f_idx], pseudo[f_idx], p_x[p_idx], orig_p[p_idx]


def _adaptive_step(
    student: torch.nn.Module, orig_model: torch.nn.Module, opt: torch.optim.Optimizer,
    f_x: torch.Tensor, orig_f: torch.Tensor, pseudo: torch.Tensor,
    p_x: torch.Tensor, orig_p: torch.Tensor, orig_params: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor], absent_mask: torch.Tensor,
    cfg: MethodConfig, kl_w: float, l2_w: float,
) -> None:
    student.train()
    f_idx = torch.randint(0, len(f_x), (cfg.batch_size,))
    p_idx = torch.randint(0, len(p_x), (cfg.batch_size,))
    f_batch = augment_training_batch(f_x[f_idx], mode=cfg.aug_mode)
    dev = next(student.parameters()).device
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        sub_f = student(preprocess(f_batch).to(dev))
    orig_batch = get_batch_orig_logits(orig_model, f_batch, orig_f[f_idx], cfg.aug_mode)
    loss_f = compute_forget_loss(sub_f, orig_batch, pseudo[f_idx].to(dev), cfg)
    loss_r = retain_kl_loss(student, p_x[p_idx], orig_p[p_idx], cfg.temperature)
    total = loss_f + kl_w * loss_r + l2_w * l2_anchor_loss(student, orig_params)
    opt.zero_grad(set_to_none=True)
    total.backward()
    for n, p in student.named_parameters():
        if p.grad is not None and n in masks:
            p.grad.mul_(masks[n])
    mask_absent_head_grads(student, absent_mask)
    opt.step()


def run_adaptive_salun_npo(
    base_model: torch.nn.Module, orig_model: torch.nn.Module, f_x: torch.Tensor,
    orig_f: torch.Tensor, p_x: torch.Tensor, orig_p: torch.Tensor,
    absent_mask: torch.Tensor, cfg: MethodConfig,
) -> torch.nn.Module:
    """Closed-loop SalUn+forget-loss with stratified probes and best-snapshot restore."""
    torch.manual_seed(cfg.seed)
    student = prepare_student(base_model, cfg.freeze_until_block)
    pseudo = orig_f.argmax(dim=-1)
    masks = compute_tempered_saliency(student, f_x, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    opt = build_optimizer(student, cfg.lr, cfg.head_lr_mult)
    kl_w, l2_w = cfg.retain_kl_weight, cfg.l2_weight
    best_score, best_sd = -1e9, None
    f_probe, pseudo_probe, p_probe, orig_p_probe = _sample_probes(f_x, pseudo, p_x, orig_p, seed=cfg.seed)

    for step in range(1, cfg.steps + 1):
        _adaptive_step(student, orig_model, opt, f_x, orig_f, pseudo, p_x, orig_p, orig_params, masks, absent_mask, cfg, kl_w, l2_w)
        if step % 15 != 0:
            continue
        ce_mean, ce_std, p50, f95, pkl, d_l2 = _probe_step(student, f_probe, pseudo_probe, p_probe, orig_p_probe, orig_params)
        kl_w, l2_w = _adjust_controller(opt, kl_w, l2_w, pkl, d_l2, p50)
        snap_score = _snapshot_score(ce_mean, ce_std, p50, f95, pkl, d_l2)
        if 0.18 <= p50 <= 0.85 and pkl <= 0.055 and snap_score > best_score:
            best_score = snap_score
            best_sd = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if p50 > cfg.target_ce_upper + 0.15 and best_sd is not None:
            break
    if best_sd is not None:
        student.load_state_dict(best_sd)
    return student.eval()


def _build_trials() -> list[tuple[str, MethodConfig]]:
    """Build multi-strategy trials including P1 two-sided/gold-mimic and P2 block-8 freeze."""
    trials: list[tuple[str, MethodConfig]] = []
    for p_star, frz, h_mult in ((0.55, 8, 1.5), (0.60, 8, 1.5), (0.60, 6, 1.0), (0.65, 8, 2.0)):
        name = f"GM_p{int(p_star*100)}_f{frz}_hm{h_mult:.1f}"
        trials.append(("adaptive", MethodConfig(name=name, strategy="salun_npo", loss_kind="gold_mimic", steps=210, lr=8e-5, gold_target_prob=p_star, retain_kl_weight=2.5, l2_weight=2e-3, salun_keep_ratio=0.25, freeze_until_block=frz, head_lr_mult=h_mult)))
    for ce_tgt, upper, frz in ((0.45, 0.65, 8), (0.50, 0.70, 8), (0.45, 0.65, 6)):
        name = f"TS_ce{ce_tgt:.2f}_u{upper:.2f}_f{frz}"
        trials.append(("adaptive", MethodConfig(name=name, strategy="salun_npo", loss_kind="two_sided_npo", steps=210, lr=8e-5, target_forget_ce=ce_tgt, target_ce_upper=upper, pullback_weight=0.5, retain_kl_weight=2.5, l2_weight=2e-3, salun_keep_ratio=0.25, freeze_until_block=frz)))
    for ce_tgt, lr, frz in ((0.50, 8e-5, 6), (0.50, 1.2e-4, 8)):
        name = f"AT_ce{ce_tgt:.2f}_lr{lr:.0e}_f{frz}"
        trials.append(("adaptive", MethodConfig(name=name, strategy="salun_npo", steps=210, lr=lr, target_forget_ce=ce_tgt, retain_kl_weight=2.5, l2_weight=2e-3, salun_keep_ratio=0.25, freeze_until_block=frz)))
    for ce_tgt, rank in ((0.45, 8), (0.50, 16)):
        trials.append(("lora_npo", MethodConfig(name=f"LORA_ce{ce_tgt:.2f}_r{rank}", strategy="lora_npo", loss_kind="two_sided_npo", steps=180, lr=2e-4, target_forget_ce=ce_tgt, target_ce_upper=0.65, retain_kl_weight=3.0, lora_rank=rank)))
    for alpha in (0.20, 0.30):
        trials.append(("ablit_adaptive", MethodConfig(name=f"FA_a{alpha:.2f}_gm60", strategy="salun_npo", loss_kind="gold_mimic", steps=180, lr=8e-5, gold_target_prob=0.60, retain_kl_weight=3.0, l2_weight=2e-3, abliteration_alpha=alpha, abliteration_rank=8, freeze_until_block=8)))
    return trials


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop 4-GPU multi-strategy auto-tuner.")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("output/autotune"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    orig_f = predict(orig, f_x, batch_size=128)
    p_x = _build_train_proxy(f_x)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f.argmax(dim=-1), minlength=NUM_CLASSES) == 0

    mine = [t for i, t in enumerate(_build_trials()) if i % args.num_workers == args.worker_id]
    rows = []
    for kind, cfg in mine:
        if kind == "adaptive":
            sub = run_adaptive_salun_npo(orig, orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
        elif kind == "ablit_adaptive":
            pre = run_abliteration(orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
            sub = run_adaptive_salun_npo(pre, orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
        elif kind == "lora_npo":
            sub = run_lora_npo(orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
        else:
            sub = run_bad_teacher(orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)

        ckpt = args.out_dir / f"{cfg.name}.pt"
        torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, ckpt)
        ev = evaluate_candidate(orig, sub, f_x, sub_path=ckpt)
        sw = score_sweet_spot(ev)
        row = {"name": cfg.name, "kind": kind, "sweet_spot": round(sw, 4), **asdict(ev.forget_raw), **asdict(ev.retain_proxy), "valid": ev.checkpoint.valid}
        rows.append(row)
        print(f"[gpu{args.worker_id}] {cfg.name:24s} | SW={sw:+.3f} | CE={row['ce_mean']:.3f}±{row['ce_std']:.2f} p50={row['ce_p50']:.3f} >95%={row['frac_gt_95']:.2f} | ProxyKL={row['proxy_kl_t2']:.4f} AbsKL={row['absent_classes_kl_t2']:.4f} dTheta={row['delta_l2_total']:.2f}", flush=True)

    rows.sort(key=lambda r: r["sweet_spot"], reverse=True)
    (args.out_dir / f"worker_{args.worker_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
