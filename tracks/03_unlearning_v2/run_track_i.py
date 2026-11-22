"""Round 1 3-Fold Weighted Rotating Selective Benchmark runner for Track 3 Piste I."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from auto_tune_4gpu import _build_train_proxy
from contracts import MethodConfig
from eval_harness import evaluate_candidate
from methods.track_i import TrackIConfig, run_track_i
from selective_benchmark import (
    FoldResult,
    SelectiveReport,
    aggregate_weighted_folds,
    evaluate_real_subset,
    make_altered_subset,
    make_rotating_class_folds,
    score_single_fold,
)
from task_template import NUM_CLASSES, build_model, load_model, predict


def _train_i_subset(
    orig: torch.nn.Module,
    f_sub: torch.Tensor,
    orig_f_sub: torch.Tensor,
    cfg: TrackIConfig,
    return_phase1: bool = False,
) -> torch.nn.Module | tuple[torch.nn.Module, torch.nn.Module]:
    """Train Piste I strictly on f_sub with zero leakage from held-out subsets."""
    p_x = _build_train_proxy(f_sub, seed=cfg.base.seed)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f_sub.argmax(dim=-1), minlength=NUM_CLASSES) == 0
    return run_track_i(orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg, return_phase1=return_phase1)


def _run_i_fold(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    labels: torch.Tensor,
    fold_id: int,
    indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: TrackIConfig,
) -> FoldResult:
    """Execute Piste I training and exact+altered evaluation on one class-rotation fold."""
    f_idx, r_in_idx, r_out_idx = indices
    fold_base = MethodConfig(**{**asdict(cfg.base), "seed": cfg.base.seed + fold_id})
    fold_cfg = TrackIConfig(**{**asdict(cfg), "base": fold_base})
    sub = _train_i_subset(orig, f_x[f_idx], orig_f[f_idx], fold_cfg)
    sub_all = predict(sub, f_x, batch_size=128)
    f_raw = evaluate_real_subset(sub_all[f_idx], orig_f[f_idx], labels[f_idx])
    r_in_raw = evaluate_real_subset(sub_all[r_in_idx], orig_f[r_in_idx], labels[r_in_idx])
    r_out_raw = evaluate_real_subset(sub_all[r_out_idx], orig_f[r_out_idx], labels[r_out_idx])
    f_alt = evaluate_real_subset(predict(sub, f_x_alt[f_idx], batch_size=128), orig_f_alt[f_idx], labels[f_idx])
    r_in_alt = evaluate_real_subset(predict(sub, f_x_alt[r_in_idx], batch_size=128), orig_f_alt[r_in_idx], labels[r_in_idx])
    gap_r = round(f_raw.ce_mean - r_in_raw.ce_mean, 4)
    gap_a = round(f_alt.ce_mean - r_in_alt.ce_mean, 4)
    f_score = score_single_fold(f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw)
    return FoldResult(fold_id, len(f_idx), f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw, gap_r, gap_a, f_score)


def _full_metrics_dict(orig: torch.nn.Module, sub: torch.nn.Module, f_x: torch.Tensor, ckpt: Path) -> dict[str, float]:
    torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, ckpt)
    ev = evaluate_candidate(orig, sub, f_x, sub_path=ckpt)
    return {
        "full_ce_mean": ev.forget_raw.ce_mean,
        "full_ce_std": ev.forget_raw.ce_std,
        "full_ce_p50": ev.forget_raw.ce_p50,
        "full_frac_gt_95": ev.forget_raw.frac_gt_95,
        "full_proxy_kl_t2": ev.retain_proxy.proxy_kl_t2,
        "full_absent_kl_t2": ev.retain_proxy.absent_classes_kl_t2,
        "full_delta_l2": ev.retain_proxy.delta_l2_total,
    }


def benchmark_track_i_candidate(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    cfg: TrackIConfig,
    out_dir: Path,
    num_folds: int = 3,
    fold_seed: int = 42,
) -> SelectiveReport:
    """Run K-fold weighted selective benchmark and save full 3,500-image checkpoint."""
    labels = orig_f.argmax(dim=-1)
    fold_indices = make_rotating_class_folds(labels, num_folds=num_folds, seed=fold_seed)
    folds = [_run_i_fold(orig, f_x, f_x_alt, orig_f, orig_f_alt, labels, k, idx_t, cfg) for k, idx_t in enumerate(fold_indices)]
    sub_full, p1_full = _train_i_subset(orig, f_x, orig_f, cfg, return_phase1=True)
    if cfg.heal_steps > 0:
        ev_p1 = evaluate_candidate(orig, p1_full, f_x)
        print(
            f"  [phase1] {cfg.base.name:24s} | CE={ev_p1.forget_raw.ce_mean:.3f}±{ev_p1.forget_raw.ce_std:.2f} "
            f"p50={ev_p1.forget_raw.ce_p50:.3f} >95%={ev_p1.forget_raw.frac_gt_95:.3f} ProxyKL={ev_p1.retain_proxy.proxy_kl_t2:.4f}",
            flush=True,
        )
    full_m = _full_metrics_dict(orig, sub_full, f_x, out_dir / f"{cfg.base.name}.pt")
    return aggregate_weighted_folds(cfg.base.name, folds, full_m)


def round1_candidates() -> list[TrackIConfig]:
    """4 Round-1 Erase-Then-Heal configurations exploring rank {4, 8}, freeze {6, 8}, ceiling {1.35..1.65}."""
    c1 = MethodConfig(
        name="I1_f8_r8_ce24_h150", strategy="track_i", steps=195, lr=1.4e-4, head_lr_mult=0.6,
        npo_beta=1.4, target_forget_ce=2.40, target_ce_upper=3.40, retain_kl_weight=1.5,
        l2_weight=8.0e-4, salun_keep_ratio=0.25, freeze_until_block=8,
    )
    c2 = MethodConfig(
        name="I2_f6_r8_ce25_h140", strategy="track_i", steps=210, lr=1.3e-4, head_lr_mult=0.5,
        npo_beta=1.35, target_forget_ce=2.50, target_ce_upper=3.50, retain_kl_weight=1.5,
        l2_weight=8.0e-4, salun_keep_ratio=0.25, freeze_until_block=6,
    )
    c3 = MethodConfig(
        name="I3_f8_r4_ce26_h165", strategy="track_i", steps=210, lr=1.5e-4, head_lr_mult=0.6,
        npo_beta=1.3, target_forget_ce=2.60, target_ce_upper=3.60, retain_kl_weight=1.4,
        l2_weight=6.0e-4, salun_keep_ratio=0.28, freeze_until_block=8,
    )
    c4 = MethodConfig(
        name="I4_f6_r4_ce28_h135", strategy="track_i", steps=210, lr=1.4e-4, head_lr_mult=0.5,
        npo_beta=1.25, target_forget_ce=2.80, target_ce_upper=3.80, retain_kl_weight=1.3,
        l2_weight=6.0e-4, salun_keep_ratio=0.28, freeze_until_block=6,
    )
    return [
        TrackIConfig(base=c1, heal_rank=8, heal_freeze_until_block=8, heal_steps=70, heal_lr=4.5e-4, heal_ce_ceiling=1.50, w_proxy_kl=2.5, w_dark_liaison=2.0, w_ceiling_heal=0.6),
        TrackIConfig(base=c2, heal_rank=8, heal_freeze_until_block=6, heal_steps=75, heal_lr=4.0e-4, heal_ce_ceiling=1.40, w_proxy_kl=2.8, w_dark_liaison=2.2, w_ceiling_heal=0.7),
        TrackIConfig(base=c3, heal_rank=4, heal_freeze_until_block=8, heal_steps=80, heal_lr=5.5e-4, heal_ce_ceiling=1.65, w_proxy_kl=2.5, w_dark_liaison=2.0, w_ceiling_heal=0.5),
        TrackIConfig(base=c4, heal_rank=4, heal_freeze_until_block=6, heal_steps=80, heal_lr=5.0e-4, heal_ce_ceiling=1.35, w_proxy_kl=3.0, w_dark_liaison=2.5, w_ceiling_heal=0.8),
    ]


def round2_candidates() -> list[TrackIConfig]:
    """4 Round-2 calibrated Erase-Then-Heal configs with floor guard + tight dTheta."""
    c1 = MethodConfig(
        name="I2R_f8_r8_ce19_fl085", strategy="track_i", steps=180, lr=1.15e-4, head_lr_mult=0.5,
        npo_beta=1.5, target_forget_ce=1.90, target_ce_upper=2.80, retain_kl_weight=2.0,
        l2_weight=1.8e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    c2 = MethodConfig(
        name="I2R_f8_r4_ce20_fl095", strategy="track_i", steps=180, lr=1.20e-4, head_lr_mult=0.5,
        npo_beta=1.5, target_forget_ce=2.00, target_ce_upper=2.90, retain_kl_weight=2.0,
        l2_weight=1.8e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    c3 = MethodConfig(
        name="I2R_f8_r4_ce16_fl075", strategy="track_i", steps=165, lr=1.10e-4, head_lr_mult=0.5,
        npo_beta=1.6, target_forget_ce=1.60, target_ce_upper=2.50, retain_kl_weight=2.2,
        l2_weight=2.0e-3, salun_keep_ratio=0.20, freeze_until_block=8,
    )
    c4 = MethodConfig(
        name="I2R_f6_r4_ce18_fl085", strategy="track_i", steps=165, lr=1.00e-4, head_lr_mult=0.45,
        npo_beta=1.55, target_forget_ce=1.80, target_ce_upper=2.70, retain_kl_weight=2.2,
        l2_weight=2.2e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    return [
        TrackIConfig(base=c1, gamma_lr_mult=1.0, phase1_target_f95=0.06, phase1_max_ce=3.2, heal_rank=8, heal_freeze_until_block=8, heal_steps=50, heal_lr=1.5e-4, heal_ce_floor=0.85, heal_ce_ceiling=1.40, w_proxy_kl=3.2, w_dark_liaison=1.2, w_ceiling_heal=0.4, w_floor_guard=1.6),
        TrackIConfig(base=c2, gamma_lr_mult=1.0, phase1_target_f95=0.05, phase1_max_ce=3.3, heal_rank=4, heal_freeze_until_block=8, heal_steps=55, heal_lr=1.8e-4, heal_ce_floor=0.95, heal_ce_ceiling=1.50, w_proxy_kl=3.2, w_dark_liaison=1.2, w_ceiling_heal=0.4, w_floor_guard=1.8),
        TrackIConfig(base=c3, gamma_lr_mult=1.0, phase1_target_f95=0.07, phase1_max_ce=2.9, heal_rank=4, heal_freeze_until_block=8, heal_steps=50, heal_lr=1.5e-4, heal_ce_floor=0.75, heal_ce_ceiling=1.30, w_proxy_kl=3.5, w_dark_liaison=1.0, w_ceiling_heal=0.4, w_floor_guard=1.6),
        TrackIConfig(base=c4, gamma_lr_mult=1.0, phase1_target_f95=0.06, phase1_max_ce=3.1, heal_rank=4, heal_freeze_until_block=6, heal_steps=50, heal_lr=1.4e-4, heal_ce_floor=0.85, heal_ce_ceiling=1.35, w_proxy_kl=3.5, w_dark_liaison=1.0, w_ceiling_heal=0.4, w_floor_guard=1.8),
    ]


def round3_candidates() -> list[TrackIConfig]:
    """4 Round-3 #32-fixed f6 1-factor-at-a-time ablations (static SalUn, unchoked head, merged L2)."""
    p1_kwargs = dict(
        strategy="track_i", steps=180, lr=1.0e-4, head_lr_mult=1.0, npo_beta=1.6,
        target_forget_ce=1.40, target_ce_upper=2.20, retain_kl_weight=2.4,
        l2_weight=2.0e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    i_kwargs = dict(
        alpha_proto=0.20, stuck_boost=1.5, gamma_lr_mult=1.0, saliency_refresh_frac=0.0,
        phase1_target_f95=0.06, phase1_max_ce=2.6, heal_freeze_until_block=6,
        heal_steps=45, heal_lr=1.2e-4, w_proxy_kl=3.5, w_dark_liaison=1.0, heal_l2_weight=1.5e-3,
    )
    c1 = MethodConfig(name="I3R_1_p1only_f6", **p1_kwargs)
    c2 = MethodConfig(name="I3R_2_lora_r4_c095_160", **p1_kwargs)
    c3 = MethodConfig(name="I3R_3_lora_r4_c105_170", **p1_kwargs)
    c4 = MethodConfig(name="I3R_4_lora_r8_c095_160", **p1_kwargs)
    return [
        TrackIConfig(base=c1, **{**i_kwargs, "heal_steps": 0}),
        TrackIConfig(base=c2, **{**i_kwargs, "heal_rank": 4, "heal_ce_floor": 0.95, "heal_ce_ceiling": 1.60, "w_ceiling_heal": 0.35, "w_floor_guard": 1.6}),
        TrackIConfig(base=c3, **{**i_kwargs, "heal_rank": 4, "heal_ce_floor": 1.05, "heal_ce_ceiling": 1.70, "w_ceiling_heal": 0.30, "w_floor_guard": 1.8}),
        TrackIConfig(base=c4, **{**i_kwargs, "heal_rank": 8, "heal_ce_floor": 0.95, "heal_ce_ceiling": 1.60, "w_ceiling_heal": 0.35, "w_floor_guard": 1.6}),
    ]


def round4_candidates() -> list[TrackIConfig]:
    """3 Round-4 #13-calibrated f6 configs (Phase 1 dTheta <= 4.30 + Phase 2 rank-8)."""
    p1_shared = dict(
        strategy="track_i", steps=150, lr=1.0e-4, head_lr_mult=1.0, npo_beta=1.6,
        retain_kl_weight=2.4, l2_weight=2.5e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    i_shared = dict(
        alpha_proto=0.20, stuck_boost=1.5, gamma_lr_mult=1.0, saliency_refresh_frac=0.0,
        phase1_probe_every=10, heal_freeze_until_block=6, heal_rank=8, heal_steps=50,
        heal_lr=1.2e-4, w_dark_liaison=1.0,
    )
    c1 = MethodConfig(name="I4R_1_r8_ce115_fl085_c150", target_forget_ce=1.15, target_ce_upper=1.95, **p1_shared)
    c2 = MethodConfig(name="I4R_2_r8_ce125_fl095_c160", target_forget_ce=1.25, target_ce_upper=2.05, **p1_shared)
    c3 = MethodConfig(name="I4R_3_r8_ce110_fl090_c155_hiKL", target_forget_ce=1.10, target_ce_upper=1.90, **p1_shared)
    return [
        TrackIConfig(base=c1, phase1_max_ce=2.05, phase1_target_f95=0.08, phase1_max_dtheta=4.30, heal_ce_floor=0.85, heal_ce_ceiling=1.50, w_proxy_kl=3.5, w_ceiling_heal=0.35, w_floor_guard=1.6, heal_l2_weight=2.0e-3, **i_shared),
        TrackIConfig(base=c2, phase1_max_ce=2.15, phase1_target_f95=0.07, phase1_max_dtheta=4.30, heal_ce_floor=0.95, heal_ce_ceiling=1.60, w_proxy_kl=3.5, w_ceiling_heal=0.30, w_floor_guard=1.8, heal_l2_weight=2.0e-3, **i_shared),
        TrackIConfig(base=c3, phase1_max_ce=2.00, phase1_target_f95=0.08, phase1_max_dtheta=4.15, heal_ce_floor=0.90, heal_ce_ceiling=1.55, w_proxy_kl=4.2, w_ceiling_heal=0.35, w_floor_guard=1.6, heal_l2_weight=2.5e-3, **i_shared),
    ]


def _eval_convex_soups(orig: torch.nn.Module, f_x: torch.Tensor, orig_f: torch.Tensor, out_dir: Path) -> None:
    """Run 1 full pass of I3R_4_lora_r8_c095_160 and evaluate 75/25 & 70/30 soups with orig."""
    i3r4 = _train_i_subset(orig, f_x, orig_f, round3_candidates()[3])
    orig_sd = {k: v.detach().cpu().float() for k, v in orig.state_dict().items()}
    i3r4_sd = {k: v.detach().cpu().float() for k, v in i3r4.state_dict().items()}
    dev = next(orig.parameters()).device
    soup_rows = []
    for alpha, name in ((0.75, "SOUP_75_I3R4_25_orig"), (0.70, "SOUP_70_I3R4_30_orig")):
        blended = {k: alpha * i3r4_sd[k] + (1.0 - alpha) * orig_sd[k] for k in orig_sd}
        ckpt = out_dir / f"{name}.pt"
        torch.save(blended, ckpt)
        m = build_model().to(dev)
        m.load_state_dict(blended, strict=True)
        ev = evaluate_candidate(orig, m.eval(), f_x, sub_path=ckpt)
        row = {"name": name, "alpha": alpha, "full_ce_mean": ev.forget_raw.ce_mean, "full_ce_std": ev.forget_raw.ce_std, "full_ce_p50": ev.forget_raw.ce_p50, "full_frac_gt_95": ev.forget_raw.frac_gt_95, "full_proxy_kl_t2": ev.retain_proxy.proxy_kl_t2, "full_absent_kl_t2": ev.retain_proxy.absent_classes_kl_t2, "full_delta_l2": ev.retain_proxy.delta_l2_total, "valid": bool(ev.checkpoint and ev.checkpoint.valid)}
        soup_rows.append(row)
        print(f"[soup] {name:24s} | Full(CE={row['full_ce_mean']:.3f}±{row['full_ce_std']:.2f} mu+s={row['full_ce_mean']+row['full_ce_std']:.2f} p50={row['full_ce_p50']:.3f} >95%={row['full_frac_gt_95']:.2f} ProxyKL={row['full_proxy_kl_t2']:.4f} AbsKL={row['full_absent_kl_t2']:.4f} dTheta={row['full_delta_l2']:.2f})", flush=True)
    (out_dir / "track_i_round4_soups.json").write_text(json.dumps(soup_rows, indent=2), encoding="utf-8")


def _log_report(rep: SelectiveReport) -> None:
    print(
        f"[track_i] {rep.name:24s} | W_SEL={rep.weighted_selective_score:.4f}±{rep.fold_score_std:.3f} | "
        f"3Fold(F_raw={rep.weighted_forget_ce_raw:.2f} F_alt={rep.weighted_forget_ce_alt:.2f} IntraAcc={rep.weighted_intra_retain_acc:.3f} Gap50={rep.weighted_intra_gap_50_50:+.2f} InterAcc={rep.weighted_inter_retain_acc:.3f} InterKL={rep.weighted_inter_retain_kl:.4f}) | "
        f"Full(CE={rep.full_ce_mean:.3f}±{rep.full_ce_std:.2f} p50={rep.full_ce_p50:.3f} >95%={rep.full_frac_gt_95:.2f} ProxyKL={rep.full_proxy_kl_t2:.4f} dTheta={rep.full_delta_l2:.2f})",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="3-Fold Weighted Selective Benchmark for Track 3 Piste I.")
    parser.add_argument("--round", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--fold-seed", type=int, default=None)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("output/track_i"))
    parser.add_argument("--json-name", type=str, default=None)
    args = parser.parse_args()

    fold_seed = args.fold_seed if args.fold_seed is not None else (777 if args.round in (3, 4) else 42)
    alt_seed = 123 if fold_seed == 42 else fold_seed + 100
    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    f_x_alt = make_altered_subset(f_x, seed=alt_seed)
    orig_f, orig_f_alt = predict(orig, f_x, batch_size=128), predict(orig, f_x_alt, batch_size=128)
    candidates = {1: round1_candidates, 2: round2_candidates, 3: round3_candidates, 4: round4_candidates}[args.round]()
    rows = []
    for cfg in candidates:
        rep = benchmark_track_i_candidate(orig, f_x, f_x_alt, orig_f, orig_f_alt, cfg, args.out_dir, args.num_folds, fold_seed)
        rows.append(asdict(rep))
        _log_report(rep)
    rows.sort(key=lambda r: r["weighted_selective_score"], reverse=True)
    (args.out_dir / (args.json_name or f"track_i_round{args.round}.json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.round == 4:
        _eval_convex_soups(orig, f_x, orig_f, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
