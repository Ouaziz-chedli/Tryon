"""Round 1+ 3-Fold Weighted Rotating Selective Benchmark runner for Track 3 Piste H."""

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import numpy as np
import torch

from auto_tune_4gpu import _build_train_proxy
from contracts import MethodConfig
from eval_harness import evaluate_candidate
from methods.track_h import TrackHConfig, run_track_h
from selective_benchmark import (
    FoldResult,
    SelectiveReport,
    aggregate_weighted_folds,
    evaluate_real_subset,
    make_altered_subset,
    make_rotating_class_folds,
    score_single_fold,
)
from task_template import DEVICE, NUM_CLASSES, build_model, load_model, predict


def _train_h_subset(
    orig: torch.nn.Module,
    f_sub: torch.Tensor,
    orig_f_sub: torch.Tensor,
    cfg: TrackHConfig,
) -> torch.nn.Module:
    """Train Piste H strictly on f_sub with zero leakage from held-out subsets."""
    p_x = _build_train_proxy(f_sub, seed=cfg.base.seed)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f_sub.argmax(dim=-1), minlength=NUM_CLASSES) == 0
    return run_track_h(orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg)


def _eval_fold_model(
    sub: torch.nn.Module,
    name: str,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    labels: torch.Tensor,
    fold_id: int,
    indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> FoldResult:
    """Evaluate an already-trained model on one class-rotation fold."""
    f_idx, r_in_idx, r_out_idx = indices
    sub_all = predict(sub, f_x, batch_size=128)
    f_raw = evaluate_real_subset(sub_all[f_idx], orig_f[f_idx], labels[f_idx])
    r_in_raw = evaluate_real_subset(sub_all[r_in_idx], orig_f[r_in_idx], labels[r_in_idx])
    r_out_raw = evaluate_real_subset(sub_all[r_out_idx], orig_f[r_out_idx], labels[r_out_idx])
    f_alt = evaluate_real_subset(predict(sub, f_x_alt[f_idx], batch_size=128), orig_f_alt[f_idx], labels[f_idx])
    r_in_alt = evaluate_real_subset(predict(sub, f_x_alt[r_in_idx], batch_size=128), orig_f_alt[r_in_idx], labels[r_in_idx])
    gap_r = round(f_raw.ce_mean - r_in_raw.ce_mean, 4)
    gap_a = round(f_alt.ce_mean - r_in_alt.ce_mean, 4)
    f_score = score_single_fold(f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw)
    print(f"  [fold {fold_id}] {name:22s} W={f_score:.4f} F={f_raw.ce_mean:.2f}/{f_alt.ce_mean:.2f} Intra={r_in_raw.top1_acc:.3f}", flush=True)
    return FoldResult(fold_id, len(f_idx), f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw, gap_r, gap_a, f_score)


def _run_h_fold(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    labels: torch.Tensor,
    fold_id: int,
    indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: TrackHConfig,
    save_fold_dir: Path | None = None,
) -> FoldResult:
    """Execute Piste H training and exact+altered evaluation on one class-rotation fold."""
    f_idx, _, _ = indices
    fold_base = MethodConfig(**{**asdict(cfg.base), "seed": cfg.base.seed + fold_id})
    fold_cfg = TrackHConfig(**{**asdict(cfg), "base": fold_base})
    sub = _train_h_subset(orig, f_x[f_idx], orig_f[f_idx], fold_cfg)
    if save_fold_dir is not None:
        torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, save_fold_dir / f"{cfg.base.name}_fold{fold_id}.pt")
    return _eval_fold_model(sub, cfg.base.name, f_x, f_x_alt, orig_f, orig_f_alt, labels, fold_id, indices)


def benchmark_track_h_candidate(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    cfg: TrackHConfig,
    out_dir: Path,
    num_folds: int = 3,
    eval_seed: int = 42,
    save_fold_dir: Path | None = None,
) -> SelectiveReport:
    """Run K-fold weighted selective benchmark and save full 3,500-image checkpoint."""
    labels = orig_f.argmax(dim=-1)
    fold_indices = make_rotating_class_folds(labels, num_folds=num_folds, seed=eval_seed)
    folds = [
        _run_h_fold(orig, f_x, f_x_alt, orig_f, orig_f_alt, labels, k, idx_tuple, cfg, save_fold_dir=save_fold_dir)
        for k, idx_tuple in enumerate(fold_indices)
    ]
    sub_full = _train_h_subset(orig, f_x, orig_f, cfg)
    ckpt = out_dir / f"{cfg.base.name}.pt"
    torch.save({k: v.detach().cpu().float() for k, v in sub_full.state_dict().items()}, ckpt)
    ev = evaluate_candidate(orig, sub_full, f_x, sub_path=ckpt)
    full_m = {
        "full_ce_mean": ev.forget_raw.ce_mean,
        "full_ce_std": ev.forget_raw.ce_std,
        "full_ce_p50": ev.forget_raw.ce_p50,
        "full_frac_gt_95": ev.forget_raw.frac_gt_95,
        "full_proxy_kl_t2": ev.retain_proxy.proxy_kl_t2,
        "full_absent_kl_t2": ev.retain_proxy.absent_classes_kl_t2,
        "full_delta_l2": ev.retain_proxy.delta_l2_total,
    }
    return aggregate_weighted_folds(cfg.base.name, folds, full_m)


def round1_candidates() -> list[TrackHConfig]:
    """Round 1 Piste H candidates specified by @h-lead."""
    c1 = MethodConfig(
        name="H1_h1h2_f8_ce125", strategy="track_h", steps=210, lr=1.1e-4, head_lr_mult=0.6,
        npo_beta=1.6, target_forget_ce=1.25, target_ce_upper=2.05, retain_kl_weight=2.2,
        l2_weight=1.5e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    c2 = MethodConfig(
        name="H2_h123_f8_ce145", strategy="track_h", steps=225, lr=1.2e-4, head_lr_mult=0.5,
        npo_beta=1.5, target_forget_ce=1.45, target_ce_upper=2.25, retain_kl_weight=2.2,
        l2_weight=1.5e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    c3 = MethodConfig(
        name="H3_h1234_circuit_f8", strategy="track_h", steps=210, lr=1.1e-4, head_lr_mult=0.6,
        npo_beta=1.6, target_forget_ce=1.25, target_ce_upper=2.05, retain_kl_weight=2.2,
        l2_weight=1.5e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    c4 = MethodConfig(
        name="H4_h123_f6_ce125", strategy="track_h", steps=210, lr=1.0e-4, head_lr_mult=0.5,
        npo_beta=1.6, target_forget_ce=1.25, target_ce_upper=2.05, retain_kl_weight=2.2,
        l2_weight=1.5e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    return [
        TrackHConfig(base=c1, alpha_proto=0.75, stuck_boost=1.0, gamma_lr_mult=1.0, use_if_then=False),
        TrackHConfig(base=c2, alpha_proto=0.80, stuck_boost=1.2, gamma_lr_mult=3.0, use_if_then=False),
        TrackHConfig(base=c3, alpha_proto=0.75, stuck_boost=1.0, gamma_lr_mult=3.0, use_if_then=True, if_then_gates=64, if_then_gain=0.8, if_then_ret_q=0.995),
        TrackHConfig(base=c4, alpha_proto=0.80, stuck_boost=1.0, gamma_lr_mult=3.0, use_if_then=False),
    ]


def round2_candidates() -> list[TrackHConfig]:
    """Server-calibrated Round 2 Piste H candidates specified by @h-lead (#24)."""
    c1 = MethodConfig(
        name="R2_h12_f8_ce090", strategy="track_h", steps=225, lr=1.05e-4, head_lr_mult=0.5,
        npo_beta=1.8, target_forget_ce=0.90, target_ce_upper=1.50, retain_kl_weight=2.4,
        l2_weight=1.8e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    c2 = MethodConfig(
        name="R2_h12_f8_ce105", strategy="track_h", steps=225, lr=1.1e-4, head_lr_mult=0.5,
        npo_beta=1.7, target_forget_ce=1.05, target_ce_upper=1.70, retain_kl_weight=2.4,
        l2_weight=1.8e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    c3 = MethodConfig(
        name="R2_h12_f6_ce095", strategy="track_h", steps=225, lr=9.5e-5, head_lr_mult=0.45,
        npo_beta=1.8, target_forget_ce=0.95, target_ce_upper=1.55, retain_kl_weight=2.6,
        l2_weight=2.2e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    c4 = MethodConfig(
        name="R2_h124_f8_circuit", strategy="track_h", steps=225, lr=1.05e-4, head_lr_mult=0.5,
        npo_beta=1.8, target_forget_ce=0.90, target_ce_upper=1.50, retain_kl_weight=2.4,
        l2_weight=1.8e-3, salun_keep_ratio=0.22, freeze_until_block=8,
    )
    return [
        TrackHConfig(base=c1, alpha_proto=0.80, stuck_boost=2.0, stuck_t4_weight=0.50, gamma_lr_mult=1.0, use_if_then=False),
        TrackHConfig(base=c2, alpha_proto=0.80, stuck_boost=2.0, stuck_t4_weight=0.50, gamma_lr_mult=1.0, use_if_then=False),
        TrackHConfig(base=c3, alpha_proto=0.82, stuck_boost=2.0, stuck_t4_weight=0.45, gamma_lr_mult=1.0, use_if_then=False),
        TrackHConfig(base=c4, alpha_proto=0.80, stuck_boost=2.0, stuck_t4_weight=0.50, gamma_lr_mult=1.0, use_if_then=True, if_then_gates=48, if_then_block=11, if_then_gain=0.35, if_then_ret_q=0.999),
    ]


def round3_candidates() -> list[TrackHConfig]:
    """Server-calibrated Round 3 Piste H candidates (#32/#35 root-cause fixes)."""
    c1 = MethodConfig(
        name="R3_f6_ce090_ap20", strategy="track_h", steps=210, lr=1.0e-4, head_lr_mult=1.0,
        npo_beta=1.8, target_forget_ce=0.90, target_ce_upper=1.55, retain_kl_weight=2.4,
        l2_weight=1.8e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    c2 = MethodConfig(
        name="R3_f6_ce100_ap15", strategy="track_h", steps=210, lr=1.0e-4, head_lr_mult=1.0,
        npo_beta=1.8, target_forget_ce=1.00, target_ce_upper=1.70, retain_kl_weight=2.4,
        l2_weight=1.8e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    c3 = MethodConfig(
        name="R3_f6_ce085_ap30_safe", strategy="track_h", steps=210, lr=9.5e-5, head_lr_mult=1.0,
        npo_beta=1.8, target_forget_ce=0.85, target_ce_upper=1.45, retain_kl_weight=2.6,
        l2_weight=2.2e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    c4 = MethodConfig(
        name="R3_f6_ce090_ap20_h4", strategy="track_h", steps=210, lr=1.0e-4, head_lr_mult=1.0,
        npo_beta=1.8, target_forget_ce=0.90, target_ce_upper=1.55, retain_kl_weight=2.4,
        l2_weight=1.8e-3, salun_keep_ratio=0.20, freeze_until_block=6,
    )
    return [
        TrackHConfig(base=c1, alpha_proto=0.20, stuck_boost=1.8, stuck_t4_weight=0.45, rollback_tau=0.20, rollback_rho=0.20, gamma_lr_mult=1.0, use_if_then=False),
        TrackHConfig(base=c2, alpha_proto=0.15, stuck_boost=2.0, stuck_t4_weight=0.50, rollback_tau=0.20, rollback_rho=0.20, gamma_lr_mult=1.0, use_if_then=False),
        TrackHConfig(base=c3, alpha_proto=0.30, stuck_boost=1.8, stuck_t4_weight=0.45, rollback_tau=0.25, rollback_rho=0.25, gamma_lr_mult=1.0, use_if_then=False),
        TrackHConfig(base=c4, alpha_proto=0.20, stuck_boost=1.8, stuck_t4_weight=0.45, rollback_tau=0.20, rollback_rho=0.20, gamma_lr_mult=1.0, use_if_then=True, if_then_gates=32, if_then_block=11, if_then_gain=0.25, if_then_ret_q=0.999),
    ]


def round4_candidates() -> list[TrackHConfig]:
    """Pure 1-factor-at-a-time ablations vs #9 (AT_ce1.00_b1.8_f6) on held-out seed=777 (#51)."""
    def _base(name: str) -> MethodConfig:
        return MethodConfig(
            name=name, strategy="track_h", steps=210, lr=1.0e-4, head_lr_mult=1.0,
            npo_beta=1.8, target_forget_ce=1.00, target_ce_upper=1.80, retain_kl_weight=2.2,
            l2_weight=1.5e-3, salun_keep_ratio=0.20, freeze_until_block=6, seed=42,
        )
    return [
        TrackHConfig(base=_base("ABL_0_baseline_AT"), alpha_proto=0.0, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=False, use_h2_damp=False, use_h3_gamma=False, use_if_then=False),
        TrackHConfig(base=_base("ABL_1_plus_H1_ortho"), alpha_proto=0.25, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=False, use_h2_damp=False, use_h3_gamma=False, use_if_then=False),
        TrackHConfig(base=_base("ABL_2_plus_H1_gov"), alpha_proto=0.0, stuck_boost=1.5, stuck_t4_weight=0.0, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=True, use_h2_damp=False, use_h3_gamma=False, use_if_then=False),
        TrackHConfig(base=_base("ABL_3_plus_H2_damp"), alpha_proto=0.0, conductance_period=15, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=False, use_h2_damp=True, use_h3_gamma=False, use_if_then=False),
    ]


def round5_candidates() -> list[TrackHConfig]:
    """Focused 2-run Round 5 on held-out seed=777 with clamp_min(1.0) fix (#61)."""
    def _base(name: str) -> MethodConfig:
        return MethodConfig(
            name=name, strategy="track_h", steps=210, lr=1.0e-4, head_lr_mult=1.0,
            npo_beta=1.8, target_forget_ce=1.00, target_ce_upper=1.80, retain_kl_weight=2.2,
            l2_weight=1.5e-3, salun_keep_ratio=0.20, freeze_until_block=6, seed=42,
        )
    return [
        TrackHConfig(base=_base("R5_h1gov_fixed"), alpha_proto=0.0, stuck_boost=1.5, stuck_t4_weight=0.0, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=True, use_h2_damp=False, use_h3_gamma=False, use_if_then=False),
        TrackHConfig(base=_base("R5_h1gov_ortho25"), alpha_proto=0.25, stuck_boost=1.5, stuck_t4_weight=0.0, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=True, use_h2_damp=False, use_h3_gamma=False, use_if_then=False),
    ]


def round6_candidates() -> list[TrackHConfig]:
    """Focused 2-run Round 6 on held-out seed=777 targeting server sweet spot (#65)."""
    def _base(name: str) -> MethodConfig:
        return MethodConfig(
            name=name, strategy="track_h", steps=240, lr=1.1e-4, head_lr_mult=1.0,
            npo_beta=1.8, target_forget_ce=1.25, target_ce_upper=2.00, retain_kl_weight=2.2,
            l2_weight=1.5e-3, salun_keep_ratio=0.20, freeze_until_block=6, seed=42,
        )
    return [
        TrackHConfig(base=_base("R6_h1gov_ce125"), alpha_proto=0.0, stuck_boost=1.5, stuck_t4_weight=0.0, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=True, use_h2_damp=False, use_h3_gamma=False, use_if_then=False),
        TrackHConfig(base=_base("R6_h1gov_ortho25_ce125"), alpha_proto=0.25, stuck_boost=1.5, stuck_t4_weight=0.0, rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=True, use_h2_damp=False, use_h3_gamma=False, use_if_then=False),
    ]


def round7_candidates() -> list[TrackHConfig]:
    """Focused 3-run Round 7 on held-out seed=777 targeting server sweet spot (#90)."""
    def _cfg(name: str, wnf: float, ce: float, upper: float, beta: float, lr: float, steps: int, kl: float, l2: float) -> TrackHConfig:
        base = MethodConfig(
            name=name, strategy="track_h", steps=steps, lr=lr, head_lr_mult=1.0,
            npo_beta=beta, target_forget_ce=ce, target_ce_upper=upper, retain_kl_weight=kl,
            l2_weight=l2, salun_keep_ratio=0.20, freeze_until_block=6, seed=42,
        )
        return TrackHConfig(
            base=base, alpha_proto=0.25, stuck_boost=1.5, stuck_t4_weight=0.0, weight_norm_floor=wnf,
            rollback_rho=0.0, gamma_lr_mult=1.0, use_h1_gov=True, use_h2_damp=False, use_h3_gamma=False, use_if_then=False,
        )
    return [
        _cfg("R7_1_wnf060_ce105", wnf=0.60, ce=1.05, upper=1.80, beta=1.80, lr=1.05e-4, steps=225, kl=2.2, l2=1.5e-3),
        _cfg("R7_2_wnf055_ce110_kl19", wnf=0.55, ce=1.10, upper=1.85, beta=1.75, lr=1.10e-4, steps=240, kl=1.9, l2=1.4e-3),
        _cfg("R7_3_wnf100_kl17_ce120", wnf=1.00, ce=1.20, upper=1.95, beta=1.75, lr=1.15e-4, steps=240, kl=1.7, l2=1.2e-3),
    ]


def _blend_two_ckpts(path_a: Path, path_b: Path, alpha: float = 0.5) -> dict[str, torch.Tensor]:
    sd_a = torch.load(path_a, map_location="cpu", weights_only=True)
    sd_b = torch.load(path_b, map_location="cpu", weights_only=True)
    return {k: (alpha * sd_a[k].float() + (1.0 - alpha) * sd_b[k].float()) for k in sd_a}


def _eval_round7_soup(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    out_dir: Path,
    fold_dir: Path,
    num_folds: int,
    eval_seed: int,
) -> SelectiveReport:
    """Evaluate 50/50 convex soup of R7_2 and R7_3 on fold models and full 3,500-image model."""
    name, name_a, name_b = "R7_SOUP_50_r72_50_r73", "R7_2_wnf055_ce110_kl19", "R7_3_wnf100_kl17_ce120"
    labels = orig_f.argmax(dim=-1)
    fold_indices = make_rotating_class_folds(labels, num_folds=num_folds, seed=eval_seed)
    model = build_model().to(DEVICE).eval()
    folds = []
    for k, idx_tuple in enumerate(fold_indices):
        model.load_state_dict(_blend_two_ckpts(fold_dir / f"{name_a}_fold{k}.pt", fold_dir / f"{name_b}_fold{k}.pt"), strict=True)
        folds.append(_eval_fold_model(model, name, f_x, f_x_alt, orig_f, orig_f_alt, labels, k, idx_tuple))
    full_sd = _blend_two_ckpts(out_dir / f"{name_a}.pt", out_dir / f"{name_b}.pt")
    ckpt = out_dir / f"{name}.pt"
    torch.save(full_sd, ckpt)
    model.load_state_dict(full_sd, strict=True)
    ev = evaluate_candidate(orig, model, f_x, sub_path=ckpt)
    full_m = {
        "full_ce_mean": ev.forget_raw.ce_mean, "full_ce_std": ev.forget_raw.ce_std,
        "full_ce_p50": ev.forget_raw.ce_p50, "full_frac_gt_95": ev.forget_raw.frac_gt_95,
        "full_proxy_kl_t2": ev.retain_proxy.proxy_kl_t2, "full_absent_kl_t2": ev.retain_proxy.absent_classes_kl_t2,
        "full_delta_l2": ev.retain_proxy.delta_l2_total,
    }
    return aggregate_weighted_folds(name, folds, full_m)


def _print_report_line(rep: SelectiveReport) -> None:
    mu_sig = rep.full_ce_mean + rep.full_ce_std
    print(
        f"[track_h] {rep.name:24s} | W_SEL={rep.weighted_selective_score:.4f}±{rep.fold_score_std:.3f} | "
        f"3Fold(F_raw={rep.weighted_forget_ce_raw:.2f} F_alt={rep.weighted_forget_ce_alt:.2f} IntraAcc={rep.weighted_intra_retain_acc:.3f} Gap50={rep.weighted_intra_gap_50_50:+.2f} InterAcc={rep.weighted_inter_retain_acc:.3f} InterKL={rep.weighted_inter_retain_kl:.4f}) | "
        f"Full(CE={rep.full_ce_mean:.3f}±{rep.full_ce_std:.2f} mu+sig={mu_sig:.2f} p50={rep.full_ce_p50:.3f} >95%={rep.full_frac_gt_95:.2f} ProxyKL={rep.full_proxy_kl_t2:.4f} AbsKL={rep.full_absent_kl_t2:.4f} dTheta={rep.full_delta_l2:.2f})",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="3-Fold Weighted Selective Benchmark for Track 3 Piste H.")
    parser.add_argument("--round", type=int, choices=[1, 2, 3, 4, 5, 6, 7], default=7)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("output/track_h"))
    parser.add_argument("--json-name", type=str, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_seed = args.eval_seed if args.eval_seed is not None else (777 if args.round >= 4 else 42)
    json_name = args.json_name or f"track_h_round{args.round}.json"
    cands_map = {1: round1_candidates, 2: round2_candidates, 3: round3_candidates, 4: round4_candidates, 5: round5_candidates, 6: round6_candidates, 7: round7_candidates}
    candidates = cands_map[args.round]()
    fold_dir = (args.out_dir / "_tmp_folds") if args.round == 7 else None
    if fold_dir is not None:
        fold_dir.mkdir(parents=True, exist_ok=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    f_x_alt = make_altered_subset(f_x, seed=eval_seed + 100)
    orig_f = predict(orig, f_x, batch_size=128)
    orig_f_alt = predict(orig, f_x_alt, batch_size=128)
    rows = []
    for cfg in candidates:
        rep = benchmark_track_h_candidate(orig, f_x, f_x_alt, orig_f, orig_f_alt, cfg, args.out_dir, args.num_folds, eval_seed=eval_seed, save_fold_dir=fold_dir)
        rows.append(asdict(rep))
        _print_report_line(rep)
        (args.out_dir / json_name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.round == 7 and fold_dir is not None:
        soup_rep = _eval_round7_soup(orig, f_x, f_x_alt, orig_f, orig_f_alt, args.out_dir, fold_dir, args.num_folds, eval_seed)
        rows.append(asdict(soup_rep))
        _print_report_line(soup_rep)
    rows.sort(key=lambda r: r["weighted_selective_score"], reverse=True)
    (args.out_dir / json_name).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
