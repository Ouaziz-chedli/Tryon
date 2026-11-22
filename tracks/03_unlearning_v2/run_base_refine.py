"""Round 1+ surgical refinement of Track 3 initial AT (SalUn+NPO) + weight soups."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import torch

from auto_tune_4gpu import _adaptive_step, _build_train_proxy, _probe_step, _sample_probes, run_adaptive_salun_npo
from contracts import MethodConfig
from eval_harness import evaluate_candidate
from methods.common import build_optimizer, prepare_student
from methods.salun_npo import compute_tempered_saliency
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


@dataclass(frozen=True)
class TightLimits:
    """Probe controller thresholds and snapshot gate for run_tight_at."""

    pkl_limit: float = 0.032
    d_l2_limit: float = 4.2
    pen_pkl_thr: float = 0.030
    pen_dl2_thr: float = 4.1
    accept_pkl_max: float = 0.042


MID_LIMITS = TightLimits(0.039, 4.35, 0.036, 4.20, 0.048)


def blend_state_dicts(
    models: list[tuple[float, dict[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    """Compute convex combination sum(w_i * sd_i) in float32."""
    if not models:
        raise ValueError("models must be non-empty")
    total_w = sum(w for w, _ in models)
    if abs(total_w - 1.0) > 1e-5:
        raise ValueError(f"weights must sum to 1.0, got {total_w:.6f}")
    out: dict[str, torch.Tensor] = {}
    for k in models[0][1]:
        acc = torch.zeros_like(models[0][1][k], dtype=torch.float32)
        for w, sd in models:
            acc.add_(sd[k].float(), alpha=w)
        out[k] = acc.to(torch.float32)
    return out


def _tight_snapshot_score(
    ce_mean: float, ce_std: float, p50: float, f95: float, pkl: float, d_l2: float,
    target_p50: float = 0.50, limits: TightLimits = TightLimits(),
) -> float:
    """Score intermediate snapshot with configurable ProxyKL/dTheta penalties."""
    tgt, tgt_std = max(target_p50, 0.45), max(1.35, 1.35 * max(target_p50, 0.45))
    d_sub = abs(p50 - tgt) / tgt + 0.20 * abs(ce_std - tgt_std) / tgt_std + 0.6 * max(0.0, f95 - 0.25)
    f_score = 1.0 - min(1.0, d_sub / 1.25)
    drift_pen = 25.0 * max(0.0, pkl - limits.pen_pkl_thr) + 1.0 * max(0.0, d_l2 - limits.pen_dl2_thr)
    return f_score - drift_pen


def _tight_adjust_controller(
    opt: torch.optim.Optimizer, kl_w: float, l2_w: float, pkl: float, d_l2: float, p50: float,
    limits: TightLimits = TightLimits(),
) -> tuple[float, float]:
    """Configurable PI-style controller braking above limits.pkl_limit or limits.d_l2_limit."""
    if pkl > limits.pkl_limit or d_l2 > limits.d_l2_limit:
        kl_w = min(10.0, kl_w * 1.35)
        l2_w = min(0.05, l2_w * 1.5)
        for g in opt.param_groups:
            g["lr"] = max(2e-5, g["lr"] * 0.85)
    elif pkl < 0.020 and d_l2 < 3.0 and p50 < 0.25:
        kl_w = max(1.0, kl_w * 0.85)
    return kl_w, l2_w


def run_tight_at(
    base_model: torch.nn.Module, orig_model: torch.nn.Module, f_x: torch.Tensor,
    orig_f: torch.Tensor, p_x: torch.Tensor, orig_p: torch.Tensor,
    absent_mask: torch.Tensor, cfg: MethodConfig, limits: TightLimits = TightLimits(),
) -> torch.nn.Module:
    """Closed-loop SalUn+NPO with configurable ProxyKL/dTheta controller and snapshot gate."""
    torch.manual_seed(cfg.seed)
    student = prepare_student(base_model, cfg.freeze_until_block)
    pseudo = orig_f.argmax(dim=-1)
    masks = compute_tempered_saliency(student, f_x, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    opt = build_optimizer(student, cfg.lr, cfg.head_lr_mult)
    kl_w, l2_w, best_score, best_sd = cfg.retain_kl_weight, cfg.l2_weight, -1e9, None
    f_probe, pseudo_probe, p_probe, orig_p_probe = _sample_probes(f_x, pseudo, p_x, orig_p, seed=cfg.seed)
    upper_p50 = max(cfg.target_ce_upper, cfg.target_forget_ce * 1.45)
    for step in range(1, cfg.steps + 1):
        _adaptive_step(student, orig_model, opt, f_x, orig_f, pseudo, p_x, orig_p, orig_params, masks, absent_mask, cfg, kl_w, l2_w)
        if step % 15 != 0:
            continue
        ce_m, ce_s, p50, f95, pkl, d_l2 = _probe_step(student, f_probe, pseudo_probe, p_probe, orig_p_probe, orig_params)
        kl_w, l2_w = _tight_adjust_controller(opt, kl_w, l2_w, pkl, d_l2, p50, limits)
        snap = _tight_snapshot_score(ce_m, ce_s, p50, f95, pkl, d_l2, cfg.target_forget_ce, limits)
        if 0.18 <= p50 <= upper_p50 + 0.25 and pkl <= limits.accept_pkl_max and snap > best_score:
            best_score, best_sd = snap, {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if p50 > upper_p50 + 0.25 and best_sd is not None:
            break
    if best_sd is not None:
        student.load_state_dict(best_sd)
    return student.eval()


def _at_cfg(
    name: str, lr: float, ce_tgt: float, ce_up: float, beta: float,
    kl_w: float, keep: float = 0.20, l2_w: float = 2e-3, aug_mode: str = "moderate", seed: int = 42,
) -> MethodConfig:
    """Construct a standard AT (freeze_until_block=6, head_lr_mult=1.0) config."""
    return MethodConfig(
        name=name, strategy="salun_npo", loss_kind="npo", steps=210, batch_size=64,
        aug_mode=aug_mode, freeze_until_block=6, head_lr_mult=1.0, seed=seed,
        lr=lr, target_forget_ce=ce_tgt, target_ce_upper=ce_up, npo_beta=beta,
        retain_kl_weight=kl_w, salun_keep_ratio=keep, l2_weight=l2_w,
    )


def build_round1_configs() -> list[MethodConfig]:
    """Return Round 1's 6 fine-grid peak AT configs + 2 server-calibrated anchors."""
    return [
        _at_cfg("AT_ce0.70_b2.0_k20_kl25", 9e-5, 0.70, 1.05, 2.0, 2.5, 0.20),
        _at_cfg("AT_ce0.80_b1.9_k20_kl25", 9e-5, 0.80, 1.20, 1.9, 2.5, 0.20),
        _at_cfg("AT_ce0.80_b1.9_k20_kl28", 9e-5, 0.80, 1.20, 1.9, 2.8, 0.20),
        _at_cfg("AT_ce0.80_b1.9_k15_kl28", 9e-5, 0.80, 1.20, 1.9, 2.8, 0.15),
        _at_cfg("AT_ce0.90_b1.8_k20_kl28", 9e-5, 0.90, 1.35, 1.8, 2.8, 0.20),
        _at_cfg("AT_ce0.90_b1.8_k15_kl25", 9.5e-5, 0.90, 1.35, 1.8, 2.5, 0.15),
        _at_cfg("ANCHOR_ce0.50_b2.0_f6", 8e-5, 0.50, 0.65, 2.0, 2.5, 0.20, 2e-3),
        _at_cfg("ANCHOR_ce1.00_b1.8_f6", 1e-4, 1.00, 1.80, 1.8, 2.2, 0.20, 1.5e-3),
    ]


def build_soup_specs() -> list[tuple[str, list[tuple[float, str]]]]:
    """Return Round 1's 6 zero-cost convex weight soup specifications."""
    return [
        ("SOUP_40_050_60_100", [(0.40, "ANCHOR_ce0.50_b2.0_f6"), (0.60, "ANCHOR_ce1.00_b1.8_f6")]),
        ("SOUP_50_050_50_100", [(0.50, "ANCHOR_ce0.50_b2.0_f6"), (0.50, "ANCHOR_ce1.00_b1.8_f6")]),
        ("SOUP_30_050_70_100", [(0.30, "ANCHOR_ce0.50_b2.0_f6"), (0.70, "ANCHOR_ce1.00_b1.8_f6")]),
        ("SOUP_50_ce70_50_ce90_kl28", [(0.50, "AT_ce0.70_b2.0_k20_kl25"), (0.50, "AT_ce0.90_b1.8_k20_kl28")]),
        ("SOUP_50_ce80kl28_50_ce90kl28", [(0.50, "AT_ce0.80_b1.9_k20_kl28"), (0.50, "AT_ce0.90_b1.8_k20_kl28")]),
        ("SOUP_top3_kl28_avg", [
            (1.0 / 3.0, "AT_ce0.70_b2.0_k20_kl25"),
            (1.0 / 3.0, "AT_ce0.80_b1.9_k20_kl28"),
            (1.0 / 3.0, "AT_ce0.90_b1.8_k20_kl28"),
        ]),
    ]


def build_round2_configs() -> list[tuple[str, MethodConfig]]:
    """Return Round 2's 6 tight-controller AT2 configs + 2 reference anchors."""
    return [
        ("tight", _at_cfg("AT2_ce0.75_b1.9_kl32_l24e3", 9e-5, 0.75, 1.15, 1.9, 3.2, 0.20, 4e-3)),
        ("tight", _at_cfg("AT2_ce0.85_b1.8_kl32_l24e3", 9e-5, 0.85, 1.30, 1.8, 3.2, 0.20, 4e-3)),
        ("tight", _at_cfg("AT2_ce0.85_b1.8_kl38_l26e3", 9e-5, 0.85, 1.30, 1.8, 3.8, 0.20, 6e-3)),
        ("tight", _at_cfg("AT2_ce0.95_b1.8_kl35_l25e3", 9e-5, 0.95, 1.45, 1.8, 3.5, 0.20, 5e-3)),
        ("tight", _at_cfg("AT2_ce0.85_b1.9_kl35_l25e3", 9e-5, 0.85, 1.30, 1.9, 3.5, 0.20, 5e-3)),
        ("tight", _at_cfg("AT2_ce0.85_b1.9_kl35_l25e3_strong", 9e-5, 0.85, 1.30, 1.9, 3.5, 0.20, 5e-3, "strong")),
        ("adaptive", _at_cfg("ANCHOR_ce1.00_b1.8_f6", 1e-4, 1.00, 1.80, 1.8, 2.2, 0.20, 1.5e-3)),
        ("adaptive", _at_cfg("AT_ce0.80_b1.9_k20_kl28", 9e-5, 0.80, 1.20, 1.9, 2.8, 0.20, 2e-3)),
    ]


def build_round2_soup_specs() -> list[tuple[str, list[tuple[float, str]]]]:
    """Return Round 2's 6 weight & ORIG-shrinkage soup specifications."""
    return [
        ("SHRINK_85_AT9", [(0.85, "ANCHOR_ce1.00_b1.8_f6"), (0.15, "ORIG")]),
        ("SHRINK_88_AT9", [(0.88, "ANCHOR_ce1.00_b1.8_f6"), (0.12, "ORIG")]),
        ("SHRINK_85_R1best", [(0.85, "AT_ce0.80_b1.9_k20_kl28"), (0.15, "ORIG")]),
        ("SOUP_50_AT2ce75_50_AT2ce95", [(0.50, "AT2_ce0.75_b1.9_kl32_l24e3"), (0.50, "AT2_ce0.95_b1.8_kl35_l25e3")]),
        ("SOUP_90_AT2mix_10_ORIG", [(0.45, "AT2_ce0.85_b1.8_kl32_l24e3"), (0.45, "AT2_ce0.95_b1.8_kl35_l25e3"), (0.10, "ORIG")]),
        ("SOUP_45_AT2kl38_45_AT9_10_ORIG", [(0.45, "AT2_ce0.85_b1.8_kl38_l26e3"), (0.45, "ANCHOR_ce1.00_b1.8_f6"), (0.10, "ORIG")]),
    ]


def build_round3_configs() -> list[tuple[str, MethodConfig]]:
    """Return Round 3's 4 mid-bracket multi-seed AT3 configs + 2 reference anchors."""
    return [
        ("tight_mid", _at_cfg("AT3_s101_ce0.90_b1.8_kl28", 9.5e-5, 0.90, 1.40, 1.8, 2.8, 0.20, 2.8e-3, seed=101)),
        ("tight_mid", _at_cfg("AT3_s202_ce0.95_b1.8_kl26", 9.5e-5, 0.95, 1.50, 1.8, 2.6, 0.20, 2.5e-3, seed=202)),
        ("tight_mid", _at_cfg("AT3_s303_ce0.85_b1.9_kl30", 9.5e-5, 0.85, 1.35, 1.9, 3.0, 0.20, 3.0e-3, seed=303)),
        ("tight_mid", _at_cfg("AT3_s42_ce0.90_b1.8_kl28", 9.5e-5, 0.90, 1.40, 1.8, 2.8, 0.20, 2.8e-3, seed=42)),
        ("adaptive", _at_cfg("ANCHOR_ce1.00_b1.8_f6", 1e-4, 1.00, 1.80, 1.8, 2.2, 0.20, 1.5e-3, seed=42)),
        ("tight", _at_cfg("AT2_ce0.85_b1.8_kl32_l24e3", 9e-5, 0.85, 1.30, 1.8, 3.2, 0.20, 4e-3, seed=42)),
    ]


def build_round3_soup_specs() -> list[tuple[str, list[tuple[float, str]]]]:
    """Return Round 3's 8 multi-seed & mid-bracket convex soups."""
    s1, s2, s3 = "AT3_s101_ce0.90_b1.8_kl28", "AT3_s202_ce0.95_b1.8_kl26", "AT3_s303_ce0.85_b1.9_kl30"
    at9, at2 = "ANCHOR_ce1.00_b1.8_f6", "AT2_ce0.85_b1.8_kl32_l24e3"
    return [
        ("SOUP3_multiseed_3way", [(1.0 / 3.0, s1), (1.0 / 3.0, s2), (1.0 / 3.0, s3)]),
        ("SOUP3_4way_with_AT9", [(0.25, s1), (0.25, s2), (0.25, s3), (0.25, at9)]),
        ("SOUP3_95_multiseed_05_ORIG", [(0.95 / 3.0, s1), (0.95 / 3.0, s2), (0.95 / 3.0, s3), (0.05, "ORIG")]),
        ("SOUP3_50_AT9_50_s101", [(0.50, at9), (0.50, s1)]),
        ("SOUP3_65_AT9_35_AT2", [(0.65, at9), (0.35, at2)]),
        ("SOUP3_75_AT9_25_AT2", [(0.75, at9), (0.25, at2)]),
        ("SHRINK_93_AT9", [(0.93, at9), (0.07, "ORIG")]),
        ("SHRINK_95_AT9", [(0.95, at9), (0.05, "ORIG")]),
    ]


def hits_target(report: SelectiveReport) -> bool:
    """Check whether a candidate meets all surgical server-peak target gates."""
    mu_sigma = report.full_ce_mean + report.full_ce_std
    return (
        1.55 <= report.full_ce_mean <= 1.85
        and 3.55 <= mu_sigma <= 3.90
        and 0.65 <= report.full_ce_p50 <= 1.00
        and report.full_frac_gt_95 <= 0.22
        and report.full_proxy_kl_t2 <= 0.038
        and report.full_delta_l2 <= 4.4
        and report.weighted_selective_score >= 0.56
    )


def _cpu_sd(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Extract a detached float32 CPU state_dict copy."""
    return {k: v.detach().cpu().float() for k, v in model.state_dict().items()}


def _model_from_sd(sd: dict[str, torch.Tensor], device: torch.device) -> torch.nn.Module:
    """Instantiate ViT-B/16 from a CPU state_dict on `device` in eval mode."""
    model = build_model()
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def _train_subset_sd(
    orig: torch.nn.Module, f_sub: torch.Tensor, orig_f_sub: torch.Tensor,
    cfg: MethodConfig, runner: str = "adaptive",
) -> dict[str, torch.Tensor]:
    """Train SalUn+NPO (`adaptive`, `tight`, or `tight_mid`) strictly on f_sub."""
    p_x = _build_train_proxy(f_sub, seed=cfg.seed)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f_sub.argmax(dim=-1), minlength=NUM_CLASSES) == 0
    if runner == "tight":
        sub = run_tight_at(orig, orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg)
    elif runner == "tight_mid":
        sub = run_tight_at(orig, orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg, MID_LIMITS)
    else:
        sub = run_adaptive_salun_npo(orig, orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg)
    return _cpu_sd(sub)


def train_base_family(
    orig: torch.nn.Module, f_x: torch.Tensor, orig_f: torch.Tensor,
    fold_indices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    cfg: MethodConfig, runner: str = "adaptive", train_full: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    """Train `cfg` on each rotating fold (+ optional full `f_x`), returning CPU state_dicts."""
    sds: dict[str, dict[str, torch.Tensor]] = {}
    for k, (f_idx, _, _) in enumerate(fold_indices):
        fold_cfg = MethodConfig(**{**asdict(cfg), "seed": cfg.seed + k})
        sds[f"fold_{k}"] = _train_subset_sd(orig, f_x[f_idx], orig_f[f_idx], fold_cfg, runner)
    if train_full:
        sds["full"] = _train_subset_sd(orig, f_x, orig_f, cfg, runner)
    return sds


def _eval_fold_sd(
    sd: dict[str, torch.Tensor], device: torch.device, f_x: torch.Tensor,
    f_x_alt: torch.Tensor, orig_f: torch.Tensor, orig_f_alt: torch.Tensor,
    labels: torch.Tensor, fold_id: int, indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> FoldResult:
    """Evaluate a fold state_dict on exact and altered subsets."""
    f_idx, r_in_idx, r_out_idx = indices
    sub = _model_from_sd(sd, device)
    sub_all = predict(sub, f_x, batch_size=128)
    f_raw = evaluate_real_subset(sub_all[f_idx], orig_f[f_idx], labels[f_idx])
    r_in_raw = evaluate_real_subset(sub_all[r_in_idx], orig_f[r_in_idx], labels[r_in_idx])
    r_out_raw = evaluate_real_subset(sub_all[r_out_idx], orig_f[r_out_idx], labels[r_out_idx])
    f_alt = evaluate_real_subset(predict(sub, f_x_alt[f_idx], batch_size=128), orig_f_alt[f_idx], labels[f_idx])
    r_in_alt = evaluate_real_subset(predict(sub, f_x_alt[r_in_idx], batch_size=128), orig_f_alt[r_in_idx], labels[r_in_idx])
    gap_r, gap_a = round(f_raw.ce_mean - r_in_raw.ce_mean, 4), round(f_alt.ce_mean - r_in_alt.ce_mean, 4)
    f_score = score_single_fold(f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw)
    return FoldResult(fold_id, len(f_idx), f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw, gap_r, gap_a, f_score)


def evaluate_family_sds(
    name: str, sds: dict[str, dict[str, torch.Tensor]], orig: torch.nn.Module,
    f_x: torch.Tensor, f_x_alt: torch.Tensor, orig_f: torch.Tensor, orig_f_alt: torch.Tensor,
    labels: torch.Tensor, fold_indices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    out_dir: Path, save_full: bool = True, full_metrics_override: dict[str, float] | None = None,
) -> SelectiveReport:
    """Evaluate 3 folds + full 3,500-image checkpoint for a state_dict family."""
    device = next(orig.parameters()).device
    folds = [_eval_fold_sd(sds[f"fold_{k}"], device, f_x, f_x_alt, orig_f, orig_f_alt, labels, k, idx) for k, idx in enumerate(fold_indices)]
    if full_metrics_override is not None:
        return aggregate_weighted_folds(name, folds, full_metrics_override)
    ckpt = out_dir / f"{name}.pt"
    if save_full:
        torch.save(sds["full"], ckpt)
    ev = evaluate_candidate(orig, _model_from_sd(sds["full"], device), f_x, sub_path=ckpt if save_full else None)
    full_m = {
        "full_ce_mean": ev.forget_raw.ce_mean, "full_ce_std": ev.forget_raw.ce_std,
        "full_ce_p50": ev.forget_raw.ce_p50, "full_frac_gt_95": ev.forget_raw.frac_gt_95,
        "full_proxy_kl_t2": ev.retain_proxy.proxy_kl_t2, "full_absent_kl_t2": ev.retain_proxy.absent_classes_kl_t2,
        "full_delta_l2": ev.retain_proxy.delta_l2_total,
    }
    return aggregate_weighted_folds(name, folds, full_m)


def _row_from_report(rep: SelectiveReport) -> dict[str, object]:
    row = asdict(rep)
    row["hits_target"] = hits_target(rep)
    return row


def _print_row(rep: SelectiveReport, prefix: str = "") -> None:
    hit, ms = ("HIT" if hits_target(rep) else "no"), rep.full_ce_mean + rep.full_ce_std
    print(
        f"{prefix}{rep.name:30s} | W_SEL={rep.weighted_selective_score:.4f}±{rep.fold_score_std:.3f} | "
        f"F={rep.weighted_forget_ce_raw:.2f}/{rep.weighted_forget_ce_alt:.2f} Gap={rep.weighted_intra_gap_50_50:+.2f} "
        f"In={rep.weighted_intra_retain_acc:.3f} Out={rep.weighted_inter_retain_acc:.3f}({rep.weighted_inter_retain_kl:.4f}) | "
        f"Full={rep.full_ce_mean:.3f}±{rep.full_ce_std:.2f} mu+s={ms:.2f} p50={rep.full_ce_p50:.3f} "
        f">95%={rep.full_frac_gt_95:.2f} PKL={rep.full_proxy_kl_t2:.4f} AKL={rep.full_absent_kl_t2:.4f} "
        f"dT={rep.full_delta_l2:.2f} | {hit}", flush=True,
    )


def _sort_key(r: dict[str, object]) -> tuple[bool, float, float]:
    return (bool(r["hits_target"]), float(r["weighted_selective_score"]), -float(r["full_proxy_kl_t2"]))


def _full_m_from_row(row: dict[str, object]) -> dict[str, float]:
    keys = ("full_ce_mean", "full_ce_std", "full_ce_p50", "full_frac_gt_95", "full_proxy_kl_t2", "full_absent_kl_t2", "full_delta_l2")
    return {k: float(row[k]) for k in keys}


def _run_heldout_seed777(
    rows: list[dict[str, object]], cfg_map: dict[str, tuple[str, MethodConfig]],
    soup_map: dict[str, list[tuple[float, str]]], orig: torch.nn.Module,
    f_x: torch.Tensor, f_x_alt: torch.Tensor, orig_f: torch.Tensor, orig_f_alt: torch.Tensor,
    labels: torch.Tensor, num_folds: int, out_dir: Path,
) -> list[dict[str, object]]:
    """Verify top AT2_*/AT3_* and top SHRINK_*/SOUP* formula on held-out seed=777 folds."""
    at_rows = sorted([r for r in rows if str(r["name"]).startswith(("AT2_", "AT3_"))], key=_sort_key, reverse=True)
    soup_rows = sorted([r for r in rows if str(r["name"]).startswith(("SHRINK_", "SOUP"))], key=_sort_key, reverse=True)
    picks = at_rows[:1] + soup_rows[:1]
    folds_777 = make_rotating_class_folds(labels, num_folds=num_folds, seed=777)
    orig_sd = _cpu_sd(orig)
    cache_777: dict[str, dict[str, dict[str, torch.Tensor]]] = {"ORIG": {f"fold_{k}": orig_sd for k in range(num_folds)}}
    heldout: list[dict[str, object]] = []
    for r in picks:
        name = str(r["name"])
        if name in cfg_map:
            rn, cg = cfg_map[name]
            cache_777.setdefault(name, train_base_family(orig, f_x, orig_f, folds_777, cg, rn, train_full=False))
            sds = cache_777[name]
        else:
            spec = soup_map[name]
            for _, src in spec:
                if src not in cache_777:
                    rn, cg = cfg_map[src]
                    cache_777[src] = train_base_family(orig, f_x, orig_f, folds_777, cg, rn, train_full=False)
            sds = {f"fold_{k}": blend_state_dicts([(w, cache_777[src][f"fold_{k}"]) for w, src in spec]) for k in range(num_folds)}
        rep = evaluate_family_sds(name, sds, orig, f_x, f_x_alt, orig_f, orig_f_alt, labels, folds_777, out_dir, save_full=False, full_metrics_override=_full_m_from_row(r))
        _print_row(rep, prefix="[seed777] ")
        heldout.append(_row_from_report(rep))
    return heldout


def _round_specs(round_num: int) -> tuple[list[tuple[str, MethodConfig]], list[tuple[str, list[tuple[float, str]]]]]:
    if round_num == 1:
        return [("adaptive", c) for c in build_round1_configs()], build_soup_specs()
    if round_num == 2:
        return build_round2_configs(), build_round2_soup_specs()
    return build_round3_configs(), build_round3_soup_specs()


def main() -> int:
    parser = argparse.ArgumentParser(description="Round 1/2/3 AT refinement + weight soups.")
    parser.add_argument("--round", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--out-dir", type=Path, default=Path("output/base_refine"))
    parser.add_argument("--num-folds", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    f_x_alt = make_altered_subset(f_x, seed=123)
    orig_f, orig_f_alt = predict(orig, f_x, batch_size=128), predict(orig, f_x_alt, batch_size=128)
    labels = orig_f.argmax(dim=-1)
    fold_indices = make_rotating_class_folds(labels, num_folds=args.num_folds, seed=42)

    base_items, soup_specs = _round_specs(args.round)
    json_path = args.out_dir / f"base_refine_round{args.round}.json"
    orig_sd = _cpu_sd(orig)
    all_sds: dict[str, dict[str, dict[str, torch.Tensor]]] = {
        "ORIG": {**{f"fold_{k}": orig_sd for k in range(args.num_folds)}, "full": orig_sd}
    }
    rows: list[dict[str, object]] = []
    for runner, cfg in base_items:
        sds = train_base_family(orig, f_x, orig_f, fold_indices, cfg, runner=runner)
        all_sds[cfg.name] = sds
        rep = evaluate_family_sds(cfg.name, sds, orig, f_x, f_x_alt, orig_f, orig_f_alt, labels, fold_indices, args.out_dir)
        _print_row(rep)
        rows.append(_row_from_report(rep))
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for soup_name, spec in soup_specs:
        blended = {slot: blend_state_dicts([(w, all_sds[src][slot]) for w, src in spec]) for slot in (*[f"fold_{k}" for k in range(args.num_folds)], "full")}
        rep = evaluate_family_sds(soup_name, blended, orig, f_x, f_x_alt, orig_f, orig_f_alt, labels, fold_indices, args.out_dir)
        _print_row(rep)
        rows.append(_row_from_report(rep))
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    rows.sort(key=_sort_key, reverse=True)
    if args.round == 1:
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return 0
    cfg_map = {c.name: (rn, c) for rn, c in base_items}
    heldout = _run_heldout_seed777(rows, cfg_map, dict(soup_specs), orig, f_x, f_x_alt, orig_f, orig_f_alt, labels, args.num_folds, args.out_dir)
    json_path.write_text(json.dumps({"results": rows, "heldout_seed777": heldout}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
