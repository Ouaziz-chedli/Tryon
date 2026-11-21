"""3-Fold Weighted Rotating Intra-Class & Inter-Class Selective Benchmark on forget_images.npz."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from augment import make_stress_suite
from auto_tune_4gpu import _build_train_proxy, run_adaptive_salun_npo
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature, evaluate_candidate
from task_template import NUM_CLASSES, load_model, predict


@dataclass(frozen=True)
class SplitMetrics:
    """Metrics on an exact or altered real-image subset of forget_images.npz."""

    ce_mean: float
    ce_std: float
    ce_p50: float
    top1_acc: float
    frac_gt_95: float
    kl_t2: float


@dataclass(frozen=True)
class FoldResult:
    """Single fold result on a selected group of active classes (exact + altered)."""

    fold_id: int
    num_forget_images: int
    forget_raw: SplitMetrics
    forget_alt: SplitMetrics
    intra_retain_raw: SplitMetrics
    intra_retain_alt: SplitMetrics
    inter_retain_raw: SplitMetrics
    gap_raw: float
    gap_alt: float
    fold_score: float


@dataclass(frozen=True)
class SelectiveReport:
    """3-Fold sample-weighted selective report + full 3,500-image checkpoint metrics."""

    name: str
    weighted_selective_score: float
    fold_score_std: float
    weighted_forget_ce_raw: float
    weighted_forget_ce_alt: float
    weighted_intra_retain_acc: float
    weighted_intra_gap_50_50: float
    weighted_inter_retain_acc: float
    weighted_inter_retain_kl: float
    full_ce_mean: float
    full_ce_std: float
    full_ce_p50: float
    full_frac_gt_95: float
    full_proxy_kl_t2: float
    full_absent_kl_t2: float
    full_delta_l2: float
    folds: list[FoldResult]


def make_rotating_class_folds(
    labels: torch.Tensor,
    num_folds: int = 3,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Partition active classes into K folds; each fold returns (f_idx, intra_r_idx, inter_r_idx)."""
    gen = torch.Generator().manual_seed(seed)
    classes = labels.unique(sorted=True)
    perm_c = classes[torch.randperm(len(classes), generator=gen)]
    fold_classes = torch.tensor_split(perm_c, num_folds)
    folds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for k, sel_c in enumerate(fold_classes):
        f_list, r_intra_list = [], []
        fold_gen = torch.Generator().manual_seed(seed + 100 * (k + 1))
        for cls_id in sel_c:
            cls_idx = torch.where(labels == cls_id)[0]
            p_idx = cls_idx[torch.randperm(len(cls_idx), generator=fold_gen)]
            half = max(1, len(p_idx) // 2)
            f_list.append(p_idx[:half])
            r_intra_list.append(p_idx[half:])
        inter_mask = ~torch.isin(labels, sel_c)
        folds.append((torch.cat(f_list), torch.cat(r_intra_list), torch.where(inter_mask)[0]))
    return folds


def make_altered_subset(images: torch.Tensor, seed: int = 123) -> torch.Tensor:
    """Create a deterministic altered copy of `images` mixing hflip, crop80, color_jitter, blur."""
    suite = make_stress_suite(images, seed=seed)
    keys = list(suite.keys())
    out = images.clone()
    for idx, key in enumerate(keys):
        out[idx :: len(keys)] = suite[key][idx :: len(keys)]
    return out


@torch.no_grad()
def evaluate_real_subset(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
) -> SplitMetrics:
    """Compute CE distribution, top-1 retention accuracy, and KL(T=2) on real images."""
    ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
    probs = F.softmax(sub_logits.float(), dim=-1)
    acc = float((sub_logits.argmax(dim=-1) == labels).float().mean().item())
    f95 = float((probs.max(dim=-1).values > 0.95).float().mean().item())
    kl2 = float(compute_kl_at_temperature(orig_logits.float(), sub_logits.float(), 2.0).mean().item())
    return SplitMetrics(
        ce_mean=round(float(ce_i.mean().item()), 4),
        ce_std=round(float(ce_i.std(correction=0).item()), 4),
        ce_p50=round(float(ce_i.median().item()), 4),
        top1_acc=round(acc, 4),
        frac_gt_95=round(f95, 4),
        kl_t2=round(kl2, 4),
    )


def score_single_fold(
    f_raw: SplitMetrics,
    f_alt: SplitMetrics,
    r_intra_raw: SplitMetrics,
    r_intra_alt: SplitMetrics,
    r_inter_raw: SplitMetrics,
) -> float:
    """Score one fold using 50/50 exact+altered forget strength, intra selectivity, and inter/intra retention."""
    if r_inter_raw.top1_acc < 0.70 or r_intra_raw.top1_acc < 0.55:
        return 0.0
    f_raw_ms = min(1.0, (f_raw.ce_mean + f_raw.ce_std) / 5.2) * (1.0 - 0.5 * f_raw.frac_gt_95)
    f_alt_ms = min(1.0, (f_alt.ce_mean + f_alt.ce_std) / 6.2) * (1.0 - 0.5 * f_alt.frac_gt_95)
    forget_50_50 = 0.5 * (f_raw_ms + f_alt_ms)
    gap_50_50 = 0.5 * ((f_raw.ce_mean - r_intra_raw.ce_mean) + (f_alt.ce_mean - r_intra_alt.ce_mean))
    sel_bonus = min(1.0, max(0.0, gap_50_50 / 0.65))
    inter_ret = r_inter_raw.top1_acc * (1.0 - min(1.0, r_inter_raw.kl_t2 / 0.30))
    intra_ret = (0.5 * (r_intra_raw.top1_acc + r_intra_alt.top1_acc)) * (1.0 - min(1.0, r_intra_raw.kl_t2 / 1.40))
    retain_q = 0.65 * inter_ret + 0.35 * intra_ret
    return round(float((0.70 * forget_50_50 + 0.30 * sel_bonus) * retain_q), 4)


def _train_on_subset(
    orig: torch.nn.Module,
    f_sub: torch.Tensor,
    orig_f_sub: torch.Tensor,
    cfg: MethodConfig,
) -> torch.nn.Module:
    """Train adaptive SalUn+NPO strictly on f_sub (zero leakage from held-out intra/inter subsets)."""
    p_x = _build_train_proxy(f_sub, seed=cfg.seed)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f_sub.argmax(dim=-1), minlength=NUM_CLASSES) == 0
    return run_adaptive_salun_npo(orig, orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg)


def _run_one_fold(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    labels: torch.Tensor,
    fold_id: int,
    indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    cfg: MethodConfig,
) -> FoldResult:
    """Execute training and exact+altered evaluation on a single class-rotation fold."""
    f_idx, r_in_idx, r_out_idx = indices
    fold_cfg = MethodConfig(**{**asdict(cfg), "seed": cfg.seed + fold_id})
    sub = _train_on_subset(orig, f_x[f_idx], orig_f[f_idx], fold_cfg)
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


def aggregate_weighted_folds(
    name: str,
    folds: list[FoldResult],
    full_metrics: dict[str, float],
) -> SelectiveReport:
    """Aggregate K rotating folds weighted by num_forget_images and penalize inter-fold variance."""
    total_n = max(1, sum(f.num_forget_images for f in folds))
    w = [f.num_forget_images / total_n for f in folds]
    w_score_mean = sum(wk * f.fold_score for wk, f in zip(w, folds))
    f_std = float(np.std([f.fold_score for f in folds]))
    w_score = round(float(w_score_mean - 0.20 * f_std), 4)
    return SelectiveReport(
        name=name,
        weighted_selective_score=w_score,
        fold_score_std=round(f_std, 4),
        weighted_forget_ce_raw=round(sum(wk * f.forget_raw.ce_mean for wk, f in zip(w, folds)), 4),
        weighted_forget_ce_alt=round(sum(wk * f.forget_alt.ce_mean for wk, f in zip(w, folds)), 4),
        weighted_intra_retain_acc=round(sum(wk * f.intra_retain_raw.top1_acc for wk, f in zip(w, folds)), 4),
        weighted_intra_gap_50_50=round(sum(wk * 0.5 * (f.gap_raw + f.gap_alt) for wk, f in zip(w, folds)), 4),
        weighted_inter_retain_acc=round(sum(wk * f.inter_retain_raw.top1_acc for wk, f in zip(w, folds)), 4),
        weighted_inter_retain_kl=round(sum(wk * f.inter_retain_raw.kl_t2 for wk, f in zip(w, folds)), 4),
        **full_metrics,
        folds=folds,
    )


def benchmark_and_save_candidate(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    cfg: MethodConfig,
    out_dir: Path,
    num_folds: int = 3,
) -> SelectiveReport:
    """Run K-fold weighted selective benchmark on forget_images.npz AND save full 3,500-image checkpoint."""
    labels = orig_f.argmax(dim=-1)
    fold_indices = make_rotating_class_folds(labels, num_folds=num_folds, seed=cfg.seed)
    folds = [
        _run_one_fold(orig, f_x, f_x_alt, orig_f, orig_f_alt, labels, k, idx_tuple, cfg)
        for k, idx_tuple in enumerate(fold_indices)
    ]
    p_x = _build_train_proxy(f_x, seed=cfg.seed)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(labels, minlength=NUM_CLASSES) == 0
    sub_full = run_adaptive_salun_npo(orig, orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
    ckpt = out_dir / f"{cfg.name}.pt"
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
    return aggregate_weighted_folds(cfg.name, folds, full_m)


def _strong_candidates() -> list[MethodConfig]:
    """8 Strong Unlearning candidates (2 per GPU on a 4-GPU JURECA node)."""
    return [
        MethodConfig(name="AT_ce1.00_b1.8_f6", strategy="salun_npo", steps=225, lr=1.0e-4, npo_beta=1.8, target_forget_ce=1.00, target_ce_upper=1.80, retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.20, freeze_until_block=6),
        MethodConfig(name="AT_ce1.40_b1.5_f6", strategy="salun_npo", steps=225, lr=1.1e-4, npo_beta=1.5, target_forget_ce=1.40, target_ce_upper=2.20, retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.20, freeze_until_block=6),
        MethodConfig(name="AT_ce1.40_b1.5_f8", strategy="salun_npo", steps=225, lr=1.2e-4, npo_beta=1.5, target_forget_ce=1.40, target_ce_upper=2.20, retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.20, freeze_until_block=8, head_lr_mult=1.5),
        MethodConfig(name="AT_ce1.80_b1.2_f6", strategy="salun_npo", steps=240, lr=1.2e-4, npo_beta=1.2, target_forget_ce=1.80, target_ce_upper=2.60, retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.25, freeze_until_block=6),
        MethodConfig(name="AT_ce2.20_b1.0_f6", strategy="salun_npo", steps=240, lr=1.3e-4, npo_beta=1.0, target_forget_ce=2.20, target_ce_upper=3.00, retain_kl_weight=2.5, l2_weight=1.5e-3, salun_keep_ratio=0.25, freeze_until_block=6),
        MethodConfig(name="TS_ce1.20_u2.40_b1.5_f6", strategy="salun_npo", loss_kind="two_sided_npo", steps=225, lr=1.1e-4, npo_beta=1.5, target_forget_ce=1.20, target_ce_upper=2.40, pullback_weight=0.4, retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.25, freeze_until_block=6),
        MethodConfig(name="TS_ce1.60_u2.80_b1.2_f6", strategy="salun_npo", loss_kind="two_sided_npo", steps=240, lr=1.2e-4, npo_beta=1.2, target_forget_ce=1.60, target_ce_upper=2.80, pullback_weight=0.4, retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.25, freeze_until_block=6),
        MethodConfig(name="TS_ce1.60_u2.80_b1.2_f8", strategy="salun_npo", loss_kind="two_sided_npo", steps=240, lr=1.3e-4, npo_beta=1.2, target_forget_ce=1.60, target_ce_upper=2.80, pullback_weight=0.4, retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.25, freeze_until_block=8, head_lr_mult=1.5),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="3-Fold Weighted Rotating Selective Benchmark on 4 GPUs.")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-folds", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, default=Path("output/selective_strong"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    f_x_alt = make_altered_subset(f_x, seed=123)
    orig_f = predict(orig, f_x, batch_size=128)
    orig_f_alt = predict(orig, f_x_alt, batch_size=128)
    mine = [c for i, c in enumerate(_strong_candidates()) if i % args.num_workers == args.worker_id]
    rows = []
    for cfg in mine:
        rep = benchmark_and_save_candidate(orig, f_x, f_x_alt, orig_f, orig_f_alt, cfg, args.out_dir, args.num_folds)
        rows.append(asdict(rep))
        print(
            f"[gpu{args.worker_id}] {rep.name:24s} | W_SEL={rep.weighted_selective_score:.4f}±{rep.fold_score_std:.3f} | "
            f"3Fold(F_raw={rep.weighted_forget_ce_raw:.2f} F_alt={rep.weighted_forget_ce_alt:.2f} IntraAcc={rep.weighted_intra_retain_acc:.3f} Gap50={rep.weighted_intra_gap_50_50:+.2f} InterAcc={rep.weighted_inter_retain_acc:.3f} InterKL={rep.weighted_inter_retain_kl:.4f}) | "
            f"Full(CE={rep.full_ce_mean:.3f}±{rep.full_ce_std:.2f} p50={rep.full_ce_p50:.3f} >95%={rep.full_frac_gt_95:.2f} ProxyKL={rep.full_proxy_kl_t2:.4f} dTheta={rep.full_delta_l2:.2f})",
            flush=True,
        )
    rows.sort(key=lambda r: r["weighted_selective_score"], reverse=True)
    (args.out_dir / f"selective_worker_{args.worker_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
