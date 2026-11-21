"""Selective Unlearning Benchmark on forget_images.npz (Intra-Class & Cross-Class splits)."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from auto_tune_4gpu import _build_train_proxy, run_adaptive_salun_npo
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
from methods.lora_unlearn import run_lora_npo
from task_template import NUM_CLASSES, load_model, predict


@dataclass(frozen=True)
class SplitMetrics:
    """Metrics on a forget or held-out real-image subset of forget_images.npz."""

    ce_mean: float
    ce_std: float
    ce_p50: float
    top1_acc: float
    frac_gt_95: float
    kl_t2: float


@dataclass(frozen=True)
class SelectiveReport:
    """Combined Intra-Class (instance selectivity) and Cross-Class (collateral drift) report."""

    name: str
    intra_forget: SplitMetrics
    intra_same_class_retain: SplitMetrics
    intra_selectivity_gap: float
    cross_forget: SplitMetrics
    cross_other_classes_retain: SplitMetrics
    selective_score: float


def make_intra_class_split(labels: torch.Tensor, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each of the 45 active classes 50/50 into (forget_idx, same_class_retain_idx)."""
    gen = torch.Generator().manual_seed(seed)
    f_list, r_list = [], []
    for cls_id in labels.unique(sorted=True):
        cls_idx = torch.where(labels == cls_id)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=gen)]
        half = max(1, len(perm) // 2)
        f_list.append(perm[:half])
        r_list.append(perm[half:])
    return torch.cat(f_list), torch.cat(r_list)


def make_cross_class_split(labels: torch.Tensor, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the 45 active classes into 22 forgotten classes and 23 held-out active classes."""
    gen = torch.Generator().manual_seed(seed)
    classes = labels.unique(sorted=True)
    perm_c = classes[torch.randperm(len(classes), generator=gen)]
    half_c = max(1, len(perm_c) // 2)
    f_mask = torch.isin(labels, perm_c[:half_c])
    return torch.where(f_mask)[0], torch.where(~f_mask)[0]


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


def score_selective_candidate(
    intra_f: SplitMetrics,
    intra_r: SplitMetrics,
    cross_f: SplitMetrics,
    cross_r: SplitMetrics,
) -> float:
    """Compute Unlearning.pdf-aligned selective score: forget quality x real-image retain score."""
    if cross_r.top1_acc < 0.70:
        return 0.0
    d_intra = abs(intra_f.ce_p50 - 0.35) + 0.25 * abs(intra_f.ce_std - 1.20) + 0.5 * max(0.0, intra_f.frac_gt_95 - 0.38)
    d_cross = abs(cross_f.ce_p50 - 0.35) + 0.25 * abs(cross_f.ce_std - 1.20) + 0.5 * max(0.0, cross_f.frac_gt_95 - 0.38)
    forget_q = 0.5 * ((1.0 - min(1.0, d_intra / 1.60)) + (1.0 - min(1.0, d_cross / 1.60)))
    intra_sel = min(1.0, max(0.0, (intra_f.ce_mean - intra_r.ce_mean) / max(intra_f.ce_mean, 0.15)))
    cross_ret = cross_r.top1_acc * (1.0 - min(1.0, cross_r.kl_t2 / 0.25))
    same_ret = intra_r.top1_acc * (1.0 - min(1.0, intra_r.kl_t2 / 0.60))
    retain_q = 0.65 * cross_ret + 0.35 * same_ret
    return round(float((0.75 * forget_q + 0.25 * intra_sel) * retain_q), 4)


def _run_method_on_subset(
    orig: torch.nn.Module,
    f_sub: torch.Tensor,
    orig_f_sub: torch.Tensor,
    kind: str,
    cfg: MethodConfig,
) -> torch.nn.Module:
    """Train candidate unlearning method strictly on f_sub (leaving held-out images unseen)."""
    p_x = _build_train_proxy(f_sub, seed=cfg.seed)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f_sub.argmax(dim=-1), minlength=NUM_CLASSES) == 0
    if kind == "lora_npo":
        return run_lora_npo(orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg)
    return run_adaptive_salun_npo(orig, orig, f_sub, orig_f_sub, p_x, orig_p, absent_mask, cfg)


def benchmark_candidate(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    kind: str,
    cfg: MethodConfig,
) -> SelectiveReport:
    """Run both Intra-Class (50/50 per class) and Cross-Class (22 vs 23 classes) benchmarks."""
    labels = orig_f.argmax(dim=-1)
    in_f_idx, in_r_idx = make_intra_class_split(labels, seed=42)
    sub_intra = _run_method_on_subset(orig, f_x[in_f_idx], orig_f[in_f_idx], kind, cfg)
    intra_f = evaluate_real_subset(predict(sub_intra, f_x[in_f_idx], batch_size=128), orig_f[in_f_idx], labels[in_f_idx])
    intra_r = evaluate_real_subset(predict(sub_intra, f_x[in_r_idx], batch_size=128), orig_f[in_r_idx], labels[in_r_idx])

    cr_f_idx, cr_r_idx = make_cross_class_split(labels, seed=42)
    sub_cross = _run_method_on_subset(orig, f_x[cr_f_idx], orig_f[cr_f_idx], kind, cfg)
    cross_f = evaluate_real_subset(predict(sub_cross, f_x[cr_f_idx], batch_size=128), orig_f[cr_f_idx], labels[cr_f_idx])
    cross_r = evaluate_real_subset(predict(sub_cross, f_x[cr_r_idx], batch_size=128), orig_f[cr_r_idx], labels[cr_r_idx])

    gap = round(intra_f.ce_mean - intra_r.ce_mean, 4)
    score = score_selective_candidate(intra_f, intra_r, cross_f, cross_r)
    return SelectiveReport(cfg.name, intra_f, intra_r, gap, cross_f, cross_r, score)


def _top4_candidates() -> list[tuple[str, MethodConfig]]:
    """The 4 top candidates from our 2xA100 Colab sweep."""
    return [
        ("adaptive", MethodConfig(name="AT_ce0.50_b3.0_f8", strategy="salun_npo", steps=210, lr=8e-5, npo_beta=3.0, target_forget_ce=0.50, retain_kl_weight=2.5, l2_weight=2e-3, salun_keep_ratio=0.20, freeze_until_block=8)),
        ("adaptive", MethodConfig(name="AT_ce0.50_b2.0_f6", strategy="salun_npo", steps=210, lr=8e-5, npo_beta=2.0, target_forget_ce=0.50, retain_kl_weight=2.5, l2_weight=2e-3, salun_keep_ratio=0.20, freeze_until_block=6)),
        ("adaptive", MethodConfig(name="TS_ce0.50_b2.5_k25", strategy="salun_npo", loss_kind="two_sided_npo", steps=210, lr=8e-5, npo_beta=2.5, target_forget_ce=0.50, target_ce_upper=0.75, pullback_weight=0.5, retain_kl_weight=2.5, l2_weight=2e-3, salun_keep_ratio=0.25, freeze_until_block=8)),
        ("lora_npo", MethodConfig(name="LORA_ce0.50_r16", strategy="lora_npo", loss_kind="two_sided_npo", steps=180, lr=2e-4, npo_beta=2.5, target_forget_ce=0.50, target_ce_upper=0.70, retain_kl_weight=3.0, lora_rank=16)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Selective Intra-Class & Cross-Class benchmark on forget_images.npz.")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("output/selective"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    orig_f = predict(orig, f_x, batch_size=128)
    mine = [c for i, c in enumerate(_top4_candidates()) if i % args.num_workers == args.worker_id]
    rows = []
    for kind, cfg in mine:
        rep = benchmark_candidate(orig, f_x, orig_f, kind, cfg)
        rows.append(asdict(rep))
        print(
            f"[sel-gpu{args.worker_id}] {rep.name:20s} | SEL={rep.selective_score:.4f} | "
            f"Intra(F_ce={rep.intra_forget.ce_mean:.3f} p50={rep.intra_forget.ce_p50:.3f} vs R_ce={rep.intra_same_class_retain.ce_mean:.3f} R_acc={rep.intra_same_class_retain.top1_acc:.3f} R_kl={rep.intra_same_class_retain.kl_t2:.4f} gap={rep.intra_selectivity_gap:+.3f}) | "
            f"Cross(F_ce={rep.cross_forget.ce_mean:.3f} vs R_acc={rep.cross_other_classes_retain.top1_acc:.3f} R_kl={rep.cross_other_classes_retain.kl_t2:.4f})",
            flush=True,
        )
    (args.out_dir / f"selective_worker_{args.worker_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
