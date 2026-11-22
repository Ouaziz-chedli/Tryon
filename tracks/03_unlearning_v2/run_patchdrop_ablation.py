"""Held-out (seed=777) 1-factor ablation of per-image Patch-Dropout Virtual Gold alignment."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from auto_tune_4gpu import _build_train_proxy
from contracts import MethodConfig
from eval_harness import evaluate_candidate
from methods.patch_drop_targets import estimate_patchdrop_gold_logits, run_patchdrop_unlearn
from selective_benchmark import (
    FoldResult,
    aggregate_weighted_folds,
    evaluate_real_subset,
    make_altered_subset,
    make_rotating_class_folds,
    score_single_fold,
)
from task_template import NUM_CLASSES, load_model, predict


def _spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-8)
    return round(float((ra * rb).sum().item() / denom.item()), 4)


def _run_pd_candidate(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    pd_all: torch.Tensor,
    cfg: MethodConfig,
    kl_w: float,
    npo_w: float,
    tgt_ce: float,
    out_dir: Path,
    eval_seed: int = 777,
) -> dict:
    labels = orig_f.argmax(dim=-1)
    folds = []
    for k, (f_idx, r_in_idx, r_out_idx) in enumerate(make_rotating_class_folds(labels, 3, seed=eval_seed)):
        p_x = _build_train_proxy(f_x[f_idx], seed=cfg.seed + k)
        orig_p = predict(orig, p_x, batch_size=128)
        absent = torch.bincount(labels[f_idx], minlength=NUM_CLASSES) == 0
        f_cfg = MethodConfig(**{**asdict(cfg), "seed": cfg.seed + k})
        sub = run_patchdrop_unlearn(
            orig, f_x[f_idx], orig_f[f_idx], p_x, orig_p, absent, f_cfg,
            pd_logits=pd_all[f_idx], kl_weight=kl_w, npo_weight=npo_w, target_mean_ce=tgt_ce,
        )
        sub_all = predict(sub, f_x, batch_size=128)
        f_raw = evaluate_real_subset(sub_all[f_idx], orig_f[f_idx], labels[f_idx])
        r_in_raw = evaluate_real_subset(sub_all[r_in_idx], orig_f[r_in_idx], labels[r_in_idx])
        r_out_raw = evaluate_real_subset(sub_all[r_out_idx], orig_f[r_out_idx], labels[r_out_idx])
        f_alt = evaluate_real_subset(predict(sub, f_x_alt[f_idx], batch_size=128), orig_f_alt[f_idx], labels[f_idx])
        r_in_alt = evaluate_real_subset(predict(sub, f_x_alt[r_in_idx], batch_size=128), orig_f_alt[r_in_idx], labels[r_in_idx])
        fs = score_single_fold(f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw)
        folds.append(FoldResult(
            k, len(f_idx), f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw,
            round(f_raw.ce_mean - r_in_raw.ce_mean, 4), round(f_alt.ce_mean - r_in_alt.ce_mean, 4), fs,
        ))

    p_full = _build_train_proxy(f_x, seed=cfg.seed)
    orig_p_full = predict(orig, p_full, batch_size=128)
    absent_full = torch.bincount(labels, minlength=NUM_CLASSES) == 0
    sub_full = run_patchdrop_unlearn(
        orig, f_x, orig_f, p_full, orig_p_full, absent_full, cfg,
        pd_logits=pd_all, kl_weight=kl_w, npo_weight=npo_w, target_mean_ce=tgt_ce,
    )
    ckpt = out_dir / f"{cfg.name}.pt"
    torch.save({k: v.detach().cpu().float() for k, v in sub_full.state_dict().items()}, ckpt)
    ev = evaluate_candidate(orig, sub_full, f_x, sub_path=ckpt)
    sub_full_f = predict(sub_full, f_x, batch_size=128)
    ce_sub = F.cross_entropy(sub_full_f.float(), labels, reduction="none")
    ce_pd = F.cross_entropy(pd_all.float(), labels, reduction="none")
    rho_pd = _spearman_corr(ce_sub, ce_pd)
    full_m = {
        "full_ce_mean": ev.forget_raw.ce_mean, "full_ce_std": ev.forget_raw.ce_std,
        "full_ce_p50": ev.forget_raw.ce_p50, "full_frac_gt_95": ev.forget_raw.frac_gt_95,
        "full_proxy_kl_t2": ev.retain_proxy.proxy_kl_t2, "full_absent_kl_t2": ev.retain_proxy.absent_classes_kl_t2,
        "full_delta_l2": ev.retain_proxy.delta_l2_total,
    }
    rep = asdict(aggregate_weighted_folds(cfg.name, folds, full_m))
    rep["spearman_vs_patchdrop"] = rho_pd
    return rep


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out ablation of per-image Patch-Dropout alignment.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/patchdrop_vgt"))
    parser.add_argument("--eval-seed", type=int, default=777)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    f_x_alt = make_altered_subset(f_x, seed=args.eval_seed + 100)
    orig_f = predict(orig, f_x, batch_size=128)
    orig_f_alt = predict(orig, f_x_alt, batch_size=128)
    pd_all = estimate_patchdrop_gold_logits(orig, f_x, drop_prob=0.45, num_views=4, seed=42)

    specs = [
        (MethodConfig(name="PD_1_kl10_npo03_ce175", strategy="salun_npo", steps=200, lr=9.5e-5, npo_beta=1.8, retain_kl_weight=2.4, l2_weight=1.8e-3, salun_keep_ratio=0.20, freeze_until_block=6, head_lr_mult=1.0), 1.0, 0.35, 1.75),
        (MethodConfig(name="PD_2_kl12_npo00_ce180", strategy="salun_npo", steps=200, lr=1.0e-4, npo_beta=1.8, retain_kl_weight=2.4, l2_weight=1.8e-3, salun_keep_ratio=0.20, freeze_until_block=6, head_lr_mult=1.0), 1.2, 0.00, 1.80),
        (MethodConfig(name="PD_3_kl10_npo05_ce165", strategy="salun_npo", steps=200, lr=9.5e-5, npo_beta=1.8, retain_kl_weight=2.6, l2_weight=2.0e-3, salun_keep_ratio=0.20, freeze_until_block=6, head_lr_mult=1.0), 1.0, 0.50, 1.65),
    ]
    rows = []
    for cfg, kl_w, npo_w, tgt_ce in specs:
        r = _run_pd_candidate(orig, f_x, f_x_alt, orig_f, orig_f_alt, pd_all, cfg, kl_w, npo_w, tgt_ce, args.out_dir, args.eval_seed)
        rows.append(r)
        print(
            f"[pd] {r['name']:24s} | W_SEL={r['weighted_selective_score']:.4f}±{r['fold_score_std']:.3f} | "
            f"rho_PD={r['spearman_vs_patchdrop']:+.3f} IntraAcc={r['weighted_intra_retain_acc']:.3f} InterKL={r['weighted_inter_retain_kl']:.4f} | "
            f"Full={r['full_ce_mean']:.3f}±{r['full_ce_std']:.2f} (mu+s={r['full_ce_mean']+r['full_ce_std']:.2f}) p50={r['full_ce_p50']:.3f} >95%={r['full_frac_gt_95']:.2f} PKL={r['full_proxy_kl_t2']:.4f} dT={r['full_delta_l2']:.2f}",
            flush=True,
        )
    (args.out_dir / "patchdrop_ablation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
