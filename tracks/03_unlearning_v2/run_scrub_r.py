"""Held-out (seed=777) 1-factor-at-a-time ablations for true SCRUB+R."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from auto_tune_4gpu import _build_train_proxy
from eval_harness import evaluate_candidate
from methods.patch_drop_targets import estimate_patchdrop_gold_logits
from methods.scrub_r import ScrubRConfig, run_scrub_r
from selective_benchmark import (
    FoldResult,
    aggregate_weighted_folds,
    evaluate_real_subset,
    make_altered_subset,
    make_rotating_class_folds,
    score_single_fold,
)
from task_template import NUM_CLASSES, load_model, predict


def round1_candidates() -> list[ScrubRConfig]:
    """Define 1-factor-at-a-time Round 1 ablations around true SCRUB+R."""
    base = ScrubRConfig(name="SR1_npo_pd_rewind")
    return [
        base,
        replace(base, name="SR2_no_rewind", use_rewind=False),
        replace(base, name="SR3_uniform_no_pd", pd_alpha=0.0, pd_gate_scale=0.0),
        replace(base, name="SR4_kl_ascent_pd", max_loss_kind="kl_ascent", lr_max=6.0e-5),
        replace(base, name="SR5_npo_pd_strong", target_forget_ce=1.80, max_epochs=7, total_epochs=11),
    ]


def _spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-8)
    return round(float((ra * rb).sum().item() / denom.item()), 4)


def _eval_fold(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    pd_all: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    f_idx: torch.Tensor,
    r_in_idx: torch.Tensor,
    r_out_idx: torch.Tensor,
    cfg: ScrubRConfig,
) -> FoldResult:
    p_x = _build_train_proxy(f_x[f_idx], seed=cfg.seed + k)
    orig_p = predict(orig, p_x, batch_size=128)
    absent = torch.bincount(labels[f_idx], minlength=NUM_CLASSES) == 0
    f_cfg = replace(cfg, seed=cfg.seed + k)
    sub, _ = run_scrub_r(orig, f_x[f_idx], orig_f[f_idx], p_x, orig_p, absent, f_cfg, pd_logits=pd_all[f_idx])
    sub_all = predict(sub, f_x, batch_size=128)
    f_raw = evaluate_real_subset(sub_all[f_idx], orig_f[f_idx], labels[f_idx])
    r_in_raw = evaluate_real_subset(sub_all[r_in_idx], orig_f[r_in_idx], labels[r_in_idx])
    r_out_raw = evaluate_real_subset(sub_all[r_out_idx], orig_f[r_out_idx], labels[r_out_idx])
    f_alt = evaluate_real_subset(predict(sub, f_x_alt[f_idx], batch_size=128), orig_f_alt[f_idx], labels[f_idx])
    r_in_alt = evaluate_real_subset(predict(sub, f_x_alt[r_in_idx], batch_size=128), orig_f_alt[r_in_idx], labels[r_in_idx])
    fs = score_single_fold(f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw)
    return FoldResult(
        k, len(f_idx), f_raw, f_alt, r_in_raw, r_in_alt, r_out_raw,
        round(f_raw.ce_mean - r_in_raw.ce_mean, 4), round(f_alt.ce_mean - r_in_alt.ce_mean, 4), fs,
    )


def _run_scrub_candidate(
    orig: torch.nn.Module,
    f_x: torch.Tensor,
    f_x_alt: torch.Tensor,
    orig_f: torch.Tensor,
    orig_f_alt: torch.Tensor,
    pd_all: torch.Tensor,
    cfg: ScrubRConfig,
    out_dir: Path,
    eval_seed: int,
) -> dict:
    labels = orig_f.argmax(dim=-1)
    folds = [
        _eval_fold(orig, f_x, f_x_alt, orig_f, orig_f_alt, pd_all, labels, k, fi, ri, ro, cfg)
        for k, (fi, ri, ro) in enumerate(make_rotating_class_folds(labels, 3, seed=eval_seed))
    ]
    p_full = _build_train_proxy(f_x, seed=cfg.seed)
    orig_p_full = predict(orig, p_full, batch_size=128)
    absent_full = torch.bincount(labels, minlength=NUM_CLASSES) == 0
    sub_full, history = run_scrub_r(orig, f_x, orig_f, p_full, orig_p_full, absent_full, cfg, pd_logits=pd_all)
    ckpt = out_dir / f"{cfg.name}.pt"
    torch.save({k: v.detach().cpu().float() for k, v in sub_full.state_dict().items()}, ckpt)
    ev = evaluate_candidate(orig, sub_full, f_x, sub_path=ckpt)
    ce_sub = F.cross_entropy(predict(sub_full, f_x, batch_size=128).float(), labels, reduction="none")
    ce_pd = F.cross_entropy(pd_all.float(), labels, reduction="none")
    full_m = {
        "full_ce_mean": ev.forget_raw.ce_mean, "full_ce_std": ev.forget_raw.ce_std,
        "full_ce_p50": ev.forget_raw.ce_p50, "full_frac_gt_95": ev.forget_raw.frac_gt_95,
        "full_proxy_kl_t2": ev.retain_proxy.proxy_kl_t2, "full_absent_kl_t2": ev.retain_proxy.absent_classes_kl_t2,
        "full_delta_l2": ev.retain_proxy.delta_l2_total,
    }
    rep = asdict(aggregate_weighted_folds(cfg.name, folds, full_m))
    rep["spearman_vs_patchdrop"] = _spearman_corr(ce_sub, ce_pd)
    rep["rewind_history"] = history
    rep["selected_epoch"] = next((h["epoch"] for h in history if h.get("selected")), -1)
    return rep


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out (seed=777) 1-factor ablations for SCRUB+R.")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("output/scrub_r"))
    parser.add_argument("--eval-seed", type=int, default=777)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.round != 1:
        raise ValueError(f"Unsupported round: {args.round}")

    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    f_x_alt = make_altered_subset(f_x, seed=args.eval_seed + 100)
    orig_f = predict(orig, f_x, batch_size=128)
    orig_f_alt = predict(orig, f_x_alt, batch_size=128)
    pd_all = estimate_patchdrop_gold_logits(orig, f_x, drop_prob=0.45, num_views=4, seed=42)

    rows = []
    for cfg in round1_candidates():
        r = _run_scrub_candidate(orig, f_x, f_x_alt, orig_f, orig_f_alt, pd_all, cfg, args.out_dir, args.eval_seed)
        rows.append(r)
        print(
            f"[scrub+r] {r['name']:20s} | ep={r['selected_epoch']} | W_SEL={r['weighted_selective_score']:.4f}±{r['fold_score_std']:.3f} | "
            f"rho_PD={r['spearman_vs_patchdrop']:+.3f} IntraAcc={r['weighted_intra_retain_acc']:.3f} InterAcc={r['weighted_inter_retain_acc']:.3f} | "
            f"Full={r['full_ce_mean']:.3f}±{r['full_ce_std']:.2f} (mu+s={r['full_ce_mean']+r['full_ce_std']:.2f}) "
            f"p50={r['full_ce_p50']:.3f} >95%={r['full_frac_gt_95']:.2f} PKL={r['full_proxy_kl_t2']:.4f} dT={r['full_delta_l2']:.2f}",
            flush=True,
        )
    (args.out_dir / f"scrub_r_round{args.round}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
