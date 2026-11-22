"""SweepMax grid: 52 configs (44 AT moderate + 4 raw-view + 4 gold-bell diamond probes) + multi-seed soups.

Two-halves diagnostic (05h4x): the EXACT/raw view is the lagging score half, so 4 grid
configs use aug_mode="none" (raw view weight 1.0). No clean/aug mix fraction knob exists in
augment.py; a true asymmetric two-view mix is Phase 3.1/3.2, NOT coded here.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from auto_tune_4gpu import _build_train_proxy
from contracts import MethodConfig
from eval_harness import evaluate_candidate, score_server_calibrated_proxy
from methods.salun_npo import run_salun_npo
from run_base_refine import _at_cfg, _model_from_sd, blend_state_dicts
from task_template import DEVICE, NUM_CLASSES, load_model, predict


def _in_window(row: dict) -> bool:
    """Full server submission window on mu+sigma, p50, ProxyKL, dTheta, frac_gt_95."""
    return (
        3.65 <= row["mu_sigma"] <= 4.00
        and 0.80 <= row["ce_p50"] <= 1.20
        and row["proxy_kl_t2"] <= 0.056
        and row["delta_l2_total"] <= 4.70
        and row["frac_gt_95"] <= 0.30
    )


def _build_grid() -> list[MethodConfig]:
    """44 moderate AT + 4 raw-view (aug none) + 4 gold-bell diamond probes (idx 48-51).

    Diamond probes (GoldBellFit docs §6 + PB empirics): PB target 1.98/u2.0 strangled the
    whole distribution (mu ~ 0.34x target, pullback 0.8), so the TIGHT branch is scanned at
    target {2.6, 3.4, 4.2} with the same u2.0 ceiling; PAcal maps the doc (1.55, 2.15) split
    via the npo empirical offset mu ~ target+0.9, p50 ~ target+0.2.
    """
    cfgs = []
    for ce in (0.70, 0.80, 0.90, 1.00):
        for beta in (1.8, 2.0, 2.5):
            for kl_w in (2.2, 2.8):
                for seed in (42, 101):
                    if beta == 2.0 and kl_w == 2.2 and ce in (0.80, 0.90):
                        continue  # slot given to the raw-view probes below
                    name = f"SG_ce{ce:.2f}_b{beta:.1f}_kl{kl_w:.1f}_s{seed}"
                    cfgs.append(_at_cfg(name, 1e-4, ce, ce + 0.45, beta, kl_w, seed=seed))
    raw_views = ((0.80, 1.8, 42), (0.90, 1.8, 42), (0.80, 2.5, 101), (0.90, 2.5, 101))
    for ce, beta, seed in raw_views:
        name = f"SGnone_ce{ce:.2f}_b{beta:.1f}_kl2.8_s{seed}"
        cfgs.append(_at_cfg(name, 1e-4, ce, ce + 0.45, beta, 2.8, aug_mode="none", seed=seed))
    probes = (
        ("PBcal_ce2.6_u2.0_b2.0_kl2.8_s101", 2.6, 2.0),
        ("PBcal_ce3.4_u2.0_b2.0_kl2.8_s101", 3.4, 2.0),
        ("PBcal_ce4.2_u2.0_b2.0_kl2.8_s101", 4.2, 2.0),
        ("PAcal_ce0.65_u3.2_b2.0_kl2.8_s101", 0.65, 3.2),
    )
    for name, ce, up in probes:
        cfgs.append(MethodConfig(
            name=name, strategy="salun_npo", loss_kind="two_sided_npo", pullback_weight=0.8,
            steps=210, lr=1e-4, npo_beta=2.0, target_forget_ce=ce, target_ce_upper=up,
            retain_kl_weight=2.8, salun_keep_ratio=0.20, freeze_until_block=6,
            head_lr_mult=1.0, l2_weight=2e-3, aug_mode="moderate", seed=101,
        ))
    return cfgs


def _eval_row(name: str, sub: torch.nn.Module, orig: torch.nn.Module, f_x: torch.Tensor, ckpt: Path, extra: dict) -> dict:
    """Full 3,500-image evaluation row with window verdict and server-calibrated proxy score."""
    ev = evaluate_candidate(orig, sub, f_x, sub_path=ckpt)
    fr, rp = ev.forget_raw, ev.retain_proxy
    row = {
        "name": name, **extra,
        "ce_mean": fr.ce_mean, "ce_std": fr.ce_std, "ce_p50": fr.ce_p50,
        "frac_gt_95": fr.frac_gt_95, "proxy_kl_t2": rp.proxy_kl_t2,
        "absent_classes_kl_t2": rp.absent_classes_kl_t2, "delta_l2_total": rp.delta_l2_total,
        "calibrated_proxy": score_server_calibrated_proxy(
            fr.ce_mean, fr.ce_std, fr.ce_p50, rp.proxy_kl_t2, rp.absent_classes_kl_t2, rp.delta_l2_total,
        ),
        "valid": ev.checkpoint.valid,
    }
    row["mu_sigma"] = round(row["ce_mean"] + row["ce_std"], 3)
    row["in_window"] = _in_window(row)
    return row


def _print_row(worker_id: int, row: dict) -> None:
    print(
        f"[gpu{worker_id}] {time.strftime('%H:%M:%S')} {row['name']:36s} | "
        f"CE={row['ce_mean']:.3f}±{row['ce_std']:.2f} mu+s={row['mu_sigma']:.2f} p50={row['ce_p50']:.3f} "
        f">95%={row['frac_gt_95']:.2f} | ProxyKL={row['proxy_kl_t2']:.4f} AbsKL={row['absent_classes_kl_t2']:.4f} "
        f"dTheta={row['delta_l2_total']:.2f} | proxy={row['calibrated_proxy']:.4f} "
        f"{'WINDOW' if row['in_window'] else 'OUT'}",
        flush=True,
    )


def _load_train_data() -> tuple:
    """Load orig model, forget set, train proxy, and absent-class mask (shared conventions)."""
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    orig_f = predict(orig, f_x, batch_size=128)
    p_x = _build_train_proxy(f_x)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f.argmax(dim=-1), minlength=NUM_CLASSES) == 0
    return orig, f_x, orig_f, p_x, orig_p, absent_mask


def _run_grid_phase(args: argparse.Namespace, mine: list[MethodConfig]) -> int:
    orig, f_x, orig_f, p_x, orig_p, absent_mask = _load_train_data()
    for cfg in mine:
        print(f"[gpu{args.worker_id}] {time.strftime('%H:%M:%S')} train {cfg.name}", flush=True)
        sub = run_salun_npo(orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
        cdir = args.out_dir / cfg.name
        cdir.mkdir(parents=True, exist_ok=True)
        ckpt = cdir / "model.pt"
        torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, ckpt)
        row = _eval_row(cfg.name, sub, orig, f_x, ckpt, {"cfg": asdict(cfg)})
        (cdir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        _print_row(args.worker_id, row)
        if not row["in_window"] and cfg.loss_kind != "two_sided_npo":
            ckpt.unlink(missing_ok=True)  # quota: window candidates + diamond probes keep checkpoints
        del sub
        torch.cuda.empty_cache()
    return 0


def _load_grid_rows(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*/metrics.json"))]


def _rank_rows(rows: list[dict]) -> list[dict]:
    """In-window first, then ProxyKL ascending, then mu+sigma closeness to the 3.84 peak."""
    return sorted(rows, key=lambda r: (not r["in_window"], r["proxy_kl_t2"], abs(r["mu_sigma"] - 3.84)))


def _top_by_seed(rows: list[dict], seed: int, n: int) -> list[dict]:
    return [r for r in _rank_rows(rows) if r["cfg"]["seed"] == seed][:n]


def _best_ce(rows: list[dict], seed: int, ce: float) -> dict | None:
    for r in _rank_rows(rows):
        if r["cfg"]["seed"] == seed and abs(r["cfg"]["target_forget_ce"] - ce) < 1e-9:
            return r
    return None


def _build_soup_specs(rows: list[dict]) -> list[tuple[str, list[tuple[float, dict]]]]:
    """8 multi-seed convex soups: 4 same-rank 50/50 pairs, 2 cross-CE pairs, 2 three-way blends."""
    top42, top101 = _top_by_seed(rows, 42, 4), _top_by_seed(rows, 101, 4)
    specs = []
    for i in range(min(4, len(top42), len(top101))):
        pair = [(0.5, top42[i]), (0.5, top101[i])]
        specs.append((f"SOUP50_{top42[i]['name']}__{top101[i]['name']}", pair))
    for ce_a, ce_b in ((0.80, 1.00), (0.90, 1.00)):
        a, b = _best_ce(rows, 42, ce_a), _best_ce(rows, 101, ce_b)
        if a and b:
            specs.append((f"SOUPX_ce{ce_a:.2f}_s42__ce{ce_b:.2f}_s101", [(0.5, a), (0.5, b)]))
    if len(top42) >= 2 and top101:
        trip = (top42[0], top101[0], top42[1])
        specs.append(("SOUP3_" + "__".join(r["name"] for r in trip), [(1.0 / 3.0, r) for r in trip]))
    if len(top101) >= 2 and top42:
        trip = (top42[0], top101[0], top101[1])
        specs.append(("SOUP3_" + "__".join(r["name"] for r in trip), [(1.0 / 3.0, r) for r in trip]))
    return specs


def _run_soup_phase(args: argparse.Namespace, mine: list[tuple]) -> int:
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    for name, parts in mine:
        sds = [(w, torch.load(args.out_dir / r["name"] / "model.pt", map_location="cpu")) for w, r in parts]
        blended = blend_state_dicts(sds)
        sub = _model_from_sd(blended, torch.device(DEVICE))
        sdir = args.out_dir / "soups" / name
        sdir.mkdir(parents=True, exist_ok=True)
        ckpt = sdir / "model.pt"
        torch.save(blended, ckpt)
        extra = {"kind": "soup", "parts": [{"weight": w, "name": r["name"], "mu_sigma": r["mu_sigma"]} for w, r in parts]}
        row = _eval_row(name, sub, orig, f_x, ckpt, extra)
        (sdir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        _print_row(args.worker_id, row)
        if not row["in_window"]:
            ckpt.unlink(missing_ok=True)  # quota: keep checkpoints only for window candidates
        del sub, blended, sds
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="SweepMax: 52-config AT grid + multi-seed soups.")
    parser.add_argument("--phase", choices=("grid", "soup"), default="grid")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("output/sweep_grid"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "grid":
        mine = [c for i, c in enumerate(_build_grid()) if i % args.num_workers == args.worker_id]
        print(f"[gpu{args.worker_id}] grid phase: {len(mine)} configs", flush=True)
        return _run_grid_phase(args, mine)
    rows = [
        r for r in _load_grid_rows(args.out_dir)
        if (args.out_dir / r["name"] / "model.pt").exists()
        and r["cfg"].get("loss_kind") != "two_sided_npo"  # diamond probes are submissions, not soup ingredients
    ]
    specs = _build_soup_specs(rows)
    mine = [s for i, s in enumerate(specs) if i % args.num_workers == args.worker_id]
    print(f"[gpu{args.worker_id}] soup phase: {len(mine)} of {len(specs)} specs", flush=True)
    return _run_soup_phase(args, mine)


if __name__ == "__main__":
    raise SystemExit(main())
