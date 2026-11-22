"""Phase 3 two-view asymmetric grid: 8 two-view runs (view x ce x seed) + 2 mono controls.

Metrics schema is identical to sweep_grid (mu_sigma, ce_p50, frac_gt_95, proxy_kl_t2,
absent_classes_kl_t2, delta_l2_total, in_window, calibrated_proxy) plus the per-half
exact_altered_gap measured by run_per_half_diagnostic.py as a subprocess on each run.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from methods.phase3_views import run_phase3_two_view
from run_base_refine import _at_cfg
from sweep_grid import _eval_row, _load_train_data, _print_row

PERHALF_SCRIPT = Path(__file__).resolve().parent / "run_per_half_diagnostic.py"
HALF_COLS = ("ce_mean", "ce_std", "ce_p50", "frac_gt_95", "calibrated_proxy_forget_only")


def _build_phase3_specs() -> list[tuple]:
    """8 two-view configs (forget_view x ce x seed) + 2 mono-view controls at ce 0.90."""
    specs = []
    for view in ("clean", "aug"):
        for ce in (0.85, 0.95):
            for seed in (42, 101):
                name = f"P3_{view}_ce{ce:.2f}_s{seed}"
                specs.append((_at_cfg(name, 1e-4, ce, ce + 0.45, 2.0, 2.8, seed=seed), view, False))
    for view in ("clean", "aug"):
        name = f"P3mono_{view}_ce0.90_s42"
        specs.append((_at_cfg(name, 1e-4, 0.90, 1.35, 2.0, 2.8, seed=42), view, True))
    return specs


def _gap_fields(halves: dict | None) -> dict:
    """exact_altered_gap = CE(altered) - CE(exact); target asymmetry < 0.10."""
    if halves is None:
        return {"exact_altered_gap": None, "gap_below_010": None, "exact_half": None, "altered_half": None}
    exact, altered = halves["exact"], halves["altered"]
    gap = round(altered["ce_mean"] - exact["ce_mean"], 4)
    return {
        "exact_altered_gap": gap,
        "gap_below_010": abs(gap) < 0.10,
        "exact_half": {c: exact[c] for c in HALF_COLS},
        "altered_half": {c: altered[c] for c in HALF_COLS},
    }


def _run_perhalf_subprocess(ckpt: Path, out_json: Path) -> dict | None:
    """Run run_per_half_diagnostic.py on one checkpoint; return its two halves or None."""
    cmd = [sys.executable, "-u", str(PERHALF_SCRIPT), "--candidates", str(ckpt), "--output", str(out_json)]
    print(f"[perhalf] subprocess {ckpt.parent.name}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[perhalf] FAILED rc={proc.returncode}\n{proc.stderr[-800:]}", flush=True)
        return None
    report = json.loads(Path(out_json).read_text(encoding="utf-8"))
    return report["candidates"][str(ckpt)]


def _run_one(args: argparse.Namespace, orig, f_x, orig_f, p_x, orig_p, absent_mask, spec: tuple) -> int:
    """Train one spec, evaluate it (sweep_grid schema), then measure the per-half gap."""
    cfg, forget_view, mono = spec
    print(f"[gpu{args.worker_id}] {time.strftime('%H:%M:%S')} train {cfg.name}", flush=True)
    sub = run_phase3_two_view(orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg, forget_view, mono)
    cdir = args.out_dir / cfg.name
    cdir.mkdir(parents=True, exist_ok=True)
    ckpt = cdir / "model.pt"
    torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, ckpt)
    row = _eval_row(cfg.name, sub, orig, f_x, ckpt, {"cfg": asdict(cfg), "forget_view": forget_view, "mono_view": mono})
    del sub
    torch.cuda.empty_cache()
    halves = _run_perhalf_subprocess(ckpt, cdir / "perhalf.json")
    row.update(_gap_fields(halves))
    (cdir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    _print_row(args.worker_id, row)
    if row["exact_altered_gap"] is not None:
        print(f"[gpu{args.worker_id}] {cfg.name} exact_altered_gap={row['exact_altered_gap']:+.3f} (target abs < 0.10)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 two-view asymmetric grid (10 runs).")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("output/phase3_views"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_phase3_specs()
    mine = [s for i, s in enumerate(specs) if i % args.num_workers == args.worker_id]
    print(f"[gpu{args.worker_id}] phase3: {len(mine)} of {len(specs)} specs", flush=True)
    orig, f_x, orig_f, p_x, orig_p, absent_mask = _load_train_data()
    for spec in mine:
        _run_one(args, orig, f_x, orig_f, p_x, orig_p, absent_mask, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
