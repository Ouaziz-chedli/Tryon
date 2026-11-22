"""SoupLab: one-job combinatorial queue of 16 soups over the in-window sweep_grid runs.

Reads output/sweep_grid/SG*/metrics.json (sweep_grid.py schema: mu_sigma, ce_p50,
frac_gt_95, proxy_kl_t2, delta_l2_total, in_window, calibrated_proxy, valid) and
blends the in-window checkpoints with run_base_refine.blend_state_dicts:

  - 4x 50/50 pairs s42 x s101 for the in-window families
    (ce0.90_b2.0_kl2.8, ce1.00_b2.0_kl2.8, ce0.70_b2.0_kl2.8, ce0.70_b1.8_kl2.8)
  - 2x 3-way: top3 in-window per seed
  - 4x shrink toward orig: 2 best in-window runs x lambda in {0.85, 0.92}
  - cross-CE 50/50 ce0.90 x ce1.00 same seed, padded with ce0.70 x ce1.00 pairs
    so the queue reaches QUEUE_TARGET (16) soups
  - 2x 4-way: top4 in-window per seed

--phase pairs (no sweep_grid dependency, runs before the grid rerun) evaluates the
JurecaSweep off-grid recommendations in CUSTOM_PAIRS: PA2 x #14 (record-#13 recipe,
50/50) and PA1 x DIA_highstd (60/40 multi-seed), parts as cluster-side absolute
paths; --pairs-file <json> swaps in any custom pair list.

output/soup_lab/ranking.json is the submission-ready priority list (in-window first,
ProxyKL ascending, mu+sigma closest to the 3.84 server peak) for the orchestrator.

Torch/eval imports are deferred into the run phases so the queue builder and the
ranking phase stay unit-testable where torch is unavailable (local workstation).
"""

import argparse
import json
from pathlib import Path
import time

QUEUE_TARGET = 16
PAIR_FAMILIES = ((0.90, 2.0, 2.8), (1.00, 2.0, 2.8), (0.70, 2.0, 2.8), (0.70, 1.8, 2.8))
SHRINK_LAMBDAS = (0.85, 0.92)
SEEDS = (42, 101)
ORIG = "ORIG"
_SCRATCH = "/p/scratch/training2666/ouaziz1/TRYON/unlearning/output/diamond4"
# JurecaSweep final-diamond recommendations (journal 08h45): off-grid amplitude x low-ProxyKL blends.
# PA2 x #14 = exact recipe of record #13; PA1 x DIA_highstd = multi-seed placement near the 3.84 peak.
CUSTOM_PAIRS = (
    ("SL_custom_PA2xAT14_5050", "custom", (
        (0.5, f"{_SCRATCH}/PA2_AT_ce095_b20_k22_kl30_s123.pt"),
        (0.5, "$HOME/tryon_ws/pack/AT_ce0.80_b1.9_k20_kl28.pt"),
    )),
    ("SL_custom_PA1xDIAhigh_6040", "custom", (
        (0.6, f"{_SCRATCH}/PA1_AT_ce085_b17_k20_kl26_s77.pt"),
        (0.4, f"{_SCRATCH}/DIA_highstd_AT_ce120_b12_k20_kl26.pt"),
    )),
)


def _in_window(row: dict) -> bool:
    """Full server submission window (same bounds as sweep_grid._in_window)."""
    return (
        3.65 <= row["mu_sigma"] <= 4.00
        and 0.80 <= row["ce_p50"] <= 1.20
        and row["proxy_kl_t2"] <= 0.056
        and row["delta_l2_total"] <= 4.70
        and row["frac_gt_95"] <= 0.30
    )


def _rank_key(row: dict) -> tuple:
    """In-window first, then ProxyKL ascending, then mu+sigma closeness to the 3.84 peak."""
    return (not row["in_window"], row["proxy_kl_t2"], abs(row["mu_sigma"] - 3.84))


def _rank_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=_rank_key)


def _seed(row: dict) -> int:
    return row["cfg"]["seed"]


def _ce(row: dict) -> float:
    return round(row["cfg"]["target_forget_ce"], 2)


def _beta(row: dict) -> float:
    return round(row["cfg"]["npo_beta"], 1)


def _kl(row: dict) -> float:
    return round(row["cfg"]["retain_kl_weight"], 1)


def _fam_row(win: list[dict], seed: int, ce: float, beta: float, kl: float) -> dict | None:
    for r in win:
        if _seed(r) == seed and _ce(r) == ce and _beta(r) == beta and _kl(r) == kl:
            return r
    return None


def _pair_specs(win: list[dict]) -> list[tuple]:
    """4x 50/50 s42 x s101 soups, one per in-window family."""
    specs = []
    for ce, beta, kl in PAIR_FAMILIES:
        r42, r101 = _fam_row(win, 42, ce, beta, kl), _fam_row(win, 101, ce, beta, kl)
        if r42 and r101:
            name = f"SL_pair_ce{ce:.2f}_b{beta:.1f}_kl{kl:.1f}"
            specs.append((name, "pair", [(0.5, r42["name"]), (0.5, r101["name"])]))
    return specs


def _top_by_seed(ranked: list[dict], seed: int, n: int) -> list[dict]:
    return [r for r in ranked if _seed(r) == seed][:n]


def _threeway_specs(ranked: list[dict]) -> list[tuple]:
    """One 3-way soup per seed from its top3 in-window runs."""
    specs = []
    for seed in SEEDS:
        top3 = _top_by_seed(ranked, seed, 3)
        if len(top3) == 3:
            parts = [(round(1.0 / 3.0, 6), r["name"]) for r in top3]
            specs.append((f"SL_3way_s{seed}", "three_way", parts))
    return specs


def _shrink_specs(ranked: list[dict]) -> list[tuple]:
    """2 best in-window runs shrunk toward orig: theta = lam*run + (1-lam)*orig."""
    specs = []
    for run in ranked[:2]:
        for lam in SHRINK_LAMBDAS:
            name = f"SL_shrink_l{lam:.2f}_{run['name']}"
            specs.append((name, "shrink", [(lam, run["name"]), (round(1.0 - lam, 6), ORIG)]))
    return specs


def _crossce_pairs(win: list[dict], ce_a: float, ce_b: float) -> list[tuple[dict, dict]]:
    """All same-seed 50/50 pairs between in-window runs at ce_a and ce_b, best-first."""
    ranked = _rank_rows(win)
    pairs = []
    for seed in SEEDS:
        a_runs = [r for r in ranked if _ce(r) == ce_a and _seed(r) == seed]
        b_runs = [r for r in ranked if _ce(r) == ce_b and _seed(r) == seed]
        pairs += [(a, b) for a in a_runs for b in b_runs]
    return pairs


def _crossce_specs(win: list[dict], budget: int) -> list[tuple]:
    """Cross-CE 50/50 same-seed soups: ce0.90 x ce1.00 first, ce0.70 x ce1.00 as fill."""
    pool = _crossce_pairs(win, 0.90, 1.00) + _crossce_pairs(win, 0.70, 1.00)
    specs = []
    for a, b in pool[:budget]:
        name = f"SL_xce_{a['name'][3:]}__{b['name'][3:]}"
        specs.append((name, "cross_ce", [(0.5, a["name"]), (0.5, b["name"])]))
    return specs


def _fourway_specs(ranked: list[dict]) -> list[tuple]:
    """One 4-way soup per seed from its top4 in-window runs."""
    specs = []
    for seed in SEEDS:
        top4 = _top_by_seed(ranked, seed, 4)
        if len(top4) == 4:
            specs.append((f"SL_4way_s{seed}", "four_way", [(0.25, r["name"]) for r in top4]))
    return specs


def _build_queue(rows: list[dict]) -> list[tuple]:
    """16-soup queue: 4 pairs + 2 three-way + 4 shrink + cross-CE fill + 2 four-way."""
    win = _rank_rows([r for r in rows if r.get("in_window") and r.get("valid")])
    core = _pair_specs(win) + _threeway_specs(win) + _shrink_specs(win) + _fourway_specs(win)
    budget = max(0, QUEUE_TARGET - len(core))
    specs = core + _crossce_specs(win, budget)
    names = [s[0] for s in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate soup names in queue: {names}")
    return specs


def _load_rows(out_dir: Path, pattern: str) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob(pattern))]

def _soup_done(sdir: Path) -> bool:
    """Resume gate: metrics exist AND checkpoint is not a quota-crash stub (64-byte incident 05-22)."""
    ckpt = sdir / "model.pt"
    return (sdir / "metrics.json").exists() and ckpt.exists() and ckpt.stat().st_size > 1_000_000


def _write_soup_row(sdir: Path, row: dict, ckpt, blended) -> None:
    """Persist soup checkpoint + metrics.json with the sweep_grid row schema."""
    sdir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(blended, ckpt)
    (sdir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")


def _is_path_ref(ref: str) -> bool:
    """True when a soup part references a checkpoint file path instead of a run name."""
    return "/" in ref or ref.endswith(".pt")


def _resolve_path(ref: str) -> Path:
    """Expand cluster-side $HOME/~ prefixes in an absolute checkpoint path."""
    import os

    return Path(os.path.expandvars(os.path.expanduser(ref)))


def _load_part_sd(ref: str, cache: dict, sweep_dir: Path, model_path: Path):
    """CPU state dict for a soup part: SG run name, absolute path, or the ORIG sentinel."""
    if ref in cache:
        return cache[ref]
    import torch

    if ref == ORIG:
        path = model_path
    elif _is_path_ref(ref):
        path = _resolve_path(ref)
    else:
        path = sweep_dir / ref / "model.pt"
    if not (path.exists() and path.stat().st_size > 1_000_000):
        raise FileNotFoundError(f"soup part checkpoint missing or a stub: {path}")
    cache[ref] = torch.load(path, map_location="cpu", weights_only=True)
    return cache[ref]


def _eval_spec(spec: tuple, ctx: dict) -> dict | None:
    """Blend, save and evaluate one soup; returns None when already done (crash resume)."""
    import torch

    from run_base_refine import _model_from_sd, blend_state_dicts
    from sweep_grid import _eval_row, _print_row
    from task_template import DEVICE

    name, kind, parts = spec
    sdir = ctx["args"].out_dir / name
    if _soup_done(sdir):
        return json.loads((sdir / "metrics.json").read_text(encoding="utf-8"))
    sds = [(w, _load_part_sd(n, ctx["cache"], ctx["args"].sweep_dir, ctx["args"].model)) for w, n in parts]
    blended = blend_state_dicts(sds)
    sub = _model_from_sd(blended, torch.device(DEVICE))
    ckpt = sdir / "model.pt"
    extra = {"kind": kind, "parts": [{"weight": w, "name": n} for w, n in parts]}
    row = _eval_row(name, sub, ctx["orig"], ctx["f_x"], ckpt, extra)
    _write_soup_row(sdir, row, ckpt, blended)
    _print_row(ctx["args"].worker_id, row)
    del sub, blended, sds
    torch.cuda.empty_cache()
    return row


def _train_ctx(args: argparse.Namespace) -> dict:
    """Shared GPU context: orig model, forget set, and the soup-part checkpoint cache."""
    import numpy as np
    import torch
    from task_template import load_model

    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    return {"args": args, "cache": {}, "orig": load_model(args.model), "f_x": f_x}


def _soup_phase(args: argparse.Namespace) -> int:
    """Evaluate this worker's shard of the soup queue on one GPU."""
    from sweep_grid import _load_grid_rows

    rows = _load_grid_rows(args.sweep_dir)
    queue = _build_queue(rows)
    mine = [s for i, s in enumerate(queue) if i % args.num_workers == args.worker_id]
    print(f"[gpu{args.worker_id}] soup queue: {len(mine)} of {len(queue)} specs", flush=True)
    if not queue:
        print("NO_IN_WINDOW_RUNS: sweep_grid metrics have no valid in-window rows", flush=True)
        return 2
    ctx = _train_ctx(args)
    for spec in mine:
        _eval_spec(spec, ctx)
    return 0


def _pairs_specs(args: argparse.Namespace) -> list[tuple]:
    """Custom off-grid pairs: CUSTOM_PAIRS, or a JSON list passed via --pairs-file."""
    if args.pairs_file:
        raw = json.loads(args.pairs_file.read_text(encoding="utf-8"))
        return [(r["name"], "custom", [(p["weight"], p["path"]) for p in r["parts"]]) for r in raw]
    return [(name, kind, list(parts)) for name, kind, parts in CUSTOM_PAIRS]


def _pairs_phase(args: argparse.Namespace) -> int:
    """Evaluate the custom off-grid pairs (no sweep_grid dependency: runs before the grid rerun)."""
    specs = _pairs_specs(args)
    print(f"[gpu{args.worker_id}] custom pairs: {len(specs)}", flush=True)
    if not specs:
        print("NO_CUSTOM_PAIRS: empty pair list", flush=True)
        return 2
    ctx = _train_ctx(args)
    for spec in specs:
        _eval_spec(spec, ctx)
    return 0


def _ranking_payloads(ranked: list[dict], out_dir: Path) -> list[dict]:
    return [
        {
            "rank": i, "name": r["name"], "kind": r.get("kind", "soup"),
            "in_window": r["in_window"], "mu_sigma": r["mu_sigma"], "ce_p50": r["ce_p50"],
            "frac_gt_95": r["frac_gt_95"], "proxy_kl_t2": r["proxy_kl_t2"],
            "delta_l2_total": r["delta_l2_total"], "calibrated_proxy": r["calibrated_proxy"],
            "valid": r.get("valid", False), "ckpt": str(out_dir / r["name"] / "model.pt"),
        }
        for i, r in enumerate(ranked, start=1)
    ]


def _rank_phase(args: argparse.Namespace) -> int:
    """Rewrite ranking.json from all evaluated soups and print the submission order."""
    rows = _load_rows(args.out_dir, "SL*/metrics.json")
    if not rows:
        print("NO_SOUP_ROWS: nothing evaluated in output/soup_lab yet", flush=True)
        return 2
    ranked = _rank_rows(rows)
    payload = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "queue": _ranking_payloads(ranked, args.out_dir)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ranking.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for p in payload["queue"]:
        flag = "SUBMIT" if p["in_window"] else "  --  "
        print(
            f"#{p['rank']:02d} {flag} {p['name']} mu+s={p['mu_sigma']:.2f} p50={p['ce_p50']:.3f} "
            f"ProxyKL={p['proxy_kl_t2']:.4f} dTheta={p['delta_l2_total']:.2f} "
            f"proxy={p['calibrated_proxy']:.4f} -> {p['ckpt']}",
            flush=True,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SoupLab: 16-soup combinatorics over in-window sweep_grid runs.")
    parser.add_argument("--phase", choices=("soup", "pairs", "rank"), default="soup")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--sweep-dir", type=Path, default=Path("output/sweep_grid"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/soup_lab"))
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--pairs-file", type=Path, default=None)
    args = parser.parse_args()
    if args.phase == "rank":
        return _rank_phase(args)
    if args.phase == "pairs":
        return _pairs_phase(args)
    return _soup_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
