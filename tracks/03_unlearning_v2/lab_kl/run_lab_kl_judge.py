"""Lab-KL local judge (P4.1): exact server formulas against gold_lab + MANDATORY ranking validation.

Every candidate checkpoint is scored with lab/server_metrics.py (the exact Track 3 formulas,
KL at T=2 included) using (orig_lab, gold_lab) as the reference pair. The judge is then
validated by Spearman correlation against the 11 known server scores; it is USABLE only
if the best oriented rho >= 0.8 (validation gate before any tuning on it).
"""

import argparse
from dataclasses import asdict
import json
import math
import sys
from pathlib import Path

import torch

HARNESS = Path(__file__).resolve().parents[1]
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from eval_harness import compute_kl_at_temperature, make_patchmix_proxy  # noqa: E402
from lab.server_metrics import evaluate_against_gold  # noqa: E402
from task_template import load_forget_images, load_model, predict  # noqa: E402

# The 11 checkpoint <-> server score pairs (journal 05h15, used by the VGT falsification).
SERVER_SCORES = {
    "SOUP_50_050_50_100": 0.1238,
    "AT_ce0.80_b1.9_k20_kl28": 0.1197,
    "AT_ce1.00_b1.8_f6": 0.1193,
    "SOUP_50_ce80kl28_50_ce90kl28": 0.1184,
    "SOUP3_75_AT9_25_AT2": 0.1176,
    "AT_ce0.50_b2.0_f6": 0.1099,
    "SHRINK_88_AT9": 0.1082,
    "H4_h123_f6_ce125": 0.0984,
    "I2R_f6_r4_ce18_fl085": 0.0950,
    "H1_h1h2_f8_ce125": 0.0813,
    "AT_ce0.50_b3.0_f8": 0.0604,
}

RANK_METRICS = ("final_score", "forget_score", "prediction_score", "mean_std_score", "retain_score", "kl_gold_sub_t2")
VALIDATION_RHO = 0.8


def rank_average_ties(values: list[float]) -> list[float]:
    """Average ranks with ties handled (shared rank for equal values)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho of two equal-length lists; NaN if either is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx, ry = rank_average_ties(xs), rank_average_ties(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def build_control_proxy(images: torch.Tensor, seed: int) -> torch.Tensor:
    """Out-of-sample control/val proxy (4x4 + 7x7 PatchMix); seeds 777/999 are disjoint from gold_lab's 42/43."""
    half = len(images) // 2
    return torch.cat([
        make_patchmix_proxy(images[:half], seed=seed, grid=4),
        make_patchmix_proxy(images[half:], seed=seed + 1, grid=7),
    ], dim=0)


def compute_raw_kls(gold_f: torch.Tensor, orig_f: torch.Tensor, sub_f: torch.Tensor) -> dict:
    """Unclipped KL diagnostics at T=2: the clip is what saturates, raw values must stay visible."""
    return {
        "kl_gold_sub_t2": round(float(compute_kl_at_temperature(gold_f, sub_f, 2.0).mean()), 4),
        "kl_gold_orig_t2": round(float(compute_kl_at_temperature(gold_f, orig_f, 2.0).mean()), 4),
    }


def judge_candidate(orig_lab, gold_lab, sub_path, forget, forget_labels, control, control_labels,
                    val, val_labels, orig_f, gold_f, batch_size) -> dict:
    """Score one candidate with the exact server formulas; per-candidate failure is recorded, not fatal."""
    try:
        sub = load_model(sub_path)
        sub_f = predict(sub, forget, batch_size=batch_size)
        report = asdict(evaluate_against_gold(
            orig_lab, gold_lab, sub, forget, forget_labels,
            control, control_labels, val, val_labels))
        return {**report, **compute_raw_kls(gold_f, orig_f, sub_f)}
    except Exception as exc:  # keep ranking the remaining candidates
        return {"error": repr(exc)}


def compute_ranking_rhos(results: dict) -> dict:
    """Spearman(lab metric, server score) over the checkpoints with a known server score."""
    stems = [s for s in SERVER_SCORES if s in results and results[s].get("error") is None]
    if len(stems) < 3:
        return {"error": f"only {len(stems)} judged checkpoints have a known server score"}
    rhos = {}
    for metric in RANK_METRICS:
        lab = [results[s][metric] for s in stems]
        srv = [SERVER_SCORES[s] for s in stems]
        rhos[metric] = round(spearman(lab, srv), 4)
    return rhos


def best_rank_signal(rhos: dict) -> tuple[str, float]:
    """Best usable ranking signal; raw KL is oriented negatively (lower KL = better candidate)."""
    oriented = {
        m: (-rho if m == "kl_gold_sub_t2" else rho)
        for m, rho in rhos.items()
        if isinstance(rho, float)
    }
    if not oriented:
        return "none", float("nan")
    return max(oriented.items(), key=lambda kv: kv[1])


def render_table(results: dict) -> str:
    """One line per known-score candidate, sorted by server score, lab metrics side by side."""
    stems = [s for s in SERVER_SCORES if s in results and results[s].get("error") is None]
    lines = []
    for s in sorted(stems, key=lambda s: -SERVER_SCORES[s]):
        cells = "  ".join(f"{m}={results[s].get(m, float('nan')):.4f}" for m in RANK_METRICS)
        lines.append(f"  {s:<36} server={SERVER_SCORES[s]:.4f}  {cells}")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig-lab", type=Path, required=True, help="output/lab_kl/orig_lab.pt")
    parser.add_argument("--gold-lab", type=Path, required=True, help="output/lab_kl/gold_lab.pt")
    parser.add_argument("--images", type=Path, required=True, help="forget_images.npz")
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True, help="candidate checkpoints to judge")
    parser.add_argument("--output", type=Path, default=Path("output/lab_kl/lab_kl_judge_report.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def judge_all(orig_lab, gold_lab, images, orig_f, gold_f, ckpts, batch_size) -> dict:
    """Judge every candidate checkpoint; returns {stem: lab report dict}."""
    forget_labels = orig_f.argmax(dim=-1)
    control = build_control_proxy(images, 999)
    control_labels = predict(orig_lab, control, batch_size=batch_size).argmax(dim=-1)
    val = build_control_proxy(images, 777)
    val_labels = predict(orig_lab, val, batch_size=batch_size).argmax(dim=-1)
    results = {}
    for path in sorted(ckpts):
        results[path.stem] = judge_candidate(
            orig_lab, gold_lab, path, images, forget_labels,
            control, control_labels, val, val_labels, orig_f, gold_f, batch_size)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"JUDGED {path.stem}", flush=True)
    return results


def build_report(results: dict, num_images: int) -> dict:
    """Assemble the judge report: candidates, Spearman vs server, verdict."""
    rhos = compute_ranking_rhos(results)
    best_name, best_rho = best_rank_signal(rhos)
    verdict = "LAB VALID (usable as local judge)" if best_rho >= VALIDATION_RHO else "LAB INVALID (do not tune on it)"
    return {
        "protocol": {"forget_images": num_images, "temperature": 2.0, "utility_gate": 0.70,
                      "control_proxy": "patchmix 4x4+7x7 seed 999", "val_proxy": "patchmix 4x4+7x7 seed 777",
                      "interpretation": "Ranking-only lab judge; validate against server scores before use."},
        "candidates": results,
        "spearman_vs_server": rhos,
        "best_signal": {"metric": best_name, "rho_oriented": None if math.isnan(best_rho) else round(best_rho, 4)},
        "verdict": verdict,
    }


def main() -> int:
    args = parse_args()
    if not args.ckpts:
        raise SystemExit("no candidate checkpoints given")
    orig_lab = load_model(args.orig_lab)
    gold_lab = load_model(args.gold_lab)
    images = load_forget_images(args.images)
    orig_f = predict(orig_lab, images, batch_size=args.batch_size)
    gold_f = predict(gold_lab, images, batch_size=args.batch_size)
    results = judge_all(orig_lab, gold_lab, images, orig_f, gold_f, args.ckpts, args.batch_size)
    report = build_report(results, len(images))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + render_table(results), flush=True)
    for m, rho in sorted(report["spearman_vs_server"].items(),
                         key=lambda kv: -abs(kv[1]) if isinstance(kv[1], float) else 0):
        print(f"  spearman({m}, server) = {rho}", flush=True)
    print(f"LABKL_VERDICT: {report['verdict']} metric={report['best_signal']['metric']} "
          f"rho={report['best_signal']['rho_oriented']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
