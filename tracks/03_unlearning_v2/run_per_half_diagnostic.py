"""Per-half server diagnostic: measure the exact vs altered halves of the 50/50 server average separately.

The Track 3 server averages the score over raw forget images and hidden altered copies
(50/50). This script evaluates each half independently for a list of candidate
checkpoints, on forget_images.npz raw (half "exact") vs the existing eval_harness
stress suite concatenated (half "altered").
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from eval_harness import (
    compute_kl_at_temperature,
    compute_stress_metrics,
    make_stress_suite,
    score_server_calibrated_proxy,
)
from task_template import DEVICE, load_forget_images, load_model, preprocess

HEARTBEAT_SECONDS = 15.0

DEFAULT_CANDIDATES = [
    "output/base_refine/SOUP_50_050_50_100.pt",
    "output/base_refine/SOUP3_75_AT9_25_AT2.pt",
    "output/base_refine/SOUP3_50_AT9_50_s101.pt",
    "output/base_refine/AT_ce0.80_b1.9_k20_kl28.pt",
    "output/track_i/I3R_4_lora_r8_c095_160.pt",
]

METRIC_COLS = [
    "ce_mean",
    "ce_std",
    "ce_p50",
    "frac_gt_95",
    "kl_t2_from_orig",
    "kl_t2_p90",
    "entropy_mean",
    "top1_agree_orig",
    "calibrated_proxy_forget_only",
]


@torch.no_grad()
def predict_with_heartbeat(
    model: torch.nn.Module,
    images_uint8: torch.Tensor,
    batch_size: int = 128,
) -> torch.Tensor:
    """predict() with a '.' heartbeat every ~15 s so long L4 runs show liveness."""
    model.eval()
    out, last, beat = [], time.monotonic(), False
    for i in range(0, len(images_uint8), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(preprocess(images_uint8[i:i + batch_size])).float().cpu())
        if time.monotonic() - last >= HEARTBEAT_SECONDS:
            print(".", end="", flush=True)
            beat, last = True, time.monotonic()
    if beat:
        print(flush=True)
    return torch.cat(out)


def build_halves(
    forget_images: torch.Tensor, stress_seed: int
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return ({half_name: images}, perturbation names); altered = concatenated stress suite."""
    suite = make_stress_suite(forget_images, seed=stress_seed)
    halves = {"exact": forget_images, "altered": torch.cat(list(suite.values()), dim=0)}
    return halves, list(suite.keys())


def evaluate_half(orig_logits: torch.Tensor, sub_logits: torch.Tensor, ref_preds: torch.Tensor) -> dict:
    """Per-half metrics reusing compute_stress_metrics + per-image KL p90 + calibrated proxy.

    shortcut: calibrated_proxy_forget_only neutralizes retain/weight-drift terms
    (they are not half-specific); use the full eval_harness for the gated score.
    """
    metrics = asdict(compute_stress_metrics(orig_logits, sub_logits, ref_preds))
    kl = compute_kl_at_temperature(orig_logits, sub_logits.float())
    metrics["kl_t2_p90"] = round(float(torch.quantile(kl, 0.9)), 4)
    metrics["calibrated_proxy_forget_only"] = score_server_calibrated_proxy(
        metrics["ce_mean"], metrics["ce_std"], metrics["ce_p50"], 0.0, 0.0, 0.0
    )
    return metrics


def format_half_summary(tag: str, half: str, m: dict) -> str:
    """One readable line per candidate half, printed as soon as it is computed."""
    return (
        f"[perhalf] {tag} {half:7s} | CE={m['ce_mean']:.2f}±{m['ce_std']:.2f} p50={m['ce_p50']:.3f} "
        f">95%={m['frac_gt_95']:.2f} | KL={m['kl_t2_from_orig']:.4f} p90={m['kl_t2_p90']:.3f} "
        f"| H={m['entropy_mean']:.3f} agree={m['top1_agree_orig']:.3f} "
        f"proxy={m['calibrated_proxy_forget_only']:.4f}"
    )


def evaluate_one_candidate(
    sub_path: Path,
    halves: dict[str, torch.Tensor],
    orig_logits: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, dict]:
    """Load one candidate checkpoint and evaluate it on both halves."""
    tag = sub_path.stem
    sub_model = load_model(sub_path)
    result = {}
    for half, images in halves.items():
        print(f"[perhalf] {tag} {half} ({len(images)} imgs) ", end="", flush=True)
        sub_logits = predict_with_heartbeat(sub_model, images, batch_size)
        # Pseudo-labels PER HALF: the altered half (28k views) must not reuse the
        # exact half's 3500 preds (batch-size mismatch crash, found on track3_h).
        result[half] = evaluate_half(orig_logits[half], sub_logits, orig_logits[half].argmax(dim=-1))
        print(format_half_summary(tag, half, result[half]), flush=True)
    return result


def collect_orig_logits(
    orig_model: torch.nn.Module, halves: dict[str, torch.Tensor], batch_size: int
) -> dict[str, torch.Tensor]:
    """Cache orig logits once per half, shared across all candidates."""
    orig_logits = {}
    for half, images in halves.items():
        print(f"[perhalf] orig {half} ({len(images)} imgs) ", end="", flush=True)
        orig_logits[half] = predict_with_heartbeat(orig_model, images, batch_size)
    return orig_logits


def run_all_candidates(
    candidate_paths: list[Path],
    halves: dict[str, torch.Tensor],
    orig_logits: dict[str, torch.Tensor],
    batch_size: int,
) -> tuple[dict[str, dict], list[str]]:
    """Evaluate every existing candidate; collect missing paths instead of failing."""
    results, missing = {}, []
    for sub_path in candidate_paths:
        if not sub_path.exists():
            print(f"[perhalf] MISSING {sub_path}, skipping", flush=True)
            missing.append(str(sub_path))
            continue
        results[str(sub_path)] = evaluate_one_candidate(
            sub_path, halves, orig_logits, batch_size
        )
    return results, missing


def print_markdown_table(results: dict[str, dict]) -> None:
    """Print the candidate x half x metrics markdown table."""
    header = "| candidate | half | " + " | ".join(METRIC_COLS) + " |"
    print("\n" + header, flush=True)
    print("|" + "---|" * (len(METRIC_COLS) + 2), flush=True)
    for path, halves in results.items():
        for half, m in halves.items():
            row = " | ".join(str(m[c]) for c in METRIC_COLS)
            print(f"| {path} | {half} | {row} |", flush=True)


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the exact vs altered halves of the 50/50 server average separately."
    )
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=Path("output/diag_perhalf/perhalf_report.json"))
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget-images", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stress-seed", type=int, default=999)
    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()
    candidate_paths = [Path(p) for p in args.candidates]

    orig_model = load_model(args.model)
    forget_images = load_forget_images(args.forget_images)
    halves, perturbation_names = build_halves(forget_images, args.stress_seed)

    orig_logits = collect_orig_logits(orig_model, halves, args.batch_size)

    results, missing = run_all_candidates(
        candidate_paths, halves, orig_logits, args.batch_size
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_forget_images": len(forget_images),
        "stress_seed": args.stress_seed,
        "altered_perturbations": perturbation_names,
        "candidates": results,
        "missing": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[perhalf] report written to {args.output}", flush=True)

    print_markdown_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
