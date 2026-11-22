"""Runner for escanade-guided weight surgery: 8 no-train configs on one moderate candidate.

Standard row mirrors sweep_grid._eval_row (mu+sigma, p50, frac_gt_95, ProxyKL, dTheta,
calibrated proxy, window verdict); forget_raw is taken from the exact half so the altered
stress suite is never scored twice. The exact/altered split follows
run_per_half_diagnostic. Checkpoints are kept only for in-window configs (quota rule).
"""

import argparse
from dataclasses import asdict, dataclass
import json
import time
from pathlib import Path

import torch

from eval_harness import (
    _build_eval_proxy,
    _compute_retain_proxy_metrics,
    score_server_calibrated_proxy,
    validate_checkpoint,
)
from methods.head_surgery import (
    build_surgery_grid,
    extract_erase_directions,
    run_head_surgery,
)
from run_per_half_diagnostic import (
    build_halves,
    collect_orig_logits,
    evaluate_half,
    predict_with_heartbeat,
)
from task_template import (
    DEVICE,
    NUM_CLASSES,
    build_model,
    load_forget_images,
    load_model,
    predict,
)

DEFAULT_EXTREME = Path("output/track_i/I2R_f6_r4_ce18_fl085.pt")
DEFAULT_CANDIDATE = Path("output/sweep_grid/SG_ce0.90_b2.0_kl2.8_s101")
STRESS_SEED = 999  # same suite as eval_harness.evaluate_candidate / run_per_half defaults


@dataclass
class SurgeryContext:
    """Per-run shared caches: orig model, halves, orig logits, retain-proxy inputs."""

    orig_model: torch.nn.Module
    halves: dict[str, torch.Tensor]
    orig_logits: dict[str, torch.Tensor]
    proxy_imgs: torch.Tensor
    orig_proxy: torch.Tensor
    active_mask: torch.Tensor
    batch_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Escanade-guided weight surgery on a moderate candidate.")
    parser.add_argument("--model", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget-images", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--extreme", type=Path, default=DEFAULT_EXTREME)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--out-dir", type=Path, default=Path("output/surgery"))
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def load_state_dict_cpu(path: Path) -> dict[str, torch.Tensor]:
    """Strict-repo state dict on CPU (same call as eval_harness.validate_checkpoint)."""
    return torch.load(path, map_location="cpu", weights_only=True)


def resolve_checkpoint(path: Path) -> Path:
    """Accept both sweep layouts: dir with model.pt, or a flat <name>.pt mirror."""
    if path.is_dir():
        return path / "model.pt"
    if path.exists():
        return path
    flat = path.with_suffix(".pt")
    if flat.exists():
        return flat
    raise FileNotFoundError(f"candidate checkpoint not found: {path} (or {flat})")


def in_window(row: dict) -> bool:
    """Mirror of sweep_grid._in_window: the shared server submission window."""
    return (
        3.65 <= row["mu_sigma"] <= 4.00
        and 0.80 <= row["ce_p50"] <= 1.20
        and row["proxy_kl_t2"] <= 0.056
        and row["delta_l2_total"] <= 4.70
        and row["frac_gt_95"] <= 0.30
    )


@torch.no_grad()
def compute_active_mask(orig_model: torch.nn.Module, forget_images: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Active classes = classes orig predicts on the forget set (eval_harness convention)."""
    preds = predict(orig_model, forget_images, batch_size=batch_size).argmax(dim=-1)
    return torch.bincount(preds, minlength=NUM_CLASSES) > 0


def build_model_from_sd(sd: dict[str, torch.Tensor]) -> torch.nn.Module:
    """Instantiate the strict ViT architecture and load the surgical state dict."""
    model = build_model()
    model.load_state_dict(sd, strict=True)
    return model.to(DEVICE).eval()


def build_row(cfg_name: str, cfg_dict: dict, exact: dict, retain, valid: bool) -> dict:
    """Standard sweep-style row from the exact-half metrics + retain proxy + window verdict."""
    row = {
        "name": cfg_name, "cfg": cfg_dict,
        "ce_mean": exact["ce_mean"], "ce_std": exact["ce_std"], "ce_p50": exact["ce_p50"],
        "frac_gt_95": exact["frac_gt_95"],
        "proxy_kl_t2": retain.proxy_kl_t2,
        "absent_classes_kl_t2": retain.absent_classes_kl_t2,
        "delta_l2_total": retain.delta_l2_total,
        "calibrated_proxy": score_server_calibrated_proxy(
            exact["ce_mean"], exact["ce_std"], exact["ce_p50"],
            retain.proxy_kl_t2, retain.absent_classes_kl_t2, retain.delta_l2_total,
        ),
        "valid": valid,
        "guard_violations": [],
    }
    row["mu_sigma"] = round(row["ce_mean"] + row["ce_std"], 3)
    row["in_window"] = in_window(row)
    return row


def evaluate_halves(sub_model: torch.nn.Module, ctx: SurgeryContext) -> dict[str, dict]:
    """Per-half exact/altered metrics for one surgical model (run_per_half scheme)."""
    per_half = {}
    for half, images in ctx.halves.items():
        sub_logits = predict_with_heartbeat(sub_model, images, ctx.batch_size)
        per_half[half] = evaluate_half(ctx.orig_logits[half], sub_logits, ctx.orig_logits[half].argmax(dim=-1))
    return per_half


def print_row(row: dict) -> None:
    """One sweep_grid-style line plus the per-half forget-only calibrated proxies."""
    exact = row["per_half"]["exact"]["calibrated_proxy_forget_only"]
    altered = row["per_half"]["altered"]["calibrated_proxy_forget_only"]
    print(
        f"[surgery] {time.strftime('%H:%M:%S')} {row['name']:16s} | "
        f"CE={row['ce_mean']:.3f}±{row['ce_std']:.2f} mu+s={row['mu_sigma']:.2f} "
        f"p50={row['ce_p50']:.3f} >95%={row['frac_gt_95']:.2f} | "
        f"ProxyKL={row['proxy_kl_t2']:.4f} AbsKL={row['absent_classes_kl_t2']:.4f} "
        f"dTheta={row['delta_l2_total']:.2f} | proxy={row['calibrated_proxy']:.4f} "
        f"half(exact/alt)={exact:.3f}/{altered:.3f} "
        f"{'WINDOW' if row['in_window'] else 'OUT'}",
        flush=True,
    )


def run_config(cfg, candidate_sd, directions, ctx: SurgeryContext, out_dir: Path) -> dict:
    """Surgery + save + standard row + per-half for one config; keep ckpt only if in window."""
    sd, violations = run_head_surgery(candidate_sd, directions, cfg, ctx.active_mask)
    if violations:
        print(f"[surgery] {cfg.name}: GUARD VIOLATIONS {violations}; config skipped", flush=True)
        return {"name": cfg.name, "cfg": asdict(cfg), "guard_violations": violations}
    ckpt_dir = out_dir / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / "model.pt"
    torch.save(sd, ckpt)
    valid = validate_checkpoint(ckpt).valid
    sub_model = build_model_from_sd(sd)
    per_half = evaluate_halves(sub_model, ctx)
    sub_proxy = predict(sub_model, ctx.proxy_imgs, batch_size=ctx.batch_size)
    retain = _compute_retain_proxy_metrics(
        ctx.orig_model, sub_model, ctx.orig_proxy, sub_proxy, ~ctx.active_mask
    )
    del sub_model
    torch.cuda.empty_cache()
    row = build_row(cfg.name, asdict(cfg), per_half["exact"], retain, valid)
    row["per_half"] = per_half
    (ckpt_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print_row(row)
    if not row["in_window"]:
        ckpt.unlink(missing_ok=True)  # quota: keep checkpoints only for window candidates
    return row


def build_context(args: argparse.Namespace, orig_model, forget_images: torch.Tensor) -> SurgeryContext:
    """Cache everything shared across the 8 configs (halves, orig logits, proxy inputs)."""
    halves, _ = build_halves(forget_images, STRESS_SEED)
    orig_logits = collect_orig_logits(orig_model, halves, args.batch_size)
    active_mask = compute_active_mask(orig_model, forget_images, args.batch_size)
    proxy_imgs = _build_eval_proxy(forget_images, seed=STRESS_SEED)
    orig_proxy = predict(orig_model, proxy_imgs, batch_size=args.batch_size)
    return SurgeryContext(
        orig_model, halves, orig_logits, proxy_imgs, orig_proxy, active_mask, args.batch_size
    )


def log_erase_directions(directions: dict[str, torch.Tensor]) -> None:
    """Print the l2 of each transplanted direction (dose-free) before the first config."""
    for key, direction in directions.items():
        print(f"[surgery] erase direction {key}: l2={float(direction.norm()):.4f}", flush=True)


def load_surgery_inputs(args: argparse.Namespace, active_mask: torch.Tensor) -> tuple[dict, dict]:
    """Validate the candidate strict; extract rank-1 erase directions from orig vs extreme."""
    candidate_sd = load_state_dict_cpu(resolve_checkpoint(args.candidate))
    build_model().load_state_dict(candidate_sd, strict=True)  # fail fast on a corrupt candidate
    orig_sd = load_state_dict_cpu(args.model)
    extreme_sd = load_state_dict_cpu(args.extreme)
    directions = extract_erase_directions(orig_sd, extreme_sd, active_mask)
    return candidate_sd, directions


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig_model = load_model(args.model)
    forget_images = load_forget_images(args.forget_images)
    ctx = build_context(args, orig_model, forget_images)
    print(
        f"[surgery] active classes: {int(ctx.active_mask.sum())}/{NUM_CLASSES} "
        f"candidate={resolve_checkpoint(args.candidate)} extreme={args.extreme}",
        flush=True,
    )
    candidate_sd, directions = load_surgery_inputs(args, ctx.active_mask)
    log_erase_directions(directions)

    rows = [
        run_config(cfg, candidate_sd, directions, ctx, args.out_dir)
        for cfg in build_surgery_grid()
    ]
    report = {
        "candidate": str(args.candidate),
        "extreme": str(args.extreme),
        "stress_seed": STRESS_SEED,
        "configs": rows,
    }
    out_path = args.out_dir / "head_surgery_results.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[surgery] report written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
