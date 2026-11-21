"""M1: Zero-training weight interpolation and checkpoint soup over bounded_sweep."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from eval_harness import evaluate_candidate
from task_template import build_model, load_model


def _blend_state_dicts(
    sd_orig: dict[str, torch.Tensor],
    deltas: list[tuple[float, dict[str, torch.Tensor]]],
    head_mult: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute sd_orig + sum(weight * delta), scaling head.* by head_mult."""
    out: dict[str, torch.Tensor] = {}
    for k, v0 in sd_orig.items():
        acc = torch.zeros_like(v0, dtype=torch.float32)
        mult = head_mult if k.startswith("head.") else 1.0
        for w, sd_u in deltas:
            acc.add_(sd_u[k].float() - v0.float(), alpha=w * mult)
        out[k] = (v0.float() + acc).to(torch.float32)
    return out


def _linear_ms_score(ce_mean: float, ce_std: float, mu_g: float, sig_g: float, d_orig: float) -> float:
    d_sub = abs(ce_mean - mu_g) + abs(ce_std - sig_g)
    return 1.0 - min(1.0, max(0.0, d_sub / max(d_orig, 1e-6)))


def _surrogate_min_grid(ev: object, proxy_kl: float, max_sum: float) -> float:
    """Compute worst-case official-PDF surrogate averaged 50/50 over raw and non-grayscale altered."""
    fr = ev.forget_raw
    alt = [m for k, m in ev.forget_stress.items() if k != "grayscale"]
    alt_mean = sum(m.ce_mean for m in alt) / max(len(alt), 1)
    alt_std = sum(m.ce_std for m in alt) / max(len(alt), 1)
    worst = 1e9
    for mu_g in (0.4, 0.7, 1.0, 1.3):
        for sig_g in (1.0, 1.4, 1.8, 2.2):
            if mu_g + sig_g > max_sum + 0.15:
                continue
            ms_raw = _linear_ms_score(fr.ce_mean, fr.ce_std, mu_g, sig_g, mu_g + sig_g)
            ms_alt = _linear_ms_score(alt_mean, alt_std, mu_g + 0.3, sig_g + 0.2, max(0.8, mu_g + sig_g - 0.5))
            s_hat = 0.5 * (ms_raw + ms_alt) / (1.0 + proxy_kl)
            worst = min(worst, s_hat)
    return worst if worst < 1e8 else 0.0


def _m1_specs(sweep_dir: Path) -> list[tuple[str, list[tuple[float, Path]], float]]:
    """Define M1 interpolation and soup candidates from job 15903458 checkpoints."""
    c5 = sweep_dir / "B_ce0.45_lr8e-05_kl2.0.pt"
    c65 = sweep_dir / "B_ce0.65_lr8e-05_kl2.0.pt"
    c85 = sweep_dir / "B_ce0.85_lr8e-05_kl2.0.pt"
    c10 = sweep_dir / "B_ce0.45_lr1e-04_kl2.5.pt"
    specs: list[tuple[str, list[tuple[float, Path]], float]] = [
        (f"interp_lam{lam:.2f}", [(lam, c5)], 1.0) for lam in (0.55, 0.65, 0.75, 0.85, 0.92)
    ]
    specs.append(("interp_blk0.65_hd0.85", [(0.65, c5)], 0.85 / 0.65))
    top4 = [c5, c65, c85, c10]
    for lam in (0.65, 0.75, 0.85):
        specs.append((f"soup4_lam{lam:.2f}", [(lam / 4.0, p) for p in top4], 1.0))
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate M1 interpolations and soups.")
    parser.add_argument("--sweep-dir", type=Path, default=Path("output/bounded_sweep"))
    parser.add_argument("--probe-bound", type=float, default=3.5, help="-ln(S0) upper bound on mu_g + sigma_g")
    args = parser.parse_args()

    orig = load_model("model.pt")
    sd_orig = {k: v.detach().cpu().float() for k, v in orig.state_dict().items()}
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    cache: dict[Path, dict[str, torch.Tensor]] = {}
    rows, best_score, best_sd = [], -1.0, None

    for name, pairs, h_mult in _m1_specs(args.sweep_dir):
        if any(not p.is_file() for _, p in pairs):
            continue
        deltas = [(w, cache.setdefault(p, torch.load(p, map_location="cpu", weights_only=True))) for w, p in pairs]
        sd = _blend_state_dicts(sd_orig, deltas, head_mult=h_mult)
        tmp_path = args.sweep_dir / f"M1_{name}.pt"
        torch.save(sd, tmp_path)
        sub = build_model()
        sub.load_state_dict(sd, strict=True)
        ev = evaluate_candidate(orig, sub.to(next(orig.parameters()).device).eval(), f_x, sub_path=tmp_path)
        fr, rp = ev.forget_raw, ev.retain_proxy
        s_hat = _surrogate_min_grid(ev, rp.proxy_kl_t2, args.probe_bound)
        row = {"name": name, "s_hat": round(s_hat, 4), **asdict(fr), **asdict(rp), "valid": ev.checkpoint.valid}
        rows.append(row)
        print(f"{name:22s} | Shat={s_hat:.4f} | CE={fr.ce_mean:.3f}±{fr.ce_std:.2f} p50={fr.ce_p50:.3f} >95%={fr.frac_gt_95:.2f} | ProxyKL={rp.proxy_kl_t2:.4f} AbsKL={rp.absent_classes_kl_t2:.4f} dTheta={rp.delta_l2_total:.2f}", flush=True)
        if ev.checkpoint.valid and s_hat > best_score:
            best_score, best_sd = s_hat, sd

    if best_sd is not None:
        torch.save(best_sd, args.sweep_dir / "M1_best.pt")
    rows.sort(key=lambda r: r["s_hat"], reverse=True)
    (args.sweep_dir / "m1_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
