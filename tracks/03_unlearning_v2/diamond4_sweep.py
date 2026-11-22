"""P1.2 diamond probe (opposite mu/sigma splits) + P2.4 prediction-attack: 4 configs, 1 per GPU.

Calibration (journal 00h07/01h08, all Full 3500-image evals, adaptive runner):
- TS_ce1.20_u2.40_b1.5 -> mu+sig 2.21 ; TS_ce1.60_u2.80_b1.2 -> 2.55  => ~+0.85 per +1.0 target_forget_ce
- AT_ce1.00_b1.8 -> 3.96 ; AT_ce1.80_b1.2 -> 5.28 (sigma etire par beta faible)
- gold_mimic caps at mean target CE ~ -log(0.15) = 1.90 (p_i clamp) => cannot reach the window CLI-only.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from auto_tune_4gpu import _build_train_proxy, run_adaptive_salun_npo
from contracts import MethodConfig
from eval_harness import evaluate_candidate, score_server_calibrated_proxy
from run_base_refine import run_tight_at
from task_template import NUM_CLASSES, load_model, predict


def _diamond_configs() -> list[tuple[str, MethodConfig]]:
    """4 rupture-plan configs: 2 diamond probes + 2 in-window prediction attacks."""
    dia_lowstd = MethodConfig(
        name="DIA_lowstd_TS_ce300_u380_pw08_b14_f6",
        strategy="salun_npo", loss_kind="two_sided_npo",
        steps=250, lr=1.2e-4, npo_beta=1.4,
        target_forget_ce=3.00, target_ce_upper=3.80, pullback_weight=0.8,
        retain_kl_weight=2.2, l2_weight=1.5e-3, salun_keep_ratio=0.25,
        freeze_until_block=6, seed=42,
    )
    dia_highstd = MethodConfig(
        name="DIA_highstd_AT_ce120_b12_k20_kl26",
        strategy="salun_npo", loss_kind="npo",
        steps=225, lr=1.05e-4, npo_beta=1.2,
        target_forget_ce=1.20, target_ce_upper=2.00,
        retain_kl_weight=2.6, l2_weight=2.5e-3, salun_keep_ratio=0.20,
        freeze_until_block=6, seed=42,
    )
    pa1 = MethodConfig(
        name="PA1_AT_ce085_b17_k20_kl26_s77",
        strategy="salun_npo", loss_kind="npo",
        steps=210, lr=9e-5, npo_beta=1.7,
        target_forget_ce=0.85, target_ce_upper=1.30,
        retain_kl_weight=2.6, l2_weight=3.0e-3, salun_keep_ratio=0.20,
        freeze_until_block=6, seed=77,
    )
    pa2 = MethodConfig(
        name="PA2_AT_ce095_b20_k22_kl30_s123",
        strategy="salun_npo", loss_kind="npo",
        steps=210, lr=9.5e-5, npo_beta=2.0,
        target_forget_ce=0.95, target_ce_upper=1.45,
        retain_kl_weight=3.0, l2_weight=3.5e-3, salun_keep_ratio=0.22,
        freeze_until_block=6, seed=123,
    )
    return [("adaptive", dia_lowstd), ("adaptive", dia_highstd), ("tight", pa1), ("tight", pa2)]


def _train_one(
    runner: str, cfg: MethodConfig, orig, f_x, orig_f, p_x, orig_p, absent_mask,
):
    if runner == "tight":
        return run_tight_at(orig, orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)
    return run_adaptive_salun_npo(orig, orig, f_x, orig_f, p_x, orig_p, absent_mask, cfg)


def _in_window(row: dict) -> bool:
    mu_sig = row["ce_mean"] + row["ce_std"]
    return (
        3.65 <= mu_sig <= 4.00
        and 0.80 <= row["ce_p50"] <= 1.20
        and row["proxy_kl_t2"] <= 0.056
        and row["delta_l2_total"] <= 4.70
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Diamond probe + prediction-attack, 1 config per GPU.")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("output/diamond4"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gpu{args.worker_id}] {time.strftime('%H:%M:%S')} loading model + forget set", flush=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    orig_f = predict(orig, f_x, batch_size=128)
    p_x = _build_train_proxy(f_x)
    orig_p = predict(orig, p_x, batch_size=128)
    absent_mask = torch.bincount(orig_f.argmax(dim=-1), minlength=NUM_CLASSES) == 0

    mine = [c for i, c in enumerate(_diamond_configs()) if i % args.num_workers == args.worker_id]
    rows = []
    for runner, cfg in mine:
        print(f"[gpu{args.worker_id}] {time.strftime('%H:%M:%S')} train {cfg.name} ({runner})", flush=True)
        sub = _train_one(runner, cfg, orig, f_x, orig_f, p_x, orig_p, absent_mask)
        ckpt = args.out_dir / f"{cfg.name}.pt"
        torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, ckpt)
        ev = evaluate_candidate(orig, sub, f_x, sub_path=ckpt)
        fr, rp = ev.forget_raw, ev.retain_proxy
        row = {
            "name": cfg.name, "runner": runner,
            "ce_mean": fr.ce_mean, "ce_std": fr.ce_std,
            "ce_p50": fr.ce_p50, "frac_gt_95": fr.frac_gt_95,
            "proxy_kl_t2": rp.proxy_kl_t2, "absent_classes_kl_t2": rp.absent_classes_kl_t2,
            "delta_l2_total": rp.delta_l2_total,
            "calibrated_proxy": score_server_calibrated_proxy(
                fr.ce_mean, fr.ce_std, fr.ce_p50, rp.proxy_kl_t2, rp.absent_classes_kl_t2, rp.delta_l2_total,
            ),
            "valid": ev.checkpoint.valid,
        }
        row["mu_sigma"] = round(row["ce_mean"] + row["ce_std"], 3)
        row["in_window"] = _in_window(row)
        rows.append(row)
        print(
            f"[gpu{args.worker_id}] {time.strftime('%H:%M:%S')} {cfg.name:36s} | "
            f"CE={row['ce_mean']:.3f}±{row['ce_std']:.2f} mu+s={row['mu_sigma']:.2f} p50={row['ce_p50']:.3f} "
            f">95%={row['frac_gt_95']:.2f} | ProxyKL={row['proxy_kl_t2']:.4f} AbsKL={row['absent_classes_kl_t2']:.4f} "
            f"dTheta={row['delta_l2_total']:.2f} | proxy={row['calibrated_proxy']:.4f} "
            f"{'WINDOW' if row['in_window'] else 'OUT'}",
            flush=True,
        )

    (args.out_dir / f"worker_{args.worker_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
