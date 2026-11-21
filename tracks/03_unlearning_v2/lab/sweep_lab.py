"""Multi-strategy sweep on cached Ground-Truth Lab (orig_lab + gold_lab)."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch

from contracts import MethodConfig
from eval_harness import make_patchmix_proxy
from lab.server_metrics import evaluate_against_gold
from methods import STRATEGY_RUNNERS
from task_template import NUM_CLASSES, load_model, predict


def _candidate_grid() -> list[MethodConfig]:
    """Build targeted candidate configs across strategies A, B, C, E, F."""
    return [
        MethodConfig(name="A_npo_lr2e5_ce0.25", strategy="salun_npo", lr=2e-5, target_forget_ce=0.25, retain_kl_weight=3.0),
        MethodConfig(name="A_npo_lr3e5_ce0.35", strategy="salun_npo", lr=3e-5, target_forget_ce=0.35, retain_kl_weight=3.0),
        MethodConfig(name="A_npo_lr5e5_ce0.40_kl4", strategy="salun_npo", lr=5e-5, target_forget_ce=0.40, retain_kl_weight=4.0),
        MethodConfig(name="B_bt_lr2e5_T3", strategy="bad_teacher", lr=2e-5, bad_teacher_temp=3.0, retain_kl_weight=3.0),
        MethodConfig(name="B_bt_lr4e5_T4", strategy="bad_teacher", lr=4e-5, bad_teacher_temp=4.0, retain_kl_weight=3.0),
        MethodConfig(name="C_ssd_a0.1_t1.5", strategy="ssd", ssd_alpha=0.1, ssd_threshold=1.5),
        MethodConfig(name="C_ssd_a0.25_t1.5", strategy="ssd", ssd_alpha=0.25, ssd_threshold=1.5),
        MethodConfig(name="C_ssd_a0.5_t2.0", strategy="ssd", ssd_alpha=0.5, ssd_threshold=2.0),
        MethodConfig(name="E_lora_npo_lr5e5_ce0.3", strategy="lora_npo", lr=5e-5, lora_rank=8, target_forget_ce=0.3, retain_kl_weight=3.0),
        MethodConfig(name="E_lora_npo_lr8e5_ce0.4", strategy="lora_npo", lr=8e-5, lora_rank=8, target_forget_ce=0.4, retain_kl_weight=3.0),
        MethodConfig(name="E_lora_bt_lr5e5_T4", strategy="lora_npo", loss_kind="bad_teacher", lr=5e-5, lora_rank=8, bad_teacher_temp=4.0, retain_kl_weight=3.0),
        MethodConfig(name="F_ablit_a0.15_r8", strategy="abliteration", abliteration_alpha=0.15, abliteration_rank=8),
        MethodConfig(name="F_ablit_a0.30_r8", strategy="abliteration", abliteration_alpha=0.30, abliteration_rank=8),
        MethodConfig(name="F_ablit_a0.50_r12", strategy="abliteration", abliteration_alpha=0.50, abliteration_rank=12),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep strategies on cached Lab.")
    parser.add_argument("--lab-dir", type=Path, default=Path("output/lab_in100"))
    args = parser.parse_args()

    orig_lab = load_model(args.lab_dir / "orig_lab.pt")
    gold_lab = load_model(args.lab_dir / "gold_lab.pt")
    splits = torch.load(args.lab_dir / "lab_splits.pt", map_location="cpu", weights_only=True)

    f_x, f_y = splits["f_x"], splits["f_y"]
    orig_f = predict(orig_lab, f_x, batch_size=128)
    proxy_x = make_patchmix_proxy(f_x, seed=42)
    orig_p = predict(orig_lab, proxy_x, batch_size=128)
    absent_mask = torch.bincount(orig_f.argmax(dim=-1), minlength=NUM_CLASSES) == 0

    rows = []
    for cfg in _candidate_grid():
        sub = STRATEGY_RUNNERS[cfg.strategy](orig_lab, f_x, orig_f, proxy_x, orig_p, absent_mask, cfg)
        rep = evaluate_against_gold(
            orig_lab, gold_lab, sub, f_x, f_y,
            splits["ctrl_x"], splits["ctrl_y"], splits["val_x"], splits["val_y"],
        )
        rows.append({"name": cfg.name, "strategy": cfg.strategy, **asdict(rep)})
        print(f"{cfg.name:24s} -> final={rep.final_score:.4f} (forget={rep.forget_score:.4f}, mean_std={rep.mean_std_score:.4f}, pred={rep.prediction_score:.4f}, retain={rep.retain_score:.4f}, val={rep.val_acc:.4f})", flush=True)

    rows.sort(key=lambda r: r["final_score"], reverse=True)
    (args.lab_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
