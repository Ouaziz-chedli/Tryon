"""Coherence check for Track G Lab: verify ranking matches our 2 server submissions."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Ground-Truth Lab against server submissions.")
    parser.add_argument("--lab-dir", type=Path, default=Path("output/lab"))
    args = parser.parse_args()

    orig_lab = load_model(args.lab_dir / "orig_lab.pt")
    gold_lab = load_model(args.lab_dir / "gold_lab.pt")
    splits = torch.load(args.lab_dir / "lab_splits.pt", map_location="cpu", weights_only=True)

    f_x, f_y = splits["f_x"], splits["f_y"]
    orig_f = predict(orig_lab, f_x, batch_size=128)
    proxy_x = make_patchmix_proxy(f_x, seed=42)
    orig_p = predict(orig_lab, proxy_x, batch_size=128)
    absent_mask = torch.bincount(orig_f.argmax(dim=-1), minlength=NUM_CLASSES) == 0

    candidates = [
        MethodConfig(name="gold_oracle", strategy="oracle"),
        MethodConfig(name="orig_no_unlearn", strategy="orig"),
        MethodConfig(name="H_ssd_lora_mod_aug", strategy="lora_npo", lr=1e-4, lora_rank=8, ssd_alpha=0.5, ssd_threshold=1.0, use_ssd_init=True, target_forget_ce=0.7, aug_mode="moderate"),
        MethodConfig(name="A_npo_mod_aug", strategy="salun_npo", lr=1e-4, target_forget_ce=0.7, aug_mode="moderate"),
    ]

    results = {}
    for cfg in candidates:
        if cfg.strategy == "oracle":
            sub = gold_lab
        elif cfg.strategy == "orig":
            sub = orig_lab
        else:
            sub = STRATEGY_RUNNERS[cfg.strategy](orig_lab, f_x, orig_f, proxy_x, orig_p, absent_mask, cfg)
        rep = evaluate_against_gold(
            orig_lab, gold_lab, sub, f_x, f_y,
            splits["ctrl_x"], splits["ctrl_y"], splits["val_x"], splits["val_y"],
        )
        results[cfg.name] = asdict(rep)
        print(f"{cfg.name:20s} -> final={rep.final_score:.4f} (forget={rep.forget_score:.4f}, retain={rep.retain_score:.4f}, val_acc={rep.val_acc:.4f})")

    (args.lab_dir / "coherence_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
