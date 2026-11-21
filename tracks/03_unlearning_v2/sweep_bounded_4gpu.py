"""4-GPU parallel sweep of per-sample bounded SalUn+NPO on real model.pt."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch

from contracts import MethodConfig
from eval_harness import evaluate_candidate, make_patchmix_proxy
from methods.salun_npo import run_salun_npo
from task_template import IMAGE_SIZE, NUM_CLASSES, load_model, predict


def _load_natural_proxy(forget_images: torch.Tensor, count: int = 3500) -> tuple[torch.Tensor, str]:
    """Build 224x224 multi-grid PatchMix proxy (4x4 + 7x7) without resolution shortcuts."""
    half = len(forget_images) // 2
    if half == 0:
        return make_patchmix_proxy(forget_images, seed=42, grid=4), "patchmix_g4_g7"
    proxy = torch.cat([
        make_patchmix_proxy(forget_images[:half], seed=42, grid=4),
        make_patchmix_proxy(forget_images[half:], seed=43, grid=7),
    ], dim=0)
    return proxy[:count], "patchmix_g4_g7"


def _build_configs() -> list[MethodConfig]:
    """12 per-sample bounded configs with paper-derived NPO beta in {2.0, 2.5, 3.0} and layer-wise SalUn."""
    cfgs = []
    for ce_tgt in (0.45, 0.55, 0.65, 0.80):
        for lr, kl_w, beta in ((8e-5, 2.0, 2.0), (8e-5, 2.5, 2.5), (1.0e-4, 2.5, 3.0)):
            name = f"B_ce{ce_tgt:.2f}_b{beta:.1f}_kl{kl_w:.1f}"
            cfgs.append(MethodConfig(
                name=name, strategy="salun_npo", steps=200, lr=lr, npo_beta=beta,
                target_forget_ce=ce_tgt, retain_kl_weight=kl_w, salun_keep_ratio=0.20, aug_mode="moderate",
            ))
    return cfgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded per-sample SalUn+NPO sweep.")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=Path("output/bounded_sweep"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig = load_model("model.pt")
    f_x = torch.from_numpy(np.load("forget_images.npz")["images"])
    orig_f = predict(orig, f_x, batch_size=128)
    proxy_x, proxy_src = _load_natural_proxy(f_x)
    orig_p = predict(orig, proxy_x, batch_size=128)
    absent_mask = torch.bincount(orig_f.argmax(dim=-1), minlength=NUM_CLASSES) == 0

    mine = [c for i, c in enumerate(_build_configs()) if i % args.num_workers == args.worker_id]
    rows = []
    for cfg in mine:
        sub = run_salun_npo(orig, f_x, orig_f, proxy_x, orig_p, absent_mask, cfg)
        ckpt = args.out_dir / f"{cfg.name}.pt"
        torch.save({k: v.detach().cpu().float() for k, v in sub.state_dict().items()}, ckpt)
        ev = evaluate_candidate(orig, sub, f_x, sub_path=ckpt)
        row = {"name": cfg.name, "proxy_src": proxy_src, **asdict(ev.forget_raw), **asdict(ev.retain_proxy), "valid": ev.checkpoint.valid}
        rows.append(row)
        print(f"[gpu{args.worker_id}] {cfg.name:26s} | CE={row['ce_mean']:.3f}±{row['ce_std']:.2f} p50={row['ce_p50']:.3f} >95%={row['frac_gt_95']:.2f} | ProxyKL={row['proxy_kl_t2']:.4f} AbsKL={row['absent_classes_kl_t2']:.4f} dTheta={row['delta_l2_total']:.2f}", flush=True)

    (args.out_dir / f"worker_{args.worker_id}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
