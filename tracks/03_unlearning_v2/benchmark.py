"""Comparative benchmark runner across all Track 3 unlearning strategies."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch

from contracts import EvalReport, MethodConfig
from eval_harness import evaluate_candidate
from methods import STRATEGY_RUNNERS
from methods.proxies import make_retain_proxy
from task_template import NUM_CLASSES, load_forget_images, load_model, predict


def build_benchmark_suite() -> list[MethodConfig]:
    """Return the v3 targeted benchmark suite (moderate aug + checkpoint candidates)."""
    return [
        MethodConfig(name="A_npo_mod_aug", strategy="salun_npo", lr=1e-4, target_forget_ce=0.7, aug_mode="moderate"),
        MethodConfig(name="A_npo_no_aug", strategy="salun_npo", lr=1e-4, target_forget_ce=0.7, aug_mode="none"),
        MethodConfig(name="E_lora_r8_mod_aug", strategy="lora_npo", lr=1e-4, lora_rank=8, target_forget_ce=0.7, aug_mode="moderate"),
        MethodConfig(name="H_ssd_lora_mod_aug", strategy="lora_npo", lr=1e-4, lora_rank=8, ssd_alpha=0.5, ssd_threshold=1.0, use_ssd_init=True, target_forget_ce=0.7, aug_mode="moderate"),
        MethodConfig(name="B_bad_teacher_kl4", strategy="bad_teacher", lr=1.5e-4, bad_teacher_temp=6.0, retain_kl_weight=4.0, target_forget_ce=0.6, aug_mode="moderate"),
    ]


def _format_summary_row(name: str, r: EvalReport, elapsed_s: float) -> str:
    stress_ce = sum(m.ce_mean for m in r.forget_stress.values()) / len(r.forget_stress)
    stress_agree = sum(m.top1_agree_orig for m in r.forget_stress.values()) / len(r.forget_stress)
    rp = r.retain_proxy
    return (
        f"| {name} | {r.forget_raw.ce_mean:.3f}±{r.forget_raw.ce_std:.2f} | "
        f"{r.forget_raw.entropy_mean:.3f} | {stress_ce:.3f} | {stress_agree:.3f} | "
        f"{rp.proxy_kl_t2:.4f} | {rp.absent_classes_kl_t2:.4f} | "
        f"{rp.delta_l2_total:.2f} | {elapsed_s:.1f}s |"
    )


def _write_summary(rows: list[str], reports: dict[str, dict], out_dir: Path) -> None:
    header = (
        "| Run | Raw CE (mean±std) | Raw Ent | Stress CE avg | Stress Agree avg | "
        "Proxy KL(T=2) | Absent KL(T=2) | Δθ L2 | Time |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    )
    (out_dir / "summary.md").write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all Track 3 unlearning strategies.")
    parser.add_argument("--model-path", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget-path", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/benchmark"))
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of run names.")
    parser.add_argument("--save-checkpoints", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    orig_model = load_model(args.model_path)
    forget_images = load_forget_images(args.forget_path)
    orig_f = predict(orig_model, forget_images, batch_size=128)
    absent_mask = torch.bincount(orig_f.argmax(dim=-1), minlength=NUM_CLASSES) == 0
    proxy_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    rows, all_reports = [], {}
    t0 = time.time()
    base_report = evaluate_candidate(orig_model, orig_model, forget_images)
    rows.append(_format_summary_row("0_orig_baseline", base_report, time.time() - t0))
    all_reports["0_orig_baseline"] = asdict(base_report)

    suite = [c for c in build_benchmark_suite() if not args.only or c.name in args.only]
    for cfg in suite:
        if cfg.proxy_kind not in proxy_cache:
            p_imgs = make_retain_proxy(forget_images, kind=cfg.proxy_kind, seed=42)
            proxy_cache[cfg.proxy_kind] = (p_imgs, predict(orig_model, p_imgs, batch_size=128))
        proxy_images, orig_p = proxy_cache[cfg.proxy_kind]
        t_start = time.time()
        sub_model = STRATEGY_RUNNERS[cfg.strategy](
            orig_model, forget_images, orig_f, proxy_images, orig_p, absent_mask, cfg
        )
        ckpt_path = args.out_dir / f"{cfg.name}.pt" if args.save_checkpoints else None
        if ckpt_path is not None:
            torch.save({k: v.detach().cpu().float() for k, v in sub_model.state_dict().items()}, ckpt_path)
        report = evaluate_candidate(orig_model, sub_model, forget_images, sub_path=ckpt_path)
        elapsed = time.time() - t_start
        rows.append(_format_summary_row(cfg.name, report, elapsed))
        all_reports[cfg.name] = asdict(report)
        print(rows[-1], flush=True)

    _write_summary(rows, all_reports, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
