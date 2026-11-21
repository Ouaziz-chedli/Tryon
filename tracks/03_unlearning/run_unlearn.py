"""CLI entrypoint for Track 3 Machine Unlearning."""

import argparse
from dataclasses import asdict
from pathlib import Path
import torch

from contracts import DiagnosticsReport, UnlearnConfig
from metrics import build_diagnostics_report, make_augmented_copy, make_patchmix_proxy, validate_submission
from task_template import load_forget_images, load_model, predict
from unlearner import unlearn_model


def parse_args() -> UnlearnConfig:
    parser = argparse.ArgumentParser(description="Run SalUn + NPO + PatchMix unlearning.")
    parser.add_argument("--model-path", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget-path", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--output-path", type=Path, default=Path("submission.pt"))
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--npo-beta", type=float, default=0.1)
    parser.add_argument("--retain-kl-weight", type=float, default=2.0)
    parser.add_argument("--l2-weight", type=float, default=1e-3)
    parser.add_argument("--salun-keep-ratio", type=float, default=0.2)
    parser.add_argument("--target-forget-loss", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return UnlearnConfig(
        model_path=args.model_path,
        forget_images_path=args.forget_path,
        output_path=args.output_path,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        npo_beta=args.npo_beta,
        retain_kl_weight=args.retain_kl_weight,
        l2_weight=args.l2_weight,
        salun_keep_ratio=args.salun_keep_ratio,
        target_forget_loss=args.target_forget_loss,
        seed=args.seed,
    )


def print_report(label: str, report: DiagnosticsReport) -> None:
    print(f"\n=== {label} ===")
    for key, value in asdict(report).items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


def main() -> int:
    cfg = parse_args()
    orig_model = load_model(cfg.model_path)
    forget_images = load_forget_images(cfg.forget_images_path)
    aug_images = make_augmented_copy(forget_images, seed=cfg.seed + 1)
    proxy_images = make_patchmix_proxy(forget_images, seed=cfg.seed + 2)

    orig_f = predict(orig_model, forget_images, batch_size=cfg.batch_size)
    orig_p = predict(orig_model, proxy_images, batch_size=cfg.batch_size)
    orig_aug = predict(orig_model, aug_images, batch_size=cfg.batch_size)
    print_report("Baseline (orig)", build_diagnostics_report(orig_f, orig_f, orig_aug, orig_p, orig_p, cfg.temperature))

    unlearned = unlearn_model(orig_model, forget_images, cfg)
    sub_f = predict(unlearned, forget_images, batch_size=cfg.batch_size)
    sub_aug = predict(unlearned, aug_images, batch_size=cfg.batch_size)
    sub_p = predict(unlearned, proxy_images, batch_size=cfg.batch_size)
    print_report("Unlearned (sub)", build_diagnostics_report(orig_f, sub_f, sub_aug, orig_p, sub_p, cfg.temperature))

    state_dict = {k: v.detach().cpu().float() for k, v in unlearned.state_dict().items()}
    torch.save(state_dict, cfg.output_path)
    val = validate_submission(cfg.output_path)
    print(f"\nValidation: valid={val.valid}, size={val.file_size_mb:.2f} MB, errors={val.errors}")
    return 0 if val.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
