"""Rank the 11 server-scored checkpoints against the Virtual Gold Teacher."""

import argparse
import json
from pathlib import Path

import torch

import research_virtual_gold as rvg


def load_model(path: Path, device: str) -> torch.nn.Module:
    import timm
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, img_size=224, init_values=1e-4)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def predict(model: torch.nn.Module, pixels: torch.Tensor, device: str) -> torch.Tensor:
    out = []
    with torch.no_grad():
        for batch in pixels.split(64):
            x = batch.to(device).float().permute(0, 3, 1, 2).div_(127.5).sub_(1.0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                out.append(model(x).float().cpu())
    return torch.cat(out)


def score_candidates(original_logits, images, labels, masked_logits, ckpts, args, device) -> dict:
    """Score every candidate checkpoint; per-candidate failures recorded, not fatal."""
    results = {}
    for path in ckpts:
        candidate = None
        try:
            candidate = load_model(path, device)
            candidate_logits = predict(candidate, images, device)
            results[path.stem] = rvg.score_virtual_gold(
                original_logits, candidate_logits, masked_logits, labels, args.views)
        except Exception as exc:  # keep ranking the remaining checkpoints
            results[path.stem] = {"error": repr(exc)}
        del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--ckpts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.30, 0.45])
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = rvg.load_images(args.images, args.max_images)
    original = load_model(args.orig, device)
    original_logits = predict(original, images, device)
    labels = original_logits.argmax(dim=-1)

    report = {
        "protocol": {"images": len(images), "views": args.views, "seed": args.seed,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     "interpretation": "Relative candidate-ranking diagnostic, not an official score."},
        "fractions": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fraction in args.fractions:
        masked = rvg.patch_mask_views(images, args.seed, fraction, args.views)
        masked_logits = predict(original, masked, device)
        report["fractions"][str(fraction)] = score_candidates(
            original_logits, images, labels, masked_logits, args.ckpts, args, device)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"FRACTION_DONE {fraction}", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
