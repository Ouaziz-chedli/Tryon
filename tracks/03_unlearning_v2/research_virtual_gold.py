"""Test whether stochastic patch masking is a useful virtual-gold oracle."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def patch_mask_views(images: torch.Tensor, seed: int, fraction: float, views: int) -> torch.Tensor:
    """Replace random 16x16 patches with channel means in the uint8 pixel domain."""
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1:] != (224, 224, 3):
        raise ValueError(f"Expected uint8 [N,224,224,3], got {images.dtype} {tuple(images.shape)}")
    if not 0.0 < fraction < 1.0 or views < 1:
        raise ValueError("fraction must be in (0,1) and views must be positive")
    n, h, w, _ = images.shape
    grid_w = w // 16
    patch_count = (h // 16) * grid_w
    masked_count = round(patch_count * fraction)
    gen = torch.Generator().manual_seed(seed)
    means = images.float().mean(dim=(0, 1, 2)).round().to(torch.uint8)
    result = images.unsqueeze(0).expand(views, -1, -1, -1, -1).clone()
    for view in range(views):
        selected_patches = torch.rand((n, patch_count), generator=gen).argsort(dim=1)[:, :masked_count]
        masks = torch.zeros((n, patch_count), dtype=torch.bool)
        masks.scatter_(1, selected_patches, True)
        for patch_id in range(patch_count):
            selected = masks[:, patch_id]
            if not selected.any():
                continue
            row, col = divmod(patch_id, grid_w)
            patch = result[view, :, row * 16:(row + 1) * 16, col * 16:(col + 1) * 16]
            patch[selected] = means
    return result.reshape(views * n, 224, 224, 3)


def load_images(path: Path, max_images: int) -> torch.Tensor:
    with np.load(path) as archive:
        key = "images" if "images" in archive else archive.files[0]
        images = torch.from_numpy(archive[key][:max_images]).to(torch.uint8)
    if images.shape[0] == 0:
        raise ValueError("forget image archive is empty")
    return images


def per_view_losses(logits: torch.Tensor, labels: torch.Tensor, views: int, count: int) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(views, count, -1).flatten(0, 1), labels.repeat(views), reduction="none").reshape(views, count)


def kl_at_temperature(teacher: torch.Tensor, student: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    t = temperature
    p_teacher = F.softmax(teacher.float() / t, dim=-1)
    log_p_student = F.log_softmax(student.float() / t, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1) * t * t


def score_virtual_gold(original_logits: torch.Tensor, candidate_logits: torch.Tensor,
                       masked_logits: torch.Tensor, labels: torch.Tensor, views: int) -> dict[str, float]:
    """Compare candidate with VGT using CE-moment and T=2 KL distances."""
    count = labels.numel()
    gold_view_losses = per_view_losses(masked_logits, labels, views, count)
    gold_logits = masked_logits.reshape(views, count, -1).mean(dim=0)
    orig_loss = F.cross_entropy(original_logits, labels, reduction="none")
    candidate_loss = F.cross_entropy(candidate_logits, labels, reduction="none")
    gold_loss_mean = gold_view_losses.mean(dim=0)
    d_candidate = (candidate_loss.mean() - gold_loss_mean.mean()).abs() + (candidate_loss.std() - gold_loss_mean.std()).abs()
    d_original = (orig_loss.mean() - gold_loss_mean.mean()).abs() + (orig_loss.std() - gold_loss_mean.std()).abs()
    ms = 1.0 - min(1.0, float(d_candidate / d_original.clamp_min(1e-8)))
    # Only a relative ranking diagnostic; virtual-gold scores are not official scores.
    kl_candidate = kl_at_temperature(gold_logits, candidate_logits).mean()
    kl_original = kl_at_temperature(gold_logits, original_logits).mean()
    pred = 1.0 - min(1.0, float(kl_candidate / kl_original.clamp_min(1e-8)))
    return {
        "virtual_gold_ce_mean": float(gold_loss_mean.mean()),
        "virtual_gold_ce_std_between_images": float(gold_loss_mean.std()),
        "virtual_gold_ce_std_within_images": float(gold_view_losses.std(dim=0).mean()),
        "virtual_gold_ce_std_pooled": float(gold_view_losses.flatten().std()),
        "mean_std_score": ms, "prediction_score": pred, "forget_score_surrogate": ms * pred,
        "kl_gold_candidate_t2": float(kl_candidate), "kl_gold_original_t2": float(kl_original),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", type=Path, default=Path("/content/vgt/model.pt"))
    parser.add_argument("--images", type=Path, default=Path("/content/vgt/forget_images.npz"))
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--masked-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    images = load_images(args.images, args.max_images)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import timm

    def load(path: Path) -> torch.nn.Module:
        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, img_size=224, init_values=1e-4)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        return model.to(device).eval()

    def predict(model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
        out = []
        with torch.no_grad():
            for batch in pixels.split(64):
                x = batch.to(device).float().permute(0, 3, 1, 2).div_(127.5).sub_(1.0)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    out.append(model(x).float().cpu())
        return torch.cat(out)

    original = load(args.orig)
    labels = predict(original, images).argmax(dim=-1)
    masked = patch_mask_views(images, args.seed, args.masked_fraction, args.views)
    masked_logits = predict(original, masked)
    original_logits = predict(original, images)
    results = {}
    for path in args.candidates:
        candidate = load(path)
        candidate_logits = predict(candidate, images)
        results[path.stem] = score_virtual_gold(original_logits, candidate_logits, masked_logits, labels, args.views)
        del candidate, candidate_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "protocol": {"images": len(images), "views": args.views, "masked_fraction": args.masked_fraction,
                     "patch_size": 16, "seed": args.seed,
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     "interpretation": "Relative candidate-ranking diagnostic, not an official score."},
        "candidates": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
