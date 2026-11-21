"""Local validation harness (Phase 3) for Track 3 Machine Unlearning."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.io as tv_io
import torchvision.transforms.functional as TF

from contracts import EvalReport, RetainProxyMetrics, StressMetrics, ValidationResult
from task_template import IMAGE_SIZE, NUM_CLASSES, build_model, load_forget_images, load_model, predict

MAX_SUBMISSION_MB = 450.0


def compute_kl_at_temperature(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute per-sample KL(p_teacher || p_student) at a given temperature."""
    p_teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)


def compute_stress_metrics(
    orig_logits: torch.Tensor,
    sub_logits: torch.Tensor,
    ref_preds: torch.Tensor,
    temperature: float = 2.0,
) -> StressMetrics:
    """Compute CE, entropy, confidence, agreement, and KL(T=2) against orig_logits."""
    sub_f = sub_logits.float()
    probs = F.softmax(sub_f, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    ce = F.cross_entropy(sub_f, ref_preds, reduction="none")
    agree = (sub_f.argmax(dim=-1) == ref_preds).float().mean()
    kl = compute_kl_at_temperature(orig_logits, sub_f, temperature).mean()
    return StressMetrics(
        ce_mean=round(float(ce.mean()), 4),
        ce_std=round(float(ce.std()), 4),
        ce_p50=round(float(torch.quantile(ce, 0.5)), 4),
        ce_p90=round(float(torch.quantile(ce, 0.9)), 4),
        entropy_mean=round(float(entropy.mean()), 4),
        max_prob_mean=round(float(top2[:, 0].mean()), 4),
        frac_gt_95=round(float((top2[:, 0] > 0.95).float().mean()), 4),
        margin_mean=round(float((top2[:, 0] - top2[:, 1]).mean()), 4),
        top1_agree_orig=round(float(agree), 4),
        kl_t2_from_orig=round(float(kl), 4),
    )


def make_patchmix_proxy(images_uint8: torch.Tensor, seed: int = 42, grid: int = 4) -> torch.Tensor:
    """Create out-of-sample retain proxy images by shuffling spatial patches across images."""
    if images_uint8.ndim != 4 or images_uint8.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected [N, {IMAGE_SIZE}, {IMAGE_SIZE}, 3], got {tuple(images_uint8.shape)}")
    gen = torch.Generator().manual_seed(seed)
    n, patch = len(images_uint8), IMAGE_SIZE // grid
    out = images_uint8.clone()
    for r in range(grid):
        for c in range(grid):
            perm = torch.randperm(n, generator=gen)
            r0, r1 = r * patch, (r + 1) * patch
            c0, c1 = c * patch, (c + 1) * patch
            out[:, r0:r1, c0:c1, :] = images_uint8[perm, r0:r1, c0:c1, :]
    return out


def _jpeg_roundtrip(nchw_uint8: torch.Tensor, quality: int = 30) -> torch.Tensor:
    decoded = [tv_io.decode_jpeg(tv_io.encode_jpeg(img, quality=quality)) for img in nchw_uint8.cpu()]
    return torch.stack(decoded, dim=0)


def make_stress_suite(images_uint8: torch.Tensor, seed: int = 42) -> dict[str, torch.Tensor]:
    """Generate deterministic altered copies simulating hidden server perturbations."""
    nchw = images_uint8.permute(0, 3, 1, 2)
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(images_uint8.shape, generator=gen) * 25.0
    color = TF.adjust_saturation(TF.adjust_brightness(nchw, 1.3), 0.4)
    return {
        "hflip": torch.flip(images_uint8, dims=[2]),
        "crop50": TF.resized_crop(nchw, 56, 56, 112, 112, [IMAGE_SIZE, IMAGE_SIZE]).permute(0, 2, 3, 1).contiguous(),
        "rotate15": TF.rotate(nchw, angle=15.0).permute(0, 2, 3, 1).contiguous(),
        "grayscale": TF.rgb_to_grayscale(nchw, num_output_channels=3).permute(0, 2, 3, 1).contiguous(),
        "color_jitter": color.permute(0, 2, 3, 1).contiguous(),
        "blur_s2": TF.gaussian_blur(nchw, kernel_size=[5, 5], sigma=[2.0, 2.0]).permute(0, 2, 3, 1).contiguous(),
        "noise_25": (images_uint8.float() + noise).clamp(0, 255).to(torch.uint8),
        "jpeg_q30": _jpeg_roundtrip(nchw, quality=30).permute(0, 2, 3, 1).contiguous(),
    }


def compute_weight_drift(orig_model: torch.nn.Module, sub_model: torch.nn.Module) -> tuple[float, float, float]:
    """Return L2 parameter drift (total, head, blocks) between orig_model and sub_model."""
    orig_sd = orig_model.state_dict()
    total_sq, head_sq, blocks_sq = 0.0, 0.0, 0.0
    for name, p_sub in sub_model.state_dict().items():
        if not torch.is_floating_point(p_sub):
            continue
        diff_sq = float((p_sub.detach().cpu().float() - orig_sd[name].detach().cpu().float()).pow(2).sum().item())
        total_sq += diff_sq
        if name.startswith("head."):
            head_sq += diff_sq
        elif name.startswith("blocks."):
            blocks_sq += diff_sq
    return round(total_sq ** 0.5, 4), round(head_sq ** 0.5, 4), round(blocks_sq ** 0.5, 4)


def _compute_retain_proxy_metrics(
    orig_model: torch.nn.Module,
    sub_model: torch.nn.Module,
    orig_proxy: torch.Tensor,
    sub_proxy: torch.Tensor,
    absent_mask: torch.Tensor,
    temperature: float = 2.0,
) -> RetainProxyMetrics:
    p_orig = F.softmax(orig_proxy.float(), dim=-1)
    p_sub = F.softmax(sub_proxy.float(), dim=-1)
    kl_all = compute_kl_at_temperature(orig_proxy, sub_proxy, temperature).mean()
    agree_all = (orig_proxy.argmax(dim=-1) == sub_proxy.argmax(dim=-1)).float().mean()
    absent_idx = absent_mask.to(orig_proxy.device)
    mass_orig = p_orig[:, absent_idx].sum(dim=-1).mean()
    mass_sub = p_sub[:, absent_idx].sum(dim=-1).mean()
    kl_absent = compute_kl_at_temperature(orig_proxy[:, absent_idx], sub_proxy[:, absent_idx], temperature).mean()
    d_total, d_head, d_blocks = compute_weight_drift(orig_model, sub_model)
    return RetainProxyMetrics(
        proxy_kl_t2=round(float(kl_all), 4),
        proxy_top1_agree=round(float(agree_all), 4),
        absent_classes_mass_orig=round(float(mass_orig), 4),
        absent_classes_mass_sub=round(float(mass_sub), 4),
        absent_classes_kl_t2=round(float(kl_absent), 4),
        delta_l2_total=d_total,
        delta_l2_head=d_head,
        delta_l2_blocks=d_blocks,
    )


def validate_checkpoint(path: Path) -> ValidationResult:
    """Verify that a saved state_dict satisfies all server submission rules."""
    if not path.is_file():
        return ValidationResult(valid=False, file_size_mb=0.0, errors=(f"Missing file: {path}",))
    size_mb = round(path.stat().st_size / (1024 * 1024), 2)
    errors: list[str] = []
    if size_mb > MAX_SUBMISSION_MB:
        errors.append(f"File size {size_mb:.2f} MB exceeds {MAX_SUBMISSION_MB} MB limit.")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        return ValidationResult(valid=False, file_size_mb=size_mb, errors=("Checkpoint is not a state_dict dict.",))
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor) or tensor.dtype != torch.float32:
            errors.append(f"Tensor {name} has dtype {tensor.dtype}, expected torch.float32.")
            break
        if not torch.isfinite(tensor).all():
            errors.append(f"Tensor {name} contains NaN or Inf.")
            break
    try:
        build_model().load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        errors.append(f"strict=True load failed: {exc}")
    return ValidationResult(valid=len(errors) == 0, file_size_mb=size_mb, errors=tuple(errors))


def evaluate_candidate(
    orig_model: torch.nn.Module,
    sub_model: torch.nn.Module,
    forget_images: torch.Tensor,
    batch_size: int = 128,
    seed: int = 42,
    sub_path: Path | None = None,
) -> EvalReport:
    """Run full local evaluation (raw forget, stress suite, retain proxy, weight drift)."""
    orig_raw = predict(orig_model, forget_images, batch_size=batch_size)
    sub_raw = predict(sub_model, forget_images, batch_size=batch_size)
    ref_preds = orig_raw.argmax(dim=-1)
    counts = torch.bincount(ref_preds, minlength=NUM_CLASSES)
    absent_mask = counts == 0

    stress_metrics = {}
    for name, pert_imgs in make_stress_suite(forget_images, seed=seed).items():
        orig_p = predict(orig_model, pert_imgs, batch_size=batch_size)
        sub_p = predict(sub_model, pert_imgs, batch_size=batch_size)
        stress_metrics[name] = compute_stress_metrics(orig_p, sub_p, ref_preds)

    proxy_imgs = make_patchmix_proxy(forget_images, seed=seed)
    orig_proxy = predict(orig_model, proxy_imgs, batch_size=batch_size)
    sub_proxy = predict(sub_model, proxy_imgs, batch_size=batch_size)
    retain = _compute_retain_proxy_metrics(orig_model, sub_model, orig_proxy, sub_proxy, absent_mask)
    ckpt = validate_checkpoint(sub_path) if sub_path is not None else None
    return EvalReport(
        num_images=len(forget_images),
        num_active_classes=int((counts > 0).sum().item()),
        num_absent_classes=int(absent_mask.sum().item()),
        forget_raw=compute_stress_metrics(orig_raw, sub_raw, ref_preds),
        forget_stress=stress_metrics,
        retain_proxy=retain,
        checkpoint=ckpt,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Track 3 local validation harness.")
    parser.add_argument("--orig-path", type=Path, default=Path("model.pt"))
    parser.add_argument("--sub-path", type=Path, default=Path("model.pt"))
    parser.add_argument("--forget-path", type=Path, default=Path("forget_images.npz"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    orig_model = load_model(args.orig_path)
    sub_model = load_model(args.sub_path)
    images = load_forget_images(args.forget_path)
    if args.max_images > 0:
        images = images[: args.max_images]
    report = evaluate_candidate(orig_model, sub_model, images, batch_size=args.batch_size, sub_path=args.sub_path)
    print(json.dumps(asdict(report), indent=2))
    return 0 if (report.checkpoint is None or report.checkpoint.valid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
