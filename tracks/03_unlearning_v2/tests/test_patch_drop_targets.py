"""Unit tests for per-image Patch-Dropout Virtual Gold target estimation and loss."""

import torch
import torch.nn.functional as F

from methods.patch_drop_targets import (
    calibrate_per_image_targets,
    mask_random_patches,
    per_sample_calibrated_loss,
)


def test_mask_random_patches_preserves_shape_and_drops_patches() -> None:
    imgs = torch.full((4, 224, 224, 3), 250, dtype=torch.uint8)
    masked = mask_random_patches(imgs, drop_prob=0.45, patch_size=16, seed=42)
    assert masked.shape == imgs.shape and masked.dtype == torch.uint8
    frac_masked = float((masked == 128).float().mean().item())
    assert 0.25 < frac_masked < 0.65


def test_calibrate_per_image_targets_preserves_ordering_and_mean_ce() -> None:
    orig = torch.zeros(20, 10)
    labels = torch.arange(20) % 10
    orig[torch.arange(20), labels] = 10.0
    pd = orig.clone()
    pd[torch.arange(20), labels] = torch.linspace(8.0, 1.5, 20)
    z_tgt, ce_tgt = calibrate_per_image_targets(pd, orig, labels, target_mean_ce=1.75)
    assert ce_tgt[0] < ce_tgt[-1]
    actual_ce = F.cross_entropy(z_tgt, labels, reduction="none")
    assert abs(float(actual_ce.mean().item()) - float(ce_tgt.mean().item())) < 0.15


def test_per_sample_calibrated_loss_produces_finite_gradients() -> None:
    sub = torch.randn(8, 10, requires_grad=True)
    orig = torch.randn(8, 10)
    labels = orig.argmax(dim=-1)
    tgt_logits, tgt_ce = calibrate_per_image_targets(orig, orig, labels, target_mean_ce=1.75)
    loss = per_sample_calibrated_loss(sub, orig, tgt_logits, tgt_ce, labels)
    loss.backward()
    assert torch.isfinite(loss) and sub.grad is not None and sub.grad.abs().sum() > 0


if __name__ == "__main__":
    test_mask_random_patches_preserves_shape_and_drops_patches()
    test_calibrate_per_image_targets_preserves_ordering_and_mean_ce()
    test_per_sample_calibrated_loss_produces_finite_gradients()
    print("PATCH_DROP_TESTS_OK")
