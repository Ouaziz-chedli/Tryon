"""Unit tests for Track 3 unlearning pipeline (CPU, synthetic ViT-tiny/mock)."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import UnlearnConfig
from metrics import (
    build_diagnostics_report,
    compute_kl_at_temperature,
    make_augmented_copy,
    make_patchmix_proxy,
)
from saliency import compute_saliency_masks
from unlearner import _npo_loss


def test_patchmix_and_augmented_shapes() -> None:
    imgs = torch.randint(0, 256, (8, 224, 224, 3), dtype=torch.uint8)
    mixed = make_patchmix_proxy(imgs, seed=0, grid=4)
    aug = make_augmented_copy(imgs, seed=1)
    assert mixed.shape == imgs.shape and mixed.dtype == torch.uint8
    assert aug.shape == imgs.shape and aug.dtype == torch.uint8
    assert not torch.equal(mixed, imgs)


def test_kl_and_diagnostics_report() -> None:
    logits = torch.randn(10, 100)
    kl_zero = compute_kl_at_temperature(logits, logits, temperature=2.0)
    assert torch.allclose(kl_zero, torch.zeros(10), atol=1e-5)

    report = build_diagnostics_report(logits, logits, logits, logits, logits, temperature=2.0)
    assert report.num_images == 10
    assert report.top1_agreement_orig == 1.0
    assert report.retain_proxy_kl < 1e-5


def test_npo_loss_is_bounded_and_finite() -> None:
    orig_logits = torch.randn(8, 100)
    sub_logits = orig_logits.clone().requires_grad_(True)
    labels = orig_logits.argmax(dim=-1)
    loss = _npo_loss(sub_logits, orig_logits, labels, beta=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert sub_logits.grad is not None and torch.isfinite(sub_logits.grad).all()


def test_saliency_mask_respects_keep_ratio() -> None:
    tiny = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 100),
    )
    imgs = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8)
    labels = torch.zeros(4, dtype=torch.long)
    masks = compute_saliency_masks(tiny, imgs, labels, keep_ratio=0.2, batch_size=2, num_batches=2)
    total = sum(m.numel() for m in masks.values())
    kept = sum(int(m.sum().item()) for m in masks.values())
    assert 0.15 <= (kept / total) <= 0.25


def test_unlearn_config_defaults() -> None:
    cfg = UnlearnConfig()
    assert cfg.temperature == 2.0
    assert cfg.num_classes == 100
