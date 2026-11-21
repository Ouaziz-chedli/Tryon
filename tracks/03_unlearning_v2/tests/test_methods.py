"""CPU unit tests for Track 3 unlearning strategies and augmentations."""

from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augment import augment_training_batch
from contracts import MethodConfig
from methods.common import (
    compute_forget_loss,
    get_batch_orig_logits,
    gold_mimic_loss,
    l2_anchor_loss,
    npo_loss,
    two_sided_npo_loss,
)
from methods.lora_unlearn import LoRALinear
from methods.proxies import make_retain_proxy
from methods.salun_npo import compute_tempered_saliency


def test_augment_training_batch_preserves_shape_and_dtype() -> None:
    imgs = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8)
    for mode in ("none", "moderate", "strong"):
        out = augment_training_batch(imgs, mode=mode)
        assert out.shape == imgs.shape
        assert out.dtype == torch.uint8


def test_lora_linear_merge_matches_forward() -> None:
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 8)
    lora = LoRALinear(base, rank=4)
    with torch.no_grad():
        lora.lora_b.normal_(std=0.1)
    x = torch.randn(5, 16)
    y_lora = lora(x)
    merged = lora.merge_to_linear()
    assert isinstance(merged, torch.nn.Linear)
    assert torch.allclose(y_lora, merged(x), atol=1e-5)


def test_npo_loss_finite_and_differentiable() -> None:
    orig = torch.randn(6, 100)
    sub = orig.clone().requires_grad_(True)
    loss = npo_loss(sub, orig, orig.argmax(dim=-1), beta=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert sub.grad is not None and torch.isfinite(sub.grad).all()


def test_tempered_saliency_shape_and_ratio() -> None:
    tiny = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 10))
    imgs = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8)
    labels = torch.zeros(4, dtype=torch.long)
    masks = compute_tempered_saliency(tiny, imgs, labels, keep_ratio=0.25, temperature=2.0, batch_size=2)
    total = sum(m.numel() for m in masks.values())
    kept = sum(int(m.sum().item()) for m in masks.values())
    assert 0.20 <= (kept / total) <= 0.30
    assert MethodConfig(name="test", strategy="salun_npo").temperature == 2.0
    assert MethodConfig(name="bt", strategy="lora_npo", loss_kind="bad_teacher").loss_kind == "bad_teacher"


def test_alternative_retain_proxies_shapes() -> None:
    imgs = torch.randint(0, 256, (6, 224, 224, 3), dtype=torch.uint8)
    for kind in ("patchmix", "mixup", "structured_noise"):
        out = make_retain_proxy(imgs, kind=kind, seed=0)
        assert out.shape == imgs.shape and out.dtype == torch.uint8
        assert not torch.equal(out, imgs)


def test_two_sided_npo_loss_pushes_low_and_pulls_high() -> None:
    orig = torch.zeros(3, 10)
    orig[:, 0] = 8.0
    labels = torch.zeros(3, dtype=torch.long)
    sub = torch.tensor([
        [8.0] + [0.0] * 9,   # ce << target_ce -> push
        [0.4] + [0.0] * 9,   # ce in [0.45, 0.65] -> neither
        [-5.0] + [0.0] * 9,  # ce >> upper_ce -> pull
    ], requires_grad=True)
    loss = two_sided_npo_loss(sub, orig, labels, target_ce=0.45, upper_ce=0.65, pullback_weight=0.5)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert sub.grad is not None and torch.isfinite(sub.grad).all()
    assert sub.grad[1].abs().max().item() == 0.0


def test_gold_mimic_loss_minimum_and_gradient() -> None:
    orig = torch.randn(4, 10)
    labels = orig.argmax(dim=-1)
    sub = orig.clone().requires_grad_(True)
    loss = gold_mimic_loss(sub, orig, labels, target_prob=0.60, temperature=2.0)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert sub.grad is not None and torch.isfinite(sub.grad).all()


def test_compute_forget_loss_and_orig_aug_helper() -> None:
    orig = torch.randn(4, 10)
    sub = orig.clone().requires_grad_(True)
    labels = orig.argmax(dim=-1)
    for kind in ("npo", "two_sided_npo", "gold_mimic"):
        cfg = MethodConfig(name=kind, strategy="salun_npo", loss_kind=kind)
        loss = compute_forget_loss(sub, orig, labels, cfg)
        assert torch.isfinite(loss)

    tiny = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 10))
    imgs = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    cached = torch.zeros(2, 10)
    assert torch.equal(get_batch_orig_logits(tiny, imgs, cached, "none"), cached)
    out_aug = get_batch_orig_logits(tiny, imgs, cached, "moderate")
    assert out_aug.shape == (2, 10) and not torch.equal(out_aug, cached)


def test_l2_anchor_loss_ignores_unanchored_params() -> None:
    model = torch.nn.Linear(4, 2)
    loss = l2_anchor_loss(model, {"weight": model.weight.detach().clone()})
    assert torch.isfinite(loss) and loss.item() == 0.0


def test_tempered_saliency_shuffled_seed_sensitivity() -> None:
    tiny = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 10))
    imgs = torch.randint(0, 256, (10, 224, 224, 3), dtype=torch.uint8)
    labels = torch.arange(10) % 10
    m1 = compute_tempered_saliency(tiny, imgs, labels, keep_ratio=0.25, num_batches=1, batch_size=2, seed=1)
    m1b = compute_tempered_saliency(tiny, imgs, labels, keep_ratio=0.25, num_batches=1, batch_size=2, seed=1)
    m2 = compute_tempered_saliency(tiny, imgs, labels, keep_ratio=0.25, num_batches=1, batch_size=2, seed=2)
    assert torch.equal(m1["1.weight"], m1b["1.weight"])
    assert not torch.equal(m1["1.weight"], m2["1.weight"])
