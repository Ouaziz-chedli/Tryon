"""CPU unit tests for Track 3 unlearning strategies and augmentations."""

from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augment import augment_training_batch
from contracts import MethodConfig
from methods.common import npo_loss
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
