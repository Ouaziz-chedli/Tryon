"""Unit tests for Track 3 Phase 3 local validation harness (CPU, synthetic)."""

from pathlib import Path
import inspect
import sys
import tempfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_harness import (
    compute_kl_at_temperature,
    compute_stress_metrics,
    compute_weight_drift,
    evaluate_candidate,
    make_patchmix_proxy,
    make_stress_suite,
    validate_checkpoint,
)


def test_stress_suite_shapes_and_dtypes() -> None:
    imgs = torch.randint(0, 256, (4, 224, 224, 3), dtype=torch.uint8)
    suite = make_stress_suite(imgs, seed=42)
    expected = {"hflip", "crop50", "rotate15", "grayscale", "color_jitter", "blur_s2", "noise_25", "jpeg_q30"}
    assert set(suite.keys()) == expected
    for tensor in suite.values():
        assert tensor.shape == imgs.shape
        assert tensor.dtype == torch.uint8


def test_patchmix_proxy_changes_pixels() -> None:
    imgs = torch.randint(0, 256, (6, 224, 224, 3), dtype=torch.uint8)
    proxy = make_patchmix_proxy(imgs, seed=7, grid=4)
    assert proxy.shape == imgs.shape and proxy.dtype == torch.uint8
    assert not torch.equal(proxy, imgs)


def test_identity_metrics_and_drift_are_zero() -> None:
    logits = torch.randn(8, 100)
    preds = logits.argmax(dim=-1)
    m = compute_stress_metrics(logits, logits, preds, temperature=2.0)
    assert m.top1_agree_orig == 1.0
    assert m.kl_t2_from_orig == 0.0
    assert m.ce_p50 >= 0.0 and m.ce_p90 >= m.ce_p50
    assert 0.0 <= m.frac_gt_95 <= 1.0
    assert torch.allclose(compute_kl_at_temperature(logits, logits, 2.0), torch.zeros(8), atol=1e-5)

    tiny = torch.nn.Linear(8, 4)
    assert compute_weight_drift(tiny, tiny) == (0.0, 0.0, 0.0)


def test_validate_checkpoint_missing_and_non_float32() -> None:
    missing = validate_checkpoint(Path("/tmp/does_not_exist_12345.pt"))
    assert not missing.valid and missing.file_size_mb == 0.0

    with tempfile.TemporaryDirectory() as tmp:
        bad_path = Path(tmp) / "bad.pt"
        torch.save({"w": torch.zeros(2, dtype=torch.float16)}, bad_path)
        res = validate_checkpoint(bad_path)
        assert not res.valid
        assert any("float32" in e for e in res.errors)


def test_evaluate_candidate_uses_out_of_sample_seed_999() -> None:
    assert inspect.signature(evaluate_candidate).parameters["seed"].default == 999
