"""CPU tests for virtual-gold patch masking and score invariants."""

import torch

from research_virtual_gold import kl_at_temperature, patch_mask_views, score_virtual_gold


def test_patch_mask_views_preserves_shape_and_masks_expected_fraction() -> None:
    images = torch.arange(3 * 224 * 224 * 3, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(3, 224, 224, 3)
    masked = patch_mask_views(images, seed=7, fraction=0.5, views=2)

    assert masked.shape == (6, 224, 224, 3)
    assert masked.dtype == torch.uint8
    changed = (masked[:3] != images).any(dim=-1)
    changed_patches = changed.reshape(3, 14, 16, 14, 16).any(dim=(2, 4))
    assert torch.equal(changed_patches.sum(dim=(1, 2)), torch.tensor([98, 98, 98]))


def test_virtual_gold_score_is_finite_for_original_and_candidate() -> None:
    torch.manual_seed(11)
    orig = torch.randn(5, 4)
    candidate = orig + torch.randn(5, 4) * 0.5
    masked = torch.stack([candidate, candidate + torch.randn_like(candidate) * 0.05])
    labels = orig.argmax(dim=-1)
    masked_logits = masked.reshape(-1, 4)

    matching = score_virtual_gold(orig, candidate, masked_logits, labels, views=2)
    original = score_virtual_gold(orig, orig, masked_logits, labels, views=2)

    assert torch.isfinite(torch.tensor(matching["forget_score_surrogate"]))
    assert torch.isfinite(torch.tensor(original["forget_score_surrogate"]))


def test_kl_at_temperature_matches_reference_distribution() -> None:
    teacher = torch.randn(7, 4)
    student = torch.randn(7, 4)
    temperature = 2.0
    expected = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student / temperature, dim=-1),
        torch.nn.functional.softmax(teacher / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1) * temperature**2
    assert torch.allclose(kl_at_temperature(teacher, student, temperature), expected)


if __name__ == "__main__":
    test_patch_mask_views_preserves_shape_and_masks_expected_fraction()
    test_virtual_gold_score_is_finite_for_original_and_candidate()
    test_kl_at_temperature_matches_reference_distribution()
    print("VIRTUAL_GOLD_TESTS_OK")
