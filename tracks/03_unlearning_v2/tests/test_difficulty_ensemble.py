"""Unit tests for the pure helpers of run_difficulty_ensemble (no GPU/model needed)."""

import torch

from eval_harness import compute_spearman_rank_corr
from run_difficulty_ensemble import (
    brightness_jitter,
    build_tta_views,
    crop_rescale,
    estimate_feature_knn,
    estimate_margin,
    fold_rhos,
    orient_and_rank,
    rankify,
    weighted_rank_mean,
)


def test_crop_rescale_preserves_shape_and_dtype() -> None:
    imgs = torch.randint(0, 256, (3, 224, 224, 3), dtype=torch.uint8)
    out = crop_rescale(imgs)
    assert out.shape == imgs.shape and out.dtype == torch.uint8


def test_brightness_jitter_bounds_and_determinism() -> None:
    imgs = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
    a = brightness_jitter(imgs, seed=11)
    b = brightness_jitter(imgs, seed=11)
    assert torch.equal(a, b) and a.dtype == torch.uint8
    assert float(a.min()) >= 0.0 and float(a.max()) <= 255.0


def test_build_tta_views_count_matches_num_views() -> None:
    imgs = torch.zeros((2, 224, 224, 3), dtype=torch.uint8)
    assert len(build_tta_views(imgs, 4)) == 4
    assert len(build_tta_views(imgs, 2)) == 2


def test_feature_knn_finds_expected_neighbour_distance() -> None:
    feats = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    knn = estimate_feature_knn(feats, k=1)
    # normalized: dist(0,1) = sqrt(2 - 2*~0.994) is small; dist(0,2) = sqrt(2) is the max
    assert knn[2].item() > knn[0].item() and knn.shape == (3,)


def test_margin_is_top1_minus_top2() -> None:
    logits = torch.tensor([[5.0, 3.0, 1.0], [0.0, 4.0, 2.0]])
    assert torch.equal(estimate_margin(logits), torch.tensor([2.0, 2.0]))


def test_orient_and_rank_flips_negative_sign() -> None:
    ce_ref = torch.tensor([0.1, 0.5, 0.9, 2.0])
    ranks, rho, sign = orient_and_rank(torch.tensor([9.0, 5.0, 3.0, 1.0]), ce_ref)
    assert sign == -1.0 and rho == -1.0
    assert torch.equal(ranks, rankify(torch.tensor([1.0, 5.0, 9.0, 13.0])))


def test_weighted_rank_mean_falls_back_to_uniform_when_all_rho_zero() -> None:
    oriented = {"a": rankify(torch.tensor([1.0, 2.0])), "b": rankify(torch.tensor([2.0, 1.0]))}
    combined, weights = weighted_rank_mean(oriented, {"a": 0.0, "b": 0.0})
    assert weights == {"a": 1.0, "b": 1.0}
    assert torch.equal(combined, torch.ones(2))


def test_fold_rhos_matches_global_on_identity_folds() -> None:
    score = torch.arange(8).float()
    ce_ref = torch.arange(8).float()
    folds = [(torch.tensor([0, 1, 2, 3]), torch.tensor([]), torch.tensor([4, 5, 6, 7]))]
    assert fold_rhos(score, ce_ref, folds) == [compute_spearman_rank_corr(score, ce_ref)]


def test_rankify_orders_ascending() -> None:
    ranks = rankify(torch.tensor([30.0, 10.0, 20.0]))
    assert torch.equal(ranks, torch.tensor([2.0, 0.0, 1.0]))
