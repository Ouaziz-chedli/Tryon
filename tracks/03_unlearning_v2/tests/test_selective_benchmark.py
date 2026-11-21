"""Unit tests for the 3-Fold Weighted Rotating Intra+Inter-Class Selective Benchmark."""

import torch

from selective_benchmark import (
    FoldResult,
    aggregate_weighted_folds,
    evaluate_real_subset,
    make_altered_subset,
    make_rotating_class_folds,
    score_single_fold,
)


def test_rotating_class_folds_cover_all_classes_disjointly() -> None:
    labels = torch.arange(180) % 15
    folds = make_rotating_class_folds(labels, num_folds=3, seed=42)
    assert len(folds) == 3
    seen_target_classes: set[int] = set()
    for f_idx, r_in_idx, r_out_idx in folds:
        assert len(f_idx) + len(r_in_idx) + len(r_out_idx) == 180
        f_set, in_set, out_set = set(f_idx.tolist()), set(r_in_idx.tolist()), set(r_out_idx.tolist())
        assert not (f_set & in_set) and not (f_set & out_set) and not (in_set & out_set)
        target_c = set(labels[f_idx].unique().tolist())
        assert target_c == set(labels[r_in_idx].unique().tolist())
        assert not (target_c & set(labels[r_out_idx].unique().tolist()))
        seen_target_classes |= target_c
    assert seen_target_classes == set(range(15))


def test_make_altered_subset_preserves_shape_and_alters_pixels() -> None:
    imgs = torch.randint(0, 256, (8, 224, 224, 3), dtype=torch.uint8)
    alt = make_altered_subset(imgs, seed=123)
    assert alt.shape == imgs.shape and alt.dtype == torch.uint8
    assert not torch.equal(imgs, alt)


def test_weighted_fold_scoring_and_aggregation() -> None:
    orig = torch.zeros(30, 10)
    orig[torch.arange(30), torch.arange(30) % 10] = 8.0
    labels = orig.argmax(dim=-1)

    sub_f = orig.clone()
    sub_f[torch.arange(30), labels] = 1.2
    sub_r_in = orig.clone()
    sub_r_in[torch.arange(30), labels] = 6.5
    sub_r_out = orig.clone()
    sub_r_out[torch.arange(30), labels] = 7.8

    m_f = evaluate_real_subset(sub_f, orig, labels)
    m_in = evaluate_real_subset(sub_r_in, orig, labels)
    m_out = evaluate_real_subset(sub_r_out, orig, labels)
    m_collapse = evaluate_real_subset(sub_f, orig, labels)

    s_good = score_single_fold(m_f, m_f, m_in, m_in, m_out)
    s_collapse = score_single_fold(m_f, m_f, m_collapse, m_collapse, m_out)
    assert s_good > s_collapse >= 0.0

    fold0 = FoldResult(0, 20, m_f, m_f, m_in, m_in, m_out, 0.8, 0.9, s_good)
    fold1 = FoldResult(1, 40, m_f, m_f, m_in, m_in, m_out, 0.7, 0.8, s_good)
    full_m = {
        "full_ce_mean": 2.1, "full_ce_std": 2.3, "full_ce_p50": 0.95,
        "full_frac_gt_95": 0.14, "full_proxy_kl_t2": 0.041,
        "full_absent_kl_t2": 0.022, "full_delta_l2": 3.9,
    }
    rep = aggregate_weighted_folds("AT_test", [fold0, fold1], full_m)
    assert abs(rep.weighted_selective_score - s_good) < 1e-4
    assert rep.fold_score_std == 0.0


if __name__ == "__main__":
    test_rotating_class_folds_cover_all_classes_disjointly()
    test_make_altered_subset_preserves_shape_and_alters_pixels()
    test_weighted_fold_scoring_and_aggregation()
    print("SELECTIVE_3FOLD_BENCHMARK_TESTS_OK")
