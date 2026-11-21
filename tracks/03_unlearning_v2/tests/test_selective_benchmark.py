"""Unit tests for the Intra-Class and Cross-Class selective benchmark on forget_images.npz."""

import torch

from selective_benchmark import (
    evaluate_real_subset,
    make_cross_class_split,
    make_intra_class_split,
    score_selective_candidate,
)


def test_intra_and_cross_splits_are_disjoint_and_exhaustive() -> None:
    labels = torch.arange(100) % 10
    in_f, in_r = make_intra_class_split(labels, seed=42)
    assert len(in_f) + len(in_r) == 100
    assert len(set(in_f.tolist()).intersection(set(in_r.tolist()))) == 0
    assert set(labels[in_f].unique().tolist()) == set(labels[in_r].unique().tolist())

    cr_f, cr_r = make_cross_class_split(labels, seed=42)
    assert len(cr_f) + len(cr_r) == 100
    assert len(set(labels[cr_f].tolist()).intersection(set(labels[cr_r].tolist()))) == 0


def test_selective_scoring_rewards_instance_selectivity_and_cross_retain() -> None:
    orig = torch.zeros(20, 10)
    orig[torch.arange(20), torch.arange(20) % 10] = 8.0
    labels = orig.argmax(dim=-1)

    sub_f = orig.clone()
    sub_f[torch.arange(20), labels] = 2.4
    sub_r_good = orig.clone()
    sub_r_good[torch.arange(20), labels] = 7.5
    sub_r_collapse = sub_f.clone()

    m_f = evaluate_real_subset(sub_f, orig, labels)
    m_r_good = evaluate_real_subset(sub_r_good, orig, labels)
    m_r_bad = evaluate_real_subset(sub_r_collapse, orig, labels)

    s_good = score_selective_candidate(m_f, m_r_good, m_f, m_r_good)
    s_bad = score_selective_candidate(m_f, m_r_bad, m_f, m_r_good)
    assert s_good > s_bad >= 0.0


if __name__ == "__main__":
    test_intra_and_cross_splits_are_disjoint_and_exhaustive()
    test_selective_scoring_rewards_instance_selectivity_and_cross_retain()
    print("SELECTIVE_BENCHMARK_TESTS_OK")
