"""Local TDD tests for trace_core (pure numpy/scipy, no torch/wmloc).

Run: python3 -m pytest tracks/01_watermark/tests/test_trace_core.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import trace_core as tc


def test_miou2_perfect_is_one():
    p = np.array([[0, 1], [1, 0]], np.uint8)
    assert tc.miou2(p, p) == pytest.approx(1.0)


def test_miou2_empty_both_is_one():
    z = np.zeros((4, 4), np.uint8)
    assert tc.miou2(z, z) == pytest.approx(1.0)


def test_miou2_half_overlap_known_value():
    # pred = left half, gt = top half of 4x4: class1 inter=4, union=12 -> 1/3;
    # class0 inter=4, union=12 -> 1/3; mean = 1/3
    p = np.zeros((4, 4), np.uint8)
    p[:, :2] = 1
    g = np.zeros((4, 4), np.uint8)
    g[:2, :] = 1
    assert tc.miou2(p, g) == pytest.approx(1.0 / 3.0)


def test_box_iou_known_values():
    assert tc.box_iou((0, 10, 0, 10), (0, 10, 0, 10)) == pytest.approx(1.0)
    assert tc.box_iou((0, 10, 0, 10), (20, 30, 20, 30)) == pytest.approx(0.0)
    # inter=25, union=100+100-25=175
    assert tc.box_iou((0, 10, 0, 10), (5, 15, 5, 15)) == pytest.approx(25.0 / 175.0)


def test_compute_fills_full_rect_is_one():
    m = np.zeros((1, 128, 128), np.uint8)
    m[0, 10:50, 20:90] = 1
    assert tc.compute_fills(m)[0] == pytest.approx(1.0)


def test_compute_fills_empty_is_one():
    m = np.zeros((1, 32, 32), np.uint8)
    assert tc.compute_fills(m)[0] == pytest.approx(1.0)


def test_compute_fills_half_is_half():
    m = np.zeros((1, 32, 32), np.uint8)
    m[0, 0:8, 0:8] = 1
    m[0, 8:16, 8:16] = 1  # bbox 16x16=256, sum=128 -> 0.5
    assert tc.compute_fills(m)[0] == pytest.approx(0.5)


def test_rect_subset_idx_picks_only_small_dense():
    m = np.zeros((3, 64, 64), np.uint8)
    m[0, 5:15, 5:25] = 1          # small dense rect -> rect
    m[1, 0:64, 0:64] = 1          # full image -> not rect (area)
    m[2, 0:32, 0:32] = 1
    m[2, 0:16, 16:32] = 0         # sparse -> not rect (fill)
    idx = tc.rect_subset_idx(m)
    assert idx.tolist() == [0]


def _coverage_scorer(gt):
    gx0, gy0 = None, None

    def score(box):
        y0, y1, x0, x1 = box
        gy0, gy1, gx0, gx1 = gt
        iy = max(0, min(y1, gy1) - max(y0, gy0))
        ix = max(0, min(x1, gx1) - max(x0, gx0))
        cov = (iy * ix) / ((gy1 - gy0) * (gx1 - gx0))
        over = ((y1 - y0) * (x1 - x0)) / ((gy1 - gy0) * (gx1 - gx0))
        return cov - 0.05 * max(0.0, over - 1.0)

    return score


def test_trace_edges_converges_to_tight_box():
    gt = (100, 200, 50, 150)
    start = (60, 240, 10, 190)
    box, best = tc.trace_edges(start, _coverage_scorer(gt), steps=(20, 10, 5), eps=1e-6)
    assert tuple(box) == gt


def test_trace_edges_flat_landscape_keeps_box():
    start = (60, 240, 10, 190)
    box, best = tc.trace_edges(start, lambda b: 0.5, steps=(8, 4), eps=0.02)
    assert tuple(box) == start
    assert best == pytest.approx(0.5)


def test_trace_edges_respects_min_side():
    gt = (100, 108, 50, 58)  # 8x8, below min_side
    start = (80, 128, 30, 78)
    box, _ = tc.trace_edges(start, _coverage_scorer(gt), steps=(8, 4), eps=1e-6, min_side=16)
    h, w = box[1] - box[0], box[3] - box[2]
    assert h >= 16 and w >= 16


def test_load_rect_boxes_prefers_v7(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n = 3
    np.savez_compressed("work_bassim_test_tmp.npz")  # placeholder removed below
    os.remove("work_bassim_test_tmp.npz")
    import shutil

    os.makedirs("work_bassim", exist_ok=True)
    b6 = np.array([[0, 4, 0, 4]] * n)
    b7 = np.array([[1, 5, 1, 5]] * n)
    np.savez_compressed("work_bassim/bbox_v6_pixelseal.npz",
                        tr_boxes=b6, tr_logp=np.ones(n), tr_facc=np.ones(n),
                        te_boxes=b6, te_logp=np.ones(n), te_facc=np.ones(n))
    oof, te = tc.load_rect_boxes("pixelseal")
    assert oof is not None and np.array_equal(oof[0], b6)
    np.savez_compressed("work_bassim/bbox_v7_pixelseal.npz",
                        tr_boxes=b7, tr_logp=np.ones(n), tr_facc=np.ones(n),
                        te_boxes=b7, te_logp=np.ones(n), te_facc=np.ones(n),
                        masks=np.zeros((n, 8, 8), np.uint8),
                        names=np.array(["a", "b", "c"]))
    oof, te = tc.load_rect_boxes("pixelseal")
    assert np.array_equal(oof[0], b7)
    shutil.rmtree("work_bassim")


def test_load_rect_boxes_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert tc.load_rect_boxes("trustmark") == (None, None)


def test_hf_energy_map_shape_and_standardization():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (2, 512, 512, 3), dtype=np.uint8)
    flat = np.full((2, 512, 512, 3), 128, np.uint8)
    en = tc.hf_energy_map(noise)
    ef = tc.hf_energy_map(flat)
    assert en.shape == (2, 1, 128, 128)
    assert abs(en.mean()) < 0.05 and abs(en.std() - 1.0) < 0.05
    assert np.all(ef == 0.0)


def test_hf_energy_map_localizes_noisy_patch():
    rng = np.random.default_rng(1)
    img = np.full((1, 512, 512, 3), 128, np.uint8)
    img[0, 0:256, 0:256] = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    e = tc.hf_energy_map(img)[0, 0]
    assert e[:64, :64].mean() > e[64:, 64:].mean() + 0.5
