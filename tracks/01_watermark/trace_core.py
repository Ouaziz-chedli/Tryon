"""Pure numpy/scipy helpers for rect analysis + edge-tracing.

Dependency-free (no torch / cv2 / wmloc) so it can be unit-tested locally.
Cluster scripts (diag_rect, rect_search_v3, train_rects, postprocess_submit_v5)
import from here; work_bassim is on PYTHONPATH in the Slurm jobs.
"""
import os

import numpy as np
from scipy.ndimage import uniform_filter


def miou2(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean IoU over the 2 classes {0, 1}; IoU=1 when union is empty."""
    s = 0.0
    for c in (0, 1):
        p = pred == c
        g = gt == c
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        s += 1.0 if union == 0 else inter / union
    return s / 2.0


def box_iou(a, b) -> float:
    """IoU of boxes (y0, y1, x0, x1)."""
    iy0, iy1 = max(a[0], b[0]), min(a[1], b[1])
    ix0, ix1 = max(a[2], b[2]), min(a[3], b[3])
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    inter = (iy1 - iy0) * (ix1 - ix0)
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / max(ua, 1)


def compute_fills(masks: np.ndarray) -> np.ndarray:
    """Fill = nonzero pixels / tight-bbox area per mask (1.0 when empty).

    Replicates the rect_part definition used in train_rects (mask itself,
    no foreground/background inversion).
    """
    out = np.ones(len(masks), np.float64)
    for i, g in enumerate(masks):
        if np.any(g):
            ys, xs = np.where(g > 0)
            out[i] = float(g.sum()) / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    return out


def rect_subset_idx(masks: np.ndarray, area_max: float = 0.55, fill_min: float = 0.95) -> np.ndarray:
    """Indices of small dense ("true rect") images."""
    area = masks.reshape(len(masks), -1).mean(axis=1)
    fills = compute_fills(masks)
    return np.where((area <= area_max) & (fills > fill_min))[0]


def trace_edges(box, score_fn, steps=(8, 4), eps=0.02, min_side=16,
                bounds=(0, 512, 0, 512), min_score=0.6):
    """Shrink each edge inward while the score stays within eps of the best.

    box: (y0, y1, x0, x1). score_fn(box_tuple) -> float. Pure function:
    no torch, no decoder — the caller wraps _score_boxes.
    Returns (traced_box_tuple, best_score). Boxes that don't decode
    (best < min_score) are returned unchanged.
    """
    y0, y1, x0, x1 = (int(v) for v in box)
    best = float(score_fn((y0, y1, x0, x1)))
    if best < min_score:
        return (y0, y1, x0, x1), best
    by0, by1, bx0, bx1 = bounds
    for s in steps:
        if s <= 0:
            continue
        moved = True
        while moved:
            moved = False
            for cand in ((y0 + s, y1, x0, x1), (y0, y1 - s, x0, x1),
                         (y0, y1, x0 + s, x1), (y0, y1, x0, x1 - s)):
                cy0, cy1, cx0, cx1 = cand
                if cy1 - cy0 < min_side or cx1 - cx0 < min_side:
                    continue
                if cy0 < by0 or cy1 > by1 or cx0 < bx0 or cx1 > bx1:
                    continue
                sc = float(score_fn(cand))
                if sc >= best - eps:
                    y0, y1, x0, x1 = cand
                    best = max(best, sc)
                    moved = True
                    break
    return (y0, y1, x0, x1), best


def load_rect_boxes(method: str, workdir: str = "."):
    """Load decoder-verified rect boxes, preferring bbox_v7 over bbox_v6.

    Returns ((oof_boxes, oof_score, oof_facc), (te_boxes, te_score, te_facc))
    or (None, None) when neither file exists. Score key is tr/te_logp
    when present, else tr/te_bacc (same fallback as postprocess_submit_v5).
    """
    for tag in ("bbox_v7", "bbox_v6"):
        p = os.path.join(workdir, f"work_bassim/{tag}_{method}.npz")
        if os.path.exists(p):
            z = np.load(p)
            s_key = "tr_logp" if "tr_logp" in z else "tr_bacc"
            t_key = "te_logp" if "te_logp" in z else "te_bacc"
            return ((z["tr_boxes"], z[s_key], z["tr_facc"]),
                    (z["te_boxes"], z[t_key], z["te_facc"]))
    return None, None


def hf_energy_map(imgs: np.ndarray) -> np.ndarray:
    """High-frequency energy map at 128x128, standardized per image.

    Mirrors the patch-CNN front end (patch_feasibility.Net.forward):
    residue = gray - 3x3 mean (reflect), energy = residue^2, downsampled
    by 4x4 block mean, then per-image zero-mean / unit-std.
    Returns (N, 1, 128, 128) float32.
    """
    g = np.asarray(imgs).astype(np.float32).mean(axis=-1)
    r = g - uniform_filter(g, size=3, mode="reflect")
    e = r * r
    n = len(e)
    e128 = e.reshape(n, 128, 4, 128, 4).mean(axis=(2, 4))
    mu = e128.mean(axis=(1, 2), keepdims=True)
    sd = e128.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((e128 - mu) / sd).astype(np.float32)[:, None]
