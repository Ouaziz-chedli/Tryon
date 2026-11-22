"""Ensemble v2 + v3 probabilities (automatically falls back to v2 if v3 doesn't beat it on OOF CV),
tune threshold + morphological cleanup / rectangle snap on OOF CV, and write submission_v3.pt.

    python work_bassim/postprocess_submit_v3.py --out output/submission_v3.pt [--submit]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score  # noqa: E402

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"


def clean_mask(m: np.ndarray, min_cc: int, rect_fill: float) -> np.ndarray:
    out = m.copy()
    if min_cc > 0 and out.sum() > 0:
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        for c in range(1, n_lbl):
            if stats[c, cv2.CC_STAT_AREA] < min_cc and out.sum() > stats[c, cv2.CC_STAT_AREA]:
                out[lbl == c] = 0
    if rect_fill <= 1.0 and out.sum() > 0:
        ys, xs = np.where(out > 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        if out.sum() / ((y1 - y0) * (x1 - x0)) >= rect_fill:
            out[y0:y1, x0:x1] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/submission_v3.pt")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    test_names = [m["image_name"] for m in LocalisationDataset("release", "test").meta]
    pred = {}
    oof_preds, oof_gts = [], []

    for method in ("pixelseal", "trustmark", "stegastamp"):
        has_v2 = os.path.exists(f"work_bassim/oof_v2_{method}.npz")
        has_v3 = os.path.exists(f"work_bassim/oof_v3_{method}.npz")
        assert has_v2 or has_v3, f"Missing predictions for {method}"

        if has_v2 and has_v3:
            z2o, z2t = np.load(f"work_bassim/oof_v2_{method}.npz"), np.load(f"work_bassim/test_v2_{method}.npz")
            z3o, z3t = np.load(f"work_bassim/oof_v3_{method}.npz"), np.load(f"work_bassim/test_v3_{method}.npz")
            p2o, p3o = z2o["probs"].astype(np.float32), z3o["probs"].astype(np.float32)
            p2t, p3t = z2t["probs"].astype(np.float32), z3t["probs"].astype(np.float32)
            masks_oof = z3o["masks"]
            names = z3t["names"].tolist()
            weights = (0.0, 0.3, 0.5, 0.7, 1.0)
        elif has_v3:
            z3o, z3t = np.load(f"work_bassim/oof_v3_{method}.npz"), np.load(f"work_bassim/test_v3_{method}.npz")
            p2o = p3o = z3o["probs"].astype(np.float32)
            p2t = p3t = z3t["probs"].astype(np.float32)
            masks_oof = z3o["masks"]
            names = z3t["names"].tolist()
            weights = (1.0,)
        else:
            z2o, z2t = np.load(f"work_bassim/oof_v2_{method}.npz"), np.load(f"work_bassim/test_v2_{method}.npz")
            p2o = p3o = z2o["probs"].astype(np.float32)
            p2t = p3t = z2t["probs"].astype(np.float32)
            masks_oof = z2o["masks"]
            names = z2t["names"].tolist()
            weights = (0.0,)

        best = (-1.0, 0.5, 0.5, 0, 2.0)  # score, w_v3, thr, min_cc, rect_fill
        for w in weights:
            po = w * p3o + (1.0 - w) * p2o
            for thr in np.linspace(0.25, 0.75, 21):
                base = [(p > thr).astype(np.uint8) for p in po]
                s0 = score(base, masks_oof)
                if s0 > best[0]:
                    best = (s0, w, float(thr), 0, 2.0)
                for min_cc in (0, 8, 20):
                    for rf in (2.0, 0.90, 0.85):
                        if min_cc == 0 and rf > 1.0:
                            continue
                        pp = [clean_mask(m, min_cc, rf) for m in base]
                        s = score(pp, masks_oof)
                        if s > best[0]:
                            best = (s, w, float(thr), min_cc, rf)

        bs, bw, bthr, bcc, brf = best
        print(f"[{method}] tuned OOF={bs:.4f} (w_v3={bw:.1f}, thr={bthr:.3f}, min_cc={bcc}, rect_fill={brf})")
        po = bw * p3o + (1.0 - bw) * p2o
        pt = bw * p3t + (1.0 - bw) * p2t
        oof_preds.extend([clean_mask((p > bthr).astype(np.uint8), bcc, brf) for p in po])
        oof_gts.extend(masks_oof)
        for n, p in zip(names, pt):
            pred[n] = clean_mask((p > bthr).astype(np.uint8), bcc, brf)

    print(f"Overall V3 Ensemble OOF mIoU (510 real imgs): {score(oof_preds, oof_gts):.4f}")
    missing = [n for n in test_names if n not in pred]
    assert not missing, f"missing {len(missing)}"
    masks = torch.from_numpy(np.stack([pred[n] for n in test_names])).bool()
    assert masks.shape == (len(test_names), 128, 128) and len(set(test_names)) == len(test_names)
    torch.save({"image_names": test_names, "masks": masks}, a.out)
    size_mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}: {tuple(masks.shape)}, {size_mb:.2f} MB, frac={masks.float().mean():.4f}")
    assert size_mb < 20

    if a.submit:
        import requests
        key = os.environ["TRYON_API_KEY"]
        with open(a.out, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": key},
                files={"file": (os.path.basename(a.out), f, "application/octet-stream")},
                timeout=(10, 600),
            )
        print("HTTP", r.status_code, r.text[:500])


if __name__ == "__main__":
    main()
