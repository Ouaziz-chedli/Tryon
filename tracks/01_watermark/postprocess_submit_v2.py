"""Tune post-processing on oof_v2_<method>.npz (strictly verified via CV) and build/submit submission_v2.pt.

    python work_bassim/postprocess_submit_v2.py --out output/submission_v2.pt --submit
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score  # noqa: E402

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"


def maybe_snap_rect(mask: np.ndarray, rect_fill_thr: float) -> np.ndarray:
    """If predicted foreground fills >= rect_fill_thr of its bounding box, snap to the filled bbox."""
    if rect_fill_thr > 1.0 or mask.sum() == 0:
        return mask
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    area = (y1 - y0) * (x1 - x0)
    if mask.sum() / area >= rect_fill_thr:
        out = np.zeros_like(mask)
        out[y0:y1, x0:x1] = 1
        return out
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/submission_v2.pt")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    test_names = [m["image_name"] for m in LocalisationDataset("release", "test").meta]
    pred = {}
    oof_preds, oof_gts = [], []

    for method in ("pixelseal", "trustmark", "stegastamp"):
        zo = np.load(f"work_bassim/oof_v2_{method}.npz")
        zt = np.load(f"work_bassim/test_v2_{method}.npz")
        probs_oof = zo["probs"].astype(np.float32)
        masks_oof = zo["masks"]

        best_s, best_thr, best_r = -1.0, 0.5, 2.0
        for thr in np.linspace(0.25, 0.75, 21):
            base_masks = [(p > thr).astype(np.uint8) for p in probs_oof]
            for r in (2.0, 0.90, 0.85, 0.80):
                pp = [maybe_snap_rect(m, r) for m in base_masks]
                s = score(pp, masks_oof)
                if s > best_s:
                    best_s, best_thr, best_r = s, float(thr), r

        print(f"[{method}] tuned OOF mIoU={best_s:.4f} (thr={best_thr:.3f}, rect_snap={best_r})")
        oof_preds.extend([maybe_snap_rect((p > best_thr).astype(np.uint8), best_r) for p in probs_oof])
        oof_gts.extend(masks_oof)

        probs_te = zt["probs"].astype(np.float32)
        for n, p in zip(zt["names"].tolist(), probs_te):
            pred[n] = maybe_snap_rect((p > best_thr).astype(np.uint8), best_r)

    print(f"Overall OOF mIoU (510 real imgs): {score(oof_preds, oof_gts):.4f}")
    missing = [n for n in test_names if n not in pred]
    assert not missing, f"missing {len(missing)}"
    masks = torch.from_numpy(np.stack([pred[n] for n in test_names])).bool()
    assert masks.shape == (len(test_names), 128, 128) and len(set(test_names)) == len(test_names)
    torch.save({"image_names": test_names, "masks": masks}, a.out)
    size_mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}: {tuple(masks.shape)}, {size_mb:.1f} MB, frac={masks.float().mean():.3f}")
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
