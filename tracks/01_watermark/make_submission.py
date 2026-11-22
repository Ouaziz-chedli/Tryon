"""Assemble the Track 1 submission from per-method test predictions and (optionally) upload it.

    python work_bassim/make_submission.py --out output/submission_v1.pt            # build + validate only
    python work_bassim/make_submission.py --out output/submission_v1.pt --submit   # also upload

Reads work_bassim/test_<method>.npz (probs, names, thr). API key is read from env TRYON_API_KEY.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from wmloc.data import LocalisationDataset  # noqa: E402

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/submission_v1.pt")
    ap.add_argument("--prefix", default="test_", help="npz prefix in work_bassim/")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    test_names = [m["image_name"] for m in LocalisationDataset("release", "test").meta]
    pred = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        z = np.load(f"work_bassim/{a.prefix}{method}.npz")
        thr = float(z["thr"])
        print(f"{method}: {len(z['names'])} images, thr={thr:.2f}, OOF mIoU={float(z['oof_miou']):.4f}")
        for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32)):
            pred[n] = (p > thr).astype(np.uint8)

    missing = [n for n in test_names if n not in pred]
    assert not missing, f"{len(missing)} test images without a prediction, e.g. {missing[:3]}"
    masks = torch.from_numpy(np.stack([pred[n] for n in test_names])).bool()
    assert masks.shape == (len(test_names), 128, 128) and len(set(test_names)) == len(test_names)
    torch.save({"image_names": test_names, "masks": masks}, a.out)
    size_mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}: {tuple(masks.shape)}, {size_mb:.1f} MB, mean watermarked fraction={masks.float().mean():.3f}")
    assert size_mb < 20, "file too large"

    if a.submit:
        import requests
        key = os.environ["TRYON_API_KEY"]
        with open(a.out, "rb") as f:
            r = requests.post(f"{BASE_URL}/submit/{TASK_ID}", headers={"X-API-Key": key},
                              files={"file": (os.path.basename(a.out), f, "application/octet-stream")}, timeout=(10, 600))
        print("HTTP", r.status_code, r.text[:500])


if __name__ == "__main__":
    main()
