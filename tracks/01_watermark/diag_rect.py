"""Phase A diagnostic: box mIoU by regime (RECT vs IRREG) from saved bbox_v6.

Splits train images into true-rect (area <= 0.55, fill > 0.95) vs irregular
partials and reports the decoder-verified box mIoU on each. Quantifies how
much of the ~0.47 "box mIoU on partial imgs" is regime-mix vs sloppy boxes.

Pure numpy (trace_core) — runs in seconds on CPU, no GPU needed.

    python work_bassim/diag_rect.py [--method pixelseal]
Prints a table per method; exit 0 always (diagnostic, never fails the job).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_core import box_iou, compute_fills, miou2, rect_subset_idx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    choices=["all", "pixelseal", "trustmark", "stegastamp"])
    a = ap.parse_args()
    methods = ["pixelseal", "trustmark", "stegastamp"] if a.method == "all" else [a.method]
    for m in methods:
        p = f"work_bassim/bbox_v6_{m}.npz"
        if not os.path.exists(p):
            print(f"[{m}] no {p}, skipping", flush=True)
            continue
        z = np.load(p)
        masks = z["masks"].astype(np.uint8)
        boxes = z["tr_boxes"].astype(int)
        logp = z["tr_logp"].astype(np.float32)
        area = masks.reshape(len(masks), -1).mean(axis=1)
        part = area <= 0.55
        rect_idx = rect_subset_idx(masks)
        irreg_idx = np.array([i for i in np.where(part)[0] if i not in set(rect_idx.tolist())])

        def subset_miou(idxs):
            sc = []
            for i in idxs:
                nm = np.zeros((128, 128), np.uint8)
                y0, y1, x0, x1 = boxes[i]
                nm[max(0, y0):min(128, y1), max(0, x0):min(128, x1)] = 1
                sc.append(miou2(nm, masks[i]))
            return float(np.mean(sc)) if sc else float("nan")

        print(f"[{m}] partial={int(part.sum())} rect={len(rect_idx)} irreg={len(irreg_idx)}", flush=True)
        print(f"[{m}] box_mIoU rect_subset={subset_miou(rect_idx):.4f} "
              f"irreg_subset={subset_miou(irreg_idx):.4f} "
              f"all_partial={subset_miou(np.where(part)[0]):.4f}", flush=True)
        if len(rect_idx):
            print(f"[{m}] median tr_logp rect={np.median(logp[rect_idx]):.2f} "
                  f"irreg={np.median(logp[irreg_idx]):.2f}" if len(irreg_idx) else
                  f"[{m}] median tr_logp rect={np.median(logp[rect_idx]):.2f}", flush=True)
        fills = compute_fills(masks)
        print(f"[{m}] fill(rect) median={np.median(fills[rect_idx]):.3f} "
              f"area(rect) median={np.median(area[rect_idx]):.4f}", flush=True)


if __name__ == "__main__":
    main()
