"""Fast standalone Step 1 script: compute multi-scale crop heatmaps + GPU coordinate-descent
bbox refinement on top of existing v2+v3+v4 ensemble probabilities, and measure real OOF gain.

    python work_bassim/bbox_eval.py --method pixelseal
Saves work_bassim/bbox_v4_<method>.npz (train+test boxes, accs, ms_maps).
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from ms_maps import compute_ms_maps, refine_bboxes  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

WEIGHTS = {
    "pixelseal": {"v2": 0.4, "v4": 0.6},
    "trustmark": {"v2": 0.4, "v4": 0.6},
    "stegastamp": {"v2": 0.4, "v3": 0.4, "v4": 0.2},
}


def load_ensemble_probs(method: str):
    oof_p, te_p = None, None
    for tag, w in WEIGHTS[method].items():
        po = np.load(f"work_bassim/oof_{tag}_{method}.npz")["probs"].astype(np.float32)
        pt = np.load(f"work_bassim/test_{tag}_{method}.npz")["probs"].astype(np.float32)
        oof_p = w * po if oof_p is None else oof_p + w * po
        te_p = w * pt if te_p is None else te_p + w * pt
    return oof_p, te_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    t0 = time.time()
    wm = load_watermark(a.method, models_dir="models", device=a.device)
    ds_tr = LocalisationDataset("release", "train", methods=[a.method])
    ds_te = LocalisationDataset("release", "test", methods=[a.method])

    Xr = np.stack([ds_tr[i]["image"] for i in range(len(ds_tr))])
    Mr = np.stack([ds_tr[i]["mask128"] for i in range(len(ds_tr))])
    Msgr = np.stack([ds_tr[i]["message"] for i in range(len(ds_tr))])
    Xte = np.stack([ds_te[i]["image"] for i in range(len(ds_te))])
    Msgte = np.stack([ds_te[i]["message"] for i in range(len(ds_te))])
    names_te = [ds_te[i]["image_name"] for i in range(len(ds_te))]

    oof_p, te_p = load_ensemble_probs(a.method)
    ms_r, top_r = compute_ms_maps(wm, a.method, Xr, Msgr, device=a.device, return_top_crops=True)
    ms_te, top_te = compute_ms_maps(wm, a.method, Xte, Msgte, device=a.device, return_top_crops=True)
    print(f"[{a.method}] ms_maps computed in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    tr_boxes, tr_bacc, tr_facc = refine_bboxes(wm, a.method, Xr, Msgr, ms_r, oof_p, device=a.device, top_crops=top_r)
    te_boxes, te_bacc, te_facc = refine_bboxes(wm, a.method, Xte, Msgte, ms_te, te_p, device=a.device, top_crops=top_te)
    print(f"[{a.method}] refine_bboxes done in {time.time() - t1:.1f}s (total {time.time() - t0:.1f}s)", flush=True)

    np.savez_compressed(
        f"work_bassim/bbox_v4_{a.method}.npz",
        tr_boxes=tr_boxes, tr_bacc=tr_bacc, tr_facc=tr_facc, tr_ms=ms_r.astype(np.float16), masks=Mr,
        te_boxes=te_boxes, te_bacc=te_bacc, te_facc=te_facc, te_ms=ms_te.astype(np.float16), names=np.array(names_te),
    )


if __name__ == "__main__":
    main()
