"""Measure decoder bit-accuracy on candidate boxes for partial-watermark train images,
split by true regime (exact rectangle vs irregular object), to design a rect-override gate.

    PYTHONPATH=. .venv/bin/python work_bassim/gt_box_acc.py
"""
import numpy as np
import torch

from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark

MARGIN = 8  # px at 512 scale


def box_acc(wm, img, msg, y0, y1, x0, x1):
    crop = img[y0:y1, x0:x1]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return float("nan")
    t = torch.from_numpy(crop).permute(2, 0, 1)[None].float().div(255)
    pred = (wm.logits(t)[0].numpy() > 0).astype(np.uint8)
    return float((pred == msg).mean())


def main():
    import sys
    methods = [sys.argv[1]] if len(sys.argv) > 1 else ["pixelseal", "trustmark", "stegastamp"]
    for m in methods:
        z4 = np.load(f"work_bassim/bbox_v4_{m}.npz")
        z5 = np.load(f"work_bassim/bbox_v5_{m}.npz")
        masks = z4["masks"]
        area = masks.mean((1, 2))
        fills = np.array([
            (g if a <= 0.55 else 1 - g).sum() / ((np.where((g if a <= 0.55 else 1 - g) > 0)[0].ptp() + 1)
                                                 * (np.where((g if a <= 0.55 else 1 - g) > 0)[1].ptp() + 1))
            for g, a in zip(masks, area)
        ])
        part = area <= 0.55
        is_rect = part & (fills > 0.95)
        is_irreg = part & (fills <= 0.95)

        ds = LocalisationDataset("release", "train", methods=[m])
        wm = load_watermark(m, models_dir="models", device="cpu")
        p5 = np.load(f"work_bassim/oof_v5_{m}.npz")["probs"].astype(np.float32)

        rows = []
        for i in np.where(part)[0]:
            item = ds[int(i)]
            img, msg = item["image"], item["message"]
            ys, xs = np.where(masks[i] > 0)
            gy0, gy1, gx0, gx1 = ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4

            b128 = z5["tr_boxes"][i]
            by0, by1, bx0, bx1 = int(b128[0]) * 4, int(b128[1]) * 4, int(b128[2]) * 4, int(b128[3]) * 4

            def clip(y0, y1, x0, x1):
                return max(0, y0), min(512, y1), max(0, x0), min(512, x1)

            g_box = clip(by0 - MARGIN, by1 + MARGIN, bx0 - MARGIN, bx1 + MARGIN)
            k_box = clip(by0 + MARGIN, by1 - MARGIN, bx0 + MARGIN, bx1 - MARGIN)

            a_gt = box_acc(wm, img, msg, gy0, gy1, gx0, gx1)
            a_s = box_acc(wm, img, msg, by0, by1, bx0, bx1)
            a_g = box_acc(wm, img, msg, *g_box)
            a_k = box_acc(wm, img, msg, *k_box) if (k_box[1] - k_box[0] >= 8 and k_box[3] - k_box[2] >= 8) else float("nan")

            sb = np.zeros((128, 128), bool)
            sb[b128[0]:b128[1], b128[2]:b128[3]] = True
            p_in = float(p5[i][sb].mean()) if sb.any() else float("nan")
            pin_thr = float((p5[i][sb] > 0.3).mean()) if sb.any() else float("nan")
            box_a = (by1 - by0) * (bx1 - bx0) / 512.0 / 512.0
            rows.append((int(i), float(is_rect[i]), area[i], a_gt, a_s, a_g, a_k, p_in, pin_thr, float(z5["tr_logp"][i]), box_a))
            print(f"  [{m}] img {i} rect={int(is_rect[i])} a_gt={a_gt:.3f} a_box={a_s:.3f} a_grow={a_g:.3f} a_shrink={a_k:.3f} lp={z5['tr_logp'][i]:.1f}", flush=True)

        arr = np.array(rows, dtype=float)
        print(f"===== {m} =====", flush=True)
        for name, sel in (("RECT", arr[:, 1] == 1), ("IRREG", arr[:, 1] == 0)):
            sub = arr[sel]
            if not len(sub):
                continue
            q = lambda c: np.nanquantile(sub[:, c], [0.1, 0.5, 0.9]).round(3)
            dg = np.nanquantile(sub[:, 4] - sub[:, 5], [0.1, 0.5, 0.9]).round(3)
            print(f"[{m}] {name} n={len(sub)}")
            print(f"   a_gt={q(3)} a_box={q(4)} a_grow={q(5)} d_grow={dg}")
            print(f"   p_in={q(7)} p_in>.3={q(8)} lp={q(9)} box_a={q(10)}", flush=True)


if __name__ == "__main__":
    main()
