"""Phase B: FINE rectangle search v3 with edge-tracing refiner.

Same anchors as rect_search_v2 (coarse sweep + ms_maps peaks + prob seeds +
Kadane for ps/tm), but the coordinate-descent refiner is replaced by
EDGE-TRACING: from the best decodable anchor box, slide each edge inward
(step 8 then 4 px) while bit-accuracy stays within eps of the best.
Result = the minimal box that still decodes = the tight rect.

Why: v2's objective (max bit-acc) rewards boxes that CONTAIN the true rect
with margin (more redundant pixels decode as well or better), so descent
drifts to oversized boxes. Tracing walks the accuracy plateau to its inner
edge instead.

    python work_bassim/rect_search_v3.py --method pixelseal [--smoke]
Saves work_bassim/bbox_v7_{method}.npz (same keys as bbox_v6 for postprocess
compat) and prints box mIoU on all partials (v2 metric) AND on the true-rect
subset (fill > 0.95, area <= 0.55).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_maps import _logits_batch, _score_boxes, _valid_box  # noqa: E402
from rect_search import _box_iou, _rect_grid, kadane_2d_boxes, local_decoder_grid, log10_binom_p  # noqa: E402
from trace_core import miou2, rect_subset_idx, trace_edges  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

from rect_search_v2 import _clip_box, _enum_windows, _expand, _small_shapes  # noqa: E402


def search_rects_v3(wm, method, imgs, msgs, probs, ms, device, max_windows,
                    max_pos=8, eps=0.02, smoke=False):
    max_ar = 2.0 if method == "trustmark" else 16.0
    min_side = 24 if method == "trustmark" else 16
    target_res = getattr(wm, "resolution", 256)
    nbits = wm.nbits
    N = len(imgs)
    shapes = _small_shapes(max_ar)
    coarse = [
        (64, 64, _rect_grid(64, 64, 32)),
        (96, 96, _rect_grid(96, 96, 48)),
    ]
    if method != "stegastamp":
        coarse.append((48, 48, _rect_grid(48, 48, 28)))
    if smoke:
        coarse = coarse[1:2]

    boxes_128 = np.zeros((N, 4), np.int32)
    best_accs = np.zeros(N, np.float32)
    best_logp = np.zeros(N, np.float32)
    full_accs = np.zeros(N, np.float32)
    grow_accs = np.zeros(N, np.float32)
    shrink_accs = np.zeros(N, np.float32)

    t0 = time.time()
    for i0 in range(0, N, 32):
        i1 = min(N, i0 + 32)
        bt = torch.from_numpy(imgs[i0:i1]).permute(0, 3, 1, 2).float().div(255)
        lg = _logits_batch(wm, method, bt, device, bs=32).numpy()
        full_accs[i0:i1] = ((lg > 0).astype(np.uint8) == msgs[i0:i1]).mean(axis=1)

    todo = [i for i in range(N) if full_accs[i] < 0.72]
    if smoke:
        todo = todo[:2]
    print(f"[{method}] {len(todo)}/{N} images to search", flush=True)

    for n_done, i in enumerate(todo):
        img_t = torch.from_numpy(imgs[i]).permute(2, 0, 1)[None].float().div(255)
        msg = msgs[i]

        coarse_boxes = []
        for wh, ww, rects in coarse:
            if wh < min_side or ww < min_side:
                continue
            sc = _score_boxes(wm, method, img_t, msg, rects, device, target_res)
            for (acc, dot), b in zip(sc, rects):
                coarse_boxes.append((acc + 1e-3 * dot, b))
        coarse_boxes.sort(reverse=True, key=lambda t: t[0])

        anchors = []
        for _, b in coarse_boxes[:2]:
            anchors.append(_expand(b, 32))
        for ch, win in ((2, 96), (1, 160)):
            m = ms[i, ch]
            py, px = np.unravel_index(np.argmax(m), m.shape)
            cy, cx = int(py) * 4, int(px) * 4
            y0 = max(0, min(512 - win, cy - win // 2))
            x0 = max(0, min(512 - win, cx - win // 2))
            anchors.append((y0, y0 + win, x0, x0 + win))
        for pthr in (0.10, 0.25):
            ys, xs = np.where(probs[i] > pthr)
            if len(ys):
                b = (int(ys.min()) * 4, (int(ys.max()) + 1) * 4, int(xs.min()) * 4, (int(xs.max()) + 1) * 4)
                anchors.append(_expand(b, 16))
        if method != "stegastamp":
            k_map = local_decoder_grid(wm, method, imgs[i], msg, device)
            if k_map is not None:
                for b in kadane_2d_boxes(k_map, 512 // k_map.shape[0], max_ar, min_side):
                    anchors.append(_expand(b, 16))

        dedup = []
        for a in anchors:
            if all(_box_iou(a, d) < 0.7 for d in dedup):
                dedup.append(a)
            if len(dedup) >= 4:
                break
        anchors = dedup

        cand_boxes = [b for _, b in coarse_boxes[:6]]
        for a in anchors:
            cand_boxes.extend(_enum_windows(a, shapes, max_pos=max_pos))
        cand_boxes = list(dict.fromkeys(cand_boxes))
        cand_boxes = [b for b in cand_boxes if _valid_box(*b, max_ar, min_side)][:max_windows]
        if not cand_boxes:
            continue

        all_scores = []
        for k0 in range(0, len(cand_boxes), 256):
            sc = _score_boxes(wm, method, img_t, msg, cand_boxes[k0:k0 + 256], device, target_res)
            all_scores.extend(sc)

        ranked = sorted(range(len(cand_boxes)), key=lambda idx: -(all_scores[idx][0] + 1e-3 * all_scores[idx][1]))
        seeds = []
        for idx in ranked:
            b = cand_boxes[idx]
            if all(_box_iou(b, sb) < 0.5 for sb, _ in seeds):
                seeds.append((b, all_scores[idx]))
            if len(seeds) >= 3:
                break
        if not seeds:
            continue

        def score_fn(box):
            y0, y1, x0, x1 = (int(v) for v in box)
            r = _score_boxes(wm, method, img_t, msg, [(y0, y1, x0, x1)], device, target_res)
            return r[0][0] + 1e-3 * r[0][1]

        best_box, best_sc = seeds[0][0], seeds[0][1][0] + 1e-3 * seeds[0][1][1]
        for seed_box, _ in seeds:
            traced, tsc = trace_edges(seed_box, score_fn, steps=(8, 4), eps=eps,
                                      min_side=min_side, bounds=(0, 512, 0, 512))
            if tsc > best_sc:
                best_box, best_sc = traced, tsc

        y0, y1, x0, x1 = best_box
        boxes_128[i] = [round(y0 / 4), round(y1 / 4), round(x0 / 4), round(x1 / 4)]
        raw = _score_boxes(wm, method, img_t, msg, [best_box], device, target_res)[0]
        best_accs[i] = raw[0]
        best_logp[i] = log10_binom_p(float(raw[0]), nbits)

        g = _clip_box((y0 - 8, y1 + 8, x0 - 8, x1 + 8))
        k = _clip_box((y0 + 8, y1 - 8, x0 + 8, x1 - 8))
        grow_accs[i] = _score_boxes(wm, method, img_t, msg, [g], device, target_res)[0][0]
        if k[1] - k[0] >= min_side and k[3] - k[2] >= min_side:
            shrink_accs[i] = _score_boxes(wm, method, img_t, msg, [k], device, target_res)[0][0]
        else:
            shrink_accs[i] = raw[0]

        if (n_done + 1) % 20 == 0:
            print(f"  [{method}] {n_done + 1}/{len(todo)} searched ({time.time() - t0:.0f}s)", flush=True)

    for i in range(N):
        if full_accs[i] >= 0.72:
            boxes_128[i] = [0, 128, 0, 128]
            best_accs[i] = full_accs[i]
            best_logp[i] = log10_binom_p(float(full_accs[i]), nbits)
    return boxes_128, best_accs, best_logp, full_accs, grow_accs, shrink_accs


def box_masks(boxes, n):
    out = np.zeros((n, 128, 128), np.uint8)
    for i, (y0, y1, x0, x1) in enumerate(boxes.astype(int)):
        out[i, max(0, y0):min(128, y1), max(0, x0):min(128, x1)] = 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_windows", type=int, default=1100)
    ap.add_argument("--max_pos", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    from bbox_eval import load_ensemble_probs
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
    zb = np.load(f"work_bassim/bbox_v4_{a.method}.npz")
    ms_r, ms_te = zb["tr_ms"].astype(np.float32), zb["te_ms"].astype(np.float32)

    if a.smoke:
        Xr, Msgr, oof_p, ms_r = Xr[:8], Msgr[:8], oof_p[:8], ms_r[:8]
        Mr = Mr[:8]

    tr_out = search_rects_v3(wm, a.method, Xr, Msgr, oof_p, ms_r, a.device,
                             a.max_windows, max_pos=a.max_pos, eps=a.eps, smoke=a.smoke)
    tr_boxes, tr_bacc, tr_lp, tr_fa, tr_grow, tr_shrink = tr_out
    if not a.smoke:
        area = Mr.mean((1, 2))
        part = np.where(area <= 0.55)[0]
        sc_all = [miou2(m, g) for m, g in zip(box_masks(tr_boxes[part], len(part)), Mr[part])]
        ridx = rect_subset_idx(Mr)
        sc_rect = [miou2(m, g) for m, g in zip(box_masks(tr_boxes[ridx], len(ridx)), Mr[ridx])]
        print(f"[{a.method}] v3 OOF box mIoU on partial imgs: {np.mean(sc_all):.4f} "
              f"| on TRUE-RECT subset (n={len(ridx)}): {np.mean(sc_rect):.4f}", flush=True)
    if a.smoke:
        print("[smoke] train-side OK, boxes:", tr_boxes.tolist(), flush=True)
        return

    te_boxes, te_bacc, te_lp, te_fa, te_grow, te_shrink = search_rects_v3(
        wm, a.method, Xte, Msgte, te_p, ms_te, a.device, a.max_windows,
        max_pos=a.max_pos, eps=a.eps)

    np.savez_compressed(
        f"work_bassim/bbox_v7_{a.method}.npz",
        tr_boxes=tr_boxes, tr_bacc=tr_bacc, tr_logp=tr_lp, tr_facc=tr_fa,
        tr_grow=tr_grow, tr_shrink=tr_shrink, masks=Mr,
        te_boxes=te_boxes, te_bacc=te_bacc, te_logp=te_lp, te_facc=te_fa,
        te_grow=te_grow, te_shrink=te_shrink, names=np.array(names_te),
    )
    print(f"[{a.method}] saved bbox_v7 in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
