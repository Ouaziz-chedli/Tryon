"""Phase 6: hierarchical FINE rectangle search for tiny partial watermarks.

bbox_v5 (rect_search.py) never proposes boxes < 48-64 px, but true rects have median
side ~20 px. Feeding the TRUE box to the decoder yields 0.72-0.99 bit-acc (gt_box_acc
on stegastamp rects), so the decoder is fine -- the proposal grid was the blocker.

Per image (only full_acc < 0.72):
  Stage 1: light coarse sweep (48/64/96 px grids, top-2) -> anchor regions (+32 px)
  Stage 2: fine windows (16..64 px, <=8x8 positions per shape) inside anchors
           (anchors: coarse top-2, ms_maps 96/160 peaks, prob seeds, Kadane for ps/tm)
  Stage 3: coordinate descent (16/8/4/2) from top-3 IoU-diverse seeds
  Records best box/acc/logp + grow(+8)/shrink(-8) accs (gate features for postprocess v5).

    python work_bassim/rect_search_v2.py --method pixelseal [--smoke]
Saves work_bassim/bbox_v6_{method}.npz
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from ms_maps import _logits_batch, _score_boxes, _valid_box  # noqa: E402
from rect_search import _box_iou, _rect_grid, kadane_2d_boxes, local_decoder_grid, log10_binom_p  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def _small_shapes(max_ar: float) -> list[tuple[int, int]]:
    shapes = ((16, 16), (16, 32), (24, 24), (24, 48), (32, 32), (32, 64), (48, 48), (48, 96), (64, 64))
    return [(h, w) for h, w in shapes if max(h / w, w / h) <= max_ar]


def _enum_windows(anchor, shapes, max_pos=8):
    ay0, ay1, ax0, ax1 = anchor
    H, W = ay1 - ay0, ax1 - ax0
    out = []
    for h, w in shapes:
        if H < h or W < w:
            continue
        s = max(4, min(h, w) // 4)
        ry = max(1, min(max_pos, (H - h) // s + 1))
        rx = max(1, min(max_pos, (W - w) // s + 1))
        dy = (H - h) // (ry - 1) if ry > 1 else 0
        dx = (W - w) // (rx - 1) if rx > 1 else 0
        for iy in range(ry):
            for ix in range(rx):
                out.append((ay0 + iy * dy, ay0 + iy * dy + h, ax0 + ix * dx, ax0 + ix * dx + w))
    return out


def _clip_box(b):
    y0, y1, x0, x1 = b
    return (max(0, y0), min(512, y1), max(0, x0), min(512, x1))


def _expand(b, m):
    y0, y1, x0, x1 = b
    return _clip_box((y0 - m, y1 + m, x0 - m, x1 + m))


def search_rects_v2(wm, method, imgs, msgs, probs, ms, device, max_windows, max_pos=8, smoke=False):
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

        best_box, best_sc = seeds[0]
        for seed_box, seed_sc in seeds:
            cur_box, cur_sc = seed_box, seed_sc
            side = max(cur_box[1] - cur_box[0], cur_box[3] - cur_box[2])
            for step in (s for s in (16, 8, 4, 2) if s <= max(2, side // 2)):
                y0, y1, x0, x1 = cur_box
                neighbors = []
                for dy0 in (-step, 0, step):
                    for dy1 in (-step, 0, step):
                        if dy0 or dy1:
                            b = (y0 + dy0, y1 + dy1, x0, x1)
                            if _valid_box(*b, max_ar, min_side):
                                neighbors.append(b)
                for dx0 in (-step, 0, step):
                    for dx1 in (-step, 0, step):
                        if dx0 or dx1:
                            b = (y0, y1, x0 + dx0, x1 + dx1)
                            if _valid_box(*b, max_ar, min_side):
                                neighbors.append(b)
                for d in (-step, step):
                    for b in ((y0 + d, y1 - d, x0 + d, x1 - d), (y0 + d, y1 + d, x0, x1), (y0, y1, x0 + d, x1 + d)):
                        if _valid_box(*b, max_ar, min_side):
                            neighbors.append(b)
                neighbors = list(dict.fromkeys(neighbors))
                if not neighbors:
                    continue
                n_sc = _score_boxes(wm, method, img_t, msg, neighbors, device, target_res)
                ni = int(np.argmax([s[0] + 1e-3 * s[1] for s in n_sc]))
                if (n_sc[ni][0] + 1e-3 * n_sc[ni][1]) > (cur_sc[0] + 1e-3 * cur_sc[1]):
                    cur_box, cur_sc = neighbors[ni], n_sc[ni]
            if (cur_sc[0] + 1e-3 * cur_sc[1]) > (best_sc[0] + 1e-3 * best_sc[1]):
                best_box, best_sc = cur_box, cur_sc

        y0, y1, x0, x1 = best_box
        boxes_128[i] = [round(y0 / 4), round(y1 / 4), round(x0 / 4), round(x1 / 4)]
        best_accs[i] = best_sc[0]
        best_logp[i] = log10_binom_p(float(best_sc[0]), nbits)

        g = _clip_box((y0 - 8, y1 + 8, x0 - 8, x1 + 8))
        k = _clip_box((y0 + 8, y1 - 8, x0 + 8, x1 - 8))
        grow_accs[i] = _score_boxes(wm, method, img_t, msg, [g], device, target_res)[0][0]
        if k[1] - k[0] >= min_side and k[3] - k[2] >= min_side:
            shrink_accs[i] = _score_boxes(wm, method, img_t, msg, [k], device, target_res)[0][0]
        else:
            shrink_accs[i] = best_sc[0]

        if (n_done + 1) % 20 == 0:
            print(f"  [{method}] {n_done + 1}/{len(todo)} searched ({time.time() - t0:.0f}s)", flush=True)

    for i in range(N):
        if full_accs[i] >= 0.72:
            boxes_128[i] = [0, 128, 0, 128]
            best_accs[i] = full_accs[i]
            best_logp[i] = log10_binom_p(float(full_accs[i]), nbits)
    return boxes_128, best_accs, best_logp, full_accs, grow_accs, shrink_accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_windows", type=int, default=1100)
    ap.add_argument("--max_pos", type=int, default=8)
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

    tr_out = search_rects_v2(wm, a.method, Xr, Msgr, oof_p, ms_r, a.device, a.max_windows, max_pos=a.max_pos, smoke=a.smoke)
    tr_boxes, tr_bacc, tr_lp, tr_fa, tr_grow, tr_shrink = tr_out
    if not a.smoke:
        from wmloc.utils import miou_image
        area = Mr.mean((1, 2))
        sc = []
        for bx, a_, g in zip(tr_boxes, area, Mr):
            if a_ <= 0.55:
                nm = np.zeros((128, 128), np.uint8)
                nm[bx[0]:bx[1], bx[2]:bx[3]] = 1
                sc.append(miou_image(nm, g))
        print(f"[{a.method}] OOF box mIoU on partial imgs: {np.mean(sc):.4f}", flush=True)
    if a.smoke:
        print("[smoke] train-side OK, boxes:", tr_boxes.tolist(), flush=True)
        return

    te_boxes, te_bacc, te_lp, te_fa, te_grow, te_shrink = search_rects_v2(
        wm, a.method, Xte, Msgte, te_p, ms_te, a.device, a.max_windows, max_pos=a.max_pos)

    np.savez_compressed(
        f"work_bassim/bbox_v6_{a.method}.npz",
        tr_boxes=tr_boxes, tr_bacc=tr_bacc, tr_logp=tr_lp, tr_facc=tr_fa,
        tr_grow=tr_grow, tr_shrink=tr_shrink, masks=Mr,
        te_boxes=te_boxes, te_bacc=te_bacc, te_logp=te_lp, te_facc=te_fa,
        te_grow=te_grow, te_shrink=te_shrink, names=np.array(names_te),
    )
    print(f"[{a.method}] saved bbox_v6 in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
