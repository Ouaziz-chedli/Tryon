"""Exhaustive Phase-2 rectangle finder (2D Kadane + dense 48/64/96 grid + top-3 multi-seed
coordinate descent + exact binomial -log10(p-value) scoring).

    python work_bassim/rect_search.py --method trustmark
Saves work_bassim/bbox_v5_<method>.npz (train+test boxes, accs, log10p, full_accs).
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binom

sys.path.insert(0, ".")
from ms_maps import _logits_batch, _score_boxes, _valid_box  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402


def _rect_grid(win_h: int, win_w: int, stride: int) -> list[tuple[int, int, int, int]]:
    ys = list(range(0, 512 - win_h + 1, stride))
    if ys[-1] != 512 - win_h:
        ys.append(512 - win_h)
    xs = list(range(0, 512 - win_w + 1, stride))
    if xs[-1] != 512 - win_w:
        xs.append(512 - win_w)
    return [(y, y + win_h, x, x + win_w) for y in ys for x in xs]


def kadane_2d_boxes(score_map: np.ndarray, scale_px: int, max_ar: float, min_side: int) -> list[tuple[int, int, int, int]]:
    """Exact 2D max-sum submatrix boxes across bias levels mu."""
    H, W = score_map.shape
    out = []
    med = float(np.median(score_map))
    std = float(score_map.std() + 1e-6)
    for k in (0.2, 0.6, 1.1, 1.6):
        M = score_map - (med + k * std)
        best_val = -1e18
        best_rect = None
        for r0 in range(H):
            col_sum = np.zeros(W, dtype=np.float64)
            for r1 in range(r0 + 1, H + 1):
                col_sum += M[r1 - 1]
                cur = 0.0
                c_start = 0
                for c in range(W):
                    if cur <= 0.0:
                        cur = col_sum[c]
                        c_start = c
                    else:
                        cur += col_sum[c]
                    if cur > best_val:
                        y0, y1 = r0 * scale_px, min(512, r1 * scale_px)
                        x0, x1 = c_start * scale_px, min(512, (c + 1) * scale_px)
                        if _valid_box(y0, y1, x0, x1, max_ar, min_side):
                            best_val = cur
                            best_rect = (y0, y1, x0, x1)
        if best_rect is not None:
            out.append(best_rect)
    return out


def _box_iou(a, b) -> float:
    iy0, iy1 = max(a[0], b[0]), min(a[1], b[1])
    ix0, ix1 = max(a[2], b[2]), min(a[3], b[3])
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    inter = (iy1 - iy0) * (ix1 - ix0)
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / max(ua, 1)


@torch.no_grad()
def local_decoder_grid(wm, method: str, img_uint8: np.ndarray, msg: np.ndarray, device: str) -> np.ndarray:
    x = to_tensor(img_uint8).to(device)
    m_pm1 = torch.from_numpy(msg).float().to(device) * 2.0 - 1.0
    if method == "pixelseal":
        xr = wm.det.preprocess(x)
        emb = wm.det.pixel_decoder.output_upscaling(wm.det.convnext(xr))
        lin = wm.det.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb, lin.weight)[0, 1:] + lin.bias[1:, None, None]
        return (torch.sign(pp) * m_pm1[:, None, None]).mean(0).cpu().numpy()
    if method == "trustmark":
        n = wm.net
        t = n.maxpool(n.relu(n.bn1(n.conv1(x * 2.0 - 1.0))))
        for k in ("layer1", "layer2", "layer3", "layer4"):
            t = getattr(n, k)(t)
        pp = torch.einsum("bchw,oc->bohw", t, n.fc.weight)[0] + n.fc.bias[:, None, None]
        return (torch.sign(pp) * m_pm1[:, None, None]).mean(0).cpu().numpy()
    return None


def log10_binom_p(acc: float, nbits: int) -> float:
    k = int(round(acc * nbits))
    p = float(binom.sf(k - 1, nbits, 0.5))
    return float(-np.log10(max(p, 1e-30)))


def _prob_seed_boxes(pm: np.ndarray, max_ar: float, min_side: int) -> list[tuple[int, int, int, int]]:
    out = []
    for pthr in (0.20, 0.35, 0.50):
        ys, xs = np.where(pm > pthr)
        if len(ys) > 0:
            b = (int(ys.min()) * 4, int(ys.max() + 1) * 4, int(xs.min()) * 4, int(xs.max() + 1) * 4)
            if _valid_box(*b, max_ar, min_side):
                out.append(b)
    return out


@torch.no_grad()
def search_rects(wm, method: str, imgs: np.ndarray, msgs: np.ndarray, probs: np.ndarray,
                 ms_maps: np.ndarray | None, device: str = "cuda", indices=None):
    max_ar = 2.0 if method == "trustmark" else 16.0
    min_side = 24 if method == "trustmark" else 16
    target_res = getattr(wm, "resolution", 256)
    nbits = wm.nbits
    N = len(imgs)

    boxes_128 = np.zeros((N, 4), np.int32)
    best_accs = np.zeros(N, np.float32)
    best_logp = np.zeros(N, np.float32)
    full_accs = np.zeros(N, np.float32)

    # Full-image bit accuracies
    for i0 in range(0, N, 32):
        i1 = min(N, i0 + 32)
        bt = torch.from_numpy(imgs[i0:i1]).permute(0, 3, 1, 2).float().div(255)
        lg = _logits_batch(wm, method, bt, device, bs=32).numpy()
        full_accs[i0:i1] = ((lg > 0).astype(np.uint8) == msgs[i0:i1]).mean(axis=1)

    target_idx = list(range(N)) if indices is None else list(indices)

    # Precompute dense multi-shape grids in batches ONLY for low full_acc images
    low_idx = [i for i in target_idx if full_accs[i] < 0.72]
    if method == "stegastamp":
        dense_shapes = [
            (64, 64, _rect_grid(64, 64, 32)),
            (64, 96, _rect_grid(64, 96, 64)),
            (96, 64, _rect_grid(96, 64, 64)),
            (96, 96, _rect_grid(96, 96, 80)),
        ]
    else:
        dense_shapes = [
            (48, 48, _rect_grid(48, 48, 28)),
            (64, 64, _rect_grid(64, 64, 32)),
            (64, 96, _rect_grid(64, 96, 48)),
            (96, 64, _rect_grid(96, 64, 48)),
            (96, 96, _rect_grid(96, 96, 80)),
        ]

    top_dense = {i: [] for i in low_idx}
    img_bs = 16 if device == "cuda" else 4
    for wh, ww, rects in dense_shapes:
        nc = len(rects)
        for j0 in range(0, len(low_idx), img_bs):
            sub_idx = low_idx[j0:j0 + img_bs]
            b_imgs = torch.from_numpy(imgs[sub_idx]).permute(0, 3, 1, 2).float().div(255)
            crops = torch.cat([
                F.interpolate(b_imgs[:, :, y0:y1, x0:x1], size=(target_res, target_res), mode="bilinear", align_corners=False)
                for (y0, y1, x0, x1) in rects
            ], dim=0)
            lg = _logits_batch(wm, method, crops, device, bs=256).view(nc, len(sub_idx), nbits)
            pred_b = (lg > 0).numpy().astype(np.uint8)
            gt_b = msgs[sub_idx][None, :, :]
            accs = (pred_b == gt_b).mean(axis=2)  # (nc, B)
            for b, img_i in enumerate(sub_idx):
                order = np.argsort(-accs[:, b])[:4]
                for ci in order:
                    y0, y1, x0, x1 = rects[int(ci)]
                    top_dense[img_i].append((float(accs[ci, b]), y0, y1, x0, x1))

    for i in target_idx:
        if full_accs[i] >= 0.72:
            boxes_128[i] = [0, 128, 0, 128]
            best_accs[i] = full_accs[i]
            best_logp[i] = log10_binom_p(float(full_accs[i]), nbits)
            continue

        img_t = torch.from_numpy(imgs[i]).permute(2, 0, 1)[None].float().div(255)
        msg = msgs[i]
        cands = []

        # 1. Dense grid top hits + strip variants around top hits
        dense_sorted = sorted(top_dense[i], reverse=True)
        for _, y0, y1, x0, x1 in dense_sorted[:8]:
            cands.append((y0, y1, x0, x1))
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            for h_s, w_s in ((32, 96), (96, 32), (48, 128), (128, 48), (40, 40)):
                yy0 = max(0, min(512 - h_s, cy - h_s // 2))
                xx0 = max(0, min(512 - w_s, cx - w_s // 2))
                if _valid_box(yy0, yy0 + h_s, xx0, xx0 + w_s, max_ar, min_side):
                    cands.append((yy0, yy0 + h_s, xx0, xx0 + w_s))

        # 2. Exact 2D Kadane boxes on local decoder grid (for pixelseal & trustmark)
        k_map = local_decoder_grid(wm, method, imgs[i], msg, device)
        if k_map is not None:
            cands.extend(kadane_2d_boxes(k_map, 512 // k_map.shape[0], max_ar, min_side))

        # 3. Probability map seeds + ms_maps peaks
        cands.extend(_prob_seed_boxes(probs[i], max_ar, min_side))
        if ms_maps is not None:
            for ch, win in ((2, 96), (1, 160), (0, 256)):
                py, px = np.unravel_index(np.argmax(ms_maps[i, ch]), (128, 128))
                y0 = max(0, min(512 - win, int(py) * 4 - win // 2))
                x0 = max(0, min(512 - win, int(px) * 4 - win // 2))
                cands.append((y0, y0 + win, x0, x0 + win))

        cands = list(dict.fromkeys(cands))
        scores = _score_boxes(wm, method, img_t, msg, cands, device, target_res)
        ranked = sorted(range(len(cands)), key=lambda idx: -(scores[idx][0] + 1e-3 * scores[idx][1]))

        # Pick top-4 (stegastamp) / top-6 spatially diverse seeds (IoU < 0.5)
        max_seeds = 4 if method == "stegastamp" else 6
        seeds = []
        for idx in ranked:
            b = cands[idx]
            if all(_box_iou(b, sb) < 0.5 for sb, _ in seeds):
                seeds.append((b, scores[idx]))
            if len(seeds) >= max_seeds:
                break

        best_overall_box, best_overall_sc = seeds[0]
        for seed_box, seed_sc in seeds:
            cur_box, cur_sc = seed_box, seed_sc
            for step in (32, 16, 8, 4, 2):
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
            if (cur_sc[0] + 1e-3 * cur_sc[1]) > (best_overall_sc[0] + 1e-3 * best_overall_sc[1]):
                best_overall_box, best_overall_sc = cur_box, cur_sc

        y0, y1, x0, x1 = best_overall_box
        boxes_128[i] = [round(y0 / 4), round(y1 / 4), round(x0 / 4), round(x1 / 4)]
        best_accs[i] = best_overall_sc[0]
        best_logp[i] = log10_binom_p(float(best_overall_sc[0]), nbits)

    return boxes_128, best_accs, best_logp, full_accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--device", default="cuda")
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
    tr_ms, te_ms = zb["tr_ms"].astype(np.float32), zb["te_ms"].astype(np.float32)

    tr_b, tr_ba, tr_lp, tr_fa = search_rects(wm, a.method, Xr, Msgr, oof_p, tr_ms, device=a.device)
    te_b, te_ba, te_lp, te_fa = search_rects(wm, a.method, Xte, Msgte, te_p, te_ms, device=a.device)
    print(f"[{a.method}] rect_search finished in {time.time() - t0:.1f}s", flush=True)

    np.savez_compressed(
        f"work_bassim/bbox_v5_{a.method}.npz",
        tr_boxes=tr_b, tr_bacc=tr_ba, tr_logp=tr_lp, tr_facc=tr_fa, masks=Mr,
        te_boxes=te_b, te_bacc=te_ba, te_logp=te_lp, te_facc=te_fa, names=np.array(names_te),
    )


if __name__ == "__main__":
    main()
