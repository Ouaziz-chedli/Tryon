"""Multi-scale sliding-crop decoder heatmaps + batched GPU coordinate-descent bbox refinement.

1. compute_ms_maps(wm, method, imgs_uint8, msgs, device):
   Runs batched crops at 3 scales (256, 160, 96 px) through wm.logits(), returning 4 channels at 128x128:
     [acc_256, acc_160, acc_96, max_acc] centered at 0.5 and scaled by 4.0.
2. refine_bboxes(wm, method, imgs_uint8, msgs, ms_maps, prob_maps, device):
   For each image, searches for the rectangular crop img[y0:y1, x0:x1] maximizing bit accuracy with
   the known message (respecting dataset aspect ratio constraints: <=2:1 for trustmark, <=16:1 others).
   Returns (boxes_128 [N,4], best_bit_accs [N], full_bit_accs [N]).
"""
import numpy as np
import torch
import torch.nn.functional as F

from wmloc.utils import bit_accuracy


def _crop_grid(win: int, stride: int) -> list[tuple[int, int]]:
    ys = list(range(0, 512 - win + 1, stride))
    if ys[-1] != 512 - win:
        ys.append(512 - win)
    xs = ys
    return [(y, x) for y in ys for x in xs]


GRID_SCALES = [
    (256, 128, _crop_grid(256, 128)),  # 9 crops
    (160, 88, _crop_grid(160, 88)),    # 25 crops
    (96, 80, _crop_grid(96, 80)),      # 36 crops -> 70 crops/image
]


@torch.no_grad()
def _logits_batch(wm, method: str, crops_t: torch.Tensor, device: str, bs: int = 256) -> torch.Tensor:
    """crops_t: (K, 3, H, W) float [0, 1]. Returns (K, nbits) logits on CPU."""
    out = []
    use_amp = (device == "cuda") and (method != "stegastamp")
    for k in range(0, len(crops_t), bs):
        sub = crops_t[k:k + bs]
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = wm.logits(sub.to(device))
        else:
            lg = wm.logits(sub)
        out.append(lg.float().cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def compute_ms_maps(wm, method: str, imgs_uint8: np.ndarray, msgs: np.ndarray, device: str = "cuda",
                    smoke: bool = False, return_top_crops: bool = False):
    """Returns float32 (N, 4, 128, 128) centered multi-scale bit-accuracy maps (and optional top crop boxes)."""
    scales = [(256, 256, [(0, 0), (256, 256)])] if smoke else GRID_SCALES
    N = len(imgs_uint8)
    maps = np.zeros((N, 4, 128, 128), np.float32)
    top_crops = [[] for _ in range(N)]

    img_batch = 16 if device == "cuda" else 4
    for s_idx, (win, _, coords) in enumerate(scales):
        w128 = win // 4
        nc = len(coords)
        for i0 in range(0, N, img_batch):
            i1 = min(N, i0 + img_batch)
            b_imgs = torch.from_numpy(imgs_uint8[i0:i1]).permute(0, 3, 1, 2).float().div(255)
            crops = torch.cat([b_imgs[:, :, y:y + win, x:x + win] for (y, x) in coords], dim=0)
            lg = _logits_batch(wm, method, crops, device, bs=256).view(nc, i1 - i0, -1)
            bits_pred = (lg > 0).numpy().astype(np.uint8)
            bits_gt = msgs[i0:i1][None, :, :]
            accs = (bits_pred == bits_gt).mean(axis=2)  # (nc, B)

            acc_map = np.zeros((i1 - i0, 128, 128), np.float32)
            cnt_map = np.zeros((128, 128), np.float32)
            for c_idx, (y, x) in enumerate(coords):
                y0, x0 = y // 4, x // 4
                acc_map[:, y0:y0 + w128, x0:x0 + w128] += accs[c_idx][:, None, None]
                cnt_map[y0:y0 + w128, x0:x0 + w128] += 1.0
            maps[i0:i1, s_idx] = (acc_map / np.maximum(cnt_map[None], 1.0) - 0.5) * 4.0

            if return_top_crops:
                for b in range(i1 - i0):
                    best_c = np.argsort(-accs[:, b])[:2]
                    for ci in best_c:
                        y, x = coords[int(ci)]
                        top_crops[i0 + b].append((float(accs[ci, b]), y, y + win, x, x + win))

    if smoke:
        maps[:, 1] = maps[:, 0]
        maps[:, 2] = maps[:, 0]
    maps[:, 3] = maps[:, :3].max(axis=1)
    if return_top_crops:
        return maps, [sorted(tc, reverse=True)[:5] for tc in top_crops]
    return maps


@torch.no_grad()
def _score_boxes(wm, method: str, img_t: torch.Tensor, msg_np: np.ndarray, boxes: list[tuple[int, int, int, int]],
                 device: str, target_res: int) -> list[tuple[float, float]]:
    """Evaluate candidate boxes (y0, y1, x0, x1) at 512px scale. Returns [(bit_acc, dot), ...]."""
    if not boxes:
        return []
    crops = torch.cat([
        F.interpolate(img_t[:, :, y0:y1, x0:x1], size=(target_res, target_res), mode="bilinear", align_corners=False)
        for (y0, y1, x0, x1) in boxes
    ], dim=0)
    lg = _logits_batch(wm, method, crops, device, bs=128).numpy()
    msg_pm1 = msg_np.astype(np.float32) * 2.0 - 1.0
    out = []
    for row in lg:
        acc = float(((row > 0).astype(np.uint8) == msg_np).mean())
        dot = float((row * msg_pm1).mean())
        out.append((acc, dot))
    return out


def _valid_box(y0: int, y1: int, x0: int, x1: int, max_ar: float, min_side: int = 16) -> bool:
    h, w = y1 - y0, x1 - x0
    if h < min_side or w < min_side or y0 < 0 or x0 < 0 or y1 > 512 or x1 > 512:
        return False
    return max(h / w, w / h) <= max_ar


@torch.no_grad()
def refine_bboxes(wm, method: str, imgs_uint8: np.ndarray, msgs: np.ndarray,
                  ms_maps: np.ndarray, prob_maps: np.ndarray, device: str = "cuda",
                  smoke: bool = False, top_crops=None):
    """Coordinate-descent search for watermarked rectangle per image.
    Returns (boxes_128 [N,4] as y0,y1,x0,x1 in 0..128, best_accs [N], full_accs [N]).
    """
    max_ar = 2.0 if method == "trustmark" else 16.0
    min_side = 24 if method == "trustmark" else 16
    target_res = getattr(wm, "resolution", 256)
    N = len(imgs_uint8)
    boxes_128 = np.zeros((N, 4), np.int32)
    best_accs = np.zeros(N, np.float32)
    full_accs = np.zeros(N, np.float32)

    for i0 in range(0, N, 32):
        i1 = min(N, i0 + 32)
        bt = torch.from_numpy(imgs_uint8[i0:i1]).permute(0, 3, 1, 2).float().div(255)
        lg = _logits_batch(wm, method, bt, device, bs=32).numpy()
        full_accs[i0:i1] = ((lg > 0).astype(np.uint8) == msgs[i0:i1]).mean(axis=1)

    if smoke:
        boxes_128[:] = [0, 128, 0, 128]
        best_accs[:] = full_accs
        return boxes_128, best_accs, full_accs

    for i in range(N):
        if full_accs[i] >= 0.72:
            boxes_128[i] = [0, 128, 0, 128]
            best_accs[i] = full_accs[i]
            continue

        img_t = torch.from_numpy(imgs_uint8[i]).permute(2, 0, 1)[None].float().div(255)
        msg = msgs[i]

        cands = []
        if top_crops is not None:
            for _, y0, y1, x0, x1 in top_crops[i]:
                if _valid_box(y0, y1, x0, x1, max_ar, min_side):
                    cands.append((y0, y1, x0, x1))

        pm = prob_maps[i]
        for pthr in (0.25, 0.45):
            ys, xs = np.where(pm > pthr)
            if len(ys) > 0:
                y0, y1 = int(ys.min()) * 4, int(ys.max() + 1) * 4
                x0, x1 = int(xs.min()) * 4, int(xs.max() + 1) * 4
                if _valid_box(y0, y1, x0, x1, max_ar, min_side):
                    cands.append((y0, y1, x0, x1))

        for ch, win in ((2, 96), (1, 160), (0, 256)):
            m = ms_maps[i, ch]
            py, px = np.unravel_index(np.argmax(m), m.shape)
            cy, cx = int(py) * 4, int(px) * 4
            for h_box, w_box in ((win, win), (win // 2, min(512, win * 2)), (min(512, win * 2), win // 2)):
                y0 = max(0, min(512 - h_box, cy - h_box // 2))
                x0 = max(0, min(512 - w_box, cx - w_box // 2))
                if _valid_box(y0, y0 + h_box, x0, x0 + w_box, max_ar, min_side):
                    cands.append((y0, y0 + h_box, x0, x0 + w_box))

        for y0 in range(0, 512 - 96 + 1, 96):
            for x0 in range(0, 512 - 96 + 1, 96):
                cands.append((y0, min(512, y0 + 128), x0, min(512, x0 + 128)))
        # Add 64x64 and 48x48 sub-boxes inside top seed regions to catch tiny <5% rectangles
        base_cands = list(dict.fromkeys(cands))
        for (sy0, sy1, sx0, sx1) in base_cands[:6]:
            sh, sw = sy1 - sy0, sx1 - sx0
            for sz in (64, 48):
                if sh >= sz and sw >= sz:
                    for ry in (0.0, 0.5, 1.0):
                        for rx in (0.0, 0.5, 1.0):
                            yy0 = int(sy0 + ry * (sh - sz))
                            xx0 = int(sx0 + rx * (sw - sz))
                            b = (yy0, yy0 + sz, xx0, xx0 + sz)
                            if _valid_box(*b, max_ar, min_side):
                                cands.append(b)
        cands = list(dict.fromkeys(cands))

        scores = _score_boxes(wm, method, img_t, msg, cands, device, target_res)
        best_idx = int(np.argmax([s[0] + 1e-3 * s[1] for s in scores]))
        best_box = cands[best_idx]
        best_sc = scores[best_idx]

        for step in (32, 16, 8, 4, 2):
            y0, y1, x0, x1 = best_box
            neighbors = []
            for dy0 in (-step, 0, step):
                for dy1 in (-step, 0, step):
                    if dy0 == 0 and dy1 == 0:
                        continue
                    b = (y0 + dy0, y1 + dy1, x0, x1)
                    if _valid_box(*b, max_ar, min_side):
                        neighbors.append(b)
            for dx0 in (-step, 0, step):
                for dx1 in (-step, 0, step):
                    if dx0 == 0 and dx1 == 0:
                        continue
                    b = (y0, y1, x0 + dx0, x1 + dx1)
                    if _valid_box(*b, max_ar, min_side):
                        neighbors.append(b)
            # Symmetric shrink/expand & shifts
            for d in (-step, step):
                for b in ((y0 + d, y1 - d, x0 + d, x1 - d),
                          (y0 + d, y1 + d, x0, x1),
                          (y0, y1, x0 + d, x1 + d)):
                    if _valid_box(*b, max_ar, min_side):
                        neighbors.append(b)
            neighbors = list(dict.fromkeys(neighbors))
            if not neighbors:
                continue
            n_scores = _score_boxes(wm, method, img_t, msg, neighbors, device, target_res)
            nb_idx = int(np.argmax([s[0] + 1e-3 * s[1] for s in n_scores]))
            if (n_scores[nb_idx][0] + 1e-3 * n_scores[nb_idx][1]) > (best_sc[0] + 1e-3 * best_sc[1]):
                best_box = neighbors[nb_idx]
                best_sc = n_scores[nb_idx]

        y0, y1, x0, x1 = best_box
        boxes_128[i] = [round(y0 / 4), round(y1 / 4), round(x0 / 4), round(x1 / 4)]
        best_accs[i] = best_sc[0]

    return boxes_128, best_accs, full_accs
