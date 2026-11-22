"""Ensemble available versions + hole/rect specialist blends + dynamic threshold (nested-CV)
+ decoder-verified bbox overrides (v4/v5/v6-fine-rect), and write submission .pt.

    python work_bassim/postprocess_submit_v5.py --out output/submission_rects.pt [--submit]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_core import load_rect_boxes  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import miou_image  # noqa: E402

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


def apply_bbox_rule(masks_list, boxes, bacc, facc, b_thr, f_thr, a_thr):
    if b_thr > 1.5:
        return masks_list
    out = []
    for m, box, ba, fa in zip(masks_list, boxes, bacc, facc):
        if fa < f_thr and ba >= b_thr and m.mean() < a_thr:
            y0, y1, x0, x1 = box
            nm = np.zeros((128, 128), np.uint8)
            nm[y0:y1, x0:x1] = 1
            out.append(nm)
        else:
            out.append(m)
    return out


def apply_rect_rule(masks_list, boxes, logp, facc, lp_thr, f_thr, a_max):
    out = []
    for m, box, lp, fa in zip(masks_list, boxes, logp, facc):
        y0, y1, x0, x1 = box
        b_area = ((y1 - y0) * (x1 - x0)) / (128.0 * 128.0)
        if fa < f_thr and lp >= lp_thr and b_area <= a_max:
            nm = np.zeros((128, 128), np.uint8)
            nm[y0:y1, x0:x1] = 1
            out.append(nm)
        else:
            out.append(m)
    return out


def simplex_weights(n: int, step: float = 0.2):
    if n == 1:
        return [(1.0,)]
    vals = [round(i * step, 2) for i in range(int(round(1.0 / step)) + 1)]
    out = []
    def rec(rem, k, cur):
        if k == 1:
            out.append(tuple(cur + [round(rem, 2)]))
            return
        for v in vals:
            if v <= rem + 1e-6:
                rec(rem - v, k - 1, cur + [v])
    rec(1.0, n, [])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/submission_rects.pt")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()

    test_names = [m["image_name"] for m in LocalisationDataset("release", "test").meta]
    pred = {}
    oof_scores = []

    for method in ("pixelseal", "trustmark", "stegastamp"):
        versions = []
        masks_oof, names = None, None
        oof_b, te_b = None, None
        for tag in ("v2", "v3", "v4", "v41", "v5", "v5b", "v6p"):
            po_p, pt_p = f"work_bassim/oof_{tag}_{method}.npz", f"work_bassim/test_{tag}_{method}.npz"
            if os.path.exists(po_p) and os.path.exists(pt_p):
                zo, zt = np.load(po_p), np.load(pt_p)
                versions.append((tag, zo["probs"].astype(np.float32), zt["probs"].astype(np.float32)))
                masks_oof = zo["masks"]
                names = zt["names"].tolist()
        bb_v4 = f"work_bassim/bbox_v4_{method}.npz"
        bb_v5 = f"work_bassim/bbox_v5_{method}.npz"
        oof_b4 = te_b4 = oof_b5 = te_b5 = None
        if os.path.exists(bb_v4):
            z4 = np.load(bb_v4)
            oof_b4 = (z4["tr_boxes"], z4["tr_bacc"], z4["tr_facc"])
            te_b4 = (z4["te_boxes"], z4["te_bacc"], z4["te_facc"])
        if os.path.exists(bb_v5):
            z5 = np.load(bb_v5)
            b5_sc = z5["tr_logp"] if "tr_logp" in z5 else z5["tr_bacc"]
            te5_sc = z5["te_logp"] if "te_logp" in z5 else z5["te_bacc"]
            oof_b5 = (z5["tr_boxes"], b5_sc, z5["tr_facc"])
            te_b5 = (z5["te_boxes"], te5_sc, z5["te_facc"])
        oof_b = oof_b4 or oof_b5
        te_b = te_b4 or te_b5
        oof_b6, te_b6 = load_rect_boxes(method)  # prefers bbox_v7, falls back to bbox_v6
        if oof_b6 is not None:
            src = "bbox_v7" if os.path.exists(f"work_bassim/bbox_v7_{method}.npz") else "bbox_v6"
            print(f"  [{method}] rect boxes from {src}", flush=True)
        assert versions, f"No predictions found for {method}"

        # 1. Precompute per-image mIoU for fast threshold search (without CC first), then tune CC/rect
        weight_grid = simplex_weights(len(versions), step=0.2 if len(versions) <= 4 else 0.25)
        best_w = (-1.0, weight_grid[0], 0.5)
        for ws in weight_grid:
            po = sum(w * v[1] for w, v in zip(ws, versions))
            for thr in np.linspace(0.25, 0.70, 19):
                s = float(np.mean([miou_image((p > thr).astype(np.uint8), g) for p, g in zip(po, masks_oof)]))
                if s > best_w[0]:
                    best_w = (s, ws, float(thr))

        bws, bthr = best_w[1], best_w[2]
        po = sum(w * v[1] for w, v in zip(bws, versions))
        pt = sum(w * v[2] for w, v in zip(bws, versions))

        # 1a. Per-regime simplex weights (full_acc >= 0.72 vs partial), OOF-validated
        if oof_b is not None:
            hi_o = oof_b[2] >= 0.72
            hi_t = te_b[2] >= 0.72

            def search_subset(sel_o):
                sub_masks = [g for g, s in zip(masks_oof, sel_o) if s]
                best = (-1.0, weight_grid[0], 0.5)
                for ws in weight_grid:
                    po_s = sum(w * v[1][sel_o] for w, v in zip(ws, versions))
                    for thr in np.linspace(0.25, 0.70, 19):
                        s = float(np.mean([miou_image((p > thr).astype(np.uint8), g) for p, g in zip(po_s, sub_masks)]))
                        if s > best[0]:
                            best = (s, ws, float(thr))
                return best

            if hi_o.any() and (~hi_o).any():
                b_hi = search_subset(hi_o)
                b_lo = search_subset(~hi_o)
                s_regime = float(b_hi[0] * hi_o.mean() + b_lo[0] * (1 - hi_o.mean()))
                if s_regime > best_w[0] + 1e-4:
                    po2 = np.zeros_like(po)
                    pt2 = np.zeros_like(pt)
                    po2[hi_o] = sum(w * v[1][hi_o] for w, v in zip(b_hi[1], versions))
                    po2[~hi_o] = sum(w * v[1][~hi_o] for w, v in zip(b_lo[1], versions))
                    pt2[hi_t] = sum(w * v[2][hi_t] for w, v in zip(b_hi[1], versions))
                    pt2[~hi_t] = sum(w * v[2][~hi_t] for w, v in zip(b_lo[1], versions))
                    best_thr, best_s = bthr, best_w[0]
                    for thr in np.linspace(0.25, 0.70, 19):
                        s = float(np.mean([miou_image((p > thr).astype(np.uint8), g) for p, g in zip(po2, masks_oof)]))
                        if s > best_s:
                            best_thr, best_s = float(thr), s
                    po, pt, bthr = po2, pt2, best_thr
                    fw_str = ", ".join(f"{v0[0]}={w:.2f}" for w, v0 in zip(b_hi[1], versions))
                    lo_str = ", ".join(f"{v0[0]}={w:.2f}" for w, v0 in zip(b_lo[1], versions))
                    print(f"  [{method}] per-regime weights accepted ({best_w[0]:.4f} -> {best_s:.4f}) fw[{fw_str}] part[{lo_str}]")

        # 1b. Optional Hole Specialist blend on full_acc >= 0.72 (strictly OOF-validated)
        hole_o_list, hole_t_list = [], []
        for htag in ("hole", "hole2"):
            ho_p, ht_p = f"work_bassim/oof_{htag}_{method}.npz", f"work_bassim/test_{htag}_{method}.npz"
            if os.path.exists(ho_p) and os.path.exists(ht_p):
                hole_o_list.append(1.0 - np.load(ho_p)["hole_probs"].astype(np.float32))
                hole_t_list.append(1.0 - np.load(ht_p)["hole_probs"].astype(np.float32))
        if hole_o_list and oof_b is not None:
            ho_wm = np.mean(hole_o_list, axis=0)
            ht_wm = np.mean(hole_t_list, axis=0)
            is_fw_o = (oof_b[2] >= 0.72)[:, None, None]
            is_fw_t = (te_b[2] >= 0.72)[:, None, None]
            cur0 = max(float(np.mean([miou_image((p > t).astype(np.uint8), g) for p, g in zip(po, masks_oof)]))
                       for t in np.linspace(0.25, 0.70, 19))
            best_h = (cur0, 0.0)
            for wh in (0.2, 0.35, 0.5, 0.7):
                cand_po = np.where(is_fw_o, (1.0 - wh) * po + wh * ho_wm, po)
                s_h = max(float(np.mean([miou_image((p > t).astype(np.uint8), g) for p, g in zip(cand_po, masks_oof)]))
                          for t in np.linspace(0.25, 0.70, 19))
                if s_h > best_h[0] + 2e-4:
                    best_h = (s_h, wh)
            if best_h[1] > 0:
                wh = best_h[1]
                po = np.where(is_fw_o, (1.0 - wh) * po + wh * ho_wm, po)
                pt = np.where(is_fw_t, (1.0 - wh) * pt + wh * ht_wm, pt)
                print(f"  [{method}] hole specialist accepted w_hole={wh:.2f} ({cur0:.4f} -> {best_h[0]:.4f})")

        # 1c. Optional Rect Specialist blend on full_acc < 0.72 (strictly OOF-validated)
        ro_p, rt_p = f"work_bassim/oof_rect_{method}.npz", f"work_bassim/test_rect_{method}.npz"
        if os.path.exists(ro_p) and os.path.exists(rt_p) and oof_b is not None:
            zro, zrt = np.load(ro_p), np.load(rt_p)
            ro = zro["probs"].astype(np.float32)
            rt_names = zrt["names"].tolist()
            by_name = {n: row for n, row in zip(rt_names, np.load(rt_p)["probs"].astype(np.float32))}
            rt = np.stack([by_name[n] for n in names])
            is_part_o = (oof_b[2] < 0.72)[:, None, None]
            is_part_t = (te_b[2] < 0.72)[:, None, None]
            cur_s = max(float(np.mean([miou_image((p > t).astype(np.uint8), g) for p, g in zip(po, masks_oof)]))
                        for t in np.linspace(0.25, 0.70, 19))
            best_r = (cur_s, 0.0)
            for wr in (0.2, 0.35, 0.5, 0.7):
                cand_po = np.where(is_part_o, (1.0 - wr) * po + wr * ro, po)
                s_r = max(float(np.mean([miou_image((p > t).astype(np.uint8), g) for p, g in zip(cand_po, masks_oof)]))
                          for t in np.linspace(0.25, 0.70, 19))
                if s_r > best_r[0] + 2e-4:
                    best_r = (s_r, wr)
            if best_r[1] > 0:
                wr = best_r[1]
                po = np.where(is_part_o, (1.0 - wr) * po + wr * ro, po)
                pt = np.where(is_part_t, (1.0 - wr) * pt + wr * rt, pt)
                print(f"  [{method}] rect specialist accepted w_rect={wr:.2f} ({cur_s:.4f} -> {best_r[0]:.4f})")

        # 2. Tune CC & rect_fill
        best_pp = (-1.0, bthr, 0, 2.0)
        for thr in np.linspace(max(0.20, bthr - 0.10), min(0.75, bthr + 0.10), 9):
            base = [(p > thr).astype(np.uint8) for p in po]
            for min_cc in (0, 8, 20):
                for rf in (2.0, 0.90):
                    pp = [clean_mask(m, min_cc, rf) for m in base] if (min_cc or rf <= 1.0) else base
                    s = float(np.mean([miou_image(m, g) for m, g in zip(pp, masks_oof)]))
                    if s > best_pp[0]:
                        best_pp = (s, float(thr), min_cc, rf)
        _, bthr, bcc, brf = best_pp

        # 3. Nested 5-fold CV check for dynamic threshold conditioned on full_acc >= 0.75
        use_dyn = False
        t_hi_final, t_lo_final = bthr, bthr
        if oof_b is not None:
            facc = oof_b[2]
            hi_grid = np.linspace(0.25, 0.65, 9)
            lo_grid = np.linspace(0.20, 0.60, 9)
            # Precompute per-image scores for each threshold on clean_mask(p > t, bcc, brf)
            sc_by_t = {}
            for t in sorted(set(hi_grid.tolist() + lo_grid.tolist() + [bthr])):
                sc_by_t[round(float(t), 4)] = np.array([
                    miou_image(clean_mask((p > t).astype(np.uint8), bcc, brf), g)
                    for p, g in zip(po, masks_oof)
                ])
            base_arr = sc_by_t[round(bthr, 4)]

            idx = np.arange(len(masks_oof))
            nested_sc = np.zeros(len(masks_oof), np.float32)
            for va in np.array_split(idx, 5):
                tr = np.setdiff1d(idx, va)
                tr_hi = facc[tr] >= 0.75
                best_t_hi = max(hi_grid, key=lambda t: sc_by_t[round(float(t), 4)][tr][tr_hi].mean() if tr_hi.any() else 0.0)
                best_t_lo = max(lo_grid, key=lambda t: sc_by_t[round(float(t), 4)][tr][~tr_hi].mean() if (~tr_hi).any() else 0.0)
                va_hi = facc[va] >= 0.75
                nested_sc[va] = np.where(va_hi, sc_by_t[round(float(best_t_hi), 4)][va], sc_by_t[round(float(best_t_lo), 4)][va])

            if nested_sc.mean() > base_arr.mean() + 1e-3:
                use_dyn = True
                tr_hi_all = facc >= 0.75
                t_hi_final = float(max(hi_grid, key=lambda t: sc_by_t[round(float(t), 4)][tr_hi_all].mean()))
                t_lo_final = float(max(lo_grid, key=lambda t: sc_by_t[round(float(t), 4)][~tr_hi_all].mean()))
                print(f"  [{method}] nested CV dyn thr accepted: {base_arr.mean():.4f} -> {nested_sc.mean():.4f} (t_hi={t_hi_final:.2f}, t_lo={t_lo_final:.2f})")
            else:
                print(f"  [{method}] nested CV dyn thr rejected ({nested_sc.mean():.4f} vs {base_arr.mean():.4f})")

        if use_dyn:
            oof_m = [clean_mask((p > (t_hi_final if fa >= 0.75 else t_lo_final)).astype(np.uint8), bcc, brf)
                     for p, fa in zip(po, oof_b[2])]
            te_m = [clean_mask((p > (t_hi_final if fa >= 0.75 else t_lo_final)).astype(np.uint8), bcc, brf)
                    for p, fa in zip(pt, te_b[2])]
        else:
            oof_m = [clean_mask((p > bthr).astype(np.uint8), bcc, brf) for p in po]
            te_m = [clean_mask((p > bthr).astype(np.uint8), bcc, brf) for p in pt]

        # 4. Tune two-stage bbox override (bbox_v4 safe medium/small + bbox_v5 tiny rect_small) strictly on OOF CV
        s_pre = float(np.mean([miou_image(m, g) for m, g in zip(oof_m, masks_oof)]))
        best_bb4 = (s_pre, 2.0, 0.0, 0.0)
        if oof_b4 is not None:
            for b_thr in (0.68, 0.72, 0.76, 0.80, 0.85):
                for f_thr in (0.58, 0.65):
                    for a_thr in (0.15, 0.25, 0.45):
                        cand = apply_bbox_rule(oof_m, *oof_b4, b_thr, f_thr, a_thr)
                        s = float(np.mean([miou_image(m, g) for m, g in zip(cand, masks_oof)]))
                        if s > best_bb4[0] + 1e-4:
                            best_bb4 = (s, b_thr, f_thr, a_thr)
            if best_bb4[1] <= 1.0:
                oof_m = apply_bbox_rule(oof_m, *oof_b4, *best_bb4[1:])
                te_m = apply_bbox_rule(te_m, *te_b4, *best_bb4[1:])

        s_post4 = float(np.mean([miou_image(m, g) for m, g in zip(oof_m, masks_oof)]))
        best_bb5 = (s_post4, 99.0, 0.0, 0.0, 0.0)
        if oof_b5 is not None:
            def apply_v5_small(mlist, boxes, sc, facc, lp_thr, f_thr, max_area):
                out = []
                for m, box, s_val, fa in zip(mlist, boxes, sc, facc):
                    y0, y1, x0, x1 = box
                    b_area = ((y1 - y0) * (x1 - x0)) / (128.0 * 128.0)
                    if fa < f_thr and s_val >= lp_thr and m.mean() < max_area and b_area < 0.16:
                        nm = np.zeros((128, 128), np.uint8)
                        nm[y0:y1, x0:x1] = 1
                        out.append(nm)
                    else:
                        out.append(m)
                return out

            for lp_thr in (9.0, 12.0, 16.0):
                for f_thr in (0.58, 0.64):
                    for max_area in (0.08, 0.12, 0.20):
                        cand = apply_v5_small(oof_m, *oof_b5, lp_thr, f_thr, max_area)
                        s = float(np.mean([miou_image(m, g) for m, g in zip(cand, masks_oof)]))
                        if s > best_bb5[0] + 1e-4:
                            best_bb5 = (s, lp_thr, f_thr, max_area)
            if best_bb5[1] < 90.0:
                oof_m = apply_v5_small(oof_m, *oof_b5, *best_bb5[1:])
                te_m = apply_v5_small(te_m, *te_b5, *best_bb5[1:])

        s_final = float(np.mean([miou_image(m, g) for m, g in zip(oof_m, masks_oof)]))

        # 5. Decoder-verified FINE rect override (bbox_v6, strictly OOF-validated)
        best_v6 = (s_final, 99.0, 0.0, 0.0)
        if oof_b6 is not None and te_b6 is not None:
            for lp_thr in (6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
                for f_thr in (0.58, 0.65):
                    for a_max in (0.16, 0.30, 0.45):
                        cand = apply_rect_rule(oof_m, *oof_b6, lp_thr, f_thr, a_max)
                        s = float(np.mean([miou_image(m, g) for m, g in zip(cand, masks_oof)]))
                        if s > best_v6[0] + 1e-4:
                            best_v6 = (s, lp_thr, f_thr, a_max)
            if best_v6[1] < 90.0:
                oof_m = apply_rect_rule(oof_m, *oof_b6, *best_v6[1:])
                te_m = apply_rect_rule(te_m, *te_b6, *best_v6[1:])
                s_final = best_v6[0]

        w_str = ", ".join(f"{v[0]}={w:.1f}" for w, v in zip(bws, versions))
        bb4_str = f"v4(>={best_bb4[1]:.2f})" if best_bb4[1] <= 1.0 else "no_v4"
        bb5_str = f"v5_sm(lp>={best_bb5[1]:.0f})" if best_bb5[1] < 90.0 else "no_v5"
        bb6_str = f"v6(lp>={best_v6[1]:.0f},f<{best_v6[2]:.2f},a<={best_v6[3]:.2f})" if best_v6[1] < 90.0 else "no_v6"
        sc_final = [miou_image(m, g) for m, g in zip(oof_m, masks_oof)]
        oof_scores.extend(sc_final)
        print(f"[{method}] OOF={np.mean(sc_final):.4f} ({w_str}, cc={bcc}, rf={brf}, {bb4_str}+{bb5_str}+{bb6_str})")
        for n, m in zip(names, te_m):
            pred[n] = m

    overall_oof = float(np.mean(oof_scores))
    print(f"Overall Ensemble OOF mIoU (510 real imgs): {overall_oof:.4f}")
    missing = [n for n in test_names if n not in pred]
    assert not missing, f"missing {len(missing)}"
    masks = torch.from_numpy(np.stack([pred[n] for n in test_names])).bool()
    assert masks.shape == (len(test_names), 128, 128) and len(set(test_names)) == len(test_names)
    torch.save({"image_names": test_names, "masks": masks}, a.out)
    with open(f"{a.out}.oof", "w") as f:
        f.write(f"{overall_oof:.6f}\n")
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
