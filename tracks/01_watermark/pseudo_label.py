"""Phase 4: Transductive self-training with triple-filtered test pseudo-labels.

Teacher = best CV-weighted ensemble over available versions (v2, v3, v4, v41, v5, v5b).
Triple filter on the 1020 test images:
  1. Cross-version agreement (low mean pixel std across teacher models)
  2. Sharp margin (low fraction of ambiguous pixels in [0.18, 0.82])
  3. Decoder verification using known test message:
     - either decoder-verified rect from bbox_v5 (log10p >= 6.0), OR
     - high full-image bit_accuracy (full_acc >= 0.72) for >= 35% area masks, OR
     - tight bbox of predicted mask decodes >= 0.75 bit accuracy.
Retrains 5-fold CV on (real train fold + synthetic + filtered test pseudo-labels),
evaluated strictly on real train OOF (never on test).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from ms_maps import compute_ms_maps  # noqa: E402
from postprocess_submit_v4 import apply_bbox_rule, clean_mask, simplex_weights  # noqa: E402
from synth_v4 import generate_synthetic_v4, load_coco_tar_and_masks  # noqa: E402
from train_v4 import SSContrib, loss_v4, predict_all, run_phase  # noqa: E402
from train_v5 import build_model_v5  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import miou_image, score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


def build_teacher_and_pseudo(method: str):
    """Build CV-optimal teacher ensemble on OOF & test, and select high-confidence pseudo-labels."""
    versions = []
    masks_oof, names_te = None, None
    for tag in ("v2", "v3", "v4", "v41", "v5", "v5b"):
        po_p, pt_p = f"work_bassim/oof_{tag}_{method}.npz", f"work_bassim/test_{tag}_{method}.npz"
        if os.path.exists(po_p) and os.path.exists(pt_p):
            zo, zt = np.load(po_p), np.load(pt_p)
            versions.append((tag, zo["probs"].astype(np.float32), zt["probs"].astype(np.float32)))
            masks_oof = zo["masks"]
            names_te = zt["names"].tolist()
    assert versions

    w_grid = simplex_weights(len(versions), step=0.2 if len(versions) <= 4 else 0.25)
    best_w = (-1.0, w_grid[0], 0.5)
    for ws in w_grid:
        po = sum(w * v[1] for w, v in zip(ws, versions))
        for thr in np.linspace(0.25, 0.70, 19):
            s = float(np.mean([miou_image((p > thr).astype(np.uint8), g) for p, g in zip(po, masks_oof)]))
            if s > best_w[0]:
                best_w = (s, ws, float(thr))

    bws, bthr = best_w[1], best_w[2]
    pt = sum(w * v[2] for w, v in zip(bws, versions))

    # Version disagreement on test (among active versions with weight > 0)
    active_te = [v[2] for w, v in zip(bws, versions) if w > 0]
    te_std = np.std(np.stack(active_te, axis=0), axis=0).mean(axis=(1, 2)) if len(active_te) > 1 else np.zeros(len(pt))
    ambig = ((pt > 0.18) & (pt < 0.82)).mean(axis=(1, 2))

    # Load bbox_v5 (or bbox_v4) for decoder verification
    bb_p = f"work_bassim/bbox_v5_{method}.npz"
    if not os.path.exists(bb_p):
        bb_p = f"work_bassim/bbox_v4_{method}.npz"
    zb = np.load(bb_p)
    te_boxes, te_bacc, te_facc = zb["te_boxes"], zb["te_bacc"], zb["te_facc"]
    te_logp = zb["te_logp"] if "te_logp" in zb else None

    pseudo_masks = (pt > bthr).astype(np.float32)
    keep = np.zeros(len(pt), dtype=bool)
    for i in range(len(pt)):
        # Criterion A: statistically verified rectangle from rect_search
        is_verified_rect = (te_logp[i] >= 6.5) if te_logp is not None else (te_bacc[i] >= 0.80)
        if te_facc[i] < 0.62 and is_verified_rect and pseudo_masks[i].mean() < 0.35:
            y0, y1, x0, x1 = te_boxes[i]
            m = np.zeros((128, 128), np.float32)
            m[y0:y1, x0:x1] = 1.0
            pseudo_masks[i] = m
            keep[i] = True
            continue
        # Criterion B: sharp ensemble agreement + decoder consistency
        area = float(pseudo_masks[i].mean())
        dec_ok = (te_facc[i] >= 0.70 and area >= 0.30) or (te_bacc[i] >= 0.72 and area < 0.30)
        if ambig[i] < 0.12 and te_std[i] < 0.14 and dec_ok:
            keep[i] = True

    print(f"[{method}] teacher OOF={best_w[0]:.4f} | selected {int(keep.sum())}/{len(pt)} pseudo-labels", flush=True)
    return keep, pseudo_masks, names_te


def main():
    import train_v4
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--n_synth", type=int, default=1400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ep1", type=int, default=8)
    ap.add_argument("--ep2", type=int, default=6)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--tag", default="v6p")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    train_v4.dev = args.device
    global dev
    dev = args.device
    train_v4.build_model = lambda m, w: build_model_v5(m, w, no_aux=(m != "pixelseal"))

    t0 = time.time()
    keep, pseudo_masks, names_te = build_teacher_and_pseudo(args.method)

    wm = load_watermark(args.method, models_dir="models", device=dev)
    covers, cover_masks, all_packed = load_coco_tar_and_masks(
        "clean_covers/val2017.tar", "clean_covers/coco_masks_4k.npz", max_covers=2000, seed=args.seed,
    )
    Xs, Ms, Msgs = generate_synthetic_v4(wm, args.method, covers, cover_masks, all_packed, n_samples=args.n_synth, seed=args.seed)

    ds_tr = LocalisationDataset("release", "train", methods=[args.method])
    ds_te = LocalisationDataset("release", "test", methods=[args.method])
    Xr = np.stack([ds_tr[i]["image"] for i in range(len(ds_tr))])
    Mr = np.stack([ds_tr[i]["mask128"] for i in range(len(ds_tr))])
    Msgr = np.stack([ds_tr[i]["message"] for i in range(len(ds_tr))])
    Xte = np.stack([ds_te[i]["image"] for i in range(len(ds_te))])
    Msgte = np.stack([ds_te[i]["message"] for i in range(len(ds_te))])

    if args.method == "trustmark":
        Exs = np.zeros((len(Xs), 1, 128, 128), np.float32)
        Exr = np.zeros((len(Xr), 1, 128, 128), np.float32)
        Exte = np.zeros((len(Xte), 1, 128, 128), np.float32)
    else:
        ms_s = compute_ms_maps(wm, args.method, Xs, Msgs, device=dev)
        zb = np.load(f"work_bassim/bbox_v4_{args.method}.npz")
        ms_r, ms_te = zb["tr_ms"].astype(np.float32), zb["te_ms"].astype(np.float32)
        if args.method == "stegastamp":
            ss_c = SSContrib(wm)
            c_s, c_r, c_te = ss_c.batch_maps(Xs, Msgs), ss_c.batch_maps(Xr, Msgr), ss_c.batch_maps(Xte, Msgte)
            mu, sd = float(c_s.mean()), float(c_s.std() + 1e-6)
            Exs = np.concatenate([(c_s - mu) / sd, ms_s], axis=1)
            Exr = np.concatenate([(c_r - mu) / sd, ms_r], axis=1)
            Exte = np.concatenate([(c_te - mu) / sd, ms_te], axis=1)
        else:
            Exs, Exr, Exte = ms_s, ms_r, ms_te

    # Combine synthetic + selected test pseudo-labels into P1 pool
    Xp = Xte[keep]
    Mp = pseudo_masks[keep]
    Msgp = Msgte[keep]
    Exp = Exte[keep]
    X_pool = np.concatenate([Xs, np.repeat(Xp, 2, axis=0)]) if len(Xp) else Xs
    M_pool = np.concatenate([Ms, np.repeat(Mp, 2, axis=0)]) if len(Xp) else Ms
    Msg_pool = np.concatenate([Msgs, np.repeat(Msgp, 2, axis=0)]) if len(Xp) else Msgs
    Ex_pool = np.concatenate([Exs, np.repeat(Exp, 2, axis=0)]) if len(Xp) else Exs

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Xr))
    oof = np.zeros(Mr.shape, np.float32)
    test_probs = np.zeros((len(Xte), 128, 128), np.float32)

    for k, va in enumerate(np.array_split(idx, args.folds)):
        tr = np.setdiff1d(idx, va)
        t1 = time.time()
        model, best_s = train_v4.train_fold(
            args.method, wm, X_pool, M_pool, Msg_pool, Ex_pool,
            Xr[tr], Mr[tr], Msgr[tr], Exr[tr],
            Xr[va], Mr[va], Msgr[va], Exr[va],
            args, args.seed + k,
        )
        oof[va] = predict_all(args.method, model, Xr[va], Msgr[va], Exr[va])
        test_probs += predict_all(args.method, model, Xte, Msgte, Exte) / args.folds
        print(f"  [fold {k}/{args.folds}] best val mIoU@0.5={best_s:.4f} ({time.time() - t1:.0f}s)", flush=True)
        np.savez_compressed(f"/tmp/partial_{args.tag}_{args.method}.npz",
                            oof=oof.astype(np.float16), test_sum=(test_probs * args.folds).astype(np.float16),
                            done_folds=k + 1)

    s05 = score([(p > 0.5).astype(np.uint8) for p in oof], Mr)
    best_t, best_s = 0.5, s05
    for t in np.linspace(0.25, 0.75, 21):
        s = score([(p > t).astype(np.uint8) for p in oof], Mr)
        if s > best_s:
            best_t, best_s = float(t), s
    print(f"[{args.method}] {args.tag.upper()} OOF mIoU@0.5={s05:.4f} | best thr={best_t:.3f} -> {best_s:.4f} | total {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(f"work_bassim/oof_{args.tag}_{args.method}.npz", probs=oof.astype(np.float16), masks=Mr, idx=idx)
    np.savez_compressed(f"work_bassim/test_{args.tag}_{args.method}.npz", probs=test_probs.astype(np.float16),
                        names=np.array(names_te), thr=best_t, oof_miou=best_s)


if __name__ == "__main__":
    main()
