"""Phase 5: Dedicated Hole Specialist (predicts clean hole = 1 - mask inside watermarked images).

Targets the 50% of images with full_watermark=True (identified at ~98% accuracy via full_acc >= 0.72),
especially tiny 1-5% clean holes where generalist models plateau at 0.50-0.65 mIoU.

- Synthetic data: 100% hole-only edits using small/medium COCO object masks (0.5%-25% clean hole).
- Real data: ONLY full_watermark=True real train images (85 per method, 5-fold CV), evaluated on all
  real train images when blended via w_hole on full_acc >= 0.72 in postprocess_submit_v4.py.
- Partial checkpoints saved to /tmp/ to avoid /p/scratch inode exhaustion.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from ms_maps import compute_ms_maps  # noqa: E402
from synth_v4 import _sample_object_mask, load_coco_tar_and_masks  # noqa: E402
from train_v4 import SSContrib, predict_all  # noqa: E402
from train_v5 import build_model_v5  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import downsample_mask, score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


def generate_hole_synth(wm, method: str, covers, cover_masks, all_packed, n_samples: int = 1400, seed: int = 31):
    rng = np.random.default_rng(seed)
    n_cov = len(covers)
    imgs = np.empty((n_samples, 512, 512, 3), np.uint8)
    hole_masks128 = np.empty((n_samples, 128, 128), np.uint8)
    msgs = np.empty((n_samples, wm.nbits), np.uint8)

    for i in range(n_samples):
        ci = int(rng.integers(0, n_cov))
        cov = covers[ci]
        bits = rng.integers(0, 2, wm.nbits, dtype=np.uint8)
        wm_full = wm.embed(cov, bits).copy()

        # Heavily bias toward small 0.5% - 15% clean holes
        prefer_small = rng.random() < 0.75
        obj = _sample_object_mask(rng, ci, cover_masks, all_packed, prefer_small)
        r = rng.random()
        if r < 0.65:
            # Pure zero-seam restoration hole
            wm_full[obj == 1] = cov[obj == 1]
        elif r < 0.85:
            # Simulated inpainting / texture fill inside hole
            blur = cv2.GaussianBlur(cov, (13, 13), 3.5)
            wm_full[obj == 1] = blur[obj == 1]
        else:
            # Clean object addition from another cover
            cov2 = covers[int(rng.integers(0, n_cov))]
            wm_full[obj == 1] = cov2[obj == 1]

        if rng.random() < 0.25:
            q = int(rng.integers(80, 98))
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(wm_full, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
            wm_full = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        imgs[i] = wm_full
        hole_masks128[i] = downsample_mask(obj)
        msgs[i] = bits

    return imgs, hole_masks128, msgs


def loss_hole(logit: torch.Tensor, y_hole: torch.Tensor) -> torch.Tensor:
    """Balanced per-image BCE + soft mIoU focusing on tiny 1-5% clean holes."""
    loss_el = F.binary_cross_entropy_with_logits(logit, y_hole, reduction="none")
    bce = loss_el.mean()
    pos_w = y_hole.sum((1, 2), keepdim=True).clamp(min=1.0)
    neg_w = (1.0 - y_hole).sum((1, 2), keepdim=True).clamp(min=1.0)
    bal_bce = 0.5 * ((loss_el * y_hole / pos_w).sum((1, 2)) + (loss_el * (1.0 - y_hole) / neg_w).sum((1, 2))).mean()

    p = torch.sigmoid(logit)
    sm = 0.05
    iou1 = ((p * y_hole).sum((1, 2)) + sm) / ((p + y_hole - p * y_hole).sum((1, 2)) + sm)
    iou0 = (((1.0 - p) * (1.0 - y_hole)).sum((1, 2)) + sm) / (((1.0 - p) + (1.0 - y_hole) - (1.0 - p) * (1.0 - y_hole)).sum((1, 2)) + sm)
    miou_l = (1.0 - 0.5 * (iou1 + iou0)).mean()
    return 0.4 * bce + 0.4 * bal_bce + 1.2 * miou_l


def run_hole_phase(method, model, opt, sched, Xtr, Htr, Msg_tr, Extr_tr, Xva, Hva, Msg_va, Exva_va,
                   epochs, bs, rng, best_s, best_state, label, predict_fn):
    import train_v4
    if method in ("pixelseal", "trustmark"):
        flip_mask = rng.random(len(Xtr)) < 0.5
        if flip_mask.any():
            Xtr = Xtr.copy(); Htr = Htr.copy(); Extr_tr = Extr_tr.copy()
            Xtr[flip_mask] = Xtr[flip_mask][:, :, ::-1]
            Htr[flip_mask] = Htr[flip_mask][:, :, ::-1]
            Extr_tr[flip_mask] = Extr_tr[flip_mask][:, :, :, ::-1]

    n = len(Xtr)
    use_amp = dev == "cuda"
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        tot_loss, nb = 0.0, 0
        for k in range(0, n, bs):
            idx = order[k:k + bs]
            xb = torch.from_numpy(Xtr[idx]).permute(0, 3, 1, 2).float().div(255).to(dev)
            yb = torch.from_numpy(Htr[idx]).float().to(dev)
            mp = torch.from_numpy(Msg_tr[idx]).float().to(dev) * 2.0 - 1.0
            eb = torch.from_numpy(Extr_tr[idx]).to(dev)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _, _ = train_v4.run_forward(method, model, xb, mp, eb)
            else:
                lg, _, _ = train_v4.run_forward(method, model, xb, mp, eb)
            loss = loss_hole(lg.float(), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            nb += 1

        if (ep + 1) % 2 == 0 or ep == epochs - 1:
            vp = predict_fn(method, model, Xva, Msg_va, Exva_va)
            s = score([(p > 0.5).astype(np.uint8) for p in vp], Hva)
            print(f"    [{label}] ep {ep + 1:2d}/{epochs} loss={tot_loss / nb:.4f} val_hole_mIoU@0.5={s:.4f}", flush=True)
            if s > best_s:
                best_s = s
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return best_s, best_state


def main():
    import train_v4
    import train_v5
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--n_synth", type=int, default=1400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ep1", type=int, default=8)
    ap.add_argument("--ep2", type=int, default=8)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--tag", default="hole")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_v4.dev = args.device
    train_v5.dev = args.device
    global dev
    dev = args.device
    if args.smoke:
        args.n_synth, args.folds, args.ep1, args.ep2, args.bs = 4, 2, 1, 1, 4

    def predict_tta(method, model, imgs, msgs, extra, bs=16):
        p = predict_all(method, model, imgs, msgs, extra, bs=bs)
        if method in ("pixelseal", "trustmark"):
            pf = predict_all(method, model, imgs[:, :, ::-1].copy(), msgs, extra[:, :, :, ::-1].copy(), bs=bs)
            p = 0.5 * (p + pf[:, :, ::-1])
        return p

    print("Device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "CPU", "| args:", vars(args), flush=True)
    t0 = time.time()

    wm = load_watermark(args.method, models_dir="models", device=dev)
    covers, cover_masks, all_packed = load_coco_tar_and_masks(
        "clean_covers/val2017.tar", "clean_covers/coco_masks_4k.npz",
        max_covers=30 if args.smoke else 2000, seed=args.seed,
    )
    Xs, Hs, Msgs = generate_hole_synth(wm, args.method, covers, cover_masks, all_packed, n_samples=args.n_synth, seed=args.seed)
    print(f"[{args.method}] generated {len(Xs)} hole-only synthetic images ({time.time() - t0:.1f}s)", flush=True)

    ds_tr = LocalisationDataset("release", "train", methods=[args.method])
    ds_te = LocalisationDataset("release", "test", methods=[args.method])
    n_tr = 16 if args.smoke else len(ds_tr)
    n_te = 4 if args.smoke else len(ds_te)
    Xr = np.stack([ds_tr[i]["image"] for i in range(n_tr)])
    Mr = np.stack([ds_tr[i]["mask128"] for i in range(n_tr)])
    Hr = (1 - Mr).astype(np.uint8)
    fw_idx = np.array([i for i in range(n_tr) if ds_tr[i]["full_watermark"]], dtype=np.int32)
    Msgr = np.stack([ds_tr[i]["message"] for i in range(n_tr)])
    Xte = np.stack([ds_te[i]["image"] for i in range(n_te)])
    Msgte = np.stack([ds_te[i]["message"] for i in range(n_te)])
    names_te = [ds_te[i]["image_name"] for i in range(n_te)]

    if args.method == "trustmark":
        Exs = np.zeros((len(Xs), 1, 128, 128), np.float32)
        Exr = np.zeros((len(Xr), 1, 128, 128), np.float32)
        Exte = np.zeros((len(Xte), 1, 128, 128), np.float32)
    else:
        ms_s = compute_ms_maps(wm, args.method, Xs, Msgs, device=dev, smoke=args.smoke)
        bb_cache = f"work_bassim/bbox_v4_{args.method}.npz"
        if not args.smoke and os.path.exists(bb_cache):
            zb = np.load(bb_cache)
            ms_r, ms_te = zb["tr_ms"].astype(np.float32), zb["te_ms"].astype(np.float32)
        else:
            ms_r = compute_ms_maps(wm, args.method, Xr, Msgr, device=dev, smoke=args.smoke)
            ms_te = compute_ms_maps(wm, args.method, Xte, Msgte, device=dev, smoke=args.smoke)
        if args.method == "stegastamp":
            ss_c = SSContrib(wm)
            c_s, c_r, c_te = ss_c.batch_maps(Xs, Msgs), ss_c.batch_maps(Xr, Msgr), ss_c.batch_maps(Xte, Msgte)
            mu, sd = float(c_s.mean()), float(c_s.std() + 1e-6)
            Exs = np.concatenate([(c_s - mu) / sd, ms_s], axis=1)
            Exr = np.concatenate([(c_r - mu) / sd, ms_r], axis=1)
            Exte = np.concatenate([(c_te - mu) / sd, ms_te], axis=1)
        else:
            Exs, Exr, Exte = ms_s, ms_r, ms_te

    rng = np.random.default_rng(args.seed)
    perm_fw = rng.permutation(fw_idx)
    oof_hole_all = np.zeros(Hr.shape, np.float32)
    test_hole_probs = np.zeros((len(Xte), 128, 128), np.float32)

    folds = np.array_split(perm_fw, args.folds)
    for k, va in enumerate(folds):
        tr = np.setdiff1d(perm_fw, va)
        t1 = time.time()
        print(f"  [fold {k}/{args.folds}] train_fw={len(tr)}, val_fw={len(va)}", flush=True)
        model, param_groups = build_model_v5(args.method, wm, no_aux=True)

        Xp1 = np.concatenate([Xs, np.repeat(Xr[tr], 6, axis=0)])
        Hp1 = np.concatenate([Hs, np.repeat(Hr[tr], 6, axis=0)])
        Mp1 = np.concatenate([Msgs, np.repeat(Msgr[tr], 6, axis=0)])
        Ep1 = np.concatenate([Exs, np.repeat(Exr[tr], 6, axis=0)])
        opt1 = torch.optim.AdamW(param_groups, weight_decay=2e-4)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, args.ep1 * ((len(Xp1) + args.bs - 1) // args.bs)), eta_min=1e-5)
        best_s, best_state = run_hole_phase(args.method, model, opt1, sched1, Xp1, Hp1, Mp1, Ep1,
                                            Xr[va], Hr[va], Msgr[va], Exr[va], args.ep1, args.bs, rng, -1.0, None, "P1", predict_tta)

        n_rep = max(2, len(tr) // 3)
        idx_s = rng.permutation(len(Xs))[:n_rep]
        Xp2 = np.concatenate([np.repeat(Xr[tr], 4, axis=0), Xs[idx_s]])
        Hp2 = np.concatenate([np.repeat(Hr[tr], 4, axis=0), Hs[idx_s]])
        Mp2 = np.concatenate([np.repeat(Msgr[tr], 4, axis=0), Msgs[idx_s]])
        Ep2 = np.concatenate([np.repeat(Exr[tr], 4, axis=0), Exs[idx_s]])
        model.load_state_dict({k_: v.to(dev) for k_, v in best_state.items()})
        opt2 = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"] * 0.25} for g in param_groups], weight_decay=5e-4)
        bs2 = min(16, args.bs)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, args.ep2 * ((len(Xp2) + bs2 - 1) // bs2)), eta_min=1e-6)
        best_s, best_state = run_hole_phase(args.method, model, opt2, sched2, Xp2, Hp2, Mp2, Ep2,
                                            Xr[va], Hr[va], Msgr[va], Exr[va], args.ep2, bs2, rng, best_s, best_state, "P2", predict_tta)

        model.load_state_dict({k_: v.to(dev) for k_, v in best_state.items()})
        oof_hole_all[va] = predict_tta(args.method, model, Xr[va], Msgr[va], Exr[va])
        # Also predict on non-fw images using fold-0 model so oof_hole_all has valid values everywhere
        if k == 0:
            non_fw = np.setdiff1d(np.arange(len(Xr)), fw_idx)
            if len(non_fw):
                oof_hole_all[non_fw] = predict_tta(args.method, model, Xr[non_fw], Msgr[non_fw], Exr[non_fw])
        test_hole_probs += predict_tta(args.method, model, Xte, Msgte, Exte) / args.folds
        print(f"  [fold {k}/{args.folds}] best hole mIoU@0.5={best_s:.4f} ({time.time() - t1:.0f}s)", flush=True)

    s_fw = score([(p > 0.5).astype(np.uint8) for p in oof_hole_all[fw_idx]], Hr[fw_idx])
    print(f"[{args.method}] HOLE specialist OOF on fw=True (n={len(fw_idx)}): {s_fw:.4f} | total {time.time() - t0:.0f}s", flush=True)

    tag = f"{args.tag}_smoke_" if args.smoke else f"{args.tag}_"
    np.savez_compressed(f"work_bassim/oof_{tag}{args.method}.npz", hole_probs=oof_hole_all.astype(np.float16), masks=Mr)
    np.savez_compressed(f"work_bassim/test_{tag}{args.method}.npz", hole_probs=test_hole_probs.astype(np.float16), names=np.array(names_te))


if __name__ == "__main__":
    main()
