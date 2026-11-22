"""Phase 7: Rect Specialist (predicts the rectangular watermark mask on partial images).

Complements decoder-based rect_search_v2: covers non-decodable tiny rects (e.g. pixelseal
rects where GT-box decode acc ~0.50). Trained on 100% rect-only zero-seam synthetic
(small_prob=0.70) + real rect_part images (fill>0.95, area<=0.55; ~43/method, 5-fold CV).
Blend gated on full_acc < 0.72, strictly OOF-validated in postprocess_submit_v5.py.

    python work_bassim/train_rects.py --tag rect --method pixelseal
Saves work_bassim/oof_rect_{method}.npz and work_bassim/test_rect_{method}.npz.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ms_maps import compute_ms_maps  # noqa: E402
from synth_v4 import _rand_rect_u_shape, load_coco_tar_and_masks  # noqa: E402
from trace_core import hf_energy_map  # noqa: E402
from train_v4 import SSContrib, loss_v4, predict_all, run_forward  # noqa: E402
from train_v5 import build_model_v5  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import downsample_mask, score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


def generate_rect_synth(wm, method: str, covers, n_samples: int = 1400, seed: int = 31):
    rng = np.random.default_rng(seed)
    max_ar = 2.0 if method == "trustmark" else 16.0
    n_cov = len(covers)
    imgs = np.empty((n_samples, 512, 512, 3), np.uint8)
    masks128 = np.empty((n_samples, 128, 128), np.uint8)
    msgs = np.empty((n_samples, wm.nbits), np.uint8)

    for i in range(n_samples):
        cov = covers[int(rng.integers(0, n_cov))]
        bits = rng.integers(0, 2, wm.nbits, dtype=np.uint8)
        y0, y1, x0, x1 = _rand_rect_u_shape(rng, max_ar, small_prob=0.70)
        out = cov.copy()
        out[y0:y1, x0:x1] = wm.embed(cov[y0:y1, x0:x1], bits)
        if rng.random() < 0.25:
            q = int(rng.integers(80, 98))
            _, enc = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), q])
            out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        m512 = np.zeros((512, 512), np.uint8)
        m512[y0:y1, x0:x1] = 1
        imgs[i] = out
        masks128[i] = downsample_mask(m512)
        msgs[i] = bits

    return imgs, masks128, msgs


def run_rect_phase(method, model, opt, sched, Xtr, Mtr, Msg_tr, Extr_tr, Xva, Mva, Msg_va, Exva_va,
                   epochs, bs, rng, best_s, best_state, label, predict_fn):
    import train_v4
    if method in ("pixelseal", "trustmark"):
        flip_mask = rng.random(len(Xtr)) < 0.5
        if flip_mask.any():
            Xtr = Xtr.copy(); Mtr = Mtr.copy(); Extr_tr = Extr_tr.copy()
            Xtr[flip_mask] = Xtr[flip_mask][:, :, ::-1]
            Mtr[flip_mask] = Mtr[flip_mask][:, :, ::-1]
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
            yb = torch.from_numpy(Mtr[idx]).float().to(dev)
            mp = torch.from_numpy(Msg_tr[idx]).float().to(dev) * 2.0 - 1.0
            eb = torch.from_numpy(Extr_tr[idx]).to(dev)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _, _ = train_v4.run_forward(method, model, xb, mp, eb)
            else:
                lg, _, _ = train_v4.run_forward(method, model, xb, mp, eb)
            loss = loss_v4(lg.float(), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            nb += 1

        if (ep + 1) % 2 == 0 or ep == epochs - 1:
            vp = predict_fn(method, model, Xva, Msg_va, Exva_va)
            s = score([(p > 0.5).astype(np.uint8) for p in vp], Mva)
            print(f"    [{label}] ep {ep + 1:2d}/{epochs} loss={tot_loss / nb:.4f} val_mIoU@0.5={s:.4f}", flush=True)
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
    ap.add_argument("--tag", default="rect")
    ap.add_argument("--hf_residue", action="store_true",
                    help="stegastamp only: append HF-residue energy channel (6ch model)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_v4.dev = args.device
    train_v5.dev = args.device
    global dev
    dev = args.device
    if args.smoke:
        args.n_synth, args.folds, args.ep1, args.ep2, args.bs = 4, 2, 1, 1, 4

    use_hf = bool(args.hf_residue and args.method == "stegastamp" and not args.smoke)
    if args.hf_residue and not use_hf:
        print(f"[rect] --hf_residue ignored (method={args.method}, smoke={args.smoke})", flush=True)

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
    Xs, Ms, Msgs = generate_rect_synth(wm, args.method, covers, n_samples=args.n_synth, seed=args.seed)
    print(f"[{args.method}] generated {len(Xs)} rect-only synthetic images ({time.time() - t0:.1f}s)", flush=True)

    ds_tr = LocalisationDataset("release", "train", methods=[args.method])
    ds_te = LocalisationDataset("release", "test", methods=[args.method])
    n_tr = 16 if args.smoke else len(ds_tr)
    n_te = 4 if args.smoke else len(ds_te)
    Xr = np.stack([ds_tr[i]["image"] for i in range(n_tr)])
    Mr = np.stack([ds_tr[i]["mask128"] for i in range(n_tr)])
    Msgr = np.stack([ds_tr[i]["message"] for i in range(n_tr)])
    Xte = np.stack([ds_te[i]["image"] for i in range(n_te)])
    Msgte = np.stack([ds_te[i]["message"] for i in range(n_te)])
    names_te = [ds_te[i]["image_name"] for i in range(n_te)]

    area_r = Mr.mean((1, 2))
    fills = np.array([
        float(g.sum()) / ((np.where(g > 0)[0].ptp() + 1) * (np.where(g > 0)[1].ptp() + 1)) if g.any() else 1.0
        for g in Mr
    ])
    rect_idx = np.array([i for i in range(n_tr) if area_r[i] <= 0.55 and fills[i] > 0.95], dtype=np.int32)
    print(f"[{args.method}] real rect_part images: {len(rect_idx)}/{n_tr}", flush=True)

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
            if use_hf:
                print("[stegastamp] appending HF-residue channel (6ch model)", flush=True)
                Exs = np.concatenate([Exs, hf_energy_map(Xs)], axis=1)
                Exr = np.concatenate([Exr, hf_energy_map(Xr)], axis=1)
                Exte = np.concatenate([Exte, hf_energy_map(Xte)], axis=1)
        else:
            Exs, Exr, Exte = ms_s, ms_r, ms_te

    rng = np.random.default_rng(args.seed)
    perm_rect = rng.permutation(rect_idx)
    oof_rect_all = np.zeros(Mr.shape, np.float32)
    test_rect_probs = np.zeros((len(Xte), 128, 128), np.float32)

    folds = np.array_split(perm_rect, args.folds)
    for k, va in enumerate(folds):
        if len(va) == 0:
            continue
        tr = np.setdiff1d(perm_rect, va)
        t1 = time.time()
        print(f"  [fold {k}/{args.folds}] train_rect={len(tr)}, val_rect={len(va)}", flush=True)
        model, param_groups = build_model_v5(args.method, wm, no_aux=True,
                                               extra_ch=6 if use_hf else 5)

        Xp1 = np.concatenate([Xs, np.repeat(Xr[tr], 6, axis=0)])
        Mp1 = np.concatenate([Ms, np.repeat(Mr[tr], 6, axis=0)])
        Msgp1 = np.concatenate([Msgs, np.repeat(Msgr[tr], 6, axis=0)])
        Ep1 = np.concatenate([Exs, np.repeat(Exr[tr], 6, axis=0)])
        opt1 = torch.optim.AdamW(param_groups, weight_decay=2e-4)
        sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, args.ep1 * ((len(Xp1) + args.bs - 1) // args.bs)), eta_min=1e-5)
        best_s, best_state = run_rect_phase(args.method, model, opt1, sched1, Xp1, Mp1, Msgp1, Ep1,
                                           Xr[va], Mr[va], Msgr[va], Exr[va], args.ep1, args.bs, rng, -1.0, None, "P1", predict_tta)

        n_rep = max(2, len(tr) // 3)
        idx_s = rng.permutation(len(Xs))[:n_rep]
        Xp2 = np.concatenate([np.repeat(Xr[tr], 4, axis=0), Xs[idx_s]])
        Mp2 = np.concatenate([np.repeat(Mr[tr], 4, axis=0), Ms[idx_s]])
        Msgp2 = np.concatenate([np.repeat(Msgr[tr], 4, axis=0), Msgs[idx_s]])
        Ep2 = np.concatenate([np.repeat(Exr[tr], 4, axis=0), Exs[idx_s]])
        model.load_state_dict({k_: v.to(dev) for k_, v in best_state.items()})
        opt2 = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"] * 0.25} for g in param_groups], weight_decay=5e-4)
        bs2 = min(16, args.bs)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, args.ep2 * ((len(Xp2) + bs2 - 1) // bs2)), eta_min=1e-6)
        best_s, best_state = run_rect_phase(args.method, model, opt2, sched2, Xp2, Mp2, Msgp2, Ep2,
                                            Xr[va], Mr[va], Msgr[va], Exr[va], args.ep2, bs2, rng, best_s, best_state, "P2", predict_tta)

        model.load_state_dict({k_: v.to(dev) for k_, v in best_state.items()})
        oof_rect_all[va] = predict_tta(args.method, model, Xr[va], Msgr[va], Exr[va])
        if k == 0:
            non_rect = np.setdiff1d(np.arange(len(Xr)), rect_idx)
            if len(non_rect):
                oof_rect_all[non_rect] = predict_tta(args.method, model, Xr[non_rect], Msgr[non_rect], Exr[non_rect])
        test_rect_probs += predict_tta(args.method, model, Xte, Msgte, Exte) / args.folds
        print(f"  [fold {k}/{args.folds}] best rect mIoU@0.5={best_s:.4f} ({time.time() - t1:.0f}s)", flush=True)

    s_rect = score([(p > 0.5).astype(np.uint8) for p in oof_rect_all[rect_idx]], Mr[rect_idx])
    print(f"[{args.method}] RECT specialist OOF on rect_part (n={len(rect_idx)}): {s_rect:.4f} | total {time.time() - t0:.0f}s", flush=True)

    tag = f"{args.tag}_smoke_" if args.smoke else f"{args.tag}_"
    np.savez_compressed(f"work_bassim/oof_{tag}{args.method}.npz", probs=oof_rect_all.astype(np.float16), masks=Mr)
    np.savez_compressed(f"work_bassim/test_{tag}{args.method}.npz", probs=test_rect_probs.astype(np.float16), names=np.array(names_te))


if __name__ == "__main__":
    main()
