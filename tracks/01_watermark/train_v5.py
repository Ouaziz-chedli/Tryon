"""Phase 3 (v5) method-optimized 7-fold CV training (using all 4 GPUs on JURECA):
  - PixelSeal : ConvNeXtV2 + 32x32 msg + ms_maps + fp_dim=128 (optionally --no_aux for diversity)
  - TrustMark : ResNet-50 + fp_dim=128 WITHOUT aux bit loss or ms_maps (which hurt TrustMark in v4/v4.1)
  - StegaStamp: ResUNet(48) + 18-ch forensic stem + 13x13 contrib + ms_maps
  - 7-fold CV (146 real train imgs/fold) + 2-phase (zero-seam COCO synth -> real fine-tune)
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from ms_maps import compute_ms_maps  # noqa: E402
from synth_v4 import generate_synthetic_v4, load_coco_tar_and_masks  # noqa: E402
from train_v4 import FPNHeadV4, SSContrib, StegaStampV4, StegaStampV4C6, loss_v4, run_phase  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


class PixelSealV5(nn.Module):
    def __init__(self, det, use_aux: bool = True):
        super().__init__()
        self.det = det
        self.use_aux = use_aux
        self.proj32 = nn.Conv2d(384, 256, 1)
        self.fpn = FPNHeadV4((96, 192, 384, 768), msg_ch=9, fp_dim=128)

    def forward(self, x: torch.Tensor, msg_pm1: torch.Tensor, ms_maps: torch.Tensor):
        xr = self.det.preprocess(x)
        cn = self.det.convnext
        feats, t = [], xr
        for i in range(4):
            t = cn.downsample_layers[i](t)
            t = cn.stages[i](t)
            feats.append(t)
        emb = self.det.pixel_decoder.output_upscaling(feats[-1])
        lin = self.det.pixel_decoder.linear
        pp16 = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
        det0 = pp16[:, 0]
        bits16 = pp16[:, 1:]
        dot16 = (bits16 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree16 = (torch.tanh(bits16) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        b32 = self.proj32(feats[2])
        dot32 = (b32 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree32 = (torch.tanh(b32) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        m128 = torch.cat([
            F.interpolate(torch.cat([det0[:, None], dot16, agree16], 1), (128, 128), mode="bilinear", align_corners=False),
            F.interpolate(torch.cat([dot32, agree32], 1), (128, 128), mode="bilinear", align_corners=False),
            ms_maps,
        ], 1)
        lg = self.fpn(feats, x, m128)
        return (lg, bits16, det0) if self.use_aux else (lg, None, None)


class TrustMarkV5(nn.Module):
    """Pure v2-style TrustMark (NO aux loss, NO ms_maps) with wider fp_dim=128."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.fpn = FPNHeadV4((256, 512, 1024, 2048), msg_ch=4, fp_dim=128)

    def forward(self, x: torch.Tensor, msg_pm1: torch.Tensor, _unused=None):
        n = self.net
        t = n.maxpool(n.relu(n.bn1(n.conv1(x * 2.0 - 1.0))))
        feats = []
        for k in ("layer1", "layer2", "layer3", "layer4"):
            t = getattr(n, k)(t)
            feats.append(t)
        pp16 = torch.einsum("bchw,oc->bohw", feats[-1], n.fc.weight) + n.fc.bias[None, :, None, None]
        dot16 = (pp16 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree16 = (torch.tanh(pp16) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)

        x245 = F.interpolate(x, size=(245, 245), mode="bilinear", align_corners=False)
        t8 = n.maxpool(n.relu(n.bn1(n.conv1(x245 * 2.0 - 1.0))))
        for k in ("layer1", "layer2", "layer3", "layer4"):
            t8 = getattr(n, k)(t8)
        pp8 = torch.einsum("bchw,oc->bohw", t8, n.fc.weight) + n.fc.bias[None, :, None, None]
        dot8 = (pp8 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree8 = (torch.tanh(pp8) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)

        m128 = torch.cat([
            F.interpolate(torch.cat([dot16, agree16], 1), (128, 128), mode="bilinear", align_corners=False),
            F.interpolate(torch.cat([dot8, agree8], 1), (128, 128), mode="bilinear", align_corners=False),
        ], 1)
        return self.fpn(feats, x, m128), None, None


def build_model_v5(method: str, base_wm, no_aux: bool = False, extra_ch: int = 5):
    if method == "pixelseal":
        model = PixelSealV5(copy.deepcopy(base_wm.det), use_aux=not no_aux).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        return model, [
            {"params": model.det.parameters(), "lr": 1e-4},
            {"params": list(model.proj32.parameters()) + list(model.fpn.parameters()), "lr": 1e-3},
        ]
    if method == "trustmark":
        model = TrustMarkV5(copy.deepcopy(base_wm.net)).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        return model, [
            {"params": model.net.parameters(), "lr": 1e-4},
            {"params": model.fpn.parameters(), "lr": 1e-3},
        ]
    model = (StegaStampV4C6 if extra_ch == 6 else StegaStampV4)(48).to(dev)
    return model, [{"params": model.parameters(), "lr": 1.5e-3}]


def main():
    import train_v4
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--n_synth", type=int, default=1800)
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--ep1", type=int, default=10)
    ap.add_argument("--ep2", type=int, default=8)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--no_aux", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_v4.dev = args.device
    global dev
    dev = args.device
    if args.smoke:
        args.n_synth, args.folds, args.ep1, args.ep2, args.bs = 4, 2, 1, 1, 4

    # Monkey-patch build_model + hflip aug/TTA (for pixelseal & trustmark, which are hflip-invariant)
    train_v4.build_model = lambda m, w: build_model_v5(m, w, no_aux=args.no_aux)
    orig_predict_all = train_v4.predict_all
    orig_run_phase = train_v4.run_phase

    def predict_tta(method, model, imgs, msgs, extra, bs=16):
        p = orig_predict_all(method, model, imgs, msgs, extra, bs=bs)
        if method in ("pixelseal", "trustmark"):
            pf = orig_predict_all(method, model, imgs[:, :, ::-1].copy(), msgs, extra[:, :, :, ::-1].copy(), bs=bs)
            p = 0.5 * (p + pf[:, :, ::-1])
        return p

    def run_phase_aug(method, model, opt, sched, Xtr, Mtr, Msg_tr, Extr_tr, Xva, Mva, Msg_va, Exva_va,
                      epochs, bs, rng, best_s, best_state, label):
        if method in ("pixelseal", "trustmark"):
            # Add hflip copies with probability 0.5 by duplicating or flipping a random half each run
            flip_mask = rng.random(len(Xtr)) < 0.5
            if flip_mask.any():
                Xtr = Xtr.copy(); Mtr = Mtr.copy(); Extr_tr = Extr_tr.copy()
                Xtr[flip_mask] = Xtr[flip_mask][:, :, ::-1]
                Mtr[flip_mask] = Mtr[flip_mask][:, :, ::-1]
                Extr_tr[flip_mask] = Extr_tr[flip_mask][:, :, :, ::-1]
        return orig_run_phase(method, model, opt, sched, Xtr, Mtr, Msg_tr, Extr_tr, Xva, Mva, Msg_va, Exva_va,
                              epochs, bs, rng, best_s, best_state, label)

    train_v4.predict_all = predict_tta
    train_v4.run_phase = run_phase_aug

    print("Device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "CPU", "| args:", vars(args), flush=True)
    t0 = time.time()

    wm = load_watermark(args.method, models_dir="models", device=dev)
    covers, cover_masks, all_packed = load_coco_tar_and_masks(
        "clean_covers/val2017.tar", "clean_covers/coco_masks_4k.npz",
        max_covers=30 if args.smoke else 2000, seed=args.seed,
    )
    Xs, Ms, Msgs = generate_synthetic_v4(wm, args.method, covers, cover_masks, all_packed, n_samples=args.n_synth, seed=args.seed)
    print(f"[{args.method}] generated {len(Xs)} synthetic images ({time.time() - t0:.1f}s)", flush=True)

    ds_tr = LocalisationDataset("release", "train", methods=[args.method])
    ds_te = LocalisationDataset("release", "test", methods=[args.method])
    n_tr = 8 if args.smoke else len(ds_tr)
    n_te = 4 if args.smoke else len(ds_te)
    Xr = np.stack([ds_tr[i]["image"] for i in range(n_tr)])
    Mr = np.stack([ds_tr[i]["mask128"] for i in range(n_tr)])
    Msgr = np.stack([ds_tr[i]["message"] for i in range(n_tr)])
    Xte = np.stack([ds_te[i]["image"] for i in range(n_te)])
    Msgte = np.stack([ds_te[i]["message"] for i in range(n_te)])
    names_te = [ds_te[i]["image_name"] for i in range(n_te)]

    if args.method == "trustmark":
        # Dummy 1-ch placeholder (unused by TrustMarkV5)
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
    idx = rng.permutation(len(Xr))
    oof = np.zeros(Mr.shape, np.float32)
    test_probs = np.zeros((len(Xte), 128, 128), np.float32)
    tag = f"{args.tag}_smoke_" if args.smoke else f"{args.tag}_"

    for k, va in enumerate(np.array_split(idx, args.folds)):
        tr = np.setdiff1d(idx, va)
        t1 = time.time()
        print(f"  [fold {k}/{args.folds}] train_real={len(tr)}, val_real={len(va)}", flush=True)
        model, best_s = train_v4.train_fold(
            args.method, wm, Xs, Ms, Msgs, Exs,
            Xr[tr], Mr[tr], Msgr[tr], Exr[tr],
            Xr[va], Mr[va], Msgr[va], Exr[va],
            args, args.seed + k,
        )
        oof[va] = train_v4.predict_all(args.method, model, Xr[va], Msgr[va], Exr[va])
        test_probs += train_v4.predict_all(args.method, model, Xte, Msgte, Exte) / args.folds
        print(f"  [fold {k}/{args.folds}] best val mIoU@0.5={best_s:.4f} ({time.time() - t1:.0f}s)", flush=True)
        np.savez_compressed(f"/tmp/partial_{tag}{args.method}.npz",
                            oof=oof.astype(np.float16), test_sum=(test_probs * args.folds).astype(np.float16),
                            done_folds=k + 1)

    s05 = score([(p > 0.5).astype(np.uint8) for p in oof], Mr)
    best_t, best_s = 0.5, s05
    for t in np.linspace(0.25, 0.75, 21):
        s = score([(p > t).astype(np.uint8) for p in oof], Mr)
        if s > best_s:
            best_t, best_s = float(t), s
    print(f"[{args.method}] {args.tag.upper()} OOF mIoU@0.5={s05:.4f} | best thr={best_t:.3f} -> {best_s:.4f} | total {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(f"work_bassim/oof_{tag}{args.method}.npz", probs=oof.astype(np.float16), masks=Mr, idx=idx)
    np.savez_compressed(f"work_bassim/test_{tag}{args.method}.npz", probs=test_probs.astype(np.float16),
                        names=np.array(names_te), thr=best_t, oof_miou=best_s)


if __name__ == "__main__":
    main()
