"""V2 method-specific watermark localisation models trained on synthetic + real data (3-fold CV).

- PixelSeal: fine-tunes PixelSeal's ConvNeXtV2 multi-scale pyramid (96@128, 192@64, 384@32, 768@16)
  + per-position message correlation + SRM high-pass residual -> 128x128 FPN/U-Net head.
- TrustMark: fine-tunes TrustMark's ResNet-50 multi-scale pyramid (256@128, 512@64, 1024@32, 2048@16)
  + per-position message correlation + SRM high-pass residual -> 128x128 FPN/U-Net head.
- StegaStamp: deeper SRM + Residual U-Net (width=48) + 13x13 dec_conv_7 message contribution.

Loss: BCE + per-image soft mIoU loss (directly matches the competition metric).
Saves work_bassim/oof_v2_<method>.npz and work_bassim/test_v2_<method>.npz (3-fold ensemble).
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from synth_v2 import generate_synthetic, load_clean_covers  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


def soft_miou_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit)
    inter1 = (p * y).sum((1, 2))
    union1 = (p + y - p * y).sum((1, 2))
    iou1 = (inter1 + 1.0) / (union1 + 1.0)
    inter0 = ((1.0 - p) * (1.0 - y)).sum((1, 2))
    union0 = ((1.0 - p) + (1.0 - y) - (1.0 - p) * (1.0 - y)).sum((1, 2))
    iou0 = (inter0 + 1.0) / (union0 + 1.0)
    return (1.0 - 0.5 * (iou1 + iou0)).mean()


def srm_highpass(x: torch.Tensor) -> torch.Tensor:
    """Multi-scale forensic high-pass residuals: 3x3, 5x5, and raw RGB."""
    hp3 = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)) * 10.0
    hp5 = (x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)) * 10.0
    return torch.cat([x * 2.0 - 1.0, hp3, hp5], dim=1)  # 9 ch


def conv_bn_act(ci: int, co: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.GELU(),
    )


class FPNHead(nn.Module):
    """Multi-scale FPN fusing [128, 64, 32, 16] backbone features + message map + high-pass stem -> 128x128."""

    def __init__(self, dims: tuple[int, int, int, int], msg_ch: int, fp_dim: int = 96):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(d, fp_dim, 1, bias=False) for d in dims])
        self.smooth = nn.ModuleList([conv_bn_act(fp_dim, fp_dim) for _ in range(3)])
        self.stem = nn.Sequential(conv_bn_act(9, 32, stride=2), conv_bn_act(32, 48, stride=2))  # 512 -> 128
        self.msg_proj = conv_bn_act(msg_ch, 32)
        self.head = nn.Sequential(
            conv_bn_act(fp_dim + 48 + 32, fp_dim),
            conv_bn_act(fp_dim, fp_dim // 2),
            nn.Conv2d(fp_dim // 2, 1, 1),
        )

    def forward(self, feats: list[torch.Tensor], x: torch.Tensor, msg_maps: torch.Tensor) -> torch.Tensor:
        # feats: [f128, f64, f32, f16]
        p = [lat(f) for lat, f in zip(self.lat, feats)]
        for i in (2, 1, 0):
            p[i] = self.smooth[i](p[i] + F.interpolate(p[i + 1], size=p[i].shape[-2:], mode="bilinear", align_corners=False))
        s = self.stem(srm_highpass(x))
        m = self.msg_proj(msg_maps)
        return self.head(torch.cat([p[0], s, m], dim=1))[:, 0]


class PixelSealSeg(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det  # ConvNeXtV2 + PixelDecoder (fine-tuned)
        self.fpn = FPNHead((96, 192, 384, 768), msg_ch=3, fp_dim=96)

    def forward(self, x: torch.Tensor, msg_pm1: torch.Tensor) -> torch.Tensor:
        # msg_pm1: (B, 256) in {-1, +1}
        xr = self.det.preprocess(x)
        cn = self.det.convnext
        feats = []
        t = xr
        for i in range(4):
            t = cn.downsample_layers[i](t)
            t = cn.stages[i](t)
            feats.append(t)
        emb = self.det.pixel_decoder.output_upscaling(feats[-1])
        lin = self.det.pixel_decoder.linear
        pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]  # B x 257 x 16 x 16
        det0 = pp[:, :1]
        bits = pp[:, 1:]
        dot = (bits * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree = (torch.tanh(bits) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        msg_128 = F.interpolate(torch.cat([det0, dot, agree], 1), size=(128, 128), mode="bilinear", align_corners=False)
        return self.fpn(feats, x, msg_128)


class TrustMarkSeg(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net  # ResNet-50 (fine-tuned)
        self.fpn = FPNHead((256, 512, 1024, 2048), msg_ch=4, fp_dim=96)

    def forward(self, x: torch.Tensor, msg_pm1: torch.Tensor) -> torch.Tensor:
        n = self.net
        t = n.maxpool(n.relu(n.bn1(n.conv1(x * 2.0 - 1.0))))
        feats = []
        for k in ("layer1", "layer2", "layer3", "layer4"):
            t = getattr(n, k)(t)
            feats.append(t)
        pp16 = torch.einsum("bchw,oc->bohw", feats[-1], n.fc.weight) + n.fc.bias[None, :, None, None]
        dot16 = (pp16 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree16 = (torch.tanh(pp16) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        # Also include native 245x245 -> 8x8 correlation
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
        return self.fpn(feats, x, m128)


class ResBlock(nn.Module):
    def __init__(self, ci: int, co: int, stride: int = 1):
        super().__init__()
        self.c1 = conv_bn_act(ci, co, stride)
        self.c2 = nn.Sequential(nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co))
        self.skip = nn.Identity() if (ci == co and stride == 1) else nn.Sequential(
            nn.Conv2d(ci, co, 1, stride=stride, bias=False), nn.BatchNorm2d(co)
        )

    def forward(self, x):
        return F.gelu(self.c2(self.c1(x)) + self.skip(x))


class StegaStampSeg(nn.Module):
    """Deep residual U-Net (512 -> 128) on multi-scale forensic high-pass + 13x13 contrib map."""

    def __init__(self, c: int = 48):
        super().__init__()
        self.stem = nn.Sequential(ResBlock(9, c), ResBlock(c, c))           # 512
        self.e1 = nn.Sequential(ResBlock(c, 2 * c, 2), ResBlock(2 * c, 2 * c))     # 256
        self.e2 = nn.Sequential(ResBlock(2 * c, 4 * c, 2), ResBlock(4 * c, 4 * c)) # 128
        self.e3 = nn.Sequential(ResBlock(4 * c, 6 * c, 2), ResBlock(6 * c, 6 * c)) # 64
        self.e4 = nn.Sequential(ResBlock(6 * c, 8 * c, 2), ResBlock(8 * c, 8 * c)) # 32
        self.e5 = nn.Sequential(ResBlock(8 * c, 8 * c, 2), ResBlock(8 * c, 8 * c)) # 16
        self.d4 = ResBlock(16 * c, 8 * c)
        self.d3 = ResBlock(14 * c, 6 * c)
        self.fe = conv_bn_act(1, 16)
        self.d2 = nn.Sequential(ResBlock(10 * c + 16, 4 * c), ResBlock(4 * c, 2 * c))
        self.out = nn.Conv2d(2 * c, 1, 1)

    def forward(self, x: torch.Tensor, contrib128: torch.Tensor) -> torch.Tensor:
        s = self.stem(srm_highpass(x))
        e1 = self.e1(s)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d4 = self.d4(torch.cat([up(e5, e4), e4], 1))
        d3 = self.d3(torch.cat([up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2, self.fe(contrib128)], 1))
        return self.out(d2)[:, 0]


class SSContrib:
    def __init__(self, wm):
        import onnx
        import onnxruntime as ort
        self.wm = wm
        m = onnx.load("models/stegastamp/stega_stamp_logits.onnx")
        init = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
        self.W1 = init["gen_decoder/dec_dense/MatMul/ReadVariableOp__75"]
        self.W2 = init["gen_decoder/decoder_out/MatMul/ReadVariableOp__77"]
        for name in ("gen_decoder/dec_conv_7/BiasAdd__443:0", "gen_decoder/dec_dense/BiasAdd:0"):
            m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        self.sess = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def batch_maps(self, imgs_uint8: np.ndarray, msgs: np.ndarray, bs: int = 32) -> np.ndarray:
        out = []
        for k in range(0, len(imgs_uint8), bs):
            x = torch.from_numpy(imgs_uint8[k:k + bs]).permute(0, 3, 1, 2).float().div(255)
            x_nhwc = self.wm._prep(x)
            _, c7, dense = self.sess.run(
                ["gen_decoder/decoder_out/BiasAdd:0", "gen_decoder/dec_conv_7/BiasAdd__443:0", "gen_decoder/dec_dense/BiasAdd:0"],
                {"image": x_nhwc, "secret": np.zeros((len(x_nhwc), 100), np.float32)},
            )
            for j in range(len(x_nhwc)):
                msg = msgs[k + j].astype(np.float32) * 2 - 1
                g = self.W1 @ ((self.W2 @ msg) * (dense[j] > 0))
                contrib = (c7[j] * g.reshape(13, 13, 128)).sum(-1)
                up = F.interpolate(torch.from_numpy(contrib)[None, None].float(), (128, 128), mode="bilinear", align_corners=False)[0, 0].numpy()
                out.append(up)
        maps = np.stack(out)[:, None].astype(np.float32)
        return maps


def build_model(method: str, base_wm):
    if method == "pixelseal":
        import copy
        model = PixelSealSeg(copy.deepcopy(base_wm.det)).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        params = [
            {"params": model.det.parameters(), "lr": 1e-4},
            {"params": model.fpn.parameters(), "lr": 1e-3},
        ]
        return model, params
    if method == "trustmark":
        import copy
        model = TrustMarkSeg(copy.deepcopy(base_wm.net)).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        params = [
            {"params": model.net.parameters(), "lr": 1e-4},
            {"params": model.fpn.parameters(), "lr": 1e-3},
        ]
        return model, params
    model = StegaStampSeg(48).to(dev)
    params = [{"params": model.parameters(), "lr": 1.5e-3}]
    return model, params


def run_forward(method: str, model, xb, msg_pm1, extra_b):
    if method == "stegastamp":
        return model(xb, extra_b)
    return model(xb, msg_pm1)


@torch.no_grad()
def predict_all(method: str, model, imgs: np.ndarray, msgs: np.ndarray, extra: np.ndarray | None, bs: int = 16) -> np.ndarray:
    model.eval()
    out = []
    for k in range(0, len(imgs), bs):
        xb = torch.from_numpy(imgs[k:k + bs]).permute(0, 3, 1, 2).float().div(255).to(dev)
        mp = (torch.from_numpy(msgs[k:k + bs]).float().to(dev) * 2.0 - 1.0)
        eb = torch.from_numpy(extra[k:k + bs]).to(dev) if extra is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = run_forward(method, model, xb, mp, eb)
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(method, base_wm, Xtr, Mtr, Msg_tr, Extr_tr, Xva, Mva, Msg_va, Exva_va, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, param_groups = build_model(method, base_wm)
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    n = len(Xtr)
    steps = args.epochs * ((n + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-5)

    best_s, best_state = -1.0, None
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(n)
        tot_loss, nb = 0.0, 0
        for k in range(0, n, args.bs):
            idx = order[k:k + args.bs]
            xb = torch.from_numpy(Xtr[idx]).permute(0, 3, 1, 2).float().div(255).to(dev)
            yb = torch.from_numpy(Mtr[idx]).float().to(dev)
            mp = torch.from_numpy(Msg_tr[idx]).float().to(dev) * 2.0 - 1.0
            eb = torch.from_numpy(Extr_tr[idx]).to(dev) if Extr_tr is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = run_forward(method, model, xb, mp, eb)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, yb) + soft_miou_loss(lg, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            nb += 1

        if (ep + 1) % 3 == 0 or ep == args.epochs - 1:
            vp = predict_all(method, model, Xva, Msg_va, Exva_va)
            s = score([(p > 0.5).astype(np.uint8) for p in vp], Mva)
            print(f"    ep {ep + 1:2d}/{args.epochs} loss={tot_loss / nb:.4f} val_mIoU@0.5={s:.4f}", flush=True)
            if s > best_s:
                best_s = s
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    return model, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--n_synth", type=int, default=1200)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--real_repeat", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_synth, args.folds, args.epochs = 16, 2, 2

    print("GPU:", torch.cuda.get_device_name(0), "| args:", vars(args), flush=True)
    t0 = time.time()

    wm = load_watermark(args.method, models_dir="models", device=dev)
    covers = load_clean_covers("clean_covers/val2017", limit=1500 if not args.smoke else 30, seed=args.seed)
    print(f"[{args.method}] loaded {len(covers)} clean covers ({time.time() - t0:.1f}s)", flush=True)

    Xs, Ms, Msgs = generate_synthetic(wm, args.method, covers, n_samples=args.n_synth, seed=args.seed)
    print(f"[{args.method}] generated {len(Xs)} synthetic images ({time.time() - t0:.1f}s)", flush=True)

    ds_tr = LocalisationDataset("release", "train", methods=[args.method])
    ds_te = LocalisationDataset("release", "test", methods=[args.method])
    n_tr = 20 if args.smoke else len(ds_tr)
    n_te = 8 if args.smoke else len(ds_te)
    Xr = np.stack([ds_tr[i]["image"] for i in range(n_tr)])
    Mr = np.stack([ds_tr[i]["mask128"] for i in range(n_tr)])
    Msgr = np.stack([ds_tr[i]["message"] for i in range(n_tr)])
    Xte = np.stack([ds_te[i]["image"] for i in range(n_te)])
    Msgte = np.stack([ds_te[i]["message"] for i in range(n_te)])
    names_te = [ds_te[i]["image_name"] for i in range(n_te)]

    Exs = Exr = Exte = None
    if args.method == "stegastamp":
        ss_c = SSContrib(wm)
        Exs = ss_c.batch_maps(Xs, Msgs)
        Exr = ss_c.batch_maps(Xr, Msgr)
        Exte = ss_c.batch_maps(Xte, Msgte)
        mu, sd = float(Exs.mean()), float(Exs.std() + 1e-6)
        Exs, Exr, Exte = (Exs - mu) / sd, (Exr - mu) / sd, (Exte - mu) / sd
        print(f"[{args.method}] extracted 13x13 contrib maps ({time.time() - t0:.1f}s)", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Xr))
    oof = np.zeros(Mr.shape, np.float32)
    test_probs = np.zeros((len(Xte), 128, 128), np.float32)

    for k, va in enumerate(np.array_split(idx, args.folds)):
        tr = np.setdiff1d(idx, va)
        Xtr = np.concatenate([Xs, np.repeat(Xr[tr], args.real_repeat, axis=0)])
        Mtr = np.concatenate([Ms, np.repeat(Mr[tr], args.real_repeat, axis=0)])
        Msg_tr = np.concatenate([Msgs, np.repeat(Msgr[tr], args.real_repeat, axis=0)])
        Extr_tr = np.concatenate([Exs, np.repeat(Exr[tr], args.real_repeat, axis=0)]) if Exs is not None else None
        Exva_va = Exr[va] if Exr is not None else None

        t1 = time.time()
        print(f"  [fold {k}] training on {len(Xtr)} imgs, val on {len(va)} real imgs...", flush=True)
        model, best_s = train_fold(args.method, wm, Xtr, Mtr, Msg_tr, Extr_tr, Xr[va], Mr[va], Msgr[va], Exva_va, args, args.seed + k)
        oof[va] = predict_all(args.method, model, Xr[va], Msgr[va], Exva_va)
        test_probs += predict_all(args.method, model, Xte, Msgte, Exte) / args.folds
        print(f"  [fold {k}] best val mIoU@0.5={best_s:.4f} ({time.time() - t1:.0f}s)", flush=True)
        tag = "smoke_" if args.smoke else ""
        # Save partial progress after every fold so a timeout never loses completed folds
        done_mask = np.zeros(len(Xr), dtype=bool)
        for j, fva in enumerate(np.array_split(idx, args.folds)):
            if j <= k:
                done_mask[fva] = True
        np.savez_compressed(f"work_bassim/partial_v2_{tag}{args.method}.npz",
                            oof=oof.astype(np.float16), test_sum=(test_probs * args.folds).astype(np.float16),
                            done_folds=k + 1, done_mask=done_mask, masks=Mr, names=np.array(names_te))

    s05 = score([(p > 0.5).astype(np.uint8) for p in oof], Mr)
    best_t, best_s = 0.5, s05
    for t in np.linspace(0.25, 0.75, 21):
        s = score([(p > t).astype(np.uint8) for p in oof], Mr)
        if s > best_s:
            best_t, best_s = float(t), s
    print(f"[{args.method}] OOF mIoU@0.5={s05:.4f} | best thr={best_t:.3f} -> {best_s:.4f} | total {time.time() - t0:.0f}s", flush=True)

    tag = "smoke_" if args.smoke else ""
    np.savez_compressed(f"work_bassim/oof_v2_{tag}{args.method}.npz", probs=oof.astype(np.float16), masks=Mr, idx=idx)
    np.savez_compressed(f"work_bassim/test_v2_{tag}{args.method}.npz", probs=test_probs.astype(np.float16),
                        names=np.array(names_te), thr=best_t, oof_miou=best_s)


if __name__ == "__main__":
    main()
