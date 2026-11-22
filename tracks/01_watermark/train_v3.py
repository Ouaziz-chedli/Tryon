"""V3 dual-branch localisation (3-fold CV + 3-fold test ensemble):
  Branch 1: Pretrained ImageNet ResNet-34 (captures edit boundaries, outpainting, copy-paste seams)
  Branch 2: Watermark-specific backbone + message correlation (fine-tuned PixelSeal ConvNeXtV2 /
            TrustMark ResNet-50 / StegaStamp SRM ResUNet + 13x13 contrib)
  Loss    : 0.5*BCE + 0.5*class_balanced_BCE + 1.5*soft_miou_loss(smooth=1e-3) for small targets.
"""
import argparse
import copy
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from synth_v3 import generate_synthetic_v3, load_coco_tar_and_masks  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


def loss_v3(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, y)
    # Per-image class-balanced BCE so tiny <5% objects or holes carry 50% gradient weight
    loss_el = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    pos_w = y.sum((1, 2), keepdim=True).clamp(min=1.0)
    neg_w = (1.0 - y).sum((1, 2), keepdim=True).clamp(min=1.0)
    bal_bce = 0.5 * ((loss_el * y / pos_w).sum((1, 2)) + (loss_el * (1.0 - y) / neg_w).sum((1, 2))).mean()

    p = torch.sigmoid(logit)
    sm = 1e-3
    inter1 = (p * y).sum((1, 2))
    union1 = (p + y - p * y).sum((1, 2))
    iou1 = (inter1 + sm) / (union1 + sm)
    inter0 = ((1.0 - p) * (1.0 - y)).sum((1, 2))
    union0 = ((1.0 - p) + (1.0 - y) - (1.0 - p) * (1.0 - y)).sum((1, 2))
    iou0 = (inter0 + sm) / (union0 + sm)
    miou_l = (1.0 - 0.5 * (iou1 + iou0)).mean()
    return 0.5 * bce + 0.5 * bal_bce + 1.5 * miou_l


def srm_highpass(x: torch.Tensor) -> torch.Tensor:
    hp3 = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)) * 10.0
    hp5 = (x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)) * 10.0
    return torch.cat([x * 2.0 - 1.0, hp3, hp5], dim=1)


def conv_bn_act(ci: int, co: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.GELU(),
    )


class DualFPNHead(nn.Module):
    """Fuses watermark pyramid [128,64,32,16] + ImageNet ResNet34 [128,64,32,16] + msg + SRM -> 128."""

    def __init__(self, wm_dims: tuple[int, int, int, int], msg_ch: int, fp_dim: int = 112):
        super().__init__()
        rgb_dims = (64, 128, 256, 512)
        self.rgb_enc = timm.create_model("resnet34", pretrained=True, features_only=True, out_indices=(1, 2, 3, 4))
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.lat = nn.ModuleList([nn.Conv2d(wd + rd, fp_dim, 1, bias=False) for wd, rd in zip(wm_dims, rgb_dims)])
        self.smooth = nn.ModuleList([conv_bn_act(fp_dim, fp_dim) for _ in range(3)])
        self.stem = nn.Sequential(conv_bn_act(9, 32, stride=2), conv_bn_act(32, 48, stride=2))  # 512 -> 128
        self.msg_proj = nn.Sequential(conv_bn_act(msg_ch, 32), conv_bn_act(32, 32))
        self.head = nn.Sequential(
            conv_bn_act(fp_dim + 48 + 32, fp_dim),
            conv_bn_act(fp_dim, fp_dim // 2),
            nn.Conv2d(fp_dim // 2, 1, 1),
        )

    def forward(self, wm_feats: list[torch.Tensor], x: torch.Tensor, msg_maps: torch.Tensor) -> torch.Tensor:
        rgb_feats = self.rgb_enc((x - self.rgb_mean) / self.rgb_std)  # [128, 64, 32, 16]
        p = [lat(torch.cat([wf, rf], dim=1)) for lat, wf, rf in zip(self.lat, wm_feats, rgb_feats)]
        for i in (2, 1, 0):
            p[i] = self.smooth[i](p[i] + F.interpolate(p[i + 1], size=p[i].shape[-2:], mode="bilinear", align_corners=False))
        return self.head(torch.cat([p[0], self.stem(srm_highpass(x)), self.msg_proj(msg_maps)], dim=1))[:, 0]


class PixelSealV3(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det
        # Add 32x32 intermediate spatial projection from stage 2 (384 -> 257) via learned adapter
        self.proj32 = nn.Conv2d(384, 256, 1)
        self.fpn = DualFPNHead((96, 192, 384, 768), msg_ch=5, fp_dim=112)

    def forward(self, x: torch.Tensor, msg_pm1: torch.Tensor) -> torch.Tensor:
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
        pp16 = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
        det0 = pp16[:, :1]
        bits16 = pp16[:, 1:]
        dot16 = (bits16 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree16 = (torch.tanh(bits16) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        # 32x32 fine correlation map
        b32 = self.proj32(feats[2])
        dot32 = (b32 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree32 = (torch.tanh(b32) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        msg_128 = torch.cat([
            F.interpolate(torch.cat([det0, dot16, agree16], 1), (128, 128), mode="bilinear", align_corners=False),
            F.interpolate(torch.cat([dot32, agree32], 1), (128, 128), mode="bilinear", align_corners=False),
        ], 1)
        return self.fpn(feats, x, msg_128)


class TrustMarkV3(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.proj32 = nn.Conv2d(1024, 100, 1)
        self.fpn = DualFPNHead((256, 512, 1024, 2048), msg_ch=6, fp_dim=112)

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

        x245 = F.interpolate(x, size=(245, 245), mode="bilinear", align_corners=False)
        t8 = n.maxpool(n.relu(n.bn1(n.conv1(x245 * 2.0 - 1.0))))
        for k in ("layer1", "layer2", "layer3", "layer4"):
            t8 = getattr(n, k)(t8)
        pp8 = torch.einsum("bchw,oc->bohw", t8, n.fc.weight) + n.fc.bias[None, :, None, None]
        dot8 = (pp8 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree8 = (torch.tanh(pp8) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)

        b32 = self.proj32(feats[2])
        dot32 = (b32 * msg_pm1[:, :, None, None]).mean(1, keepdim=True)
        agree32 = (torch.tanh(b32) * msg_pm1[:, :, None, None]).mean(1, keepdim=True)

        m128 = torch.cat([
            F.interpolate(torch.cat([dot16, agree16], 1), (128, 128), mode="bilinear", align_corners=False),
            F.interpolate(torch.cat([dot8, agree8], 1), (128, 128), mode="bilinear", align_corners=False),
            F.interpolate(torch.cat([dot32, agree32], 1), (128, 128), mode="bilinear", align_corners=False),
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


class StegaStampV3(nn.Module):
    """Deep SRM high-pass branch + ImageNet ResNet34 features + 13x13 contrib map."""

    def __init__(self, c: int = 48):
        super().__init__()
        self.stem = nn.Sequential(ResBlock(9, c), ResBlock(c, c))
        self.e1 = nn.Sequential(ResBlock(c, 2 * c, 2), ResBlock(2 * c, 2 * c))     # 256
        self.e2 = nn.Sequential(ResBlock(2 * c, 4 * c, 2), ResBlock(4 * c, 4 * c)) # 128
        self.e3 = nn.Sequential(ResBlock(4 * c, 6 * c, 2), ResBlock(6 * c, 6 * c)) # 64
        self.e4 = nn.Sequential(ResBlock(6 * c, 8 * c, 2), ResBlock(8 * c, 8 * c)) # 32
        self.e5 = nn.Sequential(ResBlock(8 * c, 8 * c, 2), ResBlock(8 * c, 8 * c)) # 16
        self.fpn = DualFPNHead((4 * c, 6 * c, 8 * c, 8 * c), msg_ch=1, fp_dim=112)

    def forward(self, x: torch.Tensor, contrib128: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(self.stem(srm_highpass(x)))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        return self.fpn([e2, e3, e4, e5], x, contrib128)


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
        prov = (["CUDAExecutionProvider", "CPUExecutionProvider"] if dev == "cuda" else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(m.SerializeToString(), providers=prov)

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
        return np.stack(out)[:, None].astype(np.float32)


def build_model(method: str, base_wm):
    if method == "pixelseal":
        model = PixelSealV3(copy.deepcopy(base_wm.det)).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        params = [
            {"params": model.det.parameters(), "lr": 1e-4},
            {"params": model.fpn.rgb_enc.parameters(), "lr": 1e-4},
            {"params": list(model.proj32.parameters()) + [p for n, p in model.fpn.named_parameters() if not n.startswith("rgb_enc.")], "lr": 1e-3},
        ]
        return model, params
    if method == "trustmark":
        model = TrustMarkV3(copy.deepcopy(base_wm.net)).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        params = [
            {"params": model.net.parameters(), "lr": 1e-4},
            {"params": model.fpn.rgb_enc.parameters(), "lr": 1e-4},
            {"params": list(model.proj32.parameters()) + [p for n, p in model.fpn.named_parameters() if not n.startswith("rgb_enc.")], "lr": 1e-3},
        ]
        return model, params
    model = StegaStampV3(48).to(dev)
    params = [
        {"params": model.fpn.rgb_enc.parameters(), "lr": 1.5e-4},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("fpn.rgb_enc.")], "lr": 1.5e-3},
    ]
    return model, params


def run_forward(method: str, model, xb, msg_pm1, extra_b):
    return model(xb, extra_b) if method == "stegastamp" else model(xb, msg_pm1)


@torch.no_grad()
def predict_all(method: str, model, imgs: np.ndarray, msgs: np.ndarray, extra: np.ndarray | None, bs: int = 16) -> np.ndarray:
    model.eval()
    out = []
    use_amp = dev == "cuda"
    for k in range(0, len(imgs), bs):
        xb = torch.from_numpy(imgs[k:k + bs]).permute(0, 3, 1, 2).float().div(255).to(dev)
        mp = torch.from_numpy(msgs[k:k + bs]).float().to(dev) * 2.0 - 1.0
        eb = torch.from_numpy(extra[k:k + bs]).to(dev) if extra is not None else None
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = run_forward(method, model, xb, mp, eb)
        else:
            lg = run_forward(method, model, xb, mp, eb)
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(method, base_wm, Xtr, Mtr, Msg_tr, Extr_tr, Xva, Mva, Msg_va, Exva_va, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, param_groups = build_model(method, base_wm)
    opt = torch.optim.AdamW(param_groups, weight_decay=2e-4)
    n = len(Xtr)
    steps = args.epochs * ((n + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=1e-5)

    best_s, best_state = -1.0, None
    use_amp = dev == "cuda"
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
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = run_forward(method, model, xb, mp, eb)
            else:
                lg = run_forward(method, model, xb, mp, eb)
            loss = loss_v3(lg.float(), yb)
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
    global dev
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--n_synth", type=int, default=1600)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--bs", type=int, default=20)
    ap.add_argument("--real_repeat", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    if args.smoke:
        args.n_synth, args.folds, args.epochs, args.bs, args.real_repeat = 6, 2, 1, 4, 1

    print("Device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "CPU", "| args:", vars(args), flush=True)
    t0 = time.time()

    wm = load_watermark(args.method, models_dir="models", device=dev)
    covers, cover_masks, all_packed = load_coco_tar_and_masks(
        "clean_covers/val2017.tar", "clean_covers/coco_masks_4k.npz",
        max_covers=40 if args.smoke else 2000, seed=args.seed,
    )
    print(f"[{args.method}] loaded {len(covers)} COCO covers+masks ({time.time() - t0:.1f}s)", flush=True)

    Xs, Ms, Msgs = generate_synthetic_v3(wm, args.method, covers, cover_masks, all_packed, n_samples=args.n_synth, seed=args.seed)
    print(f"[{args.method}] generated {len(Xs)} v3 synthetic images ({time.time() - t0:.1f}s)", flush=True)

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
    tag = "smoke_" if args.smoke else ""

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
        np.savez_compressed(f"work_bassim/partial_v3_{tag}{args.method}.npz",
                            oof=oof.astype(np.float16), test_sum=(test_probs * args.folds).astype(np.float16),
                            done_folds=k + 1, masks=Mr, names=np.array(names_te))

    s05 = score([(p > 0.5).astype(np.uint8) for p in oof], Mr)
    best_t, best_s = 0.5, s05
    for t in np.linspace(0.25, 0.75, 21):
        s = score([(p > t).astype(np.uint8) for p in oof], Mr)
        if s > best_s:
            best_t, best_s = float(t), s
    print(f"[{args.method}] V3 OOF mIoU@0.5={s05:.4f} | best thr={best_t:.3f} -> {best_s:.4f} | total {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(f"work_bassim/oof_v3_{tag}{args.method}.npz", probs=oof.astype(np.float16), masks=Mr, idx=idx)
    np.savez_compressed(f"work_bassim/test_v3_{tag}{args.method}.npz", probs=test_probs.astype(np.float16),
                        names=np.array(names_te), thr=best_t, oof_miou=best_s)


if __name__ == "__main__":
    main()
