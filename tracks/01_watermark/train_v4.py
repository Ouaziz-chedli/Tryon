"""V4 pure watermark-signal models + multi-scale sliding-crop maps + bbox refinement (5-fold CV, 2-phase):
  Phase 1: pre-train on 1800 zero-seam COCO-mask synthetic + real train split (4x)
  Phase 2: fine-tune at low LR primarily on real train images (+ 20% synthetic replay)
  Post   : batched GPU coordinate-descent bbox refinement saved alongside OOF/test probs.
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
from ms_maps import compute_ms_maps, refine_bboxes  # noqa: E402
from synth_v4 import generate_synthetic_v4, load_coco_tar_and_masks  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


def loss_v4(logit: torch.Tensor, y: torch.Tensor, pp16: torch.Tensor | None = None,
            det0_16: torch.Tensor | None = None, msg01: torch.Tensor | None = None) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, y)
    p = torch.sigmoid(logit)
    sm = 0.05
    inter1 = (p * y).sum((1, 2))
    union1 = (p + y - p * y).sum((1, 2))
    iou1 = (inter1 + sm) / (union1 + sm)
    inter0 = ((1.0 - p) * (1.0 - y)).sum((1, 2))
    union0 = ((1.0 - p) + (1.0 - y) - (1.0 - p) * (1.0 - y)).sum((1, 2))
    iou0 = (inter0 + sm) / (union0 + sm)
    miou_l = (1.0 - 0.5 * (iou1 + iou0)).mean()
    total = 0.7 * bce + 1.0 * miou_l

    if pp16 is not None and msg01 is not None:
        y16 = F.avg_pool2d(y[:, None], 8)[:, 0]
        wm_zone = (y16 > 0.5).float()[:, None]
        cl_zone = (y16 < 0.2).float()[:, None]
        target_bits = msg01[:, :, None, None].expand_as(pp16)
        bit_bce = F.binary_cross_entropy_with_logits(pp16, target_bits, reduction="none")
        pos_loss = (bit_bce * wm_zone).sum() / (wm_zone.sum() * pp16.shape[1] + 1.0)
        neg_loss = ((pp16 ** 2) * cl_zone).sum() / (cl_zone.sum() * pp16.shape[1] + 1.0)
        aux = pos_loss + 0.05 * neg_loss
        if det0_16 is not None:
            aux = aux + 0.5 * F.binary_cross_entropy_with_logits(det0_16, (y16 > 0.5).float())
        total = total + 0.15 * aux
    return total


def forensic_stem_feats(x: torch.Tensor) -> torch.Tensor:
    hp3 = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)) * 10.0
    hp5 = (x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)) * 10.0
    hp7 = (x - F.avg_pool2d(F.pad(x, (3, 3, 3, 3), mode="reflect"), 7, stride=1)) * 10.0
    dx = F.pad((x[:, :, :, 1:] - x[:, :, :, :-1]) * 10.0, (0, 1, 0, 0))
    dy = F.pad((x[:, :, 1:, :] - x[:, :, :-1, :]) * 10.0, (0, 0, 0, 1))
    return torch.cat([x * 2.0 - 1.0, hp3, hp5, hp7, dx, dy], dim=1)


def conv_bn_act(ci: int, co: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(co),
        nn.GELU(),
    )


class FPNHeadV4(nn.Module):
    def __init__(self, dims: tuple[int, int, int, int], msg_ch: int, fp_dim: int = 112):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(d, fp_dim, 1, bias=False) for d in dims])
        self.smooth = nn.ModuleList([conv_bn_act(fp_dim, fp_dim) for _ in range(3)])
        self.stem = nn.Sequential(conv_bn_act(18, 32, stride=2), conv_bn_act(32, 48, stride=2))
        self.msg_proj = nn.Sequential(conv_bn_act(msg_ch, 32), conv_bn_act(32, 32))
        self.head = nn.Sequential(
            conv_bn_act(fp_dim + 48 + 32, fp_dim),
            conv_bn_act(fp_dim, fp_dim // 2),
            nn.Conv2d(fp_dim // 2, 1, 1),
        )

    def forward(self, feats: list[torch.Tensor], x: torch.Tensor, msg_maps: torch.Tensor) -> torch.Tensor:
        p = [lat(f) for lat, f in zip(self.lat, feats)]
        for i in (2, 1, 0):
            p[i] = self.smooth[i](p[i] + F.interpolate(p[i + 1], size=p[i].shape[-2:], mode="bilinear", align_corners=False))
        return self.head(torch.cat([p[0], self.stem(forensic_stem_feats(x)), self.msg_proj(msg_maps)], dim=1))[:, 0]


class PixelSealV4(nn.Module):
    def __init__(self, det):
        super().__init__()
        self.det = det
        self.proj32 = nn.Conv2d(384, 256, 1)
        self.fpn = FPNHeadV4((96, 192, 384, 768), msg_ch=9, fp_dim=112)  # 5 + 4 ms_maps

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
        return self.fpn(feats, x, m128), bits16, det0


class TrustMarkV4(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.proj32 = nn.Conv2d(1024, 100, 1)
        self.fpn = FPNHeadV4((256, 512, 1024, 2048), msg_ch=10, fp_dim=112)  # 6 + 4 ms_maps

    def forward(self, x: torch.Tensor, msg_pm1: torch.Tensor, ms_maps: torch.Tensor):
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
            ms_maps,
        ], 1)
        return self.fpn(feats, x, m128), pp16, None


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


class StegaStampV4(nn.Module):
    def __init__(self, c: int = 48):
        super().__init__()
        self.stem = nn.Sequential(ResBlock(18, c), ResBlock(c, c))
        self.e1 = nn.Sequential(ResBlock(c, 2 * c, 2), ResBlock(2 * c, 2 * c))     # 256
        self.e2 = nn.Sequential(ResBlock(2 * c, 4 * c, 2), ResBlock(4 * c, 4 * c)) # 128
        self.e3 = nn.Sequential(ResBlock(4 * c, 6 * c, 2), ResBlock(6 * c, 6 * c)) # 64
        self.e4 = nn.Sequential(ResBlock(6 * c, 8 * c, 2), ResBlock(8 * c, 8 * c)) # 32
        self.e5 = nn.Sequential(ResBlock(8 * c, 8 * c, 2), ResBlock(8 * c, 8 * c)) # 16
        self.d4 = ResBlock(16 * c, 8 * c)
        self.d3 = ResBlock(14 * c, 6 * c)
        self.fe = conv_bn_act(5, 24)  # 1 contrib + 4 ms_maps
        self.d2 = nn.Sequential(ResBlock(10 * c + 24, 4 * c), ResBlock(4 * c, 2 * c))
        self.out = nn.Conv2d(2 * c, 1, 1)

    def forward(self, x: torch.Tensor, extra: torch.Tensor):
        e1 = self.e1(self.stem(forensic_stem_feats(x)))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d4 = self.d4(torch.cat([up(e5, e4), e4], 1))
        d3 = self.d3(torch.cat([up(d4, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2, self.fe(extra)], 1))
        return self.out(d2)[:, 0], None, None


class StegaStampV4C6(StegaStampV4):
    """StegaStampV4 with a 6-channel forensic map input (1 contrib + 4 ms + 1 HF).

    Used only under --hf_residue (train_rects). Default path keeps StegaStampV4.
    """

    def __init__(self, c: int = 48):
        super().__init__(c)
        self.fe = conv_bn_act(6, 24)

    def forward(self, x: torch.Tensor, extra: torch.Tensor):
        assert extra.shape[1] == 6, f"StegaStampV4C6 needs 6 extra channels, got {extra.shape[1]}"
        return super().forward(x, extra)


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
        model = PixelSealV4(copy.deepcopy(base_wm.det)).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        params = [
            {"params": model.det.parameters(), "lr": 1e-4},
            {"params": list(model.proj32.parameters()) + list(model.fpn.parameters()), "lr": 1e-3},
        ]
        return model, params
    if method == "trustmark":
        model = TrustMarkV4(copy.deepcopy(base_wm.net)).to(dev)
        for p in model.parameters():
            p.requires_grad_(True)
        params = [
            {"params": model.net.parameters(), "lr": 1e-4},
            {"params": list(model.proj32.parameters()) + list(model.fpn.parameters()), "lr": 1e-3},
        ]
        return model, params
    model = StegaStampV4(48).to(dev)
    params = [{"params": model.parameters(), "lr": 1.5e-3}]
    return model, params


def run_forward(method: str, model, xb, msg_pm1, extra_b):
    return model(xb, extra_b) if method == "stegastamp" else model(xb, msg_pm1, extra_b)


@torch.no_grad()
def predict_all(method: str, model, imgs: np.ndarray, msgs: np.ndarray, extra: np.ndarray, bs: int = 16) -> np.ndarray:
    model.eval()
    out = []
    use_amp = dev == "cuda"
    for k in range(0, len(imgs), bs):
        xb = torch.from_numpy(imgs[k:k + bs]).permute(0, 3, 1, 2).float().div(255).to(dev)
        mp = torch.from_numpy(msgs[k:k + bs]).float().to(dev) * 2.0 - 1.0
        eb = torch.from_numpy(extra[k:k + bs]).to(dev)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _, _ = run_forward(method, model, xb, mp, eb)
        else:
            lg, _, _ = run_forward(method, model, xb, mp, eb)
        out.append(torch.sigmoid(lg.float()).cpu().numpy())
    return np.concatenate(out)


def run_phase(method, model, opt, sched, Xtr, Mtr, Msg_tr, Extr_tr, Xva, Mva, Msg_va, Exva_va,
              epochs, bs, rng, best_s, best_state, label):
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
            m01 = torch.from_numpy(Msg_tr[idx]).float().to(dev)
            mp = m01 * 2.0 - 1.0
            eb = torch.from_numpy(Extr_tr[idx]).to(dev)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, pp16, det0 = run_forward(method, model, xb, mp, eb)
            else:
                lg, pp16, det0 = run_forward(method, model, xb, mp, eb)
            loss = loss_v4(lg.float(), yb, pp16.float() if pp16 is not None else None,
                           det0.float() if det0 is not None else None, m01)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot_loss += loss.item()
            nb += 1

        if (ep + 1) % 2 == 0 or ep == epochs - 1:
            vp = predict_all(method, model, Xva, Msg_va, Exva_va)
            s = score([(p > 0.5).astype(np.uint8) for p in vp], Mva)
            print(f"    [{label}] ep {ep + 1:2d}/{epochs} loss={tot_loss / nb:.4f} val_mIoU@0.5={s:.4f}", flush=True)
            if s > best_s:
                best_s = s
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    return best_s, best_state


def train_fold(method, base_wm, Xs, Ms, Msgs, Exs, Xr_tr, Mr_tr, Msgr_tr, Exr_tr, Xva, Mva, Msg_va, Exva_va, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model, param_groups = build_model(method, base_wm)

    Xp1 = np.concatenate([Xs, np.repeat(Xr_tr, 4, axis=0)])
    Mp1 = np.concatenate([Ms, np.repeat(Mr_tr, 4, axis=0)])
    Msg_p1 = np.concatenate([Msgs, np.repeat(Msgr_tr, 4, axis=0)])
    Exp1 = np.concatenate([Exs, np.repeat(Exr_tr, 4, axis=0)])

    opt1 = torch.optim.AdamW(param_groups, weight_decay=2e-4)
    steps1 = args.ep1 * ((len(Xp1) + args.bs - 1) // args.bs)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(1, steps1), eta_min=1e-5)
    best_s, best_state = run_phase(method, model, opt1, sched1, Xp1, Mp1, Msg_p1, Exp1,
                                   Xva, Mva, Msg_va, Exva_va, args.ep1, args.bs, rng, -1.0, None, "P1")

    n_rep = max(2, len(Xr_tr) // 4)
    idx_s = rng.permutation(len(Xs))[:n_rep]
    Xp2 = np.concatenate([np.repeat(Xr_tr, 3, axis=0), Xs[idx_s]])
    Mp2 = np.concatenate([np.repeat(Mr_tr, 3, axis=0), Ms[idx_s]])
    Msg_p2 = np.concatenate([np.repeat(Msgr_tr, 3, axis=0), Msgs[idx_s]])
    Exp2 = np.concatenate([np.repeat(Exr_tr, 3, axis=0), Exs[idx_s]])

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    low_groups = [{"params": g["params"], "lr": g["lr"] * 0.25} for g in param_groups]
    opt2 = torch.optim.AdamW(low_groups, weight_decay=5e-4)
    bs2 = min(16, args.bs)
    steps2 = args.ep2 * ((len(Xp2) + bs2 - 1) // bs2)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(1, steps2), eta_min=1e-6)
    best_s, best_state = run_phase(method, model, opt2, sched2, Xp2, Mp2, Msg_p2, Exp2,
                                   Xva, Mva, Msg_va, Exva_va, args.ep2, bs2, rng, best_s, best_state, "P2")

    model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    return model, best_s


def main():
    global dev
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--n_synth", type=int, default=1600)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--ep1", type=int, default=6)
    ap.add_argument("--ep2", type=int, default=5)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="v4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.device
    if args.smoke:
        args.n_synth, args.folds, args.ep1, args.ep2, args.bs = 4, 2, 1, 1, 4

    print("Device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "CPU", "| args:", vars(args), flush=True)
    t0 = time.time()

    wm = load_watermark(args.method, models_dir="models", device=dev)
    covers, cover_masks, all_packed = load_coco_tar_and_masks(
        "clean_covers/val2017.tar", "clean_covers/coco_masks_4k.npz",
        max_covers=30 if args.smoke else 2000, seed=args.seed,
    )
    Xs, Ms, Msgs = generate_synthetic_v4(wm, args.method, covers, cover_masks, all_packed, n_samples=args.n_synth, seed=args.seed)
    print(f"[{args.method}] generated {len(Xs)} zero-seam synthetic images ({time.time() - t0:.1f}s)", flush=True)

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

    t_ms = time.time()
    ms_s = compute_ms_maps(wm, args.method, Xs, Msgs, device=dev, smoke=args.smoke)
    bb_cache = f"work_bassim/bbox_v4_{args.method}.npz"
    if not args.smoke and os.path.exists(bb_cache):
        zb = np.load(bb_cache)
        ms_r = zb["tr_ms"].astype(np.float32)
        ms_te = zb["te_ms"].astype(np.float32)
        print(f"[{args.method}] reused cached train/test ms_maps + synth ms_maps in {time.time() - t_ms:.1f}s", flush=True)
    else:
        ms_r = compute_ms_maps(wm, args.method, Xr, Msgr, device=dev, smoke=args.smoke)
        ms_te = compute_ms_maps(wm, args.method, Xte, Msgte, device=dev, smoke=args.smoke)
        print(f"[{args.method}] computed multi-scale crop maps in {time.time() - t_ms:.1f}s", flush=True)

    if args.method == "stegastamp":
        ss_c = SSContrib(wm)
        c_s = ss_c.batch_maps(Xs, Msgs)
        c_r = ss_c.batch_maps(Xr, Msgr)
        c_te = ss_c.batch_maps(Xte, Msgte)
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
        print(f"  [fold {k}] train_real={len(tr)}, val_real={len(va)}, synth={len(Xs)}", flush=True)
        model, best_s = train_fold(
            args.method, wm, Xs, Ms, Msgs, Exs,
            Xr[tr], Mr[tr], Msgr[tr], Exr[tr],
            Xr[va], Mr[va], Msgr[va], Exr[va],
            args, args.seed + k,
        )
        oof[va] = predict_all(args.method, model, Xr[va], Msgr[va], Exr[va])
        test_probs += predict_all(args.method, model, Xte, Msgte, Exte) / args.folds
        print(f"  [fold {k}] best val mIoU@0.5={best_s:.4f} ({time.time() - t1:.0f}s)", flush=True)
        np.savez_compressed(f"work_bassim/partial_{tag}{args.method}.npz",
                            oof=oof.astype(np.float16), test_sum=(test_probs * args.folds).astype(np.float16),
                            done_folds=k + 1, masks=Mr, names=np.array(names_te))

    s05 = score([(p > 0.5).astype(np.uint8) for p in oof], Mr)
    best_t, best_s = 0.5, s05
    for t in np.linspace(0.25, 0.75, 21):
        s = score([(p > t).astype(np.uint8) for p in oof], Mr)
        if s > best_s:
            best_t, best_s = float(t), s
    print(f"[{args.method}] {args.tag.upper()} OOF mIoU@0.5={s05:.4f} | best thr={best_t:.3f} -> {best_s:.4f}", flush=True)

    if not args.smoke and os.path.exists(bb_cache):
        zb = np.load(bb_cache)
        oof_boxes, oof_bacc, oof_facc = zb["tr_boxes"], zb["tr_bacc"], zb["tr_facc"]
        te_boxes, te_bacc, te_facc = zb["te_boxes"], zb["te_bacc"], zb["te_facc"]
    else:
        t_b = time.time()
        oof_boxes, oof_bacc, oof_facc = refine_bboxes(wm, args.method, Xr, Msgr, ms_r, oof, device=dev, smoke=args.smoke)
        te_boxes, te_bacc, te_facc = refine_bboxes(wm, args.method, Xte, Msgte, ms_te, test_probs, device=dev, smoke=args.smoke)
        print(f"[{args.method}] bbox refinement done in {time.time() - t_b:.1f}s", flush=True)
    print(f"[{args.method}] total {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(f"work_bassim/oof_{tag}{args.method}.npz", probs=oof.astype(np.float16), masks=Mr, idx=idx,
                        boxes=oof_boxes, box_acc=oof_bacc, full_acc=oof_facc)
    np.savez_compressed(f"work_bassim/test_{tag}{args.method}.npz", probs=test_probs.astype(np.float16),
                        names=np.array(names_te), thr=best_t, oof_miou=best_s,
                        boxes=te_boxes, box_acc=te_bacc, full_acc=te_facc)


if __name__ == "__main__":
    main()
