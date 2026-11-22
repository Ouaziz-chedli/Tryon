"""Architecture A: U-Net + message-correlation maps from the frozen watermark decoder, 5-fold CV by image.

For each image, the decoder of its method gives coarse maps of "how much does this region vote for the KNOWN
message" (dot product and sign agreement of per-position logits with the embedded message). These maps are
upsampled to 128x128 and concatenated into the U-Net at the 128 level, next to the RGB high-pass branch.

    python work_bassim/unet_feat_cv.py --method pixelseal
Writes work_bassim/oof_feat_<method>.npz (out-of-fold probabilities).
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from unet_cv_common import blk, dice_loss, to_t  # noqa: E402
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

dev = "cuda"


# ----------------------------------------------------------------------------- feature extraction
def up128(maps):
    """list of (h, w) tensors -> (K, 128, 128)"""
    return torch.cat([F.interpolate(m[None, None].float(), (128, 128), mode="bilinear", align_corners=False)[0]
                      for m in maps], 0)


def dot_agree(pp, msg):
    """pp: (nbits, h, w) logits, msg in {-1,+1}: returns [dot, sign agreement] maps."""
    return [(pp * msg[:, None, None]).mean(0), (torch.sign(pp) * msg[:, None, None]).mean(0)]


@torch.no_grad()
def feats_pixelseal(wm, img, bits):
    x = to_tensor(img).to(dev)
    msg = torch.from_numpy(bits).float().to(dev) * 2 - 1
    pp8 = wm.features(x)["per_position_logits"][0]                      # resized 256 -> 8x8
    xr = wm.det.preprocess(x)                                            # native 512 -> 16x16
    emb = wm.det.pixel_decoder.output_upscaling(wm.det.convnext(xr))
    lin = wm.det.pixel_decoder.linear
    pp = torch.einsum("bchw,oc->bohw", emb, lin.weight)[0] + lin.bias[:, None, None]
    maps = [pp[0]] + dot_agree(pp[1:], msg) + dot_agree(pp8, msg)
    return up128(maps).cpu().numpy()


@torch.no_grad()
def feats_trustmark(wm, img, bits):
    x = to_tensor(img).to(dev)
    msg = torch.from_numpy(bits).float().to(dev) * 2 - 1
    pp8 = wm.features(x)["per_position_logits"][0]                      # resized 245 -> 8x8
    n = wm.net                                                           # native 512 -> 16x16
    t = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
    for k in ("layer1", "layer2", "layer3", "layer4"):
        t = getattr(n, k)(t)
    pp = torch.einsum("bchw,oc->bohw", t, n.fc.weight)[0] + n.fc.bias[:, None, None]
    return up128(dot_agree(pp8, msg) + dot_agree(pp, msg)).cpu().numpy()


class SSContrib:
    """StegaStamp: exact contribution of each 13x13 cell of dec_conv_7 to logits . message."""

    def __init__(self, wm):
        import onnx
        import onnxruntime as ort
        self.wm = wm
        m = onnx.load("models/stegastamp/stega_stamp_logits.onnx")
        init = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
        self.W1 = init["gen_decoder/dec_dense/MatMul/ReadVariableOp__75"]    # (21632, 512)
        self.W2 = init["gen_decoder/decoder_out/MatMul/ReadVariableOp__77"]  # (512, 100)
        for name in ("gen_decoder/dec_conv_7/BiasAdd__443:0", "gen_decoder/dec_dense/BiasAdd:0"):
            m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
        self.sess = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def __call__(self, img, bits):
        x = self.wm._prep(to_tensor(img))
        _, c7, dense = self.sess.run(
            ["gen_decoder/decoder_out/BiasAdd:0", "gen_decoder/dec_conv_7/BiasAdd__443:0", "gen_decoder/dec_dense/BiasAdd:0"],
            {"image": x, "secret": np.zeros((1, 100), np.float32)})
        msg = bits.astype(np.float32) * 2 - 1
        g = self.W1 @ ((self.W2 @ msg) * (dense[0] > 0))
        contrib = (c7[0] * g.reshape(13, 13, 128)).sum(-1)
        return up128([torch.from_numpy(contrib)]).numpy()


def extract(method, split="train", std=None, limit=None):
    ds = LocalisationDataset("release", split, methods=[method])
    wm = load_watermark(method, models_dir="models", device=dev)
    fn = {"pixelseal": feats_pixelseal, "trustmark": feats_trustmark}.get(method) or SSContrib(wm)
    n = len(ds) if limit is None else min(limit, len(ds))
    X, M, Fe, names = [], [], [], []
    for i in range(n):
        it = ds[i]
        X.append(it["image"])
        names.append(it["image_name"])
        if split == "train":
            M.append(it["mask128"])
        Fe.append(fn(it["image"], it["message"]) if method == "stegastamp" else fn(wm, it["image"], it["message"]))
    X, Fe = np.stack(X), np.stack(Fe).astype(np.float32)
    if std is None:                                              # per-channel scale, fitted on train only
        std = Fe.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    Fe /= std
    return X, (np.stack(M) if M else None), Fe, std, names


# ----------------------------------------------------------------------------- model
class UNetF(nn.Module):
    def __init__(self, k, c=24):
        super().__init__()
        self.stem = nn.Sequential(blk(3, c), blk(c, c))
        self.e1 = nn.Sequential(blk(c, 2 * c, 2), blk(2 * c, 2 * c))
        self.e2 = nn.Sequential(blk(2 * c, 4 * c, 2), blk(4 * c, 4 * c))
        self.e3 = nn.Sequential(blk(4 * c, 8 * c, 2), blk(8 * c, 8 * c))
        self.e4 = nn.Sequential(blk(8 * c, 8 * c, 2), blk(8 * c, 8 * c))
        self.d3 = nn.Sequential(blk(16 * c, 8 * c), blk(8 * c, 8 * c))
        self.fe = nn.Sequential(blk(k, 16), blk(16, 16))
        self.d2 = nn.Sequential(blk(12 * c + 16, 4 * c), blk(4 * c, 4 * c))
        self.out = nn.Conv2d(4 * c, 1, 1)

    def forward(self, x, f):
        x = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)) * 10
        e1 = self.e1(self.stem(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        d3 = self.d3(torch.cat([up(e4), e3], 1))
        d2 = self.d2(torch.cat([up(d3), e2, self.fe(f)], 1))
        return self.out(d2)[:, 0]


def predict(net, X, Fe, bs=8):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(X), bs):
            out.append(torch.sigmoid(net(to_t(X[k:k + bs]).to(dev), torch.from_numpy(Fe[k:k + bs]).to(dev)).float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(Xtr, Mtr, Ftr, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = UNetF(Ftr.shape[1], args.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    spe = (len(Xtr) * args.crops_per_img + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=args.epochs * spe)
    Xt, Mt, Ft = to_t(Xtr), torch.from_numpy(Mtr).float(), torch.from_numpy(Ftr)
    n, cs = len(Xt), args.crop
    for ep in range(args.epochs):
        net.train()
        order = np.tile(np.arange(n), args.crops_per_img)
        rng.shuffle(order)
        for k in range(0, len(order), args.bs):
            xb, mb, fb = [], [], []
            for i in order[k:k + args.bs]:
                y0 = int(rng.integers(0, (512 - cs) // 4 + 1)) * 4
                x0 = int(rng.integers(0, (512 - cs) // 4 + 1)) * 4
                xb.append(Xt[i, :, y0:y0 + cs, x0:x0 + cs])
                mb.append(Mt[i, y0 // 4:(y0 + cs) // 4, x0 // 4:(x0 + cs) // 4])
                fb.append(Ft[i, :, y0 // 4:(y0 + cs) // 4, x0 // 4:(x0 + cs) // 4])
            xb, mb, fb = torch.stack(xb).to(dev), torch.stack(mb).to(dev), torch.stack(fb).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(xb, fb)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, mb) + dice_loss(lg, mb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep + 1}/{args.epochs} loss={loss.item():.4f}", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="pixelseal", choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops_per_img", type=int, default=4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to catch bugs")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.folds = 1, 2

    print("GPU:", torch.cuda.get_device_name(0), "| args:", vars(args), flush=True)
    t0 = time.time()
    lim = 24 if args.smoke else None
    X, M, Fe, std, _ = extract(args.method, "train", limit=lim)
    Xte, _, Fte, _, names_te = extract(args.method, "test", std=std, limit=8 if args.smoke else None)
    test_probs = np.zeros((len(Xte), 128, 128), np.float32)
    print(f"[{args.method}] extracted features {Fe.shape} in {time.time() - t0:.0f}s", flush=True)
    # zero-shot reference: best single-channel threshold is NOT computed here; see OOF score below.
    idx = np.random.default_rng(args.seed).permutation(len(X))
    oof = np.zeros(M.shape, np.float32)
    for k, te in enumerate(np.array_split(idx, args.folds)):
        tr = np.setdiff1d(idx, te)
        t1 = time.time()
        net = train_fold(X[tr], M[tr], Fe[tr], args, args.seed + k)
        oof[te] = predict(net, X[te], Fe[te])
        test_probs += predict(net, Xte, Fte) / args.folds
        s = score([(p > 0.5).astype(np.uint8) for p in oof[te]], M[te])
        print(f"  fold {k}: mIoU@0.5={s:.4f} ({time.time() - t1:.0f}s)", flush=True)
    s05 = score([(p > 0.5).astype(np.uint8) for p in oof], M)
    best_t, best_s = 0.5, s05
    for t in np.linspace(0.2, 0.8, 13):
        s = score([(p > t).astype(np.uint8) for p in oof], M)
        if s > best_s:
            best_t, best_s = float(t), s
    print(f"[{args.method}] OOF mIoU@0.5={s05:.4f} | best thr={best_t:.2f} -> {best_s:.4f} | total {time.time() - t0:.0f}s", flush=True)
    tag = "smoke_" if args.smoke else ""
    np.savez_compressed(f"work_bassim/oof_feat_{tag}{args.method}.npz", probs=oof.astype(np.float16), masks=M, idx=idx)
    np.savez_compressed(f"work_bassim/test_{tag}{args.method}.npz", probs=test_probs.astype(np.float16),
                        names=np.array(names_te), thr=best_t, oof_miou=best_s)


if __name__ == "__main__":
    main()
