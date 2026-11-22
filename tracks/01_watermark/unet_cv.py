"""U-Net localisation for ONE watermark method, 5-fold CV by image on the real train split.

Input : 512x512 RGB (high-pass residual computed inside the net).
Output: 128x128 logits (1 = watermarked), scored with the official mIoU (wmloc.utils.score).
Training uses random 256x256 crops aligned on the 4 px grid (no flips: the watermark pattern is not
flip-invariant). Evaluation runs on the full 512x512 image.

Run from the task dir, e.g.:
    python work_bassim/unet_cv.py --method stegastamp
Writes work_bassim/oof_<method>.npz (out-of-fold probabilities) for later threshold tuning / post-processing.
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score  # noqa: E402

dev = "cuda"


def blk(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """512 -> 128 output grid. Skips at 256 / 128 / 64 / 32."""

    def __init__(self, c=24):
        super().__init__()
        self.stem = nn.Sequential(blk(3, c), blk(c, c))                    # 512
        self.e1 = nn.Sequential(blk(c, 2 * c, 2), blk(2 * c, 2 * c))       # 256
        self.e2 = nn.Sequential(blk(2 * c, 4 * c, 2), blk(4 * c, 4 * c))   # 128
        self.e3 = nn.Sequential(blk(4 * c, 8 * c, 2), blk(8 * c, 8 * c))   # 64
        self.e4 = nn.Sequential(blk(8 * c, 8 * c, 2), blk(8 * c, 8 * c))   # 32
        self.d3 = nn.Sequential(blk(16 * c, 8 * c), blk(8 * c, 8 * c))     # 64
        self.d2 = nn.Sequential(blk(12 * c, 4 * c), blk(4 * c, 4 * c))     # 128
        self.out = nn.Conv2d(4 * c, 1, 1)

    def forward(self, x):
        x = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)) * 10  # high-pass
        s = self.stem(x)
        e1 = self.e1(s)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d3 = self.d3(torch.cat([F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        return self.out(d2)[:, 0]


def load(method):
    ds = LocalisationDataset("release", "train", methods=[method])
    X = np.stack([ds[i]["image"] for i in range(len(ds))])
    M = np.stack([ds[i]["mask128"] for i in range(len(ds))])
    return X, M


def to_t(x):
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255)


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2))
    den = p.sum((1, 2)) + y.sum((1, 2))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def predict(net, X, bs=8):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in range(0, len(X), bs):
            out.append(torch.sigmoid(net(to_t(X[k:k + bs]).to(dev)).float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(Xtr, Mtr, args, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = UNet(args.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps_per_epoch = (len(Xtr) * args.crops_per_img + args.bs - 1) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=args.epochs * steps_per_epoch)
    Xt = to_t(Xtr)
    Mt = torch.from_numpy(Mtr).float()
    n = len(Xt)
    cs = args.crop  # crop size in pixels at 512 scale
    for ep in range(args.epochs):
        net.train()
        order = np.tile(np.arange(n), args.crops_per_img)
        rng.shuffle(order)
        for k in range(0, len(order), args.bs):
            xb, mb = [], []
            for i in order[k:k + args.bs]:
                y0 = int(rng.integers(0, (512 - cs) // 4 + 1)) * 4
                x0 = int(rng.integers(0, (512 - cs) // 4 + 1)) * 4
                xb.append(Xt[i, :, y0:y0 + cs, x0:x0 + cs])
                mb.append(Mt[i, y0 // 4:(y0 + cs) // 4, x0 // 4:(x0 + cs) // 4])
            xb, mb = torch.stack(xb).to(dev), torch.stack(mb).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(xb)
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
    ap.add_argument("--method", default="stegastamp", choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops_per_img", type=int, default=4)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("GPU:", torch.cuda.get_device_name(0), "| args:", vars(args), flush=True)
    t0 = time.time()
    X, M = load(args.method)
    print(f"[{args.method}] loaded {len(X)} images in {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X))
    folds = np.array_split(idx, args.folds)
    oof = np.zeros(M.shape, np.float32)
    for k, te in enumerate(folds):
        tr = np.setdiff1d(idx, te)
        t1 = time.time()
        net = train_fold(X[tr], M[tr], args, args.seed + k)
        oof[te] = predict(net, X[te])
        s = score([(p > 0.5).astype(np.uint8) for p in oof[te]], M[te])
        print(f"  fold {k}: mIoU@0.5={s:.4f} ({time.time() - t1:.0f}s)", flush=True)

    s05 = score([(p > 0.5).astype(np.uint8) for p in oof], M)
    best_t, best_s = 0.5, s05
    for t in np.linspace(0.2, 0.8, 13):
        s = score([(p > t).astype(np.uint8) for p in oof], M)
        if s > best_s:
            best_t, best_s = float(t), s
    print(f"[{args.method}] OOF mIoU@0.5={s05:.4f} | best thr={best_t:.2f} -> {best_s:.4f} | total {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(f"work_bassim/oof_{args.method}.npz", probs=oof.astype(np.float16), masks=M, idx=idx)


if __name__ == "__main__":
    main()
