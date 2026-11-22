"""Feasibility test: can a small CNN tell watermarked 64x64 patches from clean ones, per method?

Patches are taken from the 510 real train images, only where the 512x512 ground-truth mask is
uniform over the whole patch (all 1 or all 0). Folds are split BY IMAGE (no patch leakage).
Prints AUC / accuracy per method and fold. Run from the task dir:
    python work_bassim/patch_feasibility.py
"""
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata

sys.path.insert(0, ".")
from wmloc.data import LocalisationDataset  # noqa: E402

P = 64            # patch size (pixels at 512 scale)
STRIDE = 32
PER_CLASS = 40    # max patches per class per image
FOLDS = 5
EPOCHS = 12
SEED = 0
dev = "cuda"


def auc(scores, labels):
    r = rankdata(scores)
    n1 = labels.sum()
    n0 = len(labels) - n1
    return (r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def uniform_windows(mask):
    """Return (ys, xs, label) of PxP windows fully inside class 1 or class 0 (stride STRIDE)."""
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    mean = F.avg_pool2d(m, P, stride=STRIDE)[0, 0].numpy()
    out = []
    for lab, sel in ((1, mean > 0.999), (0, mean < 0.001)):
        ys, xs = np.nonzero(sel)
        out.append((ys * STRIDE, xs * STRIDE, lab))
    return out


def build(method, rng):
    ds = LocalisationDataset("release", "train", methods=[method])
    X, y, g = [], [], []
    for i in range(len(ds)):
        it = ds[i]
        for ys, xs, lab in uniform_windows(it["mask"]):
            if len(ys) == 0:
                continue
            idx = rng.permutation(len(ys))[:PER_CLASS]
            for j in idx:
                X.append(it["image"][ys[j]:ys[j] + P, xs[j]:xs[j] + P])
                y.append(lab)
                g.append(i)
    return np.stack(X), np.array(y), np.array(g)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.f = nn.Sequential(blk(3, 32), blk(32, 32), nn.MaxPool2d(2),
                               blk(32, 64), blk(64, 64), nn.MaxPool2d(2),
                               blk(64, 128), nn.MaxPool2d(2), blk(128, 128))
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # high-pass residual so the net looks at the watermark band, not the content
        x = (x - F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, stride=1)) * 10
        return self.fc(self.f(x).mean((2, 3)))[:, 0]


def to_t(X):
    return torch.from_numpy(X).permute(0, 3, 1, 2).float().div(255)


def run_fold(Xtr, ytr, Xte, yte):
    torch.manual_seed(SEED)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Xtr_t, ytr_t = to_t(Xtr), torch.from_numpy(ytr).float()
    n = len(Xtr_t)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * ((n + 127) // 128))
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        for k in range(0, n, 128):
            b = perm[k:k + 128]
            loss = F.binary_cross_entropy_with_logits(net(Xtr_t[b].to(dev)), ytr_t[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        s = torch.cat([net(to_t(Xte[k:k + 512]).to(dev)).cpu() for k in range(0, len(Xte), 512)]).numpy()
    return auc(s, yte), float(((s > 0) == yte).mean())


def main():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    for method in ("pixelseal", "trustmark", "stegastamp"):
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        X, y, g = build(method, rng)
        print(f"\n[{method}] patches={len(X)} pos={int(y.sum())} neg={int((1 - y).sum())} "
              f"images={len(np.unique(g))} (build {time.time() - t0:.0f}s)", flush=True)
        imgs = np.unique(g)
        rng.shuffle(imgs)
        folds = np.array_split(imgs, FOLDS)
        aucs, accs = [], []
        for k, te in enumerate(folds):
            te_m = np.isin(g, te)
            a, acc = run_fold(X[~te_m], y[~te_m], X[te_m], y[te_m])
            aucs.append(a)
            accs.append(acc)
            print(f"  fold {k}: AUC={a:.4f} acc={acc:.4f}", flush=True)
        print(f"[{method}] mean AUC={np.mean(aucs):.4f} (+-{np.std(aucs):.4f}) acc={np.mean(accs):.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
