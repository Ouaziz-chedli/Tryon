"""Read-only Q1b inspection: full class histogram, npz keys, and head.weight geometry."""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from task_template import FORGET_IMAGES_FILE, load_forget_images, load_model, predict


def main() -> None:
    npz = np.load(FORGET_IMAGES_FILE)
    npz_meta = {k: [list(npz[k].shape), str(npz[k].dtype)] for k in npz.files}

    model = load_model(Path("model.pt"))
    images = load_forget_images(FORGET_IMAGES_FILE)
    logits = predict(model, images, batch_size=128)
    counts = torch.bincount(logits.argmax(dim=-1), minlength=100).cpu()

    active_mask = counts > 0
    w = model.head.weight.detach().float().cpu()
    b = model.head.bias.detach().float().cpu()
    w_norm = w.norm(dim=-1)
    w_unit = F.normalize(w, dim=-1)
    cos = w_unit @ w_unit.T
    cos.fill_diagonal_(0.0)

    active_idx = torch.where(active_mask)[0]
    inactive_idx = torch.where(~active_mask)[0]
    pairs = [[int(i), int(counts[i])] for i in torch.argsort(counts, descending=True) if counts[i] > 0]

    report = {
        "npz_keys": npz_meta,
        "active_class_counts_desc": pairs,
        "head_weight_norm": {
            "active_mean": round(float(w_norm[active_idx].mean()), 4),
            "inactive_mean": round(float(w_norm[inactive_idx].mean()), 4),
        },
        "head_bias": {
            "active_mean": round(float(b[active_idx].mean()), 4),
            "inactive_mean": round(float(b[inactive_idx].mean()), 4),
        },
        "head_cosine_offdiag_max": {
            "active_vs_active": round(float(cos[active_idx][:, active_idx].max()), 4),
            "active_vs_inactive": round(float(cos[active_idx][:, inactive_idx].max()), 4),
            "inactive_vs_inactive": round(float(cos[inactive_idx][:, inactive_idx].max()), 4),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
