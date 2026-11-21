"""Read-only empirical exploration for Track 3 (Q1-Q4)."""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from task_template import DEVICE, load_forget_images, load_model, predict, preprocess


def analyze_q1_classes(logits: torch.Tensor) -> dict:
    """Q1: Analyze predicted class distribution across all 100 classes."""
    preds = logits.argmax(dim=-1)
    counts = torch.bincount(preds, minlength=logits.shape[1]).cpu()
    probs_cls = counts.float() / counts.sum()
    nz = probs_cls[probs_cls > 0]
    norm_ent = float(-(nz * nz.log()).sum() / torch.log(torch.tensor(float(len(counts)))))
    top_vals, top_idx = torch.topk(counts, k=10)
    return {
        "active_classes": int((counts > 0).sum().item()),
        "min_count": int(counts.min().item()),
        "median_count": float(counts.float().median().item()),
        "max_count": int(counts.max().item()),
        "std_count": float(counts.float().std().item()),
        "normalized_class_entropy": round(norm_ent, 4),
        "top10_classes": [[int(i), int(c)] for i, c in zip(top_idx.tolist(), top_vals.tolist())],
    }


def _logit_metrics(logits: torch.Tensor, ref_preds: torch.Tensor) -> dict:
    probs = F.softmax(logits.float(), dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    max_p = top2[:, 0]
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    ce = F.cross_entropy(logits.float(), ref_preds, reduction="none")
    return {
        "max_prob_mean": round(float(max_p.mean()), 4),
        "max_prob_p10": round(float(torch.quantile(max_p, 0.1)), 4),
        "max_prob_p50": round(float(torch.quantile(max_p, 0.5)), 4),
        "max_prob_p90": round(float(torch.quantile(max_p, 0.9)), 4),
        "frac_gt_95": round(float((max_p > 0.95).float().mean()), 4),
        "entropy_mean": round(float(entropy.mean()), 4),
        "margin_mean": round(float(margin.mean()), 4),
        "ce_mean": round(float(ce.mean()), 4),
        "ce_std": round(float(ce.std()), 4),
    }


def _make_perturbations(images_uint8: torch.Tensor) -> dict[str, torch.Tensor]:
    nchw = images_uint8.permute(0, 3, 1, 2)
    cropped = TF.resized_crop(nchw, top=22, left=22, height=180, width=180, size=[224, 224])
    blurred = TF.gaussian_blur(nchw, kernel_size=[5, 5], sigma=[1.5, 1.5])
    noise = torch.randn(images_uint8.shape, generator=torch.Generator().manual_seed(0)) * 15.0
    noisy = (images_uint8.float() + noise).clamp(0, 255).to(torch.uint8)
    return {
        "hflip": torch.flip(images_uint8, dims=[2]),
        "crop80": cropped.permute(0, 2, 3, 1).contiguous(),
        "blur_s1.5": blurred.permute(0, 2, 3, 1).contiguous(),
        "noise_15": noisy,
    }


def analyze_q2_memorization(model: torch.nn.Module, images: torch.Tensor, raw_logits: torch.Tensor) -> dict:
    """Q2: Measure confidence, entropy, CE, and perturbation sensitivity."""
    ref_preds = raw_logits.argmax(dim=-1)
    p_raw_t2 = F.softmax(raw_logits.float() / 2.0, dim=-1)
    out = {"raw": _logit_metrics(raw_logits, ref_preds), "perturbations": {}}
    for name, pert_imgs in _make_perturbations(images).items():
        p_logits = predict(model, pert_imgs, batch_size=128)
        agree = float((p_logits.argmax(dim=-1) == ref_preds).float().mean())
        kl_t2 = float(F.kl_div(F.log_softmax(p_logits.float() / 2.0, dim=-1), p_raw_t2, reduction="batchmean"))
        stats = _logit_metrics(p_logits, ref_preds)
        stats["top1_agree_raw"] = round(agree, 4)
        stats["kl_t2_from_raw"] = round(kl_t2, 4)
        out["perturbations"][name] = stats
    return out


@torch.no_grad()
def analyze_q3_structure(model: torch.nn.Module, images: torch.Tensor) -> dict:
    """Q3: Image channel stats and CLS embedding cosine similarity structure."""
    rgb = images.float() / 255.0
    feats = []
    for i in range(0, len(images), 128):
        x = preprocess(images[i : i + 128])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
        feats.append(F.normalize(f.float(), dim=-1))
    emb = torch.cat(feats, dim=0)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1.0)
    max_per_img = sim.max(dim=-1).values
    upper = sim[torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)]
    return {
        "shape": list(images.shape),
        "rgb_mean": [round(float(x), 4) for x in rgb.mean(dim=(0, 1, 2))],
        "rgb_std": [round(float(x), 4) for x in rgb.std(dim=(0, 1, 2))],
        "nn_cos_max": round(float(max_per_img.max()), 4),
        "nn_cos_p95": round(float(torch.quantile(max_per_img, 0.95)), 4),
        "nn_cos_median": round(float(max_per_img.median()), 4),
        "pairs_gt_95": int((upper > 0.95).sum().item()),
        "pairs_gt_90": int((upper > 0.90).sum().item()),
        "pairs_gt_85": int((upper > 0.85).sum().item()),
    }


def _group_name(param_name: str) -> str:
    if param_name.startswith("patch_embed"):
        return "patch_embed"
    if param_name in ("cls_token", "pos_embed"):
        return "pos_cls"
    if param_name.startswith("blocks."):
        idx = int(param_name.split(".")[1])
        return f"blocks_{idx // 4 * 4}-{idx // 4 * 4 + 3}"
    return "head" if param_name.startswith("head") else "norm_other"


def analyze_q4_weights_and_grads(model: torch.nn.Module, images: torch.Tensor, pseudo_labels: torch.Tensor) -> dict:
    """Q4: Parameter norms and forget-gradient energy per ViT stage."""
    model.eval()
    model.zero_grad(set_to_none=True)
    grad_sq = {n: 0.0 for n, p in model.named_parameters() if p.requires_grad}
    for i in range(0, min(512, len(images)), 64):
        loss = F.cross_entropy(model(preprocess(images[i : i + 64])), pseudo_labels[i : i + 64].to(DEVICE))
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_sq[n] += float(p.grad.detach().float().pow(2).sum().item())
        model.zero_grad(set_to_none=True)

    groups: dict[str, dict[str, float]] = {}
    for n, p in model.named_parameters():
        g = _group_name(n)
        entry = groups.setdefault(g, {"params": 0, "weight_l2_sq": 0.0, "grad_l2_sq": 0.0})
        entry["params"] += p.numel()
        entry["weight_l2_sq"] += float(p.detach().float().pow(2).sum().item())
        entry["grad_l2_sq"] += grad_sq.get(n, 0.0)
    return {
        k: {
            "params": int(v["params"]),
            "weight_l2": round(v["weight_l2_sq"] ** 0.5, 3),
            "grad_l2": round(v["grad_l2_sq"] ** 0.5, 4),
            "grad_rms": round((v["grad_l2_sq"] / max(1, v["params"])) ** 0.5, 6),
        }
        for k, v in groups.items()
    }


def main() -> None:
    model = load_model(Path("model.pt"))
    images = load_forget_images(Path("forget_images.npz"))
    raw_logits = predict(model, images, batch_size=128)
    pseudo_labels = raw_logits.argmax(dim=-1)
    report = {
        "Q1_classes": analyze_q1_classes(raw_logits),
        "Q2_memorization": analyze_q2_memorization(model, images, raw_logits),
        "Q3_structure": analyze_q3_structure(model, images),
        "Q4_weights_grads": analyze_q4_weights_and_grads(model, images, pseudo_labels),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
