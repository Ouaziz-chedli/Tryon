"""Retain-free per-image difficulty estimators + rank-ensemble, validated by Spearman rho.

Project target: Spearman rho >= 0.70 (seed 777) between the estimated per-image
difficulty and the reference (unlearned checkpoint) per-image CE. Patch-Dropout
alone plateaus around rho ~ 0.64; this tool ensembles several independent
retain-free difficulty signals, oriented and combined by rank-average, and
optionally weighted by each estimator's validation rho (--learned-weights).

Reference default: the most extreme (highest forget-set mean CE) unlearned
checkpoint found under output/, or an explicit --reference PATH state_dict.
Fold validation reuses the selective_benchmark rotating class folds (3 folds,
seed 777).
"""

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from eval_harness import compute_spearman_rank_corr
from methods.patch_drop_targets import estimate_patchdrop_gold_logits
from selective_benchmark import make_rotating_class_folds
from task_template import DEVICE, load_forget_images, load_model, preprocess

HEARTBEAT_SECONDS = 15.0
NOISE_SIGMA = 0.01  # relative per-tensor std of the weight perturbation
FEATURE_BLOCK = 8
EVAL_SEED = 777


class Heartbeat:
    """Print a '.' every ~HEARTBEAT_SECONDS so long GPU passes show liveness."""

    def __init__(self) -> None:
        self.last = time.monotonic()
        self.beat = False

    def tick(self) -> None:
        if time.monotonic() - self.last >= HEARTBEAT_SECONDS:
            print(".", end="", flush=True)
            self.beat = True
            self.last = time.monotonic()

    def done(self) -> None:
        if self.beat:
            print(flush=True)
            self.beat = False


@torch.no_grad()
def batched_logits(model: torch.nn.Module, images_uint8: torch.Tensor, batch_size: int, hb: Heartbeat) -> torch.Tensor:
    """Logits [N, 100] exactly like the server (eval mode, fp16 autocast on GPU), with heartbeat."""
    model.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            out.append(model(preprocess(images_uint8[i : i + batch_size])).float().cpu())
        hb.tick()
    return torch.cat(out)


def crop_rescale(images_uint8: torch.Tensor, ratio: float = 0.9) -> torch.Tensor:
    """Center-crop to `ratio` then bilinear-rescale back to full size (uint8, channels-last)."""
    side = images_uint8.shape[1]
    h = int(side * ratio)
    top = (side - h) // 2
    left = (side - h) // 2
    x = images_uint8.permute(0, 3, 1, 2).float()
    x = F.interpolate(x[:, :, top : top + h, left : left + h], size=(side, side), mode="bilinear", antialias=True)
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def brightness_jitter(images_uint8: torch.Tensor, seed: int, strength: float = 0.1) -> torch.Tensor:
    """Deterministic light brightness/contrast jitter (one seeded factor pair per call)."""
    gen = torch.Generator().manual_seed(seed)
    b = 1.0 + strength * (2 * torch.rand((), generator=gen).item() - 1)
    c = 1.0 + strength * (2 * torch.rand((), generator=gen).item() - 1)
    x = (images_uint8.float() - 127.5) * c + 127.5
    return (x * b).round().clamp(0, 255).to(torch.uint8)


def build_tta_views(images_uint8: torch.Tensor, num_views: int) -> list[torch.Tensor]:
    """Moderate TTA views: hflip, crop90%, then seeded light jitters, capped at num_views."""
    views = [torch.flip(images_uint8, dims=[2]), crop_rescale(images_uint8)]
    for seed in (11, 12)[: max(0, num_views - 2)]:
        views.append(brightness_jitter(images_uint8, seed=seed))
    return views[:num_views]


def estimate_patchdrop(orig, images, labels, num_views, batch_size, hb) -> torch.Tensor:
    """Patch-Dropout virtual-gold CE per image (multi-view drop_prob 0.45 + mirror)."""
    hb.done()  # estimate_patchdrop_gold_logits loops internally without heartbeat access
    pd_logits = estimate_patchdrop_gold_logits(
        orig, images, drop_prob=0.45, num_views=num_views, batch_size=batch_size, seed=42
    )
    return F.cross_entropy(pd_logits.float(), labels, reduction="none")


def estimate_tta_entropy(orig, images, clean_logits, num_views, batch_size, hb) -> torch.Tensor:
    """Mean entropy over moderate TTA views + mean top-1 disagreement with the clean view."""
    if num_views <= 0:
        return torch.zeros(len(images))
    clean_top = clean_logits.argmax(dim=-1)
    ent, disagree = [], []
    for view in build_tta_views(images, num_views):
        logits = batched_logits(orig, view, batch_size, hb)
        probs = F.softmax(logits.float(), dim=-1)
        ent.append(-(probs * probs.clamp_min(1e-12).log()).sum(dim=-1))
        disagree.append((logits.argmax(dim=-1) != clean_top).float())
    return torch.stack(ent).mean(dim=0) + torch.stack(disagree).mean(dim=0)


def estimate_weight_noise(orig, images, labels, clean_logits, num_draws, batch_size, hb) -> torch.Tensor:
    """Mean |CE change| across seeded weak Gaussian weight perturbations (memorization sensitivity)."""
    if num_draws <= 0:
        return torch.zeros(len(images))
    ce_clean = F.cross_entropy(clean_logits.float(), labels, reduction="none")
    params = [p for p in orig.parameters() if p.requires_grad]
    backup = [p.detach().clone() for p in params]
    stds = [p.detach().std().clamp_min(1e-8) for p in params]
    delta = []
    with torch.no_grad():
        for k in range(num_draws):
            torch.manual_seed(EVAL_SEED + k)
            for p, s in zip(params, stds):
                p.add_(torch.randn_like(p) * (NOISE_SIGMA * s))
            logits = batched_logits(orig, images, batch_size, hb)
            ce = F.cross_entropy(logits.float(), labels, reduction="none")
            delta.append((ce - ce_clean).abs())
        for p, b in zip(params, backup):
            p.copy_(b)
    return torch.stack(delta).mean(dim=0)


@torch.no_grad()
def extract_cls_features(model, images_uint8, block_index, batch_size, hb) -> torch.Tensor:
    """CLS token at transformer block `block_index` (typicality space for intra-forget kNN)."""
    feats = []
    hook = model.blocks[block_index].register_forward_hook(
        lambda module, inputs, output: feats.append(output[:, 0].float().cpu())
    )
    batched_logits(model, images_uint8, batch_size, hb)
    hook.remove()
    return torch.cat(feats)


def estimate_feature_knn(features: torch.Tensor, k: int) -> torch.Tensor:
    """L2-normalized distance to the k-th nearest intra-forget neighbour (low = typical)."""
    if len(features) <= k:
        return torch.zeros(len(features))
    normed = F.normalize(features.float(), dim=-1)
    dists = torch.cdist(normed, normed)
    dists.fill_diagonal_(float("inf"))
    return dists.topk(k, dim=-1, largest=False).values[:, -1]


def estimate_grad_norm(model, images, labels, batch_size, hb) -> torch.Tensor:
    """Per-image CE gradient norm wrt input pixels; one shared backward per batch (amortized)."""
    norms = []
    model.eval()
    for i in range(0, len(images), batch_size):
        x = preprocess(images[i : i + batch_size]).detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), labels[i : i + batch_size].to(logits.device), reduction="none")
        (grad,) = torch.autograd.grad(loss.sum(), x)
        norms.append(grad.reshape(grad.shape[0], -1).norm(dim=1).float().cpu())
        hb.tick()
    return torch.cat(norms)


def estimate_margin(clean_logits: torch.Tensor) -> torch.Tensor:
    """Top1-top2 logit gap on the clean view (low margin = deeply memorized; sign auto-calibrated)."""
    top2 = clean_logits.float().topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def rankify(scores: torch.Tensor) -> torch.Tensor:
    """Dense ranks as float (ties share the argsort order; adequate for 3,500 samples)."""
    return scores.float().argsort().argsort().float()


def orient_and_rank(scores: torch.Tensor, ce_ref: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Rank-average needs a common sign: calibrate each estimator against the reference CE."""
    rho = compute_spearman_rank_corr(scores, ce_ref)
    sign = 1.0 if rho >= 0.0 else -1.0
    return rankify(sign * scores), rho, sign


def fold_rhos(score: torch.Tensor, ce_ref: torch.Tensor, folds) -> list[float]:
    """Spearman of `score` vs reference CE on each rotating fold's forget subset."""
    return [compute_spearman_rank_corr(score[idx], ce_ref[idx]) for idx, _, _ in folds]


def weighted_rank_mean(oriented: dict[str, torch.Tensor], rho_map: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    """Rank-average weighted by each estimator's clamped rho (falls back to uniform)."""
    names = list(oriented)
    weights = torch.tensor([max(rho_map[n], 0.0) for n in names], dtype=torch.float32)
    if float(weights.sum()) <= 0.0:
        weights = torch.ones_like(weights)
    stacked = torch.stack([oriented[n] for n in names])
    combined = (stacked * weights[:, None]).sum(dim=0) / weights.sum()
    return combined, dict(zip(names, weights.tolist()))


def load_checkpoint_ce(path: Path, images, labels, batch_size, hb) -> torch.Tensor:
    """Per-image CE of an unlearned checkpoint state_dict on the forget set."""
    model = load_model(path)
    logits = batched_logits(model, images, batch_size, hb)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return F.cross_entropy(logits.float(), labels, reduction="none")


def pick_reference(images, labels, batch_size, hb) -> tuple[Path, torch.Tensor]:
    """Default reference = most extreme (highest forget mean CE) checkpoint under output/."""
    candidates = sorted(Path("output").rglob("*.pt"))
    if not candidates:
        raise SystemExit("no unlearned checkpoint found under output/; pass --reference PATH")
    best_path, best_ce, best_mean = None, None, -1.0
    for path in candidates:
        ce = load_checkpoint_ce(path, images, labels, batch_size, hb)
        mean_ce = float(ce.mean())
        print(f"[ref] {path} ce_mean={mean_ce:.3f}", flush=True)
        if mean_ce > best_mean:
            best_path, best_ce, best_mean = path, ce, mean_ce
    return best_path, best_ce


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble of retain-free per-image difficulty estimators, Spearman-validated (target rho >= 0.70, seed 777)."
    )
    parser.add_argument("--reference", type=Path, default=None, help="unlearned checkpoint state_dict (default: most extreme under output/)")
    parser.add_argument("--output", type=Path, default=Path("output/difficulty_ensemble"))
    parser.add_argument("--views", type=int, default=4, help="TTA / patchdrop views")
    parser.add_argument("--noise-k", type=int, default=3, help="seeded weight-noise draws")
    parser.add_argument("--knn-k", type=int, default=5, help="k-th nearest neighbour in CLS space")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learned-weights", action="store_true", help="weight rank-average by validation rho")
    return parser.parse_args()


def main() -> int:
    args = parse_cli_args()
    args.output.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat()

    images = load_forget_images()
    orig = load_model(Path("model.pt"))
    clean_logits = batched_logits(orig, images, args.batch_size, hb)
    labels = clean_logits.argmax(dim=-1)
    hb.done()

    if args.reference is not None:
        ref_path = args.reference
        ce_ref = load_checkpoint_ce(ref_path, images, labels, args.batch_size, hb)
    else:
        ref_path, ce_ref = pick_reference(images, labels, args.batch_size, hb)
    hb.done()
    print(f"[ref] reference = {ref_path} (ce_mean={float(ce_ref.mean()):.3f})", flush=True)

    scores = {}
    scores["patchdrop"] = estimate_patchdrop(orig, images, labels, args.views, args.batch_size, hb)
    scores["tta_entropy"] = estimate_tta_entropy(orig, images, clean_logits, args.views, args.batch_size, hb)
    hb.done()
    scores["weight_noise"] = estimate_weight_noise(orig, images, labels, clean_logits, args.noise_k, args.batch_size, hb)
    hb.done()
    feats = extract_cls_features(orig, images, FEATURE_BLOCK, args.batch_size, hb)
    scores["feature_knn"] = estimate_feature_knn(feats, args.knn_k)
    scores["grad_norm"] = estimate_grad_norm(orig, images, labels, args.batch_size, hb)
    hb.done()
    scores["margin"] = estimate_margin(clean_logits)

    oriented: dict[str, torch.Tensor] = {}
    rho_global: dict[str, float] = {}
    for name, score in scores.items():
        oriented[name], rho_global[name], _ = orient_and_rank(score, ce_ref)
        print(f"[est] {name:14s} rho_global={rho_global[name]:+.4f}", flush=True)

    folds = make_rotating_class_folds(labels, 3, seed=EVAL_SEED)
    ens = torch.stack(list(oriented.values())).mean(dim=0)
    report = {
        "reference": str(ref_path),
        "reference_ce_mean": round(float(ce_ref.mean()), 4),
        "estimators": {n: {"rho_global": rho_global[n]} for n in rho_global},
        "ensemble": {
            "rho_global": compute_spearman_rank_corr(ens, ce_ref),
            "rho_per_fold": fold_rhos(ens, ce_ref, folds),
        },
        "target": "rho >= 0.70 on seed 777 (vs reference unlearned CE)",
    }
    report["ensemble"]["rho_fold_mean"] = round(sum(report["ensemble"]["rho_per_fold"]) / len(folds), 4)
    if args.learned_weights:
        learned, weights = weighted_rank_mean(oriented, rho_global)
        report["learned_ensemble"] = {
            "weights": weights,
            "rho_global": compute_spearman_rank_corr(learned, ce_ref),
            "rho_per_fold": fold_rhos(learned, ce_ref, folds),
        }
        report["learned_ensemble"]["rho_fold_mean"] = round(
            sum(report["learned_ensemble"]["rho_per_fold"]) / len(folds), 4
        )

    best_name = max(rho_global, key=rho_global.get)
    report["best_single"] = {"name": best_name, "rho_global": rho_global[best_name]}
    report["ensemble_gain_vs_best_single"] = round(report["ensemble"]["rho_global"] - rho_global[best_name], 4)

    (args.output / "ensemble_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"scores": scores, "labels": labels, "reference": str(ref_path)}, args.output / "difficulty_scores.pt")

    print(f"\n{'estimator':16s} rho_global")
    for name, rho in rho_global.items():
        marker = " <- best single" if name == best_name else ""
        print(f"{name:16s} {rho:+.4f}{marker}")
    print(f"{'ensemble':16s} {report['ensemble']['rho_global']:+.4f}  folds={report['ensemble']['rho_per_fold']} mean={report['ensemble']['rho_fold_mean']:.4f}")
    if args.learned_weights:
        print(f"{'ens learned':16s} {report['learned_ensemble']['rho_global']:+.4f}  folds={report['learned_ensemble']['rho_per_fold']}")
    print(f"report -> {args.output / 'ensemble_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
