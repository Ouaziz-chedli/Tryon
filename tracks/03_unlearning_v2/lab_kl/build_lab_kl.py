"""Build the lab-KL reference pair (P4.1).

orig_lab: copy of model.pt fine-tuned to saturation (CE < 0.005) on the 3,500 forget images,
          LayerScale init_values=1e-4 preserved (weights are loaded, never re-initialised).
gold_lab: SAME starting point but trained on a retain-proxy WITHOUT the 3,500 forget images
          (multi-grid PatchMix chimeras + the 55 absent-class head rows pinned as pseudo-retain),
          so KL(p_gold_lab || p_sub) at T=2 can be measured locally like the server does.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HARNESS = Path(__file__).resolve().parents[1]
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from eval_harness import compute_kl_at_temperature, make_patchmix_proxy  # noqa: E402
from task_template import DEVICE, NUM_CLASSES, load_forget_images, load_model, predict  # noqa: E402


def prepare_input(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float [N, 3, H, W] in [-1, 1] on DEVICE (exact server preprocessing)."""
    x = images_uint8.to(DEVICE).permute(0, 3, 1, 2).float() / 255.0
    return (x - 0.5) / 0.5


@torch.no_grad()
def ce_per_sample(model: torch.nn.Module, x: torch.Tensor, labels: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Per-sample CE of model on preprocessed inputs x, full set, no grad."""
    out = []
    for i in range(0, len(x), batch_size):
        logits = model(x[i : i + batch_size])
        y = labels[i : i + batch_size].to(logits.device)
        out.append(F.cross_entropy(logits.float(), y, reduction="none").cpu())
    return torch.cat(out)


@torch.no_grad()
def pin_absent_head_rows(student: torch.nn.Module, orig_model: torch.nn.Module, absent_mask: torch.Tensor) -> None:
    """Keep the 55 absent-class classifier rows at their original values (pseudo-retain anchor)."""
    dev = student.head.weight.device
    mask = absent_mask.to(dev)
    student.head.weight.data[mask] = orig_model.head.weight.data.to(dev)[mask]
    student.head.bias.data[mask] = orig_model.head.bias.data.to(dev)[mask]


def make_optimizer(model: torch.nn.Module, lr: float, head_lr_mult: float) -> torch.optim.Optimizer:
    """AdamW (wd=0, winning recipe) with a faster head group; empty groups are dropped."""
    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    groups = [{"params": head, "lr": lr * head_lr_mult}, {"params": body, "lr": lr}]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)


def train_probe_block(model, opt, x, labels, gen, steps, probe_every, batch_size, absent_anchor) -> int:
    """Run probe_every SGD steps on random batches; returns the updated step count."""
    for _ in range(probe_every):
        idx = torch.randint(0, len(x), (batch_size,), generator=gen)
        loss = F.cross_entropy(model(x[idx.to(x.device)]).float(), labels[idx.to(labels.device)].to(x.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if absent_anchor is not None:
            pin_absent_head_rows(model, absent_anchor[0], absent_anchor[1])
        steps += 1
    return steps


def fine_tune_to_ce(model, x, labels, target_ce, max_steps, lr, head_lr_mult, batch_size, probe_every, absent_anchor=None):
    """CE fine-tune toward fixed labels until full-set CE < target_ce or max_steps; (steps, ce_mean)."""
    if len(x) == 0 or len(labels) != len(x):
        raise ValueError("fine_tune_to_ce needs matching non-empty x and labels")
    model.train()
    opt = make_optimizer(model, lr, head_lr_mult)
    gen = torch.Generator().manual_seed(42)
    steps = 0
    ce_mean = float(ce_per_sample(model, x, labels, batch_size).mean())
    while ce_mean >= target_ce and steps < max_steps:
        steps = train_probe_block(model, opt, x, labels, gen, steps, probe_every, batch_size, absent_anchor)
        ce_mean = float(ce_per_sample(model, x, labels, batch_size).mean())
    model.eval()
    return steps, ce_mean


def build_retain_proxy(orig_model: torch.nn.Module, forget_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Retain-proxy WITHOUT the 3,500 forget images: 4x4 + 7x7 PatchMix chimeras (seeds 42/43)."""
    half = len(forget_images) // 2
    chimeras = torch.cat([
        make_patchmix_proxy(forget_images[:half], seed=42, grid=4),
        make_patchmix_proxy(forget_images[half:], seed=43, grid=7),
    ], dim=0)
    return chimeras, predict(orig_model, chimeras, batch_size=128).argmax(dim=-1)


def build_orig_lab(orig_model, forget_images, labels, args):
    """Deep copy of the original fine-tuned to saturation (CE < sat_ce) on the 3,500 forget images."""
    lab = copy.deepcopy(orig_model)
    steps, ce_mean = fine_tune_to_ce(
        lab, prepare_input(forget_images), labels, args.sat_ce,
        args.orig_max_steps, args.orig_lr, args.head_lr_mult, args.batch_size, args.probe_every)
    return lab.eval(), {"steps": steps, "train_ce_mean": round(ce_mean, 6)}


def build_gold_lab(orig_model, forget_images, absent_mask, args):
    """Same starting point trained only on the retain-proxy; the 55 absent rows stay pinned to orig."""
    chimeras, proxy_labels = build_retain_proxy(orig_model, forget_images)
    gold = copy.deepcopy(orig_model)
    steps, ce_mean = fine_tune_to_ce(
        gold, prepare_input(chimeras), proxy_labels, args.gold_target_ce,
        args.gold_max_steps, args.gold_lr, args.head_lr_mult, args.batch_size, args.probe_every,
        absent_anchor=(orig_model, absent_mask))
    return gold.eval(), {"steps": steps, "proxy_ce_mean": round(ce_mean, 6)}


@torch.no_grad()
def summarize_lab(orig_lab, gold_lab, images, labels, orig_info, gold_info, absent_mask) -> dict:
    """Server-exact stats of the lab pair on the forget images (saturation + gold/orig distance)."""
    orig_f = predict(orig_lab, images, batch_size=128)
    gold_f = predict(gold_lab, images, batch_size=128)
    orig_ce = F.cross_entropy(orig_f.float(), labels, reduction="none")
    gold_ce = F.cross_entropy(gold_f.float(), labels, reduction="none")
    return {
        "num_forget_images": len(images),
        "num_absent_classes": int(absent_mask.sum().item()),
        "orig_lab": {**orig_info, "forget_ce_mean": round(float(orig_ce.mean()), 4),
                     "forget_ce_std": round(float(orig_ce.std()), 4)},
        "gold_lab": {**gold_info, "forget_ce_mean": round(float(gold_ce.mean()), 4),
                     "forget_ce_std": round(float(gold_ce.std()), 4)},
        "kl_orig_gold_t2": round(float(compute_kl_at_temperature(orig_f, gold_f, 2.0).mean()), 4),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", type=Path, required=True, help="model.pt (the original 100-class classifier)")
    parser.add_argument("--images", type=Path, required=True, help="forget_images.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("output/lab_kl"))
    parser.add_argument("--sat-ce", type=float, default=0.005, help="orig_lab saturation threshold on the forget set")
    parser.add_argument("--orig-max-steps", type=int, default=500)
    parser.add_argument("--orig-lr", type=float, default=1e-4)
    parser.add_argument("--gold-target-ce", type=float, default=0.005, help="gold_lab saturation threshold on the proxy")
    parser.add_argument("--gold-max-steps", type=int, default=2000)
    parser.add_argument("--gold-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr-mult", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-every", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    orig = load_model(args.orig)
    images = load_forget_images(args.images)
    labels = predict(orig, images, batch_size=args.batch_size).argmax(dim=-1)
    absent_mask = torch.bincount(labels, minlength=NUM_CLASSES) == 0

    orig_lab, orig_info = build_orig_lab(orig, images, labels, args)
    gold_lab, gold_info = build_gold_lab(orig, images, absent_mask, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(orig_lab.state_dict(), args.output_dir / "orig_lab.pt")
    torch.save(gold_lab.state_dict(), args.output_dir / "gold_lab.pt")
    report = {
        "protocol": {"sat_ce": args.sat_ce, "gold_target_ce": args.gold_target_ce, "device": DEVICE,
                     "interpretation": "Lab references for local KL(p_gold_lab || p_sub); ranking-only, not an official score."},
        **summarize_lab(orig_lab, gold_lab, images, labels, orig_info, gold_info, absent_mask),
    }
    (args.output_dir / "build_lab_kl_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
