"""Strategy C: Selective Synaptic Dampening (SSD) with tempered Fisher proxy."""

import torch
import torch.nn.functional as F

from contracts import MethodConfig
from methods.common import prepare_student
from task_template import preprocess


def _tempered_grad_sq(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    batch_size: int = 32,
    num_batches: int = 12,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    model.eval()
    model.zero_grad(set_to_none=True)
    dev = next(model.parameters()).device
    imp = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters() if p.requires_grad}
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(images), generator=gen)[: min(len(images), num_batches * batch_size)]
    steps = 0
    for start in range(0, len(perm), batch_size):
        idx = perm[start : start + batch_size]
        x = preprocess(images[idx]).to(dev)
        y = labels[idx].to(dev)
        loss = F.cross_entropy(model(x) / temperature, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                imp[n] += p.grad.detach().float().pow(2)
        model.zero_grad(set_to_none=True)
        steps += 1
    return {k: v / max(1, steps) for k, v in imp.items()}


def run_ssd(
    orig_model: torch.nn.Module,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
) -> torch.nn.Module:
    """One-shot Selective Synaptic Dampening on upper blocks and active head classes."""
    student = prepare_student(orig_model, cfg.freeze_until_block)
    imp_f = _tempered_grad_sq(student, forget_images, orig_forget_logits.argmax(dim=-1), cfg.temperature, seed=cfg.seed)
    imp_r = _tempered_grad_sq(student, proxy_images, orig_proxy_logits.argmax(dim=-1), cfg.temperature, seed=cfg.seed + 1)

    absent_dev = absent_mask.to(next(student.parameters()).device)
    with torch.no_grad():
        for name, param in student.named_parameters():
            if not param.requires_grad:
                continue
            ratio = imp_f[name] / (imp_r[name] + 1e-8)
            damp = torch.where(ratio > cfg.ssd_threshold, (cfg.ssd_alpha * imp_r[name] / (imp_f[name] + 1e-8)).clamp(max=1.0), 1.0)
            if name in ("head.weight", "head.bias"):
                damp[absent_dev] = 1.0
            param.mul_(damp.to(param.dtype))
    return student.eval()
