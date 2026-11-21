"""Gradient-based weight saliency mask (SalUn, ICLR 2024)."""

import torch
import torch.nn.functional as F

from task_template import preprocess


def compute_saliency_masks(
    model: torch.nn.Module,
    forget_images: torch.Tensor,
    pseudo_labels: torch.Tensor,
    keep_ratio: float = 0.2,
    batch_size: int = 32,
    num_batches: int = 16,
) -> dict[str, torch.Tensor]:
    """Return binary masks selecting the top `keep_ratio` gradient-salient weights."""
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    grad_accum = _accumulate_forget_gradients(
        model, forget_images, pseudo_labels, batch_size, num_batches
    )
    if keep_ratio == 1.0:
        return {k: torch.ones_like(v) for k, v in grad_accum.items()}
    threshold = _compute_global_quantile(grad_accum, 1.0 - keep_ratio)
    return {k: (v >= threshold).to(v.dtype) for k, v in grad_accum.items()}


def _accumulate_forget_gradients(
    model: torch.nn.Module,
    forget_images: torch.Tensor,
    pseudo_labels: torch.Tensor,
    batch_size: int,
    num_batches: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    model.zero_grad(set_to_none=True)
    accum = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in model.named_parameters() if p.requires_grad}
    limit = min(len(forget_images), num_batches * batch_size)
    for start in range(0, limit, batch_size):
        end = min(start + batch_size, limit)
        x = preprocess(forget_images[start:end])
        y = pseudo_labels[start:end].to(x.device)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                accum[name] += param.grad.detach().abs().float()
        model.zero_grad(set_to_none=True)
    return accum


def _compute_global_quantile(tensors: dict[str, torch.Tensor], q: float) -> float:
    """Estimate quantile across all parameter tensors without exceeding torch.quantile limits."""
    samples = []
    for t in tensors.values():
        flat = t.reshape(-1)
        step = max(1, flat.numel() // 4096)
        samples.append(flat[::step].cpu())
    merged = torch.cat(samples)
    return float(torch.quantile(merged, q).item())
