"""Core unlearning algorithm: SalUn + NPO + PatchMix KL Retain Anchor."""

import copy
import torch
import torch.nn.functional as F

from contracts import UnlearnConfig
from metrics import compute_kl_at_temperature, make_patchmix_proxy
from saliency import compute_saliency_masks
from task_template import DEVICE, predict, preprocess


def unlearn_model(
    orig_model: torch.nn.Module,
    forget_images: torch.Tensor,
    config: UnlearnConfig,
) -> torch.nn.Module:
    """Unlearn `forget_images` from `orig_model` without a retain set."""
    torch.manual_seed(config.seed)
    orig_model.eval().requires_grad_(False)
    student = copy.deepcopy(orig_model).to(DEVICE).requires_grad_(True)

    orig_forget_logits = predict(orig_model, forget_images, batch_size=config.batch_size)
    pseudo_labels = orig_forget_logits.argmax(dim=-1)
    proxy_images = make_patchmix_proxy(forget_images, seed=config.seed)
    orig_proxy_logits = predict(orig_model, proxy_images, batch_size=config.batch_size)

    masks = compute_saliency_masks(
        student,
        forget_images,
        pseudo_labels,
        keep_ratio=config.salun_keep_ratio,
        batch_size=min(32, config.batch_size),
        num_batches=config.salun_batches,
    )
    return _optimize_student(
        student, orig_model, forget_images, orig_forget_logits,
        pseudo_labels, proxy_images, orig_proxy_logits, masks, config,
    )


def _optimize_student(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    pseudo_labels: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    masks: dict[str, torch.Tensor],
    config: UnlearnConfig,
) -> torch.nn.Module:
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters()}
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    n_forget = len(forget_images)
    n_proxy = len(proxy_images)

    for step in range(1, config.steps + 1):
        student.train()
        f_idx = torch.randint(0, n_forget, (config.batch_size,))
        p_idx = torch.randint(0, n_proxy, (config.batch_size,))

        loss, f_ce = _compute_step_loss(
            student,
            _augment_batch(forget_images[f_idx]),
            orig_forget_logits[f_idx],
            pseudo_labels[f_idx],
            proxy_images[p_idx],
            orig_proxy_logits[p_idx],
            orig_params,
            config,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _apply_saliency_masks(student, masks)
        optimizer.step()

        if step % 25 == 0 or f_ce >= config.target_forget_loss:
            print(f"[step {step:03d}/{config.steps}] total={loss.item():.4f} forget_ce={f_ce:.4f}")
        if f_ce >= config.target_forget_loss:
            print(f"Reached target_forget_loss={config.target_forget_loss:.2f} at step {step}; stopping early.")
            break
    return student.eval()


def _compute_step_loss(
    student: torch.nn.Module,
    forget_batch: torch.Tensor,
    orig_forget_logits_b: torch.Tensor,
    pseudo_labels_b: torch.Tensor,
    proxy_batch: torch.Tensor,
    orig_proxy_logits_b: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    config: UnlearnConfig,
) -> tuple[torch.Tensor, float]:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        sub_forget_logits = student(preprocess(forget_batch))
        sub_proxy_logits = student(preprocess(proxy_batch))

    npo = _npo_loss(
        sub_forget_logits.float(),
        orig_forget_logits_b.to(DEVICE),
        pseudo_labels_b.to(DEVICE),
        config.npo_beta,
    )
    retain_kl = compute_kl_at_temperature(
        orig_proxy_logits_b.to(DEVICE), sub_proxy_logits.float(), config.temperature
    ).mean()
    l2_anchor = _l2_weight_anchor(student, orig_params)
    total = npo + config.retain_kl_weight * retain_kl + config.l2_weight * l2_anchor
    forget_ce = float(F.cross_entropy(sub_forget_logits.float(), pseudo_labels_b.to(DEVICE)).item())
    return total, forget_ce


def _npo_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Negative Preference Optimization (COLM 2024) bounded forget loss."""
    sub_log_probs = F.log_softmax(sub_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_log_probs = F.log_softmax(orig_logits, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    log_ratio = sub_log_probs - orig_log_probs
    return (2.0 / beta) * F.softplus(beta * log_ratio).mean()


def _l2_weight_anchor(
    student: torch.nn.Module, orig_params: dict[str, torch.Tensor]
) -> torch.Tensor:
    anchor = torch.tensor(0.0, device=DEVICE)
    for name, param in student.named_parameters():
        if param.requires_grad:
            anchor = anchor + F.mse_loss(param, orig_params[name], reduction="sum")
    return anchor


def _apply_saliency_masks(
    student: torch.nn.Module, masks: dict[str, torch.Tensor]
) -> None:
    for name, param in student.named_parameters():
        if param.grad is not None and name in masks:
            param.grad.mul_(masks[name].to(param.grad.device))


def _augment_batch(images_uint8: torch.Tensor) -> torch.Tensor:
    """Lightweight GPU/CPU-safe augmentations so unlearning transfers to altered copies."""
    out = images_uint8.clone()
    flip_mask = torch.rand(len(out)) < 0.5
    if flip_mask.any():
        out[flip_mask] = torch.flip(out[flip_mask], dims=[2])
    return out
