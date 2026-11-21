"""Strategy F: Heretic-inspired Directional / Arbitrary-Rank Abliteration (ARA)."""

import copy
import torch

from contracts import MethodConfig
from task_template import DEVICE, preprocess


@torch.no_grad()
def _extract_forget_subspace(
    model: torch.nn.Module,
    forget_images: torch.Tensor,
    absent_mask: torch.Tensor,
    rank: int,
    batch_size: int = 128,
) -> torch.Tensor:
    """Return [768, rank] orthonormal forget directions orthogonal to absent class prototypes."""
    feats = []
    for i in range(0, min(len(forget_images), 1024), batch_size):
        x = preprocess(forget_images[i : i + batch_size])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
        feats.append(f.float().cpu())
    mat = torch.cat(feats, dim=0)
    mat = mat - mat.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(mat, full_matrices=False)
    q = vh[:rank].T  # [768, rank]

    # Project out any component aligned with the 55 absent class head prototypes.
    absent_w = model.head.weight.detach().float().cpu()[absent_mask.cpu()]
    if len(absent_w) > 0:
        qa, _ = torch.linalg.qr(absent_w.T)  # [768, <=55]
        q = q - qa @ (qa.T @ q)
        q, _ = torch.linalg.qr(q)
    return q


@torch.no_grad()
def run_abliteration(
    orig_model: torch.nn.Module,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
) -> torch.nn.Module:
    """Dosed directional ablation on upper residual writers and active head rows."""
    del orig_forget_logits, proxy_images, orig_proxy_logits
    student = copy.deepcopy(orig_model).to(DEVICE).eval()
    q = _extract_forget_subspace(student, forget_images, absent_mask, cfg.abliteration_rank).to(DEVICE)
    proj = q @ q.T  # [768, 768]
    alpha = cfg.abliteration_alpha

    for idx in range(cfg.freeze_until_block, len(student.blocks)):
        blk = student.blocks[idx]
        for layer in (blk.attn.proj, blk.mlp.fc2):
            w = layer.weight.float()
            layer.weight.copy_((w - alpha * (proj @ w)).to(layer.weight.dtype))

    active_idx = (~absent_mask).to(DEVICE)
    hw = student.head.weight.float()
    hw[active_idx] = hw[active_idx] - alpha * (hw[active_idx] @ proj)
    student.head.weight.copy_(hw.to(student.head.weight.dtype))
    return student
