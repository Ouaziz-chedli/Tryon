"""Strategy D: Alternative retain-proxy generators (PatchMix, MixUp, Structured Noise)."""

import torch
import torch.nn.functional as F

from eval_harness import make_patchmix_proxy


def make_retain_proxy(images_uint8: torch.Tensor, kind: str = "patchmix", seed: int = 42) -> torch.Tensor:
    """Generate a synthetic retain proxy of the same shape/dtype as images_uint8."""
    if kind == "patchmix":
        return make_patchmix_proxy(images_uint8, seed=seed, grid=4)
    gen = torch.Generator().manual_seed(seed)
    n = len(images_uint8)
    if kind == "mixup":
        perm = torch.randperm(n, generator=gen)
        lam = 0.5
        mixed = lam * images_uint8.float() + (1.0 - lam) * images_uint8[perm].float()
        return mixed.clamp(0, 255).to(torch.uint8)
    if kind == "structured_noise":
        low = torch.randn((n, 3, 14, 14), generator=gen) * 65.0 + 100.0
        up = F.interpolate(low, size=images_uint8.shape[1:3], mode="bilinear", align_corners=False)
        return up.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).contiguous()
    raise ValueError(f"Unknown retain proxy kind: {kind}")
