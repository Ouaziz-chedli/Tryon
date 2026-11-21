"""Training-time augmentations (none, moderate, strong) for altered-copy generalization."""

import torch
import torchvision.transforms.v2 as T


_MOD_AUG = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.75, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02)], p=0.4),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.3, 1.0))], p=0.25),
    ]
)

_STRONG_AUG = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(15)], p=0.4),
        T.RandomApply([T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)], p=0.6),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0))], p=0.3),
    ]
)


def augment_training_batch(images_uint8: torch.Tensor, mode: str = "moderate") -> torch.Tensor:
    """Apply 'none' (hflip-only), 'moderate', or 'strong' augmentations to [B, H, W, 3] uint8."""
    if mode == "none":
        flipped = torch.flip(images_uint8, dims=[2])
        mask = torch.rand(len(images_uint8), device=images_uint8.device) < 0.5
        out = images_uint8.clone()
        out[mask] = flipped[mask]
        return out
    pipeline = _MOD_AUG if mode == "moderate" else _STRONG_AUG
    aug = pipeline(images_uint8.permute(0, 3, 1, 2))
    if mode == "strong" and torch.rand(1).item() < 0.3:
        noise = torch.randn(aug.shape, device=aug.device) * 15.0
        aug = (aug.float() + noise).clamp(0, 255).to(torch.uint8)
    return aug.permute(0, 2, 3, 1).contiguous()
