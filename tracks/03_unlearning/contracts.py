"""Strict data contracts for Track 3 Machine Unlearning."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnlearnConfig:
    """Hyperparameters and paths for an unlearning run."""

    model_path: Path = Path("model.pt")
    forget_images_path: Path = Path("forget_images.npz")
    output_path: Path = Path("submission.pt")
    num_classes: int = 100
    temperature: float = 2.0
    steps: int = 150
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.0
    npo_beta: float = 0.1
    retain_kl_weight: float = 2.0
    l2_weight: float = 1e-3
    salun_keep_ratio: float = 0.2
    salun_batches: int = 16
    target_forget_loss: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class DiagnosticsReport:
    """Summary statistics of a model on the forget set and its augmentations."""

    num_images: int
    mean_loss: float
    std_loss: float
    mean_entropy: float
    mean_max_prob: float
    aug_mean_loss: float
    aug_std_loss: float
    top1_agreement_orig: float
    retain_proxy_kl: float
    retain_proxy_agreement: float


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a submission state_dict against server rules."""

    valid: bool
    file_size_mb: float
    errors: tuple[str, ...]
