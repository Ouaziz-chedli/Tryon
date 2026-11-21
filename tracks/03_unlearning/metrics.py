"""Local proxy metrics and strict submission validation for Track 3."""

from pathlib import Path
import torch
import torch.nn.functional as F

from contracts import DiagnosticsReport, ValidationResult
from task_template import IMAGE_SIZE, NUM_CLASSES, build_model

MAX_SUBMISSION_MB = 450.0


def compute_kl_at_temperature(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute per-sample KL(p_teacher || p_student) at a given temperature."""
    p_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=-1)


def compute_loss_stats(
    logits: torch.Tensor, pseudo_labels: torch.Tensor
) -> tuple[float, float]:
    """Return (mean, std) of per-sample cross-entropy against pseudo_labels."""
    losses = F.cross_entropy(logits, pseudo_labels, reduction="none")
    return float(losses.mean().item()), float(losses.std().item())


def compute_entropy_and_confidence(logits: torch.Tensor) -> tuple[float, float]:
    """Return (mean_entropy, mean_max_prob) of softmax probabilities."""
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)
    return float(entropy.mean().item()), float(probs.max(dim=-1).values.mean().item())


def make_patchmix_proxy(
    images_uint8: torch.Tensor, seed: int = 42, grid: int = 4
) -> torch.Tensor:
    """Create out-of-sample proxy retain images by shuffling spatial patches across images."""
    if images_uint8.ndim != 4 or images_uint8.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"Expected [N, {IMAGE_SIZE}, {IMAGE_SIZE}, 3], got {tuple(images_uint8.shape)}")
    gen = torch.Generator().manual_seed(seed)
    n = len(images_uint8)
    patch = IMAGE_SIZE // grid
    out = images_uint8.clone()
    for r in range(grid):
        for c in range(grid):
            perm = torch.randperm(n, generator=gen)
            r0, r1 = r * patch, (r + 1) * patch
            c0, c1 = c * patch, (c + 1) * patch
            out[:, r0:r1, c0:c1, :] = images_uint8[perm, r0:r1, c0:c1, :]
    return out


def make_augmented_copy(images_uint8: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Create a deterministic altered copy (flip + mild crop/resize) to test forget generalization."""
    gen = torch.Generator().manual_seed(seed)
    flipped = torch.flip(images_uint8, dims=[2])
    n = len(flipped)
    perm = torch.randperm(n, generator=gen)
    half = n // 2
    out = images_uint8.clone()
    out[perm[:half]] = flipped[perm[:half]]
    return out


def build_diagnostics_report(
    orig_forget_logits: torch.Tensor,
    sub_forget_logits: torch.Tensor,
    sub_aug_logits: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    sub_proxy_logits: torch.Tensor,
    temperature: float = 2.0,
) -> DiagnosticsReport:
    """Aggregate local forget and retain proxy metrics into a DiagnosticsReport."""
    pseudo_labels = orig_forget_logits.argmax(dim=-1)
    mean_loss, std_loss = compute_loss_stats(sub_forget_logits, pseudo_labels)
    aug_mean, aug_std = compute_loss_stats(sub_aug_logits, pseudo_labels)
    entropy, max_prob = compute_entropy_and_confidence(sub_forget_logits)
    agree = float((sub_forget_logits.argmax(dim=-1) == pseudo_labels).float().mean().item())
    proxy_kl = float(
        compute_kl_at_temperature(orig_proxy_logits, sub_proxy_logits, temperature).mean().item()
    )
    proxy_agree = float(
        (orig_proxy_logits.argmax(dim=-1) == sub_proxy_logits.argmax(dim=-1)).float().mean().item()
    )
    return DiagnosticsReport(
        num_images=len(orig_forget_logits),
        mean_loss=mean_loss,
        std_loss=std_loss,
        mean_entropy=entropy,
        mean_max_prob=max_prob,
        aug_mean_loss=aug_mean,
        aug_std_loss=aug_std,
        top1_agreement_orig=agree,
        retain_proxy_kl=proxy_kl,
        retain_proxy_agreement=proxy_agree,
    )


def validate_submission(path: Path) -> ValidationResult:
    """Verify that a saved state_dict satisfies all server submission requirements."""
    errors: list[str] = []
    if not path.is_file():
        return ValidationResult(valid=False, file_size_mb=0.0, errors=(f"Missing file: {path}",))
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_SUBMISSION_MB:
        errors.append(f"File size {size_mb:.2f} MB exceeds {MAX_SUBMISSION_MB} MB limit.")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict):
        return ValidationResult(valid=False, file_size_mb=size_mb, errors=("Checkpoint is not a state_dict dict.",))
    for name, tensor in state_dict.items():
        if not torch.is_floating_point(tensor) or tensor.dtype != torch.float32:
            errors.append(f"Tensor {name} has dtype {tensor.dtype}, expected torch.float32.")
            break
        if not torch.isfinite(tensor).all():
            errors.append(f"Tensor {name} contains NaN or Inf.")
            break
    model = build_model()
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        errors.append(f"strict=True load failed: {exc}")
    if model.head.out_features != NUM_CLASSES:
        errors.append(f"Unexpected num_classes={model.head.out_features}.")
    return ValidationResult(valid=len(errors) == 0, file_size_mb=size_mb, errors=tuple(errors))
