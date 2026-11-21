"""Strict data contracts for Track 3 validation harness and unlearning."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StressMetrics:
    """Output distribution metrics on raw or altered forget images."""

    ce_mean: float
    ce_std: float
    ce_p50: float
    ce_p90: float
    entropy_mean: float
    max_prob_mean: float
    frac_gt_95: float
    margin_mean: float
    top1_agree_orig: float
    kl_t2_from_orig: float


@dataclass(frozen=True)
class RetainProxyMetrics:
    """Local retain-preservation and weight-drift metrics."""

    proxy_kl_t2: float
    proxy_top1_agree: float
    absent_classes_mass_orig: float
    absent_classes_mass_sub: float
    absent_classes_kl_t2: float
    delta_l2_total: float
    delta_l2_head: float
    delta_l2_blocks: float


@dataclass(frozen=True)
class ValidationResult:
    """Checkpoint compliance check against server submission rules."""

    valid: bool
    file_size_mb: float
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EvalReport:
    """Complete local evaluation report for a candidate unlearned model."""

    num_images: int
    num_active_classes: int
    num_absent_classes: int
    forget_raw: StressMetrics
    forget_stress: dict[str, StressMetrics]
    retain_proxy: RetainProxyMetrics
    checkpoint: ValidationResult | None = None


@dataclass(frozen=True)
class MethodConfig:
    """Hyperparameters shared across Track 3 unlearning benchmark runs."""

    name: str
    strategy: str
    steps: int = 120
    batch_size: int = 64
    lr: float = 3e-5
    temperature: float = 2.0
    npo_beta: float = 0.1
    retain_kl_weight: float = 2.0
    l2_weight: float = 1e-3
    salun_keep_ratio: float = 0.25
    target_forget_ce: float = 0.8
    bad_teacher_temp: float = 5.0
    ssd_alpha: float = 1.0
    ssd_threshold: float = 2.0
    lora_rank: int = 8
    freeze_until_block: int = 6
    aug_mode: str = "moderate"
    proxy_kind: str = "patchmix"
    abliteration_rank: int = 8
    abliteration_alpha: float = 0.4
    use_ssd_init: bool = False
    loss_kind: str = "npo"
    seed: int = 42

