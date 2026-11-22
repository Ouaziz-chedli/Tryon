"""SCRUB+R Round 4 ascent variants: SR9 rest-KL pin and SR10 per-image PD-target mimic.

Root cause: scrub_max_loss ascends only the label log-prob (gather on y), so the 99
logits c != y drift freely during forgetting and prediction stays capped ~0.143.

SR9 (rest-KL, variant basse): adds a KL(T=2) distillation of the renormalized
non-label distribution toward orig inside the max-step loss; the PD-weighted NPO
push on y keeps its deep tail (sigma ~2.3 intact) while the rest geometry stays
pinned to orig.

SR10 (pd_target, Antigravity rank 1): replaces the ascent objective with a
gold_mimic-style loss whose per-image targets come from the already-computed
Patch-Dropout gold logits, calibrated to the gold mean CE (calibrate_per_image_targets
keeps the 99 non-target logits at orig verbatim). Attacks KL_bin (per-image PD

Round 8 (SR14): the SR10 mimic was faithful but its PD targets are ~3 nats too easy
(mean CE 0.59 vs gold 1.71) while their per-image ORDER is excellent (rho 0.72). The
affine recalibration (ce_i -> a*ce_pd_i + b onto the gold mu/sigma) fixes the
distribution without touching the order, and refinement starts from the #24 champion
checkpoint (init_from) with the L2 anchor holding that init state.

Rewind (rounds 4+): the shared loop selects snapshots through a HARD anchor-budget
filter (ProxyKL <= 0.056, dTheta <= 4.70) before minimizing rewind_dist -- the
upstream p50 >= 0.15 pool excluded the healthy shallow epochs and forced selection
into the deep drift tail (SR8 landed ProxyKL 0.20 that way).

This is a NEW module: methods/scrub_r.py stays verbatim so in-flight rounds 2-3
jobs never pick up changed ascent code.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import torch
import torch.nn.functional as F

from augment import augment_training_batch
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
from methods.common import (
    build_optimizer,
    get_batch_orig_logits,
    l2_anchor_loss,
    mask_absent_head_grads,
    prepare_student,
    restore_absent_head_rows,
    retain_kl_loss,
    should_stop_early,
)
from methods.patch_drop_targets import (
    calibrate_per_image_targets,
    estimate_patchdrop_gold_logits,
)
from methods.scrub_r import (
    ScrubRConfig,
    _apply_masked_step,
    _do_max_epoch,
    _do_min_epoch,
    _eval_epoch_snapshot,
    _scale_optimizer_lr,
    _select_rewind_snapshot,
    compute_pd_difficulty_ratio,
    scrub_max_loss,
)
from methods.salun_npo import compute_tempered_saliency
from task_template import DEVICE, preprocess


# Hard anchor budget for rewind selection (server window bounds never defended upstream).
REWIND_MAX_PROXY_KL = 0.056
REWIND_MAX_DELTA_L2 = 4.70

# Gold bell moments (14-point fit): the lognormal target distribution for quantile mapping.
GOLD_BELL_MEAN, GOLD_BELL_STD = 1.721, 2.468

def _in_anchor_budget(snap: dict, delta_l2_max: float | None = REWIND_MAX_DELTA_L2) -> bool:
    """Hard ProxyKL check, plus dTheta when a budget is given (None = pkl-only)."""
    if snap["proxy_kl"] > REWIND_MAX_PROXY_KL:
        return False
    return delta_l2_max is None or snap["delta_l2"] <= delta_l2_max


def _select_rewind_snapshot_budgeted(
    snapshots: list[dict],
    use_rewind: bool,
    delta_l2_max: float | None = REWIND_MAX_DELTA_L2,
) -> dict:
    """Budgeted rewind: hard pkl (+ optional dT) pool filter BEFORE minimizing rewind_dist.

    The upstream selector pools only snapshots with p50 >= 0.15, which excludes the
    healthy shallow epochs (pkl ~0.05, p50 < 0.15) and forces selection into the
    deep drift tail. Here the pool is every snapshot inside the anchor budget; the
    dist minimization itself rejects near-no-op snapshots (their rewind_dist ~5+).
    delta_l2_max=None is the round-8 refinement mode (pkl-only): the #24 champion
    starts at dT 4.65 -- right at the fresh-run-calibrated 4.70 edge -- and its
    retain is already proven, so ProxyKL alone is the binding budget there.
    Falls back to the upstream selector when NO snapshot is inside the budget.
    """
    if not use_rewind:
        return snapshots[-1]
    budgeted = [s for s in snapshots if _in_anchor_budget(s, delta_l2_max)]
    if not budgeted:
        return _select_rewind_snapshot(snapshots, use_rewind)
    return min(budgeted, key=lambda s: s["rewind_dist"])

@dataclass(frozen=True)
class ScrubRRestConfig(ScrubRConfig):
    """SCRUB+R config + weight of the rest-class KL(T=2) distillation term (SR9)."""

    rest_kl_weight: float = 0.35


@dataclass(frozen=True)
class ScrubRBudgetedConfig(ScrubRConfig):
    """Marker config: plain upstream NPO ascent but budgeted (pkl/dT-hard) rewind.

    No new hyperparameters -- round 5 tunes existing fields (pd_gate_scale=0 flat
    gate, pd_alpha, retain_kl_weight, salun_keep_ratio); the type only routes
    training to run_scrub_r_budgeted so the hard anchor-budget selection applies.
    """

PD_AFFINE_AUTO = (-1.0, -1.0)  # runner resolves this sentinel to the gold affine (a, b)

@dataclass(frozen=True)
class ScrubRPdTargetConfig(ScrubRConfig):
    """SCRUB+R config + per-image PD-target mimic knobs (SR10/SR13/SR14).

    pd_target_affine: None keeps the mean-scaled calibration; a concrete (a, b) applies
    an affine recalibration ce_i -> a*ce_pd_i + b (order preserved, a > 0); the
    PD_AFFINE_AUTO sentinel is resolved to (a, b) from the gold stats by the runner.
    pd_target_quantile: rank-mapped targets instead (pd_ce ranks onto the gold bell
    lognormal, moments of the 14-point fit) -- takes precedence over the affine field
    and needs no sentinel (fully determined by pd_logits + the gold bell constants).
    rest_kl_weight adds the SR9-style 99-logit pin on top of the mimic (0 = off).
    init_from loads a checkpoint state dict as the student's starting point.
    rewind_delta_l2_max: dTheta budget for the rewind pool; None = pkl-only
    (round-8 refinement: the #24 champion starts at dT 4.65, right at the
    fresh-run-calibrated edge, so only ProxyKL stays binding there).
    mimic_ce_only: SR18 -- pure per-image CE match. The default kl_ce loss carries a
    disguised rest-pin: KL(z_tgt || sub) pulls all 100 logits toward z_tgt whose 99
    non-label logits are ORIG verbatim, structurally 2x stronger than the CE match,
    which explains the w-invariance and the memorization pull of rounds 10-17.
    """

    pd_target_weight: float = 1.0
    pd_target_mean_ce: float = 1.70
    pd_target_affine: tuple[float, float] | None = None
    pd_target_quantile: bool = False
    rest_kl_weight: float = 0.0
    init_from: str | None = None
    rewind_delta_l2_max: float | None = REWIND_MAX_DELTA_L2
    mimic_ce_only: bool = False


@dataclass(frozen=True)
class GateQConfig(MethodConfig):
    """Champion-trajectory (SalUn+NPO grid) config + per-image gold-bell quantile gate.

    per_image_gate=False reproduces the champion VERBATIM (flat npo gate at
    target_forget_ce, single optimizer, forget+retain in each step, median early-stop).
    per_image_gate=True replaces the flat gate by the gold-bell quantile targets ordered
    by PD ranks (hard images push to 4-5, easy ones stop at 0.1-0.5) -- the landed
    distribution IS the bell by gate construction. gate_offset shifts every gate (the
    SG grid's empirical npo offset is mu ~ gate + 0.9, so -0.9 maps the bell onto itself).
    gate_high_mult > 1 stretches only the gates ABOVE the (post-offset) median --
    the record run landed sigma 2.12 against gate sigma 2.47 because the hardest
    images' ascent efficiency compresses the tail; the mult decompresses it while
    leaving the median untouched. proxy_seed=42 matches the grid's _build_train_proxy
    default so the control reproduces the champion's exact training proxy.
    """

    per_image_gate: bool = False
    gate_offset: float = 0.0
    gate_high_mult: float = 1.0
    proxy_seed: int = 42

def rest_kl_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    rest_weight: float,
) -> torch.Tensor:
    """KL(T=2) over the non-label classes, orig || sub, both renormalized with y masked."""
    keep = torch.ones_like(sub_logits, dtype=torch.bool)
    keep.scatter_(1, labels.unsqueeze(1), False)
    sub_t = sub_logits.float().masked_fill(~keep, float("-inf")) / 2.0
    orig_t = orig_logits.float().masked_fill(~keep, float("-inf")) / 2.0
    sub_rest = F.softmax(sub_t, dim=-1).clamp_min(1e-8)
    orig_rest = F.softmax(orig_t, dim=-1).clamp_min(1e-8)
    kl = (orig_rest * (orig_rest / sub_rest).log()).sum(dim=-1)
    return rest_weight * kl.mean()


def scrub_max_rest_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    diff_ratio: torch.Tensor,
    cfg: ScrubRRestConfig,
) -> torch.Tensor:
    """SR9 ascent loss: unchanged gated PD-weighted NPO on y + rest distillation."""
    return scrub_max_loss(sub_logits, orig_logits, labels, diff_ratio, cfg) + rest_kl_loss(
        sub_logits, orig_logits, labels, cfg.rest_kl_weight
    )


@torch.no_grad()
def affine_calibrated_targets(
    pd_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    affine: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine-recalibrated PD targets: ce_i = a*ce_pd_i + b, 99 rest logits of orig verbatim."""
    a, b = affine
    if a <= 0.0:
        raise ValueError(f"affine slope must be positive (order-preserving), got a={a}")
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    ce_tgt = (a * ce_pd + b).clamp(0.05, 8.0)
    p_tgt = torch.exp(-ce_tgt).clamp(1e-4, 0.95).unsqueeze(1)
    z_orig = orig_logits.float()
    lse_rest = torch.logsumexp(z_orig.scatter(1, labels.unsqueeze(1), -1e9), dim=-1, keepdim=True)
    z_tgt = z_orig.scatter(1, labels.unsqueeze(1), lse_rest + torch.log(p_tgt / (1.0 - p_tgt)))
    return z_tgt, ce_tgt


def _gold_lognormal_params() -> tuple[float, float]:
    """(mu_log, sigma_log) of the gold bell lognormal with the fit's mean/std."""
    var_log = math.log1p((GOLD_BELL_STD / GOLD_BELL_MEAN) ** 2)
    return math.log(GOLD_BELL_MEAN) - var_log / 2.0, var_log ** 0.5


def gold_bell_quantile(ranks: torch.Tensor) -> torch.Tensor:
    """Gold bell CE targets at the given ranks in (0, 1): exp(mu_log + sigma_log * Phi^-1(r))."""
    mu_log, sigma_log = _gold_lognormal_params()
    return torch.exp(mu_log + sigma_log * torch.distributions.Normal(0.0, 1.0).icdf(ranks))


@torch.no_grad()
def quantile_calibrated_targets(
    pd_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-mapped PD targets: pd_ce ranks onto gold-bell lognormal quantiles.

    Monotone by rank, so the per-image ordering (rho 0.72) is strictly preserved and
    the target distribution is the gold bell EXACTLY by construction -- no clamp ever
    binds (lognormal quantiles live in (0, +inf); the p cap below maps to ce >= 0.051,
    far below any reachable rank). The 99 rest logits stay orig verbatim.
    """
    ce_pd = F.cross_entropy(pd_logits.float(), labels, reduction="none")
    n = len(ce_pd)
    ranks = torch.empty_like(ce_pd)
    ranks[ce_pd.argsort()] = (torch.arange(n, dtype=ce_pd.dtype, device=ce_pd.device) + 0.5) / n
    ce_tgt = gold_bell_quantile(ranks)
    p_tgt = torch.exp(-ce_tgt).clamp(1e-4, 0.95).unsqueeze(1)
    z_orig = orig_logits.float()
    lse_rest = torch.logsumexp(z_orig.scatter(1, labels.unsqueeze(1), -1e9), dim=-1, keepdim=True)
    z_tgt = z_orig.scatter(1, labels.unsqueeze(1), lse_rest + torch.log(p_tgt / (1.0 - p_tgt)))
    return z_tgt, ce_tgt

def _load_checkpoint_state(student: torch.nn.Module, ckpt_path: str) -> None:
    """Load a checkpoint state dict into the student (refinement start point)."""
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"init_from checkpoint not found: {ckpt_path}")
    device = next(student.parameters()).device
    student.load_state_dict(torch.load(ckpt_path, map_location=device))


def pd_target_mimic_loss(
    sub_logits: torch.Tensor,
    target_logits: torch.Tensor,
    target_ce: torch.Tensor,
    labels: torch.Tensor,
    cfg: ScrubRPdTargetConfig,
) -> torch.Tensor:
    """Mimic ascent loss toward per-image PD targets.

    Default (kl_ce): KL(T=2) toward z_tgt + 0.5*smooth-L1 on label CE. NOTE the KL
    term pulls all 100 logits toward z_tgt whose 99 non-label logits are ORIG verbatim
    -- a disguised rest-pin, structurally 2x the CE match, which explains the
    w-invariance and memorization pull of rounds 10-17. mimic_ce_only=True (SR18)
    drops it: the pure per-image CE match, 99 logits left free (held by the L2/retain
    anchors); target_logits is then unused but kept in the signature for the shared
    epoch wiring.
    """
    ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
    if cfg.mimic_ce_only:
        return cfg.pd_target_weight * F.smooth_l1_loss(ce_i, target_ce)
    kl_t = compute_kl_at_temperature(target_logits, sub_logits.float(), cfg.temperature).mean()
    return cfg.pd_target_weight * (kl_t + 0.5 * F.smooth_l1_loss(ce_i, target_ce))


def _do_max_rest_epoch(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt_max: torch.optim.Optimizer,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    pseudo: torch.Tensor,
    diff_ratio: torch.Tensor,
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    cfg: ScrubRRestConfig,
) -> None:
    """One SR9 ascent epoch: PD-weighted NPO on y + rest-class distillation pin."""
    student.train()
    dev = next(student.parameters()).device
    for _ in range(cfg.max_steps_per_epoch):
        idx = torch.randint(0, len(f_x), (cfg.batch_size_f,))
        f_batch = augment_training_batch(f_x[idx], mode=cfg.aug_mode)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            sub_f = student(preprocess(f_batch).to(dev))
        orig_b = get_batch_orig_logits(orig_model, f_batch, orig_f[idx], cfg.aug_mode)
        loss = scrub_max_rest_loss(sub_f, orig_b, pseudo[idx].to(dev), diff_ratio[idx].to(dev), cfg)
        _apply_masked_step(student, orig_model, opt_max, loss, masks, absent_mask)


def _do_max_pdtarget_epoch(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt_max: torch.optim.Optimizer,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    pseudo: torch.Tensor,
    z_tgt: torch.Tensor,
    ce_tgt: torch.Tensor,
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    cfg: ScrubRPdTargetConfig,
) -> None:
    """One SR10/SR14 ascent epoch: mimic toward per-image PD targets (+ optional rest pin)."""
    student.train()
    dev = next(student.parameters()).device
    for _ in range(cfg.max_steps_per_epoch):
        idx = torch.randint(0, len(f_x), (cfg.batch_size_f,))
        f_batch = augment_training_batch(f_x[idx], mode=cfg.aug_mode)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            sub_f = student(preprocess(f_batch).to(dev))
        loss = pd_target_mimic_loss(
            sub_f, z_tgt[idx].to(dev), ce_tgt[idx].to(dev), pseudo[idx].to(dev), cfg
        )
        if cfg.rest_kl_weight > 0.0:
            orig_b = get_batch_orig_logits(orig_model, f_batch, orig_f[idx], cfg.aug_mode)
            loss = loss + rest_kl_loss(sub_f, orig_b, pseudo[idx].to(dev), cfg.rest_kl_weight)
        _apply_masked_step(student, orig_model, opt_max, loss, masks, absent_mask)


def _run_alternating_scrub(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: ScrubRConfig,
    run_max_epoch,
) -> tuple[torch.nn.Module, list[dict]]:
    """Shared SCRUB+R loop (max + min + tail + rewind); run_max_epoch runs one ascent epoch.

    With cfg.init_from set (SR14 refinement), the student starts from that checkpoint,
    saliency is computed on it, and the min-step L2 anchor holds the INIT state (not
    orig) so the refinement does not erode its starting point; delta_l2 in snapshots
    stays measured against orig (window semantics unchanged). Without init_from the
    anchor is the orig state, bit-identical to the historical behavior.
    """
    torch.manual_seed(cfg.seed)
    pseudo = orig_f.argmax(dim=-1)
    student = prepare_student(orig_model, cfg.freeze_until_block)
    if getattr(cfg, "init_from", None) is not None:
        _load_checkpoint_state(student, cfg.init_from)
    masks = compute_tempered_saliency(student, f_x, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    anchor_params = {n: p.detach().clone() for n, p in student.named_parameters() if n in masks}
    opt_max = build_optimizer(student, cfg.lr_max, cfg.head_lr_mult)
    opt_min = build_optimizer(student, cfg.lr_min, cfg.head_lr_mult)
    orig_p_lbl = orig_p.argmax(dim=-1)

    snapshots: list[dict] = []
    for epoch in range(cfg.total_epochs):
        if epoch == cfg.max_epochs:
            _scale_optimizer_lr(opt_min, cfg.lr_decay_tail)
        if epoch < cfg.max_epochs:
            run_max_epoch(student, opt_max, masks, absent_mask)
        _do_min_epoch(student, orig_model, opt_min, p_x, orig_p, orig_p_lbl, anchor_params, masks, absent_mask, cfg)
        snap = _eval_epoch_snapshot(epoch, student, f_x, pseudo, p_x, orig_p, orig_params, cfg)
        snap["weights"] = {n: p.detach().cpu().clone() for n, p in student.named_parameters() if n in masks}
        snapshots.append(snap)

    d_l2_max = getattr(cfg, "rewind_delta_l2_max", REWIND_MAX_DELTA_L2)
    best = _select_rewind_snapshot_budgeted(snapshots, cfg.use_rewind, d_l2_max)
    with torch.no_grad():
        for name, param in student.named_parameters():
            if name in best["weights"]:
                param.copy_(best["weights"][name].to(param.device))
    history = [{k: v for k, v in s.items() if k != "weights"} | {"selected": s["epoch"] == best["epoch"]} for s in snapshots]
    return student.eval(), history


def _ensure_pd_logits(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    cfg: ScrubRConfig,
    pd_logits: torch.Tensor | None,
) -> torch.Tensor:
    """Estimate Patch-Dropout gold logits if the caller did not pass them."""
    if pd_logits is not None:
        return pd_logits
    return estimate_patchdrop_gold_logits(orig_model, f_x, drop_prob=0.45, num_views=4, seed=cfg.seed)


def run_scrub_r_rest(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: ScrubRRestConfig,
    pd_logits: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, list[dict]]:
    """SCRUB+R (SR9): alternating loop with rest-class KL distillation in the max step."""
    pd_logits = _ensure_pd_logits(orig_model, f_x, cfg, pd_logits)
    pseudo = orig_f.argmax(dim=-1)
    _, diff_ratio = compute_pd_difficulty_ratio(pd_logits, pseudo)

    def run_max(student, opt_max, masks, absent_mask):
        _do_max_rest_epoch(student, orig_model, opt_max, f_x, orig_f, pseudo, diff_ratio, masks, absent_mask, cfg)

    return _run_alternating_scrub(orig_model, f_x, orig_f, p_x, orig_p, absent_mask, cfg, run_max)


def run_scrub_r_pdtarget(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: ScrubRPdTargetConfig,
    pd_logits: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, list[dict]]:
    """SCRUB+R (SR10/SR13-15): per-image PD-target mimic (+ rest pin) + budgeted rewind."""
    pd_logits = _ensure_pd_logits(orig_model, f_x, cfg, pd_logits)
    pseudo = orig_f.argmax(dim=-1)
    if cfg.pd_target_affine == PD_AFFINE_AUTO:
        raise ValueError("pd_target_affine sentinel must be resolved by run_scrub_r2._run_candidate")
    if cfg.pd_target_quantile:
        z_tgt, ce_tgt = quantile_calibrated_targets(pd_logits, orig_f, pseudo)
    elif cfg.pd_target_affine is None:
        z_tgt, ce_tgt = calibrate_per_image_targets(pd_logits, orig_f, pseudo, cfg.pd_target_mean_ce)
    else:
        z_tgt, ce_tgt = affine_calibrated_targets(pd_logits, orig_f, pseudo, cfg.pd_target_affine)

    def run_max(student, opt_max, masks, absent_mask):
        _do_max_pdtarget_epoch(
            student, orig_model, opt_max, f_x, orig_f, pseudo, z_tgt, ce_tgt, masks, absent_mask, cfg
        )

    return _run_alternating_scrub(orig_model, f_x, orig_f, p_x, orig_p, absent_mask, cfg, run_max)


def run_scrub_r_budgeted(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: ScrubRBudgetedConfig,
    pd_logits: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, list[dict]]:
    """SCRUB+R (SR11): upstream NPO ascent (flat gate via cfg) + budgeted rewind."""
    pd_logits = _ensure_pd_logits(orig_model, f_x, cfg, pd_logits)
    pseudo = orig_f.argmax(dim=-1)
    _, diff_ratio = compute_pd_difficulty_ratio(pd_logits, pseudo)

    def run_max(student, opt_max, masks, absent_m):
        _do_max_epoch(student, orig_model, opt_max, f_x, orig_f, pseudo, diff_ratio, masks, absent_m, cfg)

    return _run_alternating_scrub(orig_model, f_x, orig_f, p_x, orig_p, absent_mask, cfg, run_max)


def npo_per_image_gate_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    temperature: float,
    max_ce_i: torch.Tensor,
) -> torch.Tensor:
    """NPO gated by a PER-IMAGE max_ce tensor; a constant gate is bitwise npo_loss."""
    sub_lp = F.log_softmax(sub_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / temperature, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    per_sample = (2.0 / beta) * F.softplus(beta * (sub_lp - orig_lp))
    with torch.no_grad():
        ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
    return (per_sample * (ce_i < max_ce_i).to(per_sample.dtype)).mean()


def _gateq_step(
    student: torch.nn.Module,
    orig_model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    pseudo: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    absent_mask: torch.Tensor,
    gate_i: torch.Tensor,
    cfg: GateQConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One champion step (salun_npo._salun_step verbatim) with the per-image gate."""
    student.train()
    f_idx = torch.randint(0, len(f_x), (cfg.batch_size,))
    p_idx = torch.randint(0, len(p_x), (cfg.batch_size,))
    f_batch = augment_training_batch(f_x[f_idx], mode=cfg.aug_mode)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        sub_f = student(preprocess(f_batch))
    orig_batch = get_batch_orig_logits(orig_model, f_batch, orig_f[f_idx], cfg.aug_mode)
    loss_f = npo_per_image_gate_loss(
        sub_f, orig_batch, pseudo[f_idx].to(DEVICE), cfg.npo_beta, cfg.temperature, gate_i[f_idx].to(DEVICE)
    )
    loss_r = retain_kl_loss(student, p_x[p_idx], orig_p[p_idx], cfg.temperature)
    total = loss_f + cfg.retain_kl_weight * loss_r + cfg.l2_weight * l2_anchor_loss(student, orig_params)
    opt.zero_grad(set_to_none=True)
    total.backward()
    for n, p in student.named_parameters():
        if p.grad is not None and n in masks:
            p.grad.mul_(masks[n])
    mask_absent_head_grads(student, absent_mask)
    opt.step()
    restore_absent_head_rows(student, orig_model, absent_mask)
    return f_x[f_idx], pseudo[f_idx]



def _apply_gate_transforms(gate_i: torch.Tensor, cfg: GateQConfig) -> torch.Tensor:
    """Offset first, then upper-half spread: mult on gates strictly above the median.

    The spread threshold is the median of the POST-offset gates, so offset=0 configs
    reproduce the bell median exp(mu_log) exactly, and the mult never touches the
    median row or the rank order (rho 0.72 preserved). Identity transforms (offset 0,
    mult 1.0) return the tensor unchanged -- bitwise retro-compatible with round-14.
    """
    if cfg.gate_offset != 0.0:
        gate_i = (gate_i + cfg.gate_offset).clamp_min(0.05)
    if cfg.gate_high_mult != 1.0:
        median = gate_i.median()
        gate_i = torch.where(gate_i > median, gate_i * cfg.gate_high_mult, gate_i)
    return gate_i

def run_scrub_r_gateq(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: GateQConfig,
    pd_logits: torch.Tensor | None = None,
) -> tuple[torch.nn.Module, list[dict]]:
    """Champion-trajectory SalUn+NPO with per-image gold-bell quantile gates (SR20)."""
    torch.manual_seed(cfg.seed)
    student = prepare_student(orig_model, cfg.freeze_until_block)
    pseudo = orig_f.argmax(dim=-1)
    masks = compute_tempered_saliency(student, f_x, pseudo, cfg.salun_keep_ratio, cfg.temperature, seed=cfg.seed)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if n in masks}
    pd_logits = _ensure_pd_logits(orig_model, f_x, cfg, pd_logits)
    if cfg.per_image_gate:
        _, gate_i = quantile_calibrated_targets(pd_logits, orig_f, pseudo)
        gate_i = _apply_gate_transforms(gate_i, cfg)
    else:
        gate_i = torch.full((len(f_x),), float(cfg.target_forget_ce), dtype=torch.float32)
    gate_stop = float(gate_i.median())
    opt = build_optimizer(student, cfg.lr, cfg.head_lr_mult)

    for step in range(1, cfg.steps + 1):
        f_raw, f_lbl = _gateq_step(
            student, orig_model, opt, f_x, orig_f, pseudo, p_x, orig_p,
            orig_params, masks, absent_mask, gate_i, cfg,
        )
        if step % 10 == 0 and should_stop_early(student, f_raw, f_lbl, gate_stop):
            break
    return student.eval(), []
