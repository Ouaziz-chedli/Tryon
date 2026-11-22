"""Track H: Real-time algorithmic weight tweaking (H1-H3) & in-weights MLP IF/THEN gates (H4)."""

from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F

from augment import augment_training_batch
from auto_tune_4gpu import _adjust_controller, _probe_step, _sample_probes, _snapshot_score
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
from methods.common import (
    compute_forget_loss,
    get_batch_orig_logits,
    mask_absent_head_grads,
    prepare_student,
    restore_absent_head_rows,
    retain_kl_loss,
)
from methods.salun_npo import compute_tempered_saliency
from task_template import DEVICE, NUM_CLASSES, predict, preprocess


@dataclass(frozen=True)
class TrackHConfig:
    """Hyperparameters for Piste H (H1-H4) wrapping standard MethodConfig."""

    base: MethodConfig
    alpha_proto: float = 0.75
    stuck_boost: float = 1.0
    conductance_period: int = 15
    rollback_tau: float = 0.25
    rollback_rho: float = 0.25
    gamma_lr_mult: float = 3.0
    gamma_clamp_rel: float = 0.5
    stuck_t4_weight: float = 0.0
    weight_norm_floor: float = 1.0
    use_h1_gov: bool = True
    use_h2_damp: bool = True
    use_h3_gamma: bool = True
    min_snapshot_p50: float = 0.18
    use_if_then: bool = False
    if_then_gates: int = 64
    if_then_block: int = 11
    if_then_gain: float = 0.8
    if_then_ret_q: float = 0.995


def _is_gamma_param(name: str) -> bool:
    return name.endswith(".ls1.gamma") or name.endswith(".ls2.gamma")


def build_track_h_optimizer(
    student: torch.nn.Module,
    lr: float,
    head_lr_mult: float,
    gamma_lr_mult: float,
) -> torch.optim.AdamW:
    """Build AdamW with separate lr for body, head, and LayerScale .gamma (weight_decay=0.0)."""
    head_p = [p for n, p in student.named_parameters() if p.requires_grad and n.startswith("head.")]
    gamma_p = [p for n, p in student.named_parameters() if p.requires_grad and _is_gamma_param(n)]
    body_p = [
        p for n, p in student.named_parameters()
        if p.requires_grad and not n.startswith("head.") and not _is_gamma_param(n)
    ]
    groups = [
        {"params": body_p, "lr": lr},
        {"params": head_p, "lr": lr * head_lr_mult},
        {"params": gamma_p, "lr": lr * gamma_lr_mult},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)


def _h3_l2_anchor_loss(
    student: torch.nn.Module,
    orig_params: dict[str, torch.Tensor],
    gamma_l2_scale: float = 0.1,
) -> torch.Tensor:
    """Compute L2 anchor loss with configurable LayerScale .gamma weight."""
    dev = next(student.parameters()).device
    loss = torch.tensor(0.0, device=dev)
    for name, param in student.named_parameters():
        if not param.requires_grad or name not in orig_params:
            continue
        scale = gamma_l2_scale if _is_gamma_param(name) else 1.0
        loss = loss + scale * F.mse_loss(param, orig_params[name], reduction="sum")
    return loss


@torch.no_grad()
def clamp_layerscale_gammas(
    student: torch.nn.Module,
    orig_params: dict[str, torch.Tensor],
    rel_limit: float,
) -> None:
    """Clamp unlocked LayerScale .gamma parameters within orig +/- rel_limit * |orig|."""
    if rel_limit <= 0.0:
        return
    for name, param in student.named_parameters():
        if not param.requires_grad or not _is_gamma_param(name) or name not in orig_params:
            continue
        orig_g = orig_params[name]
        delta = rel_limit * orig_g.abs().clamp_min(1e-4)
        param.data.copy_(torch.clamp(param.data, orig_g - delta, orig_g + delta))


def _t4_stuck_push(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    ce_i: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """T=4 NPO unblocking term for ultra-saturated samples (ce_i < 0.18)."""
    if weight <= 0.0:
        return torch.zeros_like(ce_i)
    sub_lp = F.log_softmax(sub_logits.float() / 4.0, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / 4.0, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    return weight * ((ce_i < 0.18).float() * F.softplus(2.0 * (sub_lp - orig_lp)))


def compute_h1_forget_loss(
    sub_logits: torch.Tensor,
    orig_logits: torch.Tensor,
    labels: torch.Tensor,
    class_ce_ema: torch.Tensor,
    cfg: TrackHConfig,
) -> torch.Tensor:
    """H1: Per-class velocity governor + stuck-sample focus + optional T=4 unblocking."""
    temp, beta, tgt = cfg.base.temperature, cfg.base.npo_beta, max(cfg.base.target_forget_ce, 1e-4)
    sub_lp = F.log_softmax(sub_logits.float() / temp, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    orig_lp = F.log_softmax(orig_logits.float() / temp, dim=-1).gather(1, labels.unsqueeze(1)).squeeze(1)
    npo_i = (2.0 / beta) * F.softplus(beta * (sub_lp - orig_lp))
    with torch.no_grad():
        ce_i = F.cross_entropy(sub_logits.float(), labels, reduction="none")
        counts = torch.bincount(labels, minlength=NUM_CLASSES).float()
        sums = torch.zeros(NUM_CLASSES, device=ce_i.device).scatter_add_(0, labels, ce_i)
        present = counts > 0
        class_ce_ema[present] = 0.85 * class_ce_ema[present] + 0.15 * (sums[present] / counts[present])
        inst_w = torch.clamp(1.0 + cfg.stuck_boost * (ce_i < 0.15).float() - 0.4 * (ce_i / tgt), 0.25, 2.5)
        inst_w = inst_w * (ce_i < tgt).float()
        cls_w = torch.clamp((tgt - class_ce_ema[labels]) / tgt, 0.10, 1.60)
        weights = inst_w * cls_w
        weights = weights / weights.mean().clamp_min(cfg.weight_norm_floor)
    stuck_push = _t4_stuck_push(sub_logits, orig_logits, labels, ce_i, cfg.stuck_t4_weight)
    return (npo_i * weights + stuck_push).mean()


@torch.no_grad()
def project_head_grad_orthogonal(
    student: torch.nn.Module,
    orig_head_weight: torch.Tensor,
    alpha_proto: float,
) -> None:
    """H1: Remove alpha_proto fraction of head.weight.grad parallel to orig_head_weight[c]."""
    grad = student.head.weight.grad
    if grad is None or alpha_proto <= 0.0:
        return
    w0 = orig_head_weight.to(grad.device, dtype=grad.dtype)
    dot = (grad * w0).sum(dim=1, keepdim=True)
    norm2 = (w0 * w0).sum(dim=1, keepdim=True) + 1e-8
    grad.sub_(alpha_proto * (dot / norm2) * w0)


def _probe_grad_abs(
    student: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    is_retain: bool,
    batch_size: int = 64,
    num_batches: int = 2,
) -> dict[str, torch.Tensor]:
    """Accumulate mean |grad| over `num_batches` random batches for forget CE or retain KL."""
    student.eval()
    accum = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in student.named_parameters() if p.requires_grad}
    for _ in range(num_batches):
        idx = torch.randint(0, len(images), (min(batch_size, len(images)),))
        student.zero_grad(set_to_none=True)
        logits = student(preprocess(images[idx]))
        if is_retain:
            loss = compute_kl_at_temperature(targets[idx].to(logits.device), logits, temperature).mean()
        else:
            loss = F.cross_entropy(logits.float() / temperature, targets[idx].to(logits.device))
        loss.backward()
        for n, p in student.named_parameters():
            if p.requires_grad and p.grad is not None:
                accum[n] += p.grad.detach().abs().float() / float(num_batches)
    student.zero_grad(set_to_none=True)
    return accum


def update_conductance_and_rollback(
    student: torch.nn.Module,
    f_x: torch.Tensor,
    pseudo: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    orig_params: dict[str, torch.Tensor],
    init_masks: dict[str, torch.Tensor],
    ema_f: dict[str, torch.Tensor],
    ema_r: dict[str, torch.Tensor],
    cfg: TrackHConfig,
    do_rollback: bool,
) -> dict[str, torch.Tensor]:
    """H2: Update EMA(|grad_f|), EMA(|grad_r|), damp within init_masks, and rollback low-g weights."""
    gf = _probe_grad_abs(student, f_x, pseudo, cfg.base.temperature, is_retain=False, batch_size=cfg.base.batch_size)
    gr = _probe_grad_abs(student, p_x, orig_p, cfg.base.temperature, is_retain=True, batch_size=cfg.base.batch_size)
    masks: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for n, p in student.named_parameters():
            if not p.requires_grad or n not in gf or n not in init_masks:
                continue
            ema_f[n] = gf[n] if n not in ema_f else (0.7 * ema_f[n] + 0.3 * gf[n])
            ema_r[n] = gr[n] if n not in ema_r else (0.7 * ema_r[n] + 0.3 * gr[n])
            active = init_masks[n] > 0
            if not active.any():
                masks[n] = init_masks[n]
                continue
            g = ema_f[n] / (ema_r[n] + 1e-8)
            med_g = g[active].median().clamp_min(1e-8)
            masks[n] = init_masks[n] * torch.clamp(g / med_g, 0.25, 1.0).to(p.dtype)
            if do_rollback and cfg.rollback_rho > 0.0:
                blend = active & (g < cfg.rollback_tau * med_g)
                p.data.copy_(torch.where(blend, (1.0 - cfg.rollback_rho) * p.data + cfg.rollback_rho * orig_params[n], p.data))
    return masks


def _track_h_step(
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
    class_ce_ema: torch.Tensor,
    cfg: TrackHConfig,
    kl_w: float,
    l2_w: float,
) -> None:
    student.train()
    f_idx = torch.randint(0, len(f_x), (cfg.base.batch_size,))
    p_idx = torch.randint(0, len(p_x), (cfg.base.batch_size,))
    f_batch = augment_training_batch(f_x[f_idx], mode=cfg.base.aug_mode)
    dev = next(student.parameters()).device
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
        sub_f = student(preprocess(f_batch).to(dev))
    orig_batch = get_batch_orig_logits(orig_model, f_batch, orig_f[f_idx], cfg.base.aug_mode)
    if cfg.use_h1_gov:
        loss_f = compute_h1_forget_loss(sub_f, orig_batch, pseudo[f_idx].to(dev), class_ce_ema, cfg)
    else:
        loss_f = compute_forget_loss(sub_f, orig_batch, pseudo[f_idx].to(dev), cfg.base)
    loss_r = retain_kl_loss(student, p_x[p_idx], orig_p[p_idx], cfg.base.temperature)
    gamma_l2 = 0.1 if cfg.use_h3_gamma else 1.0
    total = loss_f + kl_w * loss_r + l2_w * _h3_l2_anchor_loss(student, orig_params, gamma_l2_scale=gamma_l2)
    opt.zero_grad(set_to_none=True)
    total.backward()
    for n, p in student.named_parameters():
        if p.grad is not None and n in masks:
            p.grad.mul_(masks[n])
    project_head_grad_orthogonal(student, orig_model.head.weight.detach(), cfg.alpha_proto)
    mask_absent_head_grads(student, absent_mask)
    opt.step()
    if cfg.use_h3_gamma:
        clamp_layerscale_gammas(student, orig_params, cfg.gamma_clamp_rel)
    restore_absent_head_rows(student, orig_model, absent_mask)


@torch.no_grad()
def _collect_block_features(
    student: torch.nn.Module,
    images: torch.Tensor,
    block_idx: int,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture norm2 outputs and mean squared mlp.act energy across `images`."""
    student.eval()
    block = student.blocks[block_idx]
    norm2_chunks, act_sq_sum, total_tok = [], None, 0
    cache: dict[str, torch.Tensor] = {}
    h1 = block.norm2.register_forward_hook(lambda _m, _i, o: cache.__setitem__("n2", o.detach().half().cpu()))
    h2 = block.mlp.act.register_forward_hook(lambda _m, _i, o: cache.__setitem__("act", o.detach().float()))
    try:
        for s in range(0, len(images), batch_size):
            _ = student(preprocess(images[s : s + batch_size]))
            norm2_chunks.append(cache["n2"])
            act_sq = cache["act"].pow(2).sum(dim=(0, 1)).cpu()
            act_sq_sum = act_sq if act_sq_sum is None else (act_sq_sum + act_sq)
            total_tok += cache["act"].shape[0] * cache["act"].shape[1]
    finally:
        h1.remove()
        h2.remove()
    return torch.cat(norm2_chunks, dim=0), act_sq_sum / max(1, total_tok)


@torch.no_grad()
def _kmeans_unit(x_unit: torch.Tensor, k: int, seed: int, iters: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast spherical k-means on unit-normalized vectors [N, D]."""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x_unit), generator=gen)[:k]
    centroids = F.normalize(x_unit[perm].clone(), dim=-1)
    assign = torch.zeros(len(x_unit), dtype=torch.long, device=x_unit.device)
    for _ in range(iters):
        assign = (x_unit @ centroids.T).argmax(dim=-1)
        for j in range(len(centroids)):
            mask = assign == j
            if mask.any():
                centroids[j] = F.normalize(x_unit[mask].mean(dim=0), dim=-1)
    return centroids, assign


@torch.no_grad()
def _select_stubborn_forget(
    student: torch.nn.Module,
    f_x: torch.Tensor,
    pseudo: torch.Tensor,
    min_count: int,
) -> torch.Tensor:
    """Select stubborn low-CE forget indices (<0.35, fallback <1.0, fallback lowest-CE)."""
    logits = predict(student, f_x, batch_size=128)
    ce_i = F.cross_entropy(logits.float(), pseudo.to(logits.device), reduction="none").cpu()
    idx = torch.where(ce_i < 0.35)[0]
    if len(idx) >= min_count:
        return idx
    idx = torch.where(ce_i < 1.0)[0]
    if len(idx) >= min_count:
        return idx
    return torch.topk(ce_i, k=min(max(min_count * 2, 128), len(ce_i)), largest=False).indices


@torch.no_grad()
def _program_single_gate(
    block: torch.nn.Module,
    head_w: torch.Tensor,
    norm_w: torch.Tensor,
    k_idx: int,
    c_k: torch.Tensor,
    y_k: int,
    norm2_ret_flat: torch.Tensor,
    norm2_fg_cluster: torch.Tensor,
    cfg: TrackHConfig,
) -> bool:
    """Program one dormant MLP neuron (fc1 row + bias, fc2 column) with bounded scale."""
    d_ret = norm2_ret_flat @ c_k
    d_fg = norm2_fg_cluster @ c_k
    q_ret = float(torch.quantile(d_ret, cfg.if_then_ret_q).item())
    q_fg = float(torch.quantile(d_fg, 0.25).item())
    tau_k = max(q_ret + 0.05, 0.5 * (q_fg + q_ret))
    if float(d_fg.mean().item()) <= tau_k + 0.15:
        return False
    s = min(1.2, 2.5 / float(d_ret.std().clamp_min(0.5).item()))
    block.mlp.fc1.weight.data[k_idx] = (s * c_k).to(block.mlp.fc1.weight.device, dtype=block.mlp.fc1.weight.dtype)
    block.mlp.fc1.bias.data[k_idx] = torch.tensor(-s * tau_k, device=block.mlp.fc1.bias.device, dtype=block.mlp.fc1.bias.dtype)
    d_tgt = -F.normalize(head_w[y_k].float() * norm_w.float(), dim=0).to(c_k.device)
    gamma = block.ls2.gamma.detach().float().to(c_k.device) if hasattr(block, "ls2") else torch.ones_like(c_k)
    sgn = torch.where(gamma != 0, torch.sign(gamma), torch.ones_like(gamma))
    col = cfg.if_then_gain * (d_tgt * sgn)
    col_n = float(col.norm().clamp_min(1e-8).item())
    if col_n > 0.35:
        col = col * (0.35 / col_n)
    block.mlp.fc2.weight.data[:, k_idx] = col.to(block.mlp.fc2.weight.device, dtype=block.mlp.fc2.weight.dtype)
    return True


def _snapshot_block_mlp(block: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return (
        block.mlp.fc1.weight.detach().clone(),
        block.mlp.fc1.bias.detach().clone(),
        block.mlp.fc2.weight.detach().clone(),
    )


def _restore_block_mlp(block: torch.nn.Module, snap: tuple[torch.Tensor, ...]) -> None:
    block.mlp.fc1.weight.data.copy_(snap[0])
    block.mlp.fc1.bias.data.copy_(snap[1])
    block.mlp.fc2.weight.data.copy_(snap[2])


@torch.no_grad()
def compile_if_then_mlp_gates(
    student: torch.nn.Module,
    f_x: torch.Tensor,
    pseudo: torch.Tensor,
    p_x: torch.Tensor,
    cfg: TrackHConfig,
    orig_p: torch.Tensor | None = None,
) -> int:
    """H4: Compile IF/THEN gates in dormant MLP neurons with ProxyKL rollback guard."""
    block = student.blocks[cfg.if_then_block]
    snap = _snapshot_block_mlp(block)
    kl_before = float(compute_kl_at_temperature(orig_p.cpu(), predict(student, p_x, batch_size=128), 2.0).mean().item()) if orig_p is not None else 0.0
    m_gates = min(cfg.if_then_gates, block.mlp.fc1.out_features)
    norm2_ret, act_ret_sq = _collect_block_features(student, p_x, cfg.if_then_block)
    energy = act_ret_sq * block.mlp.fc2.weight.detach().float().cpu().pow(2).sum(dim=0)
    gate_neurons = torch.topk(energy, k=m_gates, largest=False).indices.tolist()
    stub_idx = _select_stubborn_forget(student, f_x, pseudo, m_gates)
    fg_cls = _collect_block_features(student, f_x[stub_idx], cfg.if_then_block)[0][:, 0, :].float()
    centroids, assign = _kmeans_unit(F.normalize(fg_cls, dim=-1), len(gate_neurons), cfg.base.seed)
    ret_flat, y_stub = norm2_ret.reshape(-1, norm2_ret.shape[-1]).float(), pseudo[stub_idx].cpu()
    head_w, norm_w = student.head.weight.detach(), student.norm.weight.detach()
    compiled = sum(
        1 for j, k_idx in enumerate(gate_neurons)
        if (assign == j).any() and _program_single_gate(
            block, head_w, norm_w, k_idx, centroids[j],
            int(torch.bincount(y_stub[assign == j], minlength=NUM_CLASSES).argmax().item()),
            ret_flat, fg_cls[assign == j], cfg,
        )
    )
    if compiled > 0 and orig_p is not None:
        kl_after = float(compute_kl_at_temperature(orig_p.cpu(), predict(student, p_x, batch_size=128), 2.0).mean().item())
        if kl_after - kl_before > 0.0015:
            _restore_block_mlp(block, snap)
            return 0
    return compiled


def run_track_h(
    orig_model: torch.nn.Module,
    f_x: torch.Tensor,
    orig_f: torch.Tensor,
    p_x: torch.Tensor,
    orig_p: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: TrackHConfig,
) -> torch.nn.Module:
    """Run Piste H (H1+H2+H3 + optional H4) closed-loop selective unlearning."""
    torch.manual_seed(cfg.base.seed)
    student = prepare_student(orig_model, cfg.base.freeze_until_block)
    pseudo = orig_f.argmax(dim=-1)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters()}
    orig_params = {n: orig_params[n] for n, p in student.named_parameters() if p.requires_grad}
    init_masks = compute_tempered_saliency(student, f_x, pseudo, cfg.base.salun_keep_ratio, cfg.base.temperature, seed=cfg.base.seed)
    masks = {k: v.clone() for k, v in init_masks.items()}
    ema_f, ema_r = {}, {}
    opt = build_track_h_optimizer(student, cfg.base.lr, cfg.base.head_lr_mult, cfg.gamma_lr_mult)
    kl_w, l2_w = cfg.base.retain_kl_weight, cfg.base.l2_weight
    class_ce_ema = torch.zeros(NUM_CLASSES, device=next(student.parameters()).device)
    f_probe, pseudo_probe, p_probe, orig_p_probe = _sample_probes(f_x, pseudo, p_x, orig_p, seed=cfg.base.seed)
    best_score, best_sd = -1e9, None
    fb_score, fb_sd = -1e9, None
    min_p50 = cfg.min_snapshot_p50
    upper_p50 = max(cfg.base.target_ce_upper, cfg.base.target_forget_ce * 1.45)

    for step in range(1, cfg.base.steps + 1):
        _track_h_step(student, orig_model, opt, f_x, orig_f, pseudo, p_x, orig_p, orig_params, masks, absent_mask, class_ce_ema, cfg, kl_w, l2_w)
        if cfg.use_h2_damp and step % cfg.conductance_period == 0:
            masks = update_conductance_and_rollback(student, f_x, pseudo, p_x, orig_p, orig_params, init_masks, ema_f, ema_r, cfg, do_rollback=True)
        if step % 15 != 0:
            continue
        ce_mean, ce_std, p50, f95, pkl, d_l2 = _probe_step(student, f_probe, pseudo_probe, p_probe, orig_p_probe, orig_params)
        kl_w, l2_w = _adjust_controller(opt, kl_w, l2_w, pkl, d_l2, p50)
        snap = _snapshot_score(ce_mean, ce_std, p50, f95, pkl, d_l2, target_p50=cfg.base.target_forget_ce)
        if min_p50 <= p50 <= upper_p50 + 0.25 and pkl <= 0.065 and snap > best_score:
            best_score, best_sd = snap, {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        elif 0.18 <= p50 <= upper_p50 + 0.25 and pkl <= 0.065 and snap > fb_score:
            fb_score, fb_sd = snap, {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        if p50 > upper_p50 + 0.25 and best_sd is not None:
            break
    chosen_sd = best_sd if best_sd is not None else fb_sd
    if chosen_sd is not None:
        student.load_state_dict(chosen_sd)
    if cfg.use_if_then:
        compile_if_then_mlp_gates(student, f_x, pseudo, p_x, cfg, orig_p=orig_p)
    return student.eval()
