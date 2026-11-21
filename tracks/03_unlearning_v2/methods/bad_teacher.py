"""Strategy B: Asymmetric SCRUB / Bad-Teacher distillation without retain set."""

import torch
import torch.nn.functional as F

from augment import augment_training_batch
from contracts import MethodConfig
from eval_harness import compute_kl_at_temperature
from methods.common import (
    l2_anchor_loss,
    mask_absent_head_grads,
    prepare_student,
    retain_kl_loss,
    should_stop_early,
)
from task_template import DEVICE, preprocess


def run_bad_teacher(
    orig_model: torch.nn.Module,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
) -> torch.nn.Module:
    """Distill forget samples toward softened orig logits while anchoring retain proxy."""
    torch.manual_seed(cfg.seed)
    student = prepare_student(orig_model, cfg.freeze_until_block)
    pseudo = orig_forget_logits.argmax(dim=-1)
    orig_params = {n: p.detach().clone() for n, p in orig_model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr)

    for step in range(1, cfg.steps + 1):
        student.train()
        f_idx = torch.randint(0, len(forget_images), (cfg.batch_size,))
        p_idx = torch.randint(0, len(proxy_images), (cfg.batch_size,))
        f_batch = augment_training_batch(forget_images[f_idx], mode=cfg.aug_mode)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            sub_f = student(preprocess(f_batch))
        soft_target = F.softmax(orig_forget_logits[f_idx].float().to(DEVICE) / cfg.bad_teacher_temp, dim=-1)
        loss_f = F.kl_div(F.log_softmax(sub_f.float() / cfg.temperature, dim=-1), soft_target, reduction="batchmean")
        loss_r = retain_kl_loss(student, proxy_images[p_idx], orig_proxy_logits[p_idx], cfg.temperature)
        total = loss_f + cfg.retain_kl_weight * loss_r + cfg.l2_weight * l2_anchor_loss(student, orig_params)
        opt.zero_grad(set_to_none=True)
        total.backward()
        mask_absent_head_grads(student, absent_mask)
        opt.step()
        if step % 10 == 0 and should_stop_early(student, forget_images[f_idx], pseudo[f_idx], cfg.target_forget_ce):
            break
    return student.eval()
