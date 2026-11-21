"""Strategy E: Low-Rank Adaptation (LoRA FP32) + NPO + Merge back to standard ViT."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from augment import augment_training_batch
from contracts import MethodConfig
from methods.common import mask_absent_head_grads, npo_loss, retain_kl_loss, should_stop_early
from methods.ssd import run_ssd
from task_template import DEVICE, preprocess


class LoRALinear(nn.Module):
    """FP32 LoRA wrapper around a frozen nn.Linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 8) -> None:
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = 2.0
        dev, dt = base.weight.device, torch.float32
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features, device=dev, dtype=dt) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * self.scale
        return self.base(x) + delta.to(x.dtype)

    def merge_to_linear(self) -> nn.Linear:
        merged = copy.deepcopy(self.base)
        with torch.no_grad():
            merged.weight.add_((self.lora_b @ self.lora_a) * self.scale)
        return merged.requires_grad_(False)


def _inject_lora(model: nn.Module, freeze_until_block: int, rank: int) -> None:
    model.requires_grad_(False)
    for idx in range(freeze_until_block, len(model.blocks)):
        blk = model.blocks[idx]
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank)
        blk.attn.proj = LoRALinear(blk.attn.proj, rank)
        blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, rank)
        blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, rank)
    model.head.requires_grad_(True)


def _merge_lora(model: nn.Module, freeze_until_block: int) -> None:
    for idx in range(freeze_until_block, len(model.blocks)):
        blk = model.blocks[idx]
        blk.attn.qkv = blk.attn.qkv.merge_to_linear()
        blk.attn.proj = blk.attn.proj.merge_to_linear()
        blk.mlp.fc1 = blk.mlp.fc1.merge_to_linear()
        blk.mlp.fc2 = blk.mlp.fc2.merge_to_linear()


def run_lora_npo(
    orig_model: nn.Module,
    forget_images: torch.Tensor,
    orig_forget_logits: torch.Tensor,
    proxy_images: torch.Tensor,
    orig_proxy_logits: torch.Tensor,
    absent_mask: torch.Tensor,
    cfg: MethodConfig,
) -> nn.Module:
    """Run NPO + Retain KL inside a low-rank FP32 LoRA subspace, then merge."""
    torch.manual_seed(cfg.seed)
    base = (
        run_ssd(orig_model, forget_images, orig_forget_logits, proxy_images, orig_proxy_logits, absent_mask, cfg)
        if cfg.use_ssd_init
        else orig_model
    )
    student = copy.deepcopy(base).to(DEVICE)
    _inject_lora(student, cfg.freeze_until_block, cfg.lora_rank)
    pseudo = orig_forget_logits.argmax(dim=-1)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr * 3.0)

    for step in range(1, cfg.steps + 1):
        student.train()
        f_idx = torch.randint(0, len(forget_images), (cfg.batch_size,))
        p_idx = torch.randint(0, len(proxy_images), (cfg.batch_size,))
        f_batch = augment_training_batch(forget_images[f_idx], mode=cfg.aug_mode)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            sub_f = student(preprocess(f_batch))
        if cfg.loss_kind == "bad_teacher":
            soft_t = F.softmax(orig_forget_logits[f_idx].float().to(DEVICE) / cfg.bad_teacher_temp, dim=-1)
            loss_f = F.kl_div(F.log_softmax(sub_f.float() / cfg.temperature, dim=-1), soft_t, reduction="batchmean")
        else:
            loss_f = npo_loss(sub_f, orig_forget_logits[f_idx].to(DEVICE), pseudo[f_idx].to(DEVICE), cfg.npo_beta, cfg.temperature)
        loss_r = retain_kl_loss(student, proxy_images[p_idx], orig_proxy_logits[p_idx], cfg.temperature)
        opt.zero_grad(set_to_none=True)
        (loss_f + cfg.retain_kl_weight * loss_r).backward()
        mask_absent_head_grads(student, absent_mask)
        opt.step()
        if step % 10 == 0 and should_stop_early(student, forget_images[f_idx], pseudo[f_idx], cfg.target_forget_ce):
            break
    _merge_lora(student, cfg.freeze_until_block)
    return student.eval()
