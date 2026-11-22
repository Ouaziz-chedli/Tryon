"""Unit tests for Track 3 Piste I (Erase-Then-Heal) on synthetic ViT-like modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from contracts import MethodConfig
from methods.lora_unlearn import LoRALinear
from methods.track_i import (
    TrackIConfig,
    _collect_lora_modules,
    _collect_orig_lora_weights,
    compute_ceiling_heal_loss,
    compute_dark_liaison_loss,
    compute_extreme_forget_loss,
    compute_floor_guard_loss,
    compute_merged_lora_l2_loss,
    inject_frozen_head_lora,
    merge_track_i_lora,
    run_track_i,
)
from task_template import DEVICE, NUM_CLASSES


class _DummyAttn(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, _k, _v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q)


class _DummyBlock(nn.Module):
    def __init__(self, dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _DummyAttn(dim)
        self.ls1 = nn.Module()
        self.ls1.gamma = nn.Parameter(torch.full((dim,), 1e-4))
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, hidden)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(hidden, dim)
        self.ls2 = nn.Module()
        self.ls2.gamma = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1.gamma * self.attn(self.norm1(x))
        return x + self.ls2.gamma * self.mlp.fc2(self.mlp.act(self.mlp.fc1(self.norm2(x))))


class _DummyViT(nn.Module):
    def __init__(self, dim: int = 16, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.blocks = nn.ModuleList([_DummyBlock(dim, dim * 2) for _ in range(12)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tok = blk(tok)
        return self.head(self.norm(tok[:, 0]))


def test_lora_inject_frozen_head_and_merge() -> None:
    vit = _DummyViT().to(DEVICE)
    inject_frozen_head_lora(vit, freeze_until_block=10, rank=4)
    assert not vit.head.weight.requires_grad and not vit.head.bias.requires_grad
    assert isinstance(vit.blocks[10].attn.qkv, LoRALinear)
    assert not isinstance(vit.blocks[9].attn.qkv, LoRALinear)
    trainable = [n for n, p in vit.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)

    x = torch.randn(2, 3, 224, 224, device=DEVICE)
    vit.blocks[11].mlp.fc1.lora_b.data.normal_(0.0, 0.1)
    out_before = vit(x).detach()
    merge_track_i_lora(vit, freeze_until_block=10)
    assert isinstance(vit.blocks[10].attn.qkv, nn.Linear)
    out_after = vit(x).detach()
    assert float((out_before - out_after).abs().max().item()) < 1e-5


def test_dark_liaison_and_ceiling_heal_losses() -> None:
    labels = torch.tensor([3, 7, 12, 25])
    orig = torch.randn(4, NUM_CLASSES)
    sub = orig.clone().requires_grad_(True)
    loss_d = compute_dark_liaison_loss(sub, orig, labels, temperature=3.0)
    loss_d.backward()
    assert float(loss_d.item()) < 1e-6
    assert float(sub.grad[torch.arange(4), labels].abs().max().item()) == 0.0

    sub2 = torch.zeros(2, NUM_CLASSES)
    sub2[0, 3] = 10.0  # low CE_0 << 1.50
    sub2[1, 7] = -5.0  # high CE_1 >> 1.50
    sub2.requires_grad_(True)
    loss_c = compute_ceiling_heal_loss(sub2, torch.tensor([3, 7]), ce_ceiling=1.50)
    loss_c.backward()
    assert float(loss_c.item()) > 0.0
    assert float(sub2.grad[0].abs().max().item()) == 0.0
    assert float(sub2.grad[1, 7].item()) < 0.0

    sub3 = sub2.detach().clone().requires_grad_(True)
    loss_fl = compute_floor_guard_loss(sub3, sub2.detach(), torch.tensor([3, 7]), ce_floor=0.85)
    loss_fl.backward()
    assert float(loss_fl.item()) > 0.0
    assert float(sub3.grad[1].abs().max().item()) == 0.0
    assert float(sub3.grad[0, 3].item()) > 0.0


def test_end_to_end_phase1_and_phase2_cpu() -> None:
    vit = _DummyViT().to(DEVICE).eval()
    f_x = torch.randint(0, 255, (16, 224, 224, 3), dtype=torch.uint8)
    p_x = torch.randint(0, 255, (16, 224, 224, 3), dtype=torch.uint8)
    orig_f = torch.randn(16, NUM_CLASSES)
    orig_p = torch.randn(16, NUM_CLASSES)
    absent = torch.ones(NUM_CLASSES, dtype=torch.bool)
    absent[:5] = False
    cfg = TrackIConfig(
        base=MethodConfig(name="ti_test", strategy="track_i", steps=4, batch_size=4, lr=1e-4, freeze_until_block=10),
        heal_steps=4,
        heal_rank=4,
        heal_freeze_until_block=10,
    )
    ema = torch.zeros(NUM_CLASSES, device=DEVICE)
    l_f = compute_extreme_forget_loss(orig_f.to(DEVICE).requires_grad_(True), orig_f.to(DEVICE), orig_f.argmax(-1).to(DEVICE), ema, cfg)
    assert l_f.isfinite()

    healed, p1 = run_track_i(vit, f_x, orig_f, p_x, orig_p, absent, cfg, return_phase1=True)
    assert isinstance(healed.blocks[10].attn.qkv, nn.Linear)
    assert torch.allclose(healed.head.weight.cpu()[absent], vit.head.weight.cpu()[absent])
    assert set(healed.state_dict().keys()) == set(vit.state_dict().keys())
    assert set(p1.state_dict().keys()) == set(vit.state_dict().keys())
    assert float(F.mse_loss(healed.head.weight.cpu(), p1.head.weight.cpu()).item()) == 0.0


def test_merged_lora_l2_and_phase1_only() -> None:
    orig = _DummyViT().to(DEVICE).eval()
    vit = _DummyViT().to(DEVICE).eval()
    orig_w = _collect_orig_lora_weights(orig, freeze_until_block=10, dev=torch.device(DEVICE))
    inject_frozen_head_lora(vit, freeze_until_block=10, rank=4)
    lora_mods = _collect_lora_modules(vit, freeze_until_block=10)
    loss_before = compute_merged_lora_l2_loss(lora_mods, orig_w)
    loss_before.backward()
    assert any(p.grad is not None and float(p.grad.abs().max()) > 0.0 for p in vit.parameters() if p.requires_grad)

    f_x = torch.randint(0, 255, (12, 224, 224, 3), dtype=torch.uint8)
    p_x = torch.randint(0, 255, (12, 224, 224, 3), dtype=torch.uint8)
    orig_f, orig_p = torch.randn(12, NUM_CLASSES), torch.randn(12, NUM_CLASSES)
    absent = torch.ones(NUM_CLASSES, dtype=torch.bool)
    absent[:4] = False
    cfg = TrackIConfig(base=MethodConfig(name="p1only", strategy="track_i", steps=3, batch_size=4, freeze_until_block=10), heal_steps=0)
    out, p1 = run_track_i(orig, f_x, orig_f, p_x, orig_p, absent, cfg, return_phase1=True)
    assert out is p1 and isinstance(out.blocks[10].attn.qkv, nn.Linear)


def test_phase1_max_dtheta_early_stop() -> None:
    orig = _DummyViT().to(DEVICE).eval()
    f_x = torch.randint(0, 255, (12, 224, 224, 3), dtype=torch.uint8)
    p_x = torch.randint(0, 255, (12, 224, 224, 3), dtype=torch.uint8)
    orig_f, orig_p = torch.randn(12, NUM_CLASSES), torch.randn(12, NUM_CLASSES)
    absent = torch.ones(NUM_CLASSES, dtype=torch.bool)
    absent[:4] = False
    cfg_cap = TrackIConfig(
        base=MethodConfig(name="cap", strategy="track_i", steps=15, batch_size=4, lr=2e-4, freeze_until_block=10, target_forget_ce=10.0),
        heal_steps=0, phase1_probe_every=1, phase1_max_ce=100.0, phase1_max_dtheta=0.01,
    )
    cfg_free = TrackIConfig(
        base=MethodConfig(name="free", strategy="track_i", steps=15, batch_size=4, lr=2e-4, freeze_until_block=10, target_forget_ce=10.0),
        heal_steps=0, phase1_probe_every=1, phase1_max_ce=100.0, phase1_max_dtheta=1e9,
    )
    out_cap = run_track_i(orig, f_x, orig_f, p_x, orig_p, absent, cfg_cap)
    out_free = run_track_i(orig, f_x, orig_f, p_x, orig_p, absent, cfg_free)
    d_cap = sum((p.detach() - q.detach()).pow(2).sum().item() for p, q in zip(out_cap.parameters(), orig.parameters())) ** 0.5
    d_free = sum((p.detach() - q.detach()).pow(2).sum().item() for p, q in zip(out_free.parameters(), orig.parameters())) ** 0.5
    assert d_cap < d_free and d_cap <= 0.012, f"d_cap={d_cap}, d_free={d_free}"


if __name__ == "__main__":
    test_lora_inject_frozen_head_and_merge()
    test_dark_liaison_and_ceiling_heal_losses()
    test_end_to_end_phase1_and_phase2_cpu()
    test_merged_lora_l2_and_phase1_only()
    test_phase1_max_dtheta_early_stop()
    print("TRACK_I_TESTS_OK")
