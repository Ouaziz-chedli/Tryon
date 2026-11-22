"""Unit tests for Track 3 Piste H (H1-H4) on synthetic ViT-like modules."""

import torch
import torch.nn as nn

from contracts import MethodConfig
from task_template import DEVICE
from methods.track_h import (
    TrackHConfig,
    build_track_h_optimizer,
    clamp_layerscale_gammas,
    compile_if_then_mlp_gates,
    compute_h1_forget_loss,
    project_head_grad_orthogonal,
    update_conductance_and_rollback,
)


class _DummyBlock(nn.Module):
    def __init__(self, dim: int = 16, hidden: int = 32) -> None:
        super().__init__()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, hidden)
        self.mlp.act = nn.GELU()
        self.mlp.fc2 = nn.Linear(hidden, dim)
        self.ls2 = nn.Module()
        self.ls2.gamma = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp.fc2(self.mlp.act(self.mlp.fc1(self.norm2(x))))
        return x + self.ls2.gamma * h


class _DummyViT(nn.Module):
    def __init__(self, dim: int = 16, num_classes: int = 100) -> None:
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


def test_h1_loss_and_orthogonal_projection() -> None:
    cfg0 = TrackHConfig(base=MethodConfig(name="t", strategy="track_h", target_forget_ce=1.2), alpha_proto=1.0, stuck_t4_weight=0.0)
    cfg4 = TrackHConfig(base=MethodConfig(name="t", strategy="track_h", target_forget_ce=1.2), alpha_proto=1.0, stuck_t4_weight=0.55)
    sub = torch.full((8, 100), -5.0)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    sub[torch.arange(8), labels] = 5.0
    sub.requires_grad_(True)
    orig = sub.detach().clone()
    ema0, ema4 = torch.zeros(100), torch.zeros(100)
    loss0 = compute_h1_forget_loss(sub, orig, labels, ema0, cfg0)
    loss4 = compute_h1_forget_loss(sub, orig, labels, ema4, cfg4)
    assert loss0.isfinite() and ema0[0].item() > 0.0 and float(loss4.item()) > float(loss0.item())

    vit = _DummyViT()
    w0 = vit.head.weight.detach().clone()
    vit.head.weight.grad = w0.clone() * 2.0
    project_head_grad_orthogonal(vit, w0, alpha_proto=1.0)
    assert float(vit.head.weight.grad.abs().max().item()) < 1e-5


def test_h2_conductance_and_h3_gamma_clamp() -> None:
    vit = _DummyViT().to(DEVICE)
    for n, p in vit.named_parameters():
        if not n.startswith("blocks.11") and not n.startswith("head"):
            p.requires_grad_(False)
    orig_params = {n: p.detach().clone() for n, p in vit.named_parameters() if p.requires_grad}
    opt = build_track_h_optimizer(vit, lr=1e-4, head_lr_mult=0.5, gamma_lr_mult=3.0)
    assert len(opt.param_groups) == 3 and abs(opt.param_groups[2]["lr"] - 3e-4) < 1e-9

    vit.blocks[11].ls2.gamma.data.fill_(1.0)
    clamp_layerscale_gammas(vit, orig_params, rel_limit=0.0)
    assert float(vit.blocks[11].ls2.gamma.abs().max().item()) == 1.0
    clamp_layerscale_gammas(vit, orig_params, rel_limit=0.5)
    assert float(vit.blocks[11].ls2.gamma.abs().max().item()) <= 1.5e-4 + 1e-7

    cfg = TrackHConfig(base=MethodConfig(name="t", strategy="track_h", salun_keep_ratio=0.5, batch_size=4))
    f_x = torch.randint(0, 255, (8, 224, 224, 3), dtype=torch.uint8)
    p_x = torch.randint(0, 255, (8, 224, 224, 3), dtype=torch.uint8)
    pseudo = torch.zeros(8, dtype=torch.long)
    orig_p = torch.zeros(8, 100)
    ema_f, ema_r = {}, {}
    init_masks = {n: (torch.rand_like(p) > 0.5).to(p.dtype) for n, p in orig_params.items()}
    masks = update_conductance_and_rollback(vit, f_x, pseudo, p_x, orig_p, orig_params, init_masks, ema_f, ema_r, cfg, do_rollback=True)
    assert "head.weight" in masks
    for n, m in masks.items():
        assert torch.all((m > 0) <= (init_masks[n] > 0))


def test_h4_if_then_compilation_runs_cleanly() -> None:
    vit = _DummyViT().to(DEVICE)
    cfg = TrackHConfig(base=MethodConfig(name="t", strategy="track_h"), use_if_then=True, if_then_gates=4, if_then_block=11, if_then_gain=0.35, if_then_ret_q=0.999)
    f_x = torch.randint(0, 255, (12, 224, 224, 3), dtype=torch.uint8)
    p_x = torch.randint(0, 255, (12, 224, 224, 3), dtype=torch.uint8)
    pseudo = torch.arange(12) % 3
    orig_p = torch.zeros(12, 100)
    fc1_before = vit.blocks[11].mlp.fc1.weight.detach().clone()
    compiled = compile_if_then_mlp_gates(vit, f_x, pseudo, p_x, cfg, orig_p=orig_p)
    assert 0 <= compiled <= 4
    if compiled > 0:
        assert float((vit.blocks[11].mlp.fc2.weight.norm(dim=0)).max().item()) <= 10.0


from run_track_h import round4_candidates, round5_candidates, round6_candidates, round7_candidates


def test_round4_ablation_isolation() -> None:
    cands = {c.base.name: c for c in round4_candidates()}
    assert set(cands) == {"ABL_0_baseline_AT", "ABL_1_plus_H1_ortho", "ABL_2_plus_H1_gov", "ABL_3_plus_H2_damp"}
    a0 = cands["ABL_0_baseline_AT"]
    assert not a0.use_h1_gov and not a0.use_h2_damp and not a0.use_h3_gamma and a0.alpha_proto == 0.0
    assert cands["ABL_1_plus_H1_ortho"].alpha_proto == 0.25 and not cands["ABL_1_plus_H1_ortho"].use_h1_gov
    assert cands["ABL_2_plus_H1_gov"].use_h1_gov and not cands["ABL_2_plus_H1_gov"].use_h2_damp
    assert cands["ABL_3_plus_H2_damp"].use_h2_damp and cands["ABL_3_plus_H2_damp"].rollback_rho == 0.0
    r5 = {c.base.name: c for c in round5_candidates()}
    assert set(r5) == {"R5_h1gov_fixed", "R5_h1gov_ortho25"}
    assert r5["R5_h1gov_fixed"].alpha_proto == 0.0 and r5["R5_h1gov_ortho25"].alpha_proto == 0.25
    r6 = {c.base.name: c for c in round6_candidates()}
    assert set(r6) == {"R6_h1gov_ce125", "R6_h1gov_ortho25_ce125"}
    assert r6["R6_h1gov_ce125"].base.target_forget_ce == 1.25 and r6["R6_h1gov_ortho25_ce125"].alpha_proto == 0.25
    r7 = {c.base.name: c for c in round7_candidates()}
    assert set(r7) == {"R7_1_wnf060_ce105", "R7_2_wnf055_ce110_kl19", "R7_3_wnf100_kl17_ce120"}
    assert r7["R7_1_wnf060_ce105"].weight_norm_floor == 0.60 and r7["R7_2_wnf055_ce110_kl19"].weight_norm_floor == 0.55


if __name__ == "__main__":
    test_h1_loss_and_orthogonal_projection()
    test_h2_conductance_and_h3_gamma_clamp()
    test_h4_if_then_compilation_runs_cleanly()
    test_round4_ablation_isolation()
    print("TRACK_H_TESTS_OK")
