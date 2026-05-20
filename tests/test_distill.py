"""Distillation: KD losses, sharded-Adam equivalence, checkpointing, trainer convergence."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tessera.config import get_preset
from tessera.distill import (
    DistillConfig,
    DistillTrainer,
    FullyShardedModule,
    HiddenStateKD,
    LocalGroup,
    distillation_loss,
    load_sharded,
    resume_or_init,
    save_sharded,
    soft_target_kd,
)
from tessera.model import Transformer


def _mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))


# -- losses ------------------------------------------------------------------
def test_kd_zero_when_student_equals_teacher():
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 16)
    assert soft_target_kd(logits, logits.clone(), temperature=2.0).abs() < 1e-6


def test_distillation_loss_blends_terms():
    torch.manual_seed(0)
    s = torch.randn(2, 4, 10, requires_grad=True)
    t = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))
    loss, metrics = distillation_loss(s, t, targets, temperature=2.0, alpha=0.5)
    assert {"kd", "ce", "loss"} <= metrics.keys()
    loss.backward()
    assert s.grad is not None


def test_hidden_state_kd_projects_and_matches():
    torch.manual_seed(0)
    kd = HiddenStateKD(student_dim=16, teacher_dim=32)
    sh = [torch.randn(2, 6, 16) for _ in range(2)]
    th = [torch.randn(2, 6, 32) for _ in range(4)]
    loss = kd(sh, th)
    assert loss.ndim == 0 and loss.item() >= 0.0


# -- FSDP --------------------------------------------------------------------
def test_sharded_adam_matches_torch_adam():
    """world_size=1 sharded Adam must equal torch.optim.Adam step-for-step."""
    model = _mlp(0)
    ref = copy.deepcopy(model)
    torch.manual_seed(7)
    x, y = torch.randn(4, 8), torch.randn(4, 4)

    fsdp = FullyShardedModule(model, LocalGroup(), lr=1e-2)
    opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    for _ in range(20):
        fsdp.forward_backward(lambda m: F.mse_loss(m(x), y))
        fsdp.step()
        opt.zero_grad()
        F.mse_loss(ref(x), y).backward()
        opt.step()

    full = fsdp.full_state_dict()
    for name, p in ref.named_parameters():
        torch.testing.assert_close(full[name], p, atol=1e-5, rtol=1e-5)


def test_fsdp_full_state_dict_roundtrips_shapes():
    model = _mlp(0)
    fsdp = FullyShardedModule(model, LocalGroup())
    sd = fsdp.full_state_dict()
    for name, p in model.named_parameters():
        assert sd[name].shape == p.shape


# -- checkpointing -----------------------------------------------------------
def test_checkpoint_save_load_roundtrip(tmp_path):
    model = _mlp(0)
    fsdp = FullyShardedModule(model, LocalGroup(), lr=1e-2)
    torch.manual_seed(7)
    x, y = torch.randn(4, 8), torch.randn(4, 4)
    fsdp.forward_backward(lambda m: F.mse_loss(m(x), y))
    fsdp.step()

    save_sharded(fsdp, step=5, ckpt_dir=tmp_path)
    snapshot = fsdp.p_shard.clone()
    fsdp.p_shard.add_(1.0)  # corrupt in-memory state
    fsdp.t = 999

    step = load_sharded(fsdp, tmp_path / "shard_rank0_step5.pt")
    assert step == 5
    torch.testing.assert_close(fsdp.p_shard, snapshot)
    assert fsdp.t == 1


def test_checkpoint_atomic_leaves_no_tmp(tmp_path):
    fsdp = FullyShardedModule(_mlp(0), LocalGroup())
    save_sharded(fsdp, step=1, ckpt_dir=tmp_path)
    files = list(tmp_path.iterdir())
    assert any(f.name == "shard_rank0_step1.pt" for f in files)
    assert not any(f.name.endswith(".tmp") for f in files)


def test_resume_picks_latest_step(tmp_path):
    fsdp = FullyShardedModule(_mlp(0), LocalGroup())
    for step in (10, 30, 20):
        save_sharded(fsdp, step=step, ckpt_dir=tmp_path)
    assert resume_or_init(fsdp, tmp_path) == 30


# -- trainer -----------------------------------------------------------------
def test_distill_trainer_reduces_kd_loss():
    torch.manual_seed(0)
    teacher = Transformer(get_preset("tessera-debug"))
    torch.manual_seed(1)
    student = Transformer(get_preset("tessera-debug"))

    cfg = DistillConfig(lr=1e-2, alpha=1.0, max_steps=80)
    trainer = DistillTrainer(student, teacher, cfg)
    inputs = torch.randint(0, teacher.cfg.vocab_size, (2, 16))

    history = [trainer.train_step(inputs) for _ in range(80)]
    assert history[-1]["kd"] < history[0]["kd"] * 0.7  # student fits the teacher
