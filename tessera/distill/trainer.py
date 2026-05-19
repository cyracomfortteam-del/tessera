"""DistillTrainer — drives student training under FSDP with KD losses + checkpointing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn

from tessera.distill.checkpoint import resume_or_init, save_sharded
from tessera.distill.fsdp import DistGroup, FullyShardedModule, LocalGroup
from tessera.distill.losses import distillation_loss


@dataclass
class DistillConfig:
    lr: float = 3e-4
    weight_decay: float = 0.0
    temperature: float = 2.0
    alpha: float = 0.9  # weight on the soft-target KD term
    max_steps: int = 1000
    ckpt_every: int = 200
    ckpt_dir: str | None = None


class DistillTrainer:
    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        cfg: DistillConfig,
        group: LocalGroup | DistGroup | None = None,
    ):
        self.cfg = cfg
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.fsdp = FullyShardedModule(
            student, group, lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.step = 0
        if cfg.ckpt_dir is not None:
            self.step = resume_or_init(self.fsdp, cfg.ckpt_dir)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor | None = None):
        last: dict[str, float] = {}

        def loss_fn(student: nn.Module) -> torch.Tensor:
            student_logits = student(inputs)
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            loss, metrics = distillation_loss(
                student_logits, teacher_logits, targets, self.cfg.temperature, self.cfg.alpha
            )
            last.update(metrics)
            return loss

        self.fsdp.forward_backward(loss_fn)
        self.fsdp.step()
        self.step += 1
        if self.cfg.ckpt_dir and self.step % self.cfg.ckpt_every == 0:
            save_sharded(self.fsdp, self.step, self.cfg.ckpt_dir)
        return last

    def fit(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor | None]]):
        history = []
        for inputs, targets in batches:
            if self.step >= self.cfg.max_steps:
                break
            history.append(self.train_step(inputs, targets))
        return history
