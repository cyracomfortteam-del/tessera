"""Knowledge-distillation losses.

The student is trained to match a (frozen) teacher. We support the standard soft-target KL
on logits with temperature scaling (Hinton et al. 2015), an optional hard cross-entropy term
against ground-truth labels, and a hidden-state feature-matching term for deeper supervision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_kd(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Temperature-scaled KL(teacher || student), averaged over tokens.

    Multiplying by T^2 keeps gradient magnitudes comparable across temperatures (the soft
    targets' gradients scale as 1/T^2).
    """
    t = temperature
    teacher = F.softmax(teacher_logits / t, dim=-1)
    student = F.log_softmax(student_logits / t, dim=-1)
    kl = F.kl_div(student, teacher, reduction="batchmean") * (t * t)
    return kl


def hard_ce(student_logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
    return F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor | None = None,
    temperature: float = 2.0,
    alpha: float = 0.9,
) -> tuple[torch.Tensor, dict[str, float]]:
    """alpha * soft-target KD + (1 - alpha) * hard CE.

    Returns (loss, metrics) where metrics holds the detached component values for logging.
    """
    kd = soft_target_kd(student_logits, teacher_logits, temperature)
    if targets is not None and alpha < 1.0:
        ce = hard_ce(student_logits, targets)
        loss = alpha * kd + (1.0 - alpha) * ce
        ce_val = float(ce.detach())
    else:
        loss = kd
        ce_val = 0.0
    return loss, {"kd": float(kd.detach()), "ce": ce_val, "loss": float(loss.detach())}


class HiddenStateKD(nn.Module):
    """MSE between projected student hidden states and teacher hidden states.

    A learned linear projection maps the (smaller) student width to the teacher width so the
    two can be compared layer-for-layer; layers are matched by a fixed stride when the depths
    differ.
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(
        self,
        student_hiddens: list[torch.Tensor],
        teacher_hiddens: list[torch.Tensor],
    ) -> torch.Tensor:
        n_s, n_t = len(student_hiddens), len(teacher_hiddens)
        stride = max(1, n_t // n_s)
        loss = student_hiddens[0].new_zeros(())
        count = 0
        for i, sh in enumerate(student_hiddens):
            th = teacher_hiddens[min(i * stride, n_t - 1)]
            loss = loss + F.mse_loss(self.proj(sh), th)
            count += 1
        return loss / max(1, count)
