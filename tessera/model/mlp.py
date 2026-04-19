"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn

from tessera import kernels
from tessera.config import ModelConfig


class SwiGLU(nn.Module):
    """FFN(x) = (silu(x Wg) * (x Wu)) Wd.

    The gate/up projections are fused by the dispatched `swiglu` op; the down projection
    is a plain linear so quantization can target it independently.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = kernels.swiglu(x, self.w_gate.weight, self.w_up.weight)
        return self.dropout(self.w_down(h))
