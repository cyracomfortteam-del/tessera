"""RMSNorm module — a thin nn.Module wrapper over the dispatched kernel op."""

from __future__ import annotations

import torch
import torch.nn as nn

from tessera import kernels


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernels.rmsnorm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}, eps={self.eps}"
