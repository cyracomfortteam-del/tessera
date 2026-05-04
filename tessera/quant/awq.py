"""AWQ-style activation-aware weight quantization.

Key observation (AWQ, Lin et al. 2023): a small fraction of weight channels are salient,
and saliency correlates with *activation* magnitude, not weight magnitude. Scaling those
input channels up before quantization (and the matching activations down at runtime) gives
them more effective quantization resolution — recovering most of the accuracy lost to naive
round-to-nearest, with zero extra runtime cost because the scale folds into the GEMM.

Identity used:  x @ Wᵀ = (x / s) @ (W · diag(s))ᵀ
We quantize W·diag(s) and divide activations by s at runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tessera import kernels
from tessera.quant.int8 import (
    QuantConfig,
    dequantize_weight,
    quantize_weight,
)


def collect_act_scale(x: torch.Tensor) -> torch.Tensor:
    """Per-input-channel activation scale: mean absolute value over all but the last dim."""
    return x.detach().abs().reshape(-1, x.shape[-1]).mean(dim=0)


def search_awq_scale(
    w: torch.Tensor,
    act_scale: torch.Tensor,
    group_size: int = 128,
    n_grid: int = 20,
) -> torch.Tensor:
    """Grid-search the per-channel scale s = act_scale**ratio that minimizes the
    activation-weighted reconstruction error of dequant(quant(W·diag(s)))."""
    act = act_scale.clamp(min=1e-8)
    best_err = torch.tensor(float("inf"))
    best_s = torch.ones_like(act)
    for i in range(n_grid):
        ratio = i / max(1, n_grid - 1)
        s = act.pow(ratio)
        s = (s / s.mean()).clamp(min=1e-4)
        w_scaled = w * s[None, :]
        q, sc, _ = quantize_weight(w_scaled, group_size, symmetric=True)
        w_deq = dequantize_weight(q, sc, None, group_size) / s[None, :]
        err = ((w - w_deq).abs() * act[None, :]).mean()
        if err < best_err:
            best_err = err
            best_s = s
    return best_s


class AWQLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, cfg: QuantConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = cfg.group_size
        n_groups = in_features // cfg.group_size
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.zeros(out_features, n_groups))
        self.register_buffer("awq_scale", torch.ones(in_features))

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, act_scale: torch.Tensor, cfg: QuantConfig
    ) -> AWQLinear:
        q = cls(linear.in_features, linear.out_features, cfg)
        s = search_awq_scale(linear.weight.data, act_scale, cfg.group_size)
        qw, scale, _ = quantize_weight(linear.weight.data * s[None, :], cfg.group_size, True)
        q.qweight.copy_(qw)
        q.scales.copy_(scale)
        q.awq_scale.copy_(s)
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / self.awq_scale
        return kernels.quant_matmul(x, self.qweight, self.scales, None, self.group_size)
