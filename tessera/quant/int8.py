"""Weight-only int8 quantization with per-group scales (GPTQ-style packing).

Quantizing only the weights (activations stay fp16/bf16) is the sweet spot for memory-bound
LLM decode: it halves weight bandwidth with negligible quality loss, and the dequant happens
inside the matmul kernel (`tessera.kernels.quant_matmul`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from tessera import kernels


@dataclass
class QuantConfig:
    bits: int = 8
    group_size: int = 128
    symmetric: bool = True


def quantize_weight(
    w: torch.Tensor, group_size: int = 128, symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize an (out, in) weight per group along the input dim.

    Returns (q_weight int8/uint8, scales (out, n_groups), zeros or None).
    `in` must be divisible by `group_size` (true for our model dims; pad otherwise).
    """
    out_features, in_features = w.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by group_size={group_size}")
    n_groups = in_features // group_size
    wg = w.reshape(out_features, n_groups, group_size).to(torch.float32)

    if symmetric:
        absmax = wg.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / 127.0).clamp(min=1e-8)
        q = torch.round(wg / scale).clamp(-127, 127).to(torch.int8)
        zeros = None
    else:
        wmin = wg.amin(dim=-1, keepdim=True)
        wmax = wg.amax(dim=-1, keepdim=True)
        scale = ((wmax - wmin) / 255.0).clamp(min=1e-8)
        zero = torch.round(-wmin / scale)
        q = (torch.round(wg / scale) + zero).clamp(0, 255).to(torch.uint8)
        zeros = zero.squeeze(-1).to(torch.float32)  # (out, n_groups)

    q = q.reshape(out_features, in_features)
    scale = scale.squeeze(-1).to(torch.float32)  # (out, n_groups)
    return q, scale, zeros


def dequantize_weight(
    q: torch.Tensor,
    scale: torch.Tensor,
    zeros: torch.Tensor | None,
    group_size: int = 128,
) -> torch.Tensor:
    out_features, in_features = q.shape
    w = q.to(torch.float32)
    if zeros is not None:
        w = w - zeros.repeat_interleave(group_size, dim=1)[:, :in_features]
    w = w * scale.repeat_interleave(group_size, dim=1)[:, :in_features]
    return w


class QuantLinear(nn.Module):
    """Drop-in int8 replacement for nn.Linear (no bias in this model)."""

    def __init__(self, in_features: int, out_features: int, cfg: QuantConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = cfg.group_size
        self.symmetric = cfg.symmetric
        n_groups = in_features // cfg.group_size
        dtype = torch.int8 if cfg.symmetric else torch.uint8
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=dtype))
        self.register_buffer("scales", torch.zeros(out_features, n_groups))
        if cfg.symmetric:
            self.zeros = None
        else:
            self.register_buffer("zeros", torch.zeros(out_features, n_groups))

    @classmethod
    def from_linear(cls, linear: nn.Linear, cfg: QuantConfig) -> QuantLinear:
        q = cls(linear.in_features, linear.out_features, cfg)
        qw, scale, zeros = quantize_weight(linear.weight.data, cfg.group_size, cfg.symmetric)
        q.qweight.copy_(qw)
        q.scales.copy_(scale)
        if zeros is not None:
            q.zeros.copy_(zeros)
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernels.quant_matmul(
            x, self.qweight, self.scales, getattr(self, "zeros", None), self.group_size
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"group_size={self.group_size}, symmetric={self.symmetric}"
        )
