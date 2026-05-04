"""FP8 (E4M3) quantization, emulated in fp32 so it runs on any device.

E4M3 = 1 sign / 4 exponent (bias 7) / 3 mantissa bits, max normal 448, with subnormals down
to 2^-9. On Hopper/Ada this is a native tensor-core dtype; here we emulate the rounding grid
so the numerics (and the dynamic per-tensor scaling you need around it) can be tested on CPU.
"""

from __future__ import annotations

import torch

E4M3_MAX = 448.0
_EXP_MIN = -6  # smallest normal exponent (unbiased)
_EXP_MAX = 8  # largest exponent before overflow
_MANT_BITS = 3


def _round_to_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round a tensor (already scaled into range) to the E4M3 representable grid."""
    sign = torch.sign(x)
    ax = x.abs().clamp(max=E4M3_MAX)
    is_zero = ax == 0
    safe = torch.where(is_zero, torch.ones_like(ax), ax)

    # Per-element exponent; clamp so subnormals share the smallest-normal step.
    exp = torch.floor(torch.log2(safe)).clamp(min=_EXP_MIN, max=_EXP_MAX)
    step = torch.ldexp(torch.ones_like(exp), (exp - _MANT_BITS).to(torch.int32))
    q = torch.round(ax / step) * step
    q = torch.where(is_zero, torch.zeros_like(q), q)
    return sign * q


def quantize_fp8(
    x: torch.Tensor, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor E4M3 quantization. Returns (dequantized tensor, scale)."""
    if scale is None:
        amax = x.detach().abs().amax().clamp(min=1e-8)
        scale = amax / E4M3_MAX
    q = _round_to_e4m3(x / scale)
    return q * scale, scale


def fp8_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """y = x @ Wᵀ with both operands cast through E4M3 (the Hopper FP8 GEMM pattern)."""
    xq, _ = quantize_fp8(x)
    wq, _ = quantize_fp8(weight)
    return torch.nn.functional.linear(xq, wq)
