"""Kernel dispatch layer.

Every fused kernel in `tessera.kernels.triton` has a matching torch implementation in
`tessera.kernels.reference`. The model code never imports a kernel directly; it calls the
ops exposed here, which pick the fastest correct backend for the incoming tensors:

    * Triton  — when the tensor is on CUDA, Triton is importable, and the user hasn't
                disabled it via TESSERA_DISABLE_TRITON=1.
    * torch   — everywhere else (CPU, Apple MPS, ROCm without Triton, CI without a GPU).

This is what lets the whole stack be developed and unit-tested on a laptop while the exact
same call sites run the hand-written kernels in production.
"""

from __future__ import annotations

import os

import torch

try:  # Triton ships Linux+NVIDIA wheels only; absence is normal on macOS/CPU CI.
    import triton  # noqa: F401

    _TRITON_IMPORTABLE = True
except Exception:  # pragma: no cover - depends on the host
    _TRITON_IMPORTABLE = False


def triton_available() -> bool:
    """True when Triton kernels can actually run (importable + a CUDA device present)."""
    if os.environ.get("TESSERA_DISABLE_TRITON", "0") == "1":
        return False
    return _TRITON_IMPORTABLE and torch.cuda.is_available()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_triton(x: torch.Tensor) -> bool:
    """Per-tensor dispatch decision used by the op wrappers below."""
    return x.is_cuda and triton_available()


# ---------------------------------------------------------------------------
# Public ops. Import lazily so that importing tessera.kernels never hard-fails
# on a machine without Triton.
# ---------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if use_triton(x):
        from tessera.kernels.triton.rmsnorm import rmsnorm_triton

        return rmsnorm_triton(x, weight, eps)
    from tessera.kernels.reference import rmsnorm_ref

    return rmsnorm_ref(x, weight, eps)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Fused scaled-dot-product attention. Shapes: (B, H, T, D)."""
    if use_triton(q):
        from tessera.kernels.triton.flash_attention import flash_attention_triton

        return flash_attention_triton(q, k, v, causal=causal, softmax_scale=softmax_scale)
    from tessera.kernels.reference import attention_ref

    return attention_ref(q, k, v, causal=causal, softmax_scale=softmax_scale)


def swiglu(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    if use_triton(x):
        from tessera.kernels.triton.swiglu import swiglu_triton

        return swiglu_triton(x, w_gate, w_up)
    from tessera.kernels.reference import swiglu_ref

    return swiglu_ref(x, w_gate, w_up)


def quant_matmul(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """y = x @ dequant(W). Triton handles the symmetric (zero-free) fast path on CUDA."""
    if zeros is None and use_triton(x):
        from tessera.kernels.triton.quant_matmul import quant_matmul_triton

        return quant_matmul_triton(x, q_weight, scales, group_size=group_size)
    from tessera.kernels.reference import dequant_matmul_ref

    return dequant_matmul_ref(x, q_weight, scales, zeros, group_size)


__all__ = [
    "triton_available",
    "default_device",
    "use_triton",
    "rmsnorm",
    "flash_attention",
    "swiglu",
    "quant_matmul",
]
