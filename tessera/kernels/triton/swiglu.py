"""Fused SwiGLU forward: a single tiled GEMM that fuses both projections + activation.

A naive SwiGLU does two GEMMs (gate, up) writing two (M, hidden) tensors to HBM, reads
them back, applies silu(gate)*up, and writes a third. This kernel keeps both accumulators
in registers, applies the activation in the epilogue, and writes the hidden tensor *once* —
cutting the activation's HBM traffic by ~3x.

The block-swizzled program ordering (GROUP_M) improves L2 reuse of the weight tiles, the
same trick used in the Triton matmul tutorial.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tessera.kernels.reference import swiglu_ref


def _configs():
    out = []
    for bm in (64, 128):
        for bn in (64, 128):
            for bk in (32, 64):
                for w in (4, 8):
                    out.append(
                        triton.Config(
                            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                            num_warps=w,
                            num_stages=3,
                        )
                    )
    return out


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _swiglu_kernel(
    x_ptr, wg_ptr, wu_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Weights are (N, K) row-major; load transposed tiles B[k, n] = w[n, k].
    wg_ptrs = wg_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    wu_ptrs = wu_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        wg = tl.load(wg_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        wu = tl.load(wu_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc_g += tl.dot(x, wg)
        acc_u += tl.dot(x, wu)
        x_ptrs += BLOCK_K * stride_xk
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk

    silu = acc_g * tl.sigmoid(acc_g)
    out = silu * acc_u

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, out.to(y_ptr.dtype.element_ty), mask=y_mask)


def _swiglu_fwd(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    *lead, k = x.shape
    x2d = x.reshape(-1, k).contiguous()
    m = x2d.shape[0]
    n = w_gate.shape[0]
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _swiglu_kernel[grid](
        x2d, w_gate, w_up, y,
        m, n, k,
        x2d.stride(0), x2d.stride(1),
        w_gate.stride(0), w_gate.stride(1),
        y.stride(0), y.stride(1),
    )
    return y.reshape(*lead, n)


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate, w_up):
        ctx.save_for_backward(x, w_gate, w_up)
        return _swiglu_fwd(x, w_gate, w_up)

    @staticmethod
    def backward(ctx, grad_out):
        x, w_gate, w_up = ctx.saved_tensors
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            gr = w_gate.detach().requires_grad_(True)
            ur = w_up.detach().requires_grad_(True)
            y = swiglu_ref(xr, gr, ur)
            gx, gg, gu = torch.autograd.grad(y, (xr, gr, ur), grad_out)
        return gx, gg, gu


def swiglu_triton(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    return _SwiGLUFn.apply(x, w_gate, w_up)
