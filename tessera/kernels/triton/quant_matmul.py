"""Weight-only quantized matmul: y = x @ dequant(W).

W is stored int8 with per-group scales along the input dim (the layout GPTQ/AWQ produce).
The kernel loads an int8 weight tile, dequantizes it *inside* the K-loop (one scale per
group), and accumulates in fp32. Because only int8 weights cross HBM, this is ~2x less
weight bandwidth than an fp16 GEMM — exactly the win that makes memory-bound decode faster.

Constraint: GROUP_SIZE must be a multiple of BLOCK_K so each K-tile lies in one group.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tessera.kernels.reference import dequant_matmul_ref


def _configs():
    out = []
    for bm in (64, 128):
        for bn in (64, 128, 256):
            for bk in (32, 64):
                out.append(
                    triton.Config(
                        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                        num_warps=4,
                        num_stages=3,
                    )
                )
    return out


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _wq_matmul_kernel(
    x_ptr, qw_ptr, scale_ptr, y_ptr,
    M, N, K, GROUP_SIZE,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sg,
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
    qw_ptrs = qw_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_rem, other=0.0).to(tl.float32)
        qw = tl.load(qw_ptrs, mask=offs_k[:, None] < k_rem, other=0)

        group = (k * BLOCK_K) // GROUP_SIZE
        s = tl.load(scale_ptr + offs_n * stride_sn + group * stride_sg)  # (BLOCK_N,)
        w = qw.to(tl.float32) * s[None, :]

        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        qw_ptrs += BLOCK_K * stride_wk

    offs_ym = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_yn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    y_ptrs = y_ptr + offs_ym[:, None] * stride_ym + offs_yn[None, :] * stride_yn
    y_mask = (offs_ym[:, None] < M) & (offs_yn[None, :] < N)
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=y_mask)


def quant_matmul_triton(
    x: torch.Tensor,
    q_weight: torch.Tensor,  # (N, K) int8
    scales: torch.Tensor,  # (N, n_groups)
    group_size: int = 128,
) -> torch.Tensor:
    """Symmetric (zero-point free) weight-only int8 GEMM. Falls back to the reference
    when zeros are needed (asymmetric); see `tessera.quant` for the packing side."""
    *lead, k = x.shape
    x2d = x.reshape(-1, k).contiguous()
    m, n = x2d.shape[0], q_weight.shape[0]
    if group_size % 32 != 0:
        return dequant_matmul_ref(x, q_weight, scales, None, group_size).reshape(*lead, n)

    y = torch.empty((m, n), device=x.device, dtype=x.dtype)

    def grid(meta):
        return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)

    _wq_matmul_kernel[grid](
        x2d, q_weight, scales, y,
        m, n, k, group_size,
        x2d.stride(0), x2d.stride(1),
        q_weight.stride(0), q_weight.stride(1),
        scales.stride(0), scales.stride(1),
        y.stride(0), y.stride(1),
    )
    return y.reshape(*lead, n)
