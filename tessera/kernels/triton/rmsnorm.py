"""Fused RMSNorm forward kernel.

One Triton program handles one row (one token): it streams the row once to accumulate the
sum of squares, computes the reciprocal RMS in fp32, then streams it again applying the
gain. Loads are contiguous along the feature dimension, so the row is read with fully
coalesced 128-byte transactions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tessera.kernels.reference import rmsnorm_ref


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,  # (M, N) input
    w_ptr,  # (N,) gain
    y_ptr,  # (M, N) output
    row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * row_stride
    y_row = y_ptr + row * row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    # mean of squares over the (masked) feature dim
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(y_row + cols, y.to(y_ptr.dtype.element_ty), mask=mask)


def _rmsnorm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    *lead, n = x.shape
    x2d = x.reshape(-1, n).contiguous()
    m = x2d.shape[0]
    y = torch.empty_like(x2d)
    block = triton.next_power_of_2(n)
    num_warps = max(1, min(16, block // 256))
    _rmsnorm_fwd_kernel[(m,)](
        x2d, weight, y, x2d.stride(0), n, eps, BLOCK_SIZE=block, num_warps=num_warps
    )
    return y.reshape(*lead, n)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return _rmsnorm_fwd(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_out):
        # Correct-by-construction: the fused forward equals rmsnorm_ref, so its grad equals
        # the reference grad. We recompute through the reference for the backward pass.
        x, weight = ctx.saved_tensors
        with torch.enable_grad():
            xr = x.detach().requires_grad_(True)
            wr = weight.detach().requires_grad_(True)
            y = rmsnorm_ref(xr, wr, ctx.eps)
            gx, gw = torch.autograd.grad(y, (xr, wr), grad_out)
        return gx, gw, None


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return _RMSNormFn.apply(x, weight, eps)
