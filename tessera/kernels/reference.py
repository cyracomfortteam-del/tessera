"""Torch reference implementations — the executable spec for every fused kernel.

These are written for clarity, not speed. Each Triton kernel is validated against the
function here in `tests/`, so if a kernel and its reference ever disagree beyond fp
tolerance the test suite fails. Keeping the spec in plain torch also means the entire
model runs correctly on CPU/MPS with zero kernels compiled.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Root-mean-square layer norm (no mean subtraction, no bias).

    Computed in fp32 for numerical stability then cast back, matching the kernel.
    """
    dtype = x.dtype
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    out = x32 * torch.rsqrt(var + eps)
    return (out.to(dtype)) * weight


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Reference scaled-dot-product attention over (B, H, T, D) tensors.

    Supports grouped-query attention: if k/v have fewer heads than q, they are
    repeat-interleaved up to q's head count (the same broadcast a fused GQA kernel
    performs implicitly by indexing the shared KV head).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    n_rep = q.shape[1] // k.shape[1]
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale  # (B,H,T,T)
    if causal:
        t_q, t_k = scores.shape[-2], scores.shape[-1]
        mask = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril(diagonal=t_k - t_q)
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn, v)


def swiglu_ref(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """SwiGLU feed-forward inner activation: silu(x @ Wg) * (x @ Wu).

    Weights are (hidden, dim) like nn.Linear; the down projection lives in the model.
    """
    gate = F.silu(F.linear(x, w_gate))
    up = F.linear(x, w_up)
    return gate * up


def dequant_matmul_ref(
    x: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Reference for weight-only quantized matmul: x @ dequant(W).

    `q_weight` is int8/uint8 of shape (out, in); `scales`/`zeros` are per-group along the
    input dimension with `group_size` columns per group. This mirrors what the Triton
    `quant_matmul` kernel does on-chip (dequantize a tile, accumulate in fp32).
    """
    out_features, in_features = q_weight.shape
    n_groups = (in_features + group_size - 1) // group_size

    w = q_weight.to(torch.float32)
    if zeros is not None:
        w = w - zeros.to(torch.float32).repeat_interleave(group_size, dim=1)[:, :in_features]
    scale_full = scales.to(torch.float32).repeat_interleave(group_size, dim=1)[:, :in_features]
    w = w * scale_full
    assert scales.shape[1] == n_groups
    return F.linear(x.to(torch.float32), w).to(x.dtype)
