"""FlashAttention-style fused attention forward.

Classic flash attention: tile the query block into SRAM, stream key/value blocks, and keep
a running max + denominator so the full (T, T) score matrix is never materialised in HBM.
This is the memory-bandwidth win that makes long-context attention feasible.

Implemented here:
  * online softmax with fp32 running statistics (m_i, l_i);
  * causal early-exit — for query block `start_m` we never iterate key blocks past it;
  * grouped-query attention — query head `h` indexes KV head `h // (H // H_KV)`, so the KV
    tensors stay small and no explicit repeat_interleave touches HBM;
  * autotuned BLOCK_M/BLOCK_N/num_warps/num_stages.

Scope: this is the prefill / training kernel (q_len == k_len). The memory-bound single-token
decode step is served by the paged-attention path in `tessera.serve`, so the wrapper below
falls back to the torch reference when q_len != k_len.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tessera.kernels.reference import attention_ref


def _configs():
    out = []
    for bm in (64, 128):
        for bn in (32, 64):
            for s in (2, 3, 4):
                for w in (4, 8):
                    out.append(
                        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=s, num_warps=w)
                    )
    return out


@triton.autotune(configs=_configs(), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, Out, scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, H_KV, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // (H // H_KV)  # GQA: map query head -> shared KV head

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h_kv * stride_kh
    v_base = V + off_z * stride_vz + off_h_kv * stride_vh
    o_base = Out + off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_cur = start_n + offs_n

        k_ptrs = k_base + offs_n_cur[None, :] * stride_kn + offs_d[:, None] * stride_kk
        k = tl.load(k_ptrs, mask=offs_n_cur[None, :] < N_CTX, other=0.0)
        qk = tl.dot(q, k) * scale  # (BLOCK_M, BLOCK_N)

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n_cur[None, :], qk, float("-inf"))
        else:
            qk = tl.where(offs_n_cur[None, :] < N_CTX, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_base + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=offs_n_cur[:, None] < N_CTX, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def _attn_fwd(q, k, v, causal, softmax_scale):
    b, h, n_ctx, head_dim = q.shape
    h_kv = k.shape[1]
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    out = torch.empty_like(q)

    def grid(meta):
        return (triton.cdiv(n_ctx, meta["BLOCK_M"]), b * h)

    _attn_fwd_kernel[grid](
        q, k, v, out, scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H=h, H_KV=h_kv, N_CTX=n_ctx,
        HEAD_DIM=head_dim, CAUSAL=causal,
    )
    return out


class _FlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, softmax_scale):
        ctx.save_for_backward(q, k, v)
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        return _attn_fwd(q, k, v, causal, softmax_scale)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v = ctx.saved_tensors
        with torch.enable_grad():
            qr, kr, vr = (t.detach().requires_grad_(True) for t in (q, k, v))
            o = attention_ref(qr, kr, vr, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
            gq, gk, gv = torch.autograd.grad(o, (qr, kr, vr), grad_out)
        return gq, gk, gv, None, None


def flash_attention_triton(q, k, v, causal=True, softmax_scale=None):
    # Single-token / ragged decode (q_len != k_len) goes through the paged-attention path;
    # the fused prefill kernel below assumes equal query/key lengths.
    if q.shape[2] != k.shape[2]:
        return attention_ref(q, k, v, causal=causal, softmax_scale=softmax_scale)
    return _FlashAttnFn.apply(q, k, v, causal, softmax_scale)
