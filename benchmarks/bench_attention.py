"""Attention micro-benchmark.

On CUDA this compares the Triton FlashAttention kernel against torch SDPA; on CPU/MPS it
times the torch reference so the script runs anywhere. Reports median latency and the
effective attention FLOP/s.
"""

from __future__ import annotations

import argparse

import torch

from tessera.kernels import default_device, triton_available
from tessera.kernels.reference import attention_ref
from tessera.profiling import time_op


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--head-dim", type=int, default=64)
    args = ap.parse_args()

    device = default_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    b, h, t, d = args.batch, args.heads, args.seq_len, args.head_dim
    q, k, v = (torch.randn(b, h, t, d, device=device, dtype=dtype) for _ in range(3))

    # 2 matmuls of (t,d)x(d,t) and (t,t)x(t,d) per head, causal ~ half => 2*b*h*t*t*d
    flops = 2.0 * b * h * t * t * d * 2

    if triton_available():
        from tessera.kernels.triton.flash_attention import flash_attention_triton

        stats = time_op(lambda: flash_attention_triton(q, k, v, causal=True))
        backend = "triton-flash"
    else:
        stats = time_op(lambda: attention_ref(q, k, v, causal=True))
        backend = "torch-reference"

    ms = stats["median_ms"]
    print(f"[{backend}] B={b} H={h} T={t} D={d} dtype={dtype} device={device}")
    print(f"  median {ms:.3f} ms   {flops / (ms * 1e-3) / 1e12:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
