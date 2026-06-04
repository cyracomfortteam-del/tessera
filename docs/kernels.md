# Kernels

Every fused kernel ships with a torch reference that *is* its specification. The reference
runs everywhere; the Triton/CUDA kernels run on a GPU and are checked against the reference
to fp tolerance in [`tests/test_kernels_gpu.py`](../tests/test_kernels_gpu.py).

## What's implemented

| Kernel | File | Idea |
|---|---|---|
| FlashAttention (fwd) | [`triton/flash_attention.py`](../tessera/kernels/triton/flash_attention.py) | Tile queries in SRAM, stream K/V, online softmax (running max/denominator) so the T×T score matrix never touches HBM. Causal early-exit; GQA via a query→KV head index map. Autotuned `BLOCK_M/BLOCK_N`. |
| RMSNorm | [`triton/rmsnorm.py`](../tessera/kernels/triton/rmsnorm.py) | One program per row; coalesced `float`-wide loads; fp32 reduction. |
| Fused SwiGLU | [`triton/swiglu.py`](../tessera/kernels/triton/swiglu.py) | One tiled GEMM computes gate+up in registers and writes `silu(gate)*up` once (~3× less activation HBM traffic). Block-swizzled for L2 reuse. |
| int8 dequant-matmul | [`triton/quant_matmul.py`](../tessera/kernels/triton/quant_matmul.py) | Dequantize int8 weight tiles inside the K-loop; fp32 accumulate. ~2× less weight bandwidth. |

The raw CUDA C++ variants in [`cuda/`](../tessera/kernels/cuda/) document the lower-level
memory choreography (shared-memory K/V staging, warp-shuffle reductions, `float4` vectorized
loads) and are the `ncu`/`nsys` baseline.

## Backward

Forward is the fused kernel; backward is a torch recompute wrapped in an `autograd.Function`.
Because the fused forward equals the reference forward, its gradient equals the reference
gradient — correct by construction, with a fully-fused backward kernel as a scoped follow-up.

## Profiling

```bash
nsys profile --trace=cuda,nvtx python examples/serve.py        # timeline + nvtx ranges
ncu --set full --kernel-name _attn_fwd_kernel  python benchmarks/bench_attention.py
```

`tessera.profiling.nvtx_range` annotates regions; `time_op` uses CUDA events on GPU and falls
back to perf_counter on CPU. Key `ncu` metrics to watch are in [cuda/README](../tessera/kernels/cuda/README.md).
