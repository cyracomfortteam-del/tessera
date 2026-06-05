# Kernels

Every fused kernel has a torch reference that defines what it should compute. The reference
runs everywhere; the Triton/CUDA kernels run on a GPU and are compared against the reference
in [`tests/test_kernels_gpu.py`](../tests/test_kernels_gpu.py).

## What's there

| Kernel | File | Notes |
|---|---|---|
| FlashAttention (fwd) | [`triton/flash_attention.py`](../tessera/kernels/triton/flash_attention.py) | Tile queries in SRAM, stream K/V, keep a running max and denominator so the full T×T score matrix never hits HBM. Causal early-exit, GQA via a query→KV head map, autotuned block sizes. |
| RMSNorm | [`triton/rmsnorm.py`](../tessera/kernels/triton/rmsnorm.py) | One program per row, coalesced loads, fp32 reduction. |
| Fused SwiGLU | [`triton/swiglu.py`](../tessera/kernels/triton/swiglu.py) | One tiled GEMM computes gate and up in registers and writes `silu(gate)*up` once, so the activation crosses HBM once instead of three times. |
| int8 dequant-matmul | [`triton/quant_matmul.py`](../tessera/kernels/triton/quant_matmul.py) | Dequantize int8 weight tiles inside the K-loop, accumulate in fp32. Roughly half the weight bandwidth of an fp16 GEMM. |

The CUDA C++ versions in [`cuda/`](../tessera/kernels/cuda/) cover the same RMSNorm and
attention at a lower level (shared-memory K/V staging, warp-shuffle reductions, float4 loads)
and are a reasonable `ncu`/`nsys` baseline.

## Backward

Forward is the fused kernel. Backward is a torch recompute wrapped in an `autograd.Function`:
since the fused forward equals the reference forward, its gradient equals the reference
gradient. A fully fused backward kernel is still on the list.

## Profiling

```bash
nsys profile --trace=cuda,nvtx python examples/serve.py
ncu --set full --kernel-name _attn_fwd_kernel python benchmarks/bench_attention.py
```

`tessera.profiling.nvtx_range` annotates regions; `time_op` uses CUDA events on GPU and
perf_counter on CPU. The `ncu` metrics worth looking at are listed in
[cuda/README](../tessera/kernels/cuda/README.md).
