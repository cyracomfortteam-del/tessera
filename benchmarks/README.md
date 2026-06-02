# Benchmarks

Quick, dependency-light micro-benchmarks. On CPU/MPS they time the torch reference path; on
CUDA `bench_attention.py` switches to the Triton FlashAttention kernel automatically.

```bash
python benchmarks/bench_attention.py  --batch 4 --heads 16 --seq-len 1024 --head-dim 64
python benchmarks/bench_throughput.py --preset tessera-tiny --requests 8 --max-new-tokens 64
```

`bench_attention.py` reports median latency + effective attention TFLOP/s.
`bench_throughput.py` reports end-to-end engine tokens/sec and the speculative-decoding
acceptance rate.

For kernel-level profiling (occupancy, memory throughput, bank conflicts) use Nsight on a GPU:

```bash
ncu --set full --kernel-name _attn_fwd_kernel python benchmarks/bench_attention.py
nsys profile --trace=cuda,nvtx python examples/serve.py
```

> Numbers in the top-level README are an Apple M2 Pro reference floor. The fused kernels are
> built for NVIDIA GPUs — run there to see the real throughput.
