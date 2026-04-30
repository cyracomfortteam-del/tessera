# Raw CUDA kernels

These `.cu` files are the lower-level companions to the Triton kernels in `../triton/`.
The Triton versions are what the model dispatches to at runtime; these document the explicit
memory choreography (shared-memory staging, coalesced loads, warp reductions) and are useful
as an `ncu`/`nsys` baseline.

## Build

As a standalone object:

```bash
nvcc -O3 -arch=sm_80 -c rmsnorm.cu -o rmsnorm.o
nvcc -O3 -arch=sm_80 -c flash_attention.cu -o flash_attention.o
```

As a PyTorch extension (loads `rmsnorm_forward` into Python):

```python
from torch.utils.cpp_extension import load
mod = load(name="tessera_rmsnorm", sources=["rmsnorm.cu"], extra_cuda_cflags=["-O3"])
y = mod.rmsnorm_forward(x.cuda(), w.cuda(), 1e-5)
```

## Profiling

Nsight Systems — timeline + nvtx ranges (pair with `tessera.profiling.nvtx_range`):

```bash
nsys profile -o tessera_timeline --trace=cuda,nvtx python examples/serve.py
```

Nsight Compute — per-kernel memory throughput, occupancy, bank conflicts:

```bash
ncu --set full --kernel-name rmsnorm_fwd_kernel  python bench.py
ncu --set full --kernel-name flash_attn_fwd_kernel python bench.py
```

What to look at:

| Metric (ncu)                          | What it tells you                                  |
|---------------------------------------|----------------------------------------------------|
| `dram__throughput.avg.pct_of_peak`    | Are we memory-bound? RMSNorm should be ~peak BW.    |
| `l1tex__data_bank_conflicts_*`        | Shared-memory bank conflicts in the attention tile. |
| `sm__warps_active.avg.pct_of_peak`    | Occupancy — register/shared-mem pressure headroom.  |
| `smsp__inst_executed_pipe_tensor`     | Tensor-core utilization (Triton GEMM path).         |

## Notes

* `rmsnorm.cu` uses `float4` vectorized loads → 16-byte coalesced transactions and a
  warp-shuffle + shared-memory block reduction.
* `flash_attention.cu` stages K/V tiles through shared memory and keeps the online-softmax
  state in registers, so the `T × T` score matrix never reaches global memory.
* Both are fp32 reference variants; the production fp16/bf16 paths live in Triton with
  autotuned tile sizes.
