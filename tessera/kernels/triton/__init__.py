"""Hand-written Triton kernels.

These modules import `triton` at module load, so they are only imported by the dispatch
layer when `tessera.kernels.triton_available()` is true (CUDA present + Triton installed).
Importing them on a CPU-only host will raise — that's intentional; use the ops in
`tessera.kernels` which guard the import for you.

Design notes that apply to every kernel here:
  * fp32 accumulation regardless of the I/O dtype (fp16/bf16) for numerical parity with
    the torch reference;
  * `@triton.autotune` over block sizes / num_warps so the same source is competitive
    across SM generations;
  * forward is the fused kernel; backward (where needed for training) is a torch
    recompute wrapped in an autograd.Function — fast forward, correct grad, with a fully
    fused backward kernel left as a clearly-scoped follow-up.
"""
