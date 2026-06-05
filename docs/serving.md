# Serving

The inference path follows the same ideas as vLLM / TensorRT-LLM, at a size you can read in
an afternoon.

## Paged KV cache ([`serve/paged_kv.py`](../tessera/serve/paged_kv.py))

KV lives in fixed-size blocks from a shared pool, and a sequence holds a block table mapping
logical position to physical block. There's no per-sequence `max_seq_len` slab to reserve, so
fragmentation is minimal, and sharing a prefix is a ref-count bump (`BlockAllocator.share`)
instead of a copy. `append`/`gather` are plain torch indexing; a fused paged-attention kernel
that reads the blocks without gathering is still to do.

## Continuous batching ([`serve/scheduler.py`](../tessera/serve/scheduler.py))

The batch is rebuilt every decode step. `schedule()` does all the admission, block allocation,
and preemption and returns the exact set the engine may run, so a finished short request frees
its blocks immediately and a waiting request gets admitted on the next step rather than waiting
behind a long generation. When memory runs out the most recently admitted sequence is
preempted back to the queue and resumes later with its progress intact. The thing that's easy
to get wrong, leaking blocks when preemption edits the running set mid-step, is covered by
[`tests/test_engine.py`](../tests/test_engine.py).

## Speculative decoding ([`serve/speculative.py`](../tessera/serve/speculative.py))

A cheap draft proposes `k` tokens and the target verifies all `k` in one forward. Proposal
`x_i` is accepted with probability `min(1, p_i/q_i)`; the first rejection resamples from the
normalized residual `max(0, p−q)`. That samples from the target distribution while doing about
one target forward per several tokens. With `draft == target` and greedy sampling every
proposal is accepted and the output is identical to plain decoding, which the tests use as a
correctness check.

## Quantization ([`quant/`](../tessera/quant/))

- int8 weight-only, per-group symmetric (`QuantLinear`).
- AWQ: per-channel scaling that protects salient channels and folds into the GEMM at no runtime
  cost.
- FP8 (E4M3), emulated in fp32 so the numerics and dynamic scaling are testable on CPU.

`quantize_model()` swaps eligible `nn.Linear` layers in place and leaves the LM head and the
fused SwiGLU gate/up projections in higher precision.

## Rust gateway ([`tessera-rs/`](../tessera-rs/))

A tokio + axum front end handles HTTP/JSON and admission back-pressure (a `Semaphore` budget)
and calls into a pluggable `Engine` (a mock for tests, PyO3 for production). The math stays in
the Python/CUDA engine; Rust handles connections and tail latency.
