# Serving

The inference path mirrors a production stack (vLLM/TensorRT-LLM-style) at a readable scale.

## Paged KV cache ([`serve/paged_kv.py`](../tessera/serve/paged_kv.py))

KV lives in fixed-size blocks drawn from a shared pool; a sequence holds a *block table*
(logical position → physical block). No per-sequence `max_seq_len` slab, almost no
fragmentation, and prefix sharing is a ref-count bump (`BlockAllocator.share`) rather than a
copy. `append`/`gather` are plain torch indexing; the fused paged-attention kernel that reads
blocks without gathering is the GPU follow-up.

## Continuous batching ([`serve/scheduler.py`](../tessera/serve/scheduler.py))

The batch is recomposed **every decode step**. `schedule()` owns all admission, block
allocation, and preemption, then returns exactly the sequences the engine may run — so a
finished 5-token request frees its blocks immediately and a waiting request is admitted on the
next step, never blocking behind a slow 500-token generation. Under memory pressure the
most-recently-admitted sequence is preempted back to the queue (and resumes later with its
progress intact). The hard part — *not* leaking blocks when preemption edits the running set
mid-step — is covered by [`tests/test_engine.py`](../tests/test_engine.py).

## Speculative decoding ([`serve/speculative.py`](../tessera/serve/speculative.py))

A cheap draft proposes `k` tokens; the target verifies all `k` in one forward. Each proposal
`x_i` is accepted with probability `min(1, p_i/q_i)`; the first rejection resamples from the
normalized residual `max(0, p−q)`. This provably samples from the target distribution while
doing ~1 target forward per several tokens. With `draft == target` and greedy sampling every
proposal is accepted and the output is identical to plain decoding — an exact correctness
anchor in the tests.

## Quantization ([`quant/`](../tessera/quant/))

* **int8** weight-only, per-group symmetric (GPTQ-style packing) → `QuantLinear`.
* **AWQ** activation-aware per-channel scaling that protects salient channels at zero runtime
  cost (the scale folds into the GEMM).
* **FP8 (E4M3)** emulated in fp32 so the numerics + dynamic scaling are CPU-testable.

`quantize_model()` swaps eligible `nn.Linear` layers in place (keeping the LM head and the
fused SwiGLU gate/up projections in higher precision).

## Rust gateway ([`tessera-rs/`](../tessera-rs/))

A tokio + axum front end handles HTTP/JSON and admission back-pressure (a `Semaphore` budget),
delegating token generation to a pluggable `Engine` (mock for tests, PyO3 for production). The
heavy math stays in the Python/CUDA engine; Rust owns connections and tail latency.
