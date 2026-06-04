# Distillation & distributed training

## Losses ([`distill/losses.py`](../tessera/distill/losses.py))

* **Soft-target KD** — temperature-scaled `KL(teacher ‖ student)`, multiplied by `T²` to keep
  gradient magnitudes stable across temperatures.
* **Hard CE** — optional ground-truth cross-entropy, blended as `α·KD + (1−α)·CE`.
* **Hidden-state KD** — MSE between a projected student hidden state and the teacher's, matched
  layer-for-layer by stride when depths differ.

## From-scratch FSDP ([`distill/fsdp.py`](../tessera/distill/fsdp.py))

ZeRO-3-style flat-parameter sharding: each rank owns `1/world_size` of the flattened
parameters, and that shard is where the Adam state lives too (optimizer memory scales as
`1/N`). One training step:

1. **all-gather** the shards → reconstruct the full flat parameter → scatter into the module,
2. forward / backward → full gradients,
3. **all-reduce + slice** → this rank's gradient shard (a reduce-scatter),
4. **Adam-update only the local shard**.

Because the Adam update is elementwise, sharded training is *numerically identical* to
single-process training. Two tests assert exactly this:

* [`test_distill.py::test_sharded_adam_matches_torch_adam`](../tests/test_distill.py) — world_size=1 matches `torch.optim.Adam` step-for-step.
* [`test_fsdp_distributed.py`](../tests/test_fsdp_distributed.py) — **2 real processes over gloo** reproduce the single-process result (runs in Linux CI; skips if gloo can't bind).

`LocalGroup` runs everything in one process so the trainer and CPU tests need no `torchrun`;
`DistGroup` wraps real `torch.distributed` collectives (gloo/NCCL).

## Fault-tolerant checkpointing ([`distill/checkpoint.py`](../tessera/distill/checkpoint.py))

Each rank persists only its own param+optimizer shard (no rank-0 bottleneck). Writes are
**atomic** — write to a temp file, `os.replace` into place — so a crash mid-write can never
corrupt the last good checkpoint; `resume_or_init` falls back to the latest complete step.
`save_consolidated` gathers the full weights into one file for inference.
