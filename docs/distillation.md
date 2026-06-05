# Distillation and distributed training

## Losses ([`distill/losses.py`](../tessera/distill/losses.py))

- Soft-target KD: temperature-scaled `KL(teacher ‖ student)`, multiplied by `T²` so gradient
  magnitudes stay comparable across temperatures.
- Hard CE: optional ground-truth cross-entropy, blended as `α·KD + (1−α)·CE`.
- Hidden-state KD: MSE between a projected student hidden state and the teacher's, matched
  layer for layer by stride when the depths differ.

## FSDP from scratch ([`distill/fsdp.py`](../tessera/distill/fsdp.py))

Flat-parameter sharding in the ZeRO-3 style: each rank owns `1/world_size` of the flattened
parameters, and the Adam state lives on that same shard, so optimizer memory scales with the
cluster. A step is:

1. all-gather the shards into the full flat parameter and scatter it into the module,
2. forward and backward to get full gradients,
3. all-reduce and slice to get this rank's gradient shard (a reduce-scatter),
4. Adam-update the local shard only.

The Adam update is elementwise, so sharded training comes out identical to single-process
training. Two tests check exactly that:

- [`test_distill.py::test_sharded_adam_matches_torch_adam`](../tests/test_distill.py) — world_size=1 matches `torch.optim.Adam` step for step.
- [`test_fsdp_distributed.py`](../tests/test_fsdp_distributed.py) — two real gloo processes reproduce the single-process result (runs in Linux CI, skips if gloo can't bind).

`LocalGroup` runs it all in one process so the trainer and the CPU tests don't need `torchrun`;
`DistGroup` wraps real `torch.distributed` collectives.

## Checkpointing ([`distill/checkpoint.py`](../tessera/distill/checkpoint.py))

Each rank writes only its own param and optimizer shard, so there's no rank-0 bottleneck.
Writes go to a temp file and `os.replace` into place, so a crash mid-write can't corrupt the
last good checkpoint; `resume_or_init` falls back to the latest complete step.
`save_consolidated` gathers the full weights into one file for inference.
