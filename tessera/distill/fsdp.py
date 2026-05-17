"""From-scratch fully-sharded data parallel (ZeRO-3 style).

Each rank owns 1/world_size of the *flattened* parameters — that shard is the source of
truth and also where the optimizer state (Adam m/v) lives, so optimizer memory scales as
1/N. For a step we:

  1. all-gather the shards to reconstruct the full flat parameter and scatter it into the
     module's tensors (`summon_full_params`),
  2. run forward/backward to get full gradients,
  3. all-reduce + slice to get this rank's gradient shard (a reduce-scatter),
  4. Adam-update only the local shard.

Because the Adam update is elementwise, sharded training is numerically identical to
single-process training — which is exactly what `tests/test_fsdp.py` asserts over gloo.

`LocalGroup` makes the whole thing run in one process (world_size=1) so the distillation
trainer and CPU tests need no torchrun; `DistGroup` wraps real torch.distributed collectives.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGroup:
    """Single-process 'process group': collectives are the identity."""

    rank = 0
    world_size = 1

    def all_gather(self, t: torch.Tensor) -> list[torch.Tensor]:
        return [t.clone()]

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        return t


class DistGroup:
    """Thin wrapper over torch.distributed (use with the gloo/nccl backend)."""

    def __init__(self) -> None:
        import torch.distributed as dist

        self.dist = dist
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def all_gather(self, t: torch.Tensor) -> list[torch.Tensor]:
        out = [torch.empty_like(t) for _ in range(self.world_size)]
        self.dist.all_gather(out, t.contiguous())
        return out

    def all_reduce_sum(self, t: torch.Tensor) -> torch.Tensor:
        self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM)
        return t


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


class FullyShardedModule:
    def __init__(
        self,
        module: nn.Module,
        group: LocalGroup | DistGroup | None = None,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.module = module
        self.group = group or LocalGroup()
        self.lr, self.betas, self.eps, self.wd = lr, betas, eps, weight_decay

        self.params = [p for p in module.parameters() if p.requires_grad]
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.total = sum(self.numels)
        world = self.group.world_size
        self.pad = (-self.total) % world
        self.shard_size = (self.total + self.pad) // world

        flat = F.pad(_flatten([p.detach() for p in self.params]), (0, self.pad))
        s = self.group.rank * self.shard_size
        self.p_shard = flat[s : s + self.shard_size].clone()
        self.m = torch.zeros_like(self.p_shard)
        self.v = torch.zeros_like(self.p_shard)
        self.t = 0

    # -- parameter materialization -------------------------------------------
    def _full_flat(self) -> torch.Tensor:
        shards = self.group.all_gather(self.p_shard)
        return torch.cat(shards)[: self.total]

    def _scatter_into_module(self) -> None:
        full = self._full_flat()
        offset = 0
        for p, n, shape in zip(self.params, self.numels, self.shapes, strict=True):
            p.data.copy_(full[offset : offset + n].view(shape))
            offset += n

    @contextlib.contextmanager
    def summon_full_params(self):
        self._scatter_into_module()
        yield self.module

    # -- training step --------------------------------------------------------
    def forward_backward(self, loss_fn) -> torch.Tensor:
        self._scatter_into_module()
        for p in self.params:
            p.grad = None
        loss = loss_fn(self.module)
        loss.backward()
        return loss

    def step(self) -> None:
        grads = [
            p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params
        ]
        g = F.pad(_flatten(grads), (0, self.pad))
        g = self.group.all_reduce_sum(g) / self.group.world_size  # reduce-scatter (avg)
        s = self.group.rank * self.shard_size
        g_local = g[s : s + self.shard_size]

        self.t += 1
        b1, b2 = self.betas
        if self.wd:
            g_local = g_local + self.wd * self.p_shard
        self.m.mul_(b1).add_(g_local, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g_local, g_local, value=1 - b2)
        m_hat = self.m / (1 - b1**self.t)
        v_hat = self.v / (1 - b2**self.t)
        self.p_shard.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    # -- checkpointing --------------------------------------------------------
    def full_state_dict(self) -> dict[str, torch.Tensor]:
        full = self._full_flat()
        out, offset = {}, 0
        names = [n for n, p in self.module.named_parameters() if p.requires_grad]
        for name, n, shape in zip(names, self.numels, self.shapes, strict=True):
            out[name] = full[offset : offset + n].view(shape).clone()
            offset += n
        return out
