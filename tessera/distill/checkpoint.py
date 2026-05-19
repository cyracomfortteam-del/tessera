"""Fault-tolerant sharded checkpointing.

Each rank persists only its own parameter + optimizer shard, so checkpoint size and write
bandwidth scale with the cluster (no rank-0 bottleneck). Writes are atomic: we write to a
temp file and `os.replace` it into place, so a crash mid-write can never corrupt the last
good checkpoint — resume just falls back to the previous step.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

from tessera.distill.fsdp import FullyShardedModule

_SHARD_RE = re.compile(r"shard_rank(\d+)_step(\d+)\.pt")


def save_sharded(fsdp: FullyShardedModule, step: int, ckpt_dir: str | os.PathLike) -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rank = fsdp.group.rank
    path = ckpt_dir / f"shard_rank{rank}_step{step}.pt"
    tmp = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "step": step,
            "rank": rank,
            "world_size": fsdp.group.world_size,
            "p_shard": fsdp.p_shard,
            "m": fsdp.m,
            "v": fsdp.v,
            "t": fsdp.t,
        },
        tmp,
    )
    os.replace(tmp, path)  # atomic on POSIX
    return path


def load_sharded(fsdp: FullyShardedModule, path: str | os.PathLike) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    fsdp.p_shard.copy_(state["p_shard"])
    fsdp.m.copy_(state["m"])
    fsdp.v.copy_(state["v"])
    fsdp.t = int(state["t"])
    return int(state["step"])


def latest_step(ckpt_dir: str | os.PathLike, rank: int) -> int | None:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    steps = []
    for p in ckpt_dir.iterdir():
        m = _SHARD_RE.fullmatch(p.name)
        if m and int(m.group(1)) == rank:
            steps.append(int(m.group(2)))
    return max(steps) if steps else None


def resume_or_init(fsdp: FullyShardedModule, ckpt_dir: str | os.PathLike) -> int:
    """Return the step to resume from (0 if no checkpoint), loading shard state if present."""
    step = latest_step(ckpt_dir, fsdp.group.rank)
    if step is None:
        return 0
    path = Path(ckpt_dir) / f"shard_rank{fsdp.group.rank}_step{step}.pt"
    return load_sharded(fsdp, path)


def save_consolidated(fsdp: FullyShardedModule, path: str | os.PathLike) -> None:
    """Rank-0 gathers the full (unsharded) weights and writes one file for inference."""
    full = fsdp.full_state_dict()
    if fsdp.group.rank == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(full, tmp)
        os.replace(tmp, path)
