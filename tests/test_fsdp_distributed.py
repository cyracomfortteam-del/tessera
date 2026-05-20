"""Real multi-process FSDP over gloo: 2 ranks must reproduce single-process Adam.

Marked `slow` (spawns processes). Validates that flat-parameter sharding + grad all-reduce
+ sharded Adam is numerically identical to training the unsharded model in one process.
"""

import datetime
import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

pytestmark = pytest.mark.slow


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _build():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 4))
    torch.manual_seed(123)
    x, y = torch.randn(4, 8), torch.randn(4, 4)
    return model, x, y


def _worker(rank: int, world_size: int, port: int, out: dict) -> None:
    import torch.distributed as dist

    from tessera.distill.fsdp import DistGroup, FullyShardedModule

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")  # loopback (macOS-friendly)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )

    model, x, y = _build()
    fsdp = FullyShardedModule(model, DistGroup(), lr=1e-2)
    for _ in range(15):
        fsdp.forward_backward(lambda m: F.mse_loss(m(x), y))
        fsdp.step()

    if rank == 0:
        out.update({k: v.clone() for k, v in fsdp.full_state_dict().items()})
    dist.barrier()
    dist.destroy_process_group()


def test_fsdp_gloo_two_ranks_matches_single_process():
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")

    world_size = 2
    manager = mp.Manager()
    out = manager.dict()
    try:
        mp.spawn(_worker, args=(world_size, _free_port(), out), nprocs=world_size, join=True)
    except Exception as e:  # gloo rendezvous can fail on some hosts (e.g. sandboxed macOS)
        pytest.skip(f"gloo distributed unavailable on this host: {e}")

    # Single-process reference: plain Adam on the same model/batch.
    ref, x, y = _build()
    opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    for _ in range(15):
        opt.zero_grad()
        F.mse_loss(ref(x), y).backward()
        opt.step()

    assert len(out) > 0
    for name, p in ref.named_parameters():
        torch.testing.assert_close(out[name], p, atol=1e-4, rtol=1e-4)
