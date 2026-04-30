"""nvtx ranges + accurate device timing.

`nvtx_range` annotates regions so they show up as named ranges in Nsight Systems
(`nsys profile`). `CudaTimer`/`time_op` use CUDA events on GPU (the only correct way to
time async kernels) and fall back to perf_counter on CPU so benchmarks run anywhere.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

try:
    import torch.cuda.nvtx as _nvtx

    _HAS_NVTX = True
except Exception:  # pragma: no cover
    _HAS_NVTX = False


@contextlib.contextmanager
def nvtx_range(name: str):
    """Push/pop an nvtx range (no-op without CUDA so call sites stay clean)."""
    on = _HAS_NVTX and torch.cuda.is_available()
    if on:
        _nvtx.range_push(name)
    try:
        yield
    finally:
        if on:
            _nvtx.range_pop()


class CudaTimer:
    """Context manager yielding elapsed milliseconds for the wrapped region."""

    def __init__(self) -> None:
        self.ms: float = 0.0
        self._use_cuda = torch.cuda.is_available()

    def __enter__(self) -> CudaTimer:
        if self._use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.ms = self._start.elapsed_time(self._end)
        else:
            self.ms = (time.perf_counter() - self._t0) * 1e3


def time_op(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 10,
    iters: int = 50,
    **kwargs: Any,
) -> dict[str, float]:
    """Median/min/max latency (ms) of `fn(*args, **kwargs)` over `iters` runs."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        with CudaTimer() as t:
            fn(*args, **kwargs)
        samples.append(t.ms)

    samples.sort()
    return {
        "median_ms": statistics.median(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": statistics.fmean(samples),
        "iters": float(iters),
    }
