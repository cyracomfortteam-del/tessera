"""Shared pytest fixtures + automatic skipping of GPU-only tests on CPU hosts."""

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.gpu tests unless a CUDA device is actually present."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device (Triton/CUDA kernels need a GPU)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def seed():
    torch.manual_seed(1234)
    return 1234
