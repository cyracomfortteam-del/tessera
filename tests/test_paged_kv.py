"""Block allocator accounting + paged KV append/gather round-trip."""

import pytest
import torch

from tessera.serve.paged_kv import BlockAllocator, PagedKVCache


def test_allocator_alloc_and_free():
    a = BlockAllocator(num_blocks=4)
    assert a.num_free == 4
    blocks = [a.allocate() for _ in range(4)]
    assert a.num_free == 0
    assert len(set(blocks)) == 4
    with pytest.raises(MemoryError):
        a.allocate()
    a.free(blocks[0])
    assert a.num_free == 1


def test_allocator_refcount_prefix_sharing():
    a = BlockAllocator(num_blocks=2)
    b = a.allocate()
    a.share(b)  # a second sequence references the same block
    assert a.ref_count(b) == 2
    a.free(b)
    assert a.ref_count(b) == 1
    assert a.num_free == 1  # still held by the sharer
    a.free(b)
    assert a.num_free == 2


def test_paged_append_gather_roundtrip():
    torch.manual_seed(0)
    n_layers, n_kv, d, bs = 2, 2, 8, 4
    cache = PagedKVCache(n_layers, n_kv, d, block_size=bs, num_blocks=8, dtype=torch.float32)

    length = 10  # spans 3 blocks of size 4
    n_blocks = cache.blocks_for_length(length)
    table = [cache.allocator.allocate() for _ in range(n_blocks)]

    ref_k = torch.randn(length, n_kv, d)
    ref_v = torch.randn(length, n_kv, d)
    for layer in range(n_layers):
        for pos in range(length):
            cache.append(layer, table, pos, ref_k[pos] + layer, ref_v[pos] + layer)

    for layer in range(n_layers):
        gk, gv = cache.gather(layer, table, length)
        assert gk.shape == (length, n_kv, d)
        torch.testing.assert_close(gk, ref_k + layer)
        torch.testing.assert_close(gv, ref_v + layer)


def test_blocks_for_length():
    cache = PagedKVCache(1, 1, 4, block_size=16, num_blocks=4, dtype=torch.float32)
    assert cache.blocks_for_length(1) == 1
    assert cache.blocks_for_length(16) == 1
    assert cache.blocks_for_length(17) == 2
