"""Block-paged KV cache — the memory manager behind continuous batching.

Instead of reserving a contiguous `max_seq_len` slab per sequence (most of which is wasted),
KV is stored in fixed-size blocks drawn from a shared pool. A sequence holds a *block table*
(logical position -> physical block). This is what lets many sequences of wildly different
lengths share one pool with almost no fragmentation, and it makes prefix sharing a
ref-count bump instead of a copy (PagedAttention, Kwon et al. 2023).

The append/gather here are plain torch indexing — correct and test-covered. The fused
paged-attention *kernel* that reads these blocks without gathering is the GPU follow-up
(see the decode-path note in kernels/triton/flash_attention.py).
"""

from __future__ import annotations

import torch


class BlockAllocator:
    """Fixed pool of physical blocks with ref-counting for copy-on-write prefix sharing."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self._free: list[int] = list(reversed(range(num_blocks)))  # used as a stack
        self._ref = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise MemoryError("KV cache out of blocks")
        block = self._free.pop()
        self._ref[block] = 1
        return block

    def share(self, block: int) -> int:
        """Bump the ref count (a second sequence now points at this block)."""
        self._ref[block] += 1
        return block

    def free(self, block: int) -> None:
        if self._ref[block] == 0:
            raise ValueError(f"double free of block {block}")
        self._ref[block] -= 1
        if self._ref[block] == 0:
            self._free.append(block)

    def ref_count(self, block: int) -> int:
        return self._ref[block]


class PagedKVCache:
    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: int = 1024,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.block_size = block_size
        self.n_layers = n_layers
        self.allocator = BlockAllocator(num_blocks)
        shape = (n_layers, num_blocks, block_size, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def blocks_for_length(self, length: int) -> int:
        return (length + self.block_size - 1) // self.block_size

    def append(
        self,
        layer: int,
        block_table: list[int],
        pos: int,
        k_vec: torch.Tensor,  # (n_kv_heads, head_dim)
        v_vec: torch.Tensor,
    ) -> None:
        block = block_table[pos // self.block_size]
        offset = pos % self.block_size
        self.k[layer, block, offset] = k_vec
        self.v[layer, block, offset] = v_vec

    def gather(
        self, layer: int, block_table: list[int], length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct contiguous (length, n_kv_heads, head_dim) K/V from the block table."""
        ks, vs = [], []
        remaining = length
        for block in block_table:
            take = min(self.block_size, remaining)
            if take <= 0:
                break
            ks.append(self.k[layer, block, :take])
            vs.append(self.v[layer, block, :take])
            remaining -= take
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)
