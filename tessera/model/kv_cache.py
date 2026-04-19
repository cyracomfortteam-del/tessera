"""A simple contiguous KV cache for single-sequence / static-batch decoding.

This is the straightforward cache used by `model.generate` and the parity tests. The
production serving path uses the block-paged cache in `tessera.serve.paged_kv`, which
trades this simplicity for non-contiguous memory and prefix sharing.
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(
        self,
        n_layers: int,
        batch: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (n_layers, batch, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def update(
        self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor, start_pos: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write the new K/V at `start_pos` and return the full cache up to the new end."""
        t = k_new.shape[2]
        end = start_pos + t
        if end > self.max_seq_len:
            raise ValueError(f"sequence length {end} exceeds cache capacity {self.max_seq_len}")
        self.k[layer, :, :, start_pos:end] = k_new
        self.v[layer, :, :, start_pos:end] = v_new
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]
