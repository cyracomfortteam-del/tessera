"""Grouped-query attention with RoPE and an optional KV cache."""

from __future__ import annotations

import torch
import torch.nn as nn

from tessera import kernels
from tessera.config import ModelConfig
from tessera.model.kv_cache import KVCache
from tessera.model.rope import apply_rope


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_kv_heads is not None
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache | None = None,
        layer_idx: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        b, t, _ = x.shape

        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin, positions)
        k = apply_rope(k, cos, sin, positions)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v, start_pos)

        # Causal masking is alignment-aware in the reference/kernel: when q is shorter
        # than k (incremental decode) the new queries attend to the cache tail.
        out = kernels.flash_attention(q, k, v, causal=True)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.wo(out)
