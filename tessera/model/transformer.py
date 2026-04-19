"""The reference decoder-only transformer.

Architecture: token embedding → N × (RMSNorm → GQA+RoPE → residual → RMSNorm → SwiGLU →
residual) → RMSNorm → tied LM head. Every heavy op routes through `tessera.kernels`, so the
same module definition runs on CPU/MPS (torch reference) or CUDA (Triton kernels).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tessera.config import ModelConfig
from tessera.model.attention import Attention
from tessera.model.kv_cache import KVCache
from tessera.model.mlp import SwiGLU
from tessera.model.rmsnorm import RMSNorm
from tessera.model.rope import build_rope_cache


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, positions, cache=None, start_pos=0):
        x = x + self.attn(
            self.attn_norm(x), cos, sin, positions, cache, self.layer_idx, start_pos
        )
        x = x + self.mlp(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, i) for i in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layers) (GPT-2 style).
        scale = (2 * cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                p.data.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.tok_emb.weight.device

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.cfg.tie_embeddings:
            n -= self.lm_head.weight.numel()
        return n

    def forward(
        self,
        tokens: torch.Tensor,
        cache: KVCache | None = None,
        start_pos: int = 0,
        return_hidden: bool = False,
    ):
        b, t = tokens.shape
        h = self.tok_emb(tokens)
        positions = torch.arange(start_pos, start_pos + t, device=tokens.device)

        hidden_states = []
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin, positions, cache, start_pos)
            if return_hidden:
                hidden_states.append(h)

        h = self.norm(h)
        logits = self.lm_head(h)
        if return_hidden:
            return logits, hidden_states
        return logits

    def new_cache(self, batch: int, dtype: torch.dtype | None = None) -> KVCache:
        assert self.cfg.n_kv_heads is not None
        return KVCache(
            n_layers=self.cfg.n_layers,
            batch=batch,
            n_kv_heads=self.cfg.n_kv_heads,
            head_dim=self.cfg.head_dim,
            max_seq_len=self.cfg.max_seq_len,
            device=self.device,
            dtype=dtype or self.tok_emb.weight.dtype,
        )

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy / temperature-sampled generation with an incremental KV cache."""
        self.eval()
        b, t = tokens.shape
        cache = self.new_cache(b)

        # Prefill the prompt, then decode one token at a time.
        logits = self(tokens, cache=cache, start_pos=0)[:, -1, :]
        out = tokens
        pos = t
        for _ in range(max_new_tokens):
            next_tok = self._sample(logits, temperature, top_k)
            out = torch.cat([out, next_tok], dim=1)
            if eos_id is not None and bool((next_tok == eos_id).all()):
                break
            logits = self(next_tok, cache=cache, start_pos=pos)[:, -1, :]
            pos += 1
        return out

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
        if temperature <= 0.0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
