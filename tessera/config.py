"""Model configuration and a small registry of presets.

A Llama-style decoder is fully described by this dataclass; everything downstream
(reference model, kernels, quantizer, JAX port) reads from it so a single config
keeps every implementation in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyper-parameters for a decoder-only transformer with GQA + SwiGLU + RoPE."""

    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    # Grouped-query attention: set n_kv_heads < n_heads to share KV projections.
    # None means full multi-head attention (n_kv_heads == n_heads).
    n_kv_heads: int | None = None
    # SwiGLU inner dimension. None => derived as round(8/3 * dim) up to a multiple.
    ffn_hidden: int | None = None
    ffn_multiple_of: int = 64
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim={self.dim} not divisible by n_heads={self.n_heads}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} not divisible by n_kv_heads={self.n_kv_heads}"
            )
        if self.ffn_hidden is None:
            # Llama/SwiGLU convention: 2/3 of the usual 4*dim, then round up.
            hidden = int(8 * self.dim / 3)
            m = self.ffn_multiple_of
            self.ffn_hidden = m * ((hidden + m - 1) // m)

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head."""
        assert self.n_kv_heads is not None
        return self.n_heads // self.n_kv_heads

    def param_count(self) -> int:
        """Approximate parameter count (embeddings + blocks + final norm)."""
        assert self.n_kv_heads is not None
        emb = self.vocab_size * self.dim
        # attention: q (dim*dim) + k,v (dim * n_kv_heads*head_dim) + o (dim*dim)
        kv_dim = self.n_kv_heads * self.head_dim
        attn = self.dim * self.dim + 2 * self.dim * kv_dim + self.dim * self.dim
        # swiglu: gate + up (dim->hidden) + down (hidden->dim)
        mlp = 3 * self.dim * self.ffn_hidden
        norms = 2 * self.dim  # two RMSNorm per block
        per_block = attn + mlp + norms
        head = 0 if self.tie_embeddings else self.vocab_size * self.dim
        return emb + self.n_layers * per_block + self.dim + head


# Teacher/student presets used by the distillation pipeline and the demos.
_PRESETS: dict[str, ModelConfig] = {
    # ~6M params — fast student, the thing we actually want to serve.
    "tessera-tiny": ModelConfig(
        vocab_size=8192, dim=256, n_layers=6, n_heads=8, n_kv_heads=2,
        max_seq_len=1024,
    ),
    # ~45M params — the teacher we distill from.
    "tessera-small": ModelConfig(
        vocab_size=8192, dim=512, n_layers=12, n_heads=8, n_kv_heads=4,
        max_seq_len=1024,
    ),
    # A debug-sized model that fits comfortably on CPU for tests.
    "tessera-debug": ModelConfig(
        vocab_size=256, dim=64, n_layers=2, n_heads=4, n_kv_heads=2,
        max_seq_len=128,
    ),
}


def get_preset(name: str) -> ModelConfig:
    if name not in _PRESETS:
        raise KeyError(f"unknown preset {name!r}; choices: {sorted(_PRESETS)}")
    # Return a copy so callers can mutate without corrupting the registry.
    base = _PRESETS[name]
    return ModelConfig(**{k: getattr(base, k) for k in base.__dataclass_fields__})


def list_presets() -> list[str]:
    return sorted(_PRESETS)
