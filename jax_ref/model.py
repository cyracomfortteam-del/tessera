"""JAX implementation of the decoder, plus a converter from torch weights.

Everything is plain `jax.numpy` over a nested params dict (no Flax) to keep the dependency
surface tiny and the math transparent. `jax_forward` is `jax.jit`-friendly: the config is
passed as a static tuple so XLA can specialize and fuse the whole forward.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _build_rope(head_dim: int, max_seq: int, theta: float):
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    t = jnp.arange(max_seq, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rope(x, cos, sin, positions):
    c = cos[positions][None, None, :, :]
    s = sin[positions][None, None, :, :]
    return x * c + _rotate_half(x) * s


def _rmsnorm(x, weight, eps):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(var + eps)) * weight


def _attention(x, blk, cos, sin, n_heads, n_kv_heads, head_dim):
    b, t, _ = x.shape
    q = (x @ blk["wq"].T).reshape(b, t, n_heads, head_dim).transpose(0, 2, 1, 3)
    k = (x @ blk["wk"].T).reshape(b, t, n_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = (x @ blk["wv"].T).reshape(b, t, n_kv_heads, head_dim).transpose(0, 2, 1, 3)

    positions = jnp.arange(t)
    q = _apply_rope(q, cos, sin, positions)
    k = _apply_rope(k, cos, sin, positions)

    n_rep = n_heads // n_kv_heads
    k = jnp.repeat(k, n_rep, axis=1)
    v = jnp.repeat(v, n_rep, axis=1)

    scale = 1.0 / jnp.sqrt(jnp.array(head_dim, dtype=jnp.float32))
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    mask = jnp.tril(jnp.ones((t, t), dtype=bool))
    scores = jnp.where(mask[None, None], scores, -jnp.inf)
    attn = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
    out = out.transpose(0, 2, 1, 3).reshape(b, t, n_heads * head_dim)
    return out @ blk["wo"].T


def _swiglu(x, blk):
    gate = jax.nn.silu(x @ blk["w_gate"].T)
    up = x @ blk["w_up"].T
    return (gate * up) @ blk["w_down"].T


@partial(jax.jit, static_argnums=(2,))
def jax_forward(params, tokens, cfg_static):
    """Forward pass.

    `cfg_static` = (n_layers, n_heads, n_kv_heads, head_dim, max_seq, eps, theta). It is a
    static (hashable) arg so XLA specializes shapes and unrolls the layer loop.
    """
    n_layers, n_heads, n_kv_heads, head_dim, max_seq, eps, theta = cfg_static
    cos, sin = _build_rope(head_dim, max_seq, theta)

    h = params["tok_emb"][tokens]  # (B, T, dim)
    for i in range(n_layers):
        blk = params["blocks"][i]
        h = h + _attention(
            _rmsnorm(h, blk["attn_norm"], eps), blk, cos, sin, n_heads, n_kv_heads, head_dim
        )
        h = h + _swiglu(_rmsnorm(h, blk["ffn_norm"], eps), blk)
    h = _rmsnorm(h, params["norm"], eps)
    return h @ params["lm_head"].T


def params_from_torch(model) -> tuple[dict, tuple]:
    """Extract a JAX params pytree + static config tuple from a torch Transformer."""
    import torch

    def arr(t):
        return jnp.asarray(t.detach().cpu().to(torch.float32).numpy())

    cfg = model.cfg
    blocks = []
    for blk in model.blocks:
        blocks.append(
            {
                "attn_norm": arr(blk.attn_norm.weight),
                "wq": arr(blk.attn.wq.weight),
                "wk": arr(blk.attn.wk.weight),
                "wv": arr(blk.attn.wv.weight),
                "wo": arr(blk.attn.wo.weight),
                "ffn_norm": arr(blk.ffn_norm.weight),
                "w_gate": arr(blk.mlp.w_gate.weight),
                "w_up": arr(blk.mlp.w_up.weight),
                "w_down": arr(blk.mlp.w_down.weight),
            }
        )
    params = {
        "tok_emb": arr(model.tok_emb.weight),
        "blocks": blocks,
        "norm": arr(model.norm.weight),
        "lm_head": arr(model.lm_head.weight),
    }
    static = (
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        int(cfg.max_seq_len),
        float(cfg.norm_eps),
        float(cfg.rope_theta),
    )
    return params, static
