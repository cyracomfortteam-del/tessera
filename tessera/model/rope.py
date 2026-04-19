"""Rotary position embeddings (GPT-NeoX / "rotate_half" convention).

We use the non-interleaved layout (first half / second half) rather than Llama's original
complex interleave. Both are correct as long as a single convention is used everywhere —
and it is: the torch model, the Triton kernels, and the JAX port all call into this layout,
which is what makes the cross-backend parity tests meaningful.
"""

from __future__ import annotations

import torch


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape (max_seq_len, head_dim)."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Rotate x of shape (B, H, T, D) by the angles at the given positions.

    `positions` is a LongTensor of shape (T,) (shared across the batch) or (B, T).
    """
    cos_p = cos[positions]
    sin_p = sin[positions]
    if cos_p.dim() == 2:  # (T, D) -> broadcast over B, H
        cos_p = cos_p[None, None, :, :]
        sin_p = sin_p[None, None, :, :]
    else:  # (B, T, D) -> broadcast over H
        cos_p = cos_p[:, None, :, :]
        sin_p = sin_p[:, None, :, :]
    return (x * cos_p) + (rotate_half(x) * sin_p)
