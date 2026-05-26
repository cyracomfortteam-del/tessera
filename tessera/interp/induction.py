"""Induction-head detection.

Induction heads (Elhage et al. 2021; Olsson et al. 2022) implement in-context copying: at a
token that previously appeared, the head attends to *the token that followed it last time*
and copies it. On a repeated random sequence [s_0..s_{P-1}, s_0..s_{P-1}] this shows up as a
bright off-diagonal "stripe" at key = query - P + 1.

We recover the attention patterns by capturing the q/k projections with forward hooks and
recomputing the (RoPE'd, GQA-expanded, causal) softmax exactly as the attention layer does —
no model surgery required.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tessera.model.rope import apply_rope


@torch.no_grad()
def attention_patterns(model, tokens: torch.Tensor) -> torch.Tensor:
    """All attention maps, shape (n_layers, B, n_heads, T, T). Rows are valid softmaxes."""
    cfg = model.cfg
    b, t = tokens.shape
    q_store: dict[int, torch.Tensor] = {}
    k_store: dict[int, torch.Tensor] = {}
    handles = []

    def saver(store, idx):
        def hook(_m, _i, out):
            store[idx] = out.detach()

        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.attn.wq.register_forward_hook(saver(q_store, i)))
        handles.append(blk.attn.wk.register_forward_hook(saver(k_store, i)))
    model(tokens)
    for h in handles:
        h.remove()

    positions = torch.arange(t, device=tokens.device)
    scale = 1.0 / math.sqrt(cfg.head_dim)
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=tokens.device))

    patterns = []
    for i in range(cfg.n_layers):
        q = q_store[i].view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = k_store[i].view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        q = apply_rope(q, model.rope_cos, model.rope_sin, positions)
        k = apply_rope(k, model.rope_cos, model.rope_sin, positions)
        k = k.repeat_interleave(cfg.n_heads // cfg.n_kv_heads, dim=1)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~causal, float("-inf"))
        patterns.append(F.softmax(scores, dim=-1))
    return torch.stack(patterns, dim=0)


def induction_score(patterns: torch.Tensor, period: int) -> torch.Tensor:
    """Per-head induction score, shape (n_layers, n_heads).

    Average attention mass on the induction stripe (key = query - period + 1) over query
    positions in the repeated second half.
    """
    n_layers, b, n_heads, t, _ = patterns.shape
    scores = torch.zeros(n_layers, n_heads)
    count = 0
    for query in range(period, t):
        key = query - period + 1
        if key < 0:
            continue
        scores += patterns[:, :, :, query, key].mean(dim=1)  # avg over batch
        count += 1
    return scores / max(1, count)


@torch.no_grad()
def find_induction_heads(model, period: int = 16, seed: int = 0):
    """Build a repeated random sequence, return (scores (L,H), best (layer, head))."""
    torch.manual_seed(seed)
    vocab = model.cfg.vocab_size
    half = torch.randint(0, vocab, (1, period))
    tokens = torch.cat([half, half], dim=1)  # [s, s]
    patterns = attention_patterns(model, tokens)
    scores = induction_score(patterns, period)
    flat = int(scores.argmax())
    best = (flat // scores.shape[1], flat % scores.shape[1])
    return scores, best
