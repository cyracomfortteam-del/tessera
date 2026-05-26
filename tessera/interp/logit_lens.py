"""Logit lens (nostalgebraist 2020).

Decode every layer's residual stream through the *final* norm + unembedding to see what the
model would predict if it stopped at that layer. Predictions typically sharpen with depth —
a quick, model-agnostic probe of where computation 'commits' to an answer.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def logit_lens(model, tokens: torch.Tensor) -> torch.Tensor:
    """Return per-layer logits of shape (n_layers, B, T, vocab).

    The last layer's slice equals the model's actual output (same norm + head), which the
    tests assert as a correctness anchor.
    """
    _, hiddens = model(tokens, return_hidden=True)
    per_layer = [model.lm_head(model.norm(h)) for h in hiddens]
    return torch.stack(per_layer, dim=0)


@torch.no_grad()
def layerwise_top_tokens(model, tokens: torch.Tensor) -> torch.Tensor:
    """Argmax token id per (layer, batch, position): shape (n_layers, B, T)."""
    return logit_lens(model, tokens).argmax(dim=-1)


@torch.no_grad()
def prediction_entropy(model, tokens: torch.Tensor) -> torch.Tensor:
    """Mean next-token entropy per layer (nats), shape (n_layers,). Tends to decrease."""
    lens = logit_lens(model, tokens)
    probs = torch.softmax(lens, dim=-1)
    ent = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)
    return ent.mean(dim=(1, 2))
