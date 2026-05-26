"""Interpretability tools: hooks, logit lens, attention patterns, induction scoring."""

import torch

from tessera.config import get_preset
from tessera.interp import (
    ActivationCache,
    attention_patterns,
    capture_residual_stream,
    find_induction_heads,
    induction_score,
    logit_lens,
)
from tessera.model import Transformer


def _model(seed=0):
    torch.manual_seed(seed)
    return Transformer(get_preset("tessera-debug")).eval()


def test_activation_cache_captures_named_modules():
    model = _model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 12))
    with ActivationCache(model, ["blocks.0.attn", "norm"]) as cache:
        model(tokens)
    assert cache["blocks.0.attn"].shape == (2, 12, model.cfg.dim)
    assert cache["norm"].shape == (2, 12, model.cfg.dim)


def test_residual_stream_shape():
    model = _model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 12))
    res = capture_residual_stream(model, tokens)
    assert res.shape == (model.cfg.n_layers, 2, 12, model.cfg.dim)


def test_logit_lens_last_layer_equals_model_output():
    model = _model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 10))
    with torch.no_grad():
        ref = model(tokens)
    lens = logit_lens(model, tokens)
    assert lens.shape == (model.cfg.n_layers, 2, 10, model.cfg.vocab_size)
    torch.testing.assert_close(lens[-1], ref, atol=1e-5, rtol=1e-5)


def test_attention_patterns_are_valid_and_causal():
    model = _model()
    tokens = torch.randint(0, model.cfg.vocab_size, (1, 16))
    patterns = attention_patterns(model, tokens)
    assert patterns.shape == (model.cfg.n_layers, 1, model.cfg.n_heads, 16, 16)
    # rows sum to 1 (valid softmax)
    sums = patterns.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)
    # strictly causal: no mass above the diagonal
    upper = torch.triu(torch.ones(16, 16), diagonal=1).bool()
    assert patterns[..., upper].abs().max() < 1e-6


def test_induction_score_range_and_shape():
    model = _model()
    scores, best = find_induction_heads(model, period=8)
    assert scores.shape == (model.cfg.n_layers, model.cfg.n_heads)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert 0 <= best[0] < model.cfg.n_layers
    assert 0 <= best[1] < model.cfg.n_heads


def test_induction_score_detects_synthetic_stripe():
    # A hand-built pattern that always attends to key = query - period + 1 should score ~1.
    n_layers, b, n_heads, t, period = 1, 1, 1, 12, 4
    patterns = torch.zeros(n_layers, b, n_heads, t, t)
    for q in range(t):
        patterns[..., q, max(0, q - period + 1)] = 1.0
    score = induction_score(patterns, period)
    assert score[0, 0] > 0.99
