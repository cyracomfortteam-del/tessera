"""Speculative decoding correctness against the autoregressive baseline."""

import torch

from tessera.config import get_preset
from tessera.model import Transformer
from tessera.serve.speculative import speculative_generate


def _model(seed=0):
    torch.manual_seed(seed)
    return Transformer(get_preset("tessera-debug")).eval()


def test_self_speculation_matches_greedy():
    """draft == target with greedy sampling must reproduce target.generate exactly and
    accept every proposal (the verifier never disagrees with itself)."""
    model = _model()
    prompt = [1, 5, 9, 3]
    greedy = model.generate(
        torch.tensor([prompt]), max_new_tokens=16, temperature=0.0
    )[0].tolist()

    res = speculative_generate(
        model, model, prompt, max_new_tokens=16, n_draft=4, temperature=0.0
    )
    assert res.tokens == greedy
    assert res.acceptance_rate == 1.0


def test_speculative_runs_with_weaker_draft():
    target = _model(0)
    draft = _model(1)  # different weights -> some rejections
    prompt = [2, 7, 1]
    res = speculative_generate(
        target, draft, prompt, max_new_tokens=20, n_draft=4, temperature=1.0
    )
    assert len(res.tokens) == len(prompt) + 20
    assert 0.0 <= res.acceptance_rate <= 1.0


def test_speculative_output_length_exact():
    model = _model()
    for n_draft in (1, 3, 8):
        res = speculative_generate(
            model, model, [4, 4], max_new_tokens=10, n_draft=n_draft, temperature=0.0
        )
        assert len(res.tokens) == 2 + 10
