"""Mechanistic interpretability tools: activation hooks, logit lens, induction heads."""

from tessera.interp.hooks import ActivationCache, capture_residual_stream
from tessera.interp.induction import (
    attention_patterns,
    find_induction_heads,
    induction_score,
)
from tessera.interp.logit_lens import (
    layerwise_top_tokens,
    logit_lens,
    prediction_entropy,
)

__all__ = [
    "ActivationCache",
    "capture_residual_stream",
    "logit_lens",
    "layerwise_top_tokens",
    "prediction_entropy",
    "attention_patterns",
    "induction_score",
    "find_induction_heads",
]
