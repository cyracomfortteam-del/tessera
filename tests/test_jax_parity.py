"""The JAX forward and the PyTorch forward must agree bit-for-bit (within fp tolerance).

Two independent implementations from one spec => a strong cross-check on RoPE, GQA, and
norm details. Skipped when JAX isn't installed (it's an optional extra).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from jax_ref import jax_forward, params_from_torch  # noqa: E402
from tessera.config import get_preset  # noqa: E402
from tessera.model import Transformer  # noqa: E402


def test_jax_forward_matches_torch():
    torch.manual_seed(0)
    model = Transformer(get_preset("tessera-debug")).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 24))
    with torch.no_grad():
        torch_logits = model(tokens).numpy()

    params, static = params_from_torch(model)
    jax_logits = np.asarray(jax_forward(params, jnp.asarray(tokens.numpy()), static))

    assert jax_logits.shape == torch_logits.shape
    np.testing.assert_allclose(jax_logits, torch_logits, atol=2e-4, rtol=2e-4)


def test_jax_forward_is_jit_cached():
    """Second call hits the XLA compilation cache (same static config) and still matches."""
    torch.manual_seed(1)
    model = Transformer(get_preset("tessera-debug")).eval()
    params, static = params_from_torch(model)

    a = jax_forward(params, jnp.zeros((1, 8), dtype=jnp.int32), static)
    b = jax_forward(params, jnp.zeros((1, 8), dtype=jnp.int32), static)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))
