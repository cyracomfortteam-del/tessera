"""A pure-JAX/XLA reference implementation of the Tessera transformer.

Why keep a second implementation? It's an independent oracle: the JAX forward and the
PyTorch forward are written from the same spec but share no code, so the parity test
(`tests/test_jax_parity.py`) catches subtle bugs in either one (RoPE convention, GQA
broadcast, norm epsilon placement). It's also the on-ramp to the JAX/XLA + `pjit`/`shard_map`
training path used in the DeepMind-style stack.
"""

from jax_ref.model import jax_forward, params_from_torch

__all__ = ["jax_forward", "params_from_torch"]
