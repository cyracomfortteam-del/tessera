"""End-to-end correctness for the reference transformer."""

import torch

from tessera.config import ModelConfig, get_preset
from tessera.model import Transformer


def _debug_model(seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(get_preset("tessera-debug")).eval()


def test_forward_shape():
    model = _debug_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, model.cfg.vocab_size)


def test_return_hidden_states():
    model = _debug_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 16))
    logits, hiddens = model(tokens, return_hidden=True)
    assert len(hiddens) == model.cfg.n_layers
    assert all(h.shape == (2, 16, model.cfg.dim) for h in hiddens)


def test_kv_cache_matches_full_forward():
    """Prefill + incremental decode must equal a single full-sequence forward pass.

    This is the load-bearing correctness test: it exercises RoPE position handling,
    causal masking with q_len < k_len, and the KV cache write/read path together.
    """
    model = _debug_model()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 20))

    full = model(tokens)

    cache = model.new_cache(batch=2)
    prefill = 12
    parts = [model(tokens[:, :prefill], cache=cache, start_pos=0)]
    for i in range(prefill, tokens.shape[1]):
        parts.append(model(tokens[:, i : i + 1], cache=cache, start_pos=i))
    incremental = torch.cat(parts, dim=1)

    assert torch.allclose(full, incremental, atol=1e-4, rtol=1e-4)


def test_generate_runs_and_is_deterministic_greedy():
    model = _debug_model()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 4))
    a = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    b = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert a.shape == (1, 12)
    assert torch.equal(a, b)  # greedy is deterministic


def test_param_count_matches_config_estimate():
    cfg = ModelConfig(vocab_size=512, dim=128, n_layers=4, n_heads=4, n_kv_heads=2)
    model = Transformer(cfg)
    # The analytic estimate should be within 1% of the realized count.
    actual = model.num_params()
    estimate = cfg.param_count()
    assert abs(actual - estimate) / actual < 0.01


def test_gqa_kv_projection_is_smaller():
    cfg = get_preset("tessera-tiny")
    model = Transformer(cfg)
    attn = model.blocks[0].attn
    assert attn.wk.weight.shape[0] == cfg.n_kv_heads * cfg.head_dim
    assert attn.wq.weight.shape[0] == cfg.n_heads * cfg.head_dim
