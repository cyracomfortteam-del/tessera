"""The torch reference ops are the spec the kernels target — pin their behaviour."""

import math

import torch
import torch.nn.functional as F

from tessera.kernels.reference import (
    attention_ref,
    rmsnorm_ref,
    swiglu_ref,
)


def test_rmsnorm_matches_manual():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 64)
    w = torch.randn(64)
    out = rmsnorm_ref(x, w, eps=1e-5)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    assert torch.allclose(out, ref, atol=1e-5)


def test_attention_matches_sdpa_mha():
    torch.manual_seed(0)
    b, h, t, d = 2, 4, 12, 16
    q, k, v = (torch.randn(b, h, t, d) for _ in range(3))
    out = attention_ref(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_attention_gqa_broadcasts_kv():
    torch.manual_seed(0)
    b, hq, hkv, t, d = 2, 8, 2, 10, 16
    q = torch.randn(b, hq, t, d)
    k = torch.randn(b, hkv, t, d)
    v = torch.randn(b, hkv, t, d)
    out = attention_ref(q, k, v, causal=True)
    # Manual reference: expand kv then SDPA.
    n_rep = hq // hkv
    ke = k.repeat_interleave(n_rep, dim=1)
    ve = v.repeat_interleave(n_rep, dim=1)
    ref = F.scaled_dot_product_attention(q, ke, ve, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-5)


def test_attention_decode_alignment():
    """A short query against a longer cache attends to the cache tail (causal offset)."""
    torch.manual_seed(0)
    b, h, d = 1, 2, 8
    full_t = 6
    q_full = torch.randn(b, h, full_t, d)
    k = torch.randn(b, h, full_t, d)
    v = torch.randn(b, h, full_t, d)
    full = attention_ref(q_full, k, v, causal=True)
    # Feed only the last query token but the whole cache.
    last = attention_ref(q_full[:, :, -1:, :], k, v, causal=True)
    assert torch.allclose(last[:, :, 0], full[:, :, -1], atol=1e-5)


def test_swiglu_matches_manual():
    torch.manual_seed(0)
    x = torch.randn(3, 7, 32)
    wg = torch.randn(64, 32)
    wu = torch.randn(64, 32)
    out = swiglu_ref(x, wg, wu)
    ref = F.silu(x @ wg.T) * (x @ wu.T)
    assert torch.allclose(out, ref, atol=1e-5)
    assert out.shape == (3, 7, 64)


def test_softmax_scale_default():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, 5, 9) for _ in range(3))
    out = attention_ref(q, k, v, causal=False, softmax_scale=1.0 / math.sqrt(9))
    out2 = attention_ref(q, k, v, causal=False)
    assert torch.allclose(out, out2, atol=1e-6)
