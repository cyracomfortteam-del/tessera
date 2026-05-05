"""Quantization correctness: int8 round-trip, AWQ improvement, FP8 grid, model swap."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from tessera.config import get_preset
from tessera.model import Transformer
from tessera.quant import (
    QuantConfig,
    QuantLinear,
    quantize_fp8,
    quantize_model,
    quantize_weight,
)
from tessera.quant.awq import search_awq_scale
from tessera.quant.int8 import dequantize_weight


def test_int8_roundtrip_error_small():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q, scale, zeros = quantize_weight(w, group_size=128, symmetric=True)
    assert q.dtype == torch.int8
    assert scale.shape == (64, 256 // 128)
    w_deq = dequantize_weight(q, scale, zeros, 128)
    rel = (w - w_deq).abs().mean() / w.abs().mean()
    assert rel < 0.02  # < 2% mean error for symmetric per-group int8


def test_quant_linear_matches_fp():
    torch.manual_seed(0)
    lin = nn.Linear(128, 64, bias=False)
    x = torch.randn(4, 10, 128)
    ref = lin(x)
    q = QuantLinear.from_linear(lin, QuantConfig(group_size=128))
    out = q(x)
    assert out.shape == ref.shape
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    assert cos > 0.999


def _awq_weighted_err(w, act_scale, scale_vec):
    ws = w * scale_vec[None, :]
    q, sc, _ = quantize_weight(ws, 128, True)
    wd = dequantize_weight(q, sc, None, 128) / scale_vec[None, :]
    return ((w - wd).abs() * act_scale[None, :]).mean()


def test_awq_never_worse_than_naive():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    act_scale = torch.ones(128)
    act_scale[:4] = 12.0  # a few salient input channels

    naive = _awq_weighted_err(w, act_scale, torch.ones(128))
    s = search_awq_scale(w, act_scale, group_size=128)
    awq = _awq_weighted_err(w, act_scale, s)
    # The grid includes the identity (ratio=0), so AWQ can never be worse.
    assert awq <= naive + 1e-8


def test_awq_reduces_to_identity_without_saliency():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    uniform = torch.ones(128)  # no channel is more salient than another
    s = search_awq_scale(w, uniform, group_size=128)
    # With uniform activations s = 1^ratio / mean = ones, so AWQ == naive.
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_fp8_idempotent_and_bounded():
    torch.manual_seed(0)
    x = torch.randn(1000) * 3.0
    xq, scale = quantize_fp8(x)
    # Re-quantizing a quantized tensor is a no-op (already on the grid).
    xqq, _ = quantize_fp8(xq, scale)
    assert torch.allclose(xq, xqq, atol=1e-6)
    rel = (x - xq).abs().mean() / x.abs().mean()
    assert rel < 0.06  # 3-bit mantissa => a few % mean error


def test_quantize_model_runs_and_preserves_output():
    torch.manual_seed(0)
    model = Transformer(get_preset("tessera-debug")).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 16))
    with torch.no_grad():
        ref = model(tokens)

    n_lin_before = sum(isinstance(m, nn.Linear) for m in model.modules())
    quantize_model(model, QuantConfig(group_size=64))
    n_quant = sum(isinstance(m, QuantLinear) for m in model.modules())
    assert n_quant > 0
    # lm_head is tied + skipped, so at least one Linear remains.
    assert n_lin_before > n_quant

    with torch.no_grad():
        out = model(tokens)
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
    assert cos > 0.99
