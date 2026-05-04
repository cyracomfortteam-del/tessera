"""Post-training quantization: weight-only int8, AWQ, and FP8 (E4M3)."""

from __future__ import annotations

import torch.nn as nn

from tessera.quant.awq import AWQLinear, collect_act_scale, search_awq_scale
from tessera.quant.fp8 import E4M3_MAX, fp8_linear, quantize_fp8
from tessera.quant.int8 import (
    QuantConfig,
    QuantLinear,
    dequantize_weight,
    quantize_weight,
)


def quantize_model(
    model: nn.Module,
    cfg: QuantConfig | None = None,
    skip: tuple[str, ...] = ("lm_head", "w_gate", "w_up"),
) -> nn.Module:
    """In-place swap of eligible nn.Linear layers for int8 QuantLinear.

    Layers whose qualified name contains any string in `skip` are left in fp. The LM head is
    kept high-precision; the SwiGLU gate/up projections are skipped because they feed the
    *fused* swiglu kernel (which consumes dense weight tensors directly). Only weights whose
    in_features divide the group size are converted; others are left intact.
    """
    cfg = cfg or QuantConfig()
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            qualified = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if any(s in qualified for s in skip):
                continue
            if child.in_features % cfg.group_size != 0:
                continue
            setattr(module, child_name, QuantLinear.from_linear(child, cfg))
    return model


__all__ = [
    "QuantConfig",
    "QuantLinear",
    "AWQLinear",
    "quantize_model",
    "quantize_weight",
    "dequantize_weight",
    "collect_act_scale",
    "search_awq_scale",
    "quantize_fp8",
    "fp8_linear",
    "E4M3_MAX",
]
