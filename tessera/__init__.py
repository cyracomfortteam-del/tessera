"""Tessera — a from-scratch LLM distillation & serving engine.

The public surface is intentionally small; sub-packages hold the real work:

    tessera.model    reference transformer (RMSNorm, RoPE, GQA, SwiGLU)
    tessera.kernels  Triton/CUDA fused kernels + the torch reference they target
    tessera.quant    weight-only int8 / AWQ / FP8 quantization
    tessera.serve    paged KV cache, continuous batching, speculative decoding
    tessera.distill  knowledge-distillation losses, trainer, FSDP-style sharding
    tessera.interp   activation hooks, logit lens, induction-head detection
    tessera.data     byte-level BPE tokenizer + multimodal preprocessing
"""

from tessera.config import ModelConfig, get_preset

__version__ = "0.4.0"

__all__ = ["ModelConfig", "get_preset", "__version__"]
