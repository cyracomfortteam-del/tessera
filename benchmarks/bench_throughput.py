"""Decode-throughput benchmark for the inference engine.

Measures end-to-end tokens/sec serving a batch of requests through the continuous-batching
engine, and reports the speculative-decoding acceptance rate (which, with a good draft,
directly multiplies decode throughput).
"""

from __future__ import annotations

import argparse
import time

import torch

from tessera.config import get_preset
from tessera.kernels import default_device
from tessera.model import Transformer
from tessera.serve import InferenceEngine, Request, SamplingParams, speculative_generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tessera-tiny")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    device = default_device()
    torch.manual_seed(0)
    model = Transformer(get_preset(args.preset)).to(device).eval()

    engine = InferenceEngine(model, max_batch=8, block_size=16, num_blocks=2048)
    reqs = [
        Request(f"r{i}", torch.randint(0, model.cfg.vocab_size, (8,)).tolist(),
                SamplingParams(max_new_tokens=args.max_new_tokens, temperature=0.0))
        for i in range(args.requests)
    ]

    t0 = time.perf_counter()
    results = engine.run(reqs)
    dt = time.perf_counter() - t0
    total = sum(len(o.output_tokens) for o in results.values())
    print(f"engine: {args.requests} reqs x {args.max_new_tokens} tok on {device}")
    print(f"  generated {total} tokens in {dt:.2f}s  ->  {total / dt:,.0f} tok/s")

    prompt = torch.randint(0, model.cfg.vocab_size, (8,)).tolist()
    res = speculative_generate(model, model, prompt, max_new_tokens=64, n_draft=4, temperature=0.0)
    print(f"  speculative acceptance (self-draft, greedy): {res.acceptance_rate:.0%}")


if __name__ == "__main__":
    main()
