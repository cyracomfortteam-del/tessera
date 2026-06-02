"""Serve several requests through the continuous-batching engine, then compare plain
autoregressive decoding against speculative decoding.

Run:  python examples/serve.py
"""

from __future__ import annotations

import torch

from tessera.config import get_preset
from tessera.kernels import default_device
from tessera.model import Transformer
from tessera.profiling import time_op
from tessera.serve import InferenceEngine, Request, SamplingParams, speculative_generate


def main() -> None:
    device = default_device()
    torch.manual_seed(0)
    target = Transformer(get_preset("tessera-tiny")).to(device).eval()

    # --- continuous batching ------------------------------------------------
    engine = InferenceEngine(target, max_batch=4, block_size=16, num_blocks=512)
    reqs = [
        Request(f"r{i}", torch.randint(0, target.cfg.vocab_size, (6,)).tolist(),
                SamplingParams(max_new_tokens=20 + 4 * i, temperature=0.0))
        for i in range(5)
    ]
    results = engine.run(reqs)
    print("continuous batching:")
    for rid, out in sorted(results.items()):
        print(f"  {rid}: {len(out.output_tokens)} tokens generated")

    # --- speculative decoding (self-speculation here for a clean demo) -------
    prompt = torch.randint(0, target.cfg.vocab_size, (8,)).tolist()
    res = speculative_generate(target, target, prompt, max_new_tokens=48, n_draft=4, temperature=0.0)
    print(f"\nspeculative decoding: acceptance rate = {res.acceptance_rate:.1%}")

    # --- latency ------------------------------------------------------------
    tokens = torch.randint(0, target.cfg.vocab_size, (1, 128), device=device)
    stats = time_op(lambda: target(tokens), warmup=3, iters=20)
    print(f"prefill 128 tok: {stats['median_ms']:.2f} ms median on {device}")


if __name__ == "__main__":
    main()
