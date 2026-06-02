"""Interpretability demo: logit-lens entropy by layer + induction-head scan.

Run:  python examples/interp_demo.py
"""

from __future__ import annotations

import torch

from tessera.config import get_preset
from tessera.interp import find_induction_heads, prediction_entropy
from tessera.model import Transformer


def main() -> None:
    torch.manual_seed(0)
    model = Transformer(get_preset("tessera-tiny")).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (2, 32))

    ent = prediction_entropy(model, tokens)
    print("logit-lens next-token entropy per layer (nats):")
    for layer, e in enumerate(ent.tolist()):
        bar = "#" * int(e / ent.max().item() * 30)
        print(f"  L{layer:2d} {e:5.2f} {bar}")

    scores, (best_layer, best_head) = find_induction_heads(model, period=16)
    print("\ninduction scan over a repeated length-16 sequence:")
    print(f"  strongest head: layer {best_layer}, head {best_head} "
          f"(score {scores[best_layer, best_head]:.3f})")
    print("  (an untrained model has weak induction; values rise sharply after training)")


if __name__ == "__main__":
    main()
