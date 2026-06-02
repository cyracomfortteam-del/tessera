"""`tessera` command-line entry point.

    tessera info                 list model presets + parameter counts
    tessera generate --preset …  run the inference engine on random tokens
    tessera bench --preset …     quick forward-pass latency benchmark
"""

from __future__ import annotations

import argparse

import torch

from tessera.config import get_preset, list_presets
from tessera.kernels import default_device, triton_available
from tessera.model import Transformer
from tessera.profiling import time_op


def cmd_info(_args: argparse.Namespace) -> None:
    print(f"tessera — device={default_device()}  triton={'yes' if triton_available() else 'no'}")
    print(f"{'preset':<16}{'params':>14}{'layers':>8}{'dim':>6}{'heads(q/kv)':>14}")
    for name in list_presets():
        cfg = get_preset(name)
        heads = f"{cfg.n_heads}/{cfg.n_kv_heads}"
        print(f"{name:<16}{cfg.param_count():>14,}{cfg.n_layers:>8}{cfg.dim:>6}{heads:>14}")


def cmd_generate(args: argparse.Namespace) -> None:
    device = default_device()
    model = Transformer(get_preset(args.preset)).to(device).eval()
    prompt = torch.randint(0, model.cfg.vocab_size, (1, args.prompt_len), device=device)
    out = model.generate(prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(f"prompt {prompt.shape[1]} tok -> {out.shape[1]} tok")
    print("token ids:", out[0].tolist())


def cmd_bench(args: argparse.Namespace) -> None:
    device = default_device()
    model = Transformer(get_preset(args.preset)).to(device).eval()
    tokens = torch.randint(0, model.cfg.vocab_size, (args.batch, args.seq_len), device=device)

    @torch.no_grad()
    def fwd():
        return model(tokens)

    stats = time_op(fwd, warmup=3, iters=args.iters)
    toks = args.batch * args.seq_len
    print(f"preset={args.preset} batch={args.batch} seq={args.seq_len} device={device}")
    print(f"forward median: {stats['median_ms']:.2f} ms  "
          f"({toks / stats['median_ms'] * 1e3:,.0f} tok/s)")


def main() -> None:
    p = argparse.ArgumentParser(prog="tessera", description="Tessera LLM toolkit")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="list presets").set_defaults(func=cmd_info)

    g = sub.add_parser("generate", help="generate from an (untrained) preset")
    g.add_argument("--preset", default="tessera-tiny")
    g.add_argument("--prompt-len", type=int, default=8)
    g.add_argument("--max-new-tokens", type=int, default=32)
    g.add_argument("--temperature", type=float, default=0.8)
    g.set_defaults(func=cmd_generate)

    b = sub.add_parser("bench", help="benchmark a forward pass")
    b.add_argument("--preset", default="tessera-tiny")
    b.add_argument("--batch", type=int, default=4)
    b.add_argument("--seq-len", type=int, default=256)
    b.add_argument("--iters", type=int, default=20)
    b.set_defaults(func=cmd_bench)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
