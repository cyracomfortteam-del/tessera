"""End-to-end knowledge distillation: a tessera-small teacher -> tessera-tiny student.

Run:  python examples/train_distill.py --steps 30
(uses synthetic data so it runs anywhere; swap in a real corpus + tokenizer to train for real)
"""

from __future__ import annotations

import argparse

import torch

from tessera.config import get_preset
from tessera.distill import DistillConfig, DistillTrainer, save_consolidated
from tessera.kernels import default_device
from tessera.model import Transformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--device", default=str(default_device()))
    ap.add_argument("--out", default=None, help="path to save the consolidated student")
    args = ap.parse_args()

    device = torch.device(args.device)
    teacher = Transformer(get_preset("tessera-small")).to(device).eval()
    student = Transformer(get_preset("tessera-tiny")).to(device)
    print(f"teacher {teacher.num_params():,} params  ->  student {student.num_params():,} params")

    cfg = DistillConfig(lr=3e-4, temperature=2.0, alpha=0.9, max_steps=args.steps)
    trainer = DistillTrainer(student, teacher, cfg)

    vocab = teacher.cfg.vocab_size
    torch.manual_seed(0)
    for step in range(args.steps):
        inputs = torch.randint(0, vocab, (args.batch, args.seq_len), device=device)
        targets = torch.randint(0, vocab, (args.batch, args.seq_len), device=device)
        metrics = trainer.train_step(inputs, targets)
        if step % 5 == 0 or step == args.steps - 1:
            print(f"step {step:3d}  kd={metrics['kd']:.4f}  loss={metrics['loss']:.4f}")

    if args.out:
        save_consolidated(trainer.fsdp, args.out)
        print(f"saved consolidated student -> {args.out}")


if __name__ == "__main__":
    main()
