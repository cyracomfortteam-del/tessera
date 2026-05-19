"""Knowledge distillation: KD losses, from-scratch FSDP, fault-tolerant checkpointing."""

from tessera.distill.checkpoint import (
    load_sharded,
    resume_or_init,
    save_consolidated,
    save_sharded,
)
from tessera.distill.fsdp import DistGroup, FullyShardedModule, LocalGroup
from tessera.distill.losses import (
    HiddenStateKD,
    distillation_loss,
    hard_ce,
    soft_target_kd,
)
from tessera.distill.trainer import DistillConfig, DistillTrainer

__all__ = [
    "soft_target_kd",
    "hard_ce",
    "distillation_loss",
    "HiddenStateKD",
    "FullyShardedModule",
    "LocalGroup",
    "DistGroup",
    "save_sharded",
    "load_sharded",
    "resume_or_init",
    "save_consolidated",
    "DistillTrainer",
    "DistillConfig",
]
