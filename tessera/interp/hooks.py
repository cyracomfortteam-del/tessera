"""Activation capture via forward hooks.

A small `nnsight`/`TransformerLens`-flavoured cache: name the submodules you care about and
get their outputs back without editing the model. Used by the logit lens and the
induction-head probe.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActivationCache:
    """Context manager capturing the forward outputs of named submodules.

    >>> with ActivationCache(model, ["blocks.0.attn", "norm"]) as cache:
    ...     model(tokens)
    ... cache["blocks.0.attn"].shape
    """

    def __init__(self, model: nn.Module, names: list[str]):
        self.model = model
        self.names = names
        self.acts: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> ActivationCache:
        modules = dict(self.model.named_modules())
        for name in self.names:
            if name not in modules:
                raise KeyError(f"no submodule named {name!r}")
            self._handles.append(modules[name].register_forward_hook(self._make(name)))
        return self

    def _make(self, name: str):
        def hook(_module, _inputs, output):
            t = output[0] if isinstance(output, tuple) else output
            if isinstance(t, torch.Tensor):
                self.acts[name] = t.detach()

        return hook

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.acts[name]


@torch.no_grad()
def capture_residual_stream(model, tokens: torch.Tensor) -> torch.Tensor:
    """Per-layer residual-stream activations, shape (n_layers, B, T, dim)."""
    _, hiddens = model(tokens, return_hidden=True)
    return torch.stack(hiddens, dim=0)
