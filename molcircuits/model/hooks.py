"""Minimal hook manager: capture activations, or replace them mid-forward.

Two operations, kept separate on purpose:

* ``capture`` -- record an activation and pass it through unchanged.
* ``edit``    -- replace an activation with the output of a callback.

The SAE reconstruction-fidelity and steering experiments both need `edit`, and
both need the *same* edit path to be used for the baseline (with alpha=0 or with
an identity callback). Sharing one implementation is what makes an honest
alpha=0 control cheap; the previous codebase generated its baseline through a
different path entirely, which confounded steering with reconstruction error.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import torch


class HookManager:
    def __init__(self):
        self._captures: dict[str, list[torch.Tensor]] = OrderedDict()
        self._capture_names: set[str] = set()
        self._edits: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        self._detach = True

    # -- registration ---------------------------------------------------------
    def capture(self, *names: str, detach: bool = True) -> "HookManager":
        self._capture_names.update(names)
        self._detach = detach
        return self

    def edit(self, name: str, fn: Callable[[torch.Tensor], torch.Tensor]) -> "HookManager":
        self._edits[name] = fn
        return self

    def clear(self) -> "HookManager":
        self._captures.clear()
        return self

    # -- called from the model ------------------------------------------------
    def wants(self, name: str) -> bool:
        return name in self._capture_names or name in self._edits

    def emit(self, name: str, value: torch.Tensor) -> None:
        if name in self._capture_names:
            self._captures.setdefault(name, []).append(
                value.detach() if self._detach else value
            )

    def maybe(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self.emit(name, value)
        fn = self._edits.get(name)
        return fn(value) if fn is not None else value

    # -- retrieval ------------------------------------------------------------
    def __getitem__(self, name: str) -> torch.Tensor:
        vals = self._captures[name]
        if len(vals) == 1:
            return vals[0]
        # Chunks from separate forward passes concatenate along batch. Under the
        # KV cache each decode step contributes a (B, 1, D) chunk, so this would
        # silently return (B*steps, 1, D) -- batch and time conflated, no error.
        # Refuse rather than hand back a plausibly-shaped wrong tensor.
        # The incremental-decode case is CONSISTENT, not inconsistent: every
        # step contributes exactly (B, 1, D). The old guard tested for differing
        # shape[1] and therefore never fired on the one case its own comment
        # names, silently returning (B*steps, 1, D).
        if vals[0].dim() >= 2 and all(v.shape[1] == 1 for v in vals):
            raise ValueError(
                f"{name!r}: {len(vals)} captured chunks each of sequence length "
                f"1 -- this is incremental decode, and concatenating them on "
                f"the batch axis would conflate batch with time. Capture with "
                f"the KV cache disabled, or read the chunks yourself.")
        if vals[0].dim() >= 2 and any(v.shape[1] != vals[0].shape[1] for v in vals):
            raise ValueError(
                f"{name!r}: captured chunks have inconsistent sequence length "
                f"{[tuple(v.shape) for v in vals[:3]]}...; capturing across "
                "incremental decode steps is not supported")
        return torch.cat(vals, dim=0)

    def get(self, name: str, default=None):
        return self[name] if name in self._captures else default

    @property
    def names(self) -> list[str]:
        return list(self._captures)


def resid_post(layer: int) -> str:
    """Canonical SAE site: end-of-block residual stream, after the MLP add.

    Every SAE in this project is trained here. Stating it in one function rather
    than three prose descriptions is the point.
    """
    return f"blocks.{layer}.hook_resid_post"


def attn_pattern(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_pattern"
