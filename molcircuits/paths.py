"""Lazy, environment-overridable path resolution for MolCircuits runs.

Bulk datasets, model checkpoints, and dense caches live outside the source tree
and are resolved lazily from environment variables on every call::

    MOLCIRCUITS_ROOT       -> ./runs
    MOLCIRCUITS_ARTIFACTS  -> $MOLCIRCUITS_ROOT/artifacts
    MOLCIRCUITS_DATA       -> $MOLCIRCUITS_ROOT/data
    MOLCIRCUITS_TOKENIZER  -> unset; loaders fall back to checkpoint vocabulary
    MOLCIRCUITS_MANIFEST   -> results/manifest.jsonl

Setting ``MOLCIRCUITS_ROOT`` relocates the whole run tree; setting
``MOLCIRCUITS_ARTIFACTS`` alone redirects artifact readers/writers while
keeping the shared corpus untouched.
"""
from __future__ import annotations

import os

DEFAULT_ROOT = "runs"
DEFAULT_MANIFEST = "results/manifest.jsonl"


def root() -> str:
    """Base directory of the bulk run tree."""
    return os.environ.get("MOLCIRCUITS_ROOT", DEFAULT_ROOT)


def artifacts() -> str:
    """Directory holding per-stage, per-seed run artifacts."""
    return os.environ.get("MOLCIRCUITS_ARTIFACTS") or os.path.join(root(), "artifacts")


def data() -> str:
    """Directory holding corpora and packed token arrays."""
    return os.environ.get("MOLCIRCUITS_DATA") or os.path.join(root(), "data")


def tokenizer() -> str | None:
    """Explicit tokenizer path, or ``None`` to let the checkpoint vocabulary win."""
    return os.environ.get("MOLCIRCUITS_TOKENIZER") or None


def manifest() -> str:
    """Path to the JSONL provenance manifest."""
    return os.environ.get("MOLCIRCUITS_MANIFEST") or DEFAULT_MANIFEST


def describe() -> dict[str, str | None]:
    """Return all resolved root paths for provenance logging."""
    return {
        "root": root(),
        "artifacts": artifacts(),
        "data": data(),
        "tokenizer": tokenizer(),
        "manifest": manifest(),
    }
