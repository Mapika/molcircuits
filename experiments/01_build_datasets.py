#!/usr/bin/env python3
"""Stage 01: Build canonical and randomized SMILES training/evaluation datasets.

Subcommands:
  canonical   Filter raw SMILES, Murcko-scaffold split, tokenize, and pack binary shards.
  randomized  Create a notation-randomized (non-canonical traversal) parallel training corpus.

Examples:
  python experiments/01_build_datasets.py canonical --input zinc20.smi --out data
  python experiments/01_build_datasets.py randomized --src data --out data_rand
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "canonical": (
        "_datasets.canonical",
        "Build scaffold-split canonical SMILES corpus, vocabulary, and packed token shards.",
    ),
    "randomized": (
        "_datasets.randomized",
        "Generate traversal-randomized SMILES training corpus sharing the canonical vocabulary.",
    ),
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            prog="01_build_datasets.py",
            description="Build canonical or randomized SMILES datasets for MolCircuits.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Subcommands:\n" + "\n".join(
                f"  {name:<12} {desc}" for name, (_, desc) in SUBCOMMANDS.items()
            ) + "\n\nRun `01_build_datasets.py <subcommand> --help` for subcommand options.",
        )
        parser.add_argument("subcommand", choices=list(SUBCOMMANDS), help="Dataset build mode")
        parser.parse_args(args)
        return

    cmd = args[0]
    if cmd not in SUBCOMMANDS:
        sys.exit(f"Unknown subcommand {cmd!r}. Choose from: {', '.join(SUBCOMMANDS)}")

    mod_name, _ = SUBCOMMANDS[cmd]
    mod = importlib.import_module(mod_name)
    saved_argv = sys.argv
    try:
        sys.argv = [f"{Path(__file__).name} {cmd}", *args[1:]]
        mod.main()
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()
