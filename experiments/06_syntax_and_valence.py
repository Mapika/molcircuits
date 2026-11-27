#!/usr/bin/env python3
"""Stage 06: String-grammar circuits, attention pointers, patching, and readouts.

Subcommands:
  pointers              Identify ring-closure, branch-return, and aromaticity pointer heads.
  ring-patching         Causal activation patching on ring-closure digit identity.
  valence-patching      Causal activation patching on unsaturated/saturated bond decisions.
  readouts              Layer-wise probe readouts for ring, branch, aromaticity, and valence.
  split-generalization  Held-out Murcko scaffold vs unseen-row cross-entropy evaluation.

Example:
  python experiments/06_syntax_and_valence.py pointers --ckpt runs/tf_seed0/ckpt_final.pt --data data --out results/pointers
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
    "pointers": (
        "_syntax.pointers",
        "Rank syntax pointer attention heads and measure ablation effects.",
    ),
    "ring-patching": (
        "_syntax.ring_patching",
        "Causal activation patching of ring-closure digit identity across layers.",
    ),
    "valence-patching": (
        "_syntax.valence_patching",
        "Causal activation patching of bond-decision valence state across layers.",
    ),
    "readouts": (
        "_syntax.readouts",
        "Layer-wise linear probe readouts across canonical model grid.",
    ),
    "split-generalization": (
        "_syntax.split_generalization",
        "Compare validation loss across unseen training rows vs held-out Murcko scaffolds.",
    ),
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            prog="06_syntax_and_valence.py",
            description="String-grammar circuits, causal patching, and readout evaluations.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Subcommands:\n" + "\n".join(
                f"  {name:<22} {desc}" for name, (_, desc) in SUBCOMMANDS.items()
            ) + "\n\nRun `06_syntax_and_valence.py <subcommand> --help` for subcommand options.",
        )
        parser.add_argument("subcommand", choices=list(SUBCOMMANDS), help="Analysis subcommand")
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
