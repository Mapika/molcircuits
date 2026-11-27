#!/usr/bin/env python3
"""Stage 07: Sparse dictionary localization, downstream property transfer, and baseline SAE steering.

Subcommands:
  wsd-controls   Held-out weighted-support-difference (WSD) fragment localization vs matched controls.
  moleculeace    MoleculeACE 30-endpoint activity-cliff benchmark (dense vs SAE vs ECFP4).
  tdc-admet      Therapeutics Data Commons (TDC) ADMET transfer benchmarks.
  steering       Baseline single-feature SAE residual steering sweep.

Example:
  python experiments/07_sparse_and_downstream.py moleculeace --ckpt runs/tf_seed0/ckpt_final.pt --sae runs/saes_seed0/sae_L3.pt --out results/moleculeace
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
    "wsd-controls": (
        "_downstream.wsd_controls",
        "Held-out SMARTS fragment localization (WSD) for SAE vs dense/PCA/random-orthogonal bases.",
    ),
    "moleculeace": (
        "_downstream.moleculeace",
        "Evaluate dense residual, SAE, and ECFP4 representations across 30 MoleculeACE endpoints.",
    ),
    "tdc-admet": (
        "_downstream.tdc_admet",
        "Evaluate dense residual, SAE, and ECFP4 representations across TDC ADMET benchmarks.",
    ),
    "steering": (
        "_downstream.steering",
        "Baseline SAE feature activation steering sweep.",
    ),
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(
            prog="07_sparse_and_downstream.py",
            description="Sparse dictionary evaluation, downstream property benchmarks, and baseline steering.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Subcommands:\n" + "\n".join(
                f"  {name:<16} {desc}" for name, (_, desc) in SUBCOMMANDS.items()
            ) + "\n\nRun `07_sparse_and_downstream.py <subcommand> --help` for subcommand options.",
        )
        parser.add_argument("subcommand", choices=list(SUBCOMMANDS), help="Evaluation subcommand")
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
