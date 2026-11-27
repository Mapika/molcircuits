"""Shared nesting assignments for fragment-level inference.

Several SMARTS targets are variants of one chemical scaffold rather than
independent scientific questions.  Stages 15 and 17 both resample at this level;
keeping the assignments here prevents their uncertainty calculations drifting.
Unlisted fragments are treated as singleton families by :func:`family_map`.
"""
from __future__ import annotations

from collections.abc import Iterable

FRAGMENT_FAMILY = {
    "benzene": "arene", "pyridine": "arene", "pyrimidine": "arene",
    "phenol": "arene", "aniline": "arene",
    "fluoro_aromatic": "arene_halide", "chloro_aromatic": "arene_halide",
    "piperidine": "sat_N_ring", "piperazine": "sat_N_ring",
    "morpholine": "sat_N_ring", "cyclohexane": "sat_ring",
    "furan": "hetero5", "thiophene": "hetero5", "imidazole": "hetero5",
    "amide": "carbonyl", "urea": "carbonyl", "ester": "carbonyl",
    "ketone": "carbonyl", "carboxylic_acid": "carbonyl",
    "aldehyde": "carbonyl",
    "ether": "heteroatom", "alcohol": "heteroatom",
    "thioether": "heteroatom", "primary_amine": "heteroatom",
    "nitrile": "heteroatom", "nitro": "heteroatom",
    "sulfonamide": "sulfur", "sulfone": "sulfur",
    "trifluoromethyl": "alkyl_halide",
    "fluoro_aliphatic": "alkyl_halide",
    "chloro_aliphatic": "alkyl_halide",
    "tert_butyl": "alkyl", "isopropyl": "alkyl", "alkene": "alkyl",
}


def family_map(fragments: Iterable[str]) -> dict[str, str]:
    """Return shared family labels, assigning unknown targets individually."""
    return {name: FRAGMENT_FAMILY.get(name, name) for name in fragments}
