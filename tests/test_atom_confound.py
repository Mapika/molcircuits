"""A coordinate that only says "this is an atom" scores a large fragment WSD.

The confound the fragment section rests on: a SMARTS match is made of atoms,
while roughly half of a SMILES string is not (ring digits, bond symbols,
brackets, parentheses).  So a synthetic coordinate reading 1 on every atom token
and 0 on every other token separates in-fragment from out-of-fragment positions
substantially -- and clears the fragment's own permutation null -- while encoding
nothing whatever about chemistry.

Scored over atom tokens *only*, the identical coordinate must give exactly 0,
because it is then constant on the population being ranked.  That is the control
which makes the fragment numbers interpretable, and it is asserted here rather
than left as a one-off script, because it is the reason the reported scores are
restricted to atom positions.

Uses the same (B, T) mask machinery as stage 06, so this is the geometry the
run reports rather than a re-implementation of it.
"""
import numpy as np
import pytest

from molcircuits.analysis.wsd import best_detector, permutation_null
from molcircuits.chem.fragments import (atom_token_mask, build_masks, coverage,
                                    valid_token_mask)

N_CTX = 64

#: Half contain a benzene ring, half do not, so the fragment mask is balanced
#: enough for a permutation null to be meaningful at this size.
SMILES = [
    "c1ccccc1CCO", "c1ccccc1C(=O)N", "Cc1ccccc1O", "c1ccccc1CC(C)C",
    "Clc1ccccc1CN", "c1ccccc1S(=O)(=O)N", "COc1ccccc1C", "c1ccccc1CCN(C)C",
    "c1ccccc1OCC(=O)O", "Nc1ccccc1C(F)(F)F", "c1ccccc1CCCO", "c1ccccc1NC(C)=O",
    "C1CCCCC1CCO", "C1CCNCC1C(=O)N", "CC(C)CCO", "C1CCOCC1CC(C)C",
    "ClCC1CCNCC1", "C1CCCCC1S(=O)(=O)N", "COC1CCCCC1C", "C1CCNCC1CCN(C)C",
    "C1CCOCC1OCC(=O)O", "NC1CCCCC1C(F)(F)F", "C1CCCCC1CCCO", "C1CCNCC1NC(C)=O",
] * 3

BENZENE = "c1ccccc1"


@pytest.fixture(scope="module")
def masks():
    fragment = build_masks(SMILES, {"benzene": BENZENE}, N_CTX, offset=1)
    return {
        "fragment": fragment["benzene"],
        "atoms": atom_token_mask(SMILES, N_CTX, offset=1),
        "real": valid_token_mask(SMILES, N_CTX, offset=1),
    }


@pytest.fixture(scope="module")
def synthetic(masks):
    """1 on atom tokens, 0 on every other real token. Shape (B, T, 1)."""
    return masks["atoms"].astype(np.float32)[:, :, None]


def test_a_smiles_string_is_substantially_not_atoms(masks):
    """The premise of the confound. If atoms were nearly all of the string
    there would be nothing for the synthetic coordinate to exploit."""
    fraction = masks["atoms"].sum() / masks["real"].sum()
    assert 0.4 < fraction < 0.95


def test_the_fragment_mask_is_a_subset_of_atom_tokens(masks):
    """A SMARTS match is made of atoms; if it were not, the control below would
    not be measuring what it claims to."""
    assert not (masks["fragment"] & ~masks["atoms"]).any()


def test_synthetic_coordinate_scores_over_every_real_token(masks, synthetic):
    """It knows only 'this position is an atom' and still separates."""
    in_fragment = masks["fragment"] & masks["atoms"]
    score = best_detector(synthetic, in_fragment, masks["real"], signed=True)
    assert score > 0.1


def test_synthetic_coordinate_clears_its_own_permutation_null(masks, synthetic):
    """So clearing a permutation null is not evidence of a chemical feature."""
    in_fragment = masks["fragment"] & masks["atoms"]
    score = best_detector(synthetic, in_fragment, masks["real"], signed=True)
    null = permutation_null(synthetic, in_fragment, masks["real"],
                            n_perm=25, seed=0)
    assert score > null["null_p95"]


def test_scored_over_atoms_only_the_same_coordinate_gives_nothing(masks, synthetic):
    """The control the reported scores rest on: restricted to atom positions the
    coordinate is constant, so it can rank nothing and must score ~0."""
    in_fragment = masks["fragment"] & masks["atoms"]
    score = best_detector(synthetic, in_fragment, masks["atoms"], signed=True)
    assert abs(score) < 1e-6


def test_coverage_is_reported_against_the_atom_mask(masks):
    covered = coverage({"benzene": masks["fragment"]}, masks["atoms"])
    assert 0.0 < covered["benzene"] < 1.0
