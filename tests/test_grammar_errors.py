"""Tests for the grammatical failure classifier used by stage 19.

The classifier decides whether an ablation broke ring tracking or merely broke
the model, so every property it relies on is pinned here: the categories are
mutually exclusive, truncation never masquerades as a grammar failure, and real
molecules never register as failures at all.
"""
import pytest

from molcircuits.chem.grammar_errors import (
    CATEGORIES,
    ErrorProfile,
    classify,
    profile,
    scan_grammar,
)
from molcircuits.chem.tokenizer import tokenize


# ------------------------------------------------------------------ scanner
def test_scan_clean_molecule_has_no_unmatched_tokens():
    g = scan_grammar(tokenize("CC(=O)Oc1ccccc1C(=O)O"))
    assert g["ring_unclosed"] == 0
    assert g["paren_unclosed"] == 0
    assert g["paren_unopened"] == 0


def test_scan_counts_two_digit_ring_labels_as_one_pair():
    """%10 is a single token; a scanner that re-read its digits would report an
    unclosed ring on a perfectly balanced molecule."""
    g = scan_grammar(tokenize("C%10CCCCC%10"))
    assert g["ring_unclosed"] == 0
    assert g["n_ring_tokens"] == 2


def test_scan_detects_each_unmatched_kind():
    assert scan_grammar(tokenize("c1ccccc"))["ring_unclosed"] == 1
    assert scan_grammar(tokenize("CC(C"))["paren_unclosed"] == 1
    assert scan_grammar(tokenize("CC)C"))["paren_unopened"] == 1


# ----------------------------------------------------------------- classify
@pytest.mark.parametrize("smiles,want", [
    ("CC(=O)Oc1ccccc1C(=O)O", "ok"),
    ("c1ccccc1", "ok"),
    ("c1ccccc", "ring_unclosed"),
    ("CC(C", "paren_unclosed"),
    ("CC)C", "paren_unopened"),
    ("C(C)(C)(C)(C)C", "valence"),
    ("", "parse"),
])
def test_classify_categories(smiles, want):
    assert classify(smiles, terminated=True) == want


def test_every_category_is_declared():
    for s in ("CCO", "c1ccccc", "CC(C", "CC)C", "C(C)(C)(C)(C)C", ""):
        assert classify(s, True) in CATEGORIES


def test_truncation_outranks_every_grammar_failure():
    """An EOS-suppressing ablation leaves rings and branches open for a reason
    that has nothing to do with ring tracking. If truncation did not win here,
    such an ablation would score as a ring failure and manufacture the result
    stage 19 exists to test."""
    assert classify("c1ccccc", terminated=False) == "truncated"
    assert classify("CC(C", terminated=False) == "truncated"


def test_syntax_outranks_chemistry():
    """A string that is both unclosed and chemically impossible is a syntax
    failure: the chemistry check never sees a complete graph to judge."""
    assert classify("C(C)(C)(C)(C)C1", True) == "ring_unclosed"


# ------------------------------------------------------------------ profile
def test_profile_excludes_truncated_from_the_denominator():
    smiles = ["CCO", "CCO", "c1ccccc", "CCO"]
    term = [True, True, True, False]
    p = profile(smiles, term)
    assert p.n == 4
    assert p.n_terminated == 3
    assert p.termination_rate == pytest.approx(0.75)
    # rates are over terminated sequences only
    assert p.validity == pytest.approx(2 / 3)
    assert p.ring_error_rate == pytest.approx(1 / 3)


def test_profile_rates_partition_the_terminated_population():
    smiles = ["CCO", "c1ccccc", "CC(C", "C(C)(C)(C)(C)C", "C11"]
    p = profile(smiles, [True] * len(smiles))
    assert p.ring_error_rate == pytest.approx(2 / 5)
    total = (p.validity + p.ring_error_rate + p.branch_error_rate
             + p.chemistry_error_rate)
    assert total == pytest.approx(1.0)


def test_empty_profile_does_not_divide_by_zero():
    p = ErrorProfile()
    assert p.validity != p.validity          # nan
    assert p.termination_rate != p.termination_rate


def test_real_molecules_are_never_grammar_failures():
    """Zero false positives is the property that makes a small rate difference
    between arms readable at all."""
    corpus = ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
              "C1CCNCC1", "c1ccc2ccccc2c1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
              "O=S(=O)(N)c1ccc(cc1)N", "C%10CCCCC%10", "FC(F)(F)c1ccccc1"]
    p = profile(corpus, [True] * len(corpus))
    assert p.validity == 1.0


def test_balanced_but_illegal_ring_closures_are_ring_failures():
    """They were classified `parse`, so a ring failure landed in the chemistry
    bucket and the ring-specific endpoint was biased downward."""
    assert classify("C11") == "ring_illegal"          # closes onto its own atom
    assert classify("C1C1") == "ring_illegal"         # re-bonds an existing bond
    assert classify("C1CCCCC1C11") == "ring_illegal"


def test_legal_rings_are_unaffected():
    for s in ("C1CCCCC1", "c1ccccc1", "C1CC1", "C1CCC2CCCCC2C1", "CC(C)(C)C"):
        assert classify(s) == "ok", s


def test_ring_illegal_does_not_shadow_an_unclosed_ring():
    assert classify("C1CC") == "ring_unclosed"


def test_illegal_ring_scan_counts_each_bad_closure_once():
    from molcircuits.chem.grammar_errors import scan_illegal_ring_bonds
    from molcircuits.chem.tokenizer import tokenize
    assert scan_illegal_ring_bonds(tokenize("C11")) == 1
    assert scan_illegal_ring_bonds(tokenize("C1CCCCC1")) == 0
