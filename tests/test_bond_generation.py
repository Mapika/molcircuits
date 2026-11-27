"""Chemical denominators, pairing, and heavy-atom standardization for bond generation."""
import pytest

from molcircuits.analysis.bond_generation import (
    heavy_atom_standardized,
    measure_attempt,
    paired_contrast,
    summarize,
)


def rows(*smiles):
    return [measure_attempt(i, *value) for i, value in enumerate(smiles)]


def test_primary_counts_sanitized_graph_bonds_only_after_eos():
    records = rows(("C=C", True), ("C#N", True), ("C1=CC=CC=C1", True),
                   ("C", True), ("C1", True), ("C=C", False), ("", True))
    benzene = records[2]
    assert (benzene["heavy_bonds"], benzene["aromatic"], benzene["double"]) == (6, 6, 0)
    stats = summarize(records)
    assert (stats["n_attempted"], stats["n_terminated"], stats["n_complete_valid"],
            stats["n_bond_eligible"], stats["n_parseable_without_eos"]) == (7, 6, 4, 3, 1)
    assert stats["multiple_fraction_valid"] == pytest.approx(2 / 3)
    assert stats["complete_valid_rate"] == pytest.approx(4 / 7)
    assert stats["double_count_per_attempt"] == stats["triple_count_per_attempt"] == pytest.approx(1 / 7)
    assert stats["mean_heavy_atoms_valid"] == pytest.approx(11 / 4)
    assert records[5]["heavy_atoms"] is None


def test_equal_molecular_weight_and_empty_eligible_endpoint():
    assert summarize(rows(("C=C", True), ("CCCC", True)))["multiple_fraction_valid"] == 0.5
    no_bonds = rows(("C", True), ("C.C", True))
    assert summarize(no_bonds)["multiple_fraction_valid"] is None
    contrast = paired_contrast(no_bonds, no_bonds, 10, 8)
    assert contrast["multiple_fraction_valid"] == {"ci95": [None, None], "difference": None,
                                                 "n_bootstrap_finite": 0}
    assert contrast["double_count_per_attempt"]["ci95"] == [0.0, 0.0]


def test_pairing_and_heavy_atom_standardization_across_common_sizes():
    left = rows(("C=C", True), ("CC", True), ("C#N", True), ("CCCC", True))
    ref = rows(("CC", True), ("CCC", True), ("CCCC", True), ("CCCC", True))
    contrast = paired_contrast(left, ref, 100, 123)
    assert contrast["multiple_fraction_valid"]["difference"] == 0.5
    sz = heavy_atom_standardized(left, ref)
    assert (sz["n_left_retained"], sz["n_reference_retained"], sz["n_common_sizes"]) == (4, 3, 2)
    assert sz["difference"] == pytest.approx(2 / 9)
    identical = paired_contrast(left, left, 50, 0)
    assert identical["multiple_fraction_valid"]["ci95"] == [0, 0]
    with pytest.raises(ValueError, match="matched"):
        paired_contrast(left, list(reversed(ref)), 50, 0)
