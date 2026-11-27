"""Valence-shaped means mass moves between bond orders, not that it all moves."""
from molcircuits.analysis.valence_shape import (linear_response, monotone,
                                            reallocates, rows_by_alpha,
                                            shape_signs, sign_reverses,
                                            summarize_direction)

SHAPED = {"dprob_-": -0.02, "dprob_=": 0.012, "dprob_#": 0.008}
COMMON = {"dprob_-": 0.02, "dprob_=": 0.012, "dprob_#": 0.008}


def test_the_shaped_pattern_is_recognised():
    out = shape_signs(SHAPED, 1.0)
    assert out["valence_shaped"]
    assert not out["common_mode"]
    assert out["n_matching"] == 3


def test_a_common_mode_shift_is_not_valence_shaped():
    """Everything moving the same way is a rescaling of the distribution and
    encodes nothing about valence -- the distinction this module exists for."""
    out = shape_signs(COMMON, 1.0)
    assert not out["valence_shaped"]
    assert out["common_mode"]


def test_negative_alpha_is_folded_onto_the_positive_convention():
    """A direction that reverses correctly must not read as contradicting
    itself on the negative half of the sweep."""
    mirrored = {k: -v for k, v in SHAPED.items()}
    assert shape_signs(mirrored, -1.0)["valence_shaped"]
    assert not shape_signs(mirrored, 1.0)["valence_shaped"]


def test_reallocation_needs_opposite_signs_across_the_bond_order_split():
    assert reallocates(SHAPED)
    assert not reallocates(COMMON)


def test_reallocation_sums_double_and_triple():
    """A tiny noisy triple-bond delta must not veto a clear reallocation."""
    row = {"dprob_-": -0.02, "dprob_=": 0.03, "dprob_#": -0.001}
    assert reallocates(row)


def test_sign_reversal_requires_a_matched_pair():
    """The bug this guards: a bare all() over the pairs found reported reversal
    vacuously for a sweep that had only positive alphas."""
    positive_only = {0.5: {"x": 1.0}, 1.0: {"x": 2.0}}
    assert not sign_reverses(positive_only, "x")


def test_sign_reversal_detects_an_antisymmetric_response():
    sweep = {-1.0: {"x": -2.0}, 1.0: {"x": 2.0}, -0.5: {"x": -1.0}, 0.5: {"x": 1.0}}
    assert sign_reverses(sweep, "x")


def test_sign_reversal_rejects_a_one_sided_response():
    sweep = {-1.0: {"x": 1.9}, 1.0: {"x": 2.0}}
    assert not sign_reverses(sweep, "x")


def test_monotone_needs_at_least_two_strengths():
    assert not monotone({1.0: {"x": 1.0}}, "x")
    assert monotone({0.5: {"x": 1.0}, 1.0: {"x": 2.0}, 2.0: {"x": 3.0}}, "x")
    assert monotone({0.5: {"x": 3.0}, 1.0: {"x": 2.0}, 2.0: {"x": 1.0}}, "x")
    assert not monotone({0.5: {"x": 1.0}, 1.0: {"x": 3.0}, 2.0: {"x": 2.0}}, "x")


def test_linear_response_recovers_a_planted_slope():
    sweep = {a: {"x": 0.25 * a + 1.0} for a in (-2.0, -1.0, 1.0, 2.0)}
    fit = linear_response(sweep, "x")
    assert abs(fit["slope"] - 0.25) < 1e-9
    assert fit["r2"] > 0.999


def test_linear_response_is_defined_for_a_flat_sweep():
    """A constant response has zero total variance; the R^2 must not divide by it."""
    sweep = {a: {"x": 0.0} for a in (-1.0, 1.0)}
    fit = linear_response(sweep, "x")
    assert fit["slope"] == 0.0
    assert not (fit["r2"] != fit["r2"])  # not NaN


def test_rows_by_alpha_indexes_the_persisted_payload():
    payload = {"probe_direction": [{"alpha_rms_units": 2.0, "dprob_-": -0.1},
                                   {"alpha_rms_units": -2.0, "dprob_-": 0.1}]}
    indexed = rows_by_alpha(payload)
    assert sorted(indexed) == [-2.0, 2.0]
    assert indexed[2.0]["dprob_-"] == -0.1


def test_summary_counts_shaped_cells_against_the_number_of_strengths():
    sweep = {1.0: SHAPED, 2.0: SHAPED, 0.5: COMMON}
    out = summarize_direction(sweep)
    assert out["n_alpha"] == 3
    assert out["n_valence_shaped"] == 2
    assert out["n_common_mode"] == 1
    assert out["n_reallocating"] == 2
