from __future__ import annotations

import pytest

from molcircuits.analysis.generation_outcomes import (fragment_outcomes,
                                                  normalized_outcome,
                                                  wilson_interval)


def test_invalid_generations_count_as_joint_failures():
    out = fragment_outcomes(["c1ccncc1", "not-smiles", "CCO", ""], "n")
    assert out["n_attempted"] == 4
    assert out["n_valid"] == 2
    assert out["n_fragment_hits"] == 1
    assert out["fragment_rate_given_valid"] == 0.5
    assert out["fragment_success_rate"] == 0.25
    assert out["validity"] == 0.5


def test_legacy_conditional_row_upgrades_without_changing_counts():
    old = {"fragment_rate": 295 / 871, "n_valid": 871, "validity": 871 / 2000}
    out = normalized_outcome(old, n_attempted=2000)
    assert out["n_fragment_hits"] == 295
    assert out["fragment_rate_given_valid"] == pytest.approx(295 / 871)
    assert out["fragment_success_rate"] == pytest.approx(295 / 2000)


def test_zero_valid_outcome_uses_json_null_for_undefined_conditionals():
    out = normalized_outcome(
        {"fragment_rate": float("nan"), "n_valid": 0}, n_attempted=400)
    assert out["n_fragment_hits"] == 0
    assert out["fragment_rate_given_valid"] is None
    assert out["fragment_rate_given_valid_ci95"] == [None, None]
    assert out["fragment_success_rate"] == 0.0


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -0.1, 1.1])
def test_legacy_nonfinite_or_invalid_conditional_is_rejected_when_valid(value):
    with pytest.raises(ValueError, match="finite legacy conditional rate"):
        normalized_outcome(
            {"fragment_rate": value, "n_valid": 4}, n_attempted=10)


def test_normalizer_rejects_impossible_counts():
    with pytest.raises(ValueError, match="inconsistent outcome counts"):
        normalized_outcome({"n_fragment_hits": 6, "n_valid": 5}, n_attempted=10)


def test_wilson_interval_stays_in_unit_interval():
    for hits in (0, 1, 9, 10):
        lo, hi = wilson_interval(hits, 10)
        assert 0.0 <= lo <= hi <= 1.0


def test_empty_wilson_interval_is_explicitly_undefined():
    assert wilson_interval(0, 0) == (None, None)
