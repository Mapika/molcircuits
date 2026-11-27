"""An effect counts only when every seed separates it from its matched control."""
import math

from molcircuits.analysis.generation_outcomes import wilson_interval
from molcircuits.analysis.replication import (intervals_disjoint, rate_interval,
                                          replicated_contrast,
                                          replicated_effects, seed_contrast)


def test_rate_interval_matches_the_count_based_implementation():
    """There is one Wilson implementation; the rate form must not become a second."""
    assert rate_interval(0.25, 400) == wilson_interval(100, 400)


def test_rate_interval_is_undefined_without_a_denominator():
    assert rate_interval(0.5, 0) == (None, None)
    assert rate_interval(float("nan"), 100) == (None, None)


def test_undefined_intervals_are_not_disjoint():
    """The bug this guards: treating a missing arm as separation would turn
    absent data into evidence of an effect."""
    assert not intervals_disjoint((None, None), (0.1, 0.2))
    assert not intervals_disjoint((0.1, 0.2), (None, None))


def test_disjointness_is_symmetric_and_strict():
    assert intervals_disjoint((0.0, 0.1), (0.2, 0.3))
    assert intervals_disjoint((0.2, 0.3), (0.0, 0.1))
    assert not intervals_disjoint((0.0, 0.2), (0.1, 0.3))


def test_separated_requires_the_treated_arm_to_be_the_higher_one():
    """Disjoint intervals with the control winning is a real finding, but not
    the one being counted, and it must not be scored as a hit."""
    control_wins = seed_contrast(0.10, 500, 0.60, 500)
    assert control_wins["disjoint"]
    assert not control_wins["separated"]

    treated_wins = seed_contrast(0.60, 500, 0.10, 500)
    assert treated_wins["separated"]


def test_overlapping_intervals_are_not_an_effect_however_large_the_lift():
    """Small n: a 0.2 raw lift that both intervals still cover is noise."""
    contrast = seed_contrast(0.6, 10, 0.4, 10)
    assert contrast["lift"] > 0.19
    assert not contrast["separated"]


def _cells(*separated_flags):
    return [seed_contrast(0.6 if ok else 0.42, 500, 0.1 if ok else 0.40, 500,
                          validity_treatment=0.9)
            for ok in separated_flags]


def test_one_dissenting_seed_withdraws_the_claim():
    assert replicated_contrast(_cells(True, True, True))["replicated"]
    assert not replicated_contrast(_cells(True, True, False))["replicated"]


def test_no_seeds_is_not_unanimous_agreement():
    """The bug this guards: `all([])` is True, so an empty cell reported as
    replicating across zero seeds."""
    summary = replicated_contrast([])
    assert not summary["replicated"]
    assert summary["n_seeds"] == 0
    assert math.isnan(summary["mean_lift"])


def test_validity_floor_separates_replicated_from_usable():
    """A lift bought by pushing the model off the valid-SMILES manifold is not
    a control interface, so it must not survive the usable filter."""
    strong = [seed_contrast(0.6, 500, 0.1, 500, validity_treatment=0.95)
              for _ in range(3)]
    fragile = [seed_contrast(0.7, 500, 0.1, 500, validity_treatment=0.40)
               for _ in range(3)]
    out = replicated_effects({("keeps", 1.0): strong, ("breaks", 1.0): fragile},
                             min_validity=0.80)
    assert out["n_replicated"] == 2
    assert out["n_usable"] == 1
    assert out["usable_keys"] == [("keeps", 1.0)]


def test_usable_keys_are_ranked_by_lift():
    big = [seed_contrast(0.9, 500, 0.1, 500, validity_treatment=0.9)
           for _ in range(3)]
    small = [seed_contrast(0.4, 500, 0.1, 500, validity_treatment=0.9)
             for _ in range(3)]
    out = replicated_effects({("small", 1.0): small, ("big", 1.0): big})
    assert out["usable_keys"][0] == ("big", 1.0)
