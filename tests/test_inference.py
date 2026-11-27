"""The four inference levels must disagree when the evidence is nested."""
import numpy as np

from molcircuits.analysis.inference import (common_fragments, family_cluster,
                                        ladder, per_fragment,
                                        per_fragment_differences, per_seed,
                                        pooled)

#: Five arenes and one alkyl. The arenes are one family, so the apparent
#: n of six targets is really two independent questions.
ARENES = ["benzene", "pyridine", "pyrimidine", "phenol", "aniline"]


def _nested(effect: float = 0.05, seeds=(0, 1, 2)):
    """Effect present only in the arene family, which is a single cluster."""
    rng = np.random.default_rng(0)
    values = {}
    for s in seeds:
        row = {}
        for f in ARENES:
            base = 0.5 + rng.normal(0, 0.001)
            row[f] = {"sae": base + effect, "dense": base}
        base = 0.5 + rng.normal(0, 0.001)
        row["tert_butyl"] = {"sae": base, "dense": base + effect}
        values[s] = row
    return values


def test_common_fragments_is_the_intersection_not_the_union():
    """A target measured in one seed only would otherwise average over a
    different number of seeds than its neighbours."""
    values = {0: {"a": {}, "b": {}}, 1: {"a": {}}}
    assert common_fragments(values) == ["a"]


def test_common_fragments_of_nothing_is_empty():
    assert common_fragments({}) == []


def test_pooled_counts_every_fragment_seed_pair():
    values = _nested()
    out = pooled(values, "sae", "dense")
    assert out["n"] == 6 * 3
    assert out["mean_diff"] > 0


def test_per_fragment_collapses_the_seed_axis():
    values = _nested()
    out = per_fragment(values, "sae", "dense")
    assert out["n"] == 6


def test_per_seed_flags_a_direction_that_flips_between_initialisations():
    """A result whose sign depends on the seed is not a result, however small
    any individual p-value is."""
    values = _nested()
    values[2] = {f: {"sae": v["dense"], "dense": v["sae"]}
                 for f, v in values[2].items()}
    assert not per_seed(values, "sae", "dense")["consistent_direction"]
    assert per_seed(_nested(), "sae", "dense")["consistent_direction"]


def test_identical_arms_do_not_raise():
    """scipy raises when every pair is identical; that is a real outcome for an
    arm compared against itself and must be reported, not crash the sweep."""
    values = _nested()
    out = pooled(values, "sae", "sae")
    assert out["pvalue"] is None
    assert out["mean_diff"] == 0.0


def test_per_fragment_differences_average_over_seeds():
    values = _nested(effect=0.05)
    diffs = per_fragment_differences(values, "sae", "dense")
    assert set(diffs) == set(ARENES) | {"tert_butyl"}
    assert abs(diffs["benzene"] - 0.05) < 1e-9
    assert abs(diffs["tert_butyl"] + 0.05) < 1e-9


def test_cluster_bootstrap_sees_fewer_independent_targets_than_fragments():
    """Five arenes are one family. The bootstrap must resample two clusters,
    not six, which is the whole point of the level."""
    out = family_cluster(_nested(), "sae", "dense", n_bootstrap=2000, seed=0)
    assert out["n_fragments"] == 6
    assert out["n_families"] == 2


def test_nesting_widens_the_interval_enough_to_cover_zero():
    """The pooled test is confident and the cluster bootstrap is not, because
    five of the six rows are the same chemical question asked five times."""
    values = _nested()
    assert pooled(values, "sae", "dense")["pvalue"] < 0.05
    assert not family_cluster(values, "sae", "dense",
                              n_bootstrap=4000, seed=0)["excludes_zero"]


def test_ladder_does_not_claim_agreement_when_the_bootstrap_covers_zero():
    out = ladder(_nested(), "sae", "dense", n_bootstrap=2000, seed=0)
    assert not out["agrees_across_levels"]
    assert set(out) >= {"pooled", "per_seed", "per_fragment", "family_cluster"}


def test_ladder_agrees_when_the_effect_is_spread_across_families():
    """Same effect size, spread over independent families instead of nested
    inside one, now survives every level."""
    rng = np.random.default_rng(1)
    fragments = ["benzene", "piperidine", "furan", "amide", "ether",
                 "sulfonamide", "trifluoromethyl", "tert_butyl"]
    values = {}
    for s in (0, 1, 2):
        values[s] = {}
        for f in fragments:
            base = 0.5 + rng.normal(0, 0.001)
            values[s][f] = {"sae": base + 0.05, "dense": base}
    out = ladder(values, "sae", "dense", n_bootstrap=4000, seed=0)
    assert out["family_cluster"]["n_families"] == 8
    assert out["family_cluster"]["excludes_zero"]
    assert out["agrees_across_levels"]
