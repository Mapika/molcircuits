"""How much of a fragment-level result is pseudoreplication?

Pooling every fragment-seed pair into one test treats three model
initialisations and a set of overlapping SMARTS targets as that many independent
observations.  They are not.  Each fragment appears once per seed, so the seed
axis is repeated measurement rather than replication; and the library nests
chemically -- benzene, pyridine and pyrimidine share a carbon skeleton,
piperidine/piperazine/morpholine differ by one heteroatom -- so neighbouring
targets are not asking independent questions either.

This module runs the same comparison at four levels of that nesting, from most
to least generous:

1. **pooled** over fragment-seed pairs -- the largest n, and the one that
   double-counts on both axes;
2. **per seed** -- three separate tests, which removes the seed
   pseudoreplication and shows whether the direction is stable across
   initialisations;
3. **per fragment**, averaged over seeds -- one row per target;
4. **fragment-family cluster bootstrap** -- resamples *families*, so pyridine
   and pyrimidine cannot both count as independent evidence.

Only the fourth is honest about how many independent targets the library really
contains, and it is the one to quote when the others disagree.  Reporting the
ladder rather than a single level is deliberate: a result that survives all four
and a result that survives only the first are different findings, and collapsing
them to one p-value hides which was obtained.

Level 4 delegates to :func:`molcircuits.analysis.fragment_controls.fragment_family_cluster_bootstrap`
and the family assignments live in :mod:`molcircuits.analysis.fragment_families`, so
the nesting used here cannot drift from the one stages 15 and 17 resample at.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats

from .fragment_controls import fragment_family_cluster_bootstrap
from .fragment_families import family_map


def common_fragments(values: Mapping[int, Mapping[str, Mapping]]) -> list[str]:
    """Fragments measured in *every* seed.

    The intersection, not the union: a target that ran in one seed and failed in
    another would otherwise enter the per-fragment and bootstrap levels with its
    mean taken over a different number of seeds than its neighbours, which is
    precisely the imbalance those levels exist to remove.
    """
    if not values:
        return []
    sets = [set(per_seed) for per_seed in values.values()]
    return sorted(set.intersection(*sets))


def _paired(values, seeds, fragments, arm_a, arm_b):
    a = [float(values[s][f][arm_a]) for s in seeds for f in fragments]
    b = [float(values[s][f][arm_b]) for s in seeds for f in fragments]
    return np.asarray(a, dtype=float), np.asarray(b, dtype=float)


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired Wilcoxon with the degenerate cases named rather than raised.

    ``scipy`` raises when every pair is identical.  That is a real outcome for a
    comparison of an arm against itself, and a summary that reports it as
    ``p=None`` is more useful than a traceback in the middle of a sweep.
    """
    diff = a - b
    if len(diff) == 0:
        return {"n": 0, "mean_diff": float("nan"), "n_wins": 0, "pvalue": None}
    out = {
        "n": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "n_wins": int((diff > 0).sum()),
    }
    if np.allclose(diff, 0.0):
        out["pvalue"] = None
        return out
    out["pvalue"] = float(stats.wilcoxon(a, b).pvalue)
    return out


def pooled(values, arm_a: str, arm_b: str) -> dict:
    """Level 1: one test over every fragment-seed pair."""
    seeds = sorted(values)
    fragments = common_fragments(values)
    a, b = _paired(values, seeds, fragments, arm_a, arm_b)
    return {"level": "pooled over fragment-seed pairs",
            "n_seeds": len(seeds), "n_fragments": len(fragments),
            **_wilcoxon(a, b)}


def per_seed(values, arm_a: str, arm_b: str) -> dict:
    """Level 2: one test per seed, so no seed contributes twice."""
    fragments = common_fragments(values)
    rows = {}
    for s in sorted(values):
        a, b = _paired(values, [s], fragments, arm_a, arm_b)
        rows[s] = _wilcoxon(a, b)
    directions = {row["mean_diff"] > 0 for row in rows.values()}
    return {"level": "per seed",
            "n_fragments": len(fragments),
            "seeds": rows,
            # A direction that flips between initialisations is not a result,
            # however small any individual p-value is.
            "consistent_direction": len(directions) == 1}


def per_fragment(values, arm_a: str, arm_b: str) -> dict:
    """Level 3: average over seeds first, so each target contributes one row."""
    seeds = sorted(values)
    fragments = common_fragments(values)
    a = np.asarray([np.mean([float(values[s][f][arm_a]) for s in seeds])
                    for f in fragments], dtype=float)
    b = np.asarray([np.mean([float(values[s][f][arm_b]) for s in seeds])
                    for f in fragments], dtype=float)
    return {"level": "per fragment, averaged over seeds",
            "n_fragments": len(fragments), **_wilcoxon(a, b)}


def per_fragment_differences(values, arm_a: str, arm_b: str) -> dict[str, float]:
    """Seed-averaged ``arm_a - arm_b`` for every commonly measured fragment."""
    seeds = sorted(values)
    return {
        f: float(np.mean([float(values[s][f][arm_a]) - float(values[s][f][arm_b])
                          for s in seeds]))
        for f in common_fragments(values)
    }


def family_cluster(values, arm_a: str, arm_b: str,
                   n_bootstrap: int = 10_000, seed: int = 0) -> dict:
    """Level 4: resample families, not fragments."""
    differences = per_fragment_differences(values, arm_a, arm_b)
    out = dict(fragment_family_cluster_bootstrap(differences, n_bootstrap, seed))
    out["level"] = "fragment-family cluster bootstrap"
    out["mean_diff"] = (float(np.mean(list(differences.values())))
                        if differences else float("nan"))
    low, high = out["ci95"]
    out["excludes_zero"] = bool(np.isfinite(low) and np.isfinite(high)
                                and (low > 0 or high < 0))
    return out


def ladder(values, arm_a: str, arm_b: str,
           n_bootstrap: int = 10_000, seed: int = 0) -> dict:
    """All four levels of the same comparison.

    ``agrees_across_levels`` is the summary worth reading first: it is true only
    when every level puts the effect in the same direction *and* the cluster
    bootstrap excludes zero, which is the conjunction that survives the nesting.
    """
    out = {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "pooled": pooled(values, arm_a, arm_b),
        "per_seed": per_seed(values, arm_a, arm_b),
        "per_fragment": per_fragment(values, arm_a, arm_b),
        "family_cluster": family_cluster(values, arm_a, arm_b, n_bootstrap, seed),
        "families": sorted(set(family_map(common_fragments(values)).values())),
    }
    signs = {out["pooled"]["mean_diff"] > 0,
             out["per_fragment"]["mean_diff"] > 0,
             out["family_cluster"]["mean_diff"] > 0}
    out["agrees_across_levels"] = bool(
        len(signs) == 1
        and out["per_seed"]["consistent_direction"]
        and out["family_cluster"]["excludes_zero"])
    return out
