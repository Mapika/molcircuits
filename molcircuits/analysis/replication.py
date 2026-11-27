"""When does a rate difference count as an effect?

A steering arm that beats its control in one seed has shown nothing: three model
initialisations produce three different dictionaries, and the largest of three
noisy contrasts is biased upward by selection alone.  The rule used throughout
is therefore deliberately strict -- a cell counts only when the two Wilson
intervals are **disjoint in every seed** and the treated arm is the higher one in
every seed.  One seed disagreeing is enough to withdraw the claim.

Two further disciplines are baked in here because leaving them to the caller is
how they get forgotten:

*Compare against the matched control, not against the unsteered baseline.*  The
interesting null is "any perturbation of this size moves the output
distribution", not "nothing was injected".  A random direction at the same
injected norm is what tests it; the unsteered rate tests something much weaker
and will happily call a direction-blind effect a success.

*Carry validity alongside the rate.*  A fragment rate bought by pushing the model
off the valid-SMILES manifold is not a control interface, and a lift quoted
without the validity it cost is not interpretable.  Every summary here reports
both, and :func:`replicated_effects` can filter on it.

The interval itself is :func:`molcircuits.analysis.generation_outcomes.wilson_interval`
-- there is exactly one Wilson implementation in this package and this module
does not add a second.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from .generation_outcomes import wilson_interval

#: Default two-sided normal quantile (95%).
Z95 = 1.96


def rate_interval(rate: float, n_total: int, z: float = Z95,
                  ) -> tuple[float | None, float | None]:
    """Wilson interval for a rate that was persisted as a float.

    The stage artifacts store ``fragment_rate`` and ``n_valid`` rather than a
    success count, so the count has to be reconstructed.  Passing ``rate *
    n_total`` back through :func:`wilson_interval` recovers the same ``p`` and
    keeps a single implementation of the interval, which matters because the
    three private copies this replaced had already drifted apart in how they
    handled an empty denominator.
    """
    if n_total <= 0 or not math.isfinite(rate):
        return None, None
    return wilson_interval(rate * n_total, n_total, z)


def intervals_disjoint(a: tuple[float | None, float | None],
                       b: tuple[float | None, float | None]) -> bool:
    """Whether two intervals fail to overlap.

    An undefined interval -- no denominator -- is not disjoint from anything.
    Treating it as separation would turn missing data into evidence.
    """
    lo_a, hi_a = a
    lo_b, hi_b = b
    if None in (lo_a, hi_a, lo_b, hi_b):
        return False
    return lo_a > hi_b or lo_b > hi_a


def seed_contrast(rate_treatment: float, n_treatment: int,
                  rate_control: float, n_control: int,
                  z: float = Z95,
                  validity_treatment: float | None = None,
                  validity_control: float | None = None) -> dict:
    """One seed's treated-against-control comparison.

    ``separated`` is the conjunction the replication rule needs: disjoint
    intervals *and* the treated arm on the higher side.  Disjointness alone
    would also fire when the control wins, which is a real finding but not the
    one being counted.
    """
    ci_t = rate_interval(rate_treatment, n_treatment, z)
    ci_c = rate_interval(rate_control, n_control, z)
    disjoint = intervals_disjoint(ci_t, ci_c)
    higher = rate_treatment > rate_control
    return {
        "rate_treatment": float(rate_treatment),
        "rate_control": float(rate_control),
        "n_treatment": int(n_treatment),
        "n_control": int(n_control),
        "ci95_treatment": list(ci_t),
        "ci95_control": list(ci_c),
        "lift": float(rate_treatment - rate_control),
        "disjoint": disjoint,
        "higher": higher,
        "separated": bool(disjoint and higher),
        "validity_treatment": validity_treatment,
        "validity_control": validity_control,
    }


def replicated_contrast(contrasts: Sequence[Mapping]) -> dict:
    """Summarise one cell across seeds.

    ``replicated`` requires separation in *every* seed present, and at least one
    seed to be present at all: an empty sequence is not unanimous agreement, and
    ``all([])`` would otherwise report it as such.
    """
    contrasts = list(contrasts)
    n_seeds = len(contrasts)
    n_separated = sum(bool(c["separated"]) for c in contrasts)
    lifts = [float(c["lift"]) for c in contrasts]
    validities = [c["validity_treatment"] for c in contrasts
                  if c.get("validity_treatment") is not None]
    return {
        "n_seeds": n_seeds,
        "n_separated": n_separated,
        "replicated": bool(n_seeds > 0 and n_separated == n_seeds),
        "mean_lift": float(sum(lifts) / n_seeds) if n_seeds else float("nan"),
        "min_lift": min(lifts) if lifts else float("nan"),
        "mean_validity": (float(sum(validities) / len(validities))
                          if validities else None),
    }


def replicated_effects(cells: Mapping[tuple, Iterable[Mapping]],
                       min_validity: float | None = None) -> dict:
    """Apply the replication rule to a whole grid of cells.

    ``cells`` maps an arbitrary hashable key -- ``(fragment, alpha)`` in the
    steering sweep -- to that cell's per-seed contrasts.  The return separates
    cells that replicate from those that additionally survive a validity floor,
    because those are two different claims and the paper makes only the second.
    """
    summaries = {key: replicated_contrast(list(rows)) for key, rows in cells.items()}
    replicated = {k: v for k, v in summaries.items() if v["replicated"]}
    if min_validity is None:
        usable = dict(replicated)
    else:
        usable = {k: v for k, v in replicated.items()
                  if v["mean_validity"] is not None
                  and v["mean_validity"] >= min_validity}
    return {
        "n_cells": len(summaries),
        "n_replicated": len(replicated),
        "n_usable": len(usable),
        "min_validity": min_validity,
        "cells": summaries,
        "replicated_keys": sorted(replicated, key=lambda k: -replicated[k]["mean_lift"]),
        "usable_keys": sorted(usable, key=lambda k: -usable[k]["mean_lift"]),
    }
