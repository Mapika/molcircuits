"""Difference maps between 2-D densities, and the floor they have to clear.

Comparing an intervened population with a baseline by differencing two kernel
density estimates is only as trustworthy as the normalisation underneath it, and
the natural choice is the misleading one.  Dividing each estimate by its own
**peak** pins both maxima to 1, so an arm that is merely *broader* than the
reference differences to zero at the centre and to a surplus everywhere else --
a ring of density that appeared from nowhere, since no mass moved to produce it.
Mass is not conserved under that normalisation, so the map can show structure
that corresponds to no redistribution at all.  Two arms that are both slightly
broader then produce two matching rings while having nothing else in common,
which is exactly the agreement a reader would take such a figure to be ruling
out.

So every difference is offered two ways:

* **peak-normalised**, which is what a figure drawn for visual contrast shows; and
* **mass-normalised**, where each estimate integrates to 1 and mass is conserved,
  so a shift in the difference map means mass really moved.

And it is scored against a floor.  :func:`split_half_floor` differences the
baseline's *own* two halves through the identical pipeline: whatever structure
that produces comes from resampling noise alone, with no intervention at all. A
steered-against-random correlation that does not exceed the half-against-half
correlation is not evidence of anything.  :func:`radial_profile` reports the
spread and centroid directly, since those are the two quantities a donut
actually encodes.

Deliberately free of any embedding or fingerprint dependency: the KDE is passed
in as a callable, so this module is exercisable on synthetic point clouds and
the figure script keeps ownership of its own projection.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def peak_normalised_difference(grid: np.ndarray,
                               reference: np.ndarray) -> np.ndarray:
    """Difference after scaling each density to unit maximum."""
    grid = np.asarray(grid, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return grid / max(grid.max(), 1e-300) - reference / max(reference.max(), 1e-300)


def mass_normalised_difference(grid: np.ndarray,
                               reference: np.ndarray) -> np.ndarray:
    """Difference after scaling each density to unit total mass."""
    grid = np.asarray(grid, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return grid / max(grid.sum(), 1e-300) - reference / max(reference.sum(), 1e-300)


#: The two normalisations, by name, for iterating over both in a report.
NORMALISATIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "peak": peak_normalised_difference,
    "mass": mass_normalised_difference,
}


def map_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two difference maps, flattened.

    ``nan`` when either map is constant.  That happens when an arm had too few
    points for a density estimate, and a correlation of 0 would misreport it as
    measured-and-unrelated rather than not measured.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def split_half_floor(points: np.ndarray, kde: Callable[[np.ndarray], np.ndarray],
                     seed: int = 0) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Two density estimates from disjoint random halves of one population.

    The halves are a permutation rather than two independent draws because the
    baseline is a fixed finite sample; splitting it is what makes the floor a
    statement about *this* sample's resampling noise, at the same per-half count
    an intervened arm would have.
    """
    points = np.asarray(points, dtype=float)
    rng = np.random.default_rng(seed)
    order = rng.permutation(points.shape[0])
    midpoint = len(order) // 2
    return kde(points[order[:midpoint]]), kde(points[order[midpoint:]])


def radial_profile(points: np.ndarray,
                   centroid: np.ndarray | None = None) -> dict:
    """Spread and centre of a point cloud -- what a donut would encode.

    ``centroid`` defaults to the cloud's own centre, but the useful call passes
    the *baseline's* centre: a mean radius measured about each arm's own centroid
    cannot distinguish an arm that moved from one that spread, and the difference
    between those two is the whole question.
    """
    points = np.asarray(points, dtype=float)
    own = points.mean(0)
    about = own if centroid is None else np.asarray(centroid, dtype=float)
    radii = np.linalg.norm(points - about, axis=1)
    return {
        "n": int(points.shape[0]),
        "centroid": [float(v) for v in own],
        "mean_radius": float(radii.mean()),
        "median_radius": float(np.median(radii)),
        "measured_about_own_centroid": centroid is None,
    }


def normalisation_diagnostic(arms: dict[str, np.ndarray], reference: np.ndarray,
                             floor: tuple[np.ndarray | None, np.ndarray | None]
                             | None = None) -> dict:
    """Pairwise difference-map correlations under both normalisations.

    ``floor`` is the :func:`split_half_floor` pair.  When it is supplied, every
    correlation is reported next to the floor's own value, because the number a
    reader needs is not "how alike are these two arms" but "how much more alike
    than two halves of the same unintervened sample".
    """
    names = sorted(arms)
    out: dict = {"arms": names, "normalisations": {}}
    for label, fn in NORMALISATIONS.items():
        maps = {name: fn(arms[name], reference) for name in names}
        pairs = {
            f"{a}|{b}": map_correlation(maps[a], maps[b])
            for i, a in enumerate(names) for b in names[i + 1:]
        }
        entry = {"pairwise_r": pairs,
                 "max_abs": {n: float(np.abs(m).max()) for n, m in maps.items()}}
        if floor is not None and floor[0] is not None and floor[1] is not None:
            half_a, half_b = fn(floor[0], reference), fn(floor[1], reference)
            entry["floor_r"] = map_correlation(half_a, half_b)
            entry["floor_max_abs"] = float(np.abs(half_a).max())
            entry["arm_vs_floor_r"] = {
                name: map_correlation(half_a, maps[name]) for name in names}
            entry["exceeds_floor"] = {
                key: bool(np.isfinite(r) and np.isfinite(entry["floor_r"])
                          and abs(r) > abs(entry["floor_r"]))
                for key, r in pairs.items()}
        out["normalisations"][label] = entry
    return out
