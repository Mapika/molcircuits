"""Peak normalisation manufactures donuts; the split-half floor is what catches it."""
import numpy as np

from molcircuits.analysis.density import (map_correlation,
                                      mass_normalised_difference,
                                      normalisation_diagnostic,
                                      peak_normalised_difference,
                                      radial_profile, split_half_floor)


def _gaussian_grid(sigma, size=41):
    """A radially symmetric density on a fixed grid, centred, unit mass."""
    axis = np.linspace(-3, 3, size)
    x, y = np.meshgrid(axis, axis)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def test_peak_normalisation_invents_density_that_no_mass_movement_produced():
    """The failure this module exists to expose. Both peaks are pinned to 1, so
    an arm that is only WIDER differences to zero at the centre and to a surplus
    everywhere else -- a ring with no matching deficit anywhere, which no
    redistribution of mass could have produced."""
    reference = _gaussian_grid(0.6)
    broader = _gaussian_grid(0.9)
    diff = peak_normalised_difference(broader, reference)

    centre = diff[diff.shape[0] // 2, diff.shape[1] // 2]
    assert abs(centre) < 1e-9              # pinned together by construction
    assert diff.min() > -1e-9              # nowhere is there a deficit
    assert diff.sum() > 0.1                # yet the map is full of surplus


def test_mass_normalisation_conserves_total_mass():
    """Each estimate integrates to 1, so the difference must sum to ~0 --
    a surplus somewhere has to be a deficit somewhere else."""
    reference = _gaussian_grid(0.6)
    broader = _gaussian_grid(0.9)
    diff = mass_normalised_difference(broader, reference)
    assert abs(diff.sum()) < 1e-9

    # And on the same pair it reports the real shape: mass left the centre.
    centre = diff[diff.shape[0] // 2, diff.shape[1] // 2]
    rim = diff[diff.shape[0] // 2, 4]
    assert centre < 0 < rim


def test_two_merely_broader_arms_look_alike_under_peak_normalisation():
    """Two arms with nothing in common except being broader correlate strongly
    once peak-normalised, which is exactly the false agreement a reader would
    take such a figure to be ruling out."""
    reference = _gaussian_grid(0.6)
    arms = {"a": _gaussian_grid(0.85), "b": _gaussian_grid(0.95)}
    out = normalisation_diagnostic(arms, reference)
    assert out["normalisations"]["peak"]["pairwise_r"]["a|b"] > 0.9


def test_map_correlation_is_undefined_for_a_constant_map():
    """A flat map means an arm had too little data; reporting r=0 would
    misread 'not measured' as 'measured and unrelated'."""
    flat = np.zeros((8, 8))
    assert np.isnan(map_correlation(flat, np.random.default_rng(0).normal(size=(8, 8))))


def test_split_half_partitions_without_overlap():
    points = np.arange(40, dtype=float).reshape(20, 2)
    seen = []
    split_half_floor(points, lambda p: seen.append(p) or np.zeros((3, 3)), seed=0)
    first, second = seen
    assert first.shape[0] + second.shape[0] == 20
    combined = {tuple(r) for r in first} | {tuple(r) for r in second}
    assert len(combined) == 20


def test_the_floor_is_reported_next_to_the_arm_correlation():
    """The number that matters is not how alike two arms are, but how much more
    alike than two halves of the same unintervened sample."""
    reference = _gaussian_grid(0.6)
    arms = {"steered": _gaussian_grid(0.85), "random": _gaussian_grid(0.9)}
    floor = (_gaussian_grid(0.61), _gaussian_grid(0.59))
    out = normalisation_diagnostic(arms, reference, floor=floor)
    peak = out["normalisations"]["peak"]
    assert "floor_r" in peak
    assert "exceeds_floor" in peak
    assert set(peak["arm_vs_floor_r"]) == {"steered", "random"}


def test_radial_profile_about_the_baseline_centre_separates_moving_from_spreading():
    """Measured about each arm's own centroid, a cloud that MOVED and one that
    SPREAD look identical -- which is the distinction being tested."""
    rng = np.random.default_rng(0)
    baseline = rng.normal(0, 1, size=(500, 2))
    centre = baseline.mean(0)
    moved = rng.normal(0, 1, size=(500, 2)) + np.array([4.0, 0.0])
    spread = rng.normal(0, 3, size=(500, 2))

    own = radial_profile(moved)
    assert own["measured_about_own_centroid"]
    assert own["mean_radius"] < 2.0            # about itself, it looks tight

    about_baseline = radial_profile(moved, centroid=centre)
    assert about_baseline["mean_radius"] > 3.5  # about the baseline, it moved
    assert radial_profile(spread, centroid=centre)["mean_radius"] > 3.0
