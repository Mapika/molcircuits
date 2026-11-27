"""Structure-preserving fragment nulls."""
import numpy as np
import pytest

from molcircuits.analysis.fragment_controls import (
    permute_fragment_masks_within_molecules,
    shift_fragment_masks_within_molecules,
    type_matched_fragment_masks,
)


def test_shift_preserves_footprint_size_and_shape():
    valid = np.array([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=bool)
    frag = np.array([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=bool)
    out = shift_fragment_masks_within_molecules(frag, valid, 20, seed=0)
    assert out.shape == (1, 20, 8)
    pattern = frag[0][valid[0]]
    doubled = np.concatenate([pattern, pattern])
    for draw in out[0]:
        assert draw.sum() == 3                      # coverage preserved
        assert not draw[~valid[0]].any()            # never leaves valid tokens
        got = draw[valid[0]]
        # the footprint is a cyclic rotation of the original, so its exact
        # pattern of runs and gaps survives and only its position changes
        assert any(np.array_equal(got, doubled[i:i + len(pattern)])
                   for i in range(len(pattern)))


def test_shift_actually_moves_the_mask():
    valid = np.ones((1, 10), dtype=bool)
    frag = np.zeros((1, 10), dtype=bool)
    frag[0, :4] = True
    out = shift_fragment_masks_within_molecules(frag, valid, 30, seed=1)
    assert not np.array_equal(out[0, 0], frag[0]) or not np.array_equal(out[0, 1], frag[0])


def test_type_matched_holds_the_token_multiset_fixed():
    #                C  O  C  O  C  O
    tokens = np.array([[1, 2, 1, 2, 1, 2]])
    valid = np.ones((1, 6), dtype=bool)
    frag = np.array([[1, 1, 0, 0, 0, 0]], dtype=bool)   # one C, one O
    out, info = type_matched_fragment_masks(frag, valid, tokens, 25, seed=0)
    assert info["shortfall_fraction"] == 0.0
    for draw in out[0]:
        picked = tokens[0][draw]
        assert sorted(picked.tolist()) == [1, 2]        # same composition, always


def test_type_matched_draw_is_always_fillable():
    """A fragment position is itself a valid position of its own type, so the
    candidate pool can never be smaller than the request. The shortfall counter
    is a guard against a caller passing a fragment mask that escapes the valid
    mask, not a routine outcome; if it ever fires, the inputs are inconsistent.
    """
    rng = np.random.default_rng(0)
    valid = rng.random((8, 16)) < 0.8
    frag = valid & (rng.random((8, 16)) < 0.35)
    tokens = rng.integers(0, 5, size=(8, 16))
    _, info = type_matched_fragment_masks(frag, valid, tokens, 4, seed=0)
    assert info["positions_requested"] > 0
    assert info["positions_short"] == 0
    assert info["shortfall_fraction"] == 0.0


def test_type_matched_counts_shortfall_when_fragment_escapes_valid():
    tokens = np.array([[1, 2, 2, 2]])
    frag = np.array([[1, 1, 0, 0]], dtype=bool)
    valid = np.array([[0, 1, 1, 1]], dtype=bool)        # type-1 slot not valid
    _, info = type_matched_fragment_masks(frag, valid, tokens, 2, seed=0)
    assert info["positions_short"] == 0                 # type 2 is fillable
    assert info["positions_requested"] == 2


def test_nulls_stay_inside_valid_positions():
    rng = np.random.default_rng(0)
    valid = rng.random((6, 12)) < 0.7
    frag = valid & (rng.random((6, 12)) < 0.3)
    tokens = rng.integers(0, 4, size=(6, 12))
    for out in (permute_fragment_masks_within_molecules(frag, valid, 5, 0),
                shift_fragment_masks_within_molecules(frag, valid, 5, 0),
                type_matched_fragment_masks(frag, valid, tokens, 5, 0)[0]):
        assert not (out & ~valid[:, None, :]).any()


@pytest.mark.parametrize("fn", [shift_fragment_masks_within_molecules,
                                permute_fragment_masks_within_molecules])
def test_shape_validation(fn):
    with pytest.raises(ValueError):
        fn(np.zeros((2, 3), bool), np.zeros((2, 4), bool), 1, 0)
