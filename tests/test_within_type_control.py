"""Within-token-identity label shuffle as a probe control."""
import numpy as np

from molcircuits.analysis.valence_probe import within_type_shuffled_labels


def test_preserves_per_type_label_multiset():
    tokens = np.array([1, 1, 1, 2, 2, 2, 3, 3])
    labels = np.array([0, 1, 2, 3, 3, 4, 1, 1])
    out = within_type_shuffled_labels(tokens, labels, seed=0)
    for t in np.unique(tokens):
        m = tokens == t
        assert sorted(out[m].tolist()) == sorted(labels[m].tolist())


def test_a_token_only_predictor_is_unaffected():
    """The per-token marginal is identical, so token-identity accuracy cannot move."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 6, size=4000)
    labels = rng.integers(0, 5, size=4000)
    out = within_type_shuffled_labels(tokens, labels, seed=1)
    for t in np.unique(tokens):
        before = np.bincount(labels[tokens == t], minlength=5)
        after = np.bincount(out[tokens == t], minlength=5)
        assert np.array_equal(before, after)


def test_actually_reassigns_when_a_type_has_mixed_labels():
    tokens = np.zeros(200, dtype=int)
    labels = np.arange(200) % 5
    out = within_type_shuffled_labels(tokens, labels, seed=2)
    assert not np.array_equal(out, labels)


def test_constant_label_type_is_left_alone():
    tokens = np.array([7, 7, 7, 7])
    labels = np.array([2, 2, 2, 2])
    assert np.array_equal(within_type_shuffled_labels(tokens, labels, 0), labels)


def test_shape_validation():
    try:
        within_type_shuffled_labels(np.zeros(4), np.zeros(5))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched shapes")


def test_block_confinement_preserves_the_invariant_per_split():
    """The point of the control: a token-only predictor must be unaffected on
    the population the accuracy is measured on, not merely in aggregate."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 5, size=2000)
    labels = rng.integers(0, 4, size=2000)
    blocks = (rng.random(2000) < 0.3).astype(int)          # a 70/30 split
    out = within_type_shuffled_labels(tokens, labels, seed=1, blocks=blocks)
    for t in np.unique(tokens):
        for b in np.unique(blocks):
            m = (tokens == t) & (blocks == b)
            assert np.array_equal(np.bincount(labels[m], minlength=4),
                                  np.bincount(out[m], minlength=4))


def test_unblocked_shuffle_breaks_the_per_split_invariant():
    """Guards the bug this parameter exists to fix: without blocks, labels cross
    the split boundary and the per-token distribution moves on each side."""
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 5, size=2000)
    labels = rng.integers(0, 4, size=2000)
    blocks = (rng.random(2000) < 0.3).astype(int)
    out = within_type_shuffled_labels(tokens, labels, seed=1)   # no blocks
    moved = sum(
        not np.array_equal(
            np.bincount(labels[(tokens == t) & (blocks == b)], minlength=4),
            np.bincount(out[(tokens == t) & (blocks == b)], minlength=4))
        for t in np.unique(tokens) for b in np.unique(blocks))
    assert moved > 0
