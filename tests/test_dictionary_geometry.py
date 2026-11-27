"""A dictionary can span a direction perfectly and still have no feature near it."""
import numpy as np
import torch

from molcircuits.analysis.dictionary_geometry import (alignment_versus_detection,
                                                  decoder_directions,
                                                  dictionary_span_recovery,
                                                  direction_alignment,
                                                  fragment_geometry)


class _SAE:
    """Minimal stand-in: the geometry only ever reads the decoder matrix."""

    def __init__(self, w_dec):
        self.W_dec = torch.as_tensor(w_dec, dtype=torch.float32)


def _overcomplete(d_in=16, n_features=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return _SAE(torch.randn(n_features, d_in, generator=g))


def test_decoder_rows_are_normalised():
    norms = decoder_directions(_overcomplete()).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_a_planted_direction_is_found_as_its_own_best_feature():
    sae = _overcomplete()
    target = 7
    direction = sae.W_dec[target].clone()
    out = direction_alignment(sae, direction, feature=target)
    assert out["best_cosine_feature"] == target
    assert out["cosine_detector"] > 0.999


def test_alignment_uses_absolute_cosine():
    """Which side of the fragment contrast was subtracted from which is a
    convention, so an anti-parallel column is just as aligned."""
    sae = _overcomplete()
    flipped = -sae.W_dec[3].clone()
    assert direction_alignment(sae, flipped, feature=3)["cosine_detector"] > 0.999


def test_random_baseline_is_the_chance_cosine():
    """Without it, 0.05 in 512 dimensions reads as 'small but present' when it
    is indistinguishable from chance."""
    out = direction_alignment(_overcomplete(d_in=16), torch.randn(16))
    assert abs(out["random_baseline"] - 1 / np.sqrt(16)) < 1e-9


def test_an_overcomplete_dictionary_spans_every_direction():
    """~1.0 by construction, which is why the module documents it as the
    expected value rather than a finding."""
    sae = _overcomplete(d_in=16, n_features=64)
    out = dictionary_span_recovery(sae, torch.randn(16))
    assert out["recovered_norm"] > 0.99
    assert out["spans_direction"]


def test_a_rank_deficient_dictionary_cannot_express_an_orthogonal_direction():
    """The contrast that makes the span number worth reporting at all."""
    w = torch.zeros(8, 4)
    w[:, 0] = torch.linspace(1, 2, 8)      # every column inside span{e0, e1}
    w[:, 1] = torch.linspace(-1, 1, 8)
    sae = _SAE(w)
    outside = torch.tensor([0.0, 0.0, 1.0, 0.0])
    out = dictionary_span_recovery(sae, outside)
    assert out["recovered_norm"] < 1e-5
    assert not out["spans_direction"]

    inside = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert dictionary_span_recovery(sae, inside)["recovered_norm"] > 0.99


def test_the_headline_case_high_span_recovery_with_a_distant_detector():
    """The finding itself: the dictionary represents the steering direction
    perfectly as a combination of many features, while the fragment's own
    detector is no closer to it than chance."""
    sae = _overcomplete(d_in=32, n_features=256, seed=3)
    g = torch.Generator().manual_seed(99)
    direction = torch.randn(32, generator=g)
    out = fragment_geometry(sae, direction, feature=0, auroc=0.93)
    assert out["recovered_norm"] > 0.99
    assert out["cosine_detector"] < 10 * out["random_baseline"]
    assert out["detector_auroc"] == 0.93


def test_alignment_versus_detection_refuses_to_report_on_too_few_points():
    """Fewer than four points cannot distinguish any monotone relation from
    chance; returning a statistic would invite quoting one."""
    rows = [{"detector_auroc": 0.9, "cosine_detector": 0.1}] * 3
    assert alignment_versus_detection(rows)["spearman"] is None


def test_alignment_versus_detection_recovers_a_planted_monotone_relation():
    rows = [{"detector_auroc": a, "cosine_detector": a / 2}
            for a in (0.6, 0.7, 0.8, 0.9, 0.95)]
    out = alignment_versus_detection(rows)
    assert out["n"] == 5
    assert out["spearman"] > 0.99


def test_non_finite_rows_are_dropped_not_propagated():
    rows = [{"detector_auroc": a, "cosine_detector": a / 2}
            for a in (0.6, 0.7, 0.8, 0.9, 0.95)]
    rows.append({"detector_auroc": float("nan"), "cosine_detector": 0.5})
    rows.append({"detector_auroc": 0.5, "cosine_detector": None})
    assert alignment_versus_detection(rows)["n"] == 5
