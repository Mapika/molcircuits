"""Cached prefix intervention matches full dense forward passes."""

import numpy as np
import pytest
import torch

from molcircuits.analysis.valence_intervene import run_with_null
from molcircuits.model.transformer import ModelConfig, MolecularTransformer


@pytest.mark.parametrize("layer", [0, 2])
def test_run_with_null_matches_every_dense_summary_field(layer):
    torch.manual_seed(15)
    model = MolecularTransformer(
        ModelConfig(vocab_size=21, n_layers=3, n_heads=2, d_model=24, d_ff=48, n_ctx=24)
    ).eval()
    tokens = torch.randint(21, (8, 20))
    positions = torch.tensor([(3, 12), (0, 0), (5, 3), (0, 5), (3, 4),
                              (0, 15), (1, 9), (6, 6), (7, 19)])
    w = np.random.default_rng(1).normal(size=24).astype("float32")
    w /= np.linalg.norm(w)
    args = (model, tokens, positions, layer, w, [-3, 0, 2], {"-": 2, "=": 8, "#": 13})
    dense = run_with_null(*args, device="cpu", n_null=7, seed=12, cache_prefixes=False)
    cached = run_with_null(*args, device="cpu", n_null=7, seed=12, cache_prefixes=True)
    assert cached.keys() == dense.keys()
    for key in ("note", "residual_rms", "n_null_directions", "alphas_in_rms_units"):
        assert cached[key] == dense[key]
    for family in ("probe_direction", "null"):
        assert len(cached[family]) == len(dense[family])
        for left, right in zip(cached[family], dense[family]):
            assert left.keys() == right.keys()
            for metric in left:
                assert left[metric] == pytest.approx(right[metric], rel=4e-5, abs=2e-5), metric
