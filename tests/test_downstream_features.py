"""Focused regressions for downstream row identity and representation fairness."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from molcircuits.chem.tokenizer import SmilesTokenizer
from molcircuits.downstream.features import (embed, load_stack,
                                         match_split_occurrences)
from molcircuits.downstream.moleculeace import Endpoint
from molcircuits.model.transformer import ModelConfig, MolecularTransformer


def _tiny_checkpoint(tmp_path: Path, *, embedded_config: bool):
    tok = SmilesTokenizer.from_corpus(["CC"])
    tok_path = tmp_path / "explicit_tokenizer.json"
    tok.save(tok_path)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=8, n_layers=1,
                      n_heads=2, d_ff=16, n_ctx=8)
    model = MolecularTransformer(cfg)
    payload = {"model": model.state_dict()}
    if embedded_config:
        payload["model_cfg"] = vars(cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save(payload, ckpt)
    return ckpt, tok_path, cfg


def test_load_stack_prefers_checkpoint_geometry_and_records_sources(tmp_path):
    ckpt, tok_path, cfg = _tiny_checkpoint(tmp_path, embedded_config=True)
    # A mutable sibling file must not silently replace embedded checkpoint
    # geometry. Its width would make the state dict fail if it were used.
    ModelConfig(vocab_size=cfg.vocab_size, d_model=16, n_layers=1,
                n_heads=2, d_ff=32, n_ctx=8).save(
                    tmp_path / "model_config.json")

    model, tok, sae, loaded, metadata = load_stack(
        str(ckpt), device="cpu", tokenizer_path=str(tok_path),
        return_metadata=True)

    assert model.cfg == cfg
    assert loaded == cfg
    assert tok.vocab == SmilesTokenizer.load(tok_path).vocab
    assert sae is None
    assert metadata["model_config_source"] == \
        f"{os.path.realpath(ckpt)}::model_cfg"
    assert metadata["tokenizer_source"] == os.path.realpath(tok_path)


def test_load_stack_legacy_checkpoint_uses_sibling_config(tmp_path):
    ckpt, tok_path, cfg = _tiny_checkpoint(tmp_path, embedded_config=False)
    config_path = tmp_path / "model_config.json"
    cfg.save(config_path)

    _, _, _, loaded, metadata = load_stack(
        str(ckpt), device="cpu", tokenizer_path=str(tok_path),
        return_metadata=True)

    assert loaded == cfg
    assert metadata["model_config_source"] == os.path.realpath(config_path)


def test_load_stack_has_no_machine_specific_tokenizer_fallback(tmp_path,
                                                                monkeypatch):
    ckpt, _, _ = _tiny_checkpoint(tmp_path, embedded_config=True)
    monkeypatch.delenv("MOLCIRCUITS_TOKENIZER", raising=False)

    with pytest.raises(FileNotFoundError, match="no tokenizer was configured"):
        load_stack(str(ckpt), device="cpu")


def test_embed_reports_and_drops_overlength_rows():
    torch.manual_seed(0)
    tok = SmilesTokenizer.from_corpus(["C"])
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=8, n_layers=1,
                      n_heads=2, d_ff=16, n_ctx=8)
    model = MolecularTransformer(cfg).eval()

    dense, codes, info = embed(
        ["C", "CCCC", "C?"], model, tok, layer=0, n_ctx=4,
        device="cpu", batch=3, return_metadata=True)

    assert codes is None
    assert info["dropped_indices"] == [1, 2]
    assert info["overlength_indices"] == [1]
    assert info["tokenizer_failed_indices"] == [2]
    assert info["drop_policy"] == "drop_from_all_representations"
    assert np.any(dense[0])
    np.testing.assert_array_equal(dense[1:], 0.0)


def test_moleculeace_applies_overlength_mask_to_ecfp(monkeypatch):
    path = Path(__file__).parents[1] / "experiments" / "_downstream" / "moleculeace.py"
    spec = importlib.util.spec_from_file_location("moleculeace_script_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    drop = {
        "drop_policy": "drop_from_all_representations",
        "n_input": 3,
        "n_ctx": 4,
        "n_dropped": 1,
        "dropped_indices": [1],
        "n_tokenizer_failed": 0,
        "tokenizer_failed_indices": [],
        "n_overlength": 1,
        "overlength_indices": [1],
    }

    def fake_embed(smiles, *args, pooling, **kwargs):
        base = np.arange(len(smiles) * 2, dtype=np.float32).reshape(len(smiles), 2)
        assert pooling == "both"
        return {"max":base, "mean":base}, {"max":base+20, "mean":base+20}, dict(drop)

    def fake_ecfp(smiles):
        return np.arange(len(smiles) * 3, dtype=np.float32).reshape(len(smiles), 3)

    monkeypatch.setattr(module, "embed", fake_embed)
    monkeypatch.setattr(module, "ecfp", fake_ecfp)
    endpoint = Endpoint("toy", ["C", "CCCC", "N"], np.arange(3.0),
                        np.array([True, True, False]),
                        np.array([False, False, True]))

    features, kept, info = module.featurize(
        endpoint, object(), object(), object(), layer=0, batch=2,
        device="cpu", n_ctx=4)

    assert kept.smiles == ["C", "N"]
    assert info["overlength_indices"] == [1]
    assert all(values.shape[0] == 2 for values in features.values())
    np.testing.assert_array_equal(features["ecfp"], fake_ecfp(endpoint.smiles)[[0, 2]])


def test_both_poolings_reuse_one_forward_and_keep_single_pool_results():
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    tok = SmilesTokenizer.from_corpus(["C", "CO", "CCO"])
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=8, n_layers=1,
                      n_heads=2, d_ff=16, n_ctx=8)
    model = MolecularTransformer(cfg).eval()
    sae = SparseAutoencoder(SAEConfig(d_in=8, expansion=2, layer=0, k=2)).eval()
    args = (["C", "CO", "C?", "CCO", "CCCCCCC"], model, tok, sae)
    separate = {p: embed(*args, layer=0, pooling=p, n_ctx=8, device="cpu", batch=2)
                for p in ("max", "mean")}
    calls = []
    hook = model.register_forward_hook(lambda *unused: calls.append(1))
    dense, sparse, dropped = embed(*args, layer=0, pooling="both", n_ctx=8, device="cpu", batch=2)
    hook.remove()
    assert len(calls) == 2
    for how, (expected_dense, expected_sparse, expected_drops) in separate.items():
        assert dropped == expected_drops == [2, 4]
        np.testing.assert_array_equal(dense[how], expected_dense)
        np.testing.assert_array_equal(sparse[how], expected_sparse)


def test_ace_cache_reuse_requires_the_same_matrix_labels_split_seed_and_head(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from molcircuits.downstream import ace_cache

    calls = []
    def fit(X, ep, seed, *args, **kwargs):
        calls.append(seed)
        return X[ep.test, 0].copy(), 3
    monkeypatch.setattr(ace_cache.moleculeace, "fit_predict", fit)
    cache = ace_cache.MoleculeACECache(tmp_path, "cpu")
    X = np.arange(20, dtype="float32").reshape(10, 2)
    ep = Endpoint("assay", ["C"]*10, X[:, 0].copy(), np.arange(10) < 7, np.arange(10) < 5)
    def run(*, seed=1, early_stop=2):
        return cache.fit(X, ep, seed, 1, .2, early_stop, "cpu")
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda i: run(), (0, 1)))
    assert sorted(r[2] for r in responses) == [False, True] and len(calls) == 1
    np.testing.assert_array_equal(responses[0][0], responses[1][0])
    assert run(seed=2)[2] is False
    assert run(early_stop=3)[2] is False
    ep.y[0] += 1
    assert run()[2] is False
    ep.train[0] = False
    assert run()[2] is False
    X[0, 0] += 1
    assert run()[2] is False
    assert run()[2] is True


def test_split_duplicates_are_consumed_once_not_expanded_quadratically():
    # Duplicate SMILES include both identical and conflicting labels. Drug IDs
    # make the exact source occurrence observable, as in PyTDC's tables.
    full = [
        ("id0", "A", 0),
        ("id1", "A", 1),
        ("id2", "A", 0),
        ("id3", "B", 1),
    ]
    parts = {
        "train": [full[0], full[1]],
        "valid": [full[2]],
        "test": [full[3]],
    }

    selected = match_split_occurrences(full, parts, keep=[0, 1, 2, 3])
    all_rows = np.concatenate(list(selected.values()))

    np.testing.assert_array_equal(selected["train"], [0, 1])
    np.testing.assert_array_equal(selected["valid"], [2])
    np.testing.assert_array_equal(selected["test"], [3])
    assert len(all_rows) == len(full)
    assert len(np.unique(all_rows)) == len(full)


def test_split_matching_consumes_dropped_duplicate_occurrences():
    full = [("A", 0), ("A", 0), ("B", 1)]
    parts = {"train": [full[0]], "valid": [full[1]], "test": [full[2]]}

    selected = match_split_occurrences(full, parts, keep=[1, 2])

    assert selected["train"].size == 0
    np.testing.assert_array_equal(selected["valid"], [0])
    np.testing.assert_array_equal(selected["test"], [1])
