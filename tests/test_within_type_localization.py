"""Within-token-type localization: the estimand that the one-hot arm cannot pass.

The primary WSD rewards any coordinate that encodes the current token's
identity whenever a fragment is enriched for some atom type. These tests pin the
within-type estimand that removes that route: exact zero for anything constant
within token type, the primary score when a molecule has one token type, the
pooled per-type formula it claims to be, and the held-out protocol, eligibility
and aggregate reporting built on it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from molcircuits.analysis.fragment_controls import (
    WSD_ARMS, aggregate_wsd_fragment_results, detector_sharing,
    heldout_within_type_detectors, heldout_wsd_detectors,
    within_type_eligible_molecules, within_type_wsd_coordinates)
from molcircuits.analysis.wsd import basis_stats, mean_wsd_coordinates

SAE = "learned_sae_topk32_8192"
ONEHOT = "atom_type_onehot"


def _synthetic(n_molecules: int, seed: int, n_tokens: int = 14):
    """Two token types (0 = C, 1 = O); the fragment is every O plus the C
    atoms bonded to one, i.e. the in-fragment carbons differ from the others
    only by context.
    """
    rng = np.random.default_rng(seed)
    valid = np.zeros((n_molecules, n_tokens), dtype=bool)
    ids = np.zeros((n_molecules, n_tokens), dtype=np.int64)
    fragment = np.zeros_like(valid)
    for b in range(n_molecules):
        length = int(rng.integers(8, n_tokens + 1))
        valid[b, :length] = True
        oxygen = rng.random(length) < 0.3
        oxygen[int(rng.integers(0, length))] = True
        ids[b, :length] = oxygen.astype(np.int64)
        neighbour = np.zeros(length, dtype=bool)
        neighbour[:-1] |= oxygen[1:]
        neighbour[1:] |= oxygen[:-1]
        fragment[b, :length] = oxygen | (neighbour & ~oxygen)
    is_oxygen = (ids == 1) & valid
    context = fragment & ~is_oxygen
    noise = rng.normal(scale=0.3, size=(n_molecules, n_tokens, 3))
    acts = np.concatenate([
        is_oxygen[..., None].astype(np.float32),                      # 0: identity
        (context + 0.3 * rng.normal(size=valid.shape))[..., None],   # 1: context
        noise,                                                        # 2-4: noise
    ], axis=-1).astype(np.float32)
    acts *= valid[..., None]
    return acts, fragment, valid, ids


def _reference(acts, fragment, valid, ids):
    """Per-type pooled form, molecule by molecule, in float64."""
    acts = np.asarray(acts, dtype=np.float64)
    B, _, K = acts.shape
    totals = np.zeros(K)
    n_eligible = 0
    for b in range(B):
        v = valid[b]
        x = acts[b, v]
        y = fragment[b, v]
        t = ids[b, v]
        sigma = x.std(axis=0) + 1e-12
        numerator, weight = np.zeros(K), 0.0
        for tid in np.unique(t):
            inside = (t == tid) & y
            outside = (t == tid) & ~y
            if inside.any() and outside.any():
                w = inside.sum() * outside.sum() / (t == tid).sum()
                numerator += w * (x[inside].mean(0) - x[outside].mean(0))
                weight += w
        if weight == 0:
            continue
        n_eligible += 1
        totals += numerator / sigma / np.sqrt(len(t) * weight)
    return totals / n_eligible, n_eligible


def test_onehot_basis_scores_exactly_zero_within_type():
    rng = np.random.default_rng(3)
    ids = rng.integers(1, 6, size=(40, 16))
    valid = rng.random((40, 16)) < 0.85
    # Enriched for type 1, as an aromatic ring is for ``c``.
    fragment = (ids == 1) | (rng.random((40, 16)) < 0.15)
    onehot = np.eye(6, dtype=np.float32)[ids] * valid[..., None]

    signed, absolute, n = within_type_wsd_coordinates(
        onehot, fragment, valid, ids, chunk=7)
    primary, _ = mean_wsd_coordinates(onehot, fragment, valid)

    assert n[0] > 0
    assert np.all(signed == 0.0) and np.all(absolute == 0.0)
    # The same basis is far from zero on the primary estimand.
    assert np.nanmax(np.abs(primary)) > 0.05


def test_any_type_constant_coordinate_is_zero_within_type():
    """Exact for 0/1 columns; for arbitrary per-type constants only the
    rounding of a float32 mean remains."""
    rng = np.random.default_rng(4)
    ids = rng.integers(0, 4, size=(30, 12))
    valid = np.ones_like(ids, dtype=bool)
    fragment = rng.random(ids.shape) < 0.4
    lookup = rng.normal(size=(4, 5)).astype(np.float32) * 1e3
    acts = lookup[ids]
    signed, absolute, n = within_type_wsd_coordinates(
        acts, fragment, valid, ids)
    primary, _ = mean_wsd_coordinates(acts, fragment, valid)
    assert n[0] > 0
    assert np.abs(signed).max() < 1e-6 and np.abs(absolute).max() < 1e-6
    assert np.abs(primary).max() > 1e3 * np.abs(signed).max()


def test_single_token_type_reduces_to_primary_wsd():
    rng = np.random.default_rng(5)
    acts = rng.normal(size=(25, 10, 4)).astype(np.float32)
    valid = np.ones((25, 10), dtype=bool)
    fragment = np.zeros_like(valid)
    for b in range(25):
        fragment[b, rng.choice(10, size=int(rng.integers(1, 9)),
                               replace=False)] = True
    ids = np.zeros((25, 10), dtype=np.int64)

    signed, absolute, n = within_type_wsd_coordinates(
        acts, fragment, valid, ids)
    primary_signed, primary_absolute = mean_wsd_coordinates(
        acts, fragment, valid)

    assert n[0] == 25
    np.testing.assert_allclose(signed, primary_signed, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(absolute, primary_absolute, rtol=1e-5, atol=1e-6)


def test_matches_pooled_per_type_reference_and_is_chunk_invariant():
    acts, fragment, valid, ids = _synthetic(60, seed=6)
    expected, n_expected = _reference(acts, fragment, valid, ids)
    stats = basis_stats(acts, valid, 11)
    for chunk, batch in ((7, 1), (64, 3)):
        masks = np.stack([fragment, fragment, ~fragment & valid], axis=1)
        signed, _, n = within_type_wsd_coordinates(
            acts, masks, valid, ids, stats=stats, chunk=chunk,
            mask_batch=batch)
        np.testing.assert_allclose(signed[0], expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(signed[1], signed[0], rtol=0, atol=0)
        assert n[0] == n_expected
        # The complement has the same eligible molecules and, within each
        # type, the opposite contrast.
        assert n[2] == n_expected
        np.testing.assert_allclose(signed[2], -signed[0], rtol=1e-4, atol=1e-6)


def test_eligibility_requires_a_shared_type_on_both_sides():
    # Toluene for benzene: every aromatic c is in the match, the methyl C is
    # not, so no type straddles the boundary.
    valid = np.ones((3, 7), dtype=bool)
    ids = np.array([[1, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2],
                    [1, 2, 2, 2, 2, 2, 2]])
    fragment = np.array([[0, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 0]], dtype=bool)
    eligible = within_type_eligible_molecules(fragment, valid, ids)
    assert eligible.tolist() == [False, False, True]

    acts = np.random.default_rng(7).normal(size=(3, 7, 2)).astype(np.float32)
    _, _, n = within_type_wsd_coordinates(acts, fragment, valid, ids)
    assert n.tolist() == [1]

    none = within_type_wsd_coordinates(
        acts[:2], fragment[:2], valid[:2], ids[:2])
    assert none[2].tolist() == [0]
    assert np.isnan(none[0]).all() and np.isnan(none[1]).all()


def test_eligibility_ignores_invalid_positions():
    valid = np.array([[1, 1, 1, 0]], dtype=bool)
    ids = np.array([[1, 2, 2, 1]])
    fragment = np.array([[1, 0, 0, 0]], dtype=bool)
    # The only other type-1 token is padding, so nothing straddles.
    assert not within_type_eligible_molecules(fragment, valid, ids)[0]


def test_context_coordinate_scores_high_within_type_identity_does_not():
    """Held out: identity wins nothing within type; context keeps its score."""
    sel = _synthetic(300, seed=10)
    ev = _synthetic(300, seed=11)

    sel_primary = mean_wsd_coordinates(sel[0], sel[1], sel[2])
    ev_primary = mean_wsd_coordinates(ev[0], ev[1], ev[2])
    sel_within = within_type_wsd_coordinates(*sel)
    ev_within = within_type_wsd_coordinates(*ev)

    # The identity coordinate alone: high on the primary estimand, exactly
    # zero within type.
    assert ev_primary[0][0, 0] > 0.3
    assert ev_within[0][0, 0] == 0.0
    # The context coordinate is high within type.
    assert ev_within[0][0, 1] > 0.3

    within = heldout_within_type_detectors(
        sel_within[0][0], sel_within[1][0], ev_within[0][0], ev_within[1][0],
        sel_within[2][0], ev_within[2][0])
    assert within["status"] == "ok"
    assert within["oriented_detector"]["selected_coordinate"] == 1
    assert within["oriented_detector"]["evaluation_oriented_wsd"] > 0.3
    assert not within["selection_signal_identically_zero"]
    assert within["n_selection_molecules_eligible"] == sel_within[2][0]

    # An identity-only basis (the identity column plus noise) scores about
    # zero held-out within type while its primary detector is strong.
    keep = [0, 2, 3, 4]
    identity_sel = within_type_wsd_coordinates(sel[0][..., keep], *sel[1:])
    identity_ev = within_type_wsd_coordinates(ev[0][..., keep], *ev[1:])
    identity = heldout_within_type_detectors(
        identity_sel[0][0], identity_sel[1][0], identity_ev[0][0],
        identity_ev[1][0], identity_sel[2][0], identity_ev[2][0])
    assert abs(identity["oriented_detector"]["evaluation_oriented_wsd"]) < 0.05
    primary = heldout_wsd_detectors(
        sel_primary[0][0][keep], sel_primary[1][0][keep],
        ev_primary[0][0][keep], ev_primary[1][0][keep])
    assert primary["oriented_detector"]["selected_coordinate"] == 0
    assert primary["oriented_detector"]["evaluation_oriented_wsd"] > 0.3


def test_within_type_detector_is_sign_invariant():
    sel = _synthetic(80, seed=12)
    ev = _synthetic(80, seed=13)
    first = heldout_within_type_detectors(
        *(x[0] for x in within_type_wsd_coordinates(*sel)[:2]),
        *(x[0] for x in within_type_wsd_coordinates(*ev)[:2]), 1, 1)
    flipped_sel = (-sel[0], *sel[1:])
    flipped_ev = (-ev[0], *ev[1:])
    second = heldout_within_type_detectors(
        *(x[0] for x in within_type_wsd_coordinates(*flipped_sel)[:2]),
        *(x[0] for x in within_type_wsd_coordinates(*flipped_ev)[:2]), 1, 1)
    a, b = first["oriented_detector"], second["oriented_detector"]
    assert a["selected_coordinate"] == b["selected_coordinate"]
    assert a["orientation_multiplier"] == -b["orientation_multiplier"]
    assert a["evaluation_oriented_wsd"] == pytest.approx(
        b["evaluation_oriented_wsd"], rel=1e-5)


def test_onehot_heldout_within_type_is_zero_and_flagged():
    zeros = np.zeros(6)
    result = heldout_within_type_detectors(zeros, zeros, zeros, zeros, 20, 18)
    assert result["status"] == "ok"
    assert result["oriented_detector"]["evaluation_oriented_wsd"] == 0.0
    assert result["selection_signal_identically_zero"] is True
    json.dumps(result, allow_nan=False)


def _arm_result(primary: float, within: float | None, coordinate: int,
                within_coordinate: int, zero: bool = False) -> dict:
    result = {
        "status": "ok",
        "oriented_detector": {
            "selected_coordinate": coordinate,
            "evaluation_oriented_wsd": primary,
            "pipeline_null": {"observed_exceeds_p95": True}},
        "absolute_detector": {
            "selected_coordinate": coordinate,
            "evaluation_mean_absolute_wsd": abs(primary),
            "pipeline_null": {"observed_exceeds_p95": True}},
    }
    if within is not None:
        result["within_type"] = {
            "status": "ok",
            "selection_signal_identically_zero": zero,
            "oriented_detector": {
                "selected_coordinate": within_coordinate,
                "evaluation_oriented_wsd": within,
                "pipeline_null_type_matched": {
                    "observed_exceeds_p95": within > 0.1}},
            "absolute_detector": {
                "selected_coordinate": within_coordinate,
                "evaluation_mean_absolute_wsd": abs(within)},
        }
    return result


def _fragments() -> dict:
    # benzene has no within-type result: every eligible molecule fell short.
    spec = {"benzene": (0.6, None, 5), "phenol": (0.5, 0.3, 5),
            "ketone": (0.4, 0.2, 7), "amide": (0.45, 0.25, 5)}
    fragments = {}
    for name, (primary, within, coordinate) in spec.items():
        arms = {}
        for arm in WSD_ARMS:
            if arm == ONEHOT:
                arms[arm] = _arm_result(primary + 0.05,
                                        None if within is None else 0.0,
                                        coordinate=1, within_coordinate=0,
                                        zero=True)
            else:
                arms[arm] = _arm_result(primary, within, coordinate,
                                        within_coordinate=coordinate + 1)
        fragments[name] = {"arms": arms}
    return fragments


def test_aggregate_reports_within_type_with_onehot_and_its_contrast():
    out = aggregate_wsd_fragment_results(
        _fragments(), n_bootstrap=200, bootstrap_seed=3)

    for block in ("primary_oriented_wsd", "within_type_oriented_wsd"):
        assert ONEHOT in out[block]["arms"]
        assert f"{SAE}_minus_{ONEHOT}" in out[block]["paired_contrasts"]

    within = out["within_type_oriented_wsd"]
    assert within["pipeline_null_key"] == "pipeline_null_type_matched"
    assert within["arms"][SAE]["n_fragments"] == 3
    assert within["arms"][SAE]["mean"] == pytest.approx(0.25)
    assert within["arms"][ONEHOT]["mean"] == 0.0
    assert within["arms"][SAE]["n_exceeds_pipeline_null_p95"] == 3
    contrast = within["paired_contrasts"][f"{SAE}_minus_{ONEHOT}"]
    assert contrast["n_fragments"] == 3
    assert contrast["mean_difference"] == pytest.approx(0.25)
    assert contrast["reference_wins"] == 3
    boot = contrast["fragment_family_cluster_bootstrap"]
    assert boot["method"] == "fragment-family cluster bootstrap"
    assert boot["n_bootstrap"] == 200 and boot["seed"] == 3

    primary = out["primary_oriented_wsd"]
    assert primary["arms"][SAE]["n_fragments"] == 4
    assert primary["paired_contrasts"][f"{SAE}_minus_{ONEHOT}"][
        "mean_difference"] == pytest.approx(-0.05)
    json.dumps(out, allow_nan=False)


def test_detector_sharing_counts_distinct_and_reused_coordinates():
    out = aggregate_wsd_fragment_results(
        _fragments(), n_bootstrap=50, bootstrap_seed=0)

    primary = out["primary_oriented_wsd"]["detector_sharing"]
    assert set(primary) == set(WSD_ARMS)
    sae = primary[SAE]
    assert sae["n_fragments_with_selected_coordinate"] == 4
    assert sae["n_distinct_coordinates"] == 2
    assert sae["shared_coordinates"] == [{
        "coordinate": 5, "n_fragments": 3,
        "fragments": ["amide", "benzene", "phenol"]}]
    assert primary[ONEHOT]["n_distinct_coordinates"] == 1

    within = out["within_type_oriented_wsd"]["detector_sharing"]
    assert within[SAE]["n_fragments_with_selected_coordinate"] == 3
    assert within[SAE]["shared_coordinates"][0]["fragments"] == [
        "amide", "phenol"]
    # A zero-signal one-hot "detector" is an arbitrary column, not reuse.
    assert within[ONEHOT]["n_distinct_coordinates"] == 0
    assert within[ONEHOT]["n_fragments_without_selection_signal"] == 3
    assert within[ONEHOT]["shared_coordinates"] == []


def test_detector_sharing_on_raw_rows():
    fragments = {
        "a": {"arms": {"x": {"oriented_detector": {"selected_coordinate": 3}}}},
        "b": {"arms": {"x": {"oriented_detector": {"selected_coordinate": 3}}}},
        "c": {"arms": {"x": {"oriented_detector": {"selected_coordinate": 4}}}},
        "d": {"arms": {"x": {"oriented_detector": {"selected_coordinate": None}}}},
    }
    out = detector_sharing(fragments, ["x", "missing"])
    assert out["x"]["n_distinct_coordinates"] == 2
    assert out["x"]["n_fragments_with_selected_coordinate"] == 3
    assert out["x"]["shared_coordinates"] == [
        {"coordinate": 3, "n_fragments": 2, "fragments": ["a", "b"]}]
    assert out["missing"]["n_distinct_coordinates"] == 0


def test_stage17_within_type_smoke(tmp_path):
    """End to end: within-type rows, eligibility, skips and aggregate keys."""
    from molcircuits.chem.tokenizer import SmilesTokenizer
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    # Every alcohol has carbons outside the C-O match, so the alcohol fragment
    # is eligible in every molecule. Every oxygen is inside an [OX2] match, so
    # that fragment has no within-type contrast and must be skipped.
    corpus = ["CCCO", "CCCCO", "OCCCC", "CC(C)CO", "CCC(C)O", "OCCC",
              "CCCCCO", "CCOCC", "OCCCCC", "CC(O)CC"]
    data = tmp_path / "data"
    tf_dir = tmp_path / "tf"
    sae_dir = tmp_path / "sae"
    out_dir = tmp_path / "out"
    data.mkdir()
    tf_dir.mkdir()
    (data / "valid.smi").write_text("\n".join(corpus))
    tokenizer = SmilesTokenizer.from_corpus(corpus)
    tokenizer.save(data / "tokenizer.json")
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=512, n_layers=1,
                      n_heads=8, d_ff=64, n_ctx=16)
    model = MolecularTransformer(cfg)
    torch.save({"model": model.state_dict(), "model_cfg": vars(cfg)},
               tf_dir / "checkpoint.pt")
    SparseAutoencoder(SAEConfig(
        d_in=512, expansion=4, variant="topk", k=32, layer=0)).save(sae_dir)
    smarts = tmp_path / "smarts.json"
    smarts.write_text(json.dumps({"oxygen": "[OX2]",
                                  "alcohol_c_o": "[CX4][OX2H]"}))

    root = Path(__file__).parents[1]
    command = [
        sys.executable,
        str(root / "experiments" / "07_sparse_and_downstream.py"),
        "wsd-controls",
        "--ckpt", str(tf_dir / "checkpoint.pt"),
        "--sae", str(sae_dir), "--data", str(data), "--out", str(out_dir),
        "--smarts", str(smarts), "--n-molecules", "10", "--n-ctx", "16",
        "--selection-split-seed", "0", "--projection-seed", "1",
        "--permutation-seed", "3", "--n-perm", "3",
        "--permutation-batch", "2", "--molecule-chunk", "3",
        "--n-bootstrap", "50", "--bootstrap-seed", "2",
        "--min-molecules-per-half", "2", "--model-batch", "4",
        "--projection-batch", "16", "--device", "cpu",
    ]
    env = os.environ.copy()
    env["MOLCIRCUITS_MANIFEST"] = str(tmp_path / "manifest.jsonl")
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(command, cwd=root, env=env, text=True,
                               capture_output=True, timeout=300)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    doc = json.loads((out_dir / "fragment_alignment_controls_v2.json").read_text())
    assert doc["status"] == "complete"
    assert doc["within_type_estimand"]["aggregate_location"] == (
        "aggregate.within_type_oriented_wsd")
    assert "oxygen" in doc["skipped_fragments_within_type"]
    assert doc["skipped_fragments_within_type"]["oxygen"][
        "n_selection_molecules_eligible"] == 0
    oxygen = doc["fragments"]["oxygen"]
    assert oxygen["within_type_eligibility"]["scored"] is False
    assert all("within_type" not in arm for arm in oxygen["arms"].values())

    alcohol = doc["fragments"]["alcohol_c_o"]
    eligibility = alcohol["within_type_eligibility"]
    assert eligibility["scored"] is True
    assert eligibility["n_selection_molecules_eligible"] >= 2
    sae_arm = "learned_sae_topk32_2048"
    arms = set(doc["completed_arms"])
    assert {sae_arm, ONEHOT} <= arms
    for arm in arms:
        within = alcohol["arms"][arm]["within_type"]
        assert within["n_selection_molecules_eligible"] == eligibility[
            "n_selection_molecules_eligible"]
        assert within["oriented_detector"]["pipeline_null_type_matched"][
            "n_perm_requested"] == 3
    onehot = alcohol["arms"][ONEHOT]["within_type"]
    assert onehot["oriented_detector"]["evaluation_oriented_wsd"] == 0.0
    assert onehot["selection_signal_identically_zero"] is True

    aggregate = doc["aggregate"]
    for block in ("primary_oriented_wsd", "within_type_oriented_wsd"):
        assert ONEHOT in aggregate[block]["arms"]
        assert f"{sae_arm}_minus_{ONEHOT}" in aggregate[block]["paired_contrasts"]
        assert set(aggregate[block]["detector_sharing"]) == arms
    assert aggregate["within_type_oriented_wsd"]["arms"][ONEHOT]["mean"] == 0.0
    assert aggregate["within_type_oriented_wsd"]["arms"][sae_arm][
        "n_fragments"] == 1


def test_aggregate_without_any_within_type_fragment_is_json_null():
    fragments = _fragments()
    for row in fragments.values():
        for result in row["arms"].values():
            result.pop("within_type", None)
    out = aggregate_wsd_fragment_results(fragments, n_bootstrap=20)
    within = out["within_type_oriented_wsd"]
    assert within["arms"][SAE]["status"] == "unavailable"
    assert within["arms"][SAE]["mean"] is None
    contrast = within["paired_contrasts"][f"{SAE}_minus_{ONEHOT}"]
    assert contrast["mean_difference"] is None
    assert contrast["fragment_family_cluster_bootstrap"]["ci95"] == [None, None]
    json.dumps(out, allow_nan=False)
