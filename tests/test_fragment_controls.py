"""Regressions and a CPU smoke test for stage-17 matched controls."""
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
    ARMS, WSD_ARMS, aggregate_fragment_results, aggregate_wsd_fragment_results,
    coordinate_aurocs, fit_selection_standardizer,
    fragment_family_cluster_bootstrap, heldout_coordinate_alignment,
    heldout_wsd_detectors, heldout_wsd_pipeline_null,
    permute_fragment_masks_within_molecules, random_unit_projection,
    rank_coordinates, relu_topk)
from molcircuits.analysis.fragment_families import family_map
from molcircuits.analysis.wsd import mean_wsd_coordinates


def test_heldout_alignment_fixes_coordinate_and_orientation_on_selection():
    labels = np.array([False, False, True, True])
    # Coordinate 0 wins selection. Coordinate 1 becomes the evaluation oracle,
    # but must not replace it there.
    selection = np.array([[0., 0.], [1., 4.], [3., 1.], [4., 3.]])
    evaluation = np.array([[0., 0.], [3., 1.], [1., 3.], [2., 4.]])

    result = heldout_coordinate_alignment(
        rank_coordinates(selection), rank_coordinates(evaluation), labels, labels)

    assert result["selected_coordinate"] == 0
    assert result["selection_alignment"] == pytest.approx(1.0)
    assert result["evaluation_alignment"] == pytest.approx(0.5)


def test_heldout_alignment_is_invariant_to_coordinate_sign():
    labels = np.array([False, False, True, True])
    selection = np.array([[4., 0.], [3., 1.], [1., 2.], [0., 3.]])
    evaluation = np.array([[3., 0.], [2., 1.], [1., 2.], [0., 3.]])
    original = heldout_coordinate_alignment(
        rank_coordinates(selection), rank_coordinates(evaluation), labels, labels)
    selection[:, 0] *= -1
    evaluation[:, 0] *= -1
    flipped = heldout_coordinate_alignment(
        rank_coordinates(selection), rank_coordinates(evaluation), labels, labels)

    assert original["selected_coordinate"] == flipped["selected_coordinate"] == 0
    assert original["orientation"] != flipped["orientation"]
    assert original["selection_alignment"] == pytest.approx(
        flipped["selection_alignment"])
    assert original["evaluation_alignment"] == pytest.approx(
        flipped["evaluation_alignment"])


def test_heldout_alignment_encodes_dead_evaluation_coordinate_as_json_null():
    """A detector that becomes constant is unavailable, not NaN or chance."""
    labels = np.array([False, False, True, True])
    selection = rank_coordinates(np.array([[0.], [1.], [3.], [4.]]))
    evaluation = rank_coordinates(np.ones((4, 1)))

    result = heldout_coordinate_alignment(
        selection, evaluation, labels, labels)

    assert result["status"] == "selected_coordinate_not_evaluable"
    assert result["selected_coordinate"] == 0
    assert result["evaluation_raw_auroc"] is None
    assert result["evaluation_alignment"] is None
    assert result["evaluation_unavailable_reason"] == (
        "selected_coordinate_constant_on_evaluation")
    json.dumps(result, allow_nan=False)


def test_dense_controls_fit_selection_only_and_match_topk():
    selection = np.array([[0., 0., 1.], [2., 4., 1.]], dtype=np.float32)
    mean, scale = fit_selection_standardizer(selection)
    np.testing.assert_allclose(mean, [1., 2., 1.])
    assert scale[2] > 0  # constant coordinate receives a finite floor

    x = torch.tensor([[-1., 4., 3., 2.], [5., -2., 1., 4.]])
    z = relu_topk(x, 2)
    assert (z > 0).sum(1).tolist() == [2, 2]
    torch.testing.assert_close(z, torch.tensor([[0., 4., 3., 0.],
                                                [5., 0., 0., 4.]]))


def test_random_projection_is_reproducible_and_unit_norm():
    first = random_unit_projection(5, 17, seed=42)
    second = random_unit_projection(5, 17, seed=42)
    other = random_unit_projection(5, 17, seed=43)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.norm(dim=0), torch.ones(17))
    assert not torch.equal(first, other)


def test_all_negative_varying_dense_coordinate_is_not_dead():
    ranks = rank_coordinates(np.array([[-4., -1.], [-3., -1.],
                                       [-2., -1.], [-1., -1.]]))
    assert ranks["dead"].tolist() == [False, True]


def test_global_token_auroc_can_be_high_when_within_molecule_wsd_is_zero():
    """A molecule-presence coordinate is context, not local alignment."""
    acts = np.zeros((4, 3, 1), dtype=np.float32)
    acts[:2] = 1.0  # constant within fragment-containing molecules
    valid = np.ones((4, 3), dtype=bool)
    fragment = np.zeros((4, 3), dtype=bool)
    fragment[:2, 0] = True

    pooled_auc = coordinate_aurocs(
        rank_coordinates(acts.reshape(-1, 1)), fragment.reshape(-1))[0]
    signed, absolute = mean_wsd_coordinates(acts, fragment, valid)

    assert pooled_auc == pytest.approx(0.8)
    assert signed[0, 0] == pytest.approx(0.0)
    assert absolute[0, 0] == pytest.approx(0.0)


def test_heldout_wsd_fixes_primary_orientation_and_secondary_coordinate():
    result = heldout_wsd_detectors(
        selection_signed=np.array([0.8, -0.2]),
        selection_absolute=np.array([0.8, 1.0]),
        evaluation_signed=np.array([-0.4, 100.0]),
        evaluation_absolute=np.array([0.4, 0.25]))

    primary = result["oriented_detector"]
    secondary = result["absolute_detector"]
    assert primary["selected_coordinate"] == 0
    assert primary["evaluation_oriented_wsd"] == pytest.approx(-0.4)
    assert secondary["selected_coordinate"] == 1
    assert secondary["evaluation_mean_absolute_wsd"] == pytest.approx(0.25)

    flipped = heldout_wsd_detectors(
        np.array([-0.8, -0.2]), np.array([0.8, 1.0]),
        np.array([0.4, 100.0]), np.array([0.4, 0.25]))
    assert flipped["oriented_detector"]["orientation_multiplier"] == -1
    assert flipped["oriented_detector"]["evaluation_oriented_wsd"] == (
        pytest.approx(primary["evaluation_oriented_wsd"]))


def test_within_molecule_permutation_preserves_coverage_and_is_reproducible():
    valid = np.array([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
    fragment = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=bool)
    first = permute_fragment_masks_within_molecules(fragment, valid, 20, seed=8)
    second = permute_fragment_masks_within_molecules(fragment, valid, 20, seed=8)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 20, 4)
    np.testing.assert_array_equal(first.sum(axis=2), [[2] * 20, [1] * 20])
    assert not (first & ~valid[:, None, :]).any()


def test_pipeline_null_reselects_on_selection_and_scores_fixed_evaluation():
    selection_signed = np.array([[2.0, 0.0], [0.0, -3.0]])
    selection_absolute = np.abs(selection_signed)
    evaluation_signed = np.array([[0.5, 99.0], [99.0, -0.25]])
    evaluation_absolute = np.array([[0.5, 99.0], [99.0, 0.25]])
    observed = heldout_wsd_detectors(
        np.array([1.0, 0.0]), np.array([1.0, 0.0]),
        np.array([0.75, 0.0]), np.array([0.75, 0.0]))

    null = heldout_wsd_pipeline_null(
        selection_signed, selection_absolute,
        evaluation_signed, evaluation_absolute, observed, 11, 12)

    primary = null["oriented_detector"]
    assert primary["n_perm_valid"] == 2
    assert primary["mean"] == pytest.approx((0.5 + 0.25) / 2)
    assert primary["selection_seed"] == 11
    assert primary["evaluation_seed"] == 12


def test_aggregate_uses_only_paired_heldout_scores():
    fragments = {
        "a": {"arms": {arm: {"status": "ok", "evaluation_alignment": 0.7}
                         for arm in ARMS}},
        "b": {"arms": {arm: {"status": "ok", "evaluation_alignment": 0.6}
                         for arm in ARMS}},
    }
    fragments["a"]["arms"]["dense_neurons_512"]["evaluation_alignment"] = 0.5
    out = aggregate_fragment_results(
        fragments, n_bootstrap=100, bootstrap_seed=17)

    assert out["arms"]["learned_sae_topk32_8192"]["n_fragments"] == 2
    contrast = out["paired_contrasts"][
        "learned_sae_topk32_8192_minus_dense_neurons_512"]
    assert contrast["n_fragments"] == 2
    assert contrast["mean_difference"] == pytest.approx(0.1)
    boot = contrast["fragment_family_cluster_bootstrap"]
    assert boot["n_bootstrap"] == 100 and boot["seed"] == 17


def test_pooled_aggregate_encodes_no_finite_scores_without_nan():
    unavailable = {
        "status": "selected_coordinate_not_evaluable",
        "evaluation_alignment": None,
    }
    fragments = {
        "oxygen": {"arms": {arm: dict(unavailable) for arm in ARMS}},
    }

    out = aggregate_fragment_results(
        fragments, n_bootstrap=100, bootstrap_seed=17)

    for row in out["arms"].values():
        assert row["status"] == "unavailable"
        assert row["n_fragments"] == 0
        assert row["mean_evaluation_alignment"] is None
    for row in out["paired_contrasts"].values():
        assert row["status"] == "unavailable"
        assert row["mean_difference"] is None
        assert row["fragment_family_cluster_bootstrap"]["ci95"] == [None, None]
    json.dumps(out, allow_nan=False)


def test_fragment_family_bootstrap_is_shared_clustered_and_deterministic():
    differences = {"benzene": 0.2, "pyridine": 0.1, "amide": -0.1}
    first = fragment_family_cluster_bootstrap(differences, 500, seed=9)
    second = fragment_family_cluster_bootstrap(differences, 500, seed=9)

    assert family_map(differences)["benzene"] == family_map(differences)["pyridine"]
    assert first == second
    assert first["n_fragments"] == 3
    assert first["n_families"] == 2
    assert first["n_bootstrap"] == 500


def test_wsd_aggregate_reports_both_estimands_with_family_bootstrap():
    fragments = {}
    for i, fragment in enumerate(("benzene", "amide")):
        fragments[fragment] = {"arms": {}}
        for arm in ARMS:
            value = 0.4 + 0.1 * i
            if arm == "dense_neurons_512":
                value -= 0.2
            fragments[fragment]["arms"][arm] = {
                "status": "ok",
                "oriented_detector": {
                    "evaluation_oriented_wsd": value,
                    "pipeline_null": {"observed_exceeds_p95": True}},
                "absolute_detector": {
                    "evaluation_mean_absolute_wsd": value + 0.1,
                    "pipeline_null": {"observed_exceeds_p95": True}},
            }

    out = aggregate_wsd_fragment_results(
        fragments, n_bootstrap=100, bootstrap_seed=17)

    primary = out["primary_oriented_wsd"]
    assert primary["arms"]["learned_sae_topk32_8192"]["mean"] == (
        pytest.approx(0.45))
    contrast = primary["paired_contrasts"][
        "learned_sae_topk32_8192_minus_dense_neurons_512"]
    assert contrast["mean_difference"] == pytest.approx(0.2)
    assert contrast["fragment_family_cluster_bootstrap"]["n_bootstrap"] == 100
    assert "secondary_mean_absolute_wsd" in out


def test_stage17_cpu_smoke_writes_versioned_complete_artifact(tmp_path):
    """Run all four arms end-to-end at the production 512/8192 widths."""
    from molcircuits.chem.tokenizer import SmilesTokenizer
    from molcircuits.model.transformer import ModelConfig, MolecularTransformer
    from molcircuits.sae.sae import SAEConfig, SparseAutoencoder

    corpus = ["CCO", "CCN", "CO", "CCC", "COC", "CCCl", "OCC", "C#N"]
    data = tmp_path / "data"
    tf_dir = tmp_path / "tf"
    sae_dir = tmp_path / "sae"
    out_dir = tmp_path / "out"
    data.mkdir()
    tf_dir.mkdir()
    with open(data / "valid.smi", "w") as handle:
        handle.write("\n".join(corpus))
    tokenizer = SmilesTokenizer.from_corpus(corpus)
    tokenizer.save(data / "tokenizer.json")

    # The fixture itself must be reproducible. The producer now handles a
    # selected coordinate becoming constant on evaluation, but changing random
    # model weights should not randomly change which branch this smoke covers.
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=512, n_layers=1,
                      n_heads=8, d_ff=64, n_ctx=16)
    model = MolecularTransformer(cfg)
    torch.save({"model": model.state_dict(), "model_cfg": vars(cfg)},
               tf_dir / "checkpoint.pt")
    sae = SparseAutoencoder(SAEConfig(
        d_in=512, expansion=16, variant="topk", k=32, layer=0))
    sae.save(sae_dir)
    smarts = tmp_path / "smarts.json"
    with open(smarts, "w") as handle:
        json.dump({"oxygen": "[OX2]"}, handle)

    root = Path(__file__).parents[1]
    env = os.environ.copy()
    env["MOLCIRCUITS_MANIFEST"] = str(tmp_path / "manifest.jsonl")
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    v2_command = [
        sys.executable,
        str(root / "experiments" / "07_sparse_and_downstream.py"),
        "wsd-controls",
        "--ckpt", str(tf_dir / "checkpoint.pt"),
        "--sae", str(sae_dir), "--data", str(data), "--out", str(out_dir),
        "--smarts", str(smarts), "--n-molecules", "8", "--n-ctx", "16",
        "--selection-split-seed", "0", "--projection-seed", "1",
        "--permutation-seed", "3", "--n-perm", "2",
        "--permutation-batch", "1", "--molecule-chunk", "2",
        "--n-bootstrap", "100", "--bootstrap-seed", "2",
        "--min-molecules-per-half", "1", "--model-batch", "4",
        "--projection-batch", "16", "--device", "cpu",
    ]
    completed = subprocess.run(v2_command, cwd=root, env=env, text=True,
                               capture_output=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    v2_path = out_dir / "fragment_alignment_controls_v2.json"
    v2 = json.load(open(v2_path))
    assert v2_path.stat().st_size < 1_000_000
    assert v2["schema_version"] == 2
    json.dumps(v2, allow_nan=False)
    assert v2["status"] == "complete"
    # The v2 screen carries one arm v1 does not: a token-identity one-hot, whose
    # score bounds what any basis reaches from atom-type composition alone.
    assert set(v2["completed_arms"]) == set(WSD_ARMS)
    assert "atom_type_onehot" in WSD_ARMS and "atom_type_onehot" not in ARMS
    arm = v2["fragments"]["oxygen"]["arms"]["learned_sae_topk32_8192"]
    assert arm["status"] == "ok"
    assert arm["oriented_detector"]["pipeline_null"]["n_perm_requested"] == 2
    assert "primary_oriented_wsd" in v2["aggregate"]
    # Every [OX2] oxygen in this corpus is inside its own match, so no token
    # type straddles the boundary and the within-type estimand must skip the
    # fragment with its counts rather than score it.
    assert v2["within_type_estimand"]["combination_rule"]
    assert v2["fragments"]["oxygen"]["within_type_eligibility"]["scored"] is False
    assert v2["skipped_fragments_within_type"]["oxygen"][
        "n_evaluation_molecules_eligible"] == 0
    assert "within_type" not in arm
    within = v2["aggregate"]["within_type_oriented_wsd"]
    assert within["arms"]["learned_sae_topk32_8192"]["n_fragments"] == 0
    for block in ("primary_oriented_wsd", "within_type_oriented_wsd"):
        assert set(v2["aggregate"][block]["detector_sharing"]) == set(WSD_ARMS)
