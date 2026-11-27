"""Regression tests for the versioned, duplicate-safe TDC artifact."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "experiments" / "_downstream" / "tdc_admet.py"


def _module():
    spec = importlib.util.spec_from_file_location("tdc_artifact_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_partition_audit_records_an_exact_disjoint_partition():
    module = _module()
    audit = module.partition_audit(
        {"train": np.array([3, 0]), "valid": np.array([2]),
         "test": np.array([1])},
        4, ["Drug", "Y"])

    assert audit["complete_disjoint_partition"] is True
    assert audit["n_assigned"] == audit["n_unique_assigned"] == 4
    assert audit["identity_columns"] == ["Drug", "Y"]
    assert len(audit["partition_sha256"]) == 64


def test_partition_audit_rejects_overlap_in_its_persisted_flag():
    module = _module()
    audit = module.partition_audit(
        {"train": np.array([0, 1]), "valid": np.array([1]),
         "test": np.array([2])},
        4, ["row"])

    assert audit["complete_disjoint_partition"] is False
    assert audit["n_assigned"] == 4
    assert audit["n_unique_assigned"] == 3


def test_aggregate_never_promotes_runtime_to_a_scientific_metric():
    module = _module()
    row = {
        arm: {"auroc": 0.7, "auprc": 0.6, "fit_seconds": 123.0,
              "n_features": 8, "best_rounds": 4, "hit_cap": False}
        for arm in module.REPRESENTATIONS
    }
    aggregate = module.aggregate([row, row])

    for arm in module.REPRESENTATIONS:
        assert set(aggregate[arm]) == {"auroc", "auprc"}


def test_pooling_means_remain_within_homogeneous_metric_groups():
    module = _module()

    def aggregate(value):
        return {arm: {"auroc": {"mean": value},
                      "pearson_r": {"mean": value}}
                for arm in module.REPRESENTATIONS}

    results = {
        "cadme": {"assays": {
            "HLM": {"splits": {"scaffold": {"es_refit": {
                "agg": aggregate(0.6)}}}},
        }},
        "tdc": {
            "AMES": {"task": "classification", "agg": aggregate(0.7)},
            "Caco2": {"task": "regression", "agg": aggregate(0.8)},
        },
    }
    _, grouped, _ = module.pooling_verdict(results)

    assert set(grouped) == {
        "cADME/scaffold/es_refit/pearson_r",
        "TDC/classification/auroc",
        "TDC/regression/pearson_r",
    }
    assert all(group["n_endpoints"] == 1 for group in grouped.values())

