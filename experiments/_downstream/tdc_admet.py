#!/usr/bin/env python3
"""Downstream transfer: cADME (Biogen) + TDC ADMET, SAE vs dense vs ECFP.

    python scripts/09_tdc_admet.py --ckpt artifacts/tf_seed0/checkpoint.pt \
        --sae artifacts/saes_seed0/layer3_x16 --out artifacts/tdc_seed0

What this reproduces, and what it fixes
---------------------------------------
The original JAX pipeline (``lmkit/experiments/downstream_eval.py``,
``tdc_benchmarks.py``) built its two learned representations like this::

    dense_pooled = masked_dense.sum(axis=1) / seq_lengths   # MEAN pool
    sae_pooled   = masked_sae.max(axis=1)                   # MAX pool

That is not a like-for-like comparison. Max-pooling a non-negative sparse code
keeps "was this feature ever strongly active anywhere in the molecule", which is
exactly the right inductive bias for a substructure-driven property. Mean-pooling
a roughly zero-centred residual stream cancels per-position signal. So a gap
between "SAE Features" and "Transformer Embeddings" in the original tables
confounds the *representation* with the *pooling operator*.

This script therefore runs the full 2x2 -- {dense, sae} x {mean, max} -- plus
ECFP, on every endpoint, so the pooling contribution can be read off directly.
Everything else (XGBoost head, hyperparameters, early stopping, refit-on-
train+valid, metric definitions, scaffold splits) follows the original.

Benchmarks
----------
*cADME*: the actual Biogen "Computational ADME" public set (Fang et al. 2023),
3521 compounds, downloaded from github.com/molecularinformatics/Computational-ADME.
This has the four assays the paper names as continuous log-scale endpoints, so
Pearson r is comparable to the paper's numbers. TDC's ``hlm``/``rlm`` are *not*
substitutes: they are binarised stability datasets, not log CLint.

*TDC*: the nine-task "core9" panel the original plotting script pins
(``make_downstream_plots.py``), loaded through PyTDC with its standard seeded
scaffold splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (average_precision_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)

from molcircuits import paths
from molcircuits.downstream.features import (ecfp, embed, load_stack,
                                         match_split_occurrences, random_active_subset)
from molcircuits.utils.manifest import Manifest, dump_json, file_sha256

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

CADME_URL = ("https://raw.githubusercontent.com/molecularinformatics/"
             "Computational-ADME/main/ADME_public_set_3521.csv")

# The four assays the paper names, mapped to their column in the public set.
CADME_ASSAYS = {
    "HLM":        "LOG HLM_CLint (mL/min/kg)",
    "MDR1_ER":    "LOG MDR1-MDCK ER (B-A/A-B)",
    "RLM":        "LOG RLM_CLint (mL/min/kg)",
    "Solubility": "LOG SOLUBILITY PH 6.8 (ug/mL)",
}

# "core9": the exact task list pinned in lmkit/make_downstream_plots.py.
# (family, task) -- family is the tdc.single_pred class name.
TDC_TASKS = [
    ("Tox",  "AMES"),
    ("ADME", "BBB_Martins"),
    ("ADME", "CYP2D6_Veith"),
    ("ADME", "CYP3A4_Veith"),
    ("ADME", "Pgp_Broccatelli"),
    ("Tox",  "hERG"),
    ("ADME", "Caco2_Wang"),
    ("ADME", "Half_Life_Obach"),
    ("ADME", "PPBR_AZ"),
    # Added here, and flagged as an addition: these are the TDC endpoints
    # closest to two of the four cADME assays, so the cADME result has a
    # public cross-check on a standard scaffold split.
    ("ADME", "Solubility_AqSolDB"),
    ("ADME", "Clearance_Microsome_AZ"),
]

ARTIFACT_SCHEMA_VERSION = 2

# The 2x2 plus the chemistry baseline. Names are stable keys in the JSON.
BASE_REPRESENTATIONS = ["sae_max", "sae_mean", "dense_max", "dense_mean", "ecfp"]

#: 512-wide coordinate subsets of the max-pooled sparse code. These cannot be
#: built in `featurize`, because the selection has to be fit on a split's TRAINING
#: rows and the splits are not known until `run_split`. They are listed here so
#: aggregation, contrasts and the JSON schema pick them up like any other arm.
#:
#: The pair is the whole point: same width, same sparsity, same axis alignment,
#: differing only in whether the coordinates were chosen by training variance. A
#: PCA or random-projection view is a dense MIXTURE and so cannot separate "a
#: sparse axis" from "a selected sparse axis".
#: Three independent random draws, not one. A single random subset estimates the
#: "unselected sparse axes" arm with a per-task standard deviation of 0.05-0.10 on
#: these benchmarks -- the same size as the effect -- so one draw cannot separate a
#: null from a noise excursion. MoleculeACE already averages three; matching that
#: here is what makes the two benchmarks comparable rather than merely both run.
RAND_DRAWS = 3
def representation_names(dim):
    return [*BASE_REPRESENTATIONS, f"sae_top{dim}", f"sae_rand{dim}",
            *(f"sae_rand{dim}_draw{r}" for r in range(1, RAND_DRAWS))]


# Historical base-model names exposed to artifact consumers.
REPRESENTATIONS = representation_names(512)
SCIENTIFIC_METRICS = {
    "auroc", "auprc", "accuracy", "pearson_r", "spearman_r", "rmse", "mae",
}

# Original arm names, so the 2x2 can be mapped back onto the paper's table.
PAPER_ARM = {"sae_max": "SAE Features (paper)",
             "dense_mean": "Transformer Embeddings (paper)",
             "ecfp": "ECFP Fingerprints (paper)",
             "sae_mean": "(new) SAE, mean-pooled",
             "dense_max": "(new) dense, max-pooled"}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------
# featurisation -- one code path, five arms
# --------------------------------------------------------------------------

def featurize(smiles, model, tok, sae, layer, batch, device, ecfp_bits, n_ctx):
    """All five representations for one molecule list.

    Returns ``(feats, keep)`` where ``feats`` maps arm name -> (n_keep, d) and
    ``keep`` indexes back into ``smiles``. Molecules the tokenizer rejects, or
    that RDKit cannot parse (their ECFP would be an all-zero row, which is not
    a representation but an absence of one), are dropped from every arm at once
    so all five see identical rows.
    """
    dense, sparse, drop_max = embed(
        smiles, model, tok, sae, layer=layer, pooling="both", n_ctx=n_ctx,
        batch=batch, device=device, return_metadata=True)
    fp = ecfp(smiles, n_bits=ecfp_bits).astype(np.float32)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    bad_rdkit = [i for i, s in enumerate(smiles)
                 if not s or Chem.MolFromSmiles(s) is None]

    drop = set(drop_max["dropped_indices"]) | set(bad_rdkit)
    keep = np.array([i for i in range(len(smiles)) if i not in drop], dtype=int)
    feats = {"dense_max": dense["max"][keep], "dense_mean": dense["mean"][keep],
             "sae_max": sparse["max"][keep], "sae_mean": sparse["mean"][keep],
             "ecfp": fp[keep]}
    dropped = sorted(drop)
    rdkit_failed = sorted(set(bad_rdkit))
    return feats, keep, {
        "drop_policy": "drop_from_all_representations",
        "n_input": len(smiles),
        "n_ctx": int(n_ctx),
        "n_dropped": len(dropped),
        "dropped_indices": dropped,
        "n_tokenizer_failed": drop_max["n_tokenizer_failed"],
        "tokenizer_failed_indices": drop_max["tokenizer_failed_indices"],
        "n_overlength": drop_max["n_overlength"],
        "overlength_indices": drop_max["overlength_indices"],
        "n_rdkit_failed": len(rdkit_failed),
        "rdkit_failed_indices": rdkit_failed,
    }


# --------------------------------------------------------------------------
# shared XGBoost head -- identical for every representation
# --------------------------------------------------------------------------

def _base_params(n_jobs, seed, max_rounds, device="cpu"):
    """The original's hyperparameters (downstream_eval.py:264-277).

    The one change is ``n_estimators``: the original caps at 2000, which on the
    8192-dim SAE arms never early-stops and costs ~13 min per fit on CPU. The
    cap is exposed as ``--max-rounds`` and applied *identically to all five
    arms*; every fit records ``best_rounds`` and ``hit_cap`` so a cap that
    actually binds is visible rather than silent.
    """
    return dict(n_estimators=max_rounds, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                reg_alpha=0.0, random_state=seed, tree_method="hist",
                n_jobs=n_jobs, device=device)


def _fit_head(X_tr, y_tr, X_va, y_va, X_te, is_clf, n_jobs, seed, early_stop,
              protocol="es_refit", fixed_rounds=500, max_rounds=600, *,
              xgb_device="cpu"):
    """Shared head. Two protocols, identical across representations.

    ``es_refit`` (default, and what the original did): early-stop on the valid
    fold, then refit on train+valid at the selected round count.

    ``fixed``: no early stopping, a fixed round count, train+valid. This exists
    because early stopping on a small valid fold is noisy and stops different
    representations at wildly different round counts -- on Caco2 the ECFP arm
    halts after a couple of trees while the SAE arm runs for hundreds. That
    asymmetry is a property of the *protocol*, not of the features, and it
    would silently flatter whichever arm happens to get a longer fit. Both
    protocols use the same hyperparameters for all five arms.
    """
    p = _base_params(n_jobs, seed, max_rounds, device=xgb_device)
    if is_clf:
        pos, neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
        p.update(objective="binary:logistic", eval_metric="auc",
                 scale_pos_weight=float(neg) / max(pos, 1) if pos else 1.0)
        Model = xgb.XGBClassifier
    else:
        p.update(objective="reg:squarederror", eval_metric="rmse")
        Model = xgb.XGBRegressor

    if protocol == "es_refit":
        m = Model(early_stopping_rounds=early_stop, **p)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        bi = getattr(m, "best_iteration", None)
        best = (int(bi) + 1) if bi is not None else max_rounds
    elif protocol == "fixed":
        best = fixed_rounds
    else:
        raise ValueError(f"unknown protocol {protocol!r}")

    hit_cap = protocol == "es_refit" and best >= max_rounds
    p["n_estimators"] = max(1, best)
    final = Model(**p)
    final.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]), verbose=False)
    pred = (final.predict_proba(X_te)[:, 1] if is_clf else final.predict(X_te))
    return pred, best, hit_cap


def _metrics(y_true, pred, is_clf):
    if is_clf:
        auroc = (float(roc_auc_score(y_true, pred))
                 if len(np.unique(y_true)) > 1 else float("nan"))
        return {"auroc": auroc,
                "auprc": float(average_precision_score(y_true, pred))}
    r = float(pearsonr(pred, y_true)[0]) if np.std(pred) > 0 else float("nan")
    return {"pearson_r": r,
            "spearman_r": float(spearmanr(pred, y_true)[0]),
            "rmse": float(math.sqrt(mean_squared_error(y_true, pred))),
            "mae": float(mean_absolute_error(y_true, pred))}


def _fit_one_rep(name, X, y, tr, va, te, is_clf, n_jobs, seed, early_stop,
                 protocol, fixed_rounds, max_rounds, xgb_device):
    """One representation's fit. Module-level so loky can pickle it."""
    t0 = time.time()
    pred, best, cap = _fit_head(X[tr], y[tr], X[va], y[va], X[te],
                                is_clf, n_jobs, seed, early_stop,
                                protocol, fixed_rounds, max_rounds,
                                xgb_device=xgb_device)
    return name, {**_metrics(y[te], pred, is_clf),
                  "n_features": int(X.shape[1]), "best_rounds": best,
                  "hit_cap": bool(cap),
                  "fit_seconds": round(time.time() - t0, 1)}


def run_split(feats, y, tr, va, te, is_clf, n_jobs, seed, early_stop,
              protocol="es_refit", fixed_rounds=500, max_rounds=600, *,
              xgb_device="cpu", fit_workers=5):
    """One split, all five representations, shared head.

    The five fits run concurrently. They are independent and individually
    seeded, so worker count changes wall-clock and nothing else. On CUDA this is
    worth 2.55x (measured: 40.0s against 102.1s to run these five arms serially,
    on a card already at 100% utilisation), and serialising them was why the
    CUDA path first looked no faster than CPU.

    Fits are logged after collection rather than inside the loop: workers are
    separate processes, so a ``log`` call in one would not reach this process's
    stdout, and interleaved output from five fits would be unreadable anyway.
    """
    # Parallel preserves input order, so `out` keeps REPRESENTATIONS order and
    # the JSON stays comparable with earlier serial runs. `_fit_one_rep` is
    # module-level and takes its arrays explicitly rather than closing over
    # `feats`: a nested function is not reliably picklable to loky workers, and
    # passing one representation's matrix beats shipping all five to each.
    # Coordinate subsets are derived here, not in featurize, so the variance is
    # measured on THIS split's training rows only and never sees validation or
    # test molecules. The random subset is drawn from the coordinates that are
    # live on train, so "unselected" does not quietly mean "empty".
    feats = dict(feats)
    S = feats["sae_max"]
    v = S[tr].var(0)
    dim = min(feats["dense_max"].shape[1], S.shape[1])
    representations = representation_names(dim)
    top = np.argsort(-v, kind="stable")[:dim]
    rs = np.random.default_rng(seed + 77000)
    feats[f"sae_top{dim}"] = np.ascontiguousarray(S[:, np.sort(top)])
    for r in range(RAND_DRAWS):
        rand = random_active_subset(v, dim, rs)
        feats[f"sae_rand{dim}{'' if r == 0 else f'_draw{r}'}"] = \
            np.ascontiguousarray(S[:, np.sort(rand)])

    out = dict(Parallel(n_jobs=min(fit_workers, len(representations)))(
        delayed(_fit_one_rep)(
            name, feats[name], y, tr, va, te, is_clf, n_jobs, seed,
            early_stop, protocol, fixed_rounds, max_rounds, xgb_device)
        for name in representations))
    primary = "auroc" if is_clf else "pearson_r"
    for name in representations:
        r = out[name]
        log(f"      {name:<11} {primary}={r[primary]:.4f} "
            f"({r['fit_seconds']:.0f}s, {r['best_rounds']} rounds"
            f"{' CAPPED' if r['hit_cap'] else ''})")
    return out


def aggregate(per_seed):
    """mean / std across split seeds, per representation per metric."""
    agg = {}
    reps = dict.fromkeys(rep for row in per_seed for rep in row
                         if rep in BASE_REPRESENTATIONS or rep.startswith(("sae_top", "sae_rand")))
    for rep in reps:
        rows = [s[rep] for s in per_seed if rep in s]
        if not rows:
            continue
        # Runtime is a diagnostic, not a scientific endpoint.  Earlier
        # artifacts silently aggregated every float and therefore represented
        # fit_seconds beside AUROC/RMSE as though it were a result metric.
        keys = [k for k in rows[0] if k in SCIENTIFIC_METRICS]
        agg[rep] = {}
        for k in keys:
            v = np.array([r[k] for r in rows], dtype=float)
            agg[rep][k] = {"mean": float(np.nanmean(v)),
                           "std": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                           "seeds": [float(x) for x in v]}
    return agg


# --------------------------------------------------------------------------
# cADME
# --------------------------------------------------------------------------

def fetch_cadme(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        log(f"  downloading cADME -> {path}")
        urllib.request.urlretrieve(CADME_URL, path)
    return pd.read_csv(path)


def _ensure_pkg_resources():
    """Provide minimal pkg_resources shim for PyTDC on setuptools>=82."""
    import importlib
    try:
        importlib.import_module("pkg_resources")
        return
    except ModuleNotFoundError:
        import importlib.metadata as md
        import importlib.resources as res
        import types
        mod = types.ModuleType("pkg_resources")
        mod.get_distribution = lambda name: types.SimpleNamespace(version=md.version(name))
        mod.resource_filename = lambda pkg, rel: str(res.files(pkg).joinpath(rel))
        sys.modules["pkg_resources"] = mod


def scaffold_split(df, seed, frac=(0.7, 0.1, 0.2), col="SMILES"):
    """TDC's own Bemis-Murcko splitter, so cADME and TDC use the same rule."""
    _ensure_pkg_resources()
    from tdc.utils.split import create_scaffold_split
    d = df.reset_index(drop=True).copy()
    d["_row"] = np.arange(len(d))
    s = create_scaffold_split(d, seed, list(frac), col)
    return tuple(s[k]["_row"].values.astype(int) for k in ("train", "valid", "test"))


def random_split(n, seed, frac=(0.7, 0.1, 0.2)):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def partition_audit(parts, n_rows: int, identity_columns) -> dict:
    """Compact evidence that train/valid/test form one exact row partition."""
    named = {name: np.asarray(indices, dtype=int)
             for name, indices in parts.items()}
    assigned = np.concatenate(tuple(named.values()))
    serializable = {name: indices.tolist() for name, indices in named.items()}
    return {
        "identity_columns": list(identity_columns),
        "n_train": int(len(named["train"])),
        "n_valid": int(len(named["valid"])),
        "n_test": int(len(named["test"])),
        "n_assigned": int(len(assigned)),
        "n_unique_assigned": int(len(np.unique(assigned))),
        "n_featurized_rows": int(n_rows),
        "complete_disjoint_partition": bool(
            len(assigned) == n_rows and len(np.unique(assigned)) == n_rows
            and (not len(assigned)
                 or (assigned.min() >= 0 and assigned.max() < n_rows))),
        "partition_sha256": hashlib.sha256(
            json.dumps(serializable, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
    }


def run_cadme(args, model, tok, sae):
    log("\n" + "=" * 78 + "\ncADME (Biogen Computational-ADME public set, 3521 cmpds)\n" + "=" * 78)
    df = fetch_cadme(args.cadme_csv)
    smiles = df["SMILES"].tolist()

    log(f"  featurizing {len(smiles)} molecules (both poolings)…")
    t0 = time.time()
    feats_all, keep, drops = featurize(
        smiles, model, tok, sae, args.layer, args.batch, args.device,
        args.ecfp_bits, args.n_ctx)
    log(f"  done in {time.time()-t0:.0f}s; {drops}")
    df = df.iloc[keep].reset_index(drop=True)

    results = {"source": CADME_URL,
               "source_csv": os.path.realpath(args.cadme_csv),
               "source_csv_sha256": file_sha256(args.cadme_csv),
               "n_molecules": int(len(df)),
               "featurization": drops, "assays": {}}

    for assay, col in CADME_ASSAYS.items():
        have = df[col].notna().values
        sub_idx = np.where(have)[0]
        y = df[col].values[sub_idx].astype(np.float64)
        feats = {k: v[sub_idx] for k, v in feats_all.items()}
        sub_df = df.iloc[sub_idx].reset_index(drop=True)
        log(f"\n  --- {assay}  ({col})  n={len(y)} ---")

        entry = {"column": col, "n": int(len(y)), "task": "regression", "splits": {}}
        for split_name in args.cadme_splits:
            # splits are drawn once per seed and shared by both head protocols
            idx = [(scaffold_split(sub_df, s) if split_name == "scaffold"
                    else random_split(len(y), s)) for s in args.seeds]
            entry["splits"][split_name] = {}
            for proto in args.protocols:
                per_seed = []
                for s, (tr, va, te) in zip(args.seeds, idx):
                    log(f"    [{split_name} seed {s} / {proto}] train={len(tr)} "
                        f"valid={len(va)} test={len(te)}")
                    split_result = run_split(
                        feats, y, tr, va, te, False, args.n_jobs, args.seed,
                        args.early_stop, proto, args.fixed_rounds,
                        args.max_rounds, xgb_device=args.xgb_device,
                        fit_workers=args.fit_workers)
                    split_result["split_seed"] = int(s)
                    split_result["split_audit"] = partition_audit(
                        {"train": tr, "valid": va, "test": te}, len(y),
                        ["post-featurization assay row index"])
                    per_seed.append(split_result)
                entry["splits"][split_name][proto] = {
                    "per_seed": per_seed, "agg": aggregate(per_seed)}
        results["assays"][assay] = entry
        _checkpoint(args, {"cadme": results})
    return results


# --------------------------------------------------------------------------
# TDC
# --------------------------------------------------------------------------

def load_tdc(family, name, path):
    _ensure_pkg_resources()
    from tdc.single_pred import ADME, Tox
    cls = {"ADME": ADME, "Tox": Tox}[family]
    return cls(name=name, path=path)


def infer_task(y):
    """Original's rule (downstream_eval.py:132-150)."""
    u = np.unique(y)
    if y.dtype.kind in ("i", "u"):
        return "classification" if u.size <= 10 else "regression"
    if y.dtype.kind == "f":
        if u.size <= 10 and set(np.round(u, 6)).issubset({0.0, 1.0}):
            return "classification"
        return "regression"
    return "classification"


def run_tdc(args, model, tok, sae):
    log("\n" + "=" * 78 + "\nTDC ADMET (core9 + 2 cADME-analogue endpoints)\n" + "=" * 78)
    out = {}
    for family, name in TDC_TASKS:
        if args.tasks and name not in args.tasks:
            continue
        log(f"\n  === {name} ({family}) ===")
        ds = load_tdc(family, name, args.tdc_path)

        # Featurize the union of molecules once; splits just index into it.
        full = ds.get_data()
        dataset_content_sha256 = hashlib.sha256(
            full.to_json(orient="split", index=False,
                         double_precision=15).encode()).hexdigest()
        smi_col = next(c for c in ("Drug", "SMILES", "smiles", "X") if c in full.columns)
        smiles = full[smi_col].tolist()
        y_all = full["Y"].values
        kind = infer_task(y_all)
        log(f"    n={len(smiles)}  task={kind}")

        t0 = time.time()
        feats, keep, drops = featurize(
            smiles, model, tok, sae, args.layer, args.batch, args.device,
            args.ecfp_bits, args.n_ctx)
        log(f"    featurized in {time.time()-t0:.0f}s; {drops}")
        y = y_all[keep].astype(np.float64)
        if kind == "classification":
            y = (y > 0.5).astype(int)
        # ROW-OCCURRENCE-INDEXED, not SMILES-keyed. A dict keyed by SMILES
        # collapses duplicate strings onto the last occurrence; expanding a
        # SMILES to all matching rows has the opposite k**2 failure. The split
        # frames reset their indices, so match their shared row identity and
        # consume each original occurrence exactly once.
        # Measured: hERG carries 7 duplicate SMILES of which 3 disagree on Y
        # (rows 167/183/224 are Y=1, keyed onto SERTINDOLE duplicates with
        # Y=0), and correcting it moves the reported AUROC 0.7929 -> 0.8311,
        # larger than the +-0.022 split spread quoted beside it. BBB_Martins
        # has 10 conflicting, Caco2_Wang 4, Half_Life_Obach 2.
        #
        # scaffold_split() above already carries an explicit row id for cADME.
        kept_smiles = np.asarray(smiles, dtype=object)[keep]
        duplicate_table = pd.DataFrame({"smiles": kept_smiles, "y": y_all[keep]})
        group_sizes = duplicate_table.groupby("smiles", dropna=False).size()
        duplicate_groups = group_sizes[group_sizes > 1]
        conflicting_groups = duplicate_table.groupby(
            "smiles", dropna=False)["y"].nunique(dropna=False)
        conflicting_groups = conflicting_groups[conflicting_groups > 1]
        duplicate_audit = {
            "identity": "row occurrence (duplicate SMILES are not collapsed)",
            "n_kept_rows": int(len(kept_smiles)),
            "n_unique_smiles": int(group_sizes.size),
            "n_duplicate_smiles_groups": int(len(duplicate_groups)),
            "n_duplicate_extra_rows": int((duplicate_groups - 1).sum()),
            "n_conflicting_duplicate_smiles_groups": int(len(conflicting_groups)),
            "n_rows_in_conflicting_duplicate_groups": int(
                group_sizes.reindex(conflicting_groups.index).sum()),
        }
        if duplicate_audit["n_duplicate_extra_rows"]:
            log(f"    {duplicate_audit['n_duplicate_extra_rows']} duplicate-row "
                "occurrences kept separately; "
                f"{duplicate_audit['n_conflicting_duplicate_smiles_groups']} "
                "SMILES groups have conflicting labels")

        entry = {"family": family, "task": kind, "n": int(len(y)),
                 "featurization": drops, "split": args.split,
                 "protocol": args.tdc_protocol,
                 "dataset_content_sha256": dataset_content_sha256,
                 "duplicate_row_audit": duplicate_audit,
                 "per_seed": []}
        for s in args.seeds:
            sp = ds.get_split(method=args.split, seed=s)
            identity_cols = [
                c for c in full.columns
                if all(c in sp[part].columns for part in ("train", "valid", "test"))
            ]
            if not identity_cols:
                raise ValueError(f"{name}: split has no shared row identity columns")
            full_rows = list(full[identity_cols].itertuples(index=False, name=None))
            split_rows = {
                part: list(sp[part][identity_cols].itertuples(index=False, name=None))
                for part in ("train", "valid", "test")
            }
            sel = match_split_occurrences(full_rows, split_rows, keep)
            empty = [part for part, indices in sel.items() if not len(indices)]
            if empty:
                raise ValueError(f"{name} seed {s}: empty split partitions {empty}")
            log(f"    [{args.split} seed {s}] train={len(sel['train'])} "
                f"valid={len(sel['valid'])} test={len(sel['test'])}")
            split_result = run_split(
                feats, y, sel["train"], sel["valid"], sel["test"],
                kind == "classification", args.n_jobs, args.seed,
                args.early_stop, args.tdc_protocol, args.fixed_rounds,
                args.max_rounds, xgb_device=args.xgb_device,
                fit_workers=args.fit_workers)
            split_result["split_seed"] = int(s)
            split_result["split_audit"] = partition_audit(
                sel, len(y), identity_cols)
            entry["per_seed"].append(split_result)
        entry["agg"] = aggregate(entry["per_seed"])
        out[name] = entry
        _checkpoint(args, {"tdc": out})
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

_PARTIAL = {}


def _checkpoint(args, update):
    """Write results incrementally so a long run is never all-or-nothing."""
    _PARTIAL.update(update)
    dump_json(_PARTIAL, os.path.join(args.out, "tdc_results.json"))


def summary_table(results):
    """One row per endpoint, one column per representation, primary metric."""
    lines, rows = [], []
    for assay, e in results.get("cadme", {}).get("assays", {}).items():
        for split_name, protos in e["splits"].items():
            for proto, sd in protos.items():
                for m in ("pearson_r", "rmse"):
                    rows.append((f"cADME/{assay}", f"{split_name}/{proto}", m,
                                 sd["agg"]))
    for name, e in results.get("tdc", {}).items():
        if "agg" not in e or not e["agg"]:
            continue
        for m in (("auroc", "auprc") if e["task"] == "classification"
                  else ("pearson_r", "rmse")):
            rows.append((f"TDC/{name}",
                         f"{e.get('split','scaffold')}/{e.get('protocol','es_refit')}",
                         m, e["agg"]))

    representations = list(dict.fromkeys(name for _, _, _, agg in rows for name in agg))
    hdr = f"{'endpoint':<28}{'split':<20}{'metric':<11}" + "".join(
        f"{r:>15}" for r in representations)
    lines += [hdr, "-" * len(hdr)]
    for ep, sp, m, agg in rows:
        cells = ""
        for r in representations:
            v = agg.get(r, {}).get(m)
            cells += f"{v['mean']:>9.4f}±{v['std']:.3f}" if v else f"{'--':>15}"
        lines.append(f"{ep:<28}{sp:<20}{m:<11}{cells}")
    return "\n".join(lines)


def pooling_verdict(results):
    """The point of the 2x2: how much of the SAE gap is pooling, not features?"""
    out, group_members, lines = {}, {}, []
    def gap(agg, m, hi_good):
        g = {}
        dim = next((key.removeprefix("sae_top") for key in agg
                    if key.startswith("sae_top") and key[7:].isdigit()), None)
        for a, b, key in (("sae_max", "dense_mean", "paper_diagonal"),
                          ("sae_max", "dense_max", "same_pool_max"),
                          ("sae_mean", "dense_mean", "same_pool_mean"),
                          ("sae_max", "ecfp", "sae_vs_ecfp"),
                          ("dense_max", "ecfp", "dense_max_vs_ecfp"),
                          # the selection decomposition, mirroring MoleculeACE
                          (f"sae_top{dim}", "dense_max", f"top{dim}_vs_dense_max"),
                          (f"sae_rand{dim}", "dense_max", f"rand{dim}_vs_dense_max"),
                          (f"sae_top{dim}", f"sae_rand{dim}", f"top{dim}_vs_rand{dim}")):
            if a in agg and b in agg:
                d = agg[a][m]["mean"] - agg[b][m]["mean"]
                g[key] = float(d if hi_good else -d)   # positive = first is better
        return g

    for assay, e in results.get("cadme", {}).get("assays", {}).items():
        for sp, protos in e["splits"].items():
            for proto, sd in protos.items():
                endpoint = f"cADME/{assay}/{sp}/{proto}"
                out[endpoint] = gap(sd["agg"], "pearson_r", True)
                group_members.setdefault(
                    f"cADME/{sp}/{proto}/pearson_r", []).append(endpoint)
    for name, e in results.get("tdc", {}).items():
        if not e.get("agg"):
            continue
        m = "auroc" if e["task"] == "classification" else "pearson_r"
        endpoint = f"TDC/{name}"
        out[endpoint] = gap(e["agg"], m, True)
        group_members.setdefault(f"TDC/{e['task']}/{m}", []).append(endpoint)

    keys = ["paper_diagonal", "same_pool_max", "same_pool_mean", "sae_vs_ecfp"]
    lines.append(f"{'endpoint':<44}" + "".join(f"{k:>18}" for k in keys))
    lines.append("-" * (44 + 18 * len(keys)))
    for ep, g in out.items():
        lines.append(f"{ep:<44}" + "".join(
            f"{g[k]:>+18.4f}" if k in g else f"{'--':>18}" for k in keys))
    grouped = {}
    lines.append("-" * (44 + 18 * len(keys)))
    for group, endpoints in sorted(group_members.items()):
        means = {
            key: float(np.mean([out[endpoint][key] for endpoint in endpoints
                                if key in out[endpoint]]))
            for key in keys
            if any(key in out[endpoint] for endpoint in endpoints)
        }
        grouped[group] = {
            "n_endpoints": len(endpoints),
            "endpoints": endpoints,
            "mean_gaps": means,
        }
        label = f"MEAN {group} (n={len(endpoints)})"
        lines.append(f"{label:<44}" + "".join(
            f"{means[k]:>+18.4f}" if k in means else f"{'--':>18}"
            for k in keys))
    return out, grouped, "\n".join(lines)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", required=True, help="SAE directory")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json used to train the checkpoint; otherwise "
                         "MOLCIRCUITS_TOKENIZER or a checkpoint-local file")
    ap.add_argument("--model-config", default=None,
                    help="model_config.json for legacy checkpoints without model_cfg")
    ap.add_argument("--out", default="artifacts/tdc")
    ap.add_argument("--seed", type=int, default=0, help="model/head seed")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer", type=int, help="default: the dictionary training layer")
    ap.add_argument("--n-ctx", type=int, default=None,
                    help="maximum encoded length; default: checkpoint n_ctx")
    ap.add_argument("--batch", type=int, default=128, help="featurization batch")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                    help="split seeds")
    ap.add_argument("--split", default="scaffold", help="TDC split method")
    ap.add_argument("--cadme-splits", nargs="+", default=["scaffold", "random"])
    ap.add_argument("--cadme-csv",
                    default=os.path.join(paths.root(), "tdc_data", "cadme", "ADME_public_set_3521.csv"))
    ap.add_argument("--tdc-path", default=os.path.join(paths.root(), "tdc_data"))
    ap.add_argument("--tasks", nargs="+", choices=[name for _, name in TDC_TASKS],
                    default=None, help="subset of TDC tasks")
    ap.add_argument("--ecfp-bits", type=int, default=2048)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--protocols", nargs="+", default=["es_refit", "fixed"],
                    choices=["es_refit", "fixed"],
                    help="head protocols for cADME (es_refit = the original's)")
    ap.add_argument("--tdc-protocol", default="es_refit",
                    choices=["es_refit", "fixed"])
    ap.add_argument("--fixed-rounds", type=int, default=300)
    ap.add_argument("--max-rounds", type=int, default=600,
                    help="boosting-round cap, identical for all arms")
    ap.add_argument("--n-jobs", type=int, default=16,
                    help="XGBoost threads PER FIT; with --fit-workers > 1 the "
                         "product is what lands on the box, so >32 total "
                         "oversubscribes and is measurably slower here")
    ap.add_argument("--fit-workers", type=int, default=5,
                    help="representations fitted concurrently (max 5). Worth "
                         "2.55x on CUDA and does not affect results: each fit "
                         "is independent and individually seeded.")
    ap.add_argument("--xgb-device", default="cpu", choices=["cpu", "cuda"],
                    help="XGBoost tree-building device. 'cuda' is ~5x faster "
                         "and this stage is otherwise CPU-bound while the GPU "
                         "idles. NOT the default and NOT safe to switch "
                         "mid-experiment: GPU and CPU hist can break ties "
                         "differently, so a run mixing them is not "
                         "seed-comparable. Turn it on for every seed or none.")
    ap.add_argument("--skip-cadme", action="store_true")
    ap.add_argument("--skip-tdc", action="store_true")
    ap.add_argument("--merge-from", nargs="*", default=None,
                    help="tdc_results.json files from other phases to fold in")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    log(f"ckpt {args.ckpt}\nsae  {args.sae}\nlayer {args.layer}  "
        f"device {args.device}  split seeds {args.seeds}")
    model, tok, sae, mcfg, stack_metadata = load_stack(
        args.ckpt, args.sae, device=args.device,
        tokenizer_path=args.tokenizer, model_config_path=args.model_config,
        return_metadata=True)
    if args.layer is None:
        args.layer = sae.cfg.layer
    if args.n_ctx is None:
        args.n_ctx = int(mcfg.n_ctx)
    log(f"loaded: d_model={mcfg.d_model} n_layers={mcfg.n_layers} "
        f"d_sae={sae.cfg.d_sae} k={getattr(sae.cfg, 'k', None)}")

    t0 = time.time()
    input_files = {
        "checkpoint": args.ckpt,
        "sae": os.path.join(args.sae, "sae.pt"),
        "tokenizer": stack_metadata.get("tokenizer_source"),
        "model_config": stack_metadata.get("model_config_source"),
    }
    input_sha256 = {
        name: {"path": os.path.realpath(path), "sha256": file_sha256(path)}
        for name, path in input_files.items()
        if path and os.path.isfile(path)
    }
    results = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "analysis": "downstream_transfer_cadme_tdc",
        "status": "partial",
        "input_sha256": input_sha256,
        "protocol": {
            "model_inputs": stack_metadata,
            "long_sequence_policy": "drop the row from every representation",
        }
    }
    _PARTIAL.clear()
    # Phases can be run as separate processes (cADME and TDC are independent and
    # each saturates a different number of cores); --merge-from folds them back
    # into a single artifact before the summary and the manifest record.
    for src in args.merge_from or []:
        with open(src) as fh:
            prev = json.load(fh)
        results.setdefault("merged_artifacts", {})[os.path.realpath(src)] = {
            "sha256": file_sha256(src),
            "config": prev.get("config"),
        }
        for key in ("cadme", "tdc"):
            if key in prev and prev[key]:
                results.setdefault(key, {})
                if key == "tdc":
                    results[key].update(prev[key])
                else:
                    results[key] = prev[key]
        # keep each phase's own config -- the merging invocation's argv is not
        # the config the numbers were produced under
        log(f"merged {src}: {list(prev.keys())}")
    _checkpoint(args, results)
    if not args.skip_cadme:
        results["cadme"] = run_cadme(args, model, tok, sae)
    if not args.skip_tdc:
        results.setdefault("tdc", {}).update(run_tdc(args, model, tok, sae))

    gaps, grouped_means, gap_txt = pooling_verdict(results)
    results["pooling_analysis"] = {
        "note": ("positive = first arm better. paper_diagonal = sae_max vs "
                 "dense_mean (what the original compared); same_pool_* = the "
                 "like-for-like comparison. Means are only taken within one "
                 "benchmark/task type/metric; no global heterogeneous mean is "
                 "defined."),
        "per_endpoint": gaps, "grouped_means": grouped_means}
    subset_dim = min(mcfg.d_model, sae.cfg.d_sae)
    results["config"] = vars(args) | {"subset_width": subset_dim, "d_model": mcfg.d_model,
                                     "d_sae": sae.cfg.d_sae, "sae_k": sae.cfg.k}
    results["representations"] = PAPER_ARM
    results["elapsed_seconds"] = round(time.time() - t0, 1)
    results["status"] = "complete"

    tbl = summary_table(results)
    results["summary_table"] = tbl
    log("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78 + "\n" + tbl)
    log("\n" + "=" * 78 + "\nPOOLING vs REPRESENTATION\n" + "=" * 78 + "\n" + gap_txt)

    path = dump_json(results, os.path.join(args.out, "tdc_results.json"))
    log(f"\nwrote {path}  ({time.time()-t0:.0f}s total)")

    n_ep = len(results.get("cadme", {}).get("assays", {})) + sum(
        1 for e in results.get("tdc", {}).values() if e.get("agg"))
    Manifest().record("tdc_admet",
                      {"n_endpoints": n_ep,
                       "representations": representation_names(subset_dim),
                       "grouped_mean_gaps": grouped_means},
                      config=vars(args), files=[path])


if __name__ == "__main__":
    sys.exit(main())
