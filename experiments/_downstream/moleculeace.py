#!/usr/bin/env python3
"""MoleculeACE transfer benchmark: 30 endpoints, one shared XGBoost head.

    python scripts/08_moleculeace.py --ckpt artifacts/transformer/checkpoint.pt \
        --sae artifacts/saes/layer3_x16 --out artifacts/moleculeace

Audits the historical downstream comparison. The original implementation
(``lmkit/experiments/downstream_eval.py``, lines 116-122) MEAN-pooled the dense
embedding and MAX-pooled the SAE code, so that dense-vs-SAE gap confounded
representation with pooling operator. This script runs the full 2x2,
``{dense, sae} x {mean, max}``, plus ECFP and a predict-the-train-mean floor.
The primary comparison is now the like-for-like ``dense_max`` vs ``sae_max``
contrast; ``dense_mean`` vs ``sae_mean`` is its pooling sensitivity analysis.

Splits and cliff labels are the ones shipped with the benchmark, never re-derived.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from molcircuits import paths
from molcircuits.downstream import moleculeace as ma
from molcircuits.downstream.ace_cache import MoleculeACECache
from molcircuits.downstream.features import ecfp, embed, load_stack
from molcircuits.utils.manifest import Manifest, dump_json

REPRESENTATIONS = ("dense_mean", "dense_max", "sae_mean", "sae_max", "ecfp")
SHORT = {"dense_mean": "d-mean", "dense_max": "d-max", "sae_mean": "s-mean",
         "sae_max": "s-max", "ecfp": "ecfp"}

#: The first two are the like-for-like dense-vs-SAE comparisons.  The third is
#: retained only as the historical mismatched-pooling diagnostic.
CONTRASTS = (("dense_max", "sae_max"), ("dense_mean", "sae_mean"),
             ("dense_mean", "sae_max"), ("dense_mean", "dense_max"),
             ("sae_mean", "sae_max"), ("sae_max", "ecfp"),
             ("dense_mean", "ecfp"))


def featurize(ep: ma.Endpoint, model, tok, sae, layer: int, batch: int,
              device: str, n_ctx: int, fingerprints=None):
    """All five representations for one endpoint, from the shared featurizer."""
    dense, sparse, drop_max = embed(
        ep.smiles, model, tok, sae, layer=layer, pooling="both", n_ctx=n_ctx,
        device=device, batch=batch, return_metadata=True)
    bad = drop_max["dropped_indices"]
    X = {"dense_max": dense["max"], "dense_mean": dense["mean"],
         "sae_max": sparse["max"], "sae_mean": sparse["mean"],
         "ecfp": (fingerprints or ecfp)(ep.smiles).astype(np.float32)}
    if bad:
        # Drop from every representation and from y together, or the rows stop
        # meaning the same molecule.
        keep = np.ones(len(ep.smiles), bool)
        keep[np.asarray(bad, int)] = False
        X = {k: v[keep] for k, v in X.items()}
        ep = ep.drop(bad)
    return X, ep, drop_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sae", required=True, help="SAE directory")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json used to train the checkpoint; otherwise "
                         "MOLCIRCUITS_TOKENIZER or a checkpoint-local file")
    ap.add_argument("--model-config", default=None,
                    help="model_config.json for legacy checkpoints without model_cfg")
    ap.add_argument("--data", default=os.path.join(paths.data(), "moleculeace"),
                    help="directory of MoleculeACE benchmark CSVs")
    ap.add_argument("--out", default="artifacts/moleculeace")
    ap.add_argument("--fit-cache", default=os.environ.get("MOLCIRCUITS_ACE_FIT_CACHE"),
                    help="shared directory for exact XGBoost predictions and Morgan fingerprints")
    ap.add_argument("--layer", type=int, default=None,
                    help="residual-stream layer; default is the SAE's own layer")
    ap.add_argument("--endpoints", nargs="+", default=None,
                    help="subset of endpoint names (default: all 30)")
    ap.add_argument("--n-ctx", type=int, default=None,
                    help="maximum encoded length; default: checkpoint n_ctx")
    ap.add_argument("--batch", type=int, default=128,
                    help="featurization batch; keep small, the GPU is shared")
    ap.add_argument("--n-jobs", type=int, default=8,
                    help="XGBoost threads per fit (CPU; the GPU is off limits here)")
    ap.add_argument("--fit-workers", type=int, default=5,
                    help="representations fitted concurrently per endpoint")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--early-stop", type=int, default=50)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--xgb-device", default="cpu", choices=["cpu", "cuda"],
                    help="XGBoost tree-building backend. 'cuda' is several "
                         "times faster, but GPU and CPU hist are not "
                         "bit-identical, so every run being compared to another "
                         "must use the same value. Defaults to cpu so that "
                         "adding the flag cannot silently change a comparison; "
                         "the value used is recorded in the results JSON.")
    args = ap.parse_args()
    cache = MoleculeACECache(args.fit_cache, args.xgb_device)

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "moleculeace_results.json")

    model, tok, sae, mcfg, stack_metadata = load_stack(
        args.ckpt, args.sae, device=args.device,
        tokenizer_path=args.tokenizer, model_config_path=args.model_config,
        return_metadata=True)
    if args.n_ctx is None:
        args.n_ctx = int(mcfg.n_ctx)
    layer = args.layer if args.layer is not None else sae.cfg.layer
    endpoints = tuple(args.endpoints) if args.endpoints else ma.ENDPOINTS
    eps = ma.load_endpoints(args.data, endpoints, download_missing=not args.no_download)
    print(f"{len(eps)} endpoints, layer {layer}, d_sae {sae.cfg.d_sae}, "
          f"{sum(len(e.smiles) for e in eps)} molecules")

    from joblib import Parallel, delayed  # noqa: PLC0415

    per_endpoint: dict[str, dict] = {}
    dropped: dict[str, dict] = {}
    cache_hits = 0
    t_start = time.time()

    for i, ep in enumerate(eps):
        t0 = time.time()
        X, ep, drop_info = featurize(ep, model, tok, sae, layer, args.batch,
                                     args.device, args.n_ctx, cache.fingerprints)
        if drop_info["n_dropped"]:
            dropped[ep.name] = drop_info

        # Parallel over representations on CUDA as well as CPU. An earlier
        # version serialised the CUDA path on the theory that N workers would
        # contend for one card; measured, that is wrong. Five concurrent fits
        # over these five arms run 2.55x faster than serialising them (40.0s vs
        # 102.1s, medians of 2, and that was on a card already at 100%
        # utilisation from other jobs). Per-fit GPU speedup and outer
        # parallelism compose here rather than cancelling; serialising threw
        # away exactly the factor that made the CUDA path look no better than
        # CPU. Worker count is not a result-affecting knob -- each fit is
        # independent and seeded -- so it can be tuned freely.
        workers = min(args.fit_workers, len(REPRESENTATIONS))
        fits = Parallel(n_jobs=workers)(
            delayed(cache.fit)(X[r], ep, args.seed, args.n_jobs,
                               args.val_frac, args.early_stop, args.xgb_device)
            for r in REPRESENTATIONS)

        row = {}
        for rep, (pred, n_best, hit) in zip(REPRESENTATIONS, fits):
            row[rep] = ma.score(pred, ep) | {"n_trees": n_best}
            cache_hits += int(hit)
        # A predict-the-train-mean floor: without it a dense RMSE near 1.05 is
        # uninterpretable, since it is roughly what "no signal at all" scores.
        floor = np.full(int(ep.test.sum()), ep.y[ep.train].mean())
        row["train_mean_baseline"] = ma.score(floor, ep)
        row["n_train"] = int(ep.train.sum())
        per_endpoint[ep.name] = row

        print(f"[{i+1:2d}/{len(eps)}] {ep.name:<17} n={len(ep.smiles):5d} "
              + "  ".join(f"{SHORT[r]}={row[r]['rmse']:.3f}" for r in REPRESENTATIONS)
              + f"  floor={row['train_mean_baseline']['rmse']:.3f}"
              + f"  [{time.time()-t0:.0f}s; {sum(hit for _, _, hit in fits)} reused fits]")

        # written every endpoint: a crash 25 endpoints in should not cost 25
        # endpoints of compute
        dump_json({"status": "partial", "per_endpoint": per_endpoint},
                  results_path)

    agg = ma.aggregate(per_endpoint)
    out = {
        "status": "complete",
        "config": vars(args) | {"layer": layer, "d_sae": int(sae.cfg.d_sae),
                                "d_model": int(mcfg.d_model), "sae_k": int(sae.cfg.k)},
        "protocol": {
            "model_inputs": stack_metadata,
            "long_sequence_policy": "drop the row from every representation",
            "splits": "shipped MoleculeACE train/test split, used verbatim",
            "cliff_labels": "shipped cliff_mol column, used verbatim",
            "target": ma.TARGET_COLUMN,
            "head": "XGBoost, hyperparameters shared across all representations",
            # Device recorded beside the params, not only in config: the
            # earliest TDC runs predate the flag and their config reads
            # xgb_device=null, which is indistinguishable from "not recorded"
            # unless the protocol block states it.
            "xgb_params": ma.XGB_PARAMS | {"device": args.xgb_device},
            "validation": (f"{args.val_frac:.0%} of the train split, seeded, for "
                           f"early stopping ({args.early_stop} rounds); refit on "
                           "the full train split with the selected tree count"),
            "note_pooling": (
                "dense_max vs sae_max is the primary like-for-like comparison. "
                "dense_mean vs sae_mean is the matched mean-pooling sensitivity "
                "analysis; dense_mean vs sae_max is retained only as the "
                "historical mismatched-pooling diagnostic."),
        },
        "molecule_drop_metadata": dropped,
        "reused_xgb_fits": cache_hits,
        "per_endpoint": per_endpoint,
        "aggregate": agg,
    }

    diag = ma.paired_contrasts(per_endpoint, CONTRASTS, seed=args.seed)
    out["contrasts"] = diag

    dump_json(out, results_path)
    Manifest().record("moleculeace", {r: agg[r]["mean_rmse"] for r in agg},
                      config=vars(args), files=[results_path])

    print(f"\n=== {len(per_endpoint)} endpoints, {time.time()-t_start:.0f}s ===")
    print(f"{'representation':<22}{'RMSE':>8}{'cliff':>8}{'noncliff':>10}"
          f"{'ratio':>8}{'pooled':>9}")
    for rep in sorted(agg, key=lambda r: agg[r]["mean_rmse"]):
        a = agg[rep]
        print(f"{rep:<22}{a['mean_rmse']:>8.3f}{a['mean_rmse_cliff']:>8.3f}"
              f"{a['mean_rmse_noncliff']:>10.3f}{a['mean_rmse_ratio']:>8.3f}"
              f"{a['pooled_rmse']:>9.3f}")
    print("\npaired dense->sae gap (positive = dense worse), 95% CI over endpoints:")
    for a, b in CONTRASTS[:3]:
        c = diag[f"{a}_minus_{b}"]
        print(f"  {a:<11} - {b:<9} {c['mean_rmse_diff']:+.3f} "
              f"[{c['rmse_ci95'][0]:+.3f}, {c['rmse_ci95'][1]:+.3f}]  "
              f"cliff {c['mean_rmse_cliff_diff']:+.3f}  "
              f"dense worse on {c['n_rmse_first_worse']}/{c['n_endpoints']}"
              + ("   <- primary like-for-like" if (a, b) == ("dense_max", "sae_max")
                 else "   <- matched mean-pooling sensitivity" if
                 (a, b) == ("dense_mean", "sae_mean")
                 else "   <- historical mismatched-pooling diagnostic"))


if __name__ == "__main__":
    main()
