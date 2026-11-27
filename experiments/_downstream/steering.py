#!/usr/bin/env python3
"""SAE feature steering with an alpha=0 SAE-in-the-loop baseline and a
random-feature control.

    python scripts/07_steering.py --ckpt artifacts/transformer/checkpoint.pt \
        --sae artifacts/saes/layer3_x16 --data data --out artifacts/steering

Targets default to a random sample of *training-split* molecules. Steering toward
molecules the corpus filter excluded (below 150 Da, say) tests nothing, since the
model was never shown anything like them.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from molcircuits.analysis.dictionary_geometry import feature_firing_rates, nonpad_residuals_by_layer
from molcircuits.analysis.steering import run_sweep
from molcircuits.analysis.valence_intervene import residual_rms
from molcircuits.chem.tokenizer import SmilesTokenizer
from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.sae.sae import SparseAutoencoder
from molcircuits.utils.manifest import Manifest, dump_json, file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/steering")
    ap.add_argument("--targets", default=None, help="file of target SMILES")
    ap.add_argument("--n-targets", type=int, default=50)
    # Multiples of the residual-stream RMS, as in the valence intervention. A
    # residual VECTOR has norm rms*sqrt(512) ~ 22.6*rms, so alpha=16 is still
    # only ~0.7 of the residual's own length.
    # 16 is not here: at that strength the steered arm returns ZERO distinct
    # output sets over all 50 targets in seed 0, so every target generates the
    # same thing and there is nothing left to compare between arms. It costs a
    # fifth of the stage's runtime to produce a column the figure then drops.
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--top-k", type=int, default=5)
    # 256, not 128: under a perturbation this size the model stops emitting EOS,
    # and a truncated SMILES has open rings and branches, so a 128 cap scores
    # "did not finish" as "invalid chemistry".
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rms-molecules", type=int, default=256,
                    help="corpus molecules used to measure the residual RMS")
    ap.add_argument("--dictionary-stats", default=None,
                    help="dictionary_stats.json beside the SAE directory; "
                         "supplies per-feature firing rates for the "
                         "rate-matched control arm. If omitted, use the file beside the "
                         "dictionary or measure the rates on 1,500 validation molecules.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg_path = args.model_config or os.path.join(
        os.path.dirname(args.ckpt), "model_config.json")
    mcfg = ModelConfig.load(cfg_path)
    model = MolecularTransformer(mcfg).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device)["model"])
    model.eval()
    sae = SparseAutoencoder.load(args.sae, args.device).to(args.device).eval()
    tok = SmilesTokenizer.load(os.path.join(args.data, "tokenizer.json"))

    if args.targets:
        targets = open(args.targets).read().split()
    else:
        pool = open(os.path.join(args.data, "train.smi")).read().split()
        rng = np.random.default_rng(args.seed)
        targets = [pool[i] for i in rng.choice(len(pool),
                                               min(args.n_targets, len(pool)), False)]

    # The alpha unit. Measured on real token positions of corpus molecules at
    # the analyzed layer, with pads excluded -- right padding otherwise puts most
    # of the weight on positions the model never saw and understates the unit.
    rms_pool = open(os.path.join(args.data, "valid.smi")).read().split()
    rng_rms = np.random.default_rng(args.seed)
    rms_smiles = [rms_pool[i] for i in rng_rms.choice(
        len(rms_pool), min(args.rms_molecules, len(rms_pool)), False)]
    n_ctx = min(128, mcfg.n_ctx)
    ids = [tok.encode(s)[:n_ctx] for s in rms_smiles]
    ids = [e + [tok.pad_id] * (n_ctx - len(e)) for e in ids]
    rms = residual_rms(model, torch.tensor(ids), sae.cfg.layer, args.device,
                       pad_id=tok.pad_id)
    print(f"residual RMS at layer {sae.cfg.layer}: {rms:.3f} "
          f"(residual vector norm ~ {rms * mcfg.d_model ** 0.5:.0f}); "
          f"alpha=1 injects a perturbation of norm {rms:.1f}")

    # Firing rates for the rate-matched control. The uniform control matches
    # count and magnitude; without this it does not match how often the injected
    # coordinates fire in ordinary contexts, which is the largest measured
    # difference between the two feature populations.
    os.makedirs(args.out, exist_ok=True)
    rate_path = args.dictionary_stats or os.path.join(os.path.dirname(os.path.normpath(args.sae)),
                                                     "dictionary_stats.json")
    key = f"layer{sae.cfg.layer}_x{sae.cfg.expansion}"
    if args.dictionary_stats or os.path.isfile(rate_path):
        with open(rate_path) as fh:
            stats = json.load(fh)
        firing_rates = stats["dictionaries"][key]["firing_rate"]
    else:
        sample = rms_pool[:1500]
        acts = nonpad_residuals_by_layer(model, tok, sample, [sae.cfg.layer],
                                        n_ctx, args.device)[sae.cfg.layer]
        firing_rates = feature_firing_rates(sae, acts, args.device).tolist()
        rate_path = dump_json({"n_tokens": len(acts), "n_molecules": len(sample),
                               "dictionaries": {key: {"firing_rate": firing_rates}}},
                              os.path.join(args.out, "steering_firing_rates.json"))
    if len(firing_rates) != sae.cfg.d_sae:
        raise ValueError("firing-rate vector width differs from the SAE dictionary")
    print(f"rate-matched control from {key}: median firing rate "
          f"{float(np.median(firing_rates)):.2e}")

    rows = []
    for i, t in enumerate(targets):
        # control_seed varies per target so the random-feature arm is a real
        # distribution over feature choices rather than one fixed draw reused
        # 50 times; the sampling seed stays shared so the arms remain paired.
        r = run_sweep(model, sae, tok, t, sae.cfg.layer, tuple(args.alphas),
                      args.n_samples, args.top_k, args.seed, args.device,
                      args.max_tokens, control_seed=args.seed + 1000 + i,
                      rms=rms, firing_rates=firing_rates)
        rows.append(r)
        summary = r["summary_vs_alpha0_baseline"]
        if not summary:
            # No arm produced a single parseable molecule, so no Tanimoto exists
            # to compare. Report it rather than dying on max() of an empty
            # sequence; the aggregate below already skips these targets.
            print(f"[{i+1}/{len(targets)}] {t[:40]:<40} no valid generations")
            continue
        best = max(summary.items(),
                   key=lambda kv: kv[1]["delta_target_tanimoto_vs_alpha0"])
        print(f"[{i+1}/{len(targets)}] {t[:40]:<40} best {best[0]}: "
              f"dSim={best[1]['delta_target_tanimoto_vs_alpha0']:+.4f} "
              f"validity={best[1]['validity']:.3f} "
              f"term={best[1].get('termination_rate', float('nan')):.3f}")

    # aggregate: steered vs random-feature, both against the alpha=0 baseline
    agg = {}
    for a in args.alphas:
        if a == 0.0:
            continue
        for arm in (f"steered_alpha={a}", f"random_feature_alpha={a}",
                    f"rate_matched_alpha={a}"):
            ds = [r["summary_vs_alpha0_baseline"][arm]["delta_target_tanimoto_vs_alpha0"]
                  for r in rows if arm in r["summary_vs_alpha0_baseline"]]
            vs = [r["summary_vs_alpha0_baseline"][arm]["validity"]
                  for r in rows if arm in r["summary_vs_alpha0_baseline"]]
            if ds:
                # CI on the MEAN, not the spread across targets. The percentile
                # form was ~5.7x too wide and made disjoint arms look overlapping.
                m, nn = float(np.mean(ds)), len(ds)
                sd = float(np.std(ds, ddof=1)) if nn > 1 else None
                half = 1.96 * sd / nn ** 0.5 if sd is not None else None
                agg[arm] = {
                    "delta_target_tanimoto_mean": m,
                    "delta_target_tanimoto_std": sd,
                    "delta_target_tanimoto_ci95_mean": [m - half, m + half] if half is not None else None,
                    "delta_target_tanimoto_spread_2p5_97p5": [
                        float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                    "validity_mean": float(np.mean(vs)),
                    "n_targets": len(ds),
                }
    # PAIRED aggregate. The marginal means above are computed over whichever
    # targets left the arm at least one parseable molecule, and that is not the
    # same set of targets for the two arms: a target whose random-feature arm
    # generated nothing valid has no Tanimoto and silently drops out. Those are
    # exactly the targets where the random direction was most destructive, so
    # dropping them makes the control look more benign than it is -- measured
    # here, the random arm survives on 11-41 of 50 targets while the steered arm
    # always survives on 50. Only the paired difference over the intersection is
    # a comparison between arms rather than between populations.
    paired = {}
    for a in args.alphas:
        if a == 0.0:
            continue
        ks = f"steered_alpha={a}"
        # Both controls are paired against the steered arm. The uniform draw
        # matches count and magnitude; the rate-matched draw additionally
        # matches how often the injected coordinates fire, which is the larger
        # difference between the two feature populations. Aggregating only the
        # uniform arm -- as this loop did -- ran the stronger control and then
        # reported the weaker one.
        for kr, tag in ((f"random_feature_alpha={a}", ""),
                        (f"rate_matched_alpha={a}", "_rate_matched")):
            # The same alpha=0 score cancels in a paired difference. Direct
            # arm means also retain targets whose common baseline failed.
            both = [r["arms"] for r in rows if all("target_tanimoto_mean" in r["arms"][k]
                                                    for k in (ks, kr))]
            diff = np.array([arms[ks]["target_tanimoto_mean"] - arms[kr]["target_tanimoto_mean"]
                             for arms in both])
            nn = len(diff)
            mean = float(diff.mean()) if nn else None
            half = (1.96 * float(np.std(diff, ddof=1)) / nn ** 0.5
                    if nn > 1 else None)
            paired[f"alpha={a}{tag}"] = {
                "control_arm": kr,
                "n_paired": nn,
                "n_targets_dropped_steered": sum(
                    1 for r in rows if "target_tanimoto_mean" not in r["arms"][ks]),
                "n_targets_dropped_control": sum(
                    1 for r in rows if "target_tanimoto_mean" not in r["arms"][kr]),
                "mean_steered_minus_control": mean,
                "ci95": [mean - half, mean + half] if half is not None else None,
                # validity over ALL targets, including those with zero valid
                # generations, which is the only unbiased way to report it
                "validity_steered_all_targets": float(
                    np.mean([r["arms"][ks]["validity"] for r in rows])),
                "validity_control_all_targets": float(
                    np.mean([r["arms"][kr]["validity"] for r in rows])),
                # Validity is only interpretable next to this: a run that never
                # emits EOS is truncated, not necessarily chemically wrong.
                "termination_steered_all_targets": float(
                    np.mean([r["arms"][ks]["termination_rate"] for r in rows])),
                "termination_control_all_targets": float(
                    np.mean([r["arms"][kr]["termination_rate"] for r in rows])),
            }

    n_unusable = sum(1 for r in rows if not r["summary_vs_alpha0_baseline"])
    out = {"per_target": rows, "aggregate": agg, "paired": paired,
           "input_sha256": {"checkpoint": file_sha256(args.ckpt),
                            "sae": file_sha256(os.path.join(args.sae, "sae.pt")),
                            "tokenizer": file_sha256(os.path.join(args.data, "tokenizer.json"))},
           "residual_rms": float(rms), "alpha_units": "residual-stream RMS",
           "d_model": mcfg.d_model, "d_sae": sae.cfg.d_sae,
           "dictionary_stats_source": os.path.realpath(rate_path),
           "relative_norm_per_unit_alpha": mcfg.d_model ** -0.5,
           "max_tokens": int(args.max_tokens),
           "n_targets": len(rows), "n_targets_no_valid_generations": n_unusable,
           "note": ("compare steered_alpha=X against a CONTROL arm, not against "
                    "unsteered generation; all arms carry identical SAE "
                    "reconstruction error and share a sampling seed. Two "
                    "controls are reported per strength: 'alpha=X' pairs against "
                    "a uniformly drawn feature set matched on count and "
                    "magnitude, 'alpha=X_rate_matched' against one additionally "
                    "matched on firing rate. The rate-matched arm is the "
                    "stronger control and is what the paper quotes. Read "
                    "'paired', not 'aggregate': the marginal means are taken over "
                    "different target sets per arm.")}
    result_path = dump_json(out, os.path.join(args.out, "steering_results.json"))
    Manifest().record("steering", {"aggregate": agg, "paired": paired},
                      config=vars(args), files=[result_path])
    for k, v in paired.items():
        # These four keys are `_control`, not `_random`: the paired block reports
        # whichever control arm it was built against (uniform or rate-matched),
        # and the names were changed when the rate-matched arm was added without
        # this print being updated. It raised KeyError after the whole stage had
        # run -- the artifact is written above, so only the summary line was lost.
        estimate = v["mean_steered_minus_control"]
        displayed_estimate = f"{estimate:+.4f}" if estimate is not None else "unavailable"
        interval = [round(x, 5) for x in v["ci95"]] if v["ci95"] is not None else "unavailable"
        print(f"{k}: steered-minus-control={displayed_estimate} ci95={interval} "
              f"(n={v['n_paired']}, control arm {v['control_arm']} lost "
              f"{v['n_targets_dropped_control']} targets) "
              f"validity {v['validity_steered_all_targets']:.3f} vs "
              f"{v['validity_control_all_targets']:.3f} | "
              f"terminated {v['termination_steered_all_targets']:.3f} vs "
              f"{v['termination_control_all_targets']:.3f}")


if __name__ == "__main__":
    main()
