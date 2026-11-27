#!/usr/bin/env python3
"""Pointer-mass and head-ablation analysis for ring and branch events.

    python scripts/04_pointers.py --ckpt artifacts/transformer/checkpoint.pt \
        --data data --out artifacts/pointers

Sweeps *all* heads under both zero and mean ablation, and reports the event
effect, the matched-control effect, their difference, and the change in overall
cross-entropy. A head with a large control term or a large delta-CE is globally
important rather than event-specific; keeping the columns separate is what makes
that visible.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from molcircuits.analysis.pointers import (build_event_index, head_effects,
                                       mean_head_activations, pointer_mass,
                                       span_curve, summarize)
from molcircuits.chem.tokenizer import SmilesTokenizer
from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.utils.manifest import Manifest, dump_json


# Ranked on relative_specificity, not the raw difference: baseline margins differ
# ~2.3x between event and control populations, so the raw index ranks global
# damage (Spearman 0.76 with delta_ce, 0.02 with pointer_mass).
_AGG_KEYS = ("delta_margin_event", "delta_margin_control",
             "delta_margin_specificity", "relative_drop_event",
             "relative_drop_control", "relative_specificity",
             "baseline_margin_event", "baseline_margin_control", "delta_ce")
_RANK_KEY = "relative_specificity"


def load_batch(smiles, tok, n_ctx, device):
    ids, tgts = [], []
    for s in smiles:
        e = tok.encode(s)[:n_ctx + 1]
        e = e + [tok.pad_id] * (n_ctx + 1 - len(e))
        ids.append(e[:n_ctx]); tgts.append(e[1:n_ctx + 1])
    return (torch.tensor(ids, device=device), torch.tensor(tgts, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--out", default="artifacts/pointers")
    ap.add_argument("--n-molecules", type=int, default=4096)
    ap.add_argument("--max-events", type=int, default=4000)
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg_path = args.model_config or os.path.join(
        os.path.dirname(args.ckpt), "model_config.json")
    mcfg = ModelConfig.load(cfg_path)
    model = MolecularTransformer(mcfg).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device)["model"])
    model.eval()
    tok = SmilesTokenizer.load(os.path.join(args.data, "tokenizer.json"))

    smiles = open(os.path.join(args.data, f"{args.split}.smi")).read().split()
    rng = np.random.default_rng(args.seed)
    if len(smiles) > args.n_molecules:
        smiles = [smiles[i] for i in rng.choice(len(smiles), args.n_molecules, False)]

    os.makedirs(args.out, exist_ok=True)
    results = {}
    for kind in ("ring", "branch"):
        ev_pairs, ev_meta, ctrl_pairs = build_event_index(
            smiles, tok, args.n_ctx, kind, args.max_events, args.seed)
        if not ev_pairs:
            print(f"no {kind} events found"); continue

        used = sorted({b for b, _ in ev_pairs} | {b for b, _ in ctrl_pairs})
        remap = {b: i for i, b in enumerate(used)}
        sub = [smiles[b] for b in used]
        ev_pairs = [(remap[b], t) for b, t in ev_pairs]
        ctrl_pairs = [(remap[b], t) for b, t in ctrl_pairs]

        pm_rows, pm_weights, ab_rows = [], [], []
        for s in range(0, len(sub), args.batch):
            chunk = sub[s:s + args.batch]
            lo, hi = s, s + len(chunk)
            ep = [(b - lo, t) for b, t in ev_pairs if lo <= b < hi]
            cp = [(b - lo, t) for b, t in ctrl_pairs if lo <= b < hi]
            em = [e for (b, _), e in zip(ev_pairs, ev_meta) if lo <= b < hi]
            if not ep:
                continue
            x, y = load_batch(chunk, tok, args.n_ctx, args.device)
            pm_rows.append(pointer_mass(model, x, em, ep, args.device))
            pm_weights.append(len(ep))          # events behind this batch's row
            # Mean ablation needs the value it substitutes; compute it on this
            # same batch, with pad positions excluded.
            mz = mean_head_activations(model, x, args.device, pad_id=tok.pad_id)
            for mode in ("zero", "mean"):
                for r in head_effects(model, x, y, ep, cp, args.device, mode=mode,
                                      mean_z=mz if mode == "mean" else None,
                                      ignore_index=tok.pad_id):
                    r["kind"] = kind
                    ab_rows.append(r)

        # POOL THE MARGINS, THEN FORM THE RATIOS ONCE.
        #
        # This used to sum each batch's row and divide by the NUMBER OF
        # BATCHES, which is wrong twice: batches contributed equally regardless
        # of how many events they carried, and relative_drop_* /
        # relative_specificity are per-batch ratios, so it averaged ratios
        # instead of computing the ratio of the pooled means -- which is what
        # eq. (2) in the results section defines. At the shipped batching the
        # error is <=0.001, but at other legal settings (--n-molecules 1024
        # --batch 1000) a top head read +0.283 against a true +0.232 and 40 of
        # 48 heads changed rank. pointer_mass is likewise weighted by events.
        pm = np.average(np.asarray(pm_rows), axis=0,
                        weights=np.asarray(pm_weights, dtype=float))

        agg: dict[tuple[int, int, str], dict] = {}
        for r in ab_rows:
            key = (r["layer"], r["head"], r["mode"])
            a = agg.setdefault(key, {"ev_base": 0.0, "ev_abl": 0.0, "n_ev": 0,
                                     "ct_base": 0.0, "ct_abl": 0.0, "n_ct": 0,
                                     "ce": 0.0, "n_tok": 0})
            ne, nc, nt = r["n_events"], r["n_controls"], r["n_scored_tokens"]
            a["ev_base"] += r["baseline_margin_event"] * ne
            a["ev_abl"] += r["ablated_margin_event"] * ne
            a["n_ev"] += ne
            a["ct_base"] += r["baseline_margin_control"] * nc
            a["ct_abl"] += r["ablated_margin_control"] * nc
            a["n_ct"] += nc
            a["ce"] += r["delta_ce"] * nt
            a["n_tok"] += nt
        rows = []
        for (l, h, mode), a in agg.items():
            be = a["ev_base"] / max(a["n_ev"], 1)
            ae = a["ev_abl"] / max(a["n_ev"], 1)
            bc = a["ct_base"] / max(a["n_ct"], 1)
            ac = a["ct_abl"] / max(a["n_ct"], 1)
            rel_ev = (be - ae) / be if be else float("nan")
            rel_ct = (bc - ac) / bc if bc else float("nan")
            rows.append({
                "layer": l, "head": h, "mode": mode,
                "pointer_mass": float(pm[l, h]),
                "baseline_margin_event": be, "baseline_margin_control": bc,
                "delta_margin_event": be - ae,
                "delta_margin_control": bc - ac,
                "delta_margin_specificity": (be - ae) - (bc - ac),
                "relative_drop_event": rel_ev,
                "relative_drop_control": rel_ct,
                "relative_specificity": rel_ev - rel_ct,
                "delta_ce": a["ce"] / max(a["n_tok"], 1),
                "n_events": a["n_ev"], "n_controls": a["n_ct"],
            })

        for mode in ("zero", "mean"):
            summarize([r for r in rows if r["mode"] == mode], key=_RANK_KEY)

        spans = span_curve([(e, 1.0) for e in ev_meta])
        results[kind] = {
            "n_events": len(ev_pairs), "n_controls": len(ctrl_pairs),
            "n_molecules": len(sub),
            "pointer_mass_grid": pm.tolist(),
            "heads": sorted(rows, key=lambda r: -r[_RANK_KEY]),
            "ranked_by": _RANK_KEY,
            "event_span_histogram": spans,
            "note": ("delta_margin_specificity = event - control; a head with a "
                     "large control term or large delta_ce is globally important "
                     "rather than event-specific"),
        }
        top = results[kind]["heads"][0]
        print(f"{kind}: {len(ev_pairs)} events, top head by {_RANK_KEY}: "
              f"L{top['layer']}H{top['head']} rel_spec={top[_RANK_KEY]:+.4f} "
              f"ptr={top['pointer_mass']:.3f} dCE={top['delta_ce']:+.3f}")

    result_path = dump_json(results, os.path.join(args.out, "pointer_results.json"))
    Manifest().record("pointers", {k: v["n_events"] for k, v in results.items()},
                      config=vars(args), files=[result_path])


if __name__ == "__main__":
    main()
