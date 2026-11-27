#!/usr/bin/env python3
"""Stage 22: is remaining valence capacity computed and used, or only present?

Section 4 reports that capacity is decodable at 0.998 and that moving along the
direction which decodes it shifts bond-order probability by about 5e-4, changing
the argmax no more often than a matched random direction. That is a null about
ONE linear intervention at ONE population, and the section says as much. Patching
asks the question without needing a direction at all: if capacity is computed,
some site carries it, and splicing that site across a minimal pair will move the
prediction that depends on it.

MINIMAL PAIR. Substitute a single token so that the *prefix-correct* remaining
capacity of the governing atom differs at a later bond decision, while every
token before the substitution is identical:

    clean      CC(=O)C ...   at the decision after the branch, the carbon has
    corrupt    CC(CO)C ...   used 3 bonds in one arm and 2 in the other

Pairs are searched for rather than hand-written: `valence.bond_decision_labels`
computes the running ledger exactly, so a candidate substitution is accepted only
when it changes capacity at a decision at least two positions downstream. Nothing
about the pair depends on a probe, a threshold or a fitted direction.

METRIC. logit(`=`) - logit(`C`) at that decision. Capacity >= 2 admits a double
bond; capacity <= 1 forbids it. So the contrast is a legality difference the
grammar fixes, not a preference the model happens to have, and it is a difference
of logits so a common-mode shift cancels rather than registering as an effect.

Recovery is (patched - corrupt) / (clean - corrupt), as in stage 21.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import bond_decision_labels
from molcircuits.model.hooks import HookManager, resid_post
from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.utils.manifest import Manifest

#: Single-character substitutions that can change a neighbour's bond ledger
#: without changing token count. Bond-order tokens change how much the governing
#: atom has spent; element tokens change what it can spend.
SUBSTITUTIONS = {"=": ["C"], "#": ["C"], "C": ["=", "O", "N"],
                 "O": ["C"], "N": ["C"]}

#: Which of the two routes to a capacity change a pair took. Every substitution
#: perturbs more than capacity -- an element swap changes what the governing atom
#: can spend, a bond-order swap changes what it has already spent and also the
#: atom count -- so neither family alone identifies capacity as the transported
#: quantity. Stratifying by family is what separates the two: if the layerwise
#: recovery profile is the same on both, the shared factor is the capacity
#: transition rather than either token edit.
SUB_FAMILY = {("=", "C"): "bond_order", ("#", "C"): "bond_order",
              ("C", "="): "bond_order",
              ("C", "O"): "element", ("C", "N"): "element",
              ("O", "C"): "element", ("N", "C"): "element"}


def capacity_at(tokens):
    """position -> remaining capacity, from the running prefix ledger."""
    return dict(bond_decision_labels(tokens))


def build_pairs(smiles, tok, n_ctx, max_pairs, min_gap=2, rng=None):
    """Minimal pairs whose capacity differs at a downstream bond decision.

    Accepted only when the substituted token is at least ``min_gap`` positions
    before the decision, so the information has to travel; a substitution sitting
    on the decision itself would be recovered by patching that one position at
    layer 0 and would say nothing about transfer.
    """
    out = []
    for s in smiles:
        try:
            toks = tokenize(s)
        except Exception:
            continue
        if len(toks) + 2 > n_ctx:
            continue
        base = capacity_at(toks)
        if not base:
            continue
        for i, t in enumerate(toks):
            for repl in SUBSTITUTIONS.get(t, []):
                alt = list(toks)
                alt[i] = repl
                try:
                    acap = capacity_at(alt)
                except Exception:
                    continue
                # a decision far enough downstream whose capacity flips across
                # the legality boundary for a double bond
                for d, cap in base.items():
                    if d not in acap or d < i + min_gap:
                        continue
                    if (cap >= 2) == (acap[d] >= 2):
                        continue
                    ce = tok.encode("".join(toks))
                    co = tok.encode("".join(alt))
                    if len(ce) != len(co) or len(ce) > n_ctx:
                        continue
                    predict = d + 1          # +1 for BOS
                    if [k for k in range(predict + 1) if ce[k] != co[k]] != [i + 1]:
                        continue
                    out.append({"clean_ids": ce, "corrupt_ids": co,
                                "sub_idx": i + 1, "predict_idx": predict,
                                "clean_cap": int(cap), "corrupt_cap": int(acap[d]),
                                "sub_from": t, "sub_to": repl,
                                "family": SUB_FAMILY[(t, repl)]})
                    break
                if len(out) >= max_pairs:
                    return out
    return out


def pad_batch(rows, key, pad_id, width):
    return torch.tensor([r[key] + [pad_id] * (width - len(r[key])) for r in rows])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=1024)
    ap.add_argument("--n-ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-scan", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig(**ckpt["model_cfg"]) if ckpt.get("model_cfg") else \
        ModelConfig.load(os.path.join(os.path.dirname(args.ckpt), "model_config.json"))
    model = MolecularTransformer(cfg).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    del ckpt
    tok = SmilesTokenizer.load(args.tokenizer)

    smiles = open(args.data).read().split()
    rng = np.random.default_rng(args.seed)
    smiles = [smiles[i] for i in rng.permutation(len(smiles))[:args.n_scan]]
    pairs = build_pairs(smiles, tok, args.n_ctx, args.n_pairs, rng=rng)
    if not pairs:
        raise SystemExit("no valence minimal pairs found")
    gaps = [p["predict_idx"] - p["sub_idx"] for p in pairs]
    print(f"  {len(pairs)} minimal pairs; substitution-to-decision gap "
          f"min {min(gaps)} median {int(np.median(gaps))} max {max(gaps)}")

    def token_id(sym):
        body = [i for i in tok.encode(sym)
                if i not in (tok.bos_id, tok.eos_id, tok.pad_id)]
        if len(body) != 1:
            raise SystemExit(f"{sym!r} is not a single token")
        return body[0]

    t_double, t_single = token_id("="), token_id("C")
    width = max(len(p["clean_ids"]) for p in pairs)
    L, H = cfg.n_layers, cfg.n_heads

    def metric(logits, rows):
        pi = torch.tensor([r["predict_idx"] for r in rows], device=logits.device)
        sel = logits[torch.arange(len(rows), device=logits.device), pi]
        return (sel[:, t_double] - sel[:, t_single]).float()

    SITES = ("substitution", "decision", "sub..decision")
    resid_s = np.zeros((L, len(SITES)))
    resid_n = np.zeros((L, len(SITES)))
    head_s = np.zeros((L, H))
    cl, co_, n_eligible = [], [], 0
    # Per-pair recoveries, kept so the aggregate can be re-cut afterwards without
    # re-running the model. The substitution-to-decision gap is the cut that
    # matters: with a median gap of a few tokens, an aggregate alone cannot show
    # whether the effect is real transfer or an artefact of adjacency.
    pp_resid, pp_head, pp_gap, pp_family = [], [], [], []

    for s0 in range(0, len(pairs), args.batch):
        rows = pairs[s0:s0 + args.batch]
        xc = pad_batch(rows, "clean_ids", tok.pad_id, width).to(args.device)
        xk = pad_batch(rows, "corrupt_ids", tok.pad_id, width).to(args.device)
        cap = HookManager().capture(*[resid_post(i) for i in range(L)],
                                    *[f"blocks.{i}.attn.hook_z" for i in range(L)])
        lc = model(xc, hooks=cap)
        clean_act = {n: cap[n].clone() for n in cap._captures}
        d_clean, d_corr = metric(lc, rows), metric(model(xk), rows)
        denom = d_clean - d_corr
        ok = denom.abs() > 1e-6
        cl.append(d_clean.cpu().numpy())
        co_.append(d_corr.cpu().numpy())
        n_eligible += int(ok.sum())
        b_resid = np.full((len(rows), L, len(SITES)), np.nan)
        b_head = np.full((len(rows), L, H), np.nan)
        okm = ok.cpu().numpy()
        pp_gap.extend(r["predict_idx"] - r["sub_idx"] for r in rows)
        pp_family.extend(r["family"] for r in rows)

        si = torch.tensor([r["sub_idx"] for r in rows], device=args.device)
        pi = torch.tensor([r["predict_idx"] for r in rows], device=args.device)
        ar = torch.arange(len(rows), device=args.device)
        span = torch.zeros(len(rows), width, dtype=torch.bool, device=args.device)
        for b, r in enumerate(rows):
            span[b, r["sub_idx"]:r["predict_idx"] + 1] = True

        for ly in range(L):
            name = resid_post(ly)
            src = clean_act[name]
            fns = [lambda v, s: v.index_put_((ar, si), s[ar, si]),
                   lambda v, s: v.index_put_((ar, pi), s[ar, pi]),
                   lambda v, s: torch.where(span[:, :, None], s, v)]
            for slot, fn in enumerate(fns):
                def edit(v, _fn=fn, _s=src):
                    return _fn(v.clone(), _s)
                lp = model(xk, hooks=HookManager().edit(name, edit))
                full = ((metric(lp, rows) - d_corr) / denom).cpu().numpy()
                b_resid[okm, ly, slot] = full[okm]
                rec = full[okm]
                resid_s[ly, slot] += rec.sum()
                resid_n[ly, slot] += int(okm.sum())

        for ly in range(L):
            name = f"blocks.{ly}.attn.hook_z"
            src = clean_act[name]
            for hd in range(H):
                def edit(v, _h=hd, _s=src):
                    v = v.clone()
                    v[ar, _h, pi] = _s[ar, _h, pi]
                    return v
                lp = model(xk, hooks=HookManager().edit(name, edit))
                full = ((metric(lp, rows) - d_corr) / denom).cpu().numpy()
                b_head[okm, ly, hd] = full[okm]
                head_s[ly, hd] += full[okm].sum()
        pp_resid.append(b_resid)
        pp_head.append(b_head)
        print(f"    batch {s0 // args.batch + 1}: clean {d_clean.mean():+.3f} "
              f"corrupt {d_corr.mean():+.3f}", flush=True)

    resid = np.divide(resid_s, np.maximum(resid_n, 1))
    if not n_eligible:
        raise RuntimeError("No pair has a nonzero behavioral contrast for recovery")
    heads = head_s / n_eligible
    best = np.unravel_index(int(np.argmax(heads)), heads.shape)
    out = {
        "artifact_schema_version": 1,
        "analysis": "valence_capacity_activation_patching",
        "config": vars(args),
        "metric": "logit('=') - logit('C') at a bond decision whose capacity flips",
        "minimal_pair": ("one token substituted; prefix-correct remaining capacity "
                         "differs across the >=2 legality boundary downstream"),
        "n_pairs": len(pairs),
        "n_pairs_with_recovery": n_eligible,
        "family_counts": {f: int(sum(1 for x in pp_family if x == f))
                          for f in sorted(set(pp_family))},
        "gap_tokens": {"min": int(min(gaps)), "median": float(np.median(gaps)),
                       "max": int(max(gaps))},
        "mean_clean_metric": float(np.concatenate(cl).mean()),
        "mean_corrupt_metric": float(np.concatenate(co_).mean()),
        "resid_post_sites": list(SITES),
        "resid_post_recovery": resid.tolist(),
        "head_z_recovery_at_decision": heads.tolist(),
        "best_patch_head": {"layer": int(best[0]), "head": int(best[1]),
                            "patch_recovery": float(heads[best])},
        "head_recovery_total": float(heads.sum()),
    }
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out, "valence_patching_per_pair.npz"),
        resid=np.concatenate(pp_resid), head=np.concatenate(pp_head),
        gap=np.asarray(pp_gap), sites=np.array(SITES),
        family=np.array(pp_family),
        clean=np.concatenate(cl), corrupt=np.concatenate(co_))
    path = os.path.join(args.out, "valence_patching.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    Manifest().record("22_valence_patching",
                      {k: out[k] for k in ("n_pairs", "best_patch_head",
                                           "head_recovery_total")},
                      config=vars(args), files=[path])
    print(f"  clean {out['mean_clean_metric']:+.3f} / "
          f"corrupt {out['mean_corrupt_metric']:+.3f}")
    print(f"  best head L{best[0]}H{best[1]} "
          f"recovery={out['best_patch_head']['patch_recovery']:+.3f}; "
          f"total over heads {heads.sum():+.3f}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
