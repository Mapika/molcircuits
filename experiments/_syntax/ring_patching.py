#!/usr/bin/env python3
"""Stage 21: where does the opener's identity live, and which head reads it?

Ablation (stage 19) shows the ring-pointer head is selectively necessary. It does
not show WHERE the information is or HOW it moves, because removing a component
tells you only that something broke. Patching localizes instead of destroying,
and answers the question the word "mechanism" actually makes.

MINIMAL PAIR. Take a molecule whose only ring label is `1` and relabel it to `2`:

    clean      Cc1ccccc1O        the closer must be `1`
    corrupted  Cc2ccccc2O        the closer must be `2`

Both are valid SMILES for the same molecule. Up to the position that predicts the
closer, the two token sequences differ in EXACTLY ONE place -- the opener digit.
So the only signal separating them is the opener's identity, which is precisely
what a ring pointer has to retrieve. That makes the pair a clean instrument: any
site whose patch restores the clean answer is carrying opener identity, and no
other difference can be responsible.

METRIC. logit(clean label) - logit(corrupt label) at the predicting position.
Recovery is (patched - corrupt) / (clean - corrupt): 1.0 restores the clean
answer completely, 0.0 leaves the corrupted answer untouched. It is a ratio of
differences rather than a raw logit, so a site that merely shifts both labels
together -- a common-mode effect -- scores 0 rather than looking like retrieval.

TWO SCANS.
  resid_post  layer x position: where the identity is, and where it has moved to
  hook_z      layer x head, at the predicting position only: which head moves it

The head scan is the one that speaks to Section 3: if the head with the highest
pointer mass is also the head whose patch restores the answer, then the detector
identified the mechanism, and the disagreement reported for margin-based ablation
is a fact about that statistic rather than about the head.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.model.hooks import HookManager, resid_post
from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.utils.manifest import Manifest


def build_pairs(smiles, tok, n_ctx, clean_label="1", corrupt_label="2",
                max_pairs=1024):
    """Minimal pairs differing only in the ring label.

    Kept only when the molecule uses `clean_label` exactly twice and never uses
    `corrupt_label`, so the relabeling is unambiguous and the corrupted string
    stays a legal SMILES for the same graph.
    """
    out = []
    for s in smiles:
        try:
            toks = tokenize(s)
        except Exception:
            continue
        if toks.count(clean_label) != 2 or corrupt_label in toks:
            continue
        idx = [i for i, t in enumerate(toks) if t == clean_label]
        open_i, close_i = idx
        # +1 for BOS: the model sees [BOS] + tokens.
        predict_idx = close_i - 1 + 1
        if predict_idx <= open_i + 1 or close_i + 1 >= n_ctx:
            continue
        corrupt = [corrupt_label if t == clean_label else t for t in toks]
        ce, co = tok.encode("".join(toks)), tok.encode("".join(corrupt))
        if len(ce) != len(co) or len(ce) > n_ctx:
            continue
        # The two inputs must differ at exactly one position at or before the
        # prediction site, otherwise the pair is not minimal and the patch is
        # not attributable to opener identity.
        diff = [i for i in range(predict_idx + 1) if ce[i] != co[i]]
        if diff != [open_i + 1]:
            continue
        out.append({"clean_ids": ce, "corrupt_ids": co,
                    "open_idx": open_i + 1, "predict_idx": predict_idx})
        if len(out) >= max_pairs:
            break
    return out


def pad_batch(rows, key, pad_id, width):
    return torch.tensor([r[key] + [pad_id] * (width - len(r[key])) for r in rows])


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True, help="corpus .smi to draw pairs from")
    ap.add_argument("--pointers", required=True, help="pointer_results.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=512)
    ap.add_argument("--n-ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-scan-positions", type=int, default=48)
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
    smiles = [smiles[i] for i in rng.permutation(len(smiles))[:200000]]
    pairs = build_pairs(smiles, tok, args.n_ctx, max_pairs=args.n_pairs)
    if not pairs:
        raise SystemExit("no minimal pairs found")
    print(f"  {len(pairs)} minimal pairs")

    # encode() wraps in BOS/EOS, so the label's own id is the middle element.
    def token_id(sym: str) -> int:
        ids = tok.encode(sym)
        body = [i for i in ids if i not in (tok.bos_id, tok.eos_id, tok.pad_id)]
        if len(body) != 1:
            raise SystemExit(f"ring label {sym!r} is not a single token")
        return body[0]

    tok_clean, tok_corr = token_id("1"), token_id("2")

    width = max(len(p["clean_ids"]) for p in pairs)
    L, H = cfg.n_layers, cfg.n_heads

    def logit_diff(logits, rows):
        """clean-label minus corrupt-label logit at each pair's predict index."""
        pi = torch.tensor([r["predict_idx"] for r in rows], device=logits.device)
        sel = logits[torch.arange(len(rows), device=logits.device), pi]
        return (sel[:, tok_clean] - sel[:, tok_corr]).float()

    RESID_SITES = ("opener", "predict", "opener..predict")
    resid_scores = np.zeros((L, len(RESID_SITES)))
    resid_n = np.zeros((L, len(RESID_SITES)))
    head_scores = np.zeros((L, H))
    clean_all, corrupt_all, n_eligible = [], [], 0

    for s0 in range(0, len(pairs), args.batch):
        rows = pairs[s0:s0 + args.batch]
        xc = pad_batch(rows, "clean_ids", tok.pad_id, width).to(args.device)
        xk = pad_batch(rows, "corrupt_ids", tok.pad_id, width).to(args.device)

        cap = HookManager().capture(*[resid_post(i) for i in range(L)],
                                    *[f"blocks.{i}.attn.hook_z" for i in range(L)])
        lc = model(xc, hooks=cap)
        clean_act = {n: cap[n].clone() for n in cap._captures}
        d_clean = logit_diff(lc, rows)
        d_corr = logit_diff(model(xk), rows)
        denom = (d_clean - d_corr)
        ok = denom.abs() > 1e-6
        clean_all.append(d_clean.cpu().numpy())
        corrupt_all.append(d_corr.cpu().numpy())
        n_eligible += int(ok.sum())

        # ---- residual scan: layer x LANDMARK ------------------------------ #
        # Landmarks are per-pair, not absolute positions. Every pair has its own
        # opener and prediction index, so averaging recovery at absolute position
        # p would pool "the opener" for one molecule with an unrelated token for
        # another and smear exactly the structure this scan is looking for.
        oi = torch.tensor([r["open_idx"] for r in rows], device=args.device)
        pi = torch.tensor([r["predict_idx"] for r in rows], device=args.device)
        ar = torch.arange(len(rows), device=args.device)
        span = torch.zeros(len(rows), width, dtype=torch.bool, device=args.device)
        for b, r in enumerate(rows):
            span[b, r["open_idx"]:r["predict_idx"] + 1] = True

        for ly in range(L):
            name = resid_post(ly)
            src = clean_act[name]
            sites = {
                0: ("opener", lambda v, s: v.index_put_(
                    (ar, oi), s[ar, oi])),
                1: ("predict", lambda v, s: v.index_put_(
                    (ar, pi), s[ar, pi])),
                2: ("opener..predict", lambda v, s: torch.where(
                    span[:, :, None], s, v)),
            }
            for slot, (_label, fn) in sites.items():
                def edit(v, _fn=fn, _s=src):
                    return _fn(v.clone(), _s)
                lp = model(xk, hooks=HookManager().edit(name, edit))
                rec = ((logit_diff(lp, rows) - d_corr) / denom)[ok]
                resid_scores[ly, slot] += rec.sum().item()
                resid_n[ly, slot] += int(ok.sum().item())

        # ---- head scan: layer x head, at the predicting position ---------- #
        for ly in range(L):
            name = f"blocks.{ly}.attn.hook_z"
            src = clean_act[name]                      # (B, H, T, d_head)
            for hd in range(H):
                def edit(v, _h=hd, _s=src, _pi=pi, _ar=ar):
                    v = v.clone()
                    v[_ar, _h, _pi] = _s[_ar, _h, _pi]
                    return v
                lp = model(xk, hooks=HookManager().edit(name, edit))
                rec = ((logit_diff(lp, rows) - d_corr) / denom)[ok]
                head_scores[ly, hd] += rec.sum().item()
        print(f"    batch {s0 // args.batch + 1}: "
              f"clean {d_clean.mean():+.3f} corrupt {d_corr.mean():+.3f}", flush=True)

    resid = np.divide(resid_scores, np.maximum(resid_n, 1))
    if not n_eligible:
        raise RuntimeError("No pair has a nonzero behavioral contrast for recovery")
    heads = head_scores / n_eligible

    ptr = json.load(open(args.pointers))
    zero = [h for h in ptr["ring"]["heads"] if h["mode"] == "zero"]
    top = max(zero, key=lambda h: h["pointer_mass"])
    best = np.unravel_index(int(np.argmax(heads)), heads.shape)

    out = {
        "artifact_schema_version": 1,
        "analysis": "ring_opener_activation_patching",
        "config": vars(args),
        "metric": ("recovery of (logit[clean label] - logit[corrupt label]) at the "
                   "position predicting the ring closer; 1.0 = clean, 0.0 = corrupt"),
        "minimal_pair": "single ring relabelled 1 -> 2; inputs differ only at the opener",
        "n_pairs": len(pairs),
        "n_pairs_with_recovery": n_eligible,
        "mean_clean_logit_diff": float(np.concatenate(clean_all).mean()),
        "mean_corrupt_logit_diff": float(np.concatenate(corrupt_all).mean()),
        "resid_post_sites": list(RESID_SITES),
        "resid_post_recovery": resid.tolist(),
        "head_z_recovery_at_predict": heads.tolist(),
        "top_pointer_head": {"layer": top["layer"], "head": top["head"],
                             "pointer_mass": top["pointer_mass"],
                             "patch_recovery": float(heads[top["layer"], top["head"]])},
        "best_patch_head": {"layer": int(best[0]), "head": int(best[1]),
                            "patch_recovery": float(heads[best])},
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ring_patching.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    Manifest().record("21_ring_patching",
                      {k: out[k] for k in ("n_pairs", "top_pointer_head",
                                           "best_patch_head")},
                      config=vars(args), files=[path])
    print(f"  top-pointer head L{top['layer']}H{top['head']} "
          f"recovery={out['top_pointer_head']['patch_recovery']:+.3f}")
    print(f"  best patch head  L{best[0]}H{best[1]} "
          f"recovery={out['best_patch_head']['patch_recovery']:+.3f}")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
