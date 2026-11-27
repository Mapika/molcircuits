"""Reproducible traversal-tree baseline + layer-6 hidden-difference graph-distance probe.

Evaluates the exact 1,000 aligned canonical validation molecules (500/500 molecule-grouped
split; 40,000 train / 40,000 test pairs; pairs_sha256 prefix d397777b9c28) across the six
Large (113.3M) checkpoints (canonical seeds 0-2 and randomized seeds 0-2) at zero-based
block index 6, comparing:
  1. Token offset d_S alone (5-feature basis)
  2. Token offset d_S + SMILES parenthesis traversal-tree path length d_T
  3. Token offset + traversal-tree distance + |h_6(i) - h_6(j)|
reported overall and specifically on pairs shortened by a ring closure (d_G < d_T).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import _element_of
from molcircuits.model.hooks import HookManager, resid_post

RDLogger.DisableLog("rdApp.*")

from importlib import import_module
_gi = import_module("04_graph_invariance")


def smiles_branch_tree_distances(smiles: str) -> np.ndarray:
    """Compute all-pairs shortest path distances on the SMILES parenthesis traversal tree.

    Ignores all ring-closure digits and ring-closing edges; connects each emitted heavy
    atom to the active parent atom on the left-to-right SMILES branch stack.
    """
    tokens = tokenize(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    n_atoms = mol.GetNumAtoms()
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    stack: list[int] = []
    cur: int | None = None
    atom_idx = 0
    for tk in tokens:
        if tk == "(":
            if cur is not None:
                stack.append(cur)
        elif tk == ")":
            if stack:
                cur = stack.pop()
        elif _gi._element_of(tk) is not None:
            if cur is not None:
                adj[cur].append(atom_idx)
                adj[atom_idx].append(cur)
            cur = atom_idx
            atom_idx += 1
    assert atom_idx == n_atoms, f"Tree atom count mismatch: {atom_idx} vs {n_atoms} for {smiles}"

    dist = np.full((n_atoms, n_atoms), -1, dtype=np.int32)
    for src in range(n_atoms):
        dist[src, src] = 0
        q = [src]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            du = dist[src, u]
            for v in adj[u]:
                if dist[src, v] < 0:
                    dist[src, v] = du + 1
                    q.append(v)
    return dist


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


@torch.no_grad()
def extract_block_reps(ckpt_path: str, pairs: list[dict], tok: SmilesTokenizer, block_idx: int, batch_size: int, device: str) -> np.ndarray:
    model = _gi.load_model(ckpt_path, device)
    hname = resid_post(block_idx)
    total_atoms = sum(p["n_atoms"] for p in pairs)
    H = np.empty((total_atoms, model.cfg.d_model), dtype=np.float32)
    cursor = 0
    for b0 in range(0, len(pairs), batch_size):
        chunk = pairs[b0 : b0 + batch_size]
        seqs = [tok.encode(p["sA"]) for p in chunk]
        max_len = max(len(s) for s in seqs)
        x = torch.full((len(chunk), max_len), tok.pad_id, dtype=torch.long, device=device)
        for bi, s in enumerate(seqs):
            x[bi, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        hm = HookManager().capture(hname)
        model(x, hooks=hm)
        arr = hm[hname].detach().float().cpu().numpy()
        for bi, p in enumerate(chunk):
            pos = p["posA"]
            n_a = p["n_atoms"]
            H[cursor : cursor + n_a] = arr[bi, pos]
            cursor += n_a
    del model
    torch.cuda.empty_cache()
    return H


def main() -> None:
    from molcircuits import paths as _paths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=_paths.data())
    ap.add_argument("--pairs-cache", default=str(Path(_paths.artifacts()) / "graph_invariance" / "pairs_cache_1000_seed0.json"))
    ap.add_argument("--block-idx", type=int, default=6)
    ap.add_argument("--max-pairs", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(REPO / "results" / "audit" / "tree_distance_probe.json"))
    args = ap.parse_args()

    smi_path, tok_path = _gi.resolve_data_and_tokenizer(args.data, None)
    tok = SmilesTokenizer.load(tok_path)
    pairs, pairs_sha = _gi.build_or_load_pairs(
        smi_path, tok, n_mols=1000, seed=args.seed, max_tokens=120, pairs_cache=args.pairs_cache
    )

    pair_u, pair_v, pair_dG, pair_dT, pair_dS, pair_mol = [], [], [], [], [], []
    cursor = 0
    for m_idx, p in enumerate(pairs):
        mol_c = Chem.MolFromSmiles(p["sA"])
        dmat_G = Chem.GetDistanceMatrix(mol_c)
        dmat_T = smiles_branch_tree_distances(p["sA"])
        ordA = p["ordA"]
        posA = p["posA"]
        n_a = p["n_atoms"]
        for u in range(n_a):
            au = ordA[u]
            tu = posA[au]
            for v in range(u + 1, n_a):
                av = ordA[v]
                tv = posA[av]
                dg = int(round(dmat_G[u, v]))
                if dg <= 0 or dg > 100:
                    continue
                pair_u.append(cursor + au)
                pair_v.append(cursor + av)
                pair_dG.append(float(dg))
                pair_dT.append(float(dmat_T[u, v]))
                pair_dS.append(int(tv - tu))
                pair_mol.append(m_idx)
        cursor += n_a

    pair_u = np.asarray(pair_u, dtype=np.int64)
    pair_v = np.asarray(pair_v, dtype=np.int64)
    pair_dG = np.asarray(pair_dG, dtype=np.float64)
    pair_dT = np.asarray(pair_dT, dtype=np.float64)
    pair_dS = np.asarray(pair_dS, dtype=np.int32)
    pair_mol = np.asarray(pair_mol, dtype=np.int32)

    # Exact 500/500 molecule-grouped split and 40,000 train / 40,000 test subsample as 04_graph_invariance.py
    n_mols = len(pairs)
    split_rng = np.random.default_rng(args.seed + 2026)
    mol_perm = split_rng.permutation(n_mols)
    train_mols = set(mol_perm[: n_mols // 2].tolist())
    is_tr = np.array([m in train_mols for m in pair_mol], dtype=bool)
    tr_idx = np.where(is_tr)[0]
    te_idx = np.where(~is_tr)[0]
    if len(tr_idx) > args.max_pairs:
        tr_idx = split_rng.choice(tr_idx, size=args.max_pairs, replace=False)
    if len(te_idx) > args.max_pairs:
        te_idx = split_rng.choice(te_idx, size=args.max_pairs, replace=False)

    y_tr = pair_dG[tr_idx]
    y_te = pair_dG[te_idx]
    ring_short_te = pair_dG[te_idx] < pair_dT[te_idx]
    tr_groups = pair_mol[tr_idx]

    X_str_tr = _gi.string_dist_features(pair_dS[tr_idx])
    X_str_te = _gi.string_dist_features(pair_dS[te_idx])
    X_tree_tr = np.column_stack([X_str_tr, pair_dT[tr_idx]])
    X_tree_te = np.column_stack([X_str_te, pair_dT[te_idx]])

    pred_str, a_str = _gi.fit_ridge_group_cv(X_str_tr, y_tr, tr_groups, X_str_te, device=args.device)
    pred_tree, a_tree = _gi.fit_ridge_group_cv(X_tree_tr, y_tr, tr_groups, X_tree_te, device=args.device)

    out = {
        "config": {
            "n_mols": n_mols,
            "pairs_sha256": pairs_sha,
            "split": "500/500 molecule-grouped (seed 2026, exact 04_graph_invariance.py split)",
            "n_train_pairs": int(len(tr_idx)),
            "n_test_pairs": int(len(te_idx)),
            "n_ring_shortened_test_pairs": int(ring_short_te.sum()),
            "block_idx_zero_based": args.block_idx,
        },
        "baselines": {
            "token_offset_only": {
                "alpha": a_str,
                "r2_all": r2_score_np(y_te, pred_str),
                "r2_ring_shortened": r2_score_np(y_te[ring_short_te], pred_str[ring_short_te]),
            },
            "token_offset_plus_branch_tree": {
                "alpha": a_tree,
                "r2_all": r2_score_np(y_te, pred_tree),
                "r2_ring_shortened": r2_score_np(y_te[ring_short_te], pred_tree[ring_short_te]),
            },
        },
        "models": {"canonical_large": {}, "randomized_large": {}},
    }

    ckpts = {
        "canonical_large": {
            f"seed{sd}": str(Path(_paths.root()) / "grid" / "zinc_large" / f"tf_seed{sd}" / "checkpoint.pt") for sd in (0, 1, 2)
        },
        "randomized_large": {
            f"seed{sd}": str(Path(_paths.root()) / "grid_rand" / "zinc_large" / f"tf_seed{sd}" / "checkpoint.pt") for sd in (0, 1, 2)
        },
    }

    for regime, seed_map in ckpts.items():
        all_r2, ring_r2, hid_only_r2 = [], [], []
        for sname, ckpt in seed_map.items():
            H = extract_block_reps(ckpt, pairs, tok, args.block_idx, args.batch, args.device)
            Hc = H - H.mean(axis=0, keepdims=True)
            D_tr = np.abs(Hc[pair_u[tr_idx]] - Hc[pair_v[tr_idx]]).astype(np.float64)
            D_te = np.abs(Hc[pair_u[te_idx]] - Hc[pair_v[te_idx]]).astype(np.float64)

            p_hid, a_hid = _gi.fit_ridge_group_cv(D_tr, y_tr, tr_groups, D_te, device=args.device)
            p_cmb, a_cmb = _gi.fit_ridge_group_cv(
                np.column_stack([X_tree_tr, D_tr]),
                y_tr,
                tr_groups,
                np.column_stack([X_tree_te, D_te]),
                device=args.device,
            )
            r_all = r2_score_np(y_te, p_cmb)
            r_ring = r2_score_np(y_te[ring_short_te], p_cmb[ring_short_te])
            r_ho = r2_score_np(y_te, p_hid)
            all_r2.append(r_all)
            ring_r2.append(r_ring)
            hid_only_r2.append(r_ho)
            out["models"][regime][sname] = {
                "ckpt": ckpt,
                "alpha_tree_plus_hidden": a_cmb,
                "r2_hidden_only_all": r_ho,
                "r2_tree_plus_hidden_all": r_all,
                "r2_tree_plus_hidden_ring_shortened": r_ring,
            }
        out["models"][regime]["mean"] = {
            "r2_hidden_only_all": float(np.mean(hid_only_r2)),
            "r2_tree_plus_hidden_all": float(np.mean(all_r2)),
            "r2_tree_plus_hidden_ring_shortened": float(np.mean(ring_r2)),
        }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["baselines"], indent=2))
    print("canonical_large mean:", out["models"]["canonical_large"]["mean"])
    print("randomized_large mean:", out["models"]["randomized_large"]["mean"])


if __name__ == "__main__":
    main()
