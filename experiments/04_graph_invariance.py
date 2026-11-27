#!/usr/bin/env python3
"""Workstream 2 — Script 1: Atom-Aligned Traversal Graph Invariance & Graph-vs-String Distance.

Measures:
  1. Atom-aligned traversal invariance across every layer (`embed` + `blocks.0..L-1.hook_resid_post`):
     - Same-atom cross-traversal cosine similarity (raw & mean-centered) between canonical SMILES
       s_A and randomized SMILES s_B, aligned 1-to-1 via RDKit `_smilesAtomOutputOrder`.
     - Null 1: within-molecule same-SMILES-token different-atom cosine similarity.
     - Null 2: exact between-molecule same-SMILES-token cosine similarity.
     - Within-molecule top-1 same-token & all-atom retrieval accuracy.
     - Linear CKA between molecule-level mean-pooled representations and between atom-aligned matrices.
     - Splits by topological context: position shift |pos_A - pos_B|, prefix Jaccard overlap,
       ring vs acyclic, aromatic vs aliphatic ring vs acyclic.
  2. Molecular graph distance vs SMILES string distance geometry:
     - Layer-wise residual geometry (cosine similarity, L2 distance, vector |h_i - h_j| Ridge probe)
       predicting shortest-path molecular graph distance d_G(i, j) controlling for token distance d_S(i, j).
     - Non-adjacent bonded (d_G=1, d_S>=3) vs non-bonded AUROC from residual geometry.
     - Per-head attention partial correlation with d_G controlling for d_S, and exact token-distance-matched
       bonded attention enrichment ratio for d_S >= 3.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import _element_of, is_atom_token
from molcircuits.model.hooks import HookManager, attn_pattern, resid_post
from molcircuits.model.transformer import ModelConfig, MolecularTransformer

RDLogger.DisableLog("rdApp.*")


def resolve_data_and_tokenizer(data_arg: str, tok_arg: str | None) -> tuple[str, str]:
    p = Path(data_arg)
    if p.is_dir():
        smi_path = str(p / "valid.smi")
        tok_path = tok_arg or str(p / "tokenizer.json")
    else:
        smi_path = str(p)
        tok_path = tok_arg or str(p.parent / "tokenizer.json")
    return smi_path, tok_path


def load_model(ckpt_path: str, device: str) -> MolecularTransformer:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_cfg"):
        cfg = ModelConfig(**ckpt["model_cfg"])
    else:
        cfg = ModelConfig.load(os.path.join(os.path.dirname(ckpt_path), "model_config.json"))
    model = MolecularTransformer(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def build_or_load_pairs(
    smi_path: str,
    tok: SmilesTokenizer,
    n_mols: int,
    seed: int,
    max_tokens: int = 120,
    min_heavy_atoms: int = 8,
    pairs_cache: str | None = None,
) -> tuple[list[dict], str]:
    """Sample n_mols valid molecules and create aligned (s_A, s_B) traversal pairs."""
    if pairs_cache and os.path.exists(pairs_cache):
        with open(pairs_cache) as fh:
            cached = json.load(fh)
        pairs = cached["pairs"]
        assert len(pairs) == n_mols, f"Cache has {len(pairs)} mols, expected {n_mols}"
        # Re-verify alignment assertion on loaded cache
        for rec in pairs:
            verify_pair_alignment(rec, tok)
        sha = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()
        return pairs, sha

    with open(smi_path) as fh:
        raw_smiles = [line.strip().split()[0] for line in fh if line.strip()]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(raw_smiles))

    seen_canon: set[str] = set()
    pairs: list[dict] = []
    vocab = tok.vocab

    for idx in perm:
        s = raw_smiles[idx]
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        n_atoms = mol.GetNumAtoms()
        if n_atoms < min_heavy_atoms:
            continue
        sA = Chem.MolToSmiles(mol, canonical=True)
        if sA in seen_canon:
            continue
        try:
            ordA = list(ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder")))
            tA = tokenize(sA)
        except Exception:
            continue
        if len(tA) + 2 > max_tokens or not all(t in vocab for t in tA):
            continue
        atA = [i for i, t in enumerate(tA) if is_atom_token(t)]
        if len(atA) != n_atoms or len(ordA) != n_atoms:
            continue

        # Generate distinct randomized SMILES s_B != s_A with clean tokenization
        found = None
        for _ in range(12):
            sB = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            if sB == sA:
                continue
            try:
                ordB = list(ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder")))
                tB = tokenize(sB)
            except Exception:
                continue
            if len(tB) + 2 > max_tokens or not all(t in vocab for t in tB):
                continue
            atB = [i for i, t in enumerate(tB) if is_atom_token(t)]
            if len(atB) != n_atoms or len(ordB) != n_atoms:
                continue
            found = (sB, tB, atB, ordB)
            break

        if found is None:
            continue
        sB, tB, atB, ordB = found

        # Build mapping from RDKit atom idx a -> 1-based token position (offset +1 for BOS)
        posA = [0] * n_atoms
        posB = [0] * n_atoms
        tokA = [""] * n_atoms
        tokB = [""] * n_atoms
        for k, a in enumerate(ordA):
            posA[a] = atA[k] + 1
            tokA[a] = tA[atA[k]]
        for k, a in enumerate(ordB):
            posB[a] = atB[k] + 1
            tokB[a] = tB[atB[k]]

        # Compute topological metadata per atom
        in_ring = [bool(mol.GetAtomWithIdx(a).IsInRing()) for a in range(n_atoms)]
        is_arom = [bool(mol.GetAtomWithIdx(a).GetIsAromatic()) for a in range(n_atoms)]
        elements = [mol.GetAtomWithIdx(a).GetSymbol() for a in range(n_atoms)]

        # Prefix Jaccard overlap: set of heavy atoms appearing before atom a in s_A vs s_B
        rankA = {a: k for k, a in enumerate(ordA)}
        rankB = {a: k for k, a in enumerate(ordB)}
        prefix_jacc = []
        for a in range(n_atoms):
            preA = set(ordA[: rankA[a]])
            preB = set(ordB[: rankB[a]])
            union = preA | preB
            prefix_jacc.append(1.0 if not union else len(preA & preB) / len(union))

        rec = {
            "sA": sA,
            "sB": sB,
            "n_atoms": n_atoms,
            "ordA": ordA,
            "ordB": ordB,
            "posA": posA,
            "posB": posB,
            "tokA": tokA,
            "tokB": tokB,
            "elements": elements,
            "in_ring": in_ring,
            "is_arom": is_arom,
            "prefix_jaccard": prefix_jacc,
        }
        verify_pair_alignment(rec, tok)
        seen_canon.add(sA)
        pairs.append(rec)
        if len(pairs) >= n_mols:
            break

    if len(pairs) < n_mols:
        raise RuntimeError(f"Only found {len(pairs)}/{n_mols} valid aligned pairs")

    sha = hashlib.sha256(json.dumps(pairs, sort_keys=True).encode()).hexdigest()
    if pairs_cache:
        os.makedirs(os.path.dirname(os.path.abspath(pairs_cache)), exist_ok=True)
        with open(pairs_cache, "w") as fh:
            json.dump({"n_mols": n_mols, "seed": seed, "sha256": sha, "pairs": pairs}, fh)
    return pairs, sha


def verify_pair_alignment(rec: dict, tok: SmilesTokenizer) -> None:
    """Assert 100% element-symbol agreement between s_A, s_B, and RDKit mol atoms."""
    tA = tokenize(rec["sA"])
    tB = tokenize(rec["sB"])
    for a in range(rec["n_atoms"]):
        ta = tA[rec["posA"][a] - 1]
        tb = tB[rec["posB"][a] - 1]
        ea = _element_of(ta)
        eb = _element_of(tb)
        er = rec["elements"][a]
        assert ea is not None and eb is not None, f"Non-atom token at atom {a}: {ta}, {tb}"
        assert ea.lower() == eb.lower() == er.lower(), (
            f"Element mismatch at atom {a}: sA={ta}({ea}) sB={tb}({eb}) rdkit={er}"
        )
        assert ta == rec["tokA"][a] and tb == rec["tokB"][a]
        assert ta in tok.vocab and tb in tok.vocab


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact Linear CKA between two (N, D) matrices using O(N D^2) Gram formulation."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    xt_y = Xc.T @ Yc
    xt_x = Xc.T @ Xc
    yt_y = Yc.T @ Yc
    num = float(np.sum(xt_y * xt_y))
    denom = float(np.sqrt(np.sum(xt_x * xt_x) * np.sum(yt_y * yt_y)))
    return num / denom if denom > 1e-12 else float("nan")


def compute_invariance_metrics(
    HA: np.ndarray,
    HB: np.ndarray,
    mol_ids: np.ndarray,
    tokA_ids: np.ndarray,
    tokB_ids: np.ndarray,
    pos_shift: np.ndarray,
    prefix_jacc: np.ndarray,
    in_ring: np.ndarray,
    is_arom: np.ndarray,
    mol_slices: list[tuple[int, int]],
    center: bool,
) -> dict:
    """Compute atom-aligned cosine invariance, Null 1, Null 2, retrieval, and context splits."""
    if center:
        mu = 0.5 * (HA.mean(axis=0, keepdims=True) + HB.mean(axis=0, keepdims=True))
        A = HA - mu
        B = HB - mu
    else:
        A = HA
        B = HB

    normA = np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)
    normB = np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-12)
    uA = A / normA
    uB = B / normB

    # Same-atom cross-traversal cosine similarity
    same_cos = np.sum(uA * uB, axis=1)

    # Exact Null 2 (between-molecule same-SMILES-token) via group sums
    # Match tokA_ids[i] against all atoms k in OTHER molecules with tokB_ids[k] == tokA_ids[i]
    uniq_toks = np.unique(np.concatenate([tokA_ids, tokB_ids]))
    tok_to_idx = {int(t): idx for idx, t in enumerate(uniq_toks)}
    D = uA.shape[1]
    T_u = len(uniq_toks)
    global_sum_B = np.zeros((T_u, D), dtype=np.float64)
    global_cnt_B = np.zeros(T_u, dtype=np.int64)
    tB_mapped = np.array([tok_to_idx[int(t)] for t in tokB_ids], dtype=np.int64)
    tA_mapped = np.array([tok_to_idx[int(t)] for t in tokA_ids], dtype=np.int64)
    np.add.at(global_sum_B, tB_mapped, uB)
    np.add.at(global_cnt_B, tB_mapped, 1)

    null1_vals = []
    null1_mol_means = []
    null2_per_atom = np.full(len(same_cos), np.nan, dtype=np.float64)
    same_mol_means = []
    top1_all_hits = 0
    top1_all_total = 0
    top1_sametok_hits = 0
    top1_sametok_total = 0

    for s, e in mol_slices:
        n_m = e - s
        uA_m = uA[s:e]
        uB_m = uB[s:e]
        tA_m = tA_mapped[s:e]
        tB_m = tB_mapped[s:e]
        sim_mat = uA_m @ uB_m.T  # (n_m, n_m)

        same_mol_means.append(float(np.mean(np.diag(sim_mat))))

        # Top-1 retrieval across all atoms in molecule
        nn_all = np.argmax(sim_mat, axis=1)
        top1_all_hits += int(np.sum(nn_all == np.arange(n_m)))
        top1_all_total += n_m

        # Null 1: within-molecule same-token different-atom (i != j, tokA[i] == tokB[j])
        mol_n1 = []
        for i in range(n_m):
            same_t_j = np.where(tB_m == tA_m[i])[0]
            other_j = same_t_j[same_t_j != i]
            if len(other_j) > 0:
                mol_n1.extend(sim_mat[i, other_j].tolist())
                # Top-1 retrieval restricted to same-token candidates (only when >= 2 candidates & i in candidates)
                if i in same_t_j:
                    best_j = same_t_j[np.argmax(sim_mat[i, same_t_j])]
                    top1_sametok_hits += int(best_j == i)
                    top1_sametok_total += 1
        if mol_n1:
            null1_vals.extend(mol_n1)
            null1_mol_means.append(float(np.mean(mol_n1)))

        # Exact Null 2 for atoms in this molecule
        # Compute per-token sums within this molecule
        local_toks = np.unique(tB_m)
        local_sum = {}
        local_cnt = {}
        for lt in local_toks:
            mask = tB_m == lt
            local_sum[int(lt)] = uB_m[mask].sum(axis=0)
            local_cnt[int(lt)] = int(mask.sum())
        for i in range(n_m):
            lt = int(tA_m[i])
            c_other = int(global_cnt_B[lt]) - local_cnt.get(lt, 0)
            if c_other > 0:
                s_other = global_sum_B[lt] - local_sum.get(lt, 0.0)
                null2_per_atom[s + i] = float(np.dot(uA_m[i], s_other) / c_other)

    valid_n2 = ~np.isnan(null2_per_atom)
    same_mean = float(np.mean(same_cos))
    null1_mean = float(np.mean(null1_vals)) if null1_vals else float("nan")
    null2_mean = float(np.mean(null2_per_atom[valid_n2])) if valid_n2.any() else float("nan")

    # Context splits
    def split_stat(mask: np.ndarray) -> dict:
        cnt = int(mask.sum())
        if cnt == 0:
            return {"n_atoms": 0, "same_atom_cos": float("nan"), "null2_cos": float("nan"), "margin_vs_null2": float("nan")}
        sc = float(same_cos[mask].mean())
        m2 = mask & valid_n2
        n2 = float(null2_per_atom[m2].mean()) if m2.any() else float("nan")
        return {
            "n_atoms": cnt,
            "same_atom_cos": sc,
            "null2_cos": n2,
            "margin_vs_null2": sc - n2 if not np.isnan(n2) else float("nan"),
        }

    splits = {
        "pos_shift_small_le3": split_stat(pos_shift <= 3),
        "pos_shift_med_4_10": split_stat((pos_shift >= 4) & (pos_shift <= 10)),
        "pos_shift_large_gt10": split_stat(pos_shift > 10),
        "prefix_jaccard_low_lt03": split_stat(prefix_jacc < 0.3),
        "prefix_jaccard_mid_03_07": split_stat((prefix_jacc >= 0.3) & (prefix_jacc < 0.7)),
        "prefix_jaccard_high_ge07": split_stat(prefix_jacc >= 0.7),
        "ring_atoms": split_stat(in_ring),
        "chain_atoms": split_stat(~in_ring),
        "aromatic_ring_atoms": split_stat(in_ring & is_arom),
        "aliphatic_ring_atoms": split_stat(in_ring & (~is_arom)),
    }

    return {
        "same_atom_cos_mean": same_mean,
        "same_atom_cos_std": float(np.std(same_cos)),
        "same_atom_cos_mol_mean": float(np.mean(same_mol_means)),
        "null1_within_mol_same_tok_mean": null1_mean,
        "null1_within_mol_mol_mean": float(np.mean(null1_mol_means)) if null1_mol_means else float("nan"),
        "null1_n_pairs": len(null1_vals),
        "null2_between_mol_same_tok_mean": null2_mean,
        "margin_vs_null1": same_mean - null1_mean,
        "margin_vs_null2": same_mean - null2_mean,
        "top1_retrieval_all_atoms": top1_all_hits / max(top1_all_total, 1),
        "top1_retrieval_same_tok": top1_sametok_hits / max(top1_sametok_total, 1),
        "top1_same_tok_n": top1_sametok_total,
        "splits": splits,
    }


def partial_pearson(x: np.ndarray, y: np.ndarray, z_feats: np.ndarray) -> float:
    """Partial Pearson correlation between x and y controlling for z_feats."""
    Z = np.column_stack([np.ones(len(x)), z_feats])
    bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
    by, *_ = np.linalg.lstsq(Z, y, rcond=None)
    rx = x - Z @ bx
    ry = y - Z @ by
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(np.dot(rx, ry) / denom) if denom > 1e-12 else float("nan")


def partial_spearman(x: np.ndarray, y: np.ndarray, z_feats: np.ndarray) -> float:
    """Partial Spearman correlation via rank-transformation + residualization on z_feats."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = np.column_stack([stats.rankdata(z_feats[:, j]) for j in range(z_feats.shape[1])])
    return partial_pearson(rx, ry, rz)


def fit_ridge_group_cv(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: np.ndarray | None = None,
    n_folds: int = 5,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Fast Ridge regression with molecule-grouped K-fold CV using exact Gram matrices."""
    if alphas is None:
        alphas = np.logspace(-2, 5, 15)
    # Standardize using train only
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs_tr = ((X_tr - mu) / sd).astype(np.float64)
    Xs_te = ((X_te - mu) / sd).astype(np.float64)
    y_mu = y_tr.mean(axis=0, keepdims=True)
    yc_tr = (y_tr - y_mu).astype(np.float64)

    uniq_g = np.unique(groups_tr)
    fold_assign = {g: i % n_folds for i, g in enumerate(uniq_g)}
    fold_ids = np.array([fold_assign[g] for g in groups_tr])

    D = Xs_tr.shape[1]
    use_gpu = device.startswith("cuda") and torch.cuda.is_available()
    if use_gpu:
        dev = torch.device(device)
        Xt = torch.from_numpy(Xs_tr).to(dev)
        yt = torch.from_numpy(yc_tr).to(dev)
        Xtet = torch.from_numpy(Xs_te).to(dev)
        eye = torch.eye(D, dtype=torch.float64, device=dev)
        G_folds, b_folds, val_slices = [], [], []
        for f in range(n_folds):
            idx = torch.from_numpy(np.where(fold_ids == f)[0]).to(dev)
            Xf = Xt.index_select(0, idx)
            yf = yt.index_select(0, idx)
            G_folds.append(Xf.T @ Xf)
            b_folds.append(Xf.T @ yf)
            val_slices.append((Xf, yf))
        G_all = sum(G_folds)
        b_all = sum(b_folds)
        best_alpha = float(alphas[0])
        best_mse = float("inf")
        for alpha in alphas:
            se = 0.0
            a_eye = float(alpha) * eye
            for f in range(n_folds):
                w = torch.linalg.solve(G_all - G_folds[f] + a_eye, b_all - b_folds[f])
                Xf, yf = val_slices[f]
                diff = yf - Xf @ w
                se += float((diff * diff).sum().item())
            if se < best_mse:
                best_mse = se
                best_alpha = float(alpha)
        w_full = torch.linalg.solve(G_all + best_alpha * eye, b_all)
        preds = (Xtet @ w_full).cpu().numpy() + y_mu
        return preds, best_alpha

    eye = np.eye(D, dtype=np.float64)
    G_folds, b_folds = [], []
    for f in range(n_folds):
        m = fold_ids == f
        Xf = Xs_tr[m]
        yf = yc_tr[m]
        G_folds.append(Xf.T @ Xf)
        b_folds.append(Xf.T @ yf)
    G_all = sum(G_folds)
    b_all = sum(b_folds)

    best_alpha = float(alphas[0])
    best_mse = float("inf")
    for alpha in alphas:
        se = 0.0
        for f in range(n_folds):
            G_tr_f = G_all - G_folds[f] + alpha * eye
            b_tr_f = b_all - b_folds[f]
            w = np.linalg.solve(G_tr_f, b_tr_f)
            m = fold_ids == f
            pred = Xs_tr[m] @ w
            se += float(np.sum((yc_tr[m] - pred) ** 2))
        if se < best_mse:
            best_mse = se
            best_alpha = float(alpha)

    w_full = np.linalg.solve(G_all + best_alpha * eye, b_all)
    preds = Xs_te @ w_full + y_mu
    return preds, best_alpha


def string_dist_features(dS: np.ndarray) -> np.ndarray:
    d = dS.astype(np.float64)
    return np.column_stack([d, np.log1p(d), 1.0 / d, d ** 2, np.sqrt(d)])


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="Path to valid.smi or data directory")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True, help="Output directory or .json file")
    ap.add_argument("--n-mols", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pairs-cache", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    smi_path, tok_path = resolve_data_and_tokenizer(args.data, args.tokenizer)
    tok = SmilesTokenizer.load(tok_path)
    model = load_model(args.ckpt, args.device)
    L, H, D = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model
    max_toks = min(120, model.cfg.n_ctx)

    print(f"[40_graph_invariance] ckpt={args.ckpt} (L={L}, H={H}, D={D}) n_mols={args.n_mols}", flush=True)
    pairs, pairs_sha = build_or_load_pairs(
        smi_path, tok, args.n_mols, args.seed,
        max_tokens=max_toks, pairs_cache=args.pairs_cache,
    )
    print(f"  aligned {len(pairs)} molecules (pairs_sha={pairs_sha[:12]})", flush=True)

    # Precompute graph distance matrices for canonical molecules & atom metadata arrays
    total_atoms = sum(p["n_atoms"] for p in pairs)
    layer_names = ["embed"] + [f"layer_{l}" for l in range(L)]
    hook_names = ["hook_embed"] + [resid_post(l) for l in range(L)]

    HA_layers = [np.empty((total_atoms, D), dtype=np.float32) for _ in range(L + 1)]
    HB_layers = [np.empty((total_atoms, D), dtype=np.float32) for _ in range(L + 1)]
    mol_mean_A = [np.empty((len(pairs), D), dtype=np.float32) for _ in range(L + 1)]
    mol_mean_B = [np.empty((len(pairs), D), dtype=np.float32) for _ in range(L + 1)]

    mol_ids = np.empty(total_atoms, dtype=np.int32)
    tokA_ids = np.empty(total_atoms, dtype=np.int32)
    tokB_ids = np.empty(total_atoms, dtype=np.int32)
    pos_shift = np.empty(total_atoms, dtype=np.int32)
    prefix_jacc = np.empty(total_atoms, dtype=np.float32)
    in_ring = np.empty(total_atoms, dtype=bool)
    is_arom = np.empty(total_atoms, dtype=bool)
    mol_slices: list[tuple[int, int]] = []

    # Pairwise graph-distance vs string-distance structures (on canonical s_A)
    MAX_DS_MATCH = 35
    attn_sum_bond = np.zeros((L, H, MAX_DS_MATCH + 1), dtype=np.float64)
    attn_sum_nonbond = np.zeros((L, H, MAX_DS_MATCH + 1), dtype=np.float64)
    cnt_bond = np.zeros(MAX_DS_MATCH + 1, dtype=np.int64)
    cnt_nonbond = np.zeros(MAX_DS_MATCH + 1, dtype=np.int64)

    # Exact online sufficient stats for partial Pearson of attention vs d_G controlling for d_S (d_S >= 2)
    # Using vectorized batch collection for a large deterministic subsample (60,000 pairs) + exact enrichment on ALL pairs
    pair_u_global = []
    pair_v_global = []
    pair_dG = []
    pair_dS = []
    pair_mol = []
    attn_sub_list = []
    attn_sub_dG = []
    attn_sub_dS = []
    sub_rng = np.random.default_rng(args.seed + 1337)

    offset = 0
    for m_idx, p in enumerate(pairs):
        n = p["n_atoms"]
        mol_slices.append((offset, offset + n))
        mol_ids[offset:offset + n] = m_idx
        tokA_ids[offset:offset + n] = [tok.vocab[t] for t in p["tokA"]]
        tokB_ids[offset:offset + n] = [tok.vocab[t] for t in p["tokB"]]
        pos_shift[offset:offset + n] = [abs(p["posA"][a] - p["posB"][a]) for a in range(n)]
        prefix_jacc[offset:offset + n] = p["prefix_jaccard"]
        in_ring[offset:offset + n] = p["in_ring"]
        is_arom[offset:offset + n] = p["is_arom"]

        mol = Chem.MolFromSmiles(p["sA"])
        dmat = Chem.GetDistanceMatrix(mol)
        # Canonical order: ordA[k] is pre-canonicalization atom idx of k-th heavy atom in s_A.
        # Reparsing s_A assigns RDKit atom indices 0..n-1 in emission order u, v, so dmat is
        # indexed by emission ranks (u, v) while hidden-state rows remain indexed by (au, av).
        ordA = p["ordA"]
        posA = p["posA"]
        # Store all causal atom pairs u < v in s_A
        for u in range(n):
            au = ordA[u]
            tu = posA[au]
            assert mol.GetAtomWithIdx(u).GetSymbol().lower() == p["elements"][au].lower()
            for v in range(u + 1, n):
                av = ordA[v]
                tv = posA[av]
                dg = int(round(dmat[u, v]))
                if dg <= 0 or dg > 100:
                    continue
                ds = tv - tu
                assert ds > 0
                pair_u_global.append(offset + au)
                pair_v_global.append(offset + av)
                pair_dG.append(dg)
                pair_dS.append(ds)
                pair_mol.append(m_idx)
        offset += n

    assert offset == total_atoms
    same_exact_tok_frac = float(np.mean(tokA_ids == tokB_ids))
    print(f"  total_atoms={total_atoms}, exact_same_token_frac={same_exact_tok_frac:.4f}, total_pairs={len(pair_dG)}", flush=True)

    # Encode sequences
    encA = [tok.encode(p["sA"]) for p in pairs]
    encB = [tok.encode(p["sB"]) for p in pairs]

    # Forward passes in batches
    attn_hooks = [attn_pattern(l) for l in range(L)]
    for b0 in range(0, len(pairs), args.batch):
        b1 = min(len(pairs), b0 + args.batch)
        batch_pairs = pairs[b0:b1]
        bsz = b1 - b0

        # ---- Traversal A (with attention capture for Part 2) ----
        wA = max(len(encA[i]) for i in range(b0, b1))
        xA = torch.full((bsz, wA), tok.pad_id, dtype=torch.long, device=args.device)
        for bi, i in enumerate(range(b0, b1)):
            seq = encA[i]
            xA[bi, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=args.device)
        hmA = HookManager().capture(*hook_names, *attn_hooks)
        model(xA, hooks=hmA)

        for li, hname in enumerate(hook_names):
            act = hmA[hname].float().cpu().numpy()  # (bsz, wA, D)
            for bi, m_idx in enumerate(range(b0, b1)):
                s, e = mol_slices[m_idx]
                pos = pairs[m_idx]["posA"]
                atom_act = act[bi, pos, :]
                HA_layers[li][s:e] = atom_act
                mol_mean_A[li][m_idx] = atom_act.mean(axis=0)

        # Accumulate attention stats on Traversal A
        pats = [hmA[attn_pattern(l)].float() for l in range(L)]  # list of (bsz, H, wA, wA)
        for bi, m_idx in enumerate(range(b0, b1)):
            p = pairs[m_idx]
            mol = Chem.MolFromSmiles(p["sA"])
            dmat = Chem.GetDistanceMatrix(mol)
            ordA = p["ordA"]
            posA = p["posA"]
            n = p["n_atoms"]

            q_idx, k_idx, dgs, dss = [], [], [], []
            for u in range(n):
                au = ordA[u]
                tu = posA[au]
                for v in range(u + 1, n):
                    av = ordA[v]
                    tv = posA[av]
                    dg = int(round(dmat[u, v]))
                    if dg <= 0 or dg > 100:
                        continue
                    q_idx.append(tv)
                    k_idx.append(tu)
                    dgs.append(dg)
                    dss.append(tv - tu)
            if not q_idx:
                continue
            q_t = torch.tensor(q_idx, dtype=torch.long, device=args.device)
            k_t = torch.tensor(k_idx, dtype=torch.long, device=args.device)
            # Stack (L, H, n_pairs_mol)
            a_mol = torch.stack([pats[l][bi, :, q_t, k_t] for l in range(L)], dim=0).cpu().numpy()
            dgs_arr = np.asarray(dgs, dtype=np.int32)
            dss_arr = np.asarray(dss, dtype=np.int32)

            # Distance-matched bonded enrichment for 3 <= d_S <= MAX_DS_MATCH
            in_range = (dss_arr >= 3) & (dss_arr <= MAX_DS_MATCH)
            if in_range.any():
                ds_r = dss_arr[in_range]
                dg_r = dgs_arr[in_range]
                a_r = a_mol[:, :, in_range]
                for ds_val in np.unique(ds_r):
                    m_b = (ds_r == ds_val) & (dg_r == 1)
                    m_nb = (ds_r == ds_val) & (dg_r > 1)
                    if m_b.any():
                        cnt_bond[ds_val] += int(m_b.sum())
                        attn_sum_bond[:, :, ds_val] += a_r[:, :, m_b].sum(axis=2)
                    if m_nb.any():
                        cnt_nonbond[ds_val] += int(m_nb.sum())
                        attn_sum_nonbond[:, :, ds_val] += a_r[:, :, m_nb].sum(axis=2)

            # Subsample ~60 pairs per molecule for partial Pearson/Spearman across all heads (d_S >= 2)
            ge2 = np.where(dss_arr >= 2)[0]
            if len(ge2) > 0:
                take = ge2 if len(ge2) <= 60 else sub_rng.choice(ge2, size=60, replace=False)
                attn_sub_list.append(a_mol[:, :, take].astype(np.float32))
                attn_sub_dG.append(dgs_arr[take])
                attn_sub_dS.append(dss_arr[take])

        del hmA, pats

        # ---- Traversal B ----
        wB = max(len(encB[i]) for i in range(b0, b1))
        xB = torch.full((bsz, wB), tok.pad_id, dtype=torch.long, device=args.device)
        for bi, i in enumerate(range(b0, b1)):
            seq = encB[i]
            xB[bi, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=args.device)
        hmB = HookManager().capture(*hook_names)
        model(xB, hooks=hmB)

        for li, hname in enumerate(hook_names):
            act = hmB[hname].float().cpu().numpy()
            for bi, m_idx in enumerate(range(b0, b1)):
                s, e = mol_slices[m_idx]
                pos = pairs[m_idx]["posB"]
                atom_act = act[bi, pos, :]
                HB_layers[li][s:e] = atom_act
                mol_mean_B[li][m_idx] = atom_act.mean(axis=0)
        del hmB

    print(f"  forward passes done in {time.time() - t0:.1f}s", flush=True)

    # =========================================================================
    # PART 1: Atom-Aligned Traversal Invariance across Depth
    # =========================================================================
    invariance_by_layer = {}
    for li, lname in enumerate(layer_names):
        HA = HA_layers[li]
        HB = HB_layers[li]
        raw_m = compute_invariance_metrics(
            HA, HB, mol_ids, tokA_ids, tokB_ids,
            pos_shift, prefix_jacc, in_ring, is_arom, mol_slices, center=False,
        )
        cen_m = compute_invariance_metrics(
            HA, HB, mol_ids, tokA_ids, tokB_ids,
            pos_shift, prefix_jacc, in_ring, is_arom, mol_slices, center=True,
        )
        cka_mol = linear_cka(mol_mean_A[li], mol_mean_B[li])
        cka_atom = linear_cka(HA, HB)
        invariance_by_layer[lname] = {
            "raw": raw_m,
            "centered": cen_m,
            "linear_cka_mol_mean": cka_mol,
            "linear_cka_atom_aligned": cka_atom,
        }
        print(
            f"  [inv] {lname:8s} cen_same={cen_m['same_atom_cos_mean']:.3f} "
            f"null1={cen_m['null1_within_mol_same_tok_mean']:.3f} "
            f"null2={cen_m['null2_between_mol_same_tok_mean']:.3f} "
            f"top1_tok={cen_m['top1_retrieval_same_tok']:.3f} "
            f"CKA_atom={cka_atom:.3f} CKA_mol={cka_mol:.3f}",
            flush=True,
        )

    # =========================================================================
    # PART 2A: Residual Geometry vs Molecular Graph Distance (controlling for d_S)
    # =========================================================================
    u_arr = np.asarray(pair_u_global, dtype=np.int32)
    v_arr = np.asarray(pair_v_global, dtype=np.int32)
    dG_arr = np.asarray(pair_dG, dtype=np.float64)
    dS_arr = np.asarray(pair_dS, dtype=np.float64)
    pmol_arr = np.asarray(pair_mol, dtype=np.int32)

    # Molecule-grouped 50/50 split
    split_rng = np.random.default_rng(args.seed + 2026)
    mol_perm = split_rng.permutation(len(pairs))
    tr_mols = set(mol_perm[: len(pairs) // 2].tolist())
    is_tr = np.array([m in tr_mols for m in pmol_arr], dtype=bool)
    is_te = ~is_tr

    # Subsample up to 40,000 train and 40,000 test pairs for vector Ridge probe
    tr_idx_all = np.where(is_tr)[0]
    te_idx_all = np.where(is_te)[0]
    tr_sub = tr_idx_all if len(tr_idx_all) <= 40000 else split_rng.choice(tr_idx_all, 40000, replace=False)
    te_sub = te_idx_all if len(te_idx_all) <= 40000 else split_rng.choice(te_idx_all, 40000, replace=False)

    Z_tr_sub = string_dist_features(dS_arr[tr_sub])
    Z_te_sub = string_dist_features(dS_arr[te_sub])
    Z_all_te = string_dist_features(dS_arr[te_idx_all])
    y_tr_sub = dG_arr[tr_sub, None]
    y_te_sub = dG_arr[te_sub]
    g_tr_sub = pmol_arr[tr_sub]

    # String-distance-only baseline
    pred_str_only, alpha_str = fit_ridge_group_cv(Z_tr_sub, y_tr_sub, g_tr_sub, Z_te_sub, device=args.device)
    r2_str_only = float(r2_score(y_te_sub, pred_str_only[:, 0]))
    pearson_str_only = float(np.corrcoef(y_te_sub, pred_str_only[:, 0])[0, 1])

    # Non-adjacent pairs mask (d_S >= 3) on test set for bonded-vs-nonbonded AUROC
    te_nonadj = te_idx_all[dS_arr[te_idx_all] >= 3]
    y_bond_nonadj = (dG_arr[te_nonadj] == 1).astype(np.int32)
    dS_nonadj_auroc = float(roc_auc_score(y_bond_nonadj, -dS_arr[te_nonadj])) if len(np.unique(y_bond_nonadj)) == 2 else float("nan")

    resid_graph_by_layer = {}
    for li, lname in enumerate(layer_names):
        HA = HA_layers[li]
        Hc = HA - HA.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(Hc, axis=1, keepdims=True).clip(min=1e-12)
        Hu = Hc / norms

        # Cosine & L2 on all test pairs
        cos_te = np.sum(Hu[u_arr[te_idx_all]] * Hu[v_arr[te_idx_all]], axis=1)
        l2_te = np.linalg.norm(Hc[u_arr[te_idx_all]] - Hc[v_arr[te_idx_all]], axis=1)

        p_pearson_cos = partial_pearson(cos_te, dG_arr[te_idx_all], Z_all_te[:, :2])
        p_spearman_cos = partial_spearman(cos_te, dG_arr[te_idx_all], Z_all_te[:, :2])
        p_pearson_l2 = partial_pearson(l2_te, dG_arr[te_idx_all], Z_all_te[:, :2])
        p_spearman_l2 = partial_spearman(l2_te, dG_arr[te_idx_all], Z_all_te[:, :2])

        # Scalar probe: [cos, l2] alone and + string distance
        cos_tr_s = np.sum(Hu[u_arr[tr_sub]] * Hu[v_arr[tr_sub]], axis=1)
        l2_tr_s = np.linalg.norm(Hc[u_arr[tr_sub]] - Hc[v_arr[tr_sub]], axis=1)
        cos_te_s = np.sum(Hu[u_arr[te_sub]] * Hu[v_arr[te_sub]], axis=1)
        l2_te_s = np.linalg.norm(Hc[u_arr[te_sub]] - Hc[v_arr[te_sub]], axis=1)

        X_scal_tr = np.column_stack([cos_tr_s, l2_tr_s])
        X_scal_te = np.column_stack([cos_te_s, l2_te_s])
        pred_scal, _ = fit_ridge_group_cv(X_scal_tr, y_tr_sub, g_tr_sub, X_scal_te, device=args.device)
        pred_scal_str, _ = fit_ridge_group_cv(
            np.column_stack([X_scal_tr, Z_tr_sub]), y_tr_sub, g_tr_sub,
            np.column_stack([X_scal_te, Z_te_sub]), device=args.device,
        )

        # Vector probe on |h_i - h_j| (centered)
        X_vec_tr = np.abs(Hc[u_arr[tr_sub]] - Hc[v_arr[tr_sub]])
        X_vec_te = np.abs(Hc[u_arr[te_sub]] - Hc[v_arr[te_sub]])
        pred_vec, alpha_vec = fit_ridge_group_cv(X_vec_tr, y_tr_sub, g_tr_sub, X_vec_te, device=args.device)
        pred_vec_str, alpha_vec_str = fit_ridge_group_cv(
            np.column_stack([X_vec_tr, Z_tr_sub]), y_tr_sub, g_tr_sub,
            np.column_stack([X_vec_te, Z_te_sub]), device=args.device,
        )

        # Non-adjacent bonded AUROC (d_S >= 3) from centered cosine similarity
        cos_nonadj = np.sum(Hu[u_arr[te_nonadj]] * Hu[v_arr[te_nonadj]], axis=1)
        auroc_cos_bond = float(roc_auc_score(y_bond_nonadj, cos_nonadj)) if len(np.unique(y_bond_nonadj)) == 2 else float("nan")

        r2_vec = float(r2_score(y_te_sub, pred_vec[:, 0]))
        r2_vec_str = float(r2_score(y_te_sub, pred_vec_str[:, 0]))
        resid_graph_by_layer[lname] = {
            "partial_pearson_cos_vs_dG_ctrl_dS": p_pearson_cos,
            "partial_spearman_cos_vs_dG_ctrl_dS": p_spearman_cos,
            "partial_pearson_l2_vs_dG_ctrl_dS": p_pearson_l2,
            "partial_spearman_l2_vs_dG_ctrl_dS": p_spearman_l2,
            "ridge_r2_scalar_only": float(r2_score(y_te_sub, pred_scal[:, 0])),
            "ridge_r2_scalar_plus_str": float(r2_score(y_te_sub, pred_scal_str[:, 0])),
            "ridge_r2_absdiff_only": r2_vec,
            "ridge_r2_absdiff_plus_str": r2_vec_str,
            "delta_r2_over_str_baseline": r2_vec_str - r2_str_only,
            "nonadj_bonded_auroc_cos": auroc_cos_bond,
            "alpha_absdiff": alpha_vec,
            "alpha_absdiff_str": alpha_vec_str,
        }
        print(
            f"  [dist] {lname:8s} p_r(cos,dG|dS)={p_pearson_cos:+.3f} "
            f"R2_vec={r2_vec:.3f} R2_vec+str={r2_vec_str:.3f} (str={r2_str_only:.3f}, dR2={r2_vec_str - r2_str_only:+.3f}) "
            f"bond_AUROC(dS>=3)={auroc_cos_bond:.3f}",
            flush=True,
        )

    # =========================================================================
    # PART 2B: Per-Head Attention Graph-Distance vs String-Distance Enrichment
    # =========================================================================
    attn_sub = np.concatenate(attn_sub_list, axis=2)  # (L, H, N_sub)
    sub_dG = np.concatenate(attn_sub_dG).astype(np.float64)
    sub_dS = np.concatenate(attn_sub_dS).astype(np.float64)
    Z_sub_attn = string_dist_features(sub_dS)[:, :2]  # [dS, log1p(dS)]

    # Exact token-distance-matched enrichment for d_S in 3..MAX_DS_MATCH
    valid_ds = (cnt_bond > 0) & (cnt_nonbond > 0)
    valid_ds[:3] = False
    w_ds = cnt_bond[valid_ds].astype(np.float64)
    w_total = float(w_ds.sum())

    head_attn_rows = []
    for l in range(L):
        for h in range(H):
            mean_b_k = attn_sum_bond[l, h, valid_ds] / cnt_bond[valid_ds]
            mean_nb_k = attn_sum_nonbond[l, h, valid_ds] / cnt_nonbond[valid_ds]
            w_mean_bond = float(np.sum(w_ds * mean_b_k) / max(w_total, 1.0))
            w_mean_nonbond = float(np.sum(w_ds * mean_nb_k) / max(w_total, 1.0))
            enrichment = w_mean_bond / max(w_mean_nonbond, 1e-12)

            a_vals = attn_sub[l, h].astype(np.float64)
            pp = partial_pearson(a_vals, sub_dG, Z_sub_attn)
            ps = partial_spearman(a_vals, sub_dG, Z_sub_attn)
            head_attn_rows.append({
                "layer": l,
                "head": h,
                "mean_attn_bonded_ds_ge3": w_mean_bond,
                "mean_attn_nonbonded_ds_matched": w_mean_nonbond,
                "bonded_enrichment_ratio": enrichment,
                "partial_pearson_attn_vs_dG_ctrl_dS": pp,
                "partial_spearman_attn_vs_dG_ctrl_dS": ps,
            })

    top_enrich = sorted(head_attn_rows, key=lambda r: r["bonded_enrichment_ratio"], reverse=True)[:5]
    print("  [attn] top-3 bonded enrichment heads (d_S >= 3, exact d_S-matched):", flush=True)
    for r in top_enrich[:3]:
        print(
            f"         L{r['layer']}H{r['head']}: ratio={r['bonded_enrichment_ratio']:.2f}x "
            f"(bond={r['mean_attn_bonded_ds_ge3']:.4f} vs matched_nb={r['mean_attn_nonbonded_ds_matched']:.4f}) "
            f"p_r(dG|dS)={r['partial_pearson_attn_vs_dG_ctrl_dS']:+.3f}",
            flush=True,
        )

    out_data = {
        "analysis": "graph_invariance_and_distance_geometry",
        "config": {
            "ckpt": args.ckpt,
            "data": smi_path,
            "n_mols": len(pairs),
            "total_atoms": total_atoms,
            "seed": args.seed,
            "pairs_sha256": pairs_sha,
            "exact_same_token_frac": same_exact_tok_frac,
            "model_cfg": {
                "n_layers": L, "n_heads": H, "d_model": D, "d_ff": model.cfg.d_ff,
            },
        },
        "part1_traversal_invariance": {
            "layers": layer_names,
            "by_layer": invariance_by_layer,
        },
        "part2_graph_vs_string_distance": {
            "n_train_mols": len(tr_mols),
            "n_test_mols": len(pairs) - len(tr_mols),
            "n_total_pairs": len(pair_dG),
            "string_distance_only_baseline": {
                "ridge_r2": r2_str_only,
                "pearson_r": pearson_str_only,
                "alpha": alpha_str,
                "nonadj_bonded_auroc_neg_dS": dS_nonadj_auroc,
                "n_nonadj_test_pairs": int(len(te_nonadj)),
                "n_nonadj_bonded_test_pairs": int(y_bond_nonadj.sum()),
            },
            "residual_by_layer": resid_graph_by_layer,
            "attention_enrichment_n_bonded_ds_ge3": int(w_total),
            "attention_enrichment_n_nonbonded_ds_ge3": int(cnt_nonbond[valid_ds].sum()),
            "attention_partial_corr_n_pairs": int(len(sub_dG)),
            "attention_by_head": head_attn_rows,
            "top5_bonded_enrichment_heads": top_enrich,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    if args.out.endswith(".json"):
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    else:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "graph_invariance.json")
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"  -> saved {out_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
