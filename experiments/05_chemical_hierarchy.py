#!/usr/bin/env python3
"""Workstream 2 — Script 2: 4-Level Chemical World Model & Multi-Head Ring Circuits.

Measures:
  Level 1 — Minimal Multi-Head Ring-Closure Circuit (`1 -> 2` activation patching):
    - On 512 single-ring relabeled pairs, ranks all L x H heads by individual patch recovery at predict_idx.
    - Simultaneously patches top-m heads (m = 1, 2, 3, 4, 6, 8, 12, all heads) vs equal-m random heads
      (10 draws each), plus all MLPs at predict_idx (and opener..predict).
    - Extracts 1 clean exemplar attention heatmap for a canonical drug-like molecule comparing the
      strongest pointer-mass head vs the most causally specific patch head.
  Level 2 & 3 — Nonlocal Topology & Aromaticity Linear Probes across Every Layer:
    - Molecule-grouped 50/50 split on 3,000 validation molecules; atom-aligned via RDKit `_smilesAtomOutputOrder`.
    - L2-regularized logistic probes at `embed` + `blocks.0..L-1.hook_resid_post` for:
        1. aliphatic_in_ring (C/N/O atoms: ring vs acyclic chain — identical token!)
        2. ring_size_5_vs_6 (ring atoms exclusively in 5- vs 6-membered ring — identical token!)
        3. fused_bicyclic (ring atoms in >=2 rings vs 1 ring — identical token!)
        4. final_heavy_degree (heavy-atom graph degree 1..4 — compared against current-token AND prefix-degree!)
        5. aliphatic_hybridization (uppercase C/N: SP vs SP2 vs SP3)
        6. aromaticity_cn (C/N aromaticity — explicitly shown alongside current-token baseline ~1.0)
    - Always reports current-token-identity baseline (accuracy & AUROC) alongside every task.
  Level 4 — Emergent Mid-Generation Global Physicochemical Properties:
    - On the same 3,000 validation molecules (1,500 train / 1,500 test), probes 8 RDKit properties
      (MolLogP, TPSA, QED, MolWt, NumHDonors, NumHAcceptors, NumAromaticRings, NumRings) via
      molecule-grouped CV Ridge regression at 25%, 50%, 75%, 100% (last SMILES token & EOS) across all layers.
    - Compares against: (a) prefix bag-of-tokens Ridge baseline, (b) partial-prefix RDKit property
      Ridge baseline, and (c) combined bag-of-tokens + partial-RDKit Ridge baseline.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import _element_of, is_atom_token
from molcircuits.model.hooks import HookManager, attn_pattern, resid_post
from molcircuits.model.transformer import ModelConfig, MolecularTransformer

RDLogger.DisableLog("rdApp.*")

PROPERTY_NAMES = [
    "MolLogP",
    "TPSA",
    "QED",
    "MolWt",
    "NumHDonors",
    "NumHAcceptors",
    "NumAromaticRings",
    "NumRings",
]

PREFIX_FRACTIONS = [
    ("25%", 0.25),
    ("50%", 0.50),
    ("75%", 0.75),
    ("100%_last", 1.00),
    ("100%_eos", 1.01),
]


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


# =============================================================================
# LEVEL 1: Multi-Head Ring-Closure Patching & Exemplar Attention Heatmap
# =============================================================================

def build_ring_pairs(smiles: list[str], tok: SmilesTokenizer, n_ctx: int = 128, max_pairs: int = 512) -> list[dict]:
    """Exact minimal-pair builder matching scripts/21_ring_patching.py (`1 -> 2`)."""
    out = []
    for s in smiles:
        try:
            toks = tokenize(s)
        except Exception:
            continue
        if toks.count("1") != 2 or "2" in toks:
            continue
        idx = [i for i, t in enumerate(toks) if t == "1"]
        open_i, close_i = idx
        predict_idx = close_i  # close_i - 1 + 1 (BOS offset)
        if predict_idx <= open_i + 1 or close_i + 1 >= n_ctx:
            continue
        if not all(t in tok.vocab for t in toks):
            continue
        corrupt = ["2" if t == "1" else t for t in toks]
        ce, co = tok.encode("".join(toks)), tok.encode("".join(corrupt))
        if len(ce) != len(co) or len(ce) > n_ctx:
            continue
        diff = [i for i in range(predict_idx + 1) if ce[i] != co[i]]
        if diff != [open_i + 1]:
            continue
        out.append({
            "smiles": s,
            "tokens": ["<bos>"] + toks + ["<eos>"],
            "clean_ids": ce,
            "corrupt_ids": co,
            "open_idx": open_i + 1,
            "predict_idx": predict_idx,
            "close_idx": close_i + 1,
        })
        if len(out) >= max_pairs:
            break
    return out


def pad_batch(rows: list[dict], key: str, pad_id: int, width: int) -> torch.Tensor:
    return torch.tensor([r[key] + [pad_id] * (width - len(r[key])) for r in rows], dtype=torch.long)


@torch.no_grad()
def run_level1_ring_circuit(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    raw_smiles: list[str],
    n_pairs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> dict:
    rng = np.random.default_rng(seed)
    shuffled = [raw_smiles[i] for i in rng.permutation(len(raw_smiles))[:200000]]
    pairs = build_ring_pairs(shuffled, tok, n_ctx=min(128, model.cfg.n_ctx), max_pairs=n_pairs)
    if len(pairs) < n_pairs:
        raise RuntimeError(f"Only found {len(pairs)}/{n_pairs} ring minimal pairs")

    tok_clean = tok.vocab["1"]
    tok_corr = tok.vocab["2"]
    width = max(len(p["clean_ids"]) for p in pairs)
    L, H = model.cfg.n_layers, model.cfg.n_heads

    def logit_diff(logits: torch.Tensor, rows: list[dict]) -> torch.Tensor:
        pi = torch.tensor([r["predict_idx"] for r in rows], device=logits.device)
        sel = logits[torch.arange(len(rows), device=logits.device), pi]
        return (sel[:, tok_clean] - sel[:, tok_corr]).float()

    # Pre-cache clean hook_z, hook_post, pointer_mass, and clean/corrupt contrasts per batch
    head_rec_sum = np.zeros((L, H), dtype=np.float64)
    ptr_mass_sum = np.zeros((L, H), dtype=np.float64)
    clean_diffs_all, corrupt_diffs_all = [], []
    n_eligible = 0

    # We do Pass 1 to rank individual heads & measure pointer mass, then Pass 2 for multi-head / MLP patching
    z_names = [f"blocks.{l}.attn.hook_z" for l in range(L)]
    pat_names = [attn_pattern(l) for l in range(L)]

    for s0 in range(0, len(pairs), batch_size):
        rows = pairs[s0:s0 + batch_size]
        bsz = len(rows)
        xc = pad_batch(rows, "clean_ids", tok.pad_id, width).to(device)
        xk = pad_batch(rows, "corrupt_ids", tok.pad_id, width).to(device)
        oi = torch.tensor([r["open_idx"] for r in rows], device=device)
        pi = torch.tensor([r["predict_idx"] for r in rows], device=device)
        ar = torch.arange(bsz, device=device)

        cap = HookManager().capture(*z_names, *pat_names)
        lc = model(xc, hooks=cap)
        d_clean = logit_diff(lc, rows)
        d_corr = logit_diff(model(xk), rows)
        denom = d_clean - d_corr
        ok = denom.abs() > 1e-6
        clean_diffs_all.append(d_clean.cpu().numpy())
        corrupt_diffs_all.append(d_corr.cpu().numpy())
        n_eligible += int(ok.sum())

        for l in range(L):
            pat = cap[attn_pattern(l)].float()  # (B, H, T, T)
            ptr_mass_sum[l] += pat[ar, :, pi, oi].sum(dim=0).cpu().numpy()

        for l in range(L):
            src = cap[z_names[l]].clone()  # (B, H, T, dh)
            for h in range(H):
                def edit_fn(v, _h=h, _s=src, _pi=pi, _ar=ar):
                    v = v.clone()
                    v[_ar, _h, _pi] = _s[_ar, _h, _pi]
                    return v
                lp = model(xk, hooks=HookManager().edit(z_names[l], edit_fn))
                rec = ((logit_diff(lp, rows) - d_corr) / denom)[ok]
                head_rec_sum[l, h] += float(rec.sum().item())

    head_rec = head_rec_sum / max(n_eligible, 1)
    ptr_mass = ptr_mass_sum / len(pairs)
    clean_arr = np.concatenate(clean_diffs_all)
    corrupt_arr = np.concatenate(corrupt_diffs_all)

    # Rank heads by individual patch recovery
    flat_heads = [(l, h, float(head_rec[l, h]), float(ptr_mass[l, h]))
                  for l in range(L) for h in range(H)]
    ranked = sorted(flat_heads, key=lambda x: x[2], reverse=True)
    top_ptr_head = max(flat_heads, key=lambda x: x[3])
    best_patch_head = ranked[0]

    n_total_heads = L * H
    m_values = [m for m in (1, 2, 3, 4, 6, 8, 12) if m <= n_total_heads]
    if n_total_heads not in m_values:
        m_values.append(n_total_heads)

    # Generate random-m head sets (10 draws per m < n_total_heads)
    rand_rng = np.random.default_rng(seed + 777)
    n_rand_draws = 10
    all_lh = [(l, h) for l in range(L) for h in range(H)]
    random_sets: dict[int, list[list[tuple[int, int]]]] = {}
    for m in m_values:
        if m < n_total_heads:
            draws = []
            for _ in range(n_rand_draws):
                idx = rand_rng.choice(n_total_heads, size=m, replace=False)
                draws.append([all_lh[i] for i in idx])
            random_sets[m] = draws

    # Pass 2: simultaneous multi-head & MLP patching
    top_m_sums = {m: 0.0 for m in m_values}
    rand_m_sums = {m: np.zeros(n_rand_draws, dtype=np.float64) for m in random_sets}
    mlp_layer_pred_sums = np.zeros(L, dtype=np.float64)
    mlp_pred_sum = 0.0
    mlp_span_sum = 0.0
    all_heads_mlps_pred_sum = 0.0

    mlp_names = [f"blocks.{l}.mlp.hook_post" for l in range(L)]

    def eval_head_set(xk, clean_z, head_set, ar, pi, d_corr, denom, ok, rows):
        by_layer: dict[int, list[int]] = {}
        for l, h in head_set:
            by_layer.setdefault(l, []).append(h)
        hm = HookManager()
        for l, hs in by_layer.items():
            src = clean_z[l]
            hs_t = torch.tensor(hs, dtype=torch.long, device=device)
            def edit_fn(v, _s=src, _hs=hs_t, _ar=ar, _pi=pi):
                v = v.clone()
                v[_ar[:, None], _hs[None, :], _pi[:, None]] = _s[_ar[:, None], _hs[None, :], _pi[:, None]]
                return v
            hm.edit(z_names[l], edit_fn)
        lp = model(xk, hooks=hm)
        rec = ((logit_diff(lp, rows) - d_corr) / denom)[ok]
        return float(rec.sum().item())

    for s0 in range(0, len(pairs), batch_size):
        rows = pairs[s0:s0 + batch_size]
        bsz = len(rows)
        xc = pad_batch(rows, "clean_ids", tok.pad_id, width).to(device)
        xk = pad_batch(rows, "corrupt_ids", tok.pad_id, width).to(device)
        pi = torch.tensor([r["predict_idx"] for r in rows], device=device)
        ar = torch.arange(bsz, device=device)
        span = torch.zeros(bsz, width, dtype=torch.bool, device=device)
        for bi, r in enumerate(rows):
            span[bi, r["open_idx"]:r["predict_idx"] + 1] = True

        cap = HookManager().capture(*z_names, *mlp_names)
        lc = model(xc, hooks=cap)
        clean_z = {l: cap[z_names[l]].clone() for l in range(L)}
        clean_mlp = {l: cap[mlp_names[l]].clone() for l in range(L)}
        d_clean = logit_diff(lc, rows)
        d_corr = logit_diff(model(xk), rows)
        denom = d_clean - d_corr
        ok = denom.abs() > 1e-6

        for m in m_values:
            head_set = [(x[0], x[1]) for x in ranked[:m]]
            top_m_sums[m] += eval_head_set(xk, clean_z, head_set, ar, pi, d_corr, denom, ok, rows)

        for m, draws in random_sets.items():
            for di, head_set in enumerate(draws):
                rand_m_sums[m][di] += eval_head_set(xk, clean_z, head_set, ar, pi, d_corr, denom, ok, rows)

        # Individual MLP layer at predict_idx
        for l in range(L):
            src = clean_mlp[l]
            def edit_ml(v, _s=src, _ar=ar, _pi=pi):
                v = v.clone()
                v[_ar, _pi] = _s[_ar, _pi]
                return v
            lp = model(xk, hooks=HookManager().edit(mlp_names[l], edit_ml))
            mlp_layer_pred_sums[l] += float((((logit_diff(lp, rows) - d_corr) / denom)[ok]).sum().item())

        # All MLPs at predict_idx
        hm_mlp = HookManager()
        for l in range(L):
            src = clean_mlp[l]
            def edit_m(v, _s=src, _ar=ar, _pi=pi):
                v = v.clone()
                v[_ar, _pi] = _s[_ar, _pi]
                return v
            hm_mlp.edit(mlp_names[l], edit_m)
        lp = model(xk, hooks=hm_mlp)
        mlp_pred_sum += float((((logit_diff(lp, rows) - d_corr) / denom)[ok]).sum().item())

        # All MLPs on opener..predict span
        hm_mlp_s = HookManager()
        for l in range(L):
            src = clean_mlp[l]
            def edit_ms(v, _s=src, _sp=span):
                return torch.where(_sp[:, :, None], _s, v)
            hm_mlp_s.edit(mlp_names[l], edit_ms)
        lp = model(xk, hooks=hm_mlp_s)
        mlp_span_sum += float((((logit_diff(lp, rows) - d_corr) / denom)[ok]).sum().item())

        # All heads + all MLPs at predict_idx
        hm_both = HookManager()
        for l in range(L):
            src_z = clean_z[l]
            src_m = clean_mlp[l]
            def edit_z_all(v, _s=src_z, _ar=ar, _pi=pi):
                v = v.clone()
                v[_ar, :, _pi] = _s[_ar, :, _pi]
                return v
            def edit_m_all(v, _s=src_m, _ar=ar, _pi=pi):
                v = v.clone()
                v[_ar, _pi] = _s[_ar, _pi]
                return v
            hm_both.edit(z_names[l], edit_z_all).edit(mlp_names[l], edit_m_all)
        lp = model(xk, hooks=hm_both)
        all_heads_mlps_pred_sum += float((((logit_diff(lp, rows) - d_corr) / denom)[ok]).sum().item())

    top_m_curve = []
    for m in m_values:
        entry = {
            "m": m,
            "label": "all_heads" if m == n_total_heads else f"top_{m}",
            "heads": [[x[0], x[1]] for x in ranked[:m]],
            "top_m_recovery": top_m_sums[m] / max(n_eligible, 1),
        }
        if m in rand_m_sums:
            r_vals = rand_m_sums[m] / max(n_eligible, 1)
            entry["random_m_mean"] = float(np.mean(r_vals))
            entry["random_m_std"] = float(np.std(r_vals))
            entry["random_m_draws"] = [float(x) for x in r_vals]
        top_m_curve.append(entry)

    # Extract 1 clean exemplar attention heatmap for a canonical drug-like molecule
    exemplar_idx = None
    for i, p in enumerate(pairs):
        n_t = len(p["tokens"])
        if 20 <= n_t <= 36 and "c" in p["tokens"] and clean_arr[i] > 2.0 and corrupt_arr[i] < -2.0:
            exemplar_idx = i
            break
    if exemplar_idx is None:
        exemplar_idx = int(np.argmax(clean_arr - corrupt_arr))
    ex = pairs[exemplar_idx]
    x_ex = torch.tensor([ex["clean_ids"]], dtype=torch.long, device=device)
    l_ptr, h_ptr = top_ptr_head[0], top_ptr_head[1]
    l_best, h_best = best_patch_head[0], best_patch_head[1]
    cap_ex = HookManager().capture(attn_pattern(l_ptr), attn_pattern(l_best))
    model(x_ex, hooks=cap_ex)
    mat_ptr = np.round(cap_ex[attn_pattern(l_ptr)][0, h_ptr].float().cpu().numpy(), 5).tolist()
    mat_best = np.round(cap_ex[attn_pattern(l_best)][0, h_best].float().cpu().numpy(), 5).tolist()

    res = {
        "n_pairs": len(pairs),
        "n_eligible": n_eligible,
        "mean_clean_logit_diff": float(clean_arr.mean()),
        "mean_corrupt_logit_diff": float(corrupt_arr.mean()),
        "head_patch_recovery": head_rec.tolist(),
        "head_pointer_mass": ptr_mass.tolist(),
        "top_pointer_head": {
            "layer": l_ptr, "head": h_ptr,
            "pointer_mass": top_ptr_head[3], "patch_recovery": top_ptr_head[2],
        },
        "best_patch_head": {
            "layer": l_best, "head": h_best,
            "pointer_mass": best_patch_head[3], "patch_recovery": best_patch_head[2],
        },
        "top_m_circuit_curve": top_m_curve,
        "mlp_by_layer_at_predict_recovery": (mlp_layer_pred_sums / max(n_eligible, 1)).tolist(),
        "all_mlps_at_predict_recovery": mlp_pred_sum / max(n_eligible, 1),
        "all_mlps_opener_to_predict_recovery": mlp_span_sum / max(n_eligible, 1),
        "all_heads_and_mlps_at_predict_recovery": all_heads_mlps_pred_sum / max(n_eligible, 1),
        "exemplar_heatmap": {
            "smiles": ex["smiles"],
            "tokens": ex["tokens"],
            "open_idx": ex["open_idx"],
            "predict_idx": ex["predict_idx"],
            "close_idx": ex["close_idx"],
            "clean_logit_diff": float(clean_arr[exemplar_idx]),
            "corrupt_logit_diff": float(corrupt_arr[exemplar_idx]),
            "strongest_pointer_head": {"layer": l_ptr, "head": h_ptr, "attention": mat_ptr},
            "best_patch_head": {"layer": l_best, "head": h_best, "attention": mat_best},
        },
    }
    print(
        f"  [L1] best_patch=L{l_best}H{h_best} ({best_patch_head[2]:+.3f}), "
        f"top_ptr=L{l_ptr}H{h_ptr} (ptr={top_ptr_head[3]:.3f}, rec={top_ptr_head[2]:+.3f}), "
        f"top4={next((x['top_m_recovery'] for x in top_m_curve if x['m'] == 4), float('nan')):+.3f}, "
        f"all_heads={top_m_curve[-1]['top_m_recovery']:+.3f}, "
        f"all_mlps@pred={res['all_mlps_at_predict_recovery']:+.3f}",
        flush=True,
    )
    return res


# =============================================================================
# Helpers for Levels 2, 3, and 4
# =============================================================================

def compute_prefix_heavy_degrees(toks: list[str]) -> list[int]:
    """Number of heavy-atom bonds already written to the left when each atom token is emitted."""
    prev_atom: int | None = None
    branch_stack: list[int | None] = []
    ring_open: dict[str, int] = {}
    deg: dict[int, int] = {}
    at_write: list[int] = []

    for i, tok in enumerate(toks):
        if tok == "(":
            branch_stack.append(prev_atom)
            continue
        if tok == ")":
            prev_atom = branch_stack.pop() if branch_stack else None
            continue
        if tok in ("-", "=", "#", "$", ":", "/", "\\"):
            continue
        if tok == ".":
            prev_atom = None
            continue
        if tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit()):
            lab = tok[1:] if tok.startswith("%") else tok
            if lab in ring_open:
                j = ring_open.pop(lab)
                if prev_atom is not None:
                    deg[prev_atom] = deg.get(prev_atom, 0) + 1
                    deg[j] = deg.get(j, 0) + 1
            elif prev_atom is not None:
                ring_open[lab] = prev_atom
            continue
        if is_atom_token(tok):
            deg[i] = 0
            if prev_atom is not None:
                deg[i] += 1
                deg[prev_atom] = deg.get(prev_atom, 0) + 1
            at_write.append(deg[i])
            prev_atom = i
    return at_write


def compute_rdkit_properties(mol: Chem.Mol) -> np.ndarray:
    return np.array([
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        QED.qed(mol),
        Descriptors.MolWt(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
    ], dtype=np.float64)


def repair_prefix_to_mol(toks_prefix: list[str]) -> Chem.Mol | None:
    """Heuristically repair a SMILES prefix into a valid partial RDKit molecule."""
    toks = list(toks_prefix)
    # Strip trailing bond symbols or open parens
    while toks and toks[-1] in ("-", "=", "#", ":", "/", "\\", "(", "."):
        toks.pop()
    if not toks:
        return None
    # Remove any unmatched ring closure digits
    ring_counts: dict[str, int] = {}
    for t in toks:
        if t.isdigit() or (t.startswith("%") and t[1:].isdigit()):
            ring_counts[t] = ring_counts.get(t, 0) + 1
    cleaned = [t for t in toks if not ((t.isdigit() or (t.startswith("%") and t[1:].isdigit())) and ring_counts[t] % 2 == 1)]
    # Remove empty parens created by stripping
    s = "".join(cleaned)
    while "()" in s:
        s = s.replace("()", "")
    # Close open parentheses
    depth = 0
    out_chars = []
    for ch in s:
        if ch == "(":
            depth += 1
            out_chars.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
                out_chars.append(ch)
        else:
            out_chars.append(ch)
    s_rep = "".join(out_chars) + (")" * depth)
    # Strip trailing bond chars again if any
    s_rep = re.sub(r"[-=#:/\\\.]+$", "", s_rep)
    if not s_rep:
        return None
    mol = Chem.MolFromSmiles(s_rep)
    if mol is not None:
        return mol
    # Fallback for unclosed aromatic fragments: uppercase organic aromatic letters
    s_aliph = re.sub(r"(?<!\[)[cnos](?!\])", lambda m: m.group(0).upper(), s_rep)
    mol = Chem.MolFromSmiles(s_aliph)
    return mol


def fit_ridge_multi_group_cv(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    groups_tr: np.ndarray,
    X_te: np.ndarray,
    alphas: np.ndarray | None = None,
    n_folds: int = 5,
) -> tuple[np.ndarray, list[float]]:
    """Multi-output Ridge regression with per-target molecule-grouped CV alpha selection."""
    if alphas is None:
        alphas = np.logspace(-2, 5, 15)
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xs_tr = ((X_tr - mu) / sd).astype(np.float64)
    Xs_te = ((X_te - mu) / sd).astype(np.float64)
    y_mu = Y_tr.mean(axis=0, keepdims=True)
    Yc_tr = (Y_tr - y_mu).astype(np.float64)

    uniq_g = np.unique(groups_tr)
    fold_assign = {g: i % n_folds for i, g in enumerate(uniq_g)}
    fold_ids = np.array([fold_assign[g] for g in groups_tr])

    D = Xs_tr.shape[1]
    K = Y_tr.shape[1]
    eye = np.eye(D, dtype=np.float64)

    G_folds, B_folds = [], []
    for f in range(n_folds):
        m = fold_ids == f
        Xf = Xs_tr[m]
        Yf = Yc_tr[m]
        G_folds.append(Xf.T @ Xf)
        B_folds.append(Xf.T @ Yf)
    G_all = sum(G_folds)
    B_all = sum(B_folds)

    cv_se = np.zeros((len(alphas), K), dtype=np.float64)
    for ai, alpha in enumerate(alphas):
        for f in range(n_folds):
            W = np.linalg.solve(G_all - G_folds[f] + alpha * eye, B_all - B_folds[f])
            m = fold_ids == f
            diff = Yc_tr[m] - Xs_tr[m] @ W
            cv_se[ai] += np.sum(diff * diff, axis=0)

    best_idx = np.argmin(cv_se, axis=0)
    preds = np.empty((len(X_te), K), dtype=np.float64)
    chosen_alphas = []
    # Group targets with identical best alpha to minimize solves
    by_alpha: dict[int, list[int]] = {}
    for k, ai in enumerate(best_idx):
        by_alpha.setdefault(int(ai), []).append(k)
        chosen_alphas.append(float(alphas[ai]))
    for ai, ks in by_alpha.items():
        W = np.linalg.solve(G_all + float(alphas[ai]) * eye, B_all[:, ks])
        preds[:, ks] = Xs_te @ W + y_mu[:, ks]
    return preds, chosen_alphas


def eval_logistic_probe(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    is_binary: bool,
    C: float = 0.5,
    device: str = "cpu",
) -> dict:
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr)
    Xs_te = scaler.transform(X_te)
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device)
        Xt = torch.from_numpy(Xs_tr.astype(np.float64)).to(dev)
        Xtet = torch.from_numpy(Xs_te.astype(np.float64)).to(dev)
        N, D = Xt.shape
        reg = 1.0 / (2.0 * C * N)
        with torch.enable_grad():
            if is_binary:
                yt = torch.from_numpy(y_tr.astype(np.float64)).to(dev)
                w = torch.zeros(D, dtype=torch.float64, device=dev, requires_grad=True)
                b = torch.zeros(1, dtype=torch.float64, device=dev, requires_grad=True)
                opt = torch.optim.LBFGS(
                    [w, b], lr=1.0, max_iter=200,
                    tolerance_grad=1e-7, tolerance_change=1e-10,
                    line_search_fn="strong_wolfe",
                )
                def closure():
                    opt.zero_grad()
                    logits = Xt @ w + b
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt) + reg * (w * w).sum()
                    loss.backward()
                    return loss
                opt.step(closure)
                with torch.no_grad():
                    p1 = torch.sigmoid(Xtet @ w + b).cpu().numpy()
                pred = (p1 >= 0.5).astype(np.int64)
                acc = float(accuracy_score(y_te, pred))
                bacc = float(balanced_accuracy_score(y_te, pred))
                auroc = float(roc_auc_score(y_te, p1))
                return {"accuracy": acc, "balanced_accuracy": bacc, "auroc": auroc}
            else:
                n_cls = int(np.max(y_tr)) + 1
                yt = torch.from_numpy(y_tr.astype(np.int64)).to(dev)
                w = torch.zeros((D, n_cls), dtype=torch.float64, device=dev, requires_grad=True)
                b = torch.zeros(n_cls, dtype=torch.float64, device=dev, requires_grad=True)
                opt = torch.optim.LBFGS(
                    [w, b], lr=1.0, max_iter=200,
                    tolerance_grad=1e-7, tolerance_change=1e-10,
                    line_search_fn="strong_wolfe",
                )
                def closure():
                    opt.zero_grad()
                    logits = Xt @ w + b
                    loss = torch.nn.functional.cross_entropy(logits, yt) + reg * (w * w).sum()
                    loss.backward()
                    return loss
                opt.step(closure)
                with torch.no_grad():
                    prob = torch.softmax(Xtet @ w + b, dim=-1).cpu().numpy()
                pred = np.argmax(prob, axis=1)
                acc = float(accuracy_score(y_te, pred))
                bacc = float(balanced_accuracy_score(y_te, pred))
                auroc = float(roc_auc_score(y_te, prob, multi_class="ovr", average="macro"))
                return {"accuracy": acc, "balanced_accuracy": bacc, "auroc": auroc}

    clf = LogisticRegression(C=C, max_iter=500, solver="lbfgs")
    clf.fit(Xs_tr, y_tr)
    pred = clf.predict(Xs_te)
    prob = clf.predict_proba(Xs_te)
    acc = float(accuracy_score(y_te, pred))
    bacc = float(balanced_accuracy_score(y_te, pred))
    if is_binary:
        auroc = float(roc_auc_score(y_te, prob[:, 1]))
    else:
        auroc = float(roc_auc_score(y_te, prob, multi_class="ovr", average="macro"))
    return {"accuracy": acc, "balanced_accuracy": bacc, "auroc": auroc}


def eval_categorical_baseline(
    cat_tr: np.ndarray,
    y_tr: np.ndarray,
    cat_te: np.ndarray,
    y_te: np.ndarray,
    is_binary: bool,
) -> dict:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_tr = enc.fit_transform(cat_tr[:, None])
    X_te = enc.transform(cat_te[:, None])
    clf = LogisticRegression(C=10.0, max_iter=300, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    prob = clf.predict_proba(X_te)
    acc = float(accuracy_score(y_te, pred))
    bacc = float(balanced_accuracy_score(y_te, pred))
    if is_binary:
        auroc = float(roc_auc_score(y_te, prob[:, 1]))
    else:
        auroc = float(roc_auc_score(y_te, prob, multi_class="ovr", average="macro"))
    return {"accuracy": acc, "balanced_accuracy": bacc, "auroc": auroc}


@torch.no_grad()
def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="Path to valid.smi or data directory")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True, help="Output directory or .json file")
    ap.add_argument("--n-mols", type=int, default=3000)
    ap.add_argument("--n-pairs", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    smi_path, tok_path = resolve_data_and_tokenizer(args.data, args.tokenizer)
    tok = SmilesTokenizer.load(tok_path)
    model = load_model(args.ckpt, args.device)
    L, H, D = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model
    layer_names = ["embed"] + [f"layer_{l}" for l in range(L)]
    hook_names = ["hook_embed"] + [resid_post(l) for l in range(L)]

    print(f"[41_chemical_hierarchy] ckpt={args.ckpt} (L={L}, H={H}, D={D})", flush=True)
    with open(smi_path) as fh:
        raw_smiles = [line.strip().split()[0] for line in fh if line.strip()]

    # =========================================================================
    # LEVEL 1: Multi-Head Ring-Closure Circuit
    # =========================================================================
    level1 = run_level1_ring_circuit(
        model, tok, raw_smiles, args.n_pairs, args.batch, args.seed, args.device
    )

    # =========================================================================
    # Sample 3,000 Molecules for Levels 2, 3, 4 (deterministic, molecule-grouped 50/50)
    # =========================================================================
    rng = np.random.default_rng(args.seed + 4100)
    perm = rng.permutation(len(raw_smiles))
    max_toks = min(128, model.cfg.n_ctx)
    vocab = tok.vocab

    mols_meta = []
    seen_canon = set()
    for idx in perm:
        s = raw_smiles[idx]
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() < 6:
            continue
        sC = Chem.MolToSmiles(mol, canonical=True)
        if sC in seen_canon:
            continue
        try:
            ordC = list(ast.literal_eval(mol.GetProp("_smilesAtomOutputOrder")))
            toks = tokenize(sC)
        except Exception:
            continue
        if len(toks) + 2 > max_toks or not all(t in vocab for t in toks):
            continue
        at_pos = [i for i, t in enumerate(toks) if is_atom_token(t)]
        n_at = mol.GetNumAtoms()
        if len(at_pos) != n_at or len(ordC) != n_at:
            continue
        # Verify 100% element alignment
        for k, a in enumerate(ordC):
            el = _element_of(toks[at_pos[k]])
            assert el is not None and el.lower() == mol.GetAtomWithIdx(a).GetSymbol().lower()

        pref_deg = compute_prefix_heavy_degrees(toks)
        assert len(pref_deg) == n_at

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        # Map atom_idx -> list of ring sizes containing it
        ring_sizes_by_atom: list[list[int]] = [[] for _ in range(n_at)]
        for ring in atom_rings:
            rlen = len(ring)
            for a in ring:
                ring_sizes_by_atom[a].append(rlen)

        props = compute_rdkit_properties(mol)
        if not np.all(np.isfinite(props)):
            continue

        # Collect atom-level attributes in SMILES emission order (k = 0..n_at-1)
        atom_tok_ids = []
        atom_tok_strs = []
        atom_pos_bos = []
        atom_in_ring = []
        atom_is_arom = []
        atom_ring_size_5v6 = []
        atom_fused = []
        atom_deg = []
        atom_hyb = []
        atom_sym = []

        hyb_map = {
            Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
        }

        for k, a in enumerate(ordC):
            at = mol.GetAtomWithIdx(a)
            tstr = toks[at_pos[k]]
            atom_tok_strs.append(tstr)
            atom_tok_ids.append(vocab[tstr])
            atom_pos_bos.append(at_pos[k] + 1)
            atom_in_ring.append(int(at.IsInRing()))
            atom_is_arom.append(int(at.GetIsAromatic()))
            rsizes = set(ring_sizes_by_atom[a])
            if rsizes == {5}:
                atom_ring_size_5v6.append(0)
            elif rsizes == {6}:
                atom_ring_size_5v6.append(1)
            else:
                atom_ring_size_5v6.append(-1)
            atom_fused.append(int(ring_info.NumAtomRings(a) >= 2) if at.IsInRing() else -1)
            d = at.GetDegree()
            atom_deg.append(d if 1 <= d <= 4 else -1)
            atom_hyb.append(hyb_map.get(at.GetHybridization(), -1))
            atom_sym.append(at.GetSymbol())

        seen_canon.add(sC)
        mols_meta.append({
            "smiles": sC,
            "toks": toks,
            "ids": tok.encode(sC),
            "n_atoms": n_at,
            "atom_pos_bos": atom_pos_bos,
            "atom_tok_ids": atom_tok_ids,
            "atom_tok_strs": atom_tok_strs,
            "atom_in_ring": atom_in_ring,
            "atom_is_arom": atom_is_arom,
            "atom_ring_size_5v6": atom_ring_size_5v6,
            "atom_fused": atom_fused,
            "atom_deg": atom_deg,
            "atom_pref_deg": pref_deg,
            "atom_hyb": atom_hyb,
            "atom_sym": atom_sym,
            "props": props,
        })
        if len(mols_meta) >= args.n_mols:
            break

    if len(mols_meta) < args.n_mols:
        raise RuntimeError(f"Only found {len(mols_meta)}/{args.n_mols} molecules")

    mol_sha = hashlib.sha256("\n".join(m["smiles"] for m in mols_meta).encode()).hexdigest()
    n_mols = len(mols_meta)
    n_train = n_mols // 2
    mol_is_train = np.zeros(n_mols, dtype=bool)
    mol_is_train[:n_train] = True  # mols_meta already came from deterministic random permutation!

    total_atoms = sum(m["n_atoms"] for m in mols_meta)
    print(f"  [L2-L4] sampled {n_mols} mols ({total_atoms} atoms, sha={mol_sha[:12]})", flush=True)

    # Allocate activation buffers
    H_atoms = [np.empty((total_atoms, D), dtype=np.float32) for _ in range(L + 1)]
    n_fracs = len(PREFIX_FRACTIONS)
    H_prefix = [np.empty((n_mols, n_fracs, D), dtype=np.float32) for _ in range(L + 1)]
    H_prefix_mean = [np.empty((n_mols, n_fracs, D), dtype=np.float32) for _ in range(L + 1)]

    # Build prefix positions & Level 4 baselines
    V = tok.vocab_size
    bot_feats = np.zeros((n_fracs, n_mols, V + 1), dtype=np.float64)
    rdkit_pref_feats = np.zeros((n_fracs, n_mols, len(PROPERTY_NAMES) + 2), dtype=np.float64)
    rdkit_parse_ok = np.zeros((n_fracs, n_mols), dtype=bool)
    Y_props = np.stack([m["props"] for m in mols_meta], axis=0)  # (n_mols, 8)
    pref_pos_mat = np.zeros((n_mols, n_fracs), dtype=np.int32)

    for mi, m in enumerate(mols_meta):
        N_tok = len(m["toks"])
        ids = m["ids"]
        for fi, (fname, frac) in enumerate(PREFIX_FRACTIONS):
            if frac > 1.0:
                pos = N_tok + 1  # <eos>
                k_tok = N_tok
            else:
                k_tok = max(1, min(N_tok, int(round(frac * N_tok))))
                pos = k_tok  # 1-based BOS offset: ids[1..k_tok] are first k_tok SMILES tokens
            pref_pos_mat[mi, fi] = pos
            # Bag-of-tokens over ids[1 : k_tok + 1]
            for tid in ids[1:k_tok + 1]:
                bot_feats[fi, mi, tid] += 1.0
            bot_feats[fi, mi, V] = float(k_tok)

            # Partial-prefix RDKit descriptors
            pmol = repair_prefix_to_mol(m["toks"][:k_tok])
            if pmol is not None:
                pvals = compute_rdkit_properties(pmol)
                if np.all(np.isfinite(pvals)):
                    rdkit_pref_feats[fi, mi, :len(PROPERTY_NAMES)] = pvals
                    rdkit_pref_feats[fi, mi, len(PROPERTY_NAMES)] = 1.0
                    rdkit_parse_ok[fi, mi] = True
            rdkit_pref_feats[fi, mi, len(PROPERTY_NAMES) + 1] = float(k_tok)

    # Impute failed partial RDKit parses with train mean (no leakage!)
    for fi in range(n_fracs):
        tr_ok = mol_is_train & rdkit_parse_ok[fi]
        col_means = rdkit_pref_feats[fi, tr_ok, :len(PROPERTY_NAMES)].mean(axis=0) if tr_ok.any() else np.zeros(len(PROPERTY_NAMES))
        bad = ~rdkit_parse_ok[fi]
        if bad.any():
            rdkit_pref_feats[fi, bad, :len(PROPERTY_NAMES)] = col_means

    # Single batched forward pass extracting BOTH atom activations (L2-3) and prefix activations (L4)
    atom_offsets = np.cumsum([0] + [m["n_atoms"] for m in mols_meta])
    for b0 in range(0, n_mols, args.batch):
        b1 = min(n_mols, b0 + args.batch)
        bsz = b1 - b0
        width = max(len(mols_meta[i]["ids"]) for i in range(b0, b1))
        x = torch.full((bsz, width), tok.pad_id, dtype=torch.long, device=args.device)
        for bi, mi in enumerate(range(b0, b1)):
            seq = mols_meta[mi]["ids"]
            x[bi, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=args.device)

        hm = HookManager().capture(*hook_names)
        model(x, hooks=hm)
        for li, hname in enumerate(hook_names):
            act = hm[hname].float().cpu().numpy()  # (bsz, width, D)
            for bi, mi in enumerate(range(b0, b1)):
                s, e = int(atom_offsets[mi]), int(atom_offsets[mi + 1])
                H_atoms[li][s:e] = act[bi, mols_meta[mi]["atom_pos_bos"], :]
                H_prefix[li][mi] = act[bi, pref_pos_mat[mi], :]
                for fi in range(n_fracs):
                    p_end = int(pref_pos_mat[mi, fi])
                    H_prefix_mean[li][mi, fi] = act[bi, 1:p_end + 1, :].mean(axis=0)
        del hm

    print(f"  [L2-L4] forward pass done in {time.time() - t0:.1f}s", flush=True)

    # =========================================================================
    # LEVEL 2 & 3: Nonlocal Topology & Aromaticity Linear Probes
    # =========================================================================
    atom_mol_id = np.concatenate([np.full(m["n_atoms"], mi, dtype=np.int32) for mi, m in enumerate(mols_meta)])
    atom_is_tr = mol_is_train[atom_mol_id]
    atom_tok_id = np.concatenate([m["atom_tok_ids"] for m in mols_meta])
    atom_tok_str = np.concatenate([m["atom_tok_strs"] for m in mols_meta])
    atom_in_ring = np.concatenate([m["atom_in_ring"] for m in mols_meta])
    atom_is_arom = np.concatenate([m["atom_is_arom"] for m in mols_meta])
    atom_r56 = np.concatenate([m["atom_ring_size_5v6"] for m in mols_meta])
    atom_fused = np.concatenate([m["atom_fused"] for m in mols_meta])
    atom_deg = np.concatenate([m["atom_deg"] for m in mols_meta])
    atom_pref_deg = np.concatenate([m["atom_pref_deg"] for m in mols_meta])
    atom_hyb = np.concatenate([m["atom_hyb"] for m in mols_meta])
    atom_sym = np.concatenate([m["atom_sym"] for m in mols_meta])

    tasks = {
        "aliphatic_in_ring": {
            "mask": np.isin(atom_tok_str, ["C", "N", "O"]),
            "y": atom_in_ring,
            "is_binary": True,
            "desc": "Aliphatic C/N/O atoms: in-ring (1) vs acyclic chain (0)",
        },
        "ring_size_5_vs_6": {
            "mask": atom_r56 >= 0,
            "y": atom_r56,
            "is_binary": True,
            "desc": "Ring atoms exclusively in 5-membered (0) vs 6-membered (1) ring",
        },
        "fused_bicyclic": {
            "mask": atom_fused >= 0,
            "y": atom_fused,
            "is_binary": True,
            "desc": "Ring atoms in >=2 rings (1, fused/spiro/bridged) vs 1 ring (0)",
        },
        "final_heavy_degree": {
            "mask": atom_deg >= 1,
            "y": atom_deg - 1,  # 0..3 for degree 1..4
            "is_binary": False,
            "desc": "Final heavy-atom graph degree (1, 2, 3, 4)",
        },
        "aliphatic_hybridization": {
            "mask": np.isin(atom_tok_str, ["C", "N"]) & (atom_hyb >= 0),
            "y": atom_hyb,
            "is_binary": False,
            "desc": "Uppercase C/N hybridization (SP=0, SP2=1, SP3=2)",
        },
        "aromaticity_cn": {
            "mask": np.isin(atom_sym, ["C", "N"]),
            "y": atom_is_arom,
            "is_binary": True,
            "desc": "C/N aromaticity (control: SMILES case encodes aromaticity directly)",
        },
    }

    level23_results = {}
    for tname, tcfg in tasks.items():
        mask = tcfg["mask"]
        tr_m = mask & atom_is_tr
        te_m = mask & (~atom_is_tr)
        y_tr = tcfg["y"][tr_m]
        y_te = tcfg["y"][te_m]
        tok_base = eval_categorical_baseline(
            atom_tok_id[tr_m], y_tr, atom_tok_id[te_m], y_te, tcfg["is_binary"]
        )
        extra_baselines = {}
        if tname == "final_heavy_degree":
            extra_baselines["prefix_degree_baseline"] = eval_categorical_baseline(
                atom_pref_deg[tr_m], y_tr, atom_pref_deg[te_m], y_te, is_binary=False
            )
            # Combined token + prefix_degree baseline
            comb_tr = atom_tok_id[tr_m] * 10 + atom_pref_deg[tr_m]
            comb_te = atom_tok_id[te_m] * 10 + atom_pref_deg[te_m]
            extra_baselines["token_plus_prefix_degree_baseline"] = eval_categorical_baseline(
                comb_tr, y_tr, comb_te, y_te, is_binary=False
            )

        by_layer = {}
        best_auroc = -1.0
        best_layer = ""
        for li, lname in enumerate(layer_names):
            m_res = eval_logistic_probe(
                H_atoms[li][tr_m], y_tr, H_atoms[li][te_m], y_te, tcfg["is_binary"], C=0.5, device=args.device
            )
            by_layer[lname] = m_res
            if m_res["auroc"] > best_auroc:
                best_auroc = m_res["auroc"]
                best_layer = lname

        level23_results[tname] = {
            "description": tcfg["desc"],
            "n_train_atoms": int(tr_m.sum()),
            "n_test_atoms": int(te_m.sum()),
            "class_balance_test": {str(int(c)): float(np.mean(y_te == c)) for c in np.unique(y_te)},
            "current_token_baseline": tok_base,
            **extra_baselines,
            "best_layer": best_layer,
            "best_metrics": by_layer[best_layer],
            "gain_over_token_acc": by_layer[best_layer]["accuracy"] - tok_base["accuracy"],
            "gain_over_token_auroc": by_layer[best_layer]["auroc"] - tok_base["auroc"],
            "by_layer": by_layer,
        }
        print(
            f"  [L2/3] {tname:24s} tok_base=(acc={tok_base['accuracy']:.3f}, auc={tok_base['auroc']:.3f}) "
            f"best@{best_layer:7s}=(acc={by_layer[best_layer]['accuracy']:.3f}, auc={by_layer[best_layer]['auroc']:.3f}) "
            f"dAUC={by_layer[best_layer]['auroc'] - tok_base['auroc']:+.3f}",
            flush=True,
        )

    # =========================================================================
    # LEVEL 4: Emergent Mid-Generation Global Physicochemical Properties
    # =========================================================================
    tr_g = np.arange(n_train, dtype=np.int32)
    Y_tr = Y_props[mol_is_train]
    Y_te = Y_props[~mol_is_train]

    def score_props(preds: np.ndarray) -> dict:
        out_p = {}
        r2s, rs = [], []
        for k, pname in enumerate(PROPERTY_NAMES):
            r2 = float(r2_score(Y_te[:, k], preds[:, k]))
            pr = float(np.corrcoef(Y_te[:, k], preds[:, k])[0, 1]) if np.std(preds[:, k]) > 1e-12 else 0.0
            out_p[pname] = {"r2": r2, "pearson_r": pr}
            r2s.append(r2)
            rs.append(pr)
        return {
            "mean_r2": float(np.mean(r2s)),
            "mean_pearson_r": float(np.mean(rs)),
            "by_property": out_p,
        }

    level4_by_fraction = {}
    for fi, (fname, frac) in enumerate(PREFIX_FRACTIONS):
        # Baselines
        X_bot_tr, X_bot_te = bot_feats[fi, mol_is_train], bot_feats[fi, ~mol_is_train]
        X_rdk_tr, X_rdk_te = rdkit_pref_feats[fi, mol_is_train], rdkit_pref_feats[fi, ~mol_is_train]
        X_cmb_tr = np.hstack([X_bot_tr, X_rdk_tr])
        X_cmb_te = np.hstack([X_bot_te, X_rdk_te])

        pred_bot, _ = fit_ridge_multi_group_cv(X_bot_tr, Y_tr, tr_g, X_bot_te)
        pred_rdk, _ = fit_ridge_multi_group_cv(X_rdk_tr, Y_tr, tr_g, X_rdk_te)
        pred_cmb, _ = fit_ridge_multi_group_cv(X_cmb_tr, Y_tr, tr_g, X_cmb_te)

        s_bot = score_props(pred_bot)
        s_rdk = score_props(pred_rdk)
        s_cmb = score_props(pred_cmb)

        by_layer = {}
        by_layer_mean = {}
        best_l = ""
        best_mean_r2 = -1e9
        best_l_pool = ""
        best_pool_r2 = -1e9
        for li, lname in enumerate(layer_names):
            X_tr = H_prefix[li][mol_is_train, fi, :]
            X_te = H_prefix[li][~mol_is_train, fi, :]
            pred_l, _ = fit_ridge_multi_group_cv(X_tr, Y_tr, tr_g, X_te)
            sl = score_props(pred_l)
            by_layer[lname] = sl
            if sl["mean_r2"] > best_mean_r2:
                best_mean_r2 = sl["mean_r2"]
                best_l = lname

            Xm_tr = H_prefix_mean[li][mol_is_train, fi, :]
            Xm_te = H_prefix_mean[li][~mol_is_train, fi, :]
            pred_m, _ = fit_ridge_multi_group_cv(Xm_tr, Y_tr, tr_g, Xm_te)
            sm = score_props(pred_m)
            by_layer_mean[lname] = sm
            if sm["mean_r2"] > best_pool_r2:
                best_pool_r2 = sm["mean_r2"]
                best_l_pool = lname

        level4_by_fraction[fname] = {
            "fraction": frac,
            "partial_rdkit_parse_rate": float(rdkit_parse_ok[fi].mean()),
            "baselines": {
                "bag_of_tokens_ridge": s_bot,
                "partial_rdkit_ridge": s_rdk,
                "combined_bot_rdkit_ridge": s_cmb,
            },
            "best_layer": best_l,
            "best_mean_r2": best_mean_r2,
            "delta_r2_vs_bag_of_tokens": best_mean_r2 - s_bot["mean_r2"],
            "delta_r2_vs_partial_rdkit": best_mean_r2 - s_rdk["mean_r2"],
            "delta_r2_vs_combined": best_mean_r2 - s_cmb["mean_r2"],
            "by_layer": by_layer,
            "prefix_mean_best_layer": best_l_pool,
            "prefix_mean_best_r2": best_pool_r2,
            "prefix_mean_delta_r2_vs_bag_of_tokens": best_pool_r2 - s_bot["mean_r2"],
            "prefix_mean_delta_r2_vs_partial_rdkit": best_pool_r2 - s_rdk["mean_r2"],
            "prefix_mean_delta_r2_vs_combined": best_pool_r2 - s_cmb["mean_r2"],
            "prefix_mean_by_layer": by_layer_mean,
        }
        print(
            f"  [L4] {fname:9s} bot={s_bot['mean_r2']:.3f} rdk={s_rdk['mean_r2']:.3f} "
            f"cmb={s_cmb['mean_r2']:.3f} | tok@{best_l:7s}={best_mean_r2:.3f} | "
            f"pool@{best_l_pool:7s}={best_pool_r2:.3f} (dBoT={best_pool_r2 - s_bot['mean_r2']:+.3f})",
            flush=True,
        )

    out_data = {
        "analysis": "chemical_hierarchy_4level_suite",
        "config": {
            "ckpt": args.ckpt,
            "data": smi_path,
            "n_mols": n_mols,
            "n_train_mols": n_train,
            "n_test_mols": n_mols - n_train,
            "total_atoms": total_atoms,
            "mol_sample_sha256": mol_sha,
            "seed": args.seed,
            "model_cfg": {
                "n_layers": L, "n_heads": H, "d_model": D, "d_ff": model.cfg.d_ff,
            },
        },
        "level1_ring_circuit": level1,
        "level2_3_topology_probes": {
            "layers": layer_names,
            "tasks": level23_results,
        },
        "level4_global_properties": {
            "layers": layer_names,
            "properties": PROPERTY_NAMES,
            "by_prefix_fraction": level4_by_fraction,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    if args.out.endswith(".json"):
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    else:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, "chemical_hierarchy.json")
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"  -> saved {out_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
