#!/usr/bin/env python3
"""Mechanistic post-mortem of why naive constant-step residual steering fails.

Script 1 of Workstream 3 (molcircuits):
  1. Downstream propagation & self-correction across layers:
     - Fits layer-l binary valence directions w_l on the exact stage-32
       molecule-grouped split (2,048 valid.smi molecules, sample_seed=47).
     - On 512 held-out molecules with valid bond-decision positions (rem >= 2),
       injects x_l <- x_l + alpha_eff * sigma_l * w_l for
       alpha in {-8, -4, -2, +2, +4, +8} vs 16 covariance-matched random
       directions at early (layer 1), middle, and late layers.
     - Measures at every subsequent layer m = l..L-1:
         * Norm ratio ||dh_m|| / ||dh_l||
         * Alignment with layer-m valence direction <dh_m, w_m> / (alpha_eff * sigma_m)
         * Cosine alignment sign(alpha) * <dh_m, w_m> / ||dh_m||
         * Top-k PC energy (k=10, 32, 64), variance ratio, and regularized
           Mahalanobis norm ratio relative to natural activation covariance
         * Next-token KL divergence D_KL(p_clean || p_steered) separately at
           bond-decision positions vs non-bond-decision positions (ring digits,
           branches, atoms where remaining valence <= 1, EOS, etc.)
  2. Collateral syntax failure decomposition:
     - Generates 1,024 samples per alpha in {-8, -4, -2, 0, +2, +4, +8}
       (+ 4 covariance-matched random directions) under constant-step layer-1
       steering and classifies every SMILES using molcircuits.chem.grammar_errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupShuffleSplit

from molcircuits.analysis.bond_generation import measure_attempt, summarize
from molcircuits.analysis.valence_intervene import (
    activation_covariance,
    covariance_matched_directions,
    participation_ratio,
)
from molcircuits.analysis.valence_probe import binary_direction
from molcircuits.chem.grammar_errors import CATEGORIES, profile as grammar_profile
from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import (
    BOND_ORDER,
    _charge_of,
    _element_of,
    _explicit_h,
    _is_aromatic_token,
    _remaining,
    bond_decision_labels,
)
from molcircuits.model.hooks import HookManager, resid_post
from molcircuits.model.sampling import generate
from molcircuits.model.transformer import ModelConfig, MolecularTransformer


ALPHAS_PROP = [-8.0, -4.0, -2.0, 2.0, 4.0, 8.0]
ALPHAS_GEN = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0]


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def tight_batch(smiles_list: list[str], tok: SmilesTokenizer, n_ctx: int) -> list[list[int]]:
    enc = [tok.encode(s)[:n_ctx] for s in smiles_list]
    width = max(len(e) for e in enc)
    return [e + [tok.pad_id] * (width - len(e)) for e in enc]


def select_injection_layers(n_layers: int) -> dict[str, int]:
    """Early (layer 1), middle, and late layer indices."""
    if n_layers == 4:
        return {"early": 1, "middle": 2, "late": 3}
    if n_layers == 6:
        return {"early": 1, "middle": 3, "late": 4}
    return {"early": 1, "middle": n_layers // 2, "late": n_layers - 2}


def _is_ring_tok(tok: str) -> bool:
    return tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit())


def position_categories_for_smiles(smiles: str, n_ctx: int) -> dict[str, list[int]]:
    """Classify every content prediction position p (predicting token p+1) in encoded sequence.

    Encoded sequence has: [BOS, tok_0, tok_1, ..., tok_{N-1}, EOS].
    Position p in [1 .. len(toks)] predicts:
      - for p < len(toks): toks[p] (given prefix toks[:p])
      - for p == len(toks): EOS (given full prefix toks[:len(toks)])
    We mirror molcircuits/chem/valence.py exactly to compute governing atom state after toks[p-1].
    """
    toks = tokenize(smiles)
    used: dict[int, float] = {}
    charges: dict[int, int] = {}
    elements: dict[int, str] = {}
    arom: dict[int, bool] = {}
    exp_h: dict[int, int] = {}

    prev_atom: int | None = None
    branch_stack: list[int | None] = []
    ring_open: dict[str, tuple[int, float | None]] = {}
    pending_bond: float | None = None

    cats: dict[str, list[int]] = {
        "bond_decision_ge2": [],
        "bond_decision_le1": [],
        "non_bond_ring_target": [],
        "non_bond_branch_target": [],
        "non_bond_atom_val_le1": [],
        "non_bond_eos_target": [],
        "non_bond_all": [],
        "eligible_gate": [],
        "ineligible_gate": [],
    }

    # Walk tokens; after processing toks[i], prediction position is p = i + 1
    # (predicting toks[i + 1], or EOS if i == len(toks) - 1).
    for i, tok in enumerate(toks):
        if tok == "(":
            branch_stack.append(prev_atom)
        elif tok == ")":
            prev_atom = branch_stack.pop() if branch_stack else None
        elif tok in BOND_ORDER:
            pending_bond = BOND_ORDER[tok]
        elif tok == ".":
            prev_atom, pending_bond = None, None
        elif _is_ring_tok(tok):
            lab = tok[1:] if tok.startswith("%") else tok
            b = pending_bond
            pending_bond = None
            if lab in ring_open:
                j, bj = ring_open.pop(lab)
                explicit = [x for x in (b, bj) if x is not None]
                if explicit:
                    order_ = max(explicit)
                else:
                    both_arom = arom.get(j, False) and arom.get(prev_atom, False)
                    order_ = 1.5 if both_arom else 1.0
                if prev_atom is not None:
                    used[prev_atom] = used.get(prev_atom, 0.0) + order_
                used[j] = used.get(j, 0.0) + order_
            elif prev_atom is not None:
                ring_open[lab] = (prev_atom, b)
        else:
            el = _element_of(tok)
            if el is not None:
                idx = i
                elements[idx] = el
                charges[idx] = _charge_of(tok)
                exp_h[idx] = _explicit_h(tok)
                arom[idx] = _is_aromatic_token(tok)
                used.setdefault(idx, 0.0)
                if prev_atom is not None:
                    b = pending_bond
                    if b is None:
                        b = 1.5 if (arom[idx] and arom.get(prev_atom, False)) else 1.0
                    used[idx] += b
                    used[prev_atom] = used.get(prev_atom, 0.0) + b
                pending_bond = None
                prev_atom = idx

        p = i + 1  # position in encoded sequence [BOS, toks[0], ..., toks[n-1], EOS]
        if p >= n_ctx - 1:
            break

        next_tok = toks[i + 1] if (i + 1) < len(toks) else "<eos>"
        cap = None
        if prev_atom is not None:
            cap = _remaining(
                elements[prev_atom],
                charges[prev_atom],
                used.get(prev_atom, 0.0),
                exp_h[prev_atom],
            )

        is_bd = next_tok in ("-", "=", "#") and prev_atom is not None and cap is not None
        gate_on = (prev_atom is not None) and (pending_bond is None) and (cap is not None) and (cap >= 2)

        if gate_on:
            cats["eligible_gate"].append(p)
        else:
            cats["ineligible_gate"].append(p)

        if is_bd:
            if cap >= 2:
                cats["bond_decision_ge2"].append(p)
            else:
                cats["bond_decision_le1"].append(p)
        else:
            cats["non_bond_all"].append(p)
            if next_tok == "<eos>":
                cats["non_bond_eos_target"].append(p)
            elif _is_ring_tok(next_tok):
                cats["non_bond_ring_target"].append(p)
            elif next_tok in ("(", ")"):
                cats["non_bond_branch_target"].append(p)
            if cap is not None and cap <= 1:
                cats["non_bond_atom_val_le1"].append(p)

    return cats


@torch.no_grad()
def collect_all_layers_bond_decisions(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    smiles: list[str],
    device: str,
    batch_size: int = 128,
):
    """Capture resid_post at EVERY layer on bond-decision positions in one pass."""
    n_layers = model.cfg.n_layers
    n_ctx = model.cfg.n_ctx
    hook_names = [resid_post(l) for l in range(n_layers)]
    acts_by_layer: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    labels_list: list[int] = []
    groups_list: list[int] = []
    pairs_list: list[tuple[int, int]] = []

    for s0 in range(0, len(smiles), batch_size):
        chunk = smiles[s0 : s0 + batch_size]
        ids = tight_batch(chunk, tok, n_ctx)
        x = torch.tensor(ids, dtype=torch.long, device=device)
        hm = HookManager().capture(*hook_names)
        model(x, hooks=hm)
        layer_acts = [hm[name].float().cpu().numpy() for name in hook_names]

        for bi, s in enumerate(chunk):
            gi = s0 + bi
            toks = tokenize(s)
            bd = [(p, rem) for p, rem in bond_decision_labels(toks, offset=1) if p < n_ctx - 1]
            for p, rem in bd:
                labels_list.append(rem)
                groups_list.append(gi)
                pairs_list.append((gi, p))
                for l in range(n_layers):
                    acts_by_layer[l].append(layer_acts[l][bi, p])

    acts = [np.asarray(a, dtype=np.float32) for a in acts_by_layer]
    labels = np.asarray(labels_list, dtype=int)
    groups = np.asarray(groups_list, dtype=int)
    return acts, labels, groups, pairs_list


def fit_all_layer_directions(
    acts_by_layer: list[np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    sample_seed: int,
):
    """Fit binary valence directions and activation covariances at every layer."""
    n_layers = len(acts_by_layer)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=sample_seed)
    fit_idx, held_idx = next(gss.split(np.zeros(len(groups)), None, groups))

    # One held-out bond-decision site per held-out molecule for RMS (matching script 33)
    by_mol: dict[int, list[int]] = {}
    for idx in held_idx:
        by_mol.setdefault(int(groups[idx]), []).append(int(idx))
    rng = np.random.default_rng(sample_seed)
    rms_selected = np.array([rng.choice(rows) for rows in by_mol.values()], dtype=int)

    directions = []
    probes = []
    covs = []
    rms_list = []
    cov_geom = []

    for l in range(n_layers):
        A = acts_by_layer[l]
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            w, meta = binary_direction(A, labels, groups, threshold=2, seed=sample_seed, max_iter=10000)
        w = np.asarray(w, dtype=np.float32)
        w = w / (np.linalg.norm(w) + 1e-12)
        cov = activation_covariance(A[fit_idx])
        rms = float(np.sqrt(np.square(A[rms_selected]).mean(axis=1)).mean())

        # Precompute eigendecomposition of covariance for PC energy & Mahalanobis
        C = 0.5 * (cov + cov.T)
        lam, U = np.linalg.eigh(C)
        # Sort descending
        order = np.argsort(lam)[::-1]
        lam = np.clip(lam[order], 0.0, None)
        U = U[:, order]
        mean_fit = A[fit_idx].mean(axis=0, dtype=np.float64)
        per_axis_var = float(np.trace(C) / C.shape[0])
        reg = 1e-3 * per_axis_var
        inv_lam_reg = 1.0 / (lam + reg)

        # Natural activation regularized Mahalanobis norm on held-out sites
        centered_held = A[held_idx].astype(np.float64) - mean_fit
        proj_held = centered_held @ U
        nat_mahal = float(np.sqrt((proj_held ** 2) @ inv_lam_reg).mean())

        directions.append(w)
        probes.append(meta)
        covs.append(cov)
        rms_list.append(rms)
        cov_geom.append({
            "U": U.astype(np.float32),
            "lam": lam.astype(np.float32),
            "inv_lam_reg": inv_lam_reg.astype(np.float32),
            "per_axis_var": per_axis_var,
            "nat_mahal": nat_mahal,
            "participation_ratio": participation_ratio(cov),
        })

    return {
        "fit_idx": fit_idx,
        "held_idx": held_idx,
        "directions": directions,
        "probes": probes,
        "covs": covs,
        "rms": rms_list,
        "cov_geom": cov_geom,
    }


@torch.no_grad()
def forward_from_layer(
    model: MolecularTransformer,
    x_l: torch.Tensor,
    start_layer: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Continue forward from resid_post(start_layer), returning post-block states and logits."""
    cos, sin = model._rope_for(x_l.device, x_l.dtype)
    states: dict[int, torch.Tensor] = {start_layer: x_l}
    x = x_l
    for m in range(start_layer + 1, model.cfg.n_layers):
        x = model.blocks[m](x, cos, sin)
        states[m] = x
    x_final = model.ln_f(x)
    w = model.embed.weight if model.lm_head is None else model.lm_head.weight
    logits = x_final @ w.T
    return states, logits


def compute_layer_metrics_gpu(
    dh: torch.Tensor,  # (N, D) float32
    dh_l_norm: torch.Tensor,  # (N,) float32
    w_m: torch.Tensor,  # (D,) float32
    alpha_eff: float,
    sigma_m: float,
    U_m: torch.Tensor,  # (D, D) float32
    lam_m: torch.Tensor,  # (D,) float32
    inv_lam_reg_m: torch.Tensor,  # (D,) float32
    per_axis_var_m: float,
    nat_mahal_m: float,
) -> dict[str, float]:
    """Compute propagation, alignment, and off-manifold metrics for dh at layer m."""
    dh_norm = dh.norm(dim=-1)  # (N,)
    norm_ratio = float((dh_norm / (dh_l_norm + 1e-12)).mean())

    proj_w = dh @ w_m  # (N,)
    alignment = float((proj_w / (alpha_eff * sigma_m)).mean())
    sign_a = 1.0 if alpha_eff >= 0 else -1.0
    cosine_w = float((sign_a * proj_w / (dh_norm + 1e-12)).mean())

    # Projections onto PCs of Sigma_m
    coords = dh @ U_m  # (N, D)
    sq = coords.pow(2)
    total_sq = sq.sum(dim=-1) + 1e-12
    pc10 = float((sq[:, :10].sum(dim=-1) / total_sq).mean())
    pc32 = float((sq[:, :32].sum(dim=-1) / total_sq).mean())
    pc64 = float((sq[:, :64].sum(dim=-1) / total_sq).mean())

    # Variance ratio of unit direction dh / ||dh|| under Sigma_m
    var_along = (sq * lam_m[None, :]).sum(dim=-1) / total_sq
    var_ratio = float((var_along / (per_axis_var_m + 1e-12)).mean())

    # Regularized Mahalanobis norm relative to natural activations
    mahal = torch.sqrt((sq * inv_lam_reg_m[None, :]).sum(dim=-1))
    mahal_ratio = float((mahal / (nat_mahal_m + 1e-12)).mean())

    return {
        "norm_ratio": norm_ratio,
        "valence_alignment": alignment,
        "valence_cosine": cosine_w,
        "pc_energy_top10": pc10,
        "pc_energy_top32": pc32,
        "pc_energy_top64": pc64,
        "variance_ratio": var_ratio,
        "mahalanobis_ratio": mahal_ratio,
    }


def kl_div_from_logits(
    clean_logp: torch.Tensor,
    clean_p: torch.Tensor,
    steered_logits: torch.Tensor,
) -> torch.Tensor:
    """D_KL(p_clean || p_steered) per position."""
    steered_logp = F.log_softmax(steered_logits.float(), dim=-1)
    return (clean_p * (clean_logp - steered_logp)).sum(dim=-1).clamp_min(0.0)


@torch.no_grad()
def run_propagation_diagnostics(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    smiles_all: list[str],
    pairs: list[tuple[int, int]],
    labels: np.ndarray,
    groups: np.ndarray,
    calib: dict,
    n_eval_mols: int,
    n_null: int,
    sample_seed: int,
    device: str,
) -> dict:
    """Part 1: downstream propagation, self-correction, and position-stratified KL."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    width_scale = math.sqrt(d_model / 512.0)

    # Select 512 held-out molecules with at least one bond-decision pos having rem >= 2
    held_idx = calib["held_idx"]
    by_mol_ge2: dict[int, list[int]] = {}
    for idx in held_idx:
        if labels[idx] >= 2:
            by_mol_ge2.setdefault(int(groups[idx]), []).append(int(idx))

    eligible_mols = sorted(by_mol_ge2.keys())
    rng = np.random.default_rng(sample_seed + 1)
    if len(eligible_mols) > n_eval_mols:
        picked_mols = sorted(rng.choice(eligible_mols, size=n_eval_mols, replace=False).tolist())
    else:
        picked_mols = eligible_mols

    eval_smiles = [smiles_all[gi] for gi in picked_mols]
    selected_pos = [pairs[int(rng.choice(by_mol_ge2[gi]))][1] for gi in picked_mols]

    # Build token tensor & position category index pairs (b, t)
    ids = tight_batch(eval_smiles, tok, model.cfg.n_ctx)
    x_eval = torch.tensor(ids, dtype=torch.long, device=device)
    B, T = x_eval.shape

    sel_b = torch.arange(B, dtype=torch.long, device=device)
    sel_t = torch.tensor(selected_pos, dtype=torch.long, device=device)

    cat_coords: dict[str, tuple[list[int], list[int]]] = {
        k: ([], [])
        for k in (
            "bond_decision_ge2",
            "bond_decision_le1",
            "non_bond_ring_target",
            "non_bond_branch_target",
            "non_bond_atom_val_le1",
            "non_bond_eos_target",
            "non_bond_all",
            "eligible_gate",
            "ineligible_gate",
        )
    }
    for bi, s in enumerate(eval_smiles):
        c = position_categories_for_smiles(s, model.cfg.n_ctx)
        for k, ps in c.items():
            for p in ps:
                if p < T:
                    cat_coords[k][0].append(bi)
                    cat_coords[k][1].append(p)

    cat_tensors = {
        k: (torch.tensor(bs, dtype=torch.long, device=device),
            torch.tensor(ts, dtype=torch.long, device=device))
        for k, (bs, ts) in cat_coords.items()
    }
    cat_counts = {k: len(bs) for k, (bs, _) in cat_coords.items()}

    # Clean forward capturing all layers
    hook_names = [resid_post(m) for m in range(n_layers)]
    hm = HookManager().capture(*hook_names)
    clean_logits = model(x_eval, hooks=hm).float()
    clean_states = {m: hm[resid_post(m)] for m in range(n_layers)}
    clean_logp = F.log_softmax(clean_logits, dim=-1)
    clean_p = clean_logp.exp()

    # Precompute GPU tensors for per-layer geometry
    w_gpu = [torch.tensor(w, dtype=torch.float32, device=device) for w in calib["directions"]]
    U_gpu = [torch.tensor(g["U"], dtype=torch.float32, device=device) for g in calib["cov_geom"]]
    lam_gpu = [torch.tensor(g["lam"], dtype=torch.float32, device=device) for g in calib["cov_geom"]]
    inv_lam_gpu = [torch.tensor(g["inv_lam_reg"], dtype=torch.float32, device=device) for g in calib["cov_geom"]]

    # Bond token IDs for dprob_multiple check
    multi_ids = torch.tensor([tok.vocab["="], tok.vocab["#"]], dtype=torch.long, device=device)
    clean_pmulti_sel = clean_p[sel_b, sel_t][:, multi_ids].sum(dim=-1)

    # Layer-to-layer cosine matrix of fitted valence directions
    dir_cos_matrix = [
        [float(np.dot(calib["directions"][i], calib["directions"][j])) for j in range(n_layers)]
        for i in range(n_layers)
    ]

    inj_layers = select_injection_layers(n_layers)
    results_by_inj: dict[str, dict] = {}

    for role, l in inj_layers.items():
        sigma_l = calib["rms"][l]
        w_l = calib["directions"][l]
        cov_null_dirs = covariance_matched_directions(w_l, n_null, calib["covs"][l], seed=sample_seed + 100 * l)
        # All directions: index 0 = valence probe, 1..n_null = cov-matched random
        all_dirs = [w_l] + [cov_null_dirs[i] for i in range(n_null)]

        base_xl = clean_states[l]
        role_out = {
            "layer": l,
            "role": role,
            "sigma_l": sigma_l,
            "probe_heldout_acc": calib["probes"][l]["heldout_accuracy"],
            "single_site": {},
            "all_position": {},
        }

        for alpha in ALPHAS_PROP:
            alpha_eff = alpha * width_scale
            disp = alpha_eff * sigma_l
            akey = f"{alpha:+g}"

            ss_probe = None
            ss_nulls = []
            ap_probe = None
            ap_nulls = []

            for di, vec in enumerate(all_dirs):
                v_t = torch.tensor(vec, dtype=base_xl.dtype, device=device)
                delta_vec = v_t * disp

                # --- Mode A: Single-site injection at selected bond-decision positions ---
                x_ss = base_xl.clone()
                x_ss[sel_b, sel_t] = x_ss[sel_b, sel_t] + delta_vec
                states_ss, logits_ss = forward_from_layer(model, x_ss, l)
                dh_l_ss = (states_ss[l][sel_b, sel_t] - clean_states[l][sel_b, sel_t]).float()
                dh_l_norm_ss = dh_l_ss.norm(dim=-1)

                layer_metrics_ss = {}
                for m in range(l, n_layers):
                    dh_m = (states_ss[m][sel_b, sel_t] - clean_states[m][sel_b, sel_t]).float()
                    layer_metrics_ss[str(m)] = compute_layer_metrics_gpu(
                        dh_m,
                        dh_l_norm_ss,
                        w_gpu[m],
                        alpha_eff,
                        calib["rms"][m],
                        U_gpu[m],
                        lam_gpu[m],
                        inv_lam_gpu[m],
                        calib["cov_geom"][m]["per_axis_var"],
                        calib["cov_geom"][m]["nat_mahal"],
                    )
                kl_ss = float(
                    kl_div_from_logits(
                        clean_logp[sel_b, sel_t],
                        clean_p[sel_b, sel_t],
                        logits_ss[sel_b, sel_t],
                    ).mean()
                )
                pmulti_ss = logits_ss[sel_b, sel_t].float().softmax(-1)[:, multi_ids].sum(dim=-1)
                dprob_multi_ss = float((pmulti_ss - clean_pmulti_sel).mean())
                entry_ss = {
                    "layers": layer_metrics_ss,
                    "kl_selected_bd_ge2": kl_ss,
                    "dprob_multiple_selected_bd_ge2": dprob_multi_ss,
                }
                if di == 0:
                    ss_probe = entry_ss
                else:
                    ss_nulls.append(entry_ss)

                # --- Mode B: All-position (constant) injection across full sequence ---
                x_ap = base_xl + delta_vec
                states_ap, logits_ap = forward_from_layer(model, x_ap, l)
                dh_l_ap = (states_ap[l][sel_b, sel_t] - clean_states[l][sel_b, sel_t]).float()
                dh_l_norm_ap = dh_l_ap.norm(dim=-1)

                layer_metrics_ap = {}
                for m in range(l, n_layers):
                    dh_m = (states_ap[m][sel_b, sel_t] - clean_states[m][sel_b, sel_t]).float()
                    layer_metrics_ap[str(m)] = compute_layer_metrics_gpu(
                        dh_m,
                        dh_l_norm_ap,
                        w_gpu[m],
                        alpha_eff,
                        calib["rms"][m],
                        U_gpu[m],
                        lam_gpu[m],
                        inv_lam_gpu[m],
                        calib["cov_geom"][m]["per_axis_var"],
                        calib["cov_geom"][m]["nat_mahal"],
                    )

                kl_full = kl_div_from_logits(clean_logp, clean_p, logits_ap)  # (B, T)
                kl_by_cat = {}
                for cname, (cb, ct) in cat_tensors.items():
                    if len(cb) > 0:
                        kl_by_cat[cname] = float(kl_full[cb, ct].mean())
                    else:
                        kl_by_cat[cname] = None
                pmulti_ap = logits_ap[sel_b, sel_t].float().softmax(-1)[:, multi_ids].sum(dim=-1)
                entry_ap = {
                    "layers": layer_metrics_ap,
                    "kl_by_category": kl_by_cat,
                    "dprob_multiple_selected_bd_ge2": float((pmulti_ap - clean_pmulti_sel).mean()),
                }
                if di == 0:
                    ap_probe = entry_ap
                else:
                    ap_nulls.append(entry_ap)

            # Aggregate null statistics cleanly
            def agg_layers(null_entries):
                out = {}
                for m in range(l, n_layers):
                    ms = str(m)
                    out[ms] = {}
                    for k in null_entries[0]["layers"][ms]:
                        vals = np.array([e["layers"][ms][k] for e in null_entries], dtype=float)
                        out[ms][k] = {"mean": float(vals.mean()), "std": float(vals.std())}
                return out

            ss_null_summary = {
                "layers": agg_layers(ss_nulls),
                "kl_selected_bd_ge2": {
                    "mean": float(np.mean([e["kl_selected_bd_ge2"] for e in ss_nulls])),
                    "std": float(np.std([e["kl_selected_bd_ge2"] for e in ss_nulls])),
                },
                "dprob_multiple_selected_bd_ge2": {
                    "mean": float(np.mean([e["dprob_multiple_selected_bd_ge2"] for e in ss_nulls])),
                    "std": float(np.std([e["dprob_multiple_selected_bd_ge2"] for e in ss_nulls])),
                },
            }
            ap_kl_null = {}
            for cname in cat_tensors:
                vals = [e["kl_by_category"][cname] for e in ap_nulls if e["kl_by_category"][cname] is not None]
                ap_kl_null[cname] = (
                    {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
                )
            ap_null_summary = {
                "layers": agg_layers(ap_nulls),
                "kl_by_category": ap_kl_null,
                "dprob_multiple_selected_bd_ge2": {
                    "mean": float(np.mean([e["dprob_multiple_selected_bd_ge2"] for e in ap_nulls])),
                    "std": float(np.std([e["dprob_multiple_selected_bd_ge2"] for e in ap_nulls])),
                },
            }

            role_out["single_site"][akey] = {
                "alpha": alpha,
                "alpha_eff": alpha_eff,
                "displacement": disp,
                "probe": ss_probe,
                "cov_null": ss_null_summary,
            }
            role_out["all_position"][akey] = {
                "alpha": alpha,
                "alpha_eff": alpha_eff,
                "displacement": disp,
                "probe": ap_probe,
                "cov_null": ap_null_summary,
            }

        results_by_inj[role] = role_out

    return {
        "n_eval_molecules": len(eval_smiles),
        "n_null_cov": n_null,
        "category_position_counts": cat_counts,
        "layer_probe_accuracy": [p["heldout_accuracy"] for p in calib["probes"]],
        "layer_rms": calib["rms"],
        "layer_participation_ratio": [g["participation_ratio"] for g in calib["cov_geom"]],
        "direction_cosine_matrix": dir_cos_matrix,
        "injections": results_by_inj,
    }


def _eval_smiles_chunk(args):
    smiles_list, term_list = args
    rows = [measure_attempt(i, s, t) for i, (s, t) in enumerate(zip(smiles_list, term_list))]
    err = grammar_profile(smiles_list, term_list)
    return rows, err


def score_generations(smiles_list: list[str], terminated: list[bool], pool: ProcessPoolExecutor | None = None) -> dict:
    n = len(smiles_list)
    if pool is not None and n >= 256:
        n_chunks = min(8, os.cpu_count() or 4)
        chunks = []
        step = math.ceil(n / n_chunks)
        for i in range(0, n, step):
            chunks.append((smiles_list[i : i + step], terminated[i : i + step]))
        rows = []
        from collections import Counter
        total_counts = Counter()
        total_term = 0
        for r_chunk, err_chunk in pool.map(_eval_smiles_chunk, chunks):
            rows.extend(r_chunk)
            total_counts.update(err_chunk.counts)
            total_term += err_chunk.n_terminated
        for idx, r in enumerate(rows):
            r["sample_id"] = idx
        from molcircuits.chem.grammar_errors import ErrorProfile
        err = ErrorProfile(n=n, n_terminated=total_term, counts=total_counts)
    else:
        rows = [measure_attempt(i, s, t) for i, (s, t) in enumerate(zip(smiles_list, terminated))]
        err = grammar_profile(smiles_list, terminated)

    bond_summ = summarize(rows)
    mult_per_attempt = float(
        np.mean([(r["double"] or 0) + (r["triple"] or 0) for r in rows])
    )
    bond_summ["multiple_bonds_per_attempt"] = mult_per_attempt

    counts_full = {cat: int(err.counts.get(cat, 0)) for cat in CATEGORIES}
    attempt_fracs = {cat: float(counts_full[cat] / max(n, 1)) for cat in CATEGORIES}
    term_rates = {
        cat: (float(counts_full[cat] / err.n_terminated) if err.n_terminated else None)
        for cat in CATEGORIES
        if cat != "truncated"
    }
    return {
        "bond_summary": bond_summ,
        "error_profile": {
            "n": err.n,
            "n_terminated": err.n_terminated,
            "termination_rate": float(err.termination_rate),
            "validity_over_terminated": float(err.validity) if err.n_terminated else None,
            "validity_over_attempts": float(counts_full["ok"] / max(n, 1)),
            "counts": counts_full,
            "attempt_fractions": attempt_fracs,
            "terminated_rates": term_rates,
            "ring_error_rate_terminated": float(err.ring_error_rate) if err.n_terminated else None,
            "branch_error_rate_terminated": float(err.branch_error_rate) if err.n_terminated else None,
            "chemistry_error_rate_terminated": float(err.chemistry_error_rate) if err.n_terminated else None,
            "ring_error_frac_attempts": float(
                (counts_full["ring_unclosed"] + counts_full["ring_unopened"] + counts_full["ring_illegal"]) / max(n, 1)
            ),
            "branch_error_frac_attempts": float(
                (counts_full["paren_unclosed"] + counts_full["paren_unopened"]) / max(n, 1)
            ),
            "chemistry_error_frac_attempts": float(
                (counts_full["parse"] + counts_full["valence"] + counts_full["aromaticity"] + counts_full["other"]) / max(n, 1)
            ),
            "truncated_frac_attempts": float(counts_full["truncated"] / max(n, 1)),
        },
    }


@torch.no_grad()
def run_syntax_failure_decomposition(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    calib: dict,
    n_samples: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    generation_seed: int,
    sample_seed: int,
    n_null_gen: int,
    device: str,
    pool: ProcessPoolExecutor | None = None,
) -> dict:
    """Part 2: Classify every failure mode under constant-step layer-1 steering."""
    layer = 1
    d_model = model.cfg.d_model
    width_scale = math.sqrt(d_model / 512.0)
    sigma_1 = calib["rms"][layer]
    w_1 = calib["directions"][layer]
    null_dirs = covariance_matched_directions(w_1, n_null_gen, calib["covs"][layer], seed=sample_seed)

    def gen_one(vec: np.ndarray | None, disp: float):
        hooks = None
        if vec is not None and disp != 0.0:
            edit = torch.as_tensor(vec, dtype=torch.float32, device=device) * disp
            hooks = HookManager().edit(resid_post(layer), lambda x: x + edit.to(x))
        meta: dict = {}
        smis = generate(
            model,
            tok,
            n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            hooks=hooks,
            seed=generation_seed,
            batch_size=batch_size,
            device=device,
            meta=meta,
            use_kv_cache=True,
        )
        return score_generations(smis, meta["terminated"], pool=pool)

    probe_arms = {}
    null_arms = {}

    for alpha in ALPHAS_GEN:
        akey = f"{alpha:+g}" if alpha != 0.0 else "0"
        disp = alpha * width_scale * sigma_1
        probe_arms[akey] = {
            "alpha": alpha,
            "displacement": disp,
            **gen_one(None if alpha == 0.0 else w_1, disp),
        }
        if alpha != 0.0 and n_null_gen > 0:
            draws = [gen_one(null_dirs[r], disp) for r in range(n_null_gen)]
            # Aggregate mean across draws
            agg_counts = {
                cat: float(np.mean([d["error_profile"]["attempt_fractions"][cat] for d in draws]))
                for cat in CATEGORIES
            }
            null_arms[akey] = {
                "alpha": alpha,
                "displacement": disp,
                "n_draws": n_null_gen,
                "mean_valid_rate": float(np.mean([d["bond_summary"]["complete_valid_rate"] for d in draws])),
                "std_valid_rate": float(np.std([d["bond_summary"]["complete_valid_rate"] for d in draws])),
                "mean_termination_rate": float(np.mean([d["error_profile"]["termination_rate"] for d in draws])),
                "mean_multiple_fraction_valid": float(
                    np.nanmean([
                        d["bond_summary"]["multiple_fraction_valid"]
                        if d["bond_summary"]["multiple_fraction_valid"] is not None
                        else np.nan
                        for d in draws
                    ])
                ),
                "mean_multiple_bonds_per_attempt": float(
                    np.mean([d["bond_summary"]["multiple_bonds_per_attempt"] for d in draws])
                ),
                "mean_attempt_fractions": agg_counts,
                "draws": draws,
            }

    return {
        "layer": layer,
        "sigma_1": sigma_1,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "generation_seed": generation_seed,
        "probe": probe_arms,
        "cov_null": null_arms,
    }


def evaluate_checkpoint(
    ckpt_path: Path,
    data_dir: Path,
    smiles_2048: list[str],
    tok: SmilesTokenizer,
    args,
    pool: ProcessPoolExecutor | None = None,
) -> dict:
    t0 = time.time()
    cfg = ModelConfig.load(str(ckpt_path.parent / "model_config.json"))
    model = MolecularTransformer(cfg).to(args.device).eval()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])

    acts, labels, groups, pairs = collect_all_layers_bond_decisions(
        model, tok, smiles_2048, device=args.device, batch_size=128
    )
    calib = fit_all_layer_directions(acts, labels, groups, sample_seed=args.sample_seed)

    t1 = time.time()
    prop_diag = run_propagation_diagnostics(
        model,
        tok,
        smiles_2048,
        pairs,
        labels,
        groups,
        calib,
        n_eval_mols=args.eval_molecules,
        n_null=args.n_null_prop,
        sample_seed=args.sample_seed,
        device=args.device,
    )
    t2 = time.time()
    syntax_diag = run_syntax_failure_decomposition(
        model,
        tok,
        calib,
        n_samples=args.n_samples,
        batch_size=args.gen_batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        generation_seed=args.generation_seed,
        sample_seed=args.sample_seed,
        n_null_gen=args.n_null_gen,
        device=args.device,
        pool=pool,
    )
    t3 = time.time()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(ckpt_path),
        "model_config": cfg.__dict__,
        "settings": {
            "sample_seed": args.sample_seed,
            "n_calib_molecules": len(smiles_2048),
            "n_eval_molecules": prop_diag["n_eval_molecules"],
            "n_null_prop": args.n_null_prop,
            "n_null_gen": args.n_null_gen,
            "n_samples": args.n_samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "generation_seed": args.generation_seed,
        },
        "timing_sec": {
            "calibration": round(t1 - t0, 2),
            "propagation": round(t2 - t1, 2),
            "syntax_decomposition": round(t3 - t2, 2),
            "total": round(t3 - t0, 2),
        },
        "propagation_diagnostics": prop_diag,
        "syntax_failure_decomposition": syntax_diag,
    }


def build_summary(per_ckpt: dict[str, dict]) -> dict:
    """Aggregate key diagnostic trends across sizes and seeds."""
    by_size: dict[str, list[dict]] = {}
    for key, res in per_ckpt.items():
        size = key.split("_seed")[0]
        by_size.setdefault(size, []).append(res)

    summary = {"checkpoints": sorted(per_ckpt.keys()), "by_size": {}}
    for size, items in by_size.items():
        # Summarize layer-1 propagation at alpha=+8 and -8 (final layer m=L-1)
        L = items[0]["model_config"]["n_layers"]
        last_m = str(L - 1)
        prop_summary = {}
        for role in ("early", "middle", "late"):
            prop_summary[role] = {}
            for akey in ("-8", "-4", "-2", "+2", "+4", "+8"):
                # Single-site final layer stats
                nr_p = [it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][last_m]["norm_ratio"] for it in items]
                al_p = [it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][last_m]["valence_alignment"] for it in items]
                pc32_p = [it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][last_m]["pc_energy_top32"] for it in items]
                mh_p = [it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][last_m]["mahalanobis_ratio"] for it in items]
                nr_n = [it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["cov_null"]["layers"][last_m]["norm_ratio"]["mean"] for it in items]

                # All-position KL by category
                kl_bd = [it["propagation_diagnostics"]["injections"][role]["all_position"][akey]["probe"]["kl_by_category"]["bond_decision_ge2"] for it in items]
                kl_ring = [it["propagation_diagnostics"]["injections"][role]["all_position"][akey]["probe"]["kl_by_category"]["non_bond_ring_target"] for it in items]
                kl_branch = [it["propagation_diagnostics"]["injections"][role]["all_position"][akey]["probe"]["kl_by_category"]["non_bond_branch_target"] for it in items]
                kl_le1 = [it["propagation_diagnostics"]["injections"][role]["all_position"][akey]["probe"]["kl_by_category"]["non_bond_atom_val_le1"] for it in items]
                kl_eos = [it["propagation_diagnostics"]["injections"][role]["all_position"][akey]["probe"]["kl_by_category"]["non_bond_eos_target"] for it in items]
                kl_nonbd = [it["propagation_diagnostics"]["injections"][role]["all_position"][akey]["probe"]["kl_by_category"]["non_bond_all"] for it in items]

                # Full layer-by-layer trajectory (averaged over seeds)
                l_start = items[0]["propagation_diagnostics"]["injections"][role]["layer"]
                layer_curve = {}
                for m in range(l_start, L):
                    ms = str(m)
                    layer_curve[ms] = {
                        "norm_ratio_probe": float(np.mean([
                            it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][ms]["norm_ratio"]
                            for it in items
                        ])),
                        "norm_ratio_covnull": float(np.mean([
                            it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["cov_null"]["layers"][ms]["norm_ratio"]["mean"]
                            for it in items
                        ])),
                        "valence_alignment_probe": float(np.mean([
                            it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][ms]["valence_alignment"]
                            for it in items
                        ])),
                        "valence_cosine_probe": float(np.mean([
                            it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][ms]["valence_cosine"]
                            for it in items
                        ])),
                        "pc32_energy_probe": float(np.mean([
                            it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][ms]["pc_energy_top32"]
                            for it in items
                        ])),
                        "mahalanobis_ratio_probe": float(np.mean([
                            it["propagation_diagnostics"]["injections"][role]["single_site"][akey]["probe"]["layers"][ms]["mahalanobis_ratio"]
                            for it in items
                        ])),
                    }

                prop_summary[role][akey] = {
                    "inj_layer": l_start,
                    "final_layer": L - 1,
                    "layer_trajectory": layer_curve,
                    "norm_ratio_probe_mean": float(np.mean(nr_p)),
                    "norm_ratio_covnull_mean": float(np.mean(nr_n)),
                    "valence_alignment_final_mean": float(np.mean(al_p)),
                    "pc32_energy_final_mean": float(np.mean(pc32_p)),
                    "mahalanobis_ratio_final_mean": float(np.mean(mh_p)),
                    "kl_bond_decision_ge2": float(np.mean(kl_bd)),
                    "kl_non_bond_all": float(np.mean(kl_nonbd)),
                    "kl_non_bond_ring": float(np.mean(kl_ring)),
                    "kl_non_bond_branch": float(np.mean(kl_branch)),
                    "kl_non_bond_val_le1": float(np.mean(kl_le1)),
                    "kl_non_bond_eos": float(np.mean(kl_eos)),
                }

        # Syntax failure decomposition summary across alphas
        syn_summary = {}
        for akey in ("-8", "-4", "-2", "0", "+2", "+4", "+8"):
            valid_rates = [it["syntax_failure_decomposition"]["probe"][akey]["bond_summary"]["complete_valid_rate"] for it in items]
            term_rates = [it["syntax_failure_decomposition"]["probe"][akey]["error_profile"]["termination_rate"] for it in items]
            mult_val = [it["syntax_failure_decomposition"]["probe"][akey]["bond_summary"]["multiple_fraction_valid"] for it in items]
            mult_att = [it["syntax_failure_decomposition"]["probe"][akey]["bond_summary"]["multiple_bonds_per_attempt"] for it in items]
            fracs = {
                cat: float(np.mean([
                    it["syntax_failure_decomposition"]["probe"][akey]["error_profile"]["attempt_fractions"][cat]
                    for it in items
                ]))
                for cat in CATEGORIES
            }
            syn_summary[akey] = {
                "valid_rate_mean": float(np.mean(valid_rates)),
                "valid_rate_std": float(np.std(valid_rates)),
                "termination_rate_mean": float(np.mean(term_rates)),
                "multiple_fraction_valid_mean": float(np.nanmean([x if x is not None else np.nan for x in mult_val])),
                "multiple_bonds_per_attempt_mean": float(np.mean(mult_att)),
                "attempt_fractions_mean": fracs,
            }

        summary["by_size"][size] = {
            "n_seeds": len(items),
            "n_layers": L,
            "d_model": items[0]["model_config"]["d_model"],
            "propagation_summary": prop_summary,
            "syntax_decomposition_summary": syn_summary,
        }
    return summary


def main():
    from molcircuits import paths as _paths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=None, help="Single checkpoint path")
    ap.add_argument("--grid-root", type=Path, default=Path(_paths.root()) / "grid")
    ap.add_argument("--all-grid", action="store_true", help="Run all 9 canonical models in grid-root")
    ap.add_argument("--sizes", nargs="+", default=["small", "base", "large"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data", type=Path, default=Path(_paths.data()))
    ap.add_argument("--out", type=Path, default=Path("results/steering"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sample-seed", type=int, default=47)
    ap.add_argument("--calib-molecules", type=int, default=2048)
    ap.add_argument("--eval-molecules", type=int, default=512)
    ap.add_argument("--n-null-prop", type=int, default=16)
    ap.add_argument("--n-null-gen", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--gen-batch-size", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--generation-seed", type=int, default=42000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tok = SmilesTokenizer.load(str(args.data / "tokenizer.json"))
    population = (args.data / "valid.smi").read_text().splitlines()
    rng = np.random.default_rng(args.sample_seed)
    idx = rng.choice(len(population), args.calib_molecules, replace=False)
    smiles_2048 = [population[i] for i in idx]

    if args.ckpt is not None:
        targets = [("single", args.ckpt)]
    elif args.all_grid:
        targets = [
            (f"{size}_seed{s}", args.grid_root / f"zinc_{size}" / f"tf_seed{s}" / "checkpoint.pt")
            for size in args.sizes
            for s in args.seeds
        ]
    else:
        ap.error("Specify either --ckpt or --all-grid")

    per_ckpt = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, ckpt in targets:
            print(f"=== Running diagnostics on {name}: {ckpt} ===", flush=True)
            res = evaluate_checkpoint(ckpt, args.data, smiles_2048, tok, args, pool=pool)
            per_ckpt[name] = res
            out_file = args.out / f"diagnostics_{name}.json"
            atomic_json(out_file, res)
            p_syn = res["syntax_failure_decomposition"]["probe"]
            e_p8 = res["propagation_diagnostics"]["injections"]["early"]["all_position"]["+8"]["probe"]
            L_last = str(res["model_config"]["n_layers"] - 1)
            print(
                f"  [{name}] {res['timing_sec']['total']:.1f}s | "
                f"L1->L{L_last} (+8) norm_ratio={e_p8['layers'][L_last]['norm_ratio']:.2f}, "
                f"align={e_p8['layers'][L_last]['valence_alignment']:.2f}, "
                f"KL(bd)={e_p8['kl_by_category']['bond_decision_ge2']:.3f}, "
                f"KL(ring)={e_p8['kl_by_category']['non_bond_ring_target']:.3f}, "
                f"KL(eos)={e_p8['kl_by_category']['non_bond_eos_target']:.3f} | "
                f"Gen valid: a=0:{p_syn['0']['bond_summary']['complete_valid_rate']:.3f}, "
                f"a=-8:{p_syn['-8']['bond_summary']['complete_valid_rate']:.3f}, "
                f"a=+8:{p_syn['+8']['bond_summary']['complete_valid_rate']:.3f}",
                flush=True,
            )

    if args.all_grid:
        summary = build_summary(per_ckpt)
        atomic_json(args.out / "summary_diagnostics.json", summary)
        print(f"Saved {args.out / 'summary_diagnostics.json'}", flush=True)


if __name__ == "__main__":
    main()
