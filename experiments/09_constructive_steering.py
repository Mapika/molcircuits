#!/usr/bin/env python3
"""Constructive steering suite on a unified Yield-vs-Effect Pareto frontier.

Script 2 of Workstream 3 (molcircuits):
  Experiment 1 — Decision-Gated vs. Constant-Step Valence Steering:
    * Incremental online SMILES valence state tracker (verified 100% against
      molcircuits.chem.valence on 500 validation molecules before running).
    * Compares across alpha in {-8, -4, -2, 0, +2, +4, +8}:
        - single_const         : layer 1, constant every step
        - single_gated         : layer 1, only when active atom rem_val >= 2 & bond-eligible
        - single_gated_nonarom : layer 1, rem_val >= 2 & non-aromatic active atom
        - multi_const          : layers 1..L-2, constant, matched total L2 norm
        - multi_gated          : layers 1..L-2, decision-gated, matched total L2 norm
        - 4 covariance-matched random directions per regime
  Experiment 2 — Global Property Steering (MolLogP, TPSA, NumAromaticRings):
    * Fits held-out RidgeCV linear directions at every layer on 4,000 validation
      molecules (2,800 train / 1,200 test) from content-token mean-pooled residuals.
    * Compares on a unified Pareto frontier (Valid Rate vs Property Shift vs
      Target-Hit Yield per Attempt vs Internal Diversity & Novelty & KL):
        - act_single    : single-layer activation steering at optimal middle layer
        - act_multi     : multi-layer activation steering across layers 1..L-2 (matched L2)
        - cov_null      : 4 covariance-matched random directions at optimal middle layer
        - logit_unemb   : single-pass logit bias via unembedding projection of W_{L-1}
        - logit_token   : single-pass bag-of-tokens linear logit bias
        - best_of_n     : Best-of-N reranking (N=1,2,4,8,16) from unsteered pool

Clean CLI supports both single checkpoint (`--ckpt --data --out --seed --device`)
and full 9-model grid (`--all-grid --grid-root ...`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit

from molcircuits.analysis.bond_generation import measure_attempt, summarize
from molcircuits.analysis.valence_intervene import (
    activation_covariance,
    covariance_matched_directions,
)
from molcircuits.analysis.valence_probe import binary_direction
from molcircuits.chem.grammar_errors import CATEGORIES, classify as classify_grammar
from molcircuits.chem.tokenizer import SPECIALS, SmilesTokenizer, tokenize
from molcircuits.chem.valence import (
    BOND_ORDER,
    _charge_of,
    _element_of,
    _explicit_h,
    _is_aromatic_token,
    _remaining,
    bond_decision_labels,
    valence_labels,
)
from molcircuits.model.hooks import HookManager, resid_post
from molcircuits.model.transformer import ModelConfig, MolecularTransformer

RDLogger.DisableLog("rdApp.*")

VALENCE_ALPHAS = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0]
PROP_ALPHAS = [-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0]
LOGIT_BETAS = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
BEST_OF_NS = [1, 2, 4, 8, 16]

PROPERTY_SPECS = {
    "MolLogP": {
        "high_thresh": 3.5,
        "high_alt_thresh": 2.8,
        "low_thresh": 1.5,
        "high_op": ">",
        "low_op": "<",
    },
    "TPSA": {
        "high_thresh": 90.0,
        "high_alt_thresh": None,
        "low_thresh": 45.0,
        "high_op": ">",
        "low_op": "<",
    },
    "NumAromaticRings": {
        "high_thresh": 3.0,
        "high_alt_thresh": 2.0,
        "low_thresh": 1.0,
        "high_op": ">=",
        "low_op": "<=",
    },
}


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def tight_batch(smiles_list: list[str], tok: SmilesTokenizer, n_ctx: int) -> list[list[int]]:
    enc = [tok.encode(s)[:n_ctx] for s in smiles_list]
    width = max(len(e) for e in enc)
    return [e + [tok.pad_id] * (width - len(e)) for e in enc]


# ---------------------------------------------------------------------------
# Fast Incremental Online SMILES Valence State Tracker
# ---------------------------------------------------------------------------

class TokenMetadataTable:
    """Precomputed token-id properties so incremental valence stepping is O(1) per sequence."""

    def __init__(self, tok: SmilesTokenizer):
        V = tok.vocab_size
        self.id_to_str = [""] * V
        for s, idx in tok.vocab.items():
            self.id_to_str[idx] = s
        self.kind = [0] * V  # 0=other/special, 1='(', 2=')', 3=bond, 4='.', 5=ring, 6=atom
        self.bond_val = [0.0] * V
        self.ring_lab = [""] * V
        self.atom_info: list[tuple[str, int, int, bool] | None] = [None] * V

        for idx, s in enumerate(self.id_to_str):
            if s in SPECIALS or not s:
                self.kind[idx] = 0
            elif s == "(":
                self.kind[idx] = 1
            elif s == ")":
                self.kind[idx] = 2
            elif s in BOND_ORDER:
                self.kind[idx] = 3
                self.bond_val[idx] = float(BOND_ORDER[s])
            elif s == ".":
                self.kind[idx] = 4
            elif s.isdigit() or (s.startswith("%") and s[1:].isdigit()):
                self.kind[idx] = 5
                self.ring_lab[idx] = s[1:] if s.startswith("%") else s
            else:
                el = _element_of(s)
                if el is not None:
                    self.kind[idx] = 6
                    self.atom_info[idx] = (el, _charge_of(s), _explicit_h(s), _is_aromatic_token(s))
                else:
                    self.kind[idx] = 0


class IncrementalValenceState:
    """Exact left-to-right incremental mirror of molcircuits.chem.valence.valence_labels."""

    __slots__ = (
        "used",
        "charges",
        "elements",
        "arom",
        "exp_h",
        "prev_atom",
        "branch_stack",
        "ring_open",
        "pending_bond",
        "next_atom_id",
        "done",
    )

    def __init__(self):
        self.used: list[float] = []
        self.charges: list[int] = []
        self.elements: list[str] = []
        self.arom: list[bool] = []
        self.exp_h: list[int] = []
        self.prev_atom: int | None = None
        self.branch_stack: list[int | None] = []
        self.ring_open: dict[str, tuple[int, float | None]] = {}
        self.pending_bond: float | None = None
        self.next_atom_id: int = 0
        self.done: bool = False

    def step(self, tid: int, table: TokenMetadataTable) -> int | None:
        """Advance state with token id `tid`. Returns atom write-time remaining if atom, else None."""
        k = table.kind[tid]
        if k == 0:
            return None
        if k == 1:  # "("
            self.branch_stack.append(self.prev_atom)
            return None
        if k == 2:  # ")"
            self.prev_atom = self.branch_stack.pop() if self.branch_stack else None
            return None
        if k == 3:  # bond
            self.pending_bond = table.bond_val[tid]
            return None
        if k == 4:  # "."
            self.prev_atom = None
            self.pending_bond = None
            return None
        if k == 5:  # ring digit
            lab = table.ring_lab[tid]
            b = self.pending_bond
            self.pending_bond = None
            if lab in self.ring_open:
                j, bj = self.ring_open.pop(lab)
                if b is not None and bj is not None:
                    order_ = b if b >= bj else bj
                elif b is not None:
                    order_ = b
                elif bj is not None:
                    order_ = bj
                else:
                    both_arom = self.arom[j] and (self.prev_atom is not None and self.arom[self.prev_atom])
                    order_ = 1.5 if both_arom else 1.0
                if self.prev_atom is not None:
                    self.used[self.prev_atom] += order_
                self.used[j] += order_
            elif self.prev_atom is not None:
                self.ring_open[lab] = (self.prev_atom, b)
            return None

        # k == 6: atom token
        el, chg, eh, ar = table.atom_info[tid]
        idx = self.next_atom_id
        self.next_atom_id = idx + 1
        self.elements.append(el)
        self.charges.append(chg)
        self.exp_h.append(eh)
        self.arom.append(ar)
        spent = 0.0
        pa = self.prev_atom
        if pa is not None:
            b = self.pending_bond
            if b is None:
                b = 1.5 if (ar and self.arom[pa]) else 1.0
            spent = b
            self.used[pa] += b
        self.used.append(spent)
        self.pending_bond = None
        self.prev_atom = idx
        return _remaining(el, chg, spent, eh)

    def current_capacity(self) -> int | None:
        pa = self.prev_atom
        if pa is None:
            return None
        return _remaining(self.elements[pa], self.charges[pa], self.used[pa], self.exp_h[pa])

    def eligible_bond_ge2(self, non_aromatic_only: bool = False) -> bool:
        if self.done or self.prev_atom is None or self.pending_bond is not None:
            return False
        if non_aromatic_only and self.arom[self.prev_atom]:
            return False
        cap = _remaining(
            self.elements[self.prev_atom],
            self.charges[self.prev_atom],
            self.used[self.prev_atom],
            self.exp_h[self.prev_atom],
        )
        return cap is not None and cap >= 2


def verify_incremental_tracker(smiles_list: list[str], tok: SmilesTokenizer) -> dict:
    """Verify IncrementalValenceState matches molcircuits.chem.valence on all 500 SMILES with 100% accuracy."""
    table = TokenMetadataTable(tok)
    n_atoms_checked = 0
    n_bd_checked = 0
    for s in smiles_list:
        toks = tokenize(s)
        ids = [tok.vocab.get(t, tok.unk_id) for t in toks]
        ref_atoms = valence_labels(toks, offset=0, semantics="prefix")
        ref_bd = bond_decision_labels(toks, offset=0)

        st = IncrementalValenceState()
        got_atoms = []
        got_bd = []
        for i, (t_str, tid) in enumerate(zip(toks, ids)):
            if t_str in ("-", "=", "#") and i > 0 and st.prev_atom is not None:
                cap = st.current_capacity()
                if cap is not None:
                    got_bd.append((i - 1, cap))
            rem_write = st.step(tid, table)
            if rem_write is not None:
                got_atoms.append((i, rem_write))

        ref_atoms_pairs = [(v.token_idx, v.remaining) for v in ref_atoms]
        if got_atoms != ref_atoms_pairs:
            raise AssertionError(f"Atom valence mismatch on {s}: got {got_atoms[:5]} vs ref {ref_atoms_pairs[:5]}")
        if got_bd != ref_bd:
            raise AssertionError(f"Bond decision mismatch on {s}: got {got_bd} vs ref {ref_bd}")
        n_atoms_checked += len(got_atoms)
        n_bd_checked += len(got_bd)
    return {
        "n_smiles": len(smiles_list),
        "n_atoms_checked": n_atoms_checked,
        "n_bond_decisions_checked": n_bd_checked,
        "exact_match": True,
    }


@torch.no_grad()
def verify_kv_cached_gated_equivalence(model: MolecularTransformer, tok: SmilesTokenizer, device: str) -> float:
    """Verify KV-cached step-gated hooks match uncached full-prefix gated forward."""
    table = TokenMetadataTable(tok)
    d = model.cfg.d_model
    rng = np.random.default_rng(123)
    w = rng.standard_normal(d).astype(np.float32)
    w = w / np.linalg.norm(w) * 2.0
    edit_t = torch.tensor(w, dtype=torch.float32, device=device)

    # Sample 4 short sequences using KV cache with decision gating, record gate mask at every step
    B = 4
    seq = torch.full((B, 1), tok.bos_id, dtype=torch.long, device=device)
    states = [IncrementalValenceState() for _ in range(B)]
    gate_t = torch.zeros((B, 1, 1), dtype=torch.float32, device=device)
    hooks_kv = HookManager().edit(resid_post(1), lambda x: x + gate_t.to(x.dtype) * edit_t.to(x.dtype))
    cache = [{} for _ in range(model.cfg.n_layers)]
    g = torch.Generator(device=device).manual_seed(999)

    step_in = seq
    gate_hist = []
    logits_kv_list = []
    for _ in range(12):
        gmask = [1.0 if st.eligible_bond_ge2() else 0.0 for st in states]
        gate_hist.append(gmask)
        gate_t.copy_(torch.tensor(gmask, dtype=torch.float32, device=device).view(B, 1, 1))
        lg = model(step_in, hooks=hooks_kv, cache=cache)[:, -1]
        logits_kv_list.append(lg)
        nxt = torch.multinomial(lg.softmax(-1), 1, generator=g)
        seq = torch.cat([seq, nxt], dim=1)
        step_in = nxt
        for bi, tid in enumerate(nxt.squeeze(1).tolist()):
            if tid == tok.eos_id:
                states[bi].done = True
            else:
                states[bi].step(tid, table)

    # Full-prefix uncached forward with the exact recorded gate_hist (B, T_in, 1)
    x_in = seq[:, :-1]  # (B, 12)
    full_gate = torch.tensor(gate_hist, dtype=torch.float32, device=device).T.unsqueeze(-1)  # (B, 12, 1)
    hooks_full = HookManager().edit(resid_post(1), lambda x: x + full_gate.to(x.dtype) * edit_t.to(x.dtype))
    logits_full = model(x_in, hooks=hooks_full)
    logits_kv = torch.stack(logits_kv_list, dim=1)
    max_diff = float((logits_full - logits_kv).abs().max())
    if max_diff > 1e-3 * max(1.0, float(logits_full.abs().max())):
        raise RuntimeError(f"KV-cached gated generation disagrees with full forward by {max_diff:.3e}")
    return max_diff


# ---------------------------------------------------------------------------
# Fast KV-Cached Generation with Optional Gating / Multi-Layer / Logit Bias
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_steered(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    n: int,
    *,
    layer_edits: dict[int, torch.Tensor] | None = None,
    logit_bias: torch.Tensor | None = None,
    gate_mode: str = "none",  # "none", "val_ge2", "val_ge2_nonarom"
    table: TokenMetadataTable | None = None,
    max_tokens: int = 256,
    temperature: float = 1.0,
    seed: int = 0,
    batch_size: int = 1024,
    device: str = "cuda:0",
) -> tuple[list[str], list[bool], float]:
    """KV-cached autoregressive sampling supporting constant, decision-gated, multi-layer, and logit steering."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    n_layers = model.cfg.n_layers
    use_gate = gate_mode != "none"
    non_arom_only = gate_mode == "val_ge2_nonarom"
    if use_gate and table is None:
        table = TokenMetadataTable(tok)

    out_smiles: list[str] = []
    out_term: list[bool] = []
    total_active_steps = 0
    total_gated_steps = 0

    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        seq = torch.full((b, 1), tok.bos_id, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        cache = [{} for _ in range(n_layers)]
        step_in = seq

        gate_t = torch.ones((b, 1, 1), dtype=torch.float32, device=device)
        hooks = None
        if layer_edits:
            hooks = HookManager()
            if use_gate:
                for l, edit_vec in layer_edits.items():
                    ev = edit_vec.to(device)
                    hooks.edit(resid_post(l), lambda x, e=ev: x + gate_t.to(x.dtype) * e.to(x.dtype))
            else:
                for l, edit_vec in layer_edits.items():
                    ev = edit_vec.to(device)
                    hooks.edit(resid_post(l), lambda x, e=ev: x + e.to(x.dtype))

        states = [IncrementalValenceState() for _ in range(b)] if use_gate else None
        done_py = [False] * b if use_gate else None

        for step in range(max_tokens):
            if use_gate:
                glist = [0.0] * b
                active_cnt = 0
                gated_cnt = 0
                for bi in range(b):
                    if not done_py[bi]:
                        active_cnt += 1
                        if states[bi].eligible_bond_ge2(non_aromatic_only=non_arom_only):
                            glist[bi] = 1.0
                            gated_cnt += 1
                total_active_steps += active_cnt
                total_gated_steps += gated_cnt
                gate_t.copy_(torch.tensor(glist, dtype=torch.float32, device=device).view(b, 1, 1))

            logits = model(step_in, hooks=hooks, cache=cache)[:, -1]
            if logit_bias is not None:
                logits = logits + logit_bias.to(logits.dtype)
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                if temperature != 1.0:
                    logits = logits / temperature
                probs = logits.softmax(-1)
                nxt = torch.multinomial(probs, 1, generator=g)

            nxt = torch.where(done[:, None], torch.full_like(nxt, tok.pad_id), nxt)
            seq = torch.cat([seq, nxt], dim=1)
            step_in = nxt
            eos_hit = nxt.squeeze(1) == tok.eos_id
            done |= eos_hit

            if use_gate:
                nxt_list = nxt.squeeze(1).tolist()
                for bi in range(b):
                    if not done_py[bi]:
                        tid = nxt_list[bi]
                        if tid == tok.eos_id:
                            done_py[bi] = True
                            states[bi].done = True
                        else:
                            states[bi].step(tid, table)
                if all(done_py):
                    break
            else:
                if step % 8 == 7 and bool(done.all()):
                    break

        for row in seq.tolist():
            out_smiles.append(tok.decode(row))
        out_term.extend(bool(v) for v in done.tolist())

    gate_frac = float(total_gated_steps / max(total_active_steps, 1)) if use_gate else 1.0
    return out_smiles[:n], out_term[:n], gate_frac


# ---------------------------------------------------------------------------
# Experiment 1: Decision-Gated vs. Constant-Step Valence Steering
# ---------------------------------------------------------------------------

def _score_valence_chunk(args):
    smis, terms = args
    rows = []
    counts = Counter()
    for s, t in zip(smis, terms):
        rows.append(measure_attempt(0, s, t))
        counts[classify_grammar(s, bool(t))] += 1
    return rows, counts


def evaluate_valence_arm(
    smis: list[str],
    terms: list[bool],
    gate_frac: float,
    pool: ProcessPoolExecutor | None = None,
) -> dict:
    n = len(smis)
    if pool is not None and n >= 256:
        n_chunks = min(8, os.cpu_count() or 4)
        step = math.ceil(n / n_chunks)
        chunks = [(smis[i : i + step], terms[i : i + step]) for i in range(0, n, step)]
        rows = []
        counts = Counter()
        for r_c, c_c in pool.map(_score_valence_chunk, chunks):
            rows.extend(r_c)
            counts.update(c_c)
        for idx, r in enumerate(rows):
            r["sample_id"] = idx
    else:
        rows = [measure_attempt(i, s, t) for i, (s, t) in enumerate(zip(smis, terms))]
        counts = Counter(classify_grammar(s, bool(t)) for s, t in zip(smis, terms))

    summ = summarize(rows)
    mult_counts_all = [(r["double"] or 0) + (r["triple"] or 0) for r in rows]
    mult_counts_valid = [r["double"] + r["triple"] for r in rows if r["complete_valid"]]
    summ["multiple_bonds_per_attempt"] = float(np.mean(mult_counts_all))
    summ["multiple_bonds_valid_mean"] = float(np.mean(mult_counts_valid)) if mult_counts_valid else None
    summ["gate_active_fraction"] = gate_frac
    summ["error_attempt_fractions"] = {cat: float(counts.get(cat, 0) / max(n, 1)) for cat in CATEGORIES}
    return summ


@torch.no_grad()
def run_experiment1_valence(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    smiles_2048: list[str],
    table: TokenMetadataTable,
    args,
    pool: ProcessPoolExecutor | None = None,
) -> dict:
    """Decision-gated vs constant-step, single-layer vs multi-layer valence steering."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    width_scale = math.sqrt(d_model / 512.0)
    mid_layers = list(range(1, n_layers - 1))  # 1 .. L-2
    K = len(mid_layers)

    # Collect bond-decision activations at layers 1..L-2 in one pass
    hook_names = [resid_post(l) for l in mid_layers]
    acts_by_l: dict[int, list[np.ndarray]] = {l: [] for l in mid_layers}
    labels_list: list[int] = []
    groups_list: list[int] = []

    for s0 in range(0, len(smiles_2048), 128):
        chunk = smiles_2048[s0 : s0 + 128]
        ids = tight_batch(chunk, tok, model.cfg.n_ctx)
        x = torch.tensor(ids, dtype=torch.long, device=args.device)
        hm = HookManager().capture(*hook_names)
        model(x, hooks=hm)
        la = {l: hm[resid_post(l)].float().cpu().numpy() for l in mid_layers}
        for bi, s in enumerate(chunk):
            gi = s0 + bi
            bd = [(p, rem) for p, rem in bond_decision_labels(tokenize(s), offset=1) if p < model.cfg.n_ctx - 1]
            for p, rem in bd:
                labels_list.append(rem)
                groups_list.append(gi)
                for l in mid_layers:
                    acts_by_l[l].append(la[l][bi, p])

    labels = np.asarray(labels_list, dtype=int)
    groups = np.asarray(groups_list, dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=args.sample_seed)
    fit_idx, held_idx = next(gss.split(np.zeros(len(groups)), None, groups))
    by_mol: dict[int, list[int]] = {}
    for idx in held_idx:
        by_mol.setdefault(int(groups[idx]), []).append(int(idx))
    rng = np.random.default_rng(args.sample_seed)
    rms_sel = np.array([rng.choice(rows) for rows in by_mol.values()], dtype=int)

    w_by_l: dict[int, np.ndarray] = {}
    rms_by_l: dict[int, float] = {}
    nulls_by_l: dict[int, np.ndarray] = {}
    probe_acc_by_l: dict[int, float] = {}

    for l in mid_layers:
        A = np.asarray(acts_by_l[l], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            w, meta = binary_direction(A, labels, groups, threshold=2, seed=args.sample_seed, max_iter=10000)
        w = np.asarray(w, dtype=np.float32)
        w = w / (np.linalg.norm(w) + 1e-12)
        cov = activation_covariance(A[fit_idx])
        rms_l = float(np.sqrt(np.square(A[rms_sel]).mean(axis=1)).mean())
        w_by_l[l] = w
        rms_by_l[l] = rms_l
        probe_acc_by_l[l] = float(meta["heldout_accuracy"])
        nulls_by_l[l] = covariance_matched_directions(w, args.n_null, cov, seed=args.sample_seed + 100 * l)

    # Baseline (alpha=0)
    base_smis, base_term, _ = generate_steered(
        model,
        tok,
        args.n_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.generation_seed,
        batch_size=args.gen_batch_size,
        device=args.device,
    )
    baseline_res = evaluate_valence_arm(base_smis, base_term, 0.0, pool=pool)

    def run_regime(regime_name: str, layers_used: list[int], gate_mode: str, run_nulls: bool) -> dict:
        k_l = len(layers_used)
        scale_k = 1.0 / math.sqrt(k_l)
        out = {"layers": layers_used, "gate_mode": gate_mode, "probe": {"0": baseline_res}, "cov_null": {}}
        for alpha in VALENCE_ALPHAS:
            if alpha == 0.0:
                continue
            akey = f"{alpha:+g}"
            edits = {
                l: torch.tensor(w_by_l[l], dtype=torch.float32, device=args.device)
                * (alpha * scale_k * width_scale * rms_by_l[l])
                for l in layers_used
            }
            smis, terms, gfrac = generate_steered(
                model,
                tok,
                args.n_samples,
                layer_edits=edits,
                gate_mode=gate_mode,
                table=table,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.generation_seed,
                batch_size=args.gen_batch_size,
                device=args.device,
            )
            out["probe"][akey] = evaluate_valence_arm(smis, terms, gfrac, pool=pool)

            if run_nulls and args.n_null > 0:
                draws = []
                for r in range(args.n_null):
                    n_edits = {
                        l: torch.tensor(nulls_by_l[l][r], dtype=torch.float32, device=args.device)
                        * (alpha * scale_k * width_scale * rms_by_l[l])
                        for l in layers_used
                    }
                    ns, nt, ng = generate_steered(
                        model,
                        tok,
                        args.n_samples,
                        layer_edits=n_edits,
                        gate_mode=gate_mode,
                        table=table,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        seed=args.generation_seed,
                        batch_size=args.gen_batch_size,
                        device=args.device,
                    )
                    draws.append(evaluate_valence_arm(ns, nt, ng, pool=pool))

                def mean_std(field):
                    vals = [d[field] for d in draws if d[field] is not None]
                    if not vals:
                        return {"mean": None, "std": None}
                    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

                out["cov_null"][akey] = {
                    "n_draws": len(draws),
                    "complete_valid_rate": mean_std("complete_valid_rate"),
                    "termination_rate": mean_std("termination_rate"),
                    "multiple_fraction_valid": mean_std("multiple_fraction_valid"),
                    "multiple_bonds_per_attempt": mean_std("multiple_bonds_per_attempt"),
                    "double_count_per_attempt": mean_std("double_count_per_attempt"),
                    "triple_count_per_attempt": mean_std("triple_count_per_attempt"),
                    "gate_active_fraction": mean_std("gate_active_fraction"),
                }
        return out

    regimes = {
        "single_const": run_regime("single_const", [1], "none", run_nulls=True),
        "single_gated": run_regime("single_gated", [1], "val_ge2", run_nulls=True),
        "single_gated_nonarom": run_regime("single_gated_nonarom", [1], "val_ge2_nonarom", run_nulls=False),
        "multi_const": run_regime("multi_const", mid_layers, "none", run_nulls=True),
        "multi_gated": run_regime("multi_gated", mid_layers, "val_ge2", run_nulls=True),
        "multi_gated_nonarom": run_regime("multi_gated_nonarom", mid_layers, "val_ge2_nonarom", run_nulls=False),
    }

    return {
        "mid_layers": mid_layers,
        "layer_rms": {str(l): rms_by_l[l] for l in mid_layers},
        "layer_probe_accuracy": {str(l): probe_acc_by_l[l] for l in mid_layers},
        "baseline": baseline_res,
        "regimes": regimes,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Global Property Steering (MolLogP, TPSA, NumAromaticRings)
# ---------------------------------------------------------------------------

def _eval_prop_chunk(args):
    smis, terms = args
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s, t in zip(smis, terms):
        err_cat = classify_grammar(s, bool(t))
        if t and s:
            m = Chem.MolFromSmiles(s)
        else:
            m = None
        valid = bool(t and m is not None and m.GetNumAtoms() > 0)
        if valid:
            canon = Chem.MolToSmiles(m)
            logp = float(Crippen.MolLogP(m))
            tpsa = float(Descriptors.TPSA(m))
            arom = float(rdMolDescriptors.CalcNumAromaticRings(m))
            ha = int(sum(a.GetAtomicNum() > 1 for a in m.GetAtoms()))
            fp = gen.GetFingerprint(m)
            out.append((True, bool(t), err_cat, len(tokenize(s)), ha, canon, logp, tpsa, arom, fp))
        else:
            n_tok = len(tokenize(s)) if s else 0
            out.append((False, bool(t), err_cat, n_tok, None, None, None, None, None, None))
    return out


def compute_property_records(
    smis: list[str],
    terms: list[bool],
    pool: ProcessPoolExecutor | None = None,
) -> list[tuple]:
    n = len(smis)
    if pool is not None and n >= 256:
        n_chunks = min(8, os.cpu_count() or 4)
        step = math.ceil(n / n_chunks)
        chunks = [(smis[i : i + step], terms[i : i + step]) for i in range(0, n, step)]
        recs = []
        for c in pool.map(_eval_prop_chunk, chunks):
            recs.extend(c)
        return recs
    return _eval_prop_chunk((smis, terms))


def _check_thresh(val: float, thresh: float, op: str) -> bool:
    if op == ">":
        return val > thresh
    if op == ">=":
        return val >= thresh
    if op == "<":
        return val < thresh
    if op == "<=":
        return val <= thresh
    raise ValueError(op)


def summarize_property_records(
    records: list[tuple],
    prop_name: str,
    baseline_mean: float | None,
    all_canon_collector: set[str] | None = None,
) -> tuple[dict, list[str]]:
    """Summarize one arm's property records; returns (metrics_dict, unique_canon_list)."""
    prop_idx = {"MolLogP": 6, "TPSA": 7, "NumAromaticRings": 8}[prop_name]
    spec = PROPERTY_SPECS[prop_name]

    n = len(records)
    n_term = sum(1 for r in records if r[1])
    valid_recs = [r for r in records if r[0]]
    n_valid = len(valid_recs)
    counts = Counter(r[2] for r in records)

    vals = np.array([r[prop_idx] for r in valid_recs], dtype=float) if n_valid else np.array([], dtype=float)
    canons = [r[5] for r in valid_recs]
    uniq_canons = list(set(canons))
    if all_canon_collector is not None:
        all_canon_collector.update(uniq_canons)

    fps = [r[9] for r in valid_recs[:500]]
    if len(fps) > 1:
        sims = []
        for i in range(len(fps)):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 : i + 200]))
        pw_tan = float(np.mean(sims)) if sims else float("nan")
        int_div = float(1.0 - pw_tan) if not math.isnan(pw_tan) else None
    else:
        pw_tan = None
        int_div = None

    high_hits = sum(1 for v in vals if _check_thresh(v, spec["high_thresh"], spec["high_op"]))
    low_hits = sum(1 for v in vals if _check_thresh(v, spec["low_thresh"], spec["low_op"]))
    high_alt_hits = (
        sum(1 for v in vals if _check_thresh(v, spec["high_alt_thresh"], spec["high_op"]))
        if spec["high_alt_thresh"] is not None
        else None
    )

    mean_val = float(vals.mean()) if n_valid else None
    std_val = float(vals.std()) if n_valid else None
    delta_val = (mean_val - baseline_mean) if (mean_val is not None and baseline_mean is not None) else None

    metrics = {
        "n_attempted": n,
        "n_terminated": n_term,
        "termination_rate": float(n_term / max(n, 1)),
        "n_complete_valid": n_valid,
        "valid_rate": float(n_valid / max(n, 1)),
        "mean_prop_valid": mean_val,
        "std_prop_valid": std_val,
        "delta_prop_valid": delta_val,
        "target_hit_high_yield_per_attempt": float(high_hits / max(n, 1)),
        "target_hit_low_yield_per_attempt": float(low_hits / max(n, 1)),
        "target_hit_high_rate_valid": float(high_hits / n_valid) if n_valid else None,
        "target_hit_low_rate_valid": float(low_hits / n_valid) if n_valid else None,
        "target_hit_high_alt_yield_per_attempt": (
            float(high_alt_hits / max(n, 1)) if high_alt_hits is not None else None
        ),
        "uniqueness": float(len(uniq_canons) / n_valid) if n_valid else None,
        "pairwise_tanimoto_mean": pw_tan if (pw_tan is not None and not math.isnan(pw_tan)) else None,
        "internal_diversity": int_div,
        "mean_decoded_tokens": float(np.mean([r[3] for r in records])),
        "mean_heavy_atoms_valid": float(np.mean([r[4] for r in valid_recs])) if n_valid else None,
        "error_attempt_fractions": {cat: float(counts.get(cat, 0) / max(n, 1)) for cat in CATEGORIES},
        "novelty_vs_train": None,  # filled in single pass over train.smi
    }
    return metrics, uniq_canons


def fill_novelty_from_train(
    train_smi_path: Path,
    arm_canon_registry: list[tuple[dict, list[str]]],
    all_canons: set[str],
) -> None:
    """Compute novelty against full train.smi in ONE ~1.5s streaming pass."""
    query_bytes = {s.encode("ascii", errors="ignore") for s in all_canons}
    seen_in_train: set[bytes] = set()
    with open(train_smi_path, "rb") as fh:
        for line in fh:
            b = line.rstrip(b"\r\n")
            if b in query_bytes:
                seen_in_train.add(b)
    seen_str = {b.decode("ascii") for b in seen_in_train}

    for metrics_dict, uniq_canons in arm_canon_registry:
        if not uniq_canons:
            metrics_dict["novelty_vs_train"] = None
        else:
            novel = sum(1 for c in uniq_canons if c not in seen_str)
            metrics_dict["novelty_vs_train"] = float(novel / len(uniq_canons))


@torch.no_grad()
def measure_step_kl_on_eval(
    model: MolecularTransformer,
    x_eval: torch.Tensor,
    content_mask: torch.Tensor,
    clean_logp: torch.Tensor,
    clean_p: torch.Tensor,
    *,
    layer_edits: dict[int, torch.Tensor] | None = None,
    logit_bias: torch.Tensor | None = None,
) -> float:
    """Mean next-token KL D_KL(p_clean || p_steered) over content positions of 256 val molecules."""
    hooks = None
    if layer_edits:
        hooks = HookManager()
        for l, ev in layer_edits.items():
            hooks.edit(resid_post(l), lambda x, e=ev: x + e.to(x.dtype))
    logits = model(x_eval, hooks=hooks).float()
    if logit_bias is not None:
        logits = logits + logit_bias.float()
    steered_logp = F.log_softmax(logits, dim=-1)
    kl = (clean_p * (clean_logp - steered_logp)).sum(dim=-1).clamp_min(0.0)
    return float(kl[content_mask].mean())


@torch.no_grad()
def fit_property_directions(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    smiles_4k: list[str],
    device: str,
    seed: int = 4300,
):
    """Collect mean-pooled content activations across all layers and fit RidgeCV directions."""
    n_layers = model.cfg.n_layers
    n_ctx = model.cfg.n_ctx
    V = tok.vocab_size
    special_ids = {tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id}

    # Compute properties and keep valid molecules
    valid_smiles = []
    props = {"MolLogP": [], "TPSA": [], "NumAromaticRings": []}
    for s in smiles_4k:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        valid_smiles.append(s)
        props["MolLogP"].append(float(Crippen.MolLogP(m)))
        props["TPSA"].append(float(Descriptors.TPSA(m)))
        props["NumAromaticRings"].append(float(rdMolDescriptors.CalcNumAromaticRings(m)))

    N = len(valid_smiles)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_tr = int(0.7 * N)
    tr_idx = perm[:n_tr]
    te_idx = perm[n_tr:]
    tr_set = set(tr_idx.tolist())

    hook_names = [resid_post(l) for l in range(n_layers)]
    pooled_by_l = [np.zeros((N, model.cfg.d_model), dtype=np.float32) for _ in range(n_layers)]
    token_bag = np.zeros((N, V), dtype=np.float32)
    # Sample content-token activations from train split for covariance and RMS
    content_samples_by_l: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    rms_sum_by_l = np.zeros(n_layers, dtype=np.float64)
    rms_cnt_by_l = np.zeros(n_layers, dtype=np.int64)

    for s0 in range(0, N, 128):
        chunk = valid_smiles[s0 : s0 + 128]
        ids = tight_batch(chunk, tok, n_ctx)
        x = torch.tensor(ids, dtype=torch.long, device=device)
        hm = HookManager().capture(*hook_names)
        model(x, hooks=hm)

        # Content mask: not PAD, BOS, EOS
        mask = (x != tok.pad_id) & (x != tok.bos_id) & (x != tok.eos_id)  # (B, T)
        mask_f = mask.float().unsqueeze(-1)
        counts = mask_f.sum(dim=1).clamp_min(1.0)  # (B, 1)

        # Bag of tokens (normalized frequency per molecule)
        x_cpu = x.cpu().numpy()
        m_cpu = mask.cpu().numpy()
        for bi in range(len(chunk)):
            gi = s0 + bi
            tids = x_cpu[bi][m_cpu[bi]]
            if len(tids) > 0:
                np.add.at(token_bag[gi], tids, 1.0)
                token_bag[gi] /= float(len(tids))

        for l in range(n_layers):
            h = hm[resid_post(l)].float()  # (B, T, D)
            pooled = (h * mask_f).sum(dim=1) / counts  # (B, D)
            pooled_by_l[l][s0 : s0 + len(chunk)] = pooled.cpu().numpy()

            # RMS and covariance samples on train molecules
            per_pos_rms = h.pow(2).mean(dim=-1).sqrt()  # (B, T)
            h_np = h.cpu().numpy()
            rms_np = per_pos_rms.cpu().numpy()
            for bi in range(len(chunk)):
                gi = s0 + bi
                if gi in tr_set:
                    pos = np.flatnonzero(m_cpu[bi])
                    if len(pos) > 0:
                        rms_sum_by_l[l] += float(rms_np[bi, pos].sum())
                        rms_cnt_by_l[l] += len(pos)
                        # Take up to 8 content positions per train molecule (~22k total)
                        if len(pos) > 8:
                            pos = pos[np.linspace(0, len(pos) - 1, 8, dtype=int)]
                        content_samples_by_l[l].append(h_np[bi, pos])

    rms_by_l = [float(rms_sum_by_l[l] / max(rms_cnt_by_l[l], 1)) for l in range(n_layers)]
    cov_by_l = [activation_covariance(np.concatenate(content_samples_by_l[l], axis=0)) for l in range(n_layers)]

    # Unembedding matrix + ln_f weight for logit-lens projection
    W_u = (model.embed.weight if model.lm_head is None else model.lm_head.weight).detach().float().cpu().numpy()
    ln_f_w = model.ln_f.weight.detach().float().cpu().numpy()
    content_vocab_mask = np.array([i not in special_ids for i in range(V)], dtype=bool)

    prop_models = {}
    for pname, y_list in props.items():
        y = np.asarray(y_list, dtype=np.float64)
        layer_dirs = []
        layer_r2 = []
        layer_rmse = []
        for l in range(n_layers):
            X = pooled_by_l[l]
            reg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
            reg.fit(X[tr_idx], y[tr_idx])
            pred = reg.predict(X[te_idx])
            ss_res = float(((y[te_idx] - pred) ** 2).sum())
            ss_tot = float(((y[te_idx] - y[te_idx].mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse = float(np.sqrt(((y[te_idx] - pred) ** 2).mean()))
            w = reg.coef_.astype(np.float64)
            w = w / (np.linalg.norm(w) + 1e-12)
            # Ensure positive correlation orientation
            if np.corrcoef(X[tr_idx] @ w, y[tr_idx])[0, 1] < 0:
                w = -w
            layer_dirs.append(w.astype(np.float32))
            layer_r2.append(float(r2))
            layer_rmse.append(rmse)

        mid_layers = list(range(1, n_layers - 1))
        best_mid = max(mid_layers, key=lambda l: layer_r2[l])

        # Logit bias 1: Unembedding projection of final-layer direction
        b_unemb = W_u @ (ln_f_w * layer_dirs[n_layers - 1].astype(np.float64))
        b_unemb[~content_vocab_mask] = 0.0
        b_unemb[content_vocab_mask] -= b_unemb[content_vocab_mask].mean()
        b_unemb = b_unemb / (np.linalg.norm(b_unemb) + 1e-12)
        if np.corrcoef(token_bag[tr_idx] @ b_unemb, y[tr_idx])[0, 1] < 0:
            b_unemb = -b_unemb

        # Logit bias 2: Bag-of-tokens Ridge regression on same 2,800 train molecules
        treg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        treg.fit(token_bag[tr_idx], y[tr_idx])
        tpred = treg.predict(token_bag[te_idx])
        t_r2 = 1.0 - float(((y[te_idx] - tpred) ** 2).sum()) / float(((y[te_idx] - y[te_idx].mean()) ** 2).sum())
        b_tok = treg.coef_.astype(np.float64)
        b_tok[~content_vocab_mask] = 0.0
        b_tok[content_vocab_mask] -= b_tok[content_vocab_mask].mean()
        b_tok = b_tok / (np.linalg.norm(b_tok) + 1e-12)
        if np.corrcoef(token_bag[tr_idx] @ b_tok, y[tr_idx])[0, 1] < 0:
            b_tok = -b_tok

        prop_models[pname] = {
            "layer_dirs": layer_dirs,
            "layer_r2": layer_r2,
            "layer_rmse": layer_rmse,
            "best_mid_layer": best_mid,
            "logit_bias_unemb": b_unemb.astype(np.float32),
            "logit_bias_token": b_tok.astype(np.float32),
            "token_bag_r2": float(t_r2),
            "unemb_token_bias_cosine": float(np.dot(b_unemb, b_tok)),
            "train_mean": float(y[tr_idx].mean()),
            "train_std": float(y[tr_idx].std()),
        }

    # Build 256-molecule KL evaluation tensor from held-out split
    kl_smiles = [valid_smiles[i] for i in te_idx[:256]]
    x_kl = torch.tensor(tight_batch(kl_smiles, tok, n_ctx), dtype=torch.long, device=device)
    kl_mask = (x_kl != tok.pad_id) & (x_kl != tok.bos_id) & (x_kl != tok.eos_id)
    clean_lg = model(x_kl).float()
    clean_logp = F.log_softmax(clean_lg, dim=-1)
    clean_p = clean_logp.exp()

    return {
        "n_molecules": N,
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "rms_by_layer": rms_by_l,
        "cov_by_layer": cov_by_l,
        "prop_models": prop_models,
        "kl_eval": (x_kl, kl_mask, clean_logp, clean_p),
    }


def compute_best_of_n(
    pool_records: list[tuple],
    prop_name: str,
    n_groups: int,
    baseline_mean: float | None,
    arm_canon_registry: list[tuple[dict, list[str]]],
    all_canons: set[str],
) -> dict:
    """Evaluate Best-of-N reranking (high and low) for N in [1, 2, 4, 8, 16] over n_groups independent slots."""
    prop_idx = {"MolLogP": 6, "TPSA": 7, "NumAromaticRings": 8}[prop_name]
    spec = PROPERTY_SPECS[prop_name]
    out = {"high": {}, "low": {}}

    for direction in ("high", "low"):
        maximize = direction == "high"
        for N in BEST_OF_NS:
            selected = []
            for gi in range(n_groups):
                group = pool_records[gi * N : (gi + 1) * N]
                valids = [r for r in group if r[0]]
                if not valids:
                    selected.append(group[0])
                else:
                    best = max(valids, key=lambda r: r[prop_idx]) if maximize else min(valids, key=lambda r: r[prop_idx])
                    selected.append(best)
            m, uniq = summarize_property_records(selected, prop_name, baseline_mean, all_canons)
            # Add compute-normalized per-attempt yield (dividing by N total attempts per output)
            m["compute_multiplier_N"] = N
            m["target_hit_high_yield_per_total_attempt"] = m["target_hit_high_yield_per_attempt"] / N
            m["target_hit_low_yield_per_total_attempt"] = m["target_hit_low_yield_per_attempt"] / N
            if m["target_hit_high_alt_yield_per_attempt"] is not None:
                m["target_hit_high_alt_yield_per_total_attempt"] = m["target_hit_high_alt_yield_per_attempt"] / N
            arm_canon_registry.append((m, uniq))
            out[direction][f"N={N}"] = m
    return out


@torch.no_grad()
def run_experiment2_properties(
    model: MolecularTransformer,
    tok: SmilesTokenizer,
    smiles_4k: list[str],
    train_smi_path: Path,
    args,
    pool: ProcessPoolExecutor | None = None,
) -> dict:
    """Global property steering across MolLogP, TPSA, and NumAromaticRings."""
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    width_scale = math.sqrt(d_model / 512.0)
    mid_layers = list(range(1, n_layers - 1))
    K = len(mid_layers)
    scale_k = 1.0 / math.sqrt(K)

    fitted = fit_property_directions(model, tok, smiles_4k, device=args.device, seed=args.prop_sample_seed)
    x_kl, kl_mask, clean_logp, clean_p = fitted["kl_eval"]

    # Generate unsteered pool of 16 * n_samples (16,384) ONCE for baseline + Best-of-N!
    pool_size = max(BEST_OF_NS) * args.n_samples
    pool_smis, pool_terms = [], []
    n_batches = math.ceil(pool_size / args.gen_batch_size)
    for bi in range(n_batches):
        nb = min(args.gen_batch_size, pool_size - len(pool_smis))
        smis, terms, _ = generate_steered(
            model,
            tok,
            nb,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.generation_seed + bi,
            batch_size=nb,
            device=args.device,
        )
        pool_smis.extend(smis)
        pool_terms.extend(terms)
    pool_records = compute_property_records(pool_smis, pool_terms, pool=pool)
    base_records = pool_records[: args.n_samples]

    arm_canon_registry: list[tuple[dict, list[str]]] = []
    all_canons: set[str] = set()
    prop_results = {}

    for pname in ("MolLogP", "TPSA", "NumAromaticRings"):
        pm = fitted["prop_models"][pname]
        best_l = pm["best_mid_layer"]
        w_best = pm["layer_dirs"][best_l]
        rms_best = fitted["rms_by_layer"][best_l]
        null_dirs = covariance_matched_directions(
            w_best, args.n_null, fitted["cov_by_layer"][best_l], seed=args.prop_sample_seed + hash(pname) % 10000
        )

        # Baseline summary (first n_samples from unsteered pool, exact generation_seed!)
        base_m, base_uniq = summarize_property_records(base_records, pname, None, all_canons)
        base_m["delta_prop_valid"] = 0.0
        base_m["step_kl"] = 0.0
        arm_canon_registry.append((base_m, base_uniq))
        base_mean = base_m["mean_prop_valid"]

        def run_act_arm(layer_edits: dict[int, torch.Tensor]):
            smis, terms, _ = generate_steered(
                model,
                tok,
                args.n_samples,
                layer_edits=layer_edits,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.generation_seed,
                batch_size=args.gen_batch_size,
                device=args.device,
            )
            recs = compute_property_records(smis, terms, pool=pool)
            m, uniq = summarize_property_records(recs, pname, base_mean, all_canons)
            m["step_kl"] = measure_step_kl_on_eval(
                model, x_kl, kl_mask, clean_logp, clean_p, layer_edits=layer_edits
            )
            arm_canon_registry.append((m, uniq))
            return m

        def run_logit_arm(bias_vec: np.ndarray, beta: float):
            b_t = torch.tensor(bias_vec * beta, dtype=torch.float32, device=args.device)
            smis, terms, _ = generate_steered(
                model,
                tok,
                args.n_samples,
                logit_bias=b_t,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                seed=args.generation_seed,
                batch_size=args.gen_batch_size,
                device=args.device,
            )
            recs = compute_property_records(smis, terms, pool=pool)
            m, uniq = summarize_property_records(recs, pname, base_mean, all_canons)
            m["step_kl"] = measure_step_kl_on_eval(
                model, x_kl, kl_mask, clean_logp, clean_p, logit_bias=b_t
            )
            arm_canon_registry.append((m, uniq))
            return m

        act_single = {"0": base_m}
        act_multi = {"0": base_m}
        cov_null = {}

        for alpha in PROP_ALPHAS:
            if alpha == 0.0:
                continue
            akey = f"{alpha:+g}"
            # Single-layer activation steering at best middle layer
            e_single = {
                best_l: torch.tensor(w_best, dtype=torch.float32, device=args.device)
                * (alpha * width_scale * rms_best)
            }
            act_single[akey] = run_act_arm(e_single)

            # Multi-layer activation steering across 1..L-2 (matched total L2 norm)
            e_multi = {
                l: torch.tensor(pm["layer_dirs"][l], dtype=torch.float32, device=args.device)
                * (alpha * scale_k * width_scale * fitted["rms_by_layer"][l])
                for l in mid_layers
            }
            act_multi[akey] = run_act_arm(e_multi)

            # Covariance-matched random directions at best_l
            if args.n_null > 0:
                draws = []
                for r in range(args.n_null):
                    e_null = {
                        best_l: torch.tensor(null_dirs[r], dtype=torch.float32, device=args.device)
                        * (alpha * width_scale * rms_best)
                    }
                    draws.append(run_act_arm(e_null))

                def agg(field):
                    vs = [d[field] for d in draws if d[field] is not None]
                    return {"mean": float(np.mean(vs)), "std": float(np.std(vs))} if vs else {"mean": None, "std": None}

                cov_null[akey] = {
                    "n_draws": len(draws),
                    "valid_rate": agg("valid_rate"),
                    "mean_prop_valid": agg("mean_prop_valid"),
                    "delta_prop_valid": agg("delta_prop_valid"),
                    "target_hit_high_yield_per_attempt": agg("target_hit_high_yield_per_attempt"),
                    "target_hit_low_yield_per_attempt": agg("target_hit_low_yield_per_attempt"),
                    "internal_diversity": agg("internal_diversity"),
                    "step_kl": agg("step_kl"),
                    "draws": draws,
                }

        # Logit-level steering (unembedding projection & token regression)
        logit_unemb = {"0": base_m}
        logit_token = {"0": base_m}
        for beta in LOGIT_BETAS:
            if beta == 0.0:
                continue
            bkey = f"{beta:+g}"
            logit_unemb[bkey] = run_logit_arm(pm["logit_bias_unemb"], beta)
            logit_token[bkey] = run_logit_arm(pm["logit_bias_token"], beta)

        # Best-of-N reranking
        best_of_n = compute_best_of_n(
            pool_records, pname, args.n_samples, base_mean, arm_canon_registry, all_canons
        )

        prop_results[pname] = {
            "spec": PROPERTY_SPECS[pname],
            "probe_r2_by_layer": pm["layer_r2"],
            "probe_rmse_by_layer": pm["layer_rmse"],
            "best_mid_layer": best_l,
            "token_bag_r2": pm["token_bag_r2"],
            "unemb_token_bias_cosine": pm["unemb_token_bias_cosine"],
            "baseline": base_m,
            "act_single": act_single,
            "act_multi": act_multi,
            "cov_null": cov_null,
            "logit_unemb": logit_unemb,
            "logit_token": logit_token,
            "best_of_n": best_of_n,
        }

    # Fill novelty_vs_train for ALL arms in a single pass over train.smi (~1.5s)
    if train_smi_path.exists():
        fill_novelty_from_train(train_smi_path, arm_canon_registry, all_canons)
        # Re-aggregate cov_null novelty means after filling draws
        for pname, pr in prop_results.items():
            for akey, cinfo in pr["cov_null"].items():
                vs = [d["novelty_vs_train"] for d in cinfo["draws"] if d["novelty_vs_train"] is not None]
                cinfo["novelty_vs_train"] = (
                    {"mean": float(np.mean(vs)), "std": float(np.std(vs))} if vs else {"mean": None, "std": None}
                )
                # Drop individual draws from JSON to keep file compact while preserving all stats
                del cinfo["draws"]

    return {
        "n_calib_molecules": fitted["n_molecules"],
        "n_train": fitted["n_train"],
        "n_test": fitted["n_test"],
        "mid_layers": mid_layers,
        "rms_by_layer": fitted["rms_by_layer"],
        "properties": prop_results,
    }


def evaluate_checkpoint(
    ckpt_path: Path,
    data_dir: Path,
    smiles_2048: list[str],
    smiles_4k: list[str],
    tok: SmilesTokenizer,
    table: TokenMetadataTable,
    args,
    pool: ProcessPoolExecutor | None = None,
) -> dict:
    t0 = time.time()
    cfg = ModelConfig.load(str(ckpt_path.parent / "model_config.json"))
    model = MolecularTransformer(cfg).to(args.device).eval()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])

    kv_diff = verify_kv_cached_gated_equivalence(model, tok, args.device)
    t1 = time.time()
    exp1 = run_experiment1_valence(model, tok, smiles_2048, table, args, pool=pool)
    t2 = time.time()
    exp2 = run_experiment2_properties(model, tok, smiles_4k, data_dir / "train.smi", args, pool=pool)
    t3 = time.time()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(ckpt_path),
        "model_config": cfg.__dict__,
        "kv_gated_verification_max_diff": kv_diff,
        "settings": {
            "n_samples": args.n_samples,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "generation_seed": args.generation_seed,
            "sample_seed": args.sample_seed,
            "prop_sample_seed": args.prop_sample_seed,
            "n_null": args.n_null,
        },
        "timing_sec": {
            "setup_verify": round(t1 - t0, 2),
            "exp1_valence": round(t2 - t1, 2),
            "exp2_properties": round(t3 - t2, 2),
            "total": round(t3 - t0, 2),
        },
        "exp1_valence_steering": exp1,
        "exp2_property_steering": exp2,
    }


def build_summary(per_ckpt: dict[str, dict]) -> dict:
    """Aggregate Pareto frontier metrics across model sizes and seeds."""
    by_size: dict[str, list[dict]] = {}
    for key, res in per_ckpt.items():
        size = key.split("_seed")[0]
        by_size.setdefault(size, []).append(res)

    summary = {"checkpoints": sorted(per_ckpt.keys()), "by_size": {}}
    for size, items in by_size.items():
        # Exp 1 summary
        exp1_sum = {}
        for regime in (
            "single_const",
            "single_gated",
            "single_gated_nonarom",
            "multi_const",
            "multi_gated",
            "multi_gated_nonarom",
        ):
            exp1_sum[regime] = {"probe": {}, "cov_null": {}}
            for akey in ("-8", "-4", "-2", "0", "+2", "+4", "+8"):
                arms = [it["exp1_valence_steering"]["regimes"][regime]["probe"][akey] for it in items]
                exp1_sum[regime]["probe"][akey] = {
                    "valid_rate_mean": float(np.mean([a["complete_valid_rate"] for a in arms])),
                    "valid_rate_std": float(np.std([a["complete_valid_rate"] for a in arms])),
                    "multiple_fraction_valid_mean": float(
                        np.nanmean([a["multiple_fraction_valid"] if a["multiple_fraction_valid"] is not None else np.nan for a in arms])
                    ),
                    "multiple_bonds_per_attempt_mean": float(np.mean([a["multiple_bonds_per_attempt"] for a in arms])),
                    "multiple_bonds_per_attempt_std": float(np.std([a["multiple_bonds_per_attempt"] for a in arms])),
                    "termination_rate_mean": float(np.mean([a["termination_rate"] for a in arms])),
                    "gate_active_fraction_mean": float(np.mean([a["gate_active_fraction"] for a in arms])),
                }
                if akey != "0" and akey in items[0]["exp1_valence_steering"]["regimes"][regime]["cov_null"]:
                    narms = [it["exp1_valence_steering"]["regimes"][regime]["cov_null"][akey] for it in items]
                    exp1_sum[regime]["cov_null"][akey] = {
                        "valid_rate_mean": float(np.mean([a["complete_valid_rate"]["mean"] for a in narms])),
                        "multiple_fraction_valid_mean": float(
                            np.nanmean([
                                a["multiple_fraction_valid"]["mean"]
                                if a["multiple_fraction_valid"]["mean"] is not None
                                else np.nan
                                for a in narms
                            ])
                        ),
                        "multiple_bonds_per_attempt_mean": float(
                            np.mean([a["multiple_bonds_per_attempt"]["mean"] for a in narms])
                        ),
                    }

        # Exp 2 summary
        exp2_sum = {}
        for pname in ("MolLogP", "TPSA", "NumAromaticRings"):
            p_out = {
                "best_mid_layers": [it["exp2_property_steering"]["properties"][pname]["best_mid_layer"] for it in items],
                "best_mid_r2_mean": float(
                    np.mean([
                        it["exp2_property_steering"]["properties"][pname]["probe_r2_by_layer"][
                            it["exp2_property_steering"]["properties"][pname]["best_mid_layer"]
                        ]
                        for it in items
                    ])
                ),
                "token_bag_r2_mean": float(
                    np.mean([it["exp2_property_steering"]["properties"][pname]["token_bag_r2"] for it in items])
                ),
            }
            for method, keys in (
                ("act_single", ["-8", "-4", "-2", "-1", "0", "+1", "+2", "+4", "+8"]),
                ("act_multi", ["-8", "-4", "-2", "-1", "0", "+1", "+2", "+4", "+8"]),
                ("logit_unemb", ["-16", "-8", "-4", "-2", "-1", "-0.5", "0", "+0.5", "+1", "+2", "+4", "+8", "+16"]),
                ("logit_token", ["-16", "-8", "-4", "-2", "-1", "-0.5", "0", "+0.5", "+1", "+2", "+4", "+8", "+16"]),
            ):
                p_out[method] = {}
                for k in keys:
                    arms = [it["exp2_property_steering"]["properties"][pname][method][k] for it in items]
                    def mavg(field):
                        vs = [a[field] for a in arms if a[field] is not None]
                        return float(np.mean(vs)) if vs else None
                    p_out[method][k] = {
                        "valid_rate": mavg("valid_rate"),
                        "mean_prop_valid": mavg("mean_prop_valid"),
                        "delta_prop_valid": mavg("delta_prop_valid"),
                        "target_hit_high_yield_per_attempt": mavg("target_hit_high_yield_per_attempt"),
                        "target_hit_low_yield_per_attempt": mavg("target_hit_low_yield_per_attempt"),
                        "target_hit_high_alt_yield_per_attempt": mavg("target_hit_high_alt_yield_per_attempt"),
                        "internal_diversity": mavg("internal_diversity"),
                        "uniqueness": mavg("uniqueness"),
                        "novelty_vs_train": mavg("novelty_vs_train"),
                        "step_kl": mavg("step_kl"),
                    }
            # Cov-null summary for Exp 2
            p_out["cov_null"] = {}
            for k in ("-8", "-4", "-2", "-1", "+1", "+2", "+4", "+8"):
                narms = [it["exp2_property_steering"]["properties"][pname]["cov_null"][k] for it in items]
                def mavg_n(field):
                    vs = [a[field]["mean"] for a in narms if a[field]["mean"] is not None]
                    return float(np.mean(vs)) if vs else None
                p_out["cov_null"][k] = {
                    "valid_rate": mavg_n("valid_rate"),
                    "delta_prop_valid": mavg_n("delta_prop_valid"),
                    "target_hit_high_yield_per_attempt": mavg_n("target_hit_high_yield_per_attempt"),
                    "target_hit_low_yield_per_attempt": mavg_n("target_hit_low_yield_per_attempt"),
                    "internal_diversity": mavg_n("internal_diversity"),
                    "novelty_vs_train": mavg_n("novelty_vs_train"),
                    "step_kl": mavg_n("step_kl"),
                }
            # Best-of-N summary
            bon_out = {"high": {}, "low": {}}
            for d in ("high", "low"):
                for N in BEST_OF_NS:
                    nk = f"N={N}"
                    arms = [it["exp2_property_steering"]["properties"][pname]["best_of_n"][d][nk] for it in items]
                    def mavg_b(field):
                        vs = [a[field] for a in arms if a[field] is not None]
                        return float(np.mean(vs)) if vs else None
                    bon_out[d][nk] = {
                        "valid_rate": mavg_b("valid_rate"),
                        "mean_prop_valid": mavg_b("mean_prop_valid"),
                        "delta_prop_valid": mavg_b("delta_prop_valid"),
                        "target_hit_high_yield_per_selected": mavg_b("target_hit_high_yield_per_attempt"),
                        "target_hit_low_yield_per_selected": mavg_b("target_hit_low_yield_per_attempt"),
                        "target_hit_high_yield_per_total_attempt": mavg_b("target_hit_high_yield_per_total_attempt"),
                        "target_hit_low_yield_per_total_attempt": mavg_b("target_hit_low_yield_per_total_attempt"),
                        "internal_diversity": mavg_b("internal_diversity"),
                        "novelty_vs_train": mavg_b("novelty_vs_train"),
                    }
            p_out["best_of_n"] = bon_out
            exp2_sum[pname] = p_out

        summary["by_size"][size] = {
            "n_seeds": len(items),
            "n_layers": items[0]["model_config"]["n_layers"],
            "d_model": items[0]["model_config"]["d_model"],
            "exp1_valence_summary": exp1_sum,
            "exp2_property_summary": exp2_sum,
        }
    return summary


def main():
    from molcircuits import paths as _paths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=None, help="Single checkpoint path")
    ap.add_argument("--data", type=Path, default=Path(_paths.data()))
    ap.add_argument("--out", type=Path, default=Path("results/steering"))
    ap.add_argument("--seed", type=int, default=43000, help="Base generation seed (alias for --generation-seed)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--all-grid", action="store_true", help="Run all 9 canonical models in --grid-root")
    ap.add_argument("--grid-root", type=Path, default=Path(_paths.root()) / "grid")
    ap.add_argument("--sizes", nargs="+", default=["small", "base", "large"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--sample-seed", type=int, default=47)
    ap.add_argument("--prop-sample-seed", type=int, default=4300)
    ap.add_argument("--calib-molecules", type=int, default=2048)
    ap.add_argument("--prop-molecules", type=int, default=4000)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=1024)
    ap.add_argument("--gen-batch-size", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    args.generation_seed = args.seed

    args.out.mkdir(parents=True, exist_ok=True)
    tok = SmilesTokenizer.load(str(args.data / "tokenizer.json"))
    table = TokenMetadataTable(tok)

    population = (args.data / "valid.smi").read_text().splitlines()
    # 1. Verify incremental online valence tracker on 500 validation SMILES!
    rng_v = np.random.default_rng(12345)
    verify_smiles = [population[i] for i in rng_v.choice(len(population), 500, replace=False)]
    v_info = verify_incremental_tracker(verify_smiles, tok)
    print(f"Incremental valence tracker verified: {v_info}", flush=True)

    # Sample calibration sets
    rng_val = np.random.default_rng(args.sample_seed)
    smiles_2048 = [population[i] for i in rng_val.choice(len(population), args.calib_molecules, replace=False)]
    rng_prop = np.random.default_rng(args.prop_sample_seed)
    smiles_4k = [population[i] for i in rng_prop.choice(len(population), args.prop_molecules, replace=False)]

    if args.ckpt is not None:
        tag = args.ckpt.parent.name
        targets = [(tag, args.ckpt)]
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
            print(f"=== Running constructive steering on {name}: {ckpt} ===", flush=True)
            res = evaluate_checkpoint(ckpt, args.data, smiles_2048, smiles_4k, tok, table, args, pool=pool)
            res["tracker_verification"] = v_info
            per_ckpt[name] = res
            out_file = (
                args.out / f"constructive_{name}.json"
                if args.all_grid or args.out.is_dir()
                else args.out
            )
            atomic_json(out_file, res)
            e1 = res["exp1_valence_steering"]["regimes"]
            print(
                f"  [{name}] {res['timing_sec']['total']:.1f}s (e1={res['timing_sec']['exp1_valence']}s, "
                f"e2={res['timing_sec']['exp2_properties']}s) | "
                f"a=+8 valid: const={e1['single_const']['probe']['+8']['complete_valid_rate']:.3f} "
                f"gated={e1['single_gated']['probe']['+8']['complete_valid_rate']:.3f} "
                f"m_gated={e1['multi_gated']['probe']['+8']['complete_valid_rate']:.3f} | "
                f"mult/att(+8): base={res['exp1_valence_steering']['baseline']['multiple_bonds_per_attempt']:.3f} "
                f"const={e1['single_const']['probe']['+8']['multiple_bonds_per_attempt']:.3f} "
                f"gated={e1['single_gated']['probe']['+8']['multiple_bonds_per_attempt']:.3f} "
                f"m_gated={e1['multi_gated']['probe']['+8']['multiple_bonds_per_attempt']:.3f}",
                flush=True,
            )

    if args.all_grid:
        summary = build_summary(per_ckpt)
        atomic_json(args.out / "summary_constructive_steering.json", summary)
        print(f"Saved {args.out / 'summary_constructive_steering.json'}", flush=True)


if __name__ == "__main__":
    main()
