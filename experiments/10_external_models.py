#!/usr/bin/env python3
"""Workstream 4: External Pretrained Chemical LMs Mechanistic Validation.

Runs a standardized mechanistic interpretability suite across public pretrained
SMILES language models on the held-out ZINC20 validation molecules
(`$MOLCIRCUITS_DATA/valid.smi`):

  1. entropy/gpt2_zinc_87m          (Causal SMILES GPT-2, 12L/12H/768d, BPE)
  2. DeepChem/ChemBERTa-77M-MLM     (Masked LM RoBERTa, 3L/12H/384d, PubChem)
  3. DeepChem/ChemBERTa-77M-MTR     (Multi-Task Property RoBERTa, 3L/12H/384d, PubChem)
  4. ibm/MoLFormer-XL-both-10pct    (Rotary Linear-Attn Encoder, 12L/12H/768d, Atom-Regex)
  5. seyonec/ChemBERTa-zinc-base-v1 (Masked LM RoBERTa, 6L/12H/768d, ZINC BPE)

Outputs per-model JSON files and `summary_external_models.json` into
`--out-dir` (default `results/external_models`).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from molcircuits.analysis.valence_probe import within_type_shuffled_labels
from molcircuits.chem.events import control_positions, find_events
from molcircuits.chem.tokenizer import token_spans, tokenize
from molcircuits.chem.valence import (
    BOND_ORDER,
    _charge_of,
    _element_of,
    _explicit_h,
    _is_aromatic_token,
    _remaining,
    is_atom_token,
    valence_labels,
)

RDLogger.DisableLog("rdApp.*")

# Pinned safetensors commit of ibm/MoLFormer-XL-both-10pct compatible with
# transformers==4.50.3 (main branch PR #7 introduced `transformers.masking_utils`
# from transformers>=4.53; weights are identical).
MOLFORMER_PINNED_REV = "7b12d946c181a37f6012b9dc3b002275de070314"

TEST_SMILES_5 = [
    "O=C(NCCc1ccccc1Br)NC[C@@H]1CCCOC1",
    "C[C@H](Cc1ccccc1F)C(=O)N1CC[C@H](N2CCCC2)[C@H]1C",
    "Cc1ccccc1O",
    "Cc2ccccc2O",
    "Cc1ccc(Cl)cc1",
]


@dataclass
class ModelSpec:
    slug: str
    hf_id: str
    kind: str  # "causal" or "encoder"
    trust_remote_code: bool = False
    revision: str | None = None
    prepend_bos_manual: bool = False
    notes: str = ""


MODEL_SPECS = [
    ModelSpec(
        slug="gpt2_zinc_87m",
        hf_id="entropy/gpt2_zinc_87m",
        kind="causal",
        prepend_bos_manual=True,
        notes="Causal SMILES GPT-2 (ZINC, BPE); requires explicit <s> BOS prefix (interior CE 1.08 vs 3.92 without BOS).",
    ),
    ModelSpec(
        slug="chemberta_77m_mlm",
        hf_id="DeepChem/ChemBERTa-77M-MLM",
        kind="encoder",
        notes="PubChem 77M RoBERTa MLM (3L/12H/384d); uses shipped AutoTokenizer matching pretraining.",
    ),
    ModelSpec(
        slug="chemberta_77m_mtr",
        hf_id="DeepChem/ChemBERTa-77M-MTR",
        kind="encoder",
        notes="PubChem 77M RoBERTa Multi-Task Regression on 200 properties (3L/12H/384d).",
    ),
    ModelSpec(
        slug="molformer_xl_10pct",
        hf_id="ibm/MoLFormer-XL-both-10pct",
        kind="encoder",
        trust_remote_code=True,
        revision=MOLFORMER_PINNED_REV,
        notes=(
            f"Rotary linear-attention encoder (12L/12H/768d, atom-regex tokenizer). "
            f"Pinned to revision {MOLFORMER_PINNED_REV[:12]} for transformers==4.50.3 compatibility "
            f"(main branch requires transformers.masking_utils from >=4.53; identical checkpoint weights)."
        ),
    ),
    ModelSpec(
        slug="chemberta_zinc_base_v1",
        hf_id="seyonec/ChemBERTa-zinc-base-v1",
        kind="encoder",
        notes="ZINC RoBERTa MLM (6L/12H/768d, BPE tokenizer — encoder counterpart to GPT-2 ZINC BPE).",
    ),
]


# --------------------------------------------------------------------------- #
# Utilities: Encoding, Offset Alignment, GPU Linear Probes, CKA
# --------------------------------------------------------------------------- #


def encode_with_offsets(tok, smiles: str, spec: ModelSpec, add_eos_causal: bool = True):
    """Encode one SMILES returning (input_ids, offsets, non_special_mask).

    For `gpt2_zinc_87m`, GPT2TokenizerFast does not automatically prepend `<s>`
    or append `</s>`; we explicitly prepend `bos_token_id` (and optionally
    append `eos_token_id`) while preserving exact character offsets into `smiles`.
    """
    if spec.prepend_bos_manual:
        enc = tok(smiles, add_special_tokens=False, return_offsets_mapping=True)
        ids = [tok.bos_token_id] + list(enc["input_ids"])
        offs = [(0, 0)] + list(enc["offset_mapping"])
        mask = [False] + [True] * len(enc["input_ids"])
        if add_eos_causal and tok.eos_token_id is not None:
            ids.append(tok.eos_token_id)
            offs.append((0, 0))
            mask.append(False)
        return ids, offs, mask
    enc = tok(smiles, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = list(enc["offset_mapping"])
    spec_ids = set(tok.all_special_ids)
    mask = [(i not in spec_ids) and (o1 > o0) for i, (o0, o1) in zip(ids, offs)]
    return ids, offs, mask


def align_atoms_to_tokens(smiles: str, offs: list[tuple[int, int]], non_special: list[bool]):
    """Map each heavy atom in SMILES emission order to external token indices.

    Returns list of dicts (one per heavy atom in left-to-right SMILES order):
      {
        "atom_smi_idx": k,
        "char_span": (c0, c1),
        "smi_tok_idx": tok_i,
        "smi_tok": str,
        "ext_tok_indices": list[int],
        "alignment_type": "single_exact" | "single_unique" | "multi_atom_shared" | "split_multi_tok" | "unmapped",
      }
    """
    toks = tokenize(smiles)
    spans = token_spans(smiles)
    atom_entries = [
        (ti, t, sp) for ti, (t, sp) in enumerate(zip(toks, spans)) if is_atom_token(t)
    ]

    # Map each atom to overlapping non-special external tokens
    per_atom_ext: list[list[int]] = []
    ext_token_atom_count: dict[int, int] = {}
    for _, _, (c0, c1) in atom_entries:
        ov = [
            ei
            for ei, ((o0, o1), ok) in enumerate(zip(offs, non_special))
            if ok and o0 < c1 and c0 < o1
        ]
        per_atom_ext.append(ov)
        for ei in ov:
            ext_token_atom_count[ei] = ext_token_atom_count.get(ei, 0) + 1

    out = []
    for k, ((ti, t, (c0, c1)), ov) in enumerate(zip(atom_entries, per_atom_ext)):
        if len(ov) == 0:
            atype = "unmapped"
        elif len(ov) > 1:
            atype = "split_multi_tok"
        else:
            ei = ov[0]
            if ext_token_atom_count[ei] > 1:
                atype = "multi_atom_shared"
            elif offs[ei] == (c0, c1):
                atype = "single_exact"
            else:
                atype = "single_unique"
        out.append(
            {
                "atom_smi_idx": k,
                "char_span": (c0, c1),
                "smi_tok_idx": ti,
                "smi_tok": t,
                "ext_tok_indices": ov,
                "alignment_type": atype,
            }
        )
    return out


def atom_vector_from_hidden(h_seq: np.ndarray, ext_indices: list[int], kind: str) -> np.ndarray | None:
    """Extract atom vector from (T, D) hidden states given overlapping token indices."""
    if not ext_indices:
        return None
    if len(ext_indices) == 1:
        return h_seq[ext_indices[0]]
    if kind == "causal":
        # Last overlapping token has causally attended over the full bracket atom
        return h_seq[ext_indices[-1]]
    return h_seq[ext_indices].mean(axis=0)


def atom_rep_token_id(ids: list[int], ext_indices: list[int], kind: str) -> int:
    """Representative token ID for the token-identity baseline."""
    if not ext_indices:
        return -1
    return ids[ext_indices[-1]] if kind == "causal" else ids[ext_indices[0]]


def audit_tokenizer(tok, spec: ModelSpec, sample_smiles: list[str]) -> dict[str, Any]:
    """Audit tokenizer on 5 test molecules and across the sample corpus."""
    examples = []
    for s in TEST_SMILES_5:
        ids, offs, mask = encode_with_offsets(tok, s, spec)
        examples.append(
            {
                "smiles": s,
                "n_tokens": len(ids),
                "tokens": tok.convert_ids_to_tokens(ids),
                "offsets": [list(o) for o in offs],
            }
        )

    counts = {
        "single_exact": 0,
        "single_unique": 0,
        "multi_atom_shared": 0,
        "split_multi_tok": 0,
        "unmapped": 0,
    }
    total_atoms = 0
    total_ext_toks = 0
    for s in sample_smiles[:1000]:
        try:
            ids, offs, mask = encode_with_offsets(tok, s, spec)
            aligned = align_atoms_to_tokens(s, offs, mask)
        except Exception:
            continue
        total_ext_toks += sum(mask)
        for a in aligned:
            counts[a["alignment_type"]] += 1
            total_atoms += 1

    fracs = {k: round(v / max(total_atoms, 1), 6) for k, v in counts.items()}
    return {
        "tokenizer_class": type(tok).__name__,
        "is_fast": bool(getattr(tok, "is_fast", False)),
        "vocab_size": int(tok.vocab_size),
        "len_tokenizer": int(len(tok)),
        "special_tokens_map": {k: str(v) for k, v in tok.special_tokens_map.items()},
        "special_ids": {
            "bos": tok.bos_token_id,
            "eos": tok.eos_token_id,
            "pad": tok.pad_token_id,
            "cls": tok.cls_token_id,
            "sep": tok.sep_token_id,
            "mask": tok.mask_token_id,
            "unk": tok.unk_token_id,
        },
        "prepend_bos_manual": spec.prepend_bos_manual,
        "notes": spec.notes,
        "test_examples_5": examples,
        "corpus_atom_coverage_1000mols": {
            "total_heavy_atoms": int(total_atoms),
            "mean_ext_tokens_per_mol": round(total_ext_toks / max(min(len(sample_smiles), 1000), 1), 3),
            "counts": counts,
            "fractions": fracs,
            "single_atom_total_frac": round(fracs["single_exact"] + fracs["single_unique"], 6),
        },
    }


def pad_id_of(tok) -> int:
    if tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    if tok.eos_token_id is not None:
        return int(tok.eos_token_id)
    return 0


@torch.no_grad()
def extract_all_hidden_states(
    model,
    tok,
    spec: ModelSpec,
    smiles_list: list[str],
    device: str,
    batch_size: int = 64,
    add_eos_causal: bool = True,
):
    """Run batched forward passes returning per-molecule (ids, offs, mask, hidden_list).

    To keep RAM/VRAM low, hidden states are stored in fp16 on CPU per molecule,
    shaped (n_layers+1, seq_len, d_model), truncated to each molecule's exact length.
    """
    pid = pad_id_of(tok)
    encoded = [encode_with_offsets(tok, s, spec, add_eos_causal=add_eos_causal) for s in smiles_list]
    results = [None] * len(smiles_list)

    for s0 in range(0, len(smiles_list), batch_size):
        chunk = encoded[s0 : s0 + batch_size]
        max_len = max(len(c[0]) for c in chunk)
        inp = torch.full((len(chunk), max_len), pid, dtype=torch.long, device=device)
        amask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for bi, (ids, _, _) in enumerate(chunk):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            amask[bi, : len(ids)] = 1

        out = model(input_ids=inp, attention_mask=amask, output_hidden_states=True)
        # Stack (L+1, B, T, D) -> fp16 CPU
        hs = torch.stack([h.detach().to(torch.float16) for h in out.hidden_states], dim=0).cpu().numpy()
        for bi, (ids, offs, nsm) in enumerate(chunk):
            Llen = len(ids)
            results[s0 + bi] = (ids, offs, nsm, hs[:, bi, :Llen, :])
    return results


def fit_torch_logreg(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_classes: int,
    C: float = 1.0,
    max_iter: int = 400,
    device: str = "cuda",
    standardize: bool = True,
) -> dict[str, float]:
    """L2-regularized multinomial/binary logistic regression via GPU LBFGS."""
    if standardize:
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr.astype(np.float32))
        Xte_s = scaler.transform(X_te.astype(np.float32))
    else:
        Xtr_s = np.ascontiguousarray(X_tr, dtype=np.float32)
        Xte_s = np.ascontiguousarray(X_te, dtype=np.float32)

    Xt = torch.as_tensor(Xtr_s, dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y_tr), dtype=torch.long, device=device)
    Xev = torch.as_tensor(Xte_s, dtype=torch.float32, device=device)

    W = torch.zeros(Xt.shape[1], n_classes, device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS(
        [W, b],
        max_iter=max_iter,
        history_size=10,
        tolerance_grad=1e-6,
        tolerance_change=1e-8,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        logits = Xt @ W + b
        loss = 0.5 * (W * W).sum() + C * F.cross_entropy(logits, yt, reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        logits_te = (Xev @ W + b).cpu().numpy()
    preds = logits_te.argmax(axis=1)
    acc = float((preds == y_te).mean())
    res = {"accuracy": acc}
    if n_classes == 2:
        probs = F.softmax(torch.tensor(logits_te), dim=-1)[:, 1].numpy()
        try:
            res["auroc"] = float(roc_auc_score(y_te, probs))
        except Exception:
            res["auroc"] = float("nan")
    return res


def token_identity_baseline(
    tok_ids: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    n_classes: int,
    device: str = "cuda",
) -> dict[str, float]:
    """One-hot token-identity logistic regression baseline on exact same split."""
    uniq = np.unique(tok_ids)
    remap = {int(t): i for i, t in enumerate(uniq)}
    mapped = np.array([remap[int(t)] for t in tok_ids], dtype=np.int64)
    onehot = np.zeros((len(mapped), len(uniq)), dtype=np.float32)
    onehot[np.arange(len(mapped)), mapped] = 1.0
    return fit_torch_logreg(
        onehot[tr], y[tr], onehot[te], y[te], n_classes=n_classes, C=1.0, device=device, standardize=False
    )


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Exact linear CKA (Kornblith et al., 2019) between (N, D) matrices."""
    Xc = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    Yc = Y.astype(np.float64) - Y.mean(axis=0, keepdims=True)
    cross = Yc.T @ Xc
    hsic_xy = float(np.sum(cross * cross))
    xx = Xc.T @ Xc
    yy = Yc.T @ Yc
    denom = float(np.sqrt(np.sum(xx * xx) * np.sum(yy * yy)))
    return hsic_xy / denom if denom > 1e-12 else float("nan")


# --------------------------------------------------------------------------- #
# Part 1: Causal SMILES GPT-2 Mechanistic Suite
# --------------------------------------------------------------------------- #


def build_single_ring_pairs_gpt2(smiles_pool: list[str], tok, spec: ModelSpec, max_pairs: int = 1024):
    """Single-ring 1->2 relabeling minimal pairs (mirrors 21_ring_patching / 30_grid_readouts)."""
    id1 = tok.encode("1", add_special_tokens=False)
    id2 = tok.encode("2", add_special_tokens=False)
    assert len(id1) == 1 and len(id2) == 1
    id1, id2 = id1[0], id2[0]

    pairs = []
    n_single_ring = 0
    for s in smiles_pool:
        try:
            toks = tokenize(s)
        except Exception:
            continue
        digits = [t for t in toks if t.isdigit() or t.startswith("%")]
        if digits != ["1", "1"]:
            continue
        n_single_ring += 1
        s2 = "".join("2" if t == "1" else t for t in toks)
        ids1, _, _ = encode_with_offsets(tok, s, spec, add_eos_causal=True)
        ids2, _, _ = encode_with_offsets(tok, s2, spec, add_eos_causal=True)
        if len(ids1) != len(ids2) or len(ids1) > 250:
            continue
        diffs = [i for i in range(len(ids1)) if ids1[i] != ids2[i]]
        if len(diffs) != 2:
            continue
        oi, ci = diffs
        if ids1[oi] != id1 or ids1[ci] != id1 or ids2[oi] != id2 or ids2[ci] != id2:
            continue
        if ci - 1 <= oi:
            continue
        pairs.append(
            {
                "smiles_clean": s,
                "smiles_corrupt": s2,
                "clean_ids": ids1,
                "corrupt_ids": ids2,
                "open_idx": oi,
                "close_idx": ci,
                "predict_idx": ci - 1,
                "span": ci - oi,
            }
        )
        if len(pairs) >= max_pairs:
            break
    return pairs, n_single_ring, id1, id2


@torch.no_grad()
def run_gpt2_ring_relabeling_and_patching(model, tok, spec: ModelSpec, smiles_pool: list[str], device: str):
    """Ring relabeling (1->2) accuracy + residual & head activation patching."""
    pairs, n_single_ring, id1, id2 = build_single_ring_pairs_gpt2(smiles_pool, tok, spec, max_pairs=1024)
    pid = pad_id_of(tok)
    batch_size = 64

    # 1. Spelling sensitivity on all 1,024 pairs
    contrasts = []
    full_hits = []
    for key, target_id in (("clean_ids", id1), ("corrupt_ids", id2)):
        scores_list = []
        for s0 in range(0, len(pairs), batch_size):
            rows = pairs[s0 : s0 + batch_size]
            w = max(len(r[key]) for r in rows)
            x = torch.full((len(rows), w), pid, dtype=torch.long, device=device)
            am = torch.zeros((len(rows), w), dtype=torch.long, device=device)
            for bi, r in enumerate(rows):
                x[bi, : len(r[key])] = torch.tensor(r[key], dtype=torch.long, device=device)
                am[bi, : len(r[key])] = 1
            logits = model(input_ids=x, attention_mask=am).logits
            pi = torch.tensor([r["predict_idx"] for r in rows], device=device)
            sel = logits[torch.arange(len(rows), device=device), pi].float().cpu()
            scores_list.append(sel)
        scores = torch.cat(scores_list, dim=0)
        contrasts.append((scores[:, id1] - scores[:, id2]).numpy())
        full_hits.append(int((scores.argmax(dim=1) == target_id).sum().item()))

    orig, relab = contrasts
    change = orig - relab
    rng = np.random.default_rng(51)
    boot = [change[rng.integers(len(change), size=len(change))].mean() for _ in range(2000)]

    relabeling = {
        "n_pairs": len(pairs),
        "n_single_ring_scanned": int(n_single_ring),
        "bpe_clean_pair_yield": round(len(pairs) / max(n_single_ring, 1), 6),
        "original_1_beats_2_count": int((orig > 0).sum()),
        "original_1_beats_2_acc": float((orig > 0).mean()),
        "renamed_2_beats_1_count": int((relab < 0).sum()),
        "renamed_2_beats_1_acc": float((relab < 0).mean()),
        "both_correct_2way_count": int(((orig > 0) & (relab < 0)).sum()),
        "both_correct_2way_acc": float(((orig > 0) & (relab < 0)).mean()),
        "original_full_vocab_top1_count": full_hits[0],
        "original_full_vocab_top1_acc": float(full_hits[0] / len(pairs)),
        "renamed_full_vocab_top1_count": full_hits[1],
        "renamed_full_vocab_top1_acc": float(full_hits[1] / len(pairs)),
        "mean_original_logit1_minus_logit2": float(orig.mean()),
        "mean_renamed_logit1_minus_logit2": float(relab.mean()),
        "paired_logit_difference": float(change.mean()),
        "paired_difference_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }

    # 2. Ring-opener activation patching on 512 pairs (resid_post + per-head hook_z)
    patch_pairs = pairs[:512]
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    resid_scores = np.zeros((n_layers + 1, 3), dtype=np.float64)
    head_scores = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_elig = 0

    for s0 in range(0, len(patch_pairs), 128):
        rows = patch_pairs[s0 : s0 + 128]
        w = max(len(r["clean_ids"]) for r in rows)
        xc = torch.full((len(rows), w), pid, dtype=torch.long, device=device)
        xk = torch.full((len(rows), w), pid, dtype=torch.long, device=device)
        am = torch.zeros((len(rows), w), dtype=torch.long, device=device)
        for bi, r in enumerate(rows):
            Llen = len(r["clean_ids"])
            xc[bi, :Llen] = torch.tensor(r["clean_ids"], dtype=torch.long, device=device)
            xk[bi, :Llen] = torch.tensor(r["corrupt_ids"], dtype=torch.long, device=device)
            am[bi, :Llen] = 1
        oi = torch.tensor([r["open_idx"] for r in rows], device=device)
        pi = torch.tensor([r["predict_idx"] for r in rows], device=device)
        ar = torch.arange(len(rows), device=device)
        span = torch.zeros((len(rows), w), dtype=torch.bool, device=device)
        for bi, r in enumerate(rows):
            span[bi, r["open_idx"] : r["predict_idx"] + 1] = True

        # Capture clean resid_post (0..n_layers) and pre-c_proj head z (0..n_layers-1)
        clean_z = {}
        handles = []
        for l in range(n_layers):
            def _cap_z(mod, inp, _l=l):
                clean_z[_l] = inp[0].detach().clone()
            handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(_cap_z))
        out_c = model(input_ids=xc, attention_mask=am, output_hidden_states=True)
        for h in handles:
            h.remove()
        clean_hs = [t.detach().clone() for t in out_c.hidden_states]

        out_k = model(input_ids=xk, attention_mask=am)
        dc = (out_c.logits[ar, pi, id1] - out_c.logits[ar, pi, id2]).float()
        dk = (out_k.logits[ar, pi, id1] - out_k.logits[ar, pi, id2]).float()
        denom = dc - dk
        ok = denom.abs() > 1e-6
        n_elig += int(ok.sum().item())

        # Residual scan: layer 0 = drop output, layer 1..n_layers-1 = h[l-1] output,
        # layer n_layers = ln_f output (since HF GPT2 hidden_states[n_layers] is post-ln_f).
        for l in range(n_layers + 1):
            src = clean_hs[l]
            mod = (
                model.transformer.drop
                if l == 0
                else (model.transformer.ln_f if l == n_layers else model.transformer.h[l - 1])
            )
            is_tensor_out = l == 0 or l == n_layers
            for slot in range(3):
                def _edit_resid(module, inp, out, _is_t=is_tensor_out, _slot=slot, _src=src):
                    t = out if _is_t else out[0]
                    t = t.clone()
                    if _slot == 0:
                        t[ar, oi] = _src[ar, oi]
                    elif _slot == 1:
                        t[ar, pi] = _src[ar, pi]
                    else:
                        t = torch.where(span[:, :, None], _src, t)
                    return t if _is_t else (t,) + out[1:]
                hh = mod.register_forward_hook(_edit_resid)
                lp = model(input_ids=xk, attention_mask=am).logits
                hh.remove()
                dp = (lp[ar, pi, id1] - lp[ar, pi, id2]).float()
                rec = ((dp - dk) / denom)[ok]
                resid_scores[l, slot] += float(rec.sum().item())

        # Per-head z patching at predict_idx
        for l in range(n_layers):
            src_z = clean_z[l]
            cproj = model.transformer.h[l].attn.c_proj
            for hd in range(n_heads):
                sl = slice(hd * d_head, (hd + 1) * d_head)
                def _edit_head(module, inp, _sl=sl, _src=src_z):
                    z = inp[0].clone()
                    z[ar, pi, _sl] = _src[ar, pi, _sl]
                    return (z,)
                hh = cproj.register_forward_pre_hook(_edit_head)
                lp = model(input_ids=xk, attention_mask=am).logits
                hh.remove()
                dp = (lp[ar, pi, id1] - lp[ar, pi, id2]).float()
                rec = ((dp - dk) / denom)[ok]
                head_scores[l, hd] += float(rec.sum().item())

    resid_rec = (resid_scores / max(n_elig, 1)).tolist()
    head_rec = head_scores / max(n_elig, 1)
    best_h = np.unravel_index(int(np.argmax(head_rec)), head_rec.shape)

    patching = {
        "n_pairs": len(patch_pairs),
        "n_eligible": int(n_elig),
        "resid_sites": ["opener", "predict", "opener..predict"],
        "resid_recovery_by_layer": resid_rec,
        "head_z_recovery_at_predict": head_rec.tolist(),
        "best_patch_head": {
            "layer": int(best_h[0]),
            "head": int(best_h[1]),
            "patch_recovery": float(head_rec[best_h]),
        },
        "top5_patch_heads": [
            {"layer": int(i // n_heads), "head": int(i % n_heads), "patch_recovery": float(head_rec.ravel()[i])}
            for i in np.argsort(-head_rec.ravel())[:5]
        ],
    }
    return relabeling, patching


def _ring_digit_to_active_atom(toks: list[str]) -> dict[int, tuple[int, int]]:
    """Map each ring-digit SMILES token index to (active_atom_smi_tok_idx, rdkit_atom_idx).

    Walks SMILES grammar with a branch stack so stacked ring digits (e.g. `C12`),
    explicit bond symbols (e.g. `C=1`), and closed branches (e.g. `C(O)1`) all
    resolve to the true active heavy atom rather than `pos - 1`.
    """
    prev_atom_tok: int | None = None
    prev_atom_rd: int | None = None
    branch_stack: list[tuple[int | None, int | None]] = []
    rd_count = 0
    out: dict[int, tuple[int, int]] = {}
    for i, t in enumerate(toks):
        if t == "(":
            branch_stack.append((prev_atom_tok, prev_atom_rd))
        elif t == ")":
            prev_atom_tok, prev_atom_rd = branch_stack.pop() if branch_stack else (None, None)
        elif t in BOND_ORDER or t in ("/", "\\", "$"):
            continue
        elif t == ".":
            prev_atom_tok, prev_atom_rd = None, None
        elif t.isdigit() or (t.startswith("%") and t[1:].isdigit()):
            if prev_atom_tok is not None and prev_atom_rd is not None:
                out[i] = (prev_atom_tok, prev_atom_rd)
        elif is_atom_token(t):
            prev_atom_tok = i
            prev_atom_rd = rd_count
            rd_count += 1
    return out


def build_ring_events_external(smiles_list: list[str], tok, spec: ModelSpec, max_events: int = 2000, seed: int = 51):
    """Find ring-closure events & matched controls mapped cleanly into external token indices."""
    rng = np.random.default_rng(seed)
    mols_kept = []
    ev_records = []  # (mol_local_idx, open_ext, close_ext, predict_ext, open_atom_ext, close_atom_ext, span_ext)
    ctrl_records = []  # (mol_local_idx, pred_ext)

    for s in smiles_list:
        try:
            toks = tokenize(s)
            spans = token_spans(s)
        except Exception:
            continue
        all_evs = find_events(toks, offset=0)
        ring_evs = [e for e in all_evs if e.kind == "ring"]
        if not ring_evs:
            continue
        ids, offs, nsm = encode_with_offsets(tok, s, spec, add_eos_causal=True)
        if len(ids) > 200:
            continue

        mol = Chem.MolFromSmiles(s)
        ring_atom_map = _ring_digit_to_active_atom(toks)
        n_atom_toks = sum(1 for t in toks if is_atom_token(t))
        mol_ok = mol is not None and mol.GetNumAtoms() == n_atom_toks

        # Map SMILES char span to unique external token index
        def map_exact(smi_idx: int) -> int | None:
            c0, c1 = spans[smi_idx]
            hits = [ei for ei, ((o0, o1), ok) in enumerate(zip(offs, nsm)) if ok and o0 == c0 and o1 == c1]
            return hits[0] if len(hits) == 1 else None

        def map_overlap(smi_idx: int) -> int | None:
            c0, c1 = spans[smi_idx]
            hits = [ei for ei, ((o0, o1), ok) in enumerate(zip(offs, nsm)) if ok and o0 < c1 and c0 < o1]
            if not hits:
                return None
            return hits[-1] if spec.kind == "causal" else hits[0]

        mol_evs = []
        for e in ring_evs:
            oe = map_exact(e.open_idx)
            ce = map_exact(e.close_idx)
            if oe is None or ce is None or ce <= oe + 1:
                continue
            # Trace each ring digit back to its true active heavy atom & verify RDKit bond
            oa, ca = None, None
            o_info = ring_atom_map.get(e.open_idx)
            c_info = ring_atom_map.get(e.close_idx)
            if mol_ok and o_info is not None and c_info is not None:
                o_tok_i, o_rd = o_info
                c_tok_i, c_rd = c_info
                if o_rd != c_rd and mol.GetBondBetweenAtoms(o_rd, c_rd) is not None:
                    oa = map_overlap(o_tok_i)
                    ca = map_overlap(c_tok_i)
            mol_evs.append((oe, ce, ce - 1, oa, ca, ce - oe))
        if not mol_evs:
            continue

        # Exclude all closure prediction positions from control pool
        all_closure_pred_ext = set()
        for e in all_evs:
            ce = map_overlap(e.close_idx)
            if ce is not None and ce > 0:
                all_closure_pred_ext.add(ce - 1)

        m_idx = len(mols_kept)
        mols_kept.append((s, ids, offs, nsm))
        for oe, ce, pe, oa, ca, sp in mol_evs:
            ev_records.append((m_idx, oe, ce, pe, oa, ca, sp))

        # Draw 4 controls per ring event in this molecule
        first_valid = 1 if (spec.prepend_bos_manual or not nsm[0]) else 0
        last_valid = len(ids) - 2  # predict ids[t+1], excluding EOS prediction
        cand_ctrl = [t for t in range(first_valid, last_valid) if t not in all_closure_pred_ext and nsm[t] and nsm[t + 1]]
        want = 4 * len(mol_evs)
        if cand_ctrl:
            pick = rng.choice(cand_ctrl, size=min(want, len(cand_ctrl)), replace=False)
            for p in sorted(int(x) for x in pick):
                ctrl_records.append((m_idx, p))

        if len(ev_records) >= max_events:
            break

    return mols_kept, ev_records, ctrl_records


@torch.no_grad()
def run_gpt2_ring_pointer_and_ablation(
    model, tok, spec: ModelSpec, smiles_list: list[str], device: str, ring_patch_grid: list[list[float]] | None = None
):
    """Measure pointer attention mass + per-head zero-ablation specificity across all 144 heads."""
    mols_kept, ev_records, ctrl_records = build_ring_events_external(
        smiles_list, tok, spec, max_events=1800, seed=51
    )
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    pid = pad_id_of(tok)

    def margin_at(logits: torch.Tensor, targets: torch.Tensor, pairs_bt: list[tuple[int, int]]) -> torch.Tensor:
        if not pairs_bt:
            return torch.zeros((0,), device=logits.device)
        pos = torch.tensor(pairs_bt, dtype=torch.long, device=logits.device)
        b, t = pos[:, 0], pos[:, 1]
        lg = logits[b, t].float()
        gold = targets[b, t]
        gold_logit = lg.gather(1, gold[:, None]).squeeze(1)
        masked = lg.scatter(1, gold[:, None], torch.finfo(lg.dtype).min)
        return gold_logit - masked.max(dim=-1).values

    pm_pred_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    pm_close_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_ev_total = 0

    # Accumulators for pooled margins & CE across batches
    base_ev_sum = 0.0
    base_ct_sum = 0.0
    base_ce_sum = 0.0
    n_ct_total = 0
    n_tok_total = 0

    abl_ev_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    abl_ct_sum = np.zeros((n_layers, n_heads), dtype=np.float64)
    abl_ce_sum = np.zeros((n_layers, n_heads), dtype=np.float64)

    batch_size = 128
    # Index events/controls by mol_idx
    ev_by_mol: dict[int, list] = {}
    for rec in ev_records:
        ev_by_mol.setdefault(rec[0], []).append(rec)
    ct_by_mol: dict[int, list] = {}
    for rec in ctrl_records:
        ct_by_mol.setdefault(rec[0], []).append(rec[1])

    for s0 in range(0, len(mols_kept), batch_size):
        chunk = mols_kept[s0 : s0 + batch_size]
        max_len = max(len(c[1]) for c in chunk)
        inp = torch.full((len(chunk), max_len), pid, dtype=torch.long, device=device)
        am = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for bi, (_, ids, _, _) in enumerate(chunk):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            am[bi, : len(ids)] = 1

        x_in = inp[:, :-1]
        y_tgt = inp[:, 1:]
        am_in = am[:, :-1]
        # Mask pad targets
        y_masked = y_tgt.clone()
        y_masked[am[:, 1:] == 0] = -100

        b_ev_bt, b_ev_full, b_ct_bt = [], [], []
        for bi in range(len(chunk)):
            gm = s0 + bi
            for rec in ev_by_mol.get(gm, []):
                _, oe, ce, pe, _, _, _ = rec
                if pe < x_in.shape[1] and ce < inp.shape[1]:
                    b_ev_bt.append((bi, pe))
                    b_ev_full.append((bi, oe, ce, pe))
            for p in ct_by_mol.get(gm, []):
                if p < x_in.shape[1]:
                    b_ct_bt.append((bi, p))
        if not b_ev_bt:
            continue

        # Baseline forward WITH attentions (on full inp so close_idx->open_idx is also measurable)
        out_full = model(input_ids=inp, attention_mask=am, output_attentions=True)
        for l in range(n_layers):
            pat = out_full.attentions[l]  # (B, H, T, T)
            for bi, oe, ce, pe in b_ev_full:
                pm_pred_sum[l] += pat[bi, :, pe, oe].float().cpu().numpy()
                pm_close_sum[l] += pat[bi, :, ce, oe].float().cpu().numpy()
        n_ev_total += len(b_ev_full)

        base_logits = out_full.logits[:, :-1, :]
        bev = margin_at(base_logits, y_tgt, b_ev_bt)
        bct = margin_at(base_logits, y_tgt, b_ct_bt)
        base_ev_sum += float(bev.sum().item())
        base_ct_sum += float(bct.sum().item())
        n_ct_total += len(b_ct_bt)
        n_tok_b = int((y_masked != -100).sum().item())
        base_ce_b = F.cross_entropy(
            base_logits.reshape(-1, base_logits.size(-1)).float(),
            y_masked.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        base_ce_sum += float(base_ce_b)
        n_tok_total += n_tok_b

        # Zero-ablate every head (l, h) in turn via c_proj pre-hook
        for l in range(n_layers):
            cproj = model.transformer.h[l].attn.c_proj
            for hd in range(n_heads):
                sl = slice(hd * d_head, (hd + 1) * d_head)
                def _zero_head(module, hinp, _sl=sl):
                    z = hinp[0].clone()
                    z[:, :, _sl] = 0.0
                    return (z,)
                hh = cproj.register_forward_pre_hook(_zero_head)
                lg = model(input_ids=x_in, attention_mask=am_in).logits
                hh.remove()
                aev = margin_at(lg, y_tgt, b_ev_bt)
                act = margin_at(lg, y_tgt, b_ct_bt)
                ace = F.cross_entropy(
                    lg.reshape(-1, lg.size(-1)).float(),
                    y_masked.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
                abl_ev_sum[l, hd] += float(aev.sum().item())
                abl_ct_sum[l, hd] += float(act.sum().item())
                abl_ce_sum[l, hd] += float(ace)

    pm_pred = pm_pred_sum / max(n_ev_total, 1)
    pm_close = pm_close_sum / max(n_ev_total, 1)
    be = base_ev_sum / max(n_ev_total, 1)
    bc = base_ct_sum / max(n_ct_total, 1)
    bce = base_ce_sum / max(n_tok_total, 1)

    rows = []
    for l in range(n_layers):
        for h in range(n_heads):
            ae = abl_ev_sum[l, h] / max(n_ev_total, 1)
            ac = abl_ct_sum[l, h] / max(n_ct_total, 1)
            ace = abl_ce_sum[l, h] / max(n_tok_total, 1)
            d_ev = be - ae
            d_ct = bc - ac
            rel_ev = d_ev / be if be else float("nan")
            rel_ct = d_ct / bc if bc else float("nan")
            row = {
                "layer": l,
                "head": h,
                "pointer_mass_predict": float(pm_pred[l, h]),
                "pointer_mass_closer": float(pm_close[l, h]),
                "delta_margin_event": float(d_ev),
                "delta_margin_control": float(d_ct),
                "delta_margin_specificity": float(d_ev - d_ct),
                "relative_drop_event": float(rel_ev),
                "relative_drop_control": float(rel_ct),
                "relative_specificity": float(rel_ev - rel_ct),
                "delta_ce": float(ace - bce),
            }
            if ring_patch_grid is not None:
                row["patch_recovery"] = float(ring_patch_grid[l][h])
            rows.append(row)

    # Compute 1-based ranks (1 = highest)
    by_pm = sorted(rows, key=lambda r: -r["pointer_mass_predict"])
    by_raw = sorted(rows, key=lambda r: -r["delta_margin_specificity"])
    by_rel = sorted(rows, key=lambda r: -r["relative_specificity"])
    raw_rank = {(r["layer"], r["head"]): idx + 1 for idx, r in enumerate(by_raw)}
    rel_rank = {(r["layer"], r["head"]): idx + 1 for idx, r in enumerate(by_rel)}
    pm_rank = {(r["layer"], r["head"]): idx + 1 for idx, r in enumerate(by_pm)}
    for r in rows:
        k = (r["layer"], r["head"])
        r["rank_pointer_mass"] = pm_rank[k]
        r["rank_delta_margin_specificity"] = raw_rank[k]
        r["rank_relative_specificity"] = rel_rank[k]

    top_ptr = by_pm[0]
    pm_arr = np.array([r["pointer_mass_predict"] for r in rows])
    raw_arr = np.array([r["delta_margin_specificity"] for r in rows])
    rel_arr = np.array([r["relative_specificity"] for r in rows])
    dce_arr = np.array([r["delta_ce"] for r in rows])

    corrs = {
        "spearman_pointer_vs_raw_specificity": float(spearmanr(pm_arr, raw_arr).statistic),
        "spearman_pointer_vs_relative_specificity": float(spearmanr(pm_arr, rel_arr).statistic),
        "spearman_raw_specificity_vs_delta_ce": float(spearmanr(raw_arr, dce_arr).statistic),
        "spearman_relative_specificity_vs_delta_ce": float(spearmanr(rel_arr, dce_arr).statistic),
    }
    if ring_patch_grid is not None:
        pr_arr = np.array([r["patch_recovery"] for r in rows])
        corrs["spearman_pointer_vs_patch_recovery"] = float(spearmanr(pm_arr, pr_arr).statistic)

    return {
        "n_molecules": len(mols_kept),
        "n_events": int(n_ev_total),
        "n_controls": int(n_ct_total),
        "n_heads_total": int(n_layers * n_heads),
        "baseline_margin_event": float(be),
        "baseline_margin_control": float(bc),
        "baseline_ce": float(bce),
        "strongest_pointer_head": top_ptr,
        "top5_by_pointer_mass": by_pm[:5],
        "top5_by_delta_margin_specificity": by_raw[:5],
        "top5_by_relative_specificity": by_rel[:5],
        "correlations": corrs,
        "pointer_mass_predict_grid": pm_pred.tolist(),
        "pointer_mass_closer_grid": pm_close.tolist(),
        "all_heads": rows,
    }


def _find_governing_atoms_for_double_bonds(toks: list[str]):
    """O(L) pass returning (decision_idx, gov_atom_idx, cap, element) for explicit '=' tokens."""
    used, charges, elements, arom, exp_h = {}, {}, {}, {}, {}
    prev_atom = None
    branch_stack = []
    ring_open = {}
    pending_bond = None
    out = []
    for i, tok in enumerate(toks):
        if tok == "(":
            branch_stack.append(prev_atom)
            continue
        if tok == ")":
            prev_atom = branch_stack.pop() if branch_stack else None
            continue
        if tok in BOND_ORDER:
            if tok == "=" and i > 0 and prev_atom is not None:
                cap = _remaining(
                    elements[prev_atom], charges[prev_atom], used.get(prev_atom, 0.0), exp_h[prev_atom]
                )
                if cap is not None:
                    out.append((i - 1, prev_atom, cap, elements[prev_atom]))
            pending_bond = BOND_ORDER[tok]
            continue
        if tok == ".":
            prev_atom, pending_bond = None, None
            continue
        if tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit()):
            lab = tok[1:] if tok.startswith("%") else tok
            b = pending_bond
            pending_bond = None
            if lab in ring_open:
                j, bj = ring_open.pop(lab)
                explicit = [x for x in (b, bj) if x is not None]
                order_ = (
                    max(explicit)
                    if explicit
                    else (1.5 if (arom.get(j, False) and arom.get(prev_atom, False)) else 1.0)
                )
                if prev_atom is not None:
                    used[prev_atom] = used.get(prev_atom, 0.0) + order_
                used[j] = used.get(j, 0.0) + order_
            elif prev_atom is not None:
                ring_open[lab] = (prev_atom, b)
            continue
        el = _element_of(tok)
        if el is None:
            continue
        idx = i
        elements[idx] = el
        charges[idx] = _charge_of(tok)
        exp_h[idx] = _explicit_h(tok)
        arom[idx] = _is_aromatic_token(tok)
        used.setdefault(idx, 0.0)
        if prev_atom is not None:
            b = (
                pending_bond
                if pending_bond is not None
                else (1.5 if (arom[idx] and arom.get(prev_atom, False)) else 1.0)
            )
            used[idx] += b
            used[prev_atom] = used.get(prev_atom, 0.0) + b
        pending_bond = None
        prev_atom = idx
    return out


@torch.no_grad()
def run_gpt2_valence_patching(model, tok, spec: ModelSpec, valid_smi_path: str, device: str):
    """Activation patching on paired upstream-edit prefixes (near <= 3 vs far >= 10 tokens).

    Because GPT-2's BPE merges `)` + `=` into a single token `)=` at branch boundaries
    (the exact syntactic mechanism creating far >= 10 token gaps between a governing atom
    and its downstream double-bond decision), we evaluate BOTH:
      1. `standalone_eq_near` (tok_gap <= 3): metric = logit('=') - logit('C')
      2. `branch_boundary_near` (tok_gap <= 3) & `branch_boundary_far` (tok_gap >= 10):
         metric = logit(')=') - logit(')') (and also log truncated-prefix logit('=') - logit('C'))!
    """
    id_eq = tok.encode("=", add_special_tokens=False)[0]
    id_C = tok.encode("C", add_special_tokens=False)[0]
    id_peq = tok.encode(")=", add_special_tokens=False)[0]
    id_p = tok.encode(")", add_special_tokens=False)[0]

    standalone_near, branch_near, branch_far = [], [], []
    max_per_bucket = 256

    with open(valid_smi_path) as fh:
        for line_idx, line in enumerate(fh):
            if line_idx >= 300000:
                break
            if (
                len(standalone_near) >= max_per_bucket
                and len(branch_near) >= max_per_bucket
                and len(branch_far) >= max_per_bucket
            ):
                break
            s = line.strip()
            if "=" not in s:
                continue
            try:
                toks = tokenize(s)
                govs = _find_governing_atoms_for_double_bonds(toks)
            except Exception:
                continue
            if not govs:
                continue
            spans = token_spans(s)
            enc1 = None
            for d, ga, cap, el in govs:
                if cap < 2 or el != "C" or toks[ga] != "C":
                    continue
                if max(0, cap - 2) >= 2:
                    continue
                smi_gap = d - ga
                if smi_gap < 1:
                    continue
                if enc1 is None:
                    ids1, offs1, _ = encode_with_offsets(tok, s, spec, add_eos_causal=True)
                    enc1 = (ids1, offs1)
                else:
                    ids1, offs1 = enc1
                if len(ids1) > 200:
                    continue
                alt = list(toks)
                alt[ga] = "O"
                s_alt = "".join(alt)
                ids2, _, _ = encode_with_offsets(tok, s_alt, spec, add_eos_causal=True)
                if len(ids1) != len(ids2):
                    continue
                b_c0, b_c1 = spans[d + 1]
                # Standalone '='
                btoks = [ti for ti, (o0, o1) in enumerate(offs1) if o0 == b_c0 and o1 == b_c1 and ids1[ti] == id_eq]
                if len(btoks) == 1:
                    pidx = btoks[0] - 1
                    diffs = [k for k in range(pidx + 1) if ids1[k] != ids2[k]]
                    if len(diffs) == 1:
                        tg = pidx - diffs[0]
                        if 1 <= tg <= 3 and len(standalone_near) < max_per_bucket:
                            standalone_near.append(
                                {
                                    "clean_ids": ids1,
                                    "corrupt_ids": ids2,
                                    "sub_idx": diffs[0],
                                    "predict_idx": pidx,
                                    "tok_gap": tg,
                                    "smi_gap": smi_gap,
                                    "clean_cap": cap,
                                    "corrupt_cap": max(0, cap - 2),
                                    "smiles_clean": s,
                                    "smiles_corrupt": s_alt,
                                }
                            )
                            break
                else:
                    # Merged ')=' at branch close
                    mtoks = [
                        ti
                        for ti, (o0, o1) in enumerate(offs1)
                        if o0 <= b_c0 and b_c1 <= o1 and ids1[ti] == id_peq
                    ]
                    if len(mtoks) == 1:
                        pidx = mtoks[0] - 1
                        diffs = [k for k in range(pidx + 1) if ids1[k] != ids2[k]]
                        if len(diffs) == 1:
                            tg = pidx - diffs[0]
                            row = {
                                "clean_ids": ids1,
                                "corrupt_ids": ids2,
                                "sub_idx": diffs[0],
                                "predict_idx": pidx,
                                "tok_gap": tg,
                                "smi_gap": smi_gap,
                                "clean_cap": cap,
                                "corrupt_cap": max(0, cap - 2),
                                "smiles_clean": s,
                                "smiles_corrupt": s_alt,
                            }
                            if 1 <= tg <= 3 and len(branch_near) < max_per_bucket:
                                branch_near.append(row)
                                break
                            elif tg >= 10 and len(branch_far) < max_per_bucket:
                                branch_far.append(row)
                                break

    pid = pad_id_of(tok)
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    def patch_bucket(rows: list[dict], t_pos: int, t_neg: int, label: str) -> dict[str, Any]:
        if not rows:
            return {"n_pairs": 0}
        resid_s = np.zeros((n_layers + 1, 3), dtype=np.float64)
        head_s = np.zeros((n_layers, n_heads), dtype=np.float64)
        dc_all, dk_all = [], []
        n_elig = 0
        for s0 in range(0, len(rows), 128):
            chunk = rows[s0 : s0 + 128]
            w = max(len(r["clean_ids"]) for r in chunk)
            xc = torch.full((len(chunk), w), pid, dtype=torch.long, device=device)
            xk = torch.full((len(chunk), w), pid, dtype=torch.long, device=device)
            am = torch.zeros((len(chunk), w), dtype=torch.long, device=device)
            for bi, r in enumerate(chunk):
                Llen = len(r["clean_ids"])
                xc[bi, :Llen] = torch.tensor(r["clean_ids"], dtype=torch.long, device=device)
                xk[bi, :Llen] = torch.tensor(r["corrupt_ids"], dtype=torch.long, device=device)
                am[bi, :Llen] = 1
            si = torch.tensor([r["sub_idx"] for r in chunk], device=device)
            pi = torch.tensor([r["predict_idx"] for r in chunk], device=device)
            ar = torch.arange(len(chunk), device=device)
            span = torch.zeros((len(chunk), w), dtype=torch.bool, device=device)
            for bi, r in enumerate(chunk):
                span[bi, r["sub_idx"] : r["predict_idx"] + 1] = True

            clean_z = {}
            handles = []
            for l in range(n_layers):
                def _cap_z(mod, inp, _l=l):
                    clean_z[_l] = inp[0].detach().clone()
                handles.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(_cap_z))
            out_c = model(input_ids=xc, attention_mask=am, output_hidden_states=True)
            for h in handles:
                h.remove()
            clean_hs = [t.detach().clone() for t in out_c.hidden_states]

            out_k = model(input_ids=xk, attention_mask=am)
            dc = (out_c.logits[ar, pi, t_pos] - out_c.logits[ar, pi, t_neg]).float()
            dk = (out_k.logits[ar, pi, t_pos] - out_k.logits[ar, pi, t_neg]).float()
            dc_all.append(dc.cpu().numpy())
            dk_all.append(dk.cpu().numpy())
            denom = dc - dk
            # Filter out tiny denominators (< 0.1) to prevent ratio blowup
            ok = denom.abs() > 0.1
            n_elig += int(ok.sum().item())

            for l in range(n_layers + 1):
                src = clean_hs[l]
                mod = (
                    model.transformer.drop
                    if l == 0
                    else (model.transformer.ln_f if l == n_layers else model.transformer.h[l - 1])
                )
                is_tensor_out = l == 0 or l == n_layers
                for slot in range(3):
                    def _edit_resid(module, inp, out, _is_t=is_tensor_out, _slot=slot, _src=src):
                        t = out if _is_t else out[0]
                        t = t.clone()
                        if _slot == 0:
                            t[ar, si] = _src[ar, si]
                        elif _slot == 1:
                            t[ar, pi] = _src[ar, pi]
                        else:
                            t = torch.where(span[:, :, None], _src, t)
                        return t if _is_t else (t,) + out[1:]
                    hh = mod.register_forward_hook(_edit_resid)
                    lp = model(input_ids=xk, attention_mask=am).logits
                    hh.remove()
                    dp = (lp[ar, pi, t_pos] - lp[ar, pi, t_neg]).float()
                    rec = ((dp - dk) / denom)[ok]
                    resid_s[l, slot] += float(rec.sum().item())

            for l in range(n_layers):
                src_z = clean_z[l]
                cproj = model.transformer.h[l].attn.c_proj
                for hd in range(n_heads):
                    sl = slice(hd * d_head, (hd + 1) * d_head)
                    def _edit_head(module, inp, _sl=sl, _src=src_z):
                        z = inp[0].clone()
                        z[ar, pi, _sl] = _src[ar, pi, _sl]
                        return (z,)
                    hh = cproj.register_forward_pre_hook(_edit_head)
                    lp = model(input_ids=xk, attention_mask=am).logits
                    hh.remove()
                    dp = (lp[ar, pi, t_pos] - lp[ar, pi, t_neg]).float()
                    rec = ((dp - dk) / denom)[ok]
                    head_s[l, hd] += float(rec.sum().item())

        dc_cat = np.concatenate(dc_all)
        dk_cat = np.concatenate(dk_all)
        resid_rec = resid_s / max(n_elig, 1)
        head_rec = head_s / max(n_elig, 1)
        best_h = np.unravel_index(int(np.argmax(head_rec)), head_rec.shape)
        gaps = [r["tok_gap"] for r in rows]
        sgaps = [r["smi_gap"] for r in rows]
        return {
            "bucket": label,
            "metric": f"logit({tok.convert_ids_to_tokens([t_pos])[0]!r}) - logit({tok.convert_ids_to_tokens([t_neg])[0]!r})",
            "n_pairs": len(rows),
            "n_eligible_abs_delta_gt_0p1": int(n_elig),
            "tok_gap_stats": {"min": int(min(gaps)), "median": float(np.median(gaps)), "max": int(max(gaps))},
            "smi_gap_stats": {"min": int(min(sgaps)), "median": float(np.median(sgaps)), "max": int(max(sgaps))},
            "mean_clean_metric": float(dc_cat.mean()),
            "mean_corrupt_metric": float(dk_cat.mean()),
            "mean_delta": float((dc_cat - dk_cat).mean()),
            "resid_sites": ["substitution", "decision", "sub..decision"],
            "resid_recovery_by_layer": resid_rec.tolist(),
            "best_decision_layer": int(np.argmax(resid_rec[:, 1])),
            "max_decision_recovery": float(np.max(resid_rec[:, 1])),
            "head_z_recovery_at_decision": head_rec.tolist(),
            "best_patch_head": {
                "layer": int(best_h[0]),
                "head": int(best_h[1]),
                "patch_recovery": float(head_rec[best_h]),
            },
            "top5_patch_heads": [
                {"layer": int(i // n_heads), "head": int(i % n_heads), "patch_recovery": float(head_rec.ravel()[i])}
                for i in np.argsort(-head_rec.ravel())[:5]
            ],
            "example_pair": {
                "clean": rows[0]["smiles_clean"],
                "corrupt": rows[0]["smiles_corrupt"],
                "tok_gap": rows[0]["tok_gap"],
                "smi_gap": rows[0]["smi_gap"],
            },
        }

    # Also test exact logit('=') - logit('C') on standalone_near AND on branch_far (as a secondary
    # readout right before ')=' so both metrics are transparently recorded)
    res_standalone_near = patch_bucket(standalone_near, id_eq, id_C, "standalone_eq_near_le3")
    res_branch_near = patch_bucket(branch_near, id_peq, id_p, "branch_peq_near_le3")
    res_branch_far = patch_bucket(branch_far, id_peq, id_p, "branch_peq_far_ge10")
    res_branch_far_eqC = patch_bucket(branch_far, id_eq, id_C, "branch_far_ge10_eq_vs_C_check")

    return {
        "note": (
            "Governing-atom element substitution (C -> O) flips remaining valence across the >=2 double-bond "
            "legality boundary without introducing illegal SMILES syntax. Because GPT-2 BPE merges ')' + '=' into "
            "a single token ')=', long-range branch-spanning pairs (tok_gap >= 10) naturally contrast logit(')=') - logit(')'), "
            "while non-branch near pairs contrast standalone logit('=') - logit('C'). Both are reported."
        ),
        "standalone_eq_near_le3": res_standalone_near,
        "branch_peq_near_le3": res_branch_near,
        "branch_peq_far_ge10": res_branch_far,
        "branch_far_ge10_eq_vs_C_check": res_branch_far_eqC,
    }


# --------------------------------------------------------------------------- #
# Part 2: Atom-Aligned Traversal Graph Invariance (ALL Models)
# --------------------------------------------------------------------------- #


def build_traversal_pairs(smiles_list: list[str], n_mols: int = 800, seed: int = 51):
    """Create canonical vs randomized SMILES pairs with exact RDKit atom output orders."""
    rng = np.random.default_rng(seed)
    pairs = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() < 5:
            continue
        s_can = Chem.MolToSmiles(mol, canonical=True)
        ord_can = [int(x) for x in mol.GetProp("_smilesAtomOutputOrder").strip("[],").split(",") if x]
        s_rand = None
        ord_rand = None
        for _ in range(10):
            cand = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            if cand != s_can:
                s_rand = cand
                ord_rand = [int(x) for x in mol.GetProp("_smilesAtomOutputOrder").strip("[],").split(",") if x]
                break
        if s_rand is None:
            continue
        try:
            toks_c = [t for t in tokenize(s_can) if is_atom_token(t)]
            toks_r = [t for t in tokenize(s_rand) if is_atom_token(t)]
        except Exception:
            continue
        if len(toks_c) != mol.GetNumAtoms() or len(toks_r) != mol.GetNumAtoms():
            continue
        if len(ord_can) != mol.GetNumAtoms() or len(ord_rand) != mol.GetNumAtoms():
            continue
        atomic_nums = [mol.GetAtomWithIdx(a).GetAtomicNum() for a in range(mol.GetNumAtoms())]
        pairs.append(
            {
                "s_can": s_can,
                "s_rand": s_rand,
                "ord_can": ord_can,  # position k in s_can -> RDKit atom idx
                "ord_rand": ord_rand,  # position k in s_rand -> RDKit atom idx
                "atomic_nums": atomic_nums,
            }
        )
        if len(pairs) >= n_mols:
            break
    return pairs


def compute_cosine_invariance(
    H_can: np.ndarray,
    H_rand: np.ndarray,
    mol_ids: np.ndarray,
    elems: np.ndarray,
    center: bool = True,
    seed: int = 51,
) -> dict[str, float]:
    """Matched atom cosine vs within-mol same-element null & between-mol same-element null."""
    U = H_can.astype(np.float64)
    V = H_rand.astype(np.float64)
    if center:
        U = U - U.mean(axis=0, keepdims=True)
        V = V - V.mean(axis=0, keepdims=True)
    Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    matched = np.sum(Un * Vn, axis=1)
    matched_mean = float(matched.mean())

    # Within-molecule same-element null + within-molecule retrieval
    within_null_vals = []
    ret_all_hits = 0
    ret_all_total = 0
    ret_same_el_hits = 0
    ret_same_el_total = 0

    uniq_mols = np.unique(mol_ids)
    # Fast grouped indexing via sort
    order = np.argsort(mol_ids, kind="stable")
    m_sorted = mol_ids[order]
    splits = np.flatnonzero(np.diff(m_sorted)) + 1
    groups = np.split(order, splits)
    for g in groups:
        if len(g) < 2:
            continue
        S = Un[g] @ Vn[g].T  # (k, k)
        el_g = elems[g]
        # Top-1 retrieval over all atoms in molecule
        pred_all = S.argmax(axis=1)
        ret_all_hits += int((pred_all == np.arange(len(g))).sum())
        ret_all_total += len(g)
        # Same-element pairs
        for i in range(len(g)):
            same = np.flatnonzero(el_g == el_g[i])
            if len(same) >= 2:
                others = same[same != i]
                within_null_vals.append(float(S[i, others].mean()))
                best_same = same[int(np.argmax(S[i, same]))]
                ret_same_el_hits += int(best_same == i)
                ret_same_el_total += 1

    # Between-molecule same-element null (10 random partners per atom)
    rng = np.random.default_rng(seed)
    idx_by_el = {int(e): np.flatnonzero(elems == e) for e in np.unique(elems)}
    between_vals = []
    for e, idxs in idx_by_el.items():
        if len(idxs) < 20:
            continue
        # Sample 10 random partners per atom and drop same-molecule hits
        picks = rng.choice(idxs, size=(len(idxs), 10), replace=True)
        diff_mol = mol_ids[picks] != mol_ids[idxs][:, None]
        dots = np.einsum("nd,nkd->nk", Un[idxs], Vn[picks])
        valid = dots[diff_mol]
        if len(valid):
            between_vals.append(valid)
    between_null = float(np.concatenate(between_vals).mean()) if between_vals else float("nan")
    within_null = float(np.mean(within_null_vals)) if within_null_vals else float("nan")

    return {
        "n_atoms": int(len(H_can)),
        "matched_cos": matched_mean,
        "within_mol_same_elem_null": within_null,
        "between_mol_same_elem_null": between_null,
        "gap_vs_within_null": float(matched_mean - within_null),
        "gap_vs_between_null": float(matched_mean - between_null),
        "within_mol_top1_all_atoms": float(ret_all_hits / max(ret_all_total, 1)),
        "within_mol_top1_same_elem": float(ret_same_el_hits / max(ret_same_el_total, 1)),
    }


def run_graph_invariance(model, tok, spec: ModelSpec, trav_pairs: list[dict], device: str) -> dict[str, Any]:
    """Run Atom-Aligned Traversal Graph Invariance + Molecule-Level CKA across every layer."""
    can_smiles = [p["s_can"] for p in trav_pairs]
    rand_smiles = [p["s_rand"] for p in trav_pairs]
    ext_can = extract_all_hidden_states(model, tok, spec, can_smiles, device=device, batch_size=64)
    ext_rand = extract_all_hidden_states(model, tok, spec, rand_smiles, device=device, batch_size=64)
    n_layers_plus1 = ext_can[0][3].shape[0]

    # Pre-align atoms for every molecule once
    aligned_meta = []  # list of (mol_idx, elem, can_ext_idxs, rand_ext_idxs, both_single_unique)
    mol_valid_mask = []
    for mi, p in enumerate(trav_pairs):
        ids_c, offs_c, nsm_c, _ = ext_can[mi]
        ids_r, offs_r, nsm_r, _ = ext_rand[mi]
        ac = align_atoms_to_tokens(p["s_can"], offs_c, nsm_c)
        ar = align_atoms_to_tokens(p["s_rand"], offs_r, nsm_r)
        # Map RDKit atom idx -> alignment entry
        can_by_rd = {p["ord_can"][entry["atom_smi_idx"]]: entry for entry in ac}
        rand_by_rd = {p["ord_rand"][entry["atom_smi_idx"]]: entry for entry in ar}
        any_mapped = False
        for rd_a, anum in enumerate(p["atomic_nums"]):
            ec = can_by_rd.get(rd_a)
            er = rand_by_rd.get(rd_a)
            if ec is None or er is None:
                continue
            if not ec["ext_tok_indices"] or not er["ext_tok_indices"]:
                continue
            both_single = (
                ec["alignment_type"] in ("single_exact", "single_unique")
                and er["alignment_type"] in ("single_exact", "single_unique")
            )
            aligned_meta.append((mi, anum, ec["ext_tok_indices"], er["ext_tok_indices"], both_single))
            any_mapped = True
        mol_valid_mask.append(any_mapped and any(nsm_c) and any(nsm_r))

    mol_ids = np.array([m[0] for m in aligned_meta], dtype=np.int32)
    elems = np.array([m[1] for m in aligned_meta], dtype=np.int32)
    both_single_mask = np.array([m[4] for m in aligned_meta], dtype=bool)
    valid_mols = [i for i, ok in enumerate(mol_valid_mask) if ok]

    by_layer = []
    for l in range(n_layers_plus1):
        # Atom matrices
        H_can = np.stack(
            [atom_vector_from_hidden(ext_can[mi][3][l], ci, spec.kind) for mi, _, ci, _, _ in aligned_meta],
            axis=0,
        )
        H_rand = np.stack(
            [atom_vector_from_hidden(ext_rand[mi][3][l], ri, spec.kind) for mi, _, _, ri, _ in aligned_meta],
            axis=0,
        )
        raw_all = compute_cosine_invariance(H_can, H_rand, mol_ids, elems, center=False, seed=51)
        ctr_all = compute_cosine_invariance(H_can, H_rand, mol_ids, elems, center=True, seed=51)

        # Strict single-atom tokens subset (important for BPE models)
        strict_ctr = None
        if both_single_mask.sum() >= 500 and both_single_mask.mean() < 0.995:
            strict_ctr = compute_cosine_invariance(
                H_can[both_single_mask],
                H_rand[both_single_mask],
                mol_ids[both_single_mask],
                elems[both_single_mask],
                center=True,
                seed=51,
            )

        # Molecule-level mean-pooled CKA & retrieval
        M_can = np.stack(
            [ext_can[mi][3][l][np.array(ext_can[mi][2], dtype=bool)].mean(axis=0) for mi in valid_mols],
            axis=0,
        ).astype(np.float64)
        M_rand = np.stack(
            [ext_rand[mi][3][l][np.array(ext_rand[mi][2], dtype=bool)].mean(axis=0) for mi in valid_mols],
            axis=0,
        ).astype(np.float64)
        cka_val = linear_cka(M_can, M_rand)

        # Molecule mean-centered cosine + retrieval top-1 across all 800 molecules
        Mc = M_can - M_can.mean(axis=0, keepdims=True)
        Mr = M_rand - M_rand.mean(axis=0, keepdims=True)
        Mc_n = Mc / np.maximum(np.linalg.norm(Mc, axis=1, keepdims=True), 1e-12)
        Mr_n = Mr / np.maximum(np.linalg.norm(Mr, axis=1, keepdims=True), 1e-12)
        S_mol = Mc_n @ Mr_n.T
        mol_matched_cos = float(np.diag(S_mol).mean())
        mol_null_cos = float((S_mol.sum() - np.trace(S_mol)) / (len(valid_mols) * (len(valid_mols) - 1)))
        mol_top1 = float((S_mol.argmax(axis=1) == np.arange(len(valid_mols))).mean())

        entry = {
            "layer": l,
            "atom_raw": raw_all,
            "atom_centered": ctr_all,
            "mol_meanpool_cka": float(cka_val),
            "mol_meanpool_centered_cos": mol_matched_cos,
            "mol_meanpool_null_cos": mol_null_cos,
            "mol_meanpool_top1_retrieval": mol_top1,
        }
        if strict_ctr is not None:
            entry["atom_centered_strict_single_tok"] = strict_ctr
        by_layer.append(entry)

    del ext_can, ext_rand
    gc.collect()

    best_atom_l = max(by_layer[1:], key=lambda r: r["atom_centered"]["gap_vs_within_null"])
    best_cka_l = max(by_layer[1:], key=lambda r: r["mol_meanpool_cka"])
    return {
        "n_molecules": len(valid_mols),
        "n_aligned_atoms": int(len(aligned_meta)),
        "strict_single_tok_frac": float(both_single_mask.mean()),
        "by_layer": by_layer,
        "best_atom_invariance_layer": {
            "layer": best_atom_l["layer"],
            "matched_cos": best_atom_l["atom_centered"]["matched_cos"],
            "within_null": best_atom_l["atom_centered"]["within_mol_same_elem_null"],
            "between_null": best_atom_l["atom_centered"]["between_mol_same_elem_null"],
            "gap_vs_within_null": best_atom_l["atom_centered"]["gap_vs_within_null"],
            "within_mol_top1_same_elem": best_atom_l["atom_centered"]["within_mol_top1_same_elem"],
        },
        "best_mol_cka_layer": {
            "layer": best_cka_l["layer"],
            "cka": best_cka_l["mol_meanpool_cka"],
            "mol_top1_retrieval": best_cka_l["mol_meanpool_top1_retrieval"],
        },
    }


# --------------------------------------------------------------------------- #
# Part 3: Chemical Hierarchy Probes Across Depth (ALL Models)
# --------------------------------------------------------------------------- #


def build_hierarchy_dataset(smiles_list: list[str], n_mols: int = 2000):
    """Precompute atom-level & molecule-level chemical ground truth on fixed validation molecules."""
    records = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() < 4:
            continue
        try:
            toks = tokenize(s)
            v_pref = valence_labels(toks, offset=0, semantics="prefix")
            v_fin = valence_labels(toks, offset=0, semantics="final")
        except Exception:
            continue
        atom_toks = [t for t in toks if is_atom_token(t)]
        if len(atom_toks) != mol.GetNumAtoms():
            continue
        # Map smi_tok_idx -> valence labels
        vp_map = {vl.token_idx: vl.remaining for vl in v_pref}
        vf_map = {vl.token_idx: vl.remaining for vl in v_fin}
        ri = mol.GetRingInfo()
        ring_sizes = [[] for _ in range(mol.GetNumAtoms())]
        for ring in ri.AtomRings():
            sz = len(ring)
            for a in ring:
                ring_sizes[a].append(sz)

        atom_meta = []
        atom_smi_indices = [ti for ti, t in enumerate(toks) if is_atom_token(t)]
        for a in range(mol.GetNumAtoms()):
            at = mol.GetAtomWithIdx(a)
            smi_ti = atom_smi_indices[a]
            is_arom = bool(at.GetIsAromatic())
            in_ring = bool(at.IsInRing())
            szs = set(ring_sizes[a])
            r56 = 0 if szs == {5} else (1 if szs == {6} else -1)
            fused = int(ri.NumAtomRings(a) >= 2) if in_ring else -1
            atom_meta.append(
                {
                    "smi_tok_idx": smi_ti,
                    "val_prefix": vp_map.get(smi_ti, -1),
                    "val_final": vf_map.get(smi_ti, -1),
                    "aliph_in_ring": int(in_ring) if not is_arom else -1,
                    "ring_5_vs_6": r56,
                    "fused_ring": fused,
                }
            )
        mol_props = {
            "MolLogP": float(Descriptors.MolLogP(mol)),
            "TPSA": float(Descriptors.TPSA(mol)),
            "QED": float(Descriptors.qed(mol)),
            "MolWt": float(Descriptors.MolWt(mol)),
            "NumAromaticRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        }
        records.append({"smiles": s, "atoms": atom_meta, "props": mol_props, "smi_toks": toks})
        if len(records) >= n_mols:
            break
    return records


def run_chemical_hierarchy_probes(model, tok, spec: ModelSpec, records: list[dict], device: str) -> dict[str, Any]:
    """Run atom-level (valence, aliphatic in-ring, 5 vs 6 ring, fused ring) and
    whole-molecule (LogP, TPSA, QED, MW, NumAromaticRings) probes across all layers."""
    smiles_list = [r["smiles"] for r in records]
    extracted = extract_all_hidden_states(model, tok, spec, smiles_list, device=device, batch_size=64)
    n_layers_plus1 = extracted[0][3].shape[0]
    n_mols = len(records)

    # Fixed 70/30 molecule-level split (zero leakage!)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=51)
    mol_tr, mol_te = next(gss.split(np.arange(n_mols), groups=np.arange(n_mols)))
    tr_mol_set = set(int(x) for x in mol_tr)
    te_mol_set = set(int(x) for x in mol_te)
    assert tr_mol_set.isdisjoint(te_mol_set)

    # Align atoms once
    atom_rows = []  # (mol_idx, ext_idxs, rep_tok_id, meta_dict)
    for mi, rec in enumerate(records):
        ids, offs, nsm, _ = extracted[mi]
        aligned = align_atoms_to_tokens(rec["smiles"], offs, nsm)
        if len(aligned) != len(rec["atoms"]):
            continue
        for al, meta in zip(aligned, rec["atoms"]):
            if not al["ext_tok_indices"]:
                continue
            rtok = atom_rep_token_id(ids, al["ext_tok_indices"], spec.kind)
            atom_rows.append((mi, al["ext_tok_indices"], rtok, meta))

    mol_arr = np.array([r[0] for r in atom_rows], dtype=np.int32)
    tok_arr = np.array([r[2] for r in atom_rows], dtype=np.int32)
    is_tr_atom = np.array([m in tr_mol_set for m in mol_arr], dtype=bool)
    is_te_atom = np.array([m in te_mol_set for m in mol_arr], dtype=bool)

    tasks = [
        ("remaining_valence_prefix", "val_prefix", 5),
        ("implicit_h_final", "val_final", 5),
        ("aliphatic_in_ring", "aliph_in_ring", 2),
        ("ring_size_5_vs_6", "ring_5_vs_6", 2),
        ("fused_ring", "fused_ring", 2),
    ]

    # Precompute labels, task masks, majority baselines, token-identity baselines, within-type controls
    task_info = {}
    for tname, field, n_cls in tasks:
        y_all = np.array([r[3][field] for r in atom_rows], dtype=np.int64)
        valid = y_all >= 0
        tr_idx = np.flatnonzero(valid & is_tr_atom)
        te_idx = np.flatnonzero(valid & is_te_atom)
        assert set(mol_arr[tr_idx]).isdisjoint(set(mol_arr[te_idx]))

        counts_tr = np.bincount(y_all[tr_idx], minlength=n_cls)
        maj_cls = int(counts_tr.argmax())
        maj_acc = float((y_all[te_idx] == maj_cls).mean())

        # Token-identity baseline on exact same split
        valid_idx = np.flatnonzero(valid)
        pos_map = {int(v): i for i, v in enumerate(valid_idx)}
        tr_loc = np.array([pos_map[int(i)] for i in tr_idx], dtype=np.int64)
        te_loc = np.array([pos_map[int(i)] for i in te_idx], dtype=np.int64)
        tok_base = token_identity_baseline(
            tok_arr[valid_idx], y_all[valid_idx], tr_loc, te_loc, n_classes=n_cls, device=device
        )

        # Within-type shuffled labels for valence selectivity control
        split_blk = np.zeros(len(valid_idx), dtype=int)
        split_blk[te_loc] = 1
        shuf_y = within_type_shuffled_labels(tok_arr[valid_idx], y_all[valid_idx], seed=51, blocks=split_blk)

        task_info[tname] = {
            "n_classes": n_cls,
            "y_all": y_all,
            "valid_idx": valid_idx,
            "tr_loc": tr_loc,
            "te_loc": te_loc,
            "shuf_y": shuf_y,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "majority_accuracy": maj_acc,
            "token_identity_baseline": tok_base,
            "by_layer": [],
        }

    # Evaluate atom-level probes across all layers
    for l in range(n_layers_plus1):
        H_all = np.stack(
            [atom_vector_from_hidden(extracted[mi][3][l], eidx, spec.kind) for mi, eidx, _, _ in atom_rows],
            axis=0,
        ).astype(np.float32)

        for tname, field, n_cls in tasks:
            ti = task_info[tname]
            Hv = H_all[ti["valid_idx"]]
            yv = ti["y_all"][ti["valid_idx"]]
            tr_l, te_l = ti["tr_loc"], ti["te_loc"]
            res = fit_torch_logreg(Hv[tr_l], yv[tr_l], Hv[te_l], yv[te_l], n_classes=n_cls, device=device)
            layer_rec = {"layer": l, **res}
            if tname == "remaining_valence_prefix":
                ctrl = fit_torch_logreg(
                    Hv[tr_l], ti["shuf_y"][tr_l], Hv[te_l], ti["shuf_y"][te_l], n_classes=n_cls, device=device
                )
                layer_rec["within_type_shuffled_accuracy"] = ctrl["accuracy"]
                layer_rec["within_type_selectivity"] = float(res["accuracy"] - ctrl["accuracy"])
            ti["by_layer"].append(layer_rec)

    # Whole-molecule physicochemical property probes (RidgeCV) + Bag-of-Tokens baselines
    prop_names = ["MolLogP", "TPSA", "QED", "MolWt", "NumAromaticRings"]
    Y_props = {p: np.array([r["props"][p] for r in records], dtype=np.float64) for p in prop_names}

    # Build Bag-of-Tokens count matrices (model vocab AND canonical atom-regex vocab)
    ext_vocab_seen = {}
    ext_tok_lists = []
    for mi in range(n_mols):
        ids, _, nsm, _ = extracted[mi]
        toks_i = [ids[k] for k, ok in enumerate(nsm) if ok]
        ext_tok_lists.append(toks_i)
        for tid in toks_i:
            if tid not in ext_vocab_seen:
                ext_vocab_seen[tid] = len(ext_vocab_seen)
    X_bot_ext = np.zeros((n_mols, len(ext_vocab_seen)), dtype=np.float64)
    for mi, toks_i in enumerate(ext_tok_lists):
        for tid in toks_i:
            X_bot_ext[mi, ext_vocab_seen[tid]] += 1.0

    smi_vocab_seen = {}
    for rec in records:
        for t in rec["smi_toks"]:
            if t not in smi_vocab_seen:
                smi_vocab_seen[t] = len(smi_vocab_seen)
    X_bot_smi = np.zeros((n_mols, len(smi_vocab_seen)), dtype=np.float64)
    for mi, rec in enumerate(records):
        for t in rec["smi_toks"]:
            X_bot_smi[mi, smi_vocab_seen[t]] += 1.0

    alphas = np.logspace(-3, 5, 17)

    def eval_ridge(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[mol_tr])
        Xte = sc.transform(X[mol_te])
        reg = RidgeCV(alphas=alphas)
        reg.fit(Xtr, y[mol_tr])
        pred = reg.predict(Xte)
        r2 = float(r2_score(y[mol_te], pred))
        rmse = float(np.sqrt(np.mean((y[mol_te] - pred) ** 2)))
        return {"r2": r2, "rmse": rmse, "alpha": float(reg.alpha_)}

    mol_prop_results = {}
    for p in prop_names:
        mol_prop_results[p] = {
            "bag_of_model_tokens_baseline": eval_ridge(X_bot_ext, Y_props[p]),
            "bag_of_smiles_tokens_baseline": eval_ridge(X_bot_smi, Y_props[p]),
            "by_layer": [],
        }

    for l in range(n_layers_plus1):
        X_mol = np.stack(
            [
                extracted[mi][3][l][np.array(extracted[mi][2], dtype=bool)].mean(axis=0)
                if any(extracted[mi][2])
                else extracted[mi][3][l].mean(axis=0)
                for mi in range(n_mols)
            ],
            axis=0,
        ).astype(np.float64)
        for p in prop_names:
            mol_prop_results[p]["by_layer"].append({"layer": l, **eval_ridge(X_mol, Y_props[p])})

    # Clean up temporary numpy arrays from task_info and compute best-layer summaries
    atom_probes_out = {}
    for tname, _, n_cls in tasks:
        ti = task_info[tname]
        metric_key = "auroc" if n_cls == 2 else "accuracy"
        best_l = max(ti["by_layer"][1:], key=lambda r: r[metric_key])
        atom_probes_out[tname] = {
            "n_classes": n_cls,
            "n_train_atoms": ti["n_train"],
            "n_test_atoms": ti["n_test"],
            "majority_accuracy": ti["majority_accuracy"],
            "token_identity_baseline": ti["token_identity_baseline"],
            "best_layer": best_l,
            "by_layer": ti["by_layer"],
        }

    for p in prop_names:
        best_l = max(mol_prop_results[p]["by_layer"][1:], key=lambda r: r["r2"])
        mol_prop_results[p]["best_layer"] = best_l
        mol_prop_results[p]["gain_over_bot_model_r2"] = float(
            best_l["r2"] - mol_prop_results[p]["bag_of_model_tokens_baseline"]["r2"]
        )

    del extracted
    gc.collect()

    return {
        "n_molecules": n_mols,
        "n_train_molecules": int(len(mol_tr)),
        "n_test_molecules": int(len(mol_te)),
        "split": "GroupShuffleSplit(test_size=0.3, random_state=51) by molecule",
        "atom_probes": atom_probes_out,
        "molecule_property_probes": mol_prop_results,
    }


# --------------------------------------------------------------------------- #
# Part 4: Ring-Opener Attention Mass for Encoder Models
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_encoder_ring_attention(model, tok, spec: ModelSpec, smiles_list: list[str], device: str) -> dict[str, Any]:
    """Measure ring-opener/closer attention mass across all heads of bidirectional encoders."""
    mols_kept, ev_records, _ = build_ring_events_external(smiles_list, tok, spec, max_events=1800, seed=51)
    cfg = model.config
    n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
    n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
    pid = pad_id_of(tok)

    ev_by_mol: dict[int, list] = {}
    for rec in ev_records:
        ev_by_mol.setdefault(rec[0], []).append(rec)

    c2o = np.zeros((n_layers, n_heads), dtype=np.float64)
    o2c = np.zeros((n_layers, n_heads), dtype=np.float64)
    p2o = np.zeros((n_layers, n_heads), dtype=np.float64)
    ca2oa = np.zeros((n_layers, n_heads), dtype=np.float64)
    n_ev = 0
    n_atom_ev = 0

    for s0 in range(0, len(mols_kept), 64):
        chunk = mols_kept[s0 : s0 + 64]
        w = max(len(c[1]) for c in chunk)
        inp = torch.full((len(chunk), w), pid, dtype=torch.long, device=device)
        am = torch.zeros((len(chunk), w), dtype=torch.long, device=device)
        for bi, (_, ids, _, _) in enumerate(chunk):
            inp[bi, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            am[bi, : len(ids)] = 1

        out = model(input_ids=inp, attention_mask=am, output_attentions=True)
        for l in range(n_layers):
            pat = out.attentions[l]  # (B, H, T, T)
            for bi in range(len(chunk)):
                gm = s0 + bi
                for _, oe, ce, pe, oa, ca, _ in ev_by_mol.get(gm, []):
                    c2o[l] += pat[bi, :, ce, oe].float().cpu().numpy()
                    o2c[l] += pat[bi, :, oe, ce].float().cpu().numpy()
                    p2o[l] += pat[bi, :, pe, oe].float().cpu().numpy()
                    if oa is not None and ca is not None:
                        ca2oa[l] += pat[bi, :, ca, oa].float().cpu().numpy()
        for bi in range(len(chunk)):
            gm = s0 + bi
            for _, oe, ce, pe, oa, ca, _ in ev_by_mol.get(gm, []):
                n_ev += 1
                if oa is not None and ca is not None:
                    n_atom_ev += 1

    c2o /= max(n_ev, 1)
    o2c /= max(n_ev, 1)
    p2o /= max(n_ev, 1)
    ca2oa /= max(n_atom_ev, 1)

    def top_summary(grid: np.ndarray):
        idx = np.argsort(-grid.ravel())[:5]
        return {
            "max_layer": int(idx[0] // n_heads),
            "max_head": int(idx[0] % n_heads),
            "max_mass": float(grid.ravel()[idx[0]]),
            "mean_mass": float(grid.mean()),
            "median_mass": float(np.median(grid)),
            "top5": [
                {"layer": int(i // n_heads), "head": int(i % n_heads), "mass": float(grid.ravel()[i])}
                for i in idx
            ],
            "grid": grid.tolist(),
        }

    return {
        "n_molecules": len(mols_kept),
        "n_ring_events": int(n_ev),
        "n_atom_pair_events": int(n_atom_ev),
        "closer_to_opener": top_summary(c2o),
        "opener_to_closer": top_summary(o2c),
        "predict_to_opener": top_summary(p2o),
        "closer_atom_to_opener_atom": top_summary(ca2oa),
    }


# --------------------------------------------------------------------------- #
# Model Loader & Orchestration
# --------------------------------------------------------------------------- #


def load_model_and_tokenizer(spec: ModelSpec, device: str):
    kw = {"trust_remote_code": spec.trust_remote_code}
    if spec.revision is not None:
        kw["revision"] = spec.revision
    tok = AutoTokenizer.from_pretrained(spec.hf_id, **kw)
    cfg = AutoConfig.from_pretrained(spec.hf_id, **kw)
    if spec.kind == "causal":
        model = AutoModelForCausalLM.from_pretrained(spec.hf_id, attn_implementation="eager", **kw)
    else:
        try:
            model = AutoModel.from_pretrained(spec.hf_id, attn_implementation="eager", **kw)
        except Exception:
            model = AutoModel.from_pretrained(spec.hf_id, **kw)
    model = model.to(device).eval()
    return model, tok, cfg


def main():
    from molcircuits import paths as _paths

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--valid-smi", default=str(Path(_paths.data()) / "valid.smi"))
    ap.add_argument("--out-dir", default="results/external_models")
    ap.add_argument("--n-trav-mols", type=int, default=800)
    ap.add_argument("--n-probe-mols", type=int, default=2000)
    ap.add_argument("--models", nargs="*", default=None, help="Optional subset of slugs to run")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"Loading validation molecules from {args.valid_smi} ...", flush=True)
    with open(args.valid_smi) as fh:
        all_lines = [line.strip() for line in fh if line.strip()]

    rng = np.random.default_rng(51)
    sample_idx = rng.choice(len(all_lines), size=min(200_000, len(all_lines)), replace=False)
    pool_smiles = list(dict.fromkeys(all_lines[i] for i in sample_idx))

    print(f"Preparing shared datasets: {args.n_trav_mols} traversal pairs, {args.n_probe_mols} probe mols ...", flush=True)
    trav_pairs = build_traversal_pairs(pool_smiles, n_mols=args.n_trav_mols, seed=51)
    probe_records = build_hierarchy_dataset(pool_smiles[5000:], n_mols=args.n_probe_mols)
    ring_attn_smiles = pool_smiles[10000:13000]
    print(
        f"  Ready: {len(trav_pairs)} traversal pairs, {len(probe_records)} hierarchy probe mols "
        f"[{time.time() - t_start:.1f}s]",
        flush=True,
    )

    specs = MODEL_SPECS
    if args.models:
        specs = [s for s in MODEL_SPECS if s.slug in set(args.models)]

    summary_rows = {}

    for spec in specs:
        t0 = time.time()
        print("\n" + "=" * 80, flush=True)
        print(f"MODEL: {spec.hf_id} ({spec.slug}, kind={spec.kind})", flush=True)
        try:
            model, tok, cfg = load_model_and_tokenizer(spec, args.device)
        except Exception as e:
            print(f"  FAILED TO LOAD {spec.hf_id}: {e}", flush=True)
            summary_rows[spec.slug] = {"hf_id": spec.hf_id, "status": "failed", "error": str(e)}
            continue

        n_params = sum(p.numel() for p in model.parameters())
        n_layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
        n_heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
        d_model = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
        print(f"  Loaded: {n_params/1e6:.2f}M params, {n_layers}L/{n_heads}H/{d_model}d", flush=True)

        res: dict[str, Any] = {
            "slug": spec.slug,
            "hf_id": spec.hf_id,
            "kind": spec.kind,
            "revision": spec.revision,
            "architecture": {
                "model_type": getattr(cfg, "model_type", None),
                "n_params": int(n_params),
                "n_layers": n_layers,
                "n_heads": n_heads,
                "d_model": d_model,
            },
            "tokenizer_audit": audit_tokenizer(tok, spec, [r["smiles"] for r in probe_records]),
        }
        cov = res["tokenizer_audit"]["corpus_atom_coverage_1000mols"]
        print(
            f"  Tokenizer: {res['tokenizer_audit']['tokenizer_class']} vocab={tok.vocab_size} "
            f"single_atom={cov['single_atom_total_frac']:.3f} multi_bpe={cov['fractions']['multi_atom_shared']:.3f} "
            f"split={cov['fractions']['split_multi_tok']:.3f}",
            flush=True,
        )

        # Causal-specific mechanistic suite (GPT-2)
        if spec.kind == "causal":
            print("  [1/5] Ring relabeling (1->2) & activation patching ...", flush=True)
            relabel, rpatch = run_gpt2_ring_relabeling_and_patching(model, tok, spec, pool_smiles, args.device)
            res["ring_relabeling"] = relabel
            res["ring_activation_patching"] = rpatch
            print(
                f"    orig 1>2={relabel['original_1_beats_2_acc']:.3f} (top1={relabel['original_full_vocab_top1_acc']:.3f}), "
                f"renamed 2>1={relabel['renamed_2_beats_1_acc']:.3f} (top1={relabel['renamed_full_vocab_top1_acc']:.3f}), "
                f"best patch head L{rpatch['best_patch_head']['layer']}H{rpatch['best_patch_head']['head']}="
                f"{rpatch['best_patch_head']['patch_recovery']:+.3f}",
                flush=True,
            )

            print("  [2/5] Ring attention-vs-ablation dissociation (all 144 heads) ...", flush=True)
            dissoc = run_gpt2_ring_pointer_and_ablation(
                model, tok, spec, ring_attn_smiles, args.device, ring_patch_grid=rpatch["head_z_recovery_at_predict"]
            )
            res["ring_attention_vs_ablation"] = dissoc
            tp = dissoc["strongest_pointer_head"]
            print(
                f"    top pointer head L{tp['layer']}H{tp['head']}: mass={tp['pointer_mass_predict']:.3f}, "
                f"raw_spec_rank={tp['rank_delta_margin_specificity']}/{dissoc['n_heads_total']}, "
                f"rel_spec_rank={tp['rank_relative_specificity']}/{dissoc['n_heads_total']}, "
                f"patch_rec={tp.get('patch_recovery', float('nan')):+.3f}, "
                f"rho(pm,raw)={dissoc['correlations']['spearman_pointer_vs_raw_specificity']:+.3f}, "
                f"rho(pm,rel)={dissoc['correlations']['spearman_pointer_vs_relative_specificity']:+.3f}",
                flush=True,
            )

            print("  [3/5] Valence activation patching (near <=3 vs far >=10) ...", flush=True)
            vpatch = run_gpt2_valence_patching(model, tok, spec, args.valid_smi, args.device)
            res["valence_activation_patching"] = vpatch
            for bk in ("standalone_eq_near_le3", "branch_peq_near_le3", "branch_peq_far_ge10"):
                b = vpatch[bk]
                if b.get("n_pairs", 0):
                    print(
                        f"    {bk}: n={b['n_pairs']} delta={b['mean_delta']:+.2f} "
                        f"max_dec_rec={b['max_decision_recovery']:+.3f}@L{b['best_decision_layer']} "
                        f"best_head=L{b['best_patch_head']['layer']}H{b['best_patch_head']['head']}"
                        f"({b['best_patch_head']['patch_recovery']:+.3f})",
                        flush=True,
                    )

        # Graph invariance (all models)
        step_lbl = "[4/5]" if spec.kind == "causal" else "[1/3]"
        print(f"  {step_lbl} Atom-aligned traversal graph invariance & CKA ({len(trav_pairs)} mols) ...", flush=True)
        ginv = run_graph_invariance(model, tok, spec, trav_pairs, args.device)
        res["traversal_graph_invariance"] = ginv
        ba = ginv["best_atom_invariance_layer"]
        bc = ginv["best_mol_cka_layer"]
        print(
            f"    best atom inv L{ba['layer']}: cos={ba['matched_cos']:.3f} vs within_null={ba['within_null']:.3f} "
            f"(gap={ba['gap_vs_within_null']:+.3f}, top1_same_el={ba['within_mol_top1_same_elem']:.3f}) | "
            f"best CKA L{bc['layer']}={bc['cka']:.3f} (mol_top1={bc['mol_top1_retrieval']:.3f})",
            flush=True,
        )

        # Chemical hierarchy probes (all models)
        step_lbl = "[5/5]" if spec.kind == "causal" else "[2/3]"
        print(f"  {step_lbl} Chemical hierarchy probes ({len(probe_records)} mols) ...", flush=True)
        hprobes = run_chemical_hierarchy_probes(model, tok, spec, probe_records, args.device)
        res["chemical_hierarchy_probes"] = hprobes
        ap_res = hprobes["atom_probes"]
        mp_res = hprobes["molecule_property_probes"]
        print(
            f"    val_pref={ap_res['remaining_valence_prefix']['best_layer']['accuracy']:.3f} "
            f"(tok={ap_res['remaining_valence_prefix']['token_identity_baseline']['accuracy']:.3f}) | "
            f"aliph_ring={ap_res['aliphatic_in_ring']['best_layer']['auroc']:.3f} "
            f"r56={ap_res['ring_size_5_vs_6']['best_layer']['auroc']:.3f} "
            f"fused={ap_res['fused_ring']['best_layer']['auroc']:.3f} | "
            f"LogP={mp_res['MolLogP']['best_layer']['r2']:.3f} TPSA={mp_res['TPSA']['best_layer']['r2']:.3f} "
            f"QED={mp_res['QED']['best_layer']['r2']:.3f} MW={mp_res['MolWt']['best_layer']['r2']:.3f} "
            f"AromR={mp_res['NumAromaticRings']['best_layer']['r2']:.3f}",
            flush=True,
        )

        # Ring-opener attention mass for encoders
        if spec.kind == "encoder":
            print("  [3/3] Ring-opener attention mass ...", flush=True)
            rattn = run_encoder_ring_attention(model, tok, spec, ring_attn_smiles, args.device)
            res["encoder_ring_attention"] = rattn
            print(
                f"    closer->opener max={rattn['closer_to_opener']['max_mass']:.3f} "
                f"(L{rattn['closer_to_opener']['max_layer']}H{rattn['closer_to_opener']['max_head']}), "
                f"opener->closer max={rattn['opener_to_closer']['max_mass']:.3f}, "
                f"pred->opener max={rattn['predict_to_opener']['max_mass']:.3f}, "
                f"catom->oatom max={rattn['closer_atom_to_opener_atom']['max_mass']:.3f}",
                flush=True,
            )

        res["elapsed_seconds"] = round(time.time() - t0, 2)
        out_path = out_dir / f"{spec.slug}.json"
        out_path.write_text(json.dumps(res, indent=2) + "\n")
        print(f"  Saved {out_path} [{res['elapsed_seconds']:.1f}s]", flush=True)

        # Build compact summary entry
        srow = {
            "hf_id": spec.hf_id,
            "kind": spec.kind,
            "revision": spec.revision,
            "n_params_m": round(n_params / 1e6, 2),
            "layers": n_layers,
            "heads": n_heads,
            "d_model": d_model,
            "tokenizer_class": res["tokenizer_audit"]["tokenizer_class"],
            "vocab_size": int(tok.vocab_size),
            "single_atom_token_frac": cov["single_atom_total_frac"],
            "multi_atom_bpe_frac": cov["fractions"]["multi_atom_shared"],
            "best_atom_inv_layer": ba["layer"],
            "best_atom_matched_cos_centered": ba["matched_cos"],
            "best_atom_within_null_centered": ba["within_null"],
            "best_atom_between_null_centered": ba["between_null"],
            "best_atom_gap_vs_within": ba["gap_vs_within_null"],
            "best_atom_top1_same_elem": ba["within_mol_top1_same_elem"],
            "best_mol_cka_layer": bc["layer"],
            "best_mol_cka": bc["cka"],
            "best_mol_top1_retrieval": bc["mol_top1_retrieval"],
            "probes_best": {
                "remaining_valence_prefix_acc": ap_res["remaining_valence_prefix"]["best_layer"]["accuracy"],
                "remaining_valence_prefix_tok_base": ap_res["remaining_valence_prefix"]["token_identity_baseline"]["accuracy"],
                "implicit_h_final_acc": ap_res["implicit_h_final"]["best_layer"]["accuracy"],
                "aliphatic_in_ring_auroc": ap_res["aliphatic_in_ring"]["best_layer"]["auroc"],
                "aliphatic_in_ring_tok_base_auroc": ap_res["aliphatic_in_ring"]["token_identity_baseline"]["auroc"],
                "ring_size_5_vs_6_auroc": ap_res["ring_size_5_vs_6"]["best_layer"]["auroc"],
                "ring_size_5_vs_6_tok_base_auroc": ap_res["ring_size_5_vs_6"]["token_identity_baseline"]["auroc"],
                "fused_ring_auroc": ap_res["fused_ring"]["best_layer"]["auroc"],
                "fused_ring_tok_base_auroc": ap_res["fused_ring"]["token_identity_baseline"]["auroc"],
                "MolLogP_r2": mp_res["MolLogP"]["best_layer"]["r2"],
                "MolLogP_bot_r2": mp_res["MolLogP"]["bag_of_model_tokens_baseline"]["r2"],
                "TPSA_r2": mp_res["TPSA"]["best_layer"]["r2"],
                "TPSA_bot_r2": mp_res["TPSA"]["bag_of_model_tokens_baseline"]["r2"],
                "QED_r2": mp_res["QED"]["best_layer"]["r2"],
                "QED_bot_r2": mp_res["QED"]["bag_of_model_tokens_baseline"]["r2"],
                "MolWt_r2": mp_res["MolWt"]["best_layer"]["r2"],
                "MolWt_bot_r2": mp_res["MolWt"]["bag_of_model_tokens_baseline"]["r2"],
                "NumAromaticRings_r2": mp_res["NumAromaticRings"]["best_layer"]["r2"],
                "NumAromaticRings_bot_r2": mp_res["NumAromaticRings"]["bag_of_model_tokens_baseline"]["r2"],
            },
        }
        if spec.kind == "causal":
            srow["causal_ring_relabeling"] = res["ring_relabeling"]
            srow["causal_ring_dissociation"] = {
                "strongest_pointer_head": res["ring_attention_vs_ablation"]["strongest_pointer_head"],
                "correlations": res["ring_attention_vs_ablation"]["correlations"],
                "best_patch_head": res["ring_activation_patching"]["best_patch_head"],
            }
            srow["causal_valence_patching_summary"] = {
                k: {
                    "n_pairs": res["valence_activation_patching"][k]["n_pairs"],
                    "mean_delta": res["valence_activation_patching"][k]["mean_delta"],
                    "max_decision_recovery": res["valence_activation_patching"][k]["max_decision_recovery"],
                    "best_decision_layer": res["valence_activation_patching"][k]["best_decision_layer"],
                    "best_patch_head": res["valence_activation_patching"][k]["best_patch_head"],
                }
                for k in ("standalone_eq_near_le3", "branch_peq_near_le3", "branch_peq_far_ge10")
                if res["valence_activation_patching"][k].get("n_pairs", 0)
            }
        else:
            ra = res["encoder_ring_attention"]
            srow["encoder_ring_attention_max"] = {
                "closer_to_opener": ra["closer_to_opener"]["max_mass"],
                "closer_to_opener_head": f"L{ra['closer_to_opener']['max_layer']}H{ra['closer_to_opener']['max_head']}",
                "opener_to_closer": ra["opener_to_closer"]["max_mass"],
                "opener_to_closer_head": f"L{ra['opener_to_closer']['max_layer']}H{ra['opener_to_closer']['max_head']}",
                "predict_to_opener": ra["predict_to_opener"]["max_mass"],
                "closer_atom_to_opener_atom": ra["closer_atom_to_opener_atom"]["max_mass"],
            }
        summary_rows[spec.slug] = srow

        del model
        torch.cuda.empty_cache()
        gc.collect()

    summary_path = out_dir / "summary_external_models.json"
    summary_payload = {
        "valid_smi": args.valid_smi,
        "n_traversal_molecules": len(trav_pairs),
        "n_hierarchy_probe_molecules": len(probe_records),
        "total_elapsed_seconds": round(time.time() - t_start, 2),
        "models": summary_rows,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    print(f"\nALL DONE in {time.time() - t_start:.1f}s -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
