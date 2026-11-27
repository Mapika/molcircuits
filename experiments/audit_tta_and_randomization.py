"""Inference-only audit of RDKit randomization structure and multi-traversal TTA graph likelihood.

1. Structural audit on 5,000 ZINC20 validation molecules comparing:
   - Canonical SMILES
   - RDKit doRandom=True (Arus-Pous unrestricted randomization; our training corpus)
   - Arus-Pous restricted randomization (random atom renumbering + canonical=False)
   Reports exact token counts, ring closures/mol (cycle rank), max branch depth, and
   fraction of molecules closing a ring inside a branch.

2. Multi-traversal Test-Time Augmentation (TTA) & rigorous marginal graph-NLL upper bound
   across all 18 controlled ZINC20 models (9 canonical + 9 randomized) on 1,000 held-out
   validation molecules with up to K=32 distinct traversals per molecule:
   - Exact single-string canonical NLL: -log p(s_canon)
   - Exact single-string random mean NLL: mean_k [-log p(s_k)]
   - Exact finite-sum marginal graph-NLL upper bound (K in {1, 5, 10, 20, 32}):
       -log sum_{k=1..K} p(s_k) >= -log p(G)   (exact inequality over distinct strings)
     evaluated both on {s_canon + distinct random} and on {distinct random only}.
   - Output-space traversal invariance (free of CKA and layer choice):
     within-molecule NLL SD across traversals, and Spearman rho between two independent
     traversals (rand_a vs rand_b, and canon vs rand_a).
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from scipy.special import logsumexp
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize

RDLogger.DisableLog("rdApp.*")
_gi = import_module("04_graph_invariance")


def structural_stats(smis: list[str]) -> dict:
    tok_lens, char_lens, ring_closures, max_depths, close_in_branch = [], [], [], [], []
    for s in smis:
        char_lens.append(len(s))
        toks = tokenize(s)
        tok_lens.append(len(toks))
        ring_closures.append(len(re.findall(r"%\d\d|\d", re.sub(r"\[[^\]]+\]", "", s))) / 2.0)
        d = md = 0
        open_rings: set[str] = set()
        reopened = False
        for tk in toks:
            if tk == "(":
                d += 1
                if d > md:
                    md = d
            elif tk == ")":
                d = max(0, d - 1)
            elif tk.isdigit() or (tk.startswith("%") and tk[1:].isdigit()):
                if tk in open_rings:
                    if d > 0:
                        reopened = True
                    open_rings.remove(tk)
                else:
                    open_rings.add(tk)
        max_depths.append(md)
        close_in_branch.append(int(reopened))
    n = len(smis)
    return {
        "n": n,
        "mean_tokens": float(np.mean(tok_lens)),
        "mean_chars": float(np.mean(char_lens)),
        "mean_ring_closures": float(np.mean(ring_closures)),
        "mean_max_branch_depth": float(np.mean(max_depths)),
        "frac_close_inside_branch": float(np.mean(close_in_branch)),
    }


def run_structural_audit(valid_smi_path: Path, n_mols: int = 5000) -> dict:
    canon = []
    with open(valid_smi_path) as fh:
        for ln in fh:
            s = ln.strip().split()[0]
            m = Chem.MolFromSmiles(s)
            if m is not None:
                canon.append(Chem.MolToSmiles(m, canonical=True))
                if len(canon) >= n_mols:
                    break
    mols = [Chem.MolFromSmiles(s) for s in canon]
    rng = random.Random(0)
    unrest = [Chem.MolToSmiles(m, canonical=False, doRandom=True) for m in mols]
    rest = []
    for m in mols:
        order = list(range(m.GetNumAtoms()))
        rng.shuffle(order)
        rm = Chem.RenumberAtoms(m, order)
        rest.append(Chem.MolToSmiles(rm, canonical=False))
    return {
        "n_molecules": len(canon),
        "canonical": structural_stats(canon),
        "unrestricted_doRandom": structural_stats(unrest),
        "restricted_renumber": structural_stats(rest),
    }


@torch.no_grad()
def score_strings_nll(
    model,
    encoded: list[list[int]],
    pad_id: int,
    device: str,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact per-string NLL (nats/mol, sum over target tokens incl. <eos>) and target lengths."""
    n = len(encoded)
    order = np.argsort([len(x) for x in encoded])
    nlls = np.empty(n, dtype=np.float64)
    lens = np.empty(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        idxs = order[start : start + batch_size]
        seqs = [encoded[i] for i in idxs]
        max_len = max(len(s) for s in seqs)
        buf = np.full((len(seqs), max_len), pad_id, dtype=np.int64)
        mask = np.zeros((len(seqs), max_len - 1), dtype=np.float32)
        for r, s in enumerate(seqs):
            buf[r, : len(s)] = s
            mask[r, : len(s) - 1] = 1.0
        x = torch.from_numpy(buf[:, :-1]).to(device)
        y = torch.from_numpy(buf[:, 1:]).to(device)
        m = torch.from_numpy(mask).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape(len(seqs), max_len - 1)
        seq_nll = (loss * m).sum(dim=1).cpu().numpy()
        seq_len = m.sum(dim=1).cpu().numpy()
        for r, orig_i in enumerate(idxs):
            nlls[orig_i] = float(seq_nll[r])
            lens[orig_i] = int(seq_len[r])
    return nlls, lens


def build_tta_corpus(valid_smi_path: Path, tok: SmilesTokenizer, n_mols: int = 1000, k_max: int = 32):
    """For each molecule, collect canonical string + up to k_max distinct doRandom=True traversals."""
    random.seed(42)
    np.random.seed(42)
    records = []
    with open(valid_smi_path) as fh:
        for ln in fh:
            s = ln.strip().split()[0]
            if len(s) > 120:
                continue
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            c = Chem.MolToSmiles(m, canonical=True)
            try:
                c_ids = tok.encode(c)
            except ValueError:
                continue
            if len(c_ids) > 250:
                continue
            # Collect up to k_max distinct random traversals (preferring non-canonical for pure-rand view)
            distinct_rand = []
            seen = {c}
            for _ in range(k_max * 6):
                r = Chem.MolToSmiles(m, canonical=False, doRandom=True)
                if r in seen:
                    continue
                try:
                    r_ids = tok.encode(r)
                except ValueError:
                    continue
                if len(r_ids) > 250:
                    continue
                seen.add(r)
                distinct_rand.append((r, r_ids))
                if len(distinct_rand) >= k_max:
                    break
            if len(distinct_rand) < 10:
                continue
            records.append({"canon": (c, c_ids), "rands": distinct_rand})
            if len(records) >= n_mols:
                break
    return records


def main():
    from molcircuits import paths as _paths

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    valid_smi_path = Path(_paths.data()) / "valid.smi"
    _, tok_path = _gi.resolve_data_and_tokenizer(_paths.data(), None)
    tok = SmilesTokenizer.load(tok_path)

    print("[1/2] Running 5,000-molecule RDKit randomization structural audit...", flush=True)
    struct_res = run_structural_audit(valid_smi_path, n_mols=5000)
    print(json.dumps(struct_res, indent=2), flush=True)

    print("[2/2] Building 1,000-molecule x K<=32 distinct-traversal TTA corpus...", flush=True)
    records = build_tta_corpus(valid_smi_path, tok, n_mols=1000, k_max=32)
    flat_ids: list[list[int]] = []
    mol_slices: list[tuple[int, int]] = []  # (canon_idx, rand_start, rand_end)
    for rec in records:
        c_idx = len(flat_ids)
        flat_ids.append(rec["canon"][1])
        r_start = len(flat_ids)
        for _, r_ids in rec["rands"]:
            flat_ids.append(r_ids)
        r_end = len(flat_ids)
        mol_slices.append((c_idx, r_start, r_end))
    mean_k_rand = float(np.mean([e - s for _, s, e in mol_slices]))
    print(f"  Built {len(records)} molecules, {len(flat_ids)} total strings (mean {mean_k_rand:.2f} distinct non-canon/mol)", flush=True)

    ckpts = []
    for reg, root in [
        ("canon", Path(_paths.root()) / "grid"),
        ("rand", Path(_paths.root()) / "grid_rand"),
    ]:
        for sz in ("small", "base", "large"):
            for sd in (0, 1, 2):
                p = root / f"zinc_{sz}" / f"tf_seed{sd}" / "checkpoint.pt"
                if p.exists():
                    ckpts.append((reg, sz, sd, p))

    k_grid = (1, 5, 10, 20, 32)
    per_ckpt = {}
    for reg, sz, sd, p in ckpts:
        key = f"{reg}_{sz}_seed{sd}"
        model = _gi.load_model(str(p), device)
        nlls, lens = score_strings_nll(model, flat_ids, tok.pad_id, device, batch_size=512)
        del model
        torch.cuda.empty_cache()

        canon_nlls = np.array([nlls[c_idx] for c_idx, _, _ in mol_slices])
        canon_npt = np.array([nlls[c_idx] / lens[c_idx] for c_idx, _, _ in mol_slices])
        rand_mean_nlls = np.array([np.mean(nlls[rs:re]) for _, rs, re in mol_slices])
        rand_mean_npt = np.array([np.mean(nlls[rs:re] / lens[rs:re]) for _, rs, re in mol_slices])
        within_sd_nll = np.array([np.std(nlls[rs:re]) for _, rs, re in mol_slices])
        within_sd_npt = np.array([np.std(nlls[rs:re] / lens[rs:re]) for _, rs, re in mol_slices])

        # Independent pair Spearman rho (output-space ranking invariance)
        rand_a = np.array([nlls[rs] for _, rs, _ in mol_slices])
        rand_b = np.array([nlls[rs + 1] for _, rs, _ in mol_slices])
        rho_rr = float(spearmanr(rand_a, rand_b).statistic)
        rho_cr = float(spearmanr(canon_nlls, rand_a).statistic)
        rand_a_npt = np.array([nlls[rs] / lens[rs] for _, rs, _ in mol_slices])
        rand_b_npt = np.array([nlls[rs + 1] / lens[rs + 1] for _, rs, _ in mol_slices])
        rho_rr_npt = float(spearmanr(rand_a_npt, rand_b_npt).statistic)
        rho_cr_npt = float(spearmanr(canon_npt, rand_a_npt).statistic)

        # Rigorous finite-sum marginal graph-NLL upper bound: -log sum_{k=1..K} p(s_k)
        # (a) {canon + K-1 distinct random}, (b) {K distinct random only}
        bound_incl_canon = {}
        bound_rand_only = {}
        for K in k_grid:
            vals_ic, vals_ro = [], []
            for c_idx, rs, re in mol_slices:
                logp_c = -nlls[c_idx]
                logp_r = -nlls[rs:re]
                # include canonical + first K-1 distinct random
                sel_ic = np.concatenate(([logp_c], logp_r[: max(0, K - 1)]))
                vals_ic.append(-float(logsumexp(sel_ic)))
                sel_ro = logp_r[: min(K, len(logp_r))]
                vals_ro.append(-float(logsumexp(sel_ro)))
            bound_incl_canon[str(K)] = float(np.mean(vals_ic))
            bound_rand_only[str(K)] = float(np.mean(vals_ro))

        per_ckpt[key] = {
            "regime": reg,
            "scale": sz,
            "seed": sd,
            "canon_nll_per_mol": float(np.mean(canon_nlls)),
            "canon_nats_per_tok": float(np.mean(canon_npt)),
            "rand_mean_nll_per_mol": float(np.mean(rand_mean_nlls)),
            "rand_mean_nats_per_tok": float(np.mean(rand_mean_npt)),
            "within_mol_nll_sd_mean": float(np.mean(within_sd_nll)),
            "within_mol_nll_sd_median": float(np.median(within_sd_nll)),
            "within_mol_npt_sd_mean": float(np.mean(within_sd_npt)),
            "spearman_rand_vs_rand_nll": rho_rr,
            "spearman_canon_vs_rand_nll": rho_cr,
            "spearman_rand_vs_rand_npt": rho_rr_npt,
            "spearman_canon_vs_rand_npt": rho_cr_npt,
            "graph_nll_upper_bound_incl_canon": bound_incl_canon,
            "graph_nll_upper_bound_rand_only": bound_rand_only,
        }
        print(
            f"  {key:18s}: canon={per_ckpt[key]['canon_nll_per_mol']:6.2f}  "
            f"rand1={per_ckpt[key]['rand_mean_nll_per_mol']:6.2f}  "
            f"UB(K=32,ic)={bound_incl_canon['32']:6.2f}  "
            f"UB(K=32,ro)={bound_rand_only['32']:6.2f}  "
            f"sd_nll={per_ckpt[key]['within_mol_nll_sd_mean']:5.2f}  "
            f"rho_rr={rho_rr:.4f}  rho_cr={rho_cr:.4f}",
            flush=True,
        )

    # Aggregate 3-seed means per (regime, scale)
    summary = {}
    for reg in ("canon", "rand"):
        for sz in ("small", "base", "large"):
            rows = [per_ckpt[f"{reg}_{sz}_seed{sd}"] for sd in (0, 1, 2) if f"{reg}_{sz}_seed{sd}" in per_ckpt]
            if not rows:
                continue
            summary[f"{reg}_{sz}"] = {
                "n_seeds": len(rows),
                "canon_nll_per_mol": float(np.mean([r["canon_nll_per_mol"] for r in rows])),
                "rand_mean_nll_per_mol": float(np.mean([r["rand_mean_nll_per_mol"] for r in rows])),
                "within_mol_nll_sd_mean": float(np.mean([r["within_mol_nll_sd_mean"] for r in rows])),
                "spearman_rand_vs_rand_nll": float(np.mean([r["spearman_rand_vs_rand_nll"] for r in rows])),
                "spearman_canon_vs_rand_nll": float(np.mean([r["spearman_canon_vs_rand_nll"] for r in rows])),
                "spearman_rand_vs_rand_npt": float(np.mean([r["spearman_rand_vs_rand_npt"] for r in rows])),
                "spearman_canon_vs_rand_npt": float(np.mean([r["spearman_canon_vs_rand_npt"] for r in rows])),
                "graph_nll_upper_bound_incl_canon": {
                    str(K): float(np.mean([r["graph_nll_upper_bound_incl_canon"][str(K)] for r in rows]))
                    for K in k_grid
                },
                "graph_nll_upper_bound_rand_only": {
                    str(K): float(np.mean([r["graph_nll_upper_bound_rand_only"][str(K)] for r in rows]))
                    for K in k_grid
                },
            }

    out = {
        "structural_randomization_audit": struct_res,
        "tta_config": {
            "n_molecules": len(records),
            "k_max_distinct_noncanon": 32,
            "mean_distinct_noncanon_per_mol": mean_k_rand,
            "total_strings_scored": len(flat_ids),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "summary_3seed_mean": summary,
        "per_checkpoint": per_ckpt,
    }
    out_path = REPO / "results" / "audit" / "tta_and_randomization_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
