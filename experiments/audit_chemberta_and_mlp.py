"""Causal verification of ChemBERTa-77M-MTR head L1H4 and Large canonical MLP ablation necessity.

1. ChemBERTa-77M-MTR all-36-head zero ablation:
   Measures causal impact of zeroing each attention head (l=0..2, h=0..11) on:
     - Native 200-task MTR regression outputs (MSE shift on ring vs acyclic molecules)
     - Layer-2 molecule cross-traversal Linear CKA (800 pairs)
     - Layer-2 constitutional-isomer identical-histogram retrieval (300x6 = 1,800 mols)
2. Large canonical (113.3M, seeds 0-2) layer-wise MLP zero-ablation necessity on ring closure:
   Measures closing-digit logit-diff and accuracy drop when zeroing MLP l=0..11 at predict_idx
   (and jointly zeroing late MLPs 9..11), alongside matched non-ring control positions.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.model.hooks import HookManager, resid_post

RDLogger.DisableLog("rdApp.*")
_ext = import_module("10_external_models")
_gi = import_module("04_graph_invariance")
_ch = import_module("05_chemical_hierarchy")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = X.astype(np.float64) - X.mean(axis=0, keepdims=True)
    Yc = Y.astype(np.float64) - Y.mean(axis=0, keepdims=True)
    cross = Yc.T @ Xc
    num = float(np.sum(cross * cross))
    den = float(np.sqrt(np.sum((Xc.T @ Xc) ** 2) * np.sum((Yc.T @ Yc) ** 2)))
    return num / den if den > 1e-12 else float("nan")


@torch.no_grad()
def run_chemberta_mtr_head_ablation(device: str) -> dict:
    hf_id = "DeepChem/ChemBERTa-77M-MTR"
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id).to(device).eval()

    from molcircuits import paths as _paths

    # 1. Load 800 traversal pairs (same deterministic seed as 10_external_models)
    valid_smi = Path(_paths.data()) / "valid.smi"
    smiles_pool = []
    with open(valid_smi) as fh:
        for i, ln in enumerate(fh):
            if i >= 60000:
                break
            smiles_pool.append(ln.strip().split()[0])

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(smiles_pool))
    can_list, rand_list, y_logp, y_tpsa, y_nrings = [], [], [], [], []
    for idx in perm:
        s = smiles_pool[idx]
        m = Chem.MolFromSmiles(s)
        if m is None or m.GetNumHeavyAtoms() < 6 or len(s) > 90:
            continue
        sc = Chem.MolToSmiles(m, canonical=True)
        sr = Chem.MolToSmiles(m, doRandom=True)
        if sr != sc:
            can_list.append(sc)
            rand_list.append(sr)
            y_logp.append(float(Descriptors.MolLogP(m)))
            y_tpsa.append(float(Descriptors.TPSA(m)))
            y_nrings.append(float(rdMolDescriptors.CalcNumRings(m)))
        if len(can_list) >= 800:
            break
    y_logp = np.asarray(y_logp)
    y_tpsa = np.asarray(y_tpsa)
    y_nrings = np.asarray(y_nrings)

    # Build 200 constitutional-isomer buckets (6 distinct isomers each = 1,200 mols)
    buckets: dict[tuple, list[str]] = {}
    seen = set()
    for s in smiles_pool:
        if len(s) > 80:
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        sc = Chem.MolToSmiles(m, canonical=True)
        if sc in seen:
            continue
        seen.add(sc)
        atom_toks = [t for t in tokenize(sc) if _gi._element_of(t) is not None]
        key = tuple(sorted(Counter(atom_toks).items()))
        lst = buckets.setdefault(key, [])
        if len(lst) < 6:
            lst.append(sc)
    iso_buckets = [v for v in buckets.values() if len(v) == 6][:200]
    iso_can, iso_rand = [], []
    for b in iso_buckets:
        for sc in b:
            m = Chem.MolFromSmiles(sc)
            sr = Chem.MolToSmiles(m, doRandom=True)
            for _ in range(5):
                if sr != sc:
                    break
                sr = Chem.MolToSmiles(m, doRandom=True)
            iso_can.append(sc)
            iso_rand.append(sr)

    def forward_batch(smis: list[str], ablate: tuple[int, int] | None = None):
        hook_handle = None
        if ablate is not None:
            al, ah = ablate
            target_mod = model.encoder.layer[al].attention.self
            n_heads = model.config.num_attention_heads
            head_dim = model.config.hidden_size // n_heads

            def _hook(_mod, _inp, out):
                ctx = out[0].clone()
                ctx[:, :, ah * head_dim : (ah + 1) * head_dim] = 0.0
                return (ctx,) + out[1:]

            hook_handle = target_mod.register_forward_hook(_hook)

        l2_pools, cls_l3 = [], []
        bsz = 128
        try:
            for i0 in range(0, len(smis), bsz):
                batch = tok(smis[i0 : i0 + bsz], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                out = model(**batch, output_hidden_states=True)
                hs2 = out.hidden_states[2]  # Layer-2 output (after L0 and L1; 0=emb, 1=L0, 2=L1, 3=L2)
                hs3 = out.hidden_states[3]  # Final Layer-3 output (after L2)
                ids = batch["input_ids"]
                mask = (ids != tok.cls_token_id) & (ids != tok.sep_token_id) & (ids != tok.pad_token_id)
                mask_f = mask.unsqueeze(-1).float()
                pooled = (hs2 * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
                l2_pools.append(pooled.float().cpu().numpy())
                cls_l3.append(hs3[:, 0, :].float().cpu().numpy())
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        return np.concatenate(l2_pools, axis=0), np.concatenate(cls_l3, axis=0)

    def iso_top1(Mc: np.ndarray, Mr: np.ndarray) -> float:
        mu = 0.5 * (Mc.mean(0, keepdims=True) + Mr.mean(0, keepdims=True))
        A = Mc - mu
        B = Mr - mu
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        hits = 0
        for b in range(len(iso_buckets)):
            s0, s1 = b * 6, (b + 1) * 6
            sim = A[s0:s1] @ B[s0:s1].T
            hits += int(np.sum(sim.argmax(axis=1) == np.arange(6)))
        return hits / (len(iso_buckets) * 6)

    # Clean baselines + frozen Ridge probes on clean [CLS] (500 train / 300 test)
    M_can_0, CLS_can_0 = forward_batch(can_list, None)
    M_rand_0, _ = forward_batch(rand_list, None)
    cka_0 = linear_cka(M_can_0, M_rand_0)

    I_can_0, _ = forward_batch(iso_can, None)
    I_rand_0, _ = forward_batch(iso_rand, None)
    iso_0 = iso_top1(I_can_0, I_rand_0)

    def fit_frozen_probe(X_tr, y_tr, alpha=10.0):
        mu = X_tr.mean(0, keepdims=True)
        Xc = (X_tr - mu).astype(np.float64)
        ym = float(y_tr.mean())
        w = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ (y_tr - ym))
        return mu, ym, w

    def eval_r2(X_te, y_te, probe):
        mu, ym, w = probe
        pred = (X_te.astype(np.float64) - mu) @ w + ym
        return 1.0 - float(np.sum((y_te - pred) ** 2)) / float(np.sum((y_te - y_te.mean()) ** 2))

    p_logp = fit_frozen_probe(CLS_can_0[:500], y_logp[:500])
    p_tpsa = fit_frozen_probe(CLS_can_0[:500], y_tpsa[:500])
    p_nrings = fit_frozen_probe(CLS_can_0[:500], y_nrings[:500])
    r2_logp_0 = eval_r2(CLS_can_0[500:], y_logp[500:], p_logp)
    r2_tpsa_0 = eval_r2(CLS_can_0[500:], y_tpsa[500:], p_tpsa)
    r2_nrings_0 = eval_r2(CLS_can_0[500:], y_nrings[500:], p_nrings)

    head_rows = []
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    for l in range(n_layers):
        for h in range(n_heads):
            Mc, CLSc = forward_batch(can_list, (l, h))
            Mr, _ = forward_batch(rand_list, (l, h))
            cka_h = linear_cka(Mc, Mr)
            Ic, _ = forward_batch(iso_can, (l, h))
            Ir, _ = forward_batch(iso_rand, (l, h))
            iso_h = iso_top1(Ic, Ir)

            r2_lp = eval_r2(CLSc[500:], y_logp[500:], p_logp)
            r2_tp = eval_r2(CLSc[500:], y_tpsa[500:], p_tpsa)
            r2_nr = eval_r2(CLSc[500:], y_nrings[500:], p_nrings)
            cls_l2 = float(np.mean(np.linalg.norm(CLSc - CLS_can_0, axis=1)))
            head_rows.append({
                "layer": l,
                "head": h,
                "name": f"L{l}H{h}",
                "cka_l2": cka_h,
                "delta_cka_l2": cka_h - cka_0,
                "iso_top1_l2": iso_h,
                "delta_iso_top1_l2": iso_h - iso_0,
                "delta_r2_nrings": r2_nr - r2_nrings_0,
                "delta_r2_logp": r2_lp - r2_logp_0,
                "delta_r2_tpsa": r2_tp - r2_tpsa_0,
                "cls_l3_l2_shift": cls_l2,
            })

    by_nrings_drop = sorted(head_rows, key=lambda r: r["delta_r2_nrings"])
    by_iso_drop = sorted(head_rows, key=lambda r: r["delta_iso_top1_l2"])
    by_cka_drop = sorted(head_rows, key=lambda r: r["delta_cka_l2"])
    l1h4 = next(r for r in head_rows if r["name"] == "L1H4")
    l1h4["rank_nrings_drop"] = 1 + next(i for i, r in enumerate(by_nrings_drop) if r["name"] == "L1H4")
    l1h4["rank_iso_drop"] = 1 + next(i for i, r in enumerate(by_iso_drop) if r["name"] == "L1H4")
    l1h4["rank_cka_drop"] = 1 + next(i for i, r in enumerate(by_cka_drop) if r["name"] == "L1H4")

    del model
    torch.cuda.empty_cache()
    return {
        "model": hf_id,
        "n_cka_pairs": len(can_list),
        "n_iso_buckets": len(iso_buckets),
        "baseline": {
            "cka_l2": cka_0,
            "iso_top1_l2": iso_0,
            "cls_l3_r2_logp": r2_logp_0,
            "cls_l3_r2_tpsa": r2_tpsa_0,
            "cls_l3_r2_nrings": r2_nrings_0,
        },
        "L1H4": l1h4,
        "top5_nrings_drop": by_nrings_drop[:5],
        "top5_iso_drop": by_iso_drop[:5],
        "top5_cka_drop": by_cka_drop[:5],
        "all_36_heads": head_rows,
    }


@torch.no_grad()
def run_large_mlp_ablation_necessity(device: str) -> dict:
    """Layer-wise and joint late-MLP zero-ablation on 512 single-ring pairs across Large seeds 0-2."""
    from molcircuits import paths as _paths

    smi_path, tok_path = _gi.resolve_data_and_tokenizer(_paths.data(), None)
    tok = SmilesTokenizer.load(tok_path)
    id1, id2 = tok.vocab["1"], tok.vocab["2"]

    # Collect 512 single-ring validation molecules (same deterministic criteria as 05_chemical_hierarchy)
    pairs = []
    with open(smi_path) as fh:
        for ln in fh:
            s = ln.strip().split()[0]
            if len(s) > 100:
                continue
            tks = tokenize(s)
            if tks.count("1") == 2 and all(d not in tks for d in ("2", "3", "4", "5", "6")):
                pos = [i for i, t in enumerate(tks) if t == "1"]
                close_pos = pos[1]
                pred_idx = close_pos  # in [BOS] + tks, token at index close_pos predicts token close_pos+1 (= '1')
                if close_pos >= 4:
                    pairs.append((s, pred_idx))
            if len(pairs) >= 512:
                break

    enc = [tok.encode(s) for s, _ in pairs]
    max_len = max(len(e) for e in enc)
    x = torch.full((len(enc), max_len), tok.pad_id, dtype=torch.long, device=device)
    for i, e in enumerate(enc):
        x[i, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)
    pred_pos = torch.tensor([p for _, p in pairs], dtype=torch.long, device=device)
    # Non-ring control position: 2 tokens before closing prediction (always >= 2)
    ctrl_pos = (pred_pos - 2).clamp(min=1)
    ctrl_targets = x[torch.arange(len(enc), device=device), ctrl_pos + 1]

    out = {"n_pairs": len(pairs), "per_seed": {}}
    for sd in (0, 1, 2):
        ckpt = str(Path(_paths.root()) / "grid" / "zinc_large" / f"tf_seed{sd}" / "checkpoint.pt")
        model = _gi.load_model(ckpt, device)
        L = model.cfg.n_layers

        def eval_ablate(layers_to_zero: list[int], pos_tensor: torch.Tensor):
            handles = []
            for l in layers_to_zero:
                def _make_hook(pt):
                    def _hk(_m, _inp, out_t):
                        z = out_t.clone()
                        z[torch.arange(z.shape[0], device=z.device), pt, :] = 0.0
                        return z
                    return _hk
                handles.append(model.blocks[l].mlp.register_forward_hook(_make_hook(pos_tensor)))
            logits_list = []
            try:
                for i0 in range(0, len(enc), 128):
                    i1 = min(len(enc), i0 + 128)
                    # re-register per batch slice cleanly
                    for h in handles:
                        h.remove()
                    handles = []
                    pt_b = pos_tensor[i0:i1]
                    for l in layers_to_zero:
                        def _make_hook_b(ptb):
                            def _hk(_m, _inp, out_t):
                                z = out_t.clone()
                                z[torch.arange(z.shape[0], device=z.device), ptb, :] = 0.0
                                return z
                            return _hk
                        handles.append(model.blocks[l].mlp.register_forward_hook(_make_hook_b(pt_b)))
                    lg = model(x[i0:i1])
                    logits_list.append(lg[torch.arange(i1 - i0, device=device), pt_b, :].float().cpu())
            finally:
                for h in handles:
                    h.remove()
            return torch.cat(logits_list, dim=0)

        clean_ring = eval_ablate([], pred_pos)
        clean_ctrl = eval_ablate([], ctrl_pos)
        clean_ld = float((clean_ring[:, id1] - clean_ring[:, id2]).mean().item())
        clean_acc = float((clean_ring.argmax(dim=-1) == id1).float().mean().item())
        clean_ctrl_acc = float((clean_ctrl.argmax(dim=-1) == ctrl_targets.cpu()).float().mean().item())

        by_layer = []
        for l in range(L):
            ab_r = eval_ablate([l], pred_pos)
            ab_c = eval_ablate([l], ctrl_pos)
            ld = float((ab_r[:, id1] - ab_r[:, id2]).mean().item())
            acc = float((ab_r.argmax(dim=-1) == id1).float().mean().item())
            c_acc = float((ab_c.argmax(dim=-1) == ctrl_targets.cpu()).float().mean().item())
            by_layer.append({
                "layer": l,
                "ring_logit_diff": ld,
                "delta_ring_logit_diff": ld - clean_ld,
                "ring_acc": acc,
                "delta_ring_acc": acc - clean_acc,
                "ctrl_acc": c_acc,
                "delta_ctrl_acc": c_acc - clean_ctrl_acc,
            })

        ab_late = eval_ablate([9, 10, 11], pred_pos)
        ab_late_c = eval_ablate([9, 10, 11], ctrl_pos)
        out["per_seed"][f"seed{sd}"] = {
            "clean_ring_logit_diff": clean_ld,
            "clean_ring_acc": clean_acc,
            "clean_ctrl_acc": clean_ctrl_acc,
            "by_layer": by_layer,
            "joint_late_9_10_11": {
                "ring_logit_diff": float((ab_late[:, id1] - ab_late[:, id2]).mean().item()),
                "delta_ring_logit_diff": float((ab_late[:, id1] - ab_late[:, id2]).mean().item()) - clean_ld,
                "ring_acc": float((ab_late.argmax(dim=-1) == id1).float().mean().item()),
                "delta_ring_acc": float((ab_late.argmax(dim=-1) == id1).float().mean().item()) - clean_acc,
                "ctrl_acc": float((ab_late_c.argmax(dim=-1) == ctrl_targets.cpu()).float().mean().item()),
                "delta_ctrl_acc": float((ab_late_c.argmax(dim=-1) == ctrl_targets.cpu()).float().mean().item()) - clean_ctrl_acc,
            },
        }
        del model
        torch.cuda.empty_cache()

    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {
        "chemberta_mtr_head_ablation": run_chemberta_mtr_head_ablation(device),
        "large_mlp_ablation_necessity": run_large_mlp_ablation_necessity(device),
    }
    out_p = REPO / "results" / "audit" / "chemberta_and_mlp_ablation.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2) + "\n")

    cm = res["chemberta_mtr_head_ablation"]
    print("=== CHEMBERTA-77M-MTR L1H4 CAUSAL ABLATION ===")
    print("Baseline:", cm["baseline"])
    print("L1H4:", cm["L1H4"])
    print("Top-3 NumRings R2 drop:", [(r["name"], round(r["delta_r2_nrings"], 4), round(r["delta_cka_l2"], 4)) for r in cm["top5_nrings_drop"][:3]])
    print("Top-3 CKA drop:", [(r["name"], round(r["delta_cka_l2"], 4)) for r in cm["top5_cka_drop"][:3]])
    print("Top-3 isomer drop:", [(r["name"], round(r["delta_iso_top1_l2"], 4)) for r in cm["top5_iso_drop"][:3]])

    ml = res["large_mlp_ablation_necessity"]
    print("\n=== LARGE MLP ZERO-ABLATION NECESSITY (SEED 0) ===")
    s0 = ml["per_seed"]["seed0"]
    print(f"Clean ring acc={s0['clean_ring_acc']:.4f}, ld={s0['clean_ring_logit_diff']:.2f}, ctrl_acc={s0['clean_ctrl_acc']:.4f}")
    for row in s0["by_layer"]:
        print(f"  L{row['layer']:2d}: d_ld={row['delta_ring_logit_diff']:+6.2f}, d_ring_acc={100*row['delta_ring_acc']:+6.2f}pp, d_ctrl_acc={100*row['delta_ctrl_acc']:+6.2f}pp")
    print("  Joint L9-11:", s0["joint_late_9_10_11"])


if __name__ == "__main__":
    main()
