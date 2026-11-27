"""Late-layer residual steering control (L8-L11) and RDKit-completable minimal-pair valence patching.

1. Late-layer residual steering control on Large canonical seed 0 (KV-cached, 512 attempts/arm):
   Tests single-layer residual injection at layers 5 (baseline best_mid), 8, 9, 10, and 11
   for MolLogP, TPSA, and NumAromaticRings (alpha in {-8, -4, +4, +8}).
2. Strictly RDKit-completable minimal-pair valence patching (1 pair per parent molecule):
   Builds up to 256 near (<=3), 256 far (>=10), and 256 far capacity-preserving control pairs
   where BOTH original and edited prefixes up to decision d are verified chemically valid and
   RDKit-completable (full RDKit sanitization after closing open branches/rings, and verifying
   that appending '=C' at d is RDKit-legal on the cap>=2 prefix and RDKit-illegal on cap<=1).
"""
from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import bond_decision_labels
from molcircuits.model.hooks import HookManager, resid_post

RDLogger.DisableLog("rdApp.*")
_gi = import_module("04_graph_invariance")
_st = import_module("09_constructive_steering")


def is_prefix_rdkit_completable(prefix_toks: list[str], append_tail: str = "C") -> bool:
    """Verify a SMILES prefix can be completed into a fully RDKit-sanitized molecule.

    Appends `append_tail` at the active atom, closes any open branch parentheses, and closes
    any open ring digits via valid single-bonded carbon links. If RDKit sanitizes the result,
    no syntax error or pre-prediction valence overflow exists anywhere in the prefix.
    """
    s = "".join(prefix_toks)
    if "=(" in s or "==" in s or "=#" in s or "#=" in s or "##" in s or "()" in s:
        return False
    # Count open parens and unmatched ring digits
    depth = 0
    open_rings: list[str] = []
    for tk in prefix_toks:
        if tk == "(":
            depth += 1
        elif tk == ")":
            depth -= 1
            if depth < 0:
                return False
        elif tk.isdigit() or (tk.startswith("%") and tk[1:].isdigit()):
            if tk in open_rings:
                open_rings.remove(tk)
            else:
                open_rings.append(tk)
    # Build completion: append_tail at active atom, close parens, then close any open rings on a chain
    # First attach any open ring closures to separate carbons so no ring closes onto itself
    tail = append_tail
    if open_rings:
        # Close each open ring via -CC<ring> so even a ring opened on the active atom forms a >=3-atom ring
        ring_tail = "".join(f"CC{r}" for r in open_rings)
        cand = s + tail + (")" * depth) + ring_tail
        if Chem.MolFromSmiles(cand) is not None:
            return True
        cand2 = s + tail + ring_tail + (")" * depth)
        return Chem.MolFromSmiles(cand2) is not None
    cand = s + tail + (")" * depth)
    return Chem.MolFromSmiles(cand) is not None


@torch.no_grad()
def run_late_layer_steering(device: str) -> dict:
    from molcircuits import paths as _paths

    ckpt = str(Path(_paths.root()) / "grid" / "zinc_large" / "tf_seed0" / "checkpoint.pt")
    smi_path, tok_path = _gi.resolve_data_and_tokenizer(_paths.data(), None)
    tok = SmilesTokenizer.load(tok_path)
    model = _gi.load_model(ckpt, device)
    D = model.cfg.d_model

    smis = []
    with open(smi_path) as fh:
        for ln in fh:
            s = ln.strip().split()[0]
            if len(s) <= 120:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    smis.append((s, float(Descriptors.MolLogP(m)), float(Descriptors.TPSA(m)), float(rdMolDescriptors.CalcNumAromaticRings(m))))
            if len(smis) >= 4000:
                break

    test_layers = [5, 8, 9, 10, 11]
    hook_names = [resid_post(l) for l in test_layers]
    H_by_l = {l: np.empty((len(smis), D), dtype=np.float32) for l in test_layers}
    rms_by_l = {}

    for i0 in range(0, len(smis), 128):
        chunk = [r[0] for r in smis[i0 : i0 + 128]]
        seqs = [tok.encode(s) for s in chunk]
        ml = max(len(e) for e in seqs)
        x = torch.full((len(chunk), ml), tok.pad_id, dtype=torch.long, device=device)
        for bi, e in enumerate(seqs):
            x[bi, : len(e)] = torch.tensor(e, dtype=torch.long, device=device)
        hm = HookManager().capture(*hook_names)
        model(x, hooks=hm)
        for l in test_layers:
            arr = hm[resid_post(l)].float().cpu().numpy()
            for bi, e in enumerate(seqs):
                H_by_l[l][i0 + bi] = arr[bi, 1 : len(e) - 1].mean(axis=0)

    for l in test_layers:
        rms_by_l[l] = float(np.sqrt(np.mean(H_by_l[l] ** 2)))

    props = {
        "MolLogP": (np.array([r[1] for r in smis]), lambda m: float(Descriptors.MolLogP(m)), 3.5),
        "TPSA": (np.array([r[2] for r in smis]), lambda m: float(Descriptors.TPSA(m)), 90.0),
        "NumAromaticRings": (np.array([r[3] for r in smis]), lambda m: float(rdMolDescriptors.CalcNumAromaticRings(m)), 3.0),
    }

    dirs = {p: {} for p in props}
    r2s = {p: {} for p in props}
    for pname, (y_all, _, _) in props.items():
        y_tr, y_te = y_all[:2800], y_all[2800:]
        for l in test_layers:
            X = H_by_l[l].astype(np.float64)
            X_tr, X_te = X[:2800], X[2800:]
            mu = X_tr.mean(0, keepdims=True)
            Xc = X_tr - mu
            ym = y_tr.mean()
            w = np.linalg.solve(Xc.T @ Xc + 100.0 * np.eye(D), Xc.T @ (y_tr - ym))
            pred = (X_te - mu) @ w + ym
            r2s[pname][f"L{l}"] = 1.0 - float(np.sum((y_te - pred) ** 2)) / float(np.sum((y_te - y_te.mean()) ** 2))
            w_hat = w / max(float(np.linalg.norm(w)), 1e-12)
            dirs[pname][l] = torch.tensor(w_hat, dtype=torch.float32, device=device)

    n_samp = 512
    base_smis, _, _ = _st.generate_steered(model, tok, n_samp, seed=123, batch_size=512, max_tokens=128, device=device)
    base_valid = [m for s in base_smis if (m := Chem.MolFromSmiles(s)) is not None]
    out = {
        "model": "zinc_large_seed0",
        "n_samples_per_arm": n_samp,
        "probe_r2": r2s,
        "unsteered": {
            "valid_rate": len(base_valid) / float(n_samp),
            **{
                p: {
                    "mean": float(np.mean([fn(m) for m in base_valid])),
                    "hit_yield": float(sum(fn(m) >= thr if p == "NumAromaticRings" else fn(m) > thr for m in base_valid) / float(n_samp)),
                }
                for p, (_, fn, thr) in props.items()
            },
        },
        "by_property": {},
    }

    for pname, (_, fn, thr) in props.items():
        base_m = out["unsteered"][pname]["mean"]
        out["by_property"][pname] = {}
        for l in test_layers:
            l_res = {}
            for alpha in (-8.0, -4.0, 4.0, 8.0):
                delta = alpha * rms_by_l[l] * dirs[pname][l]
                gen, _, _ = _st.generate_steered(
                    model, tok, n_samp, layer_edits={l: delta}, seed=123, batch_size=512, max_tokens=128, device=device
                )
                vmols = [m for s in gen if (m := Chem.MolFromSmiles(s)) is not None]
                vr = len(vmols) / float(n_samp)
                vals = [fn(m) for m in vmols]
                mean_v = float(np.mean(vals)) if vals else float("nan")
                hits = sum((v >= thr if pname == "NumAromaticRings" else v > thr) for v in vals)
                l_res[f"{alpha:+.0f}"] = {
                    "valid_rate": vr,
                    "mean": mean_v,
                    "delta_mean": mean_v - base_m if vals else float("nan"),
                    "hit_yield": hits / float(n_samp),
                }
            out["by_property"][pname][f"L{l}"] = l_res

    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def run_rdkit_valid_valence_patching(device: str) -> dict:
    """Strictly RDKit-completable minimal pairs (1 pair/parent molecule): near, far, and far capacity-preserving control."""
    from molcircuits import paths as _paths

    smi_path, tok_path = _gi.resolve_data_and_tokenizer(_paths.data(), None)
    tok = SmilesTokenizer.load(tok_path)
    eq_id, c_id = tok.vocab["="], tok.vocab["C"]
    subs = {"=": ["C"], "#": ["C"], "C": ["=", "O", "N"], "O": ["C"], "N": ["C"]}

    near_pairs, far_pairs, far_ctrl_pairs = [], [], []
    seen_parents = set()

    with open(smi_path) as fh:
        for line_i, ln in enumerate(fh):
            if line_i >= 120000 or (len(near_pairs) >= 256 and len(far_pairs) >= 256 and len(far_ctrl_pairs) >= 256):
                break
            s = ln.strip().split()[0]
            if len(s) > 110 or s in seen_parents:
                continue
            tks = tokenize(s)
            base_cap = dict(bond_decision_labels(tks))
            if not base_cap:
                continue
            need_far = len(far_pairs) < 256
            need_near = len(near_pairs) < 256
            need_ctrl = len(far_ctrl_pairs) < 256

            found_here = False
            for i, tk in enumerate(tks):
                if found_here:
                    break
                if not any(
                    (need_far and d - i >= 10) or (need_ctrl and d - i >= 10) or (need_near and 2 <= d - i <= 3)
                    for d in base_cap
                ):
                    continue
                for repl in subs.get(tk, []):
                    alt = list(tks)
                    alt[i] = repl
                    try:
                        alt_cap = dict(bond_decision_labels(alt))
                    except Exception:
                        continue
                    matched_d, matched_kind = None, None
                    for d, cap in base_cap.items():
                        if d not in alt_cap:
                            continue
                        gap = d - i
                        if gap < 2:
                            continue
                        flipped = (cap >= 2) != (alt_cap[d] >= 2)
                        if flipped and gap >= 10 and need_far:
                            matched_d, matched_kind = d, "far"
                            break
                        elif flipped and 2 <= gap <= 3 and need_near:
                            matched_d, matched_kind = d, "near"
                            break
                        elif (not flipped) and gap >= 10 and cap >= 2 and need_ctrl:
                            matched_d, matched_kind = d, "ctrl"
                    if matched_d is None:
                        continue

                    # Prefix tokens UP TO AND INCLUDING governing position d (before bond decision d+1)
                    pref_a = tks[: matched_d + 1]
                    pref_b = alt[: matched_d + 1]
                    cap_a = base_cap[matched_d]
                    cap_b = alt_cap[matched_d]

                    # Verify BOTH prefixes are chemically completable via RDKit with a single-bonded C!
                    if not (is_prefix_rdkit_completable(pref_a, "C") and is_prefix_rdkit_completable(pref_b, "C")):
                        continue
                    if matched_kind in ("near", "far"):
                        # Verify '=C' is RDKit-legal on the cap>=2 prefix and RDKit-illegal on cap<=1!
                        hi_pref, lo_pref = (pref_a, pref_b) if cap_a >= 2 else (pref_b, pref_a)
                        if not is_prefix_rdkit_completable(hi_pref, "=C"):
                            continue
                        if is_prefix_rdkit_completable(lo_pref, "=C"):
                            continue

                    ce = tok.encode(s)
                    co = tok.encode("".join(alt))
                    if len(ce) != len(co) or len(ce) > 128:
                        continue
                    pred = matched_d + 1
                    if [k for k in range(pred + 1) if ce[k] != co[k]] != [i + 1]:
                        continue
                    gap = matched_d - i
                    rec = (ce, co, pred, gap, int(cap_a >= 2), int(cap_b >= 2))
                    if matched_kind == "far":
                        far_pairs.append(rec if cap_a >= 2 else (co, ce, pred, gap, int(cap_b >= 2), int(cap_a >= 2)))
                    elif matched_kind == "near":
                        near_pairs.append(rec if cap_a >= 2 else (co, ce, pred, gap, int(cap_b >= 2), int(cap_a >= 2)))
                    else:
                        far_ctrl_pairs.append(rec)
                    seen_parents.add(s)
                    found_here = True
                    break

    print(f"Collected RDKit-completable pairs: near={len(near_pairs)}, far={len(far_pairs)}, far_ctrl={len(far_ctrl_pairs)}", flush=True)

    def eval_group(model, grp):
        L = model.cfg.n_layers
        n = len(grp)
        ml = max(max(len(r[0]), len(r[1])) for r in grp)
        xc = torch.full((n, ml), tok.pad_id, dtype=torch.long, device=device)
        xo = torch.full((n, ml), tok.pad_id, dtype=torch.long, device=device)
        for bi, (ce, co, _, _, _, _) in enumerate(grp):
            xc[bi, : len(ce)] = torch.tensor(ce, dtype=torch.long, device=device)
            xo[bi, : len(co)] = torch.tensor(co, dtype=torch.long, device=device)
        pp = torch.tensor([r[2] for r in grp], dtype=torch.long, device=device)
        ar = torch.arange(n, device=device)

        hnames = [resid_post(l) for l in range(L)]
        hm_c = HookManager().capture(*hnames)
        lg_c = model(xc, hooks=hm_c)[ar, pp]
        lg_o = model(xo)[ar, pp]
        clean_ld = (lg_c[:, eq_id] - lg_c[:, c_id]).float().cpu().numpy()
        corrupt_ld = (lg_o[:, eq_id] - lg_o[:, c_id]).float().cpu().numpy()
        denom = clean_ld - corrupt_ld

        rec_by_l = []
        for l in range(L):
            src = hm_c[resid_post(l)][ar, pp].clone()

            def _patch(_m, _inp, out_t, s_t=src):
                z = out_t.clone()
                z[ar, pp] = s_t
                return z

            h = model.blocks[l].register_forward_hook(_patch)
            try:
                lg_p = model(xo)[ar, pp]
            finally:
                h.remove()
            p_ld = (lg_p[:, eq_id] - lg_p[:, c_id]).float().cpu().numpy()
            rec = float(np.sum(p_ld - corrupt_ld) / max(float(np.sum(denom)), 1e-6))
            rec_by_l.append(rec)

        first50 = next((l + 1 for l, r in enumerate(rec_by_l) if r >= 0.5), None)
        return {
            "n_pairs": n,
            "mean_clean_ld": float(np.mean(clean_ld)),
            "mean_corrupt_ld": float(np.mean(corrupt_ld)),
            "mean_contrast": float(np.mean(denom)),
            "recovery_by_block_1indexed": rec_by_l,
            "first_block_ge_50pct": first50,
            "median_gap": float(np.median([r[3] for r in grp])),
        }

    out = {"per_seed": {}}
    for sd in (0, 1, 2):
        ckpt = str(Path(_paths.root()) / "grid" / "zinc_large" / f"tf_seed{sd}" / "checkpoint.pt")
        model = _gi.load_model(ckpt, device)
        out["per_seed"][f"seed{sd}"] = {
            "near_rdkit_valid": eval_group(model, near_pairs),
            "far_rdkit_valid": eval_group(model, far_pairs),
            "far_capacity_preserving_ctrl": eval_group(model, far_ctrl_pairs),
        }
        del model
        torch.cuda.empty_cache()
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {
        "late_layer_steering_large_s0": run_late_layer_steering(device),
        "rdkit_valid_valence_patching": run_rdkit_valid_valence_patching(device),
    }
    out_p = REPO / "results" / "audit" / "steering_and_valence_controls.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2) + "\n")

    print("=== 1. LATE-LAYER RESIDUAL STEERING (LARGE SEED 0, KV-CACHED, N=512) ===")
    st = res["late_layer_steering_large_s0"]
    print("Unsteered:", st["unsteered"])
    for prop, lmap in st["by_property"].items():
        print(f"\nProperty: {prop}")
        for ln, amap in lmap.items():
            row_s = " | ".join(
                f"{a}: val={v['valid_rate']*100:.1f}%, d={v['delta_mean']:+.2f}, yld={v['hit_yield']*100:.1f}%"
                for a, v in amap.items()
            )
            print(f"  {ln:3s}: {row_s}")

    print("\n=== 2. RDKIT-COMPLETABLE MINIMAL-PAIR VALENCE PATCHING (1 PAIR / PARENT MOL) ===")
    vp = res["rdkit_valid_valence_patching"]["per_seed"]
    for sd, d in vp.items():
        n = d["near_rdkit_valid"]
        f = d["far_rdkit_valid"]
        c = d["far_capacity_preserving_ctrl"]
        print(
            f"  {sd}: near(n={n['n_pairs']}, contrast={n['mean_contrast']:.2f}) -> 50% at block {n['first_block_ge_50pct']} | "
            f"far(n={f['n_pairs']}, med_gap={f['median_gap']:.0f}, contrast={f['mean_contrast']:.2f}) -> 50% at block {f['first_block_ge_50pct']} | "
            f"far_ctrl(n={c['n_pairs']}, contrast={c['mean_contrast']:.2f})"
        )


if __name__ == "__main__":
    main()
