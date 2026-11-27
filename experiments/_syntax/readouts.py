"""Compare valence readout and ring relabeling on identical inputs across model sizes.

Pass --root for the directory containing data/ and grid/. The valence probe uses
the layer nearest two thirds of network depth. The ring comparison retains the
same molecular graph and asks whether the closure follows the altered opener.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import torch

from molcircuits.analysis.valence_probe import _fit_eval, bootstrap_ci
from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.chem.valence import valence_labels
from molcircuits.model.hooks import HookManager, resid_post
from molcircuits.model.transformer import ModelConfig, MolecularTransformer

from _syntax import ring_patching as ring_analysis


@torch.no_grad()
def _collect_valence_residuals(model, tok, smiles, layer, n_ctx, device, batch=128):
    A, Y, G, T = [], [], [], []
    for s0 in range(0, len(smiles), batch):
        chunk = smiles[s0:s0 + batch]
        enc = [tok.encode(s)[:n_ctx] for s in chunk]
        width = max(len(e) for e in enc)
        ids = [e + [tok.pad_id] * (width - len(e)) for e in enc]
        x = torch.tensor(ids, device=device)
        h = HookManager().capture(resid_post(layer))
        model(x, hooks=h)
        acts = h[resid_post(layer)].float().cpu().numpy()
        for bi, s in enumerate(chunk):
            toks = tokenize(s)
            for lab in valence_labels(toks, offset=1):
                if lab.token_idx >= n_ctx:
                    continue
                A.append(acts[bi, lab.token_idx])
                Y.append(lab.remaining)
                G.append(s0 + bi)
                T.append(ids[bi][lab.token_idx])
    return (np.array(A, dtype=np.float32), np.array(Y), np.array(G), np.array(T))


def only_ring_one(smiles):
    for row in smiles:
        digits = [t for t in tokenize(row) if t.isdigit() or t.startswith("%")]
        if digits == ["1", "1"]:
            yield row


@torch.no_grad()
def ring_metrics(model, tok, pairs, device):
    contrasts, full_hits = [], []
    width = max(len(p["clean_ids"]) for p in pairs)
    for key, digit in (("clean_ids", "1"), ("corrupt_ids", "2")):
        logits = []
        for start in range(0, len(pairs), 64):
            batch = pairs[start:start + 64]
            ids = ring_analysis.pad_batch(batch, key, tok.pad_id, width).to(device)
            positions = [p["predict_idx"] for p in batch]
            logits.append(model(ids)[torch.arange(len(batch), device=device), positions].cpu())
        scores = torch.cat(logits)
        contrasts.append((scores[:, tok.vocab["1"]] - scores[:, tok.vocab["2"]]).numpy())
        full_hits.append(int((scores.argmax(1) == tok.vocab[digit]).sum()))
    original, relabeled = contrasts
    change = original - relabeled
    rng = np.random.default_rng(51)
    boot = [change[rng.integers(len(change), size=len(change))].mean() for _ in range(2000)]
    return {
        "original_1_beats_2": int((original > 0).sum()),
        "renamed_2_beats_1": int((relabeled < 0).sum()),
        "both_correct_in_digit_comparison": int(((original > 0) & (relabeled < 0)).sum()),
        "original_full_vocab_top1": full_hits[0], "renamed_full_vocab_top1": full_hits[1],
        "mean_original_logit1_minus_logit2": float(original.mean()),
        "mean_renamed_logit1_minus_logit2": float(relabeled.mean()),
        "paired_logit_difference": float(change.mean()),
        "paired_difference_bootstrap_ci": np.quantile(boot, [.025, .975]).tolist(),
    }


def atom_metrics(model, tok, molecules, device):
    depth = model.cfg.n_layers
    layer = round(2 * depth / 3) - 1
    with torch.no_grad():
        x, y, groups, tokens = _collect_valence_residuals(
            model, tok, molecules, layer, model.cfg.n_ctx, device, batch=64)
    classifier, accuracy, tr, te = _fit_eval(x, y, groups, seed=51, device=device)
    assert not set(groups[tr]).intersection(groups[te])
    # Fit the most frequent capacity per token on the training molecules.
    counts = np.zeros((tok.vocab_size, 5), dtype=int)
    np.add.at(counts, (tokens[tr], y[tr]), 1)
    modes = counts.argmax(axis=1)
    modes[counts.sum(axis=1) == 0] = np.bincount(y[tr]).argmax()
    token_accuracy = (modes[tokens[te]] == y[te]).mean()
    _, lo, hi = bootstrap_ci(y[te], classifier.predict(x[te]), groups[te], n_boot=1000, seed=51)
    return {
        "zero_based_layer": layer, "n_train_molecules": len(set(groups[tr])),
        "n_test_molecules": len(set(groups[te])), "n_test_atoms": len(te),
        "test_labels_and_groups_sha256": hashlib.sha256(np.stack((y[te], groups[te], tokens[te])).tobytes()).hexdigest(),
        "atom_accuracy": accuracy, "molecule_bootstrap_ci": [lo, hi],
        "current_token_accuracy": float(token_accuracy),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    data = args.root / "data"
    tok = SmilesTokenizer.load(data / "tokenizer.json")
    population = (data / "valid.smi").read_text().splitlines()
    rng = np.random.default_rng(51)
    rows = [population[i] for i in rng.choice(len(population), 200_000, replace=False)]
    atoms = list(dict.fromkeys(rows))[:2000]
    pairs = ring_analysis.build_pairs(only_ring_one(dict.fromkeys(rows)), tok, n_ctx=128, max_pairs=1024)
    assert len(atoms) == 2000 and len(pairs) == 1024
    out = {
        "sample_seed": 51, "atom_sample_sha256": hashlib.sha256("\n".join(atoms).encode()).hexdigest(),
        "ring_sample_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
        "n_ring_pairs": len(pairs), "runs": {},
    }
    for size in ["small", "base", "large"]:
        for seed in range(3):
            path = args.root / "grid" / f"zinc_{size}" / f"tf_seed{seed}"
            cfg = ModelConfig.load(path / "model_config.json")
            model = MolecularTransformer(cfg).to(args.device).eval()
            model.load_state_dict(torch.load(path / "checkpoint.pt", map_location="cpu")["model"])
            ring = ring_metrics(model, tok, pairs, args.device)
            atom = atom_metrics(model, tok, atoms, args.device)
            print(size, seed, atom["atom_accuracy"], ring["renamed_2_beats_1"], flush=True)
            out["runs"][f"{size}_{seed}"] = {"checkpoint": str(path.relative_to(args.root)),
                                              "valence_readout": atom, "ring_relabeling": ring}
    splits = {r["valence_readout"]["test_labels_and_groups_sha256"] for r in out["runs"].values()}
    assert len(splits) == 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
