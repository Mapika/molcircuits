"""Graph-based bond outcomes for matched generation attempts."""
from collections import defaultdict

import numpy as np
from rdkit import Chem, rdBase

from molcircuits.analysis.generation_outcomes import wilson_interval
from molcircuits.chem.tokenizer import tokenize


def measure_attempt(sample_id, smiles, terminated):
    """Count heavy-atom bonds after EOS and RDKit sanitization, including aromatic bonds in the denominator."""
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles) if smiles else None
    valid = mol is not None and mol.GetNumAtoms() > 0
    row = {
        "sample_id": sample_id, "smiles": smiles, "terminated": bool(terminated),
        "parseable": valid, "complete_valid": bool(terminated and valid),
        "decoded_tokens": len(tokenize(smiles)), "heavy_atoms": None, "heavy_bonds": None,
        "double": None, "triple": None, "aromatic": None,
    }
    if row["complete_valid"]:
        bonds = [bond for bond in mol.GetBonds()
                 if bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1]
        nonaromatic = [bond.GetBondType() for bond in bonds if not bond.GetIsAromatic()]
        row.update(heavy_atoms=sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms()),
                   heavy_bonds=len(bonds), double=nonaromatic.count(Chem.BondType.DOUBLE),
                   triple=nonaromatic.count(Chem.BondType.TRIPLE), aromatic=len(bonds) - len(nonaromatic))
    return row


def metric_arrays(rows):
    """NaN denotes an ineligible molecule. Per-attempt count endpoints give failed attempts zero."""
    return {
        "multiple_fraction_valid": np.array([(r["double"] + r["triple"]) / r["heavy_bonds"]
                                            if r["heavy_bonds"] else np.nan for r in rows]),
        "complete_valid_rate": np.array([r["complete_valid"] for r in rows], dtype=float),
        "double_count_per_attempt": np.array([r["double"] or 0 for r in rows], dtype=float),
        "triple_count_per_attempt": np.array([r["triple"] or 0 for r in rows], dtype=float),
        "mean_decoded_tokens": np.array([r["decoded_tokens"] for r in rows], dtype=float),
        "mean_heavy_atoms_valid": np.array([r["heavy_atoms"] for r in rows], dtype=float),
    }


def summarize(rows):
    values = metric_arrays(rows)
    result = {}
    for k, vector in values.items():
        finite = vector[np.isfinite(vector)]
        result[k] = float(finite.mean()) if len(finite) else None
    valid = sum(r["complete_valid"] for r in rows)
    ended = sum(r["terminated"] for r in rows)
    result.update(n_attempted=len(rows), n_complete_valid=valid, n_terminated=ended,
                  n_bond_eligible=int(np.isfinite(values["multiple_fraction_valid"]).sum()),
                  termination_rate=ended / len(rows) if rows else None,
                  complete_valid_ci95=list(wilson_interval(valid, len(rows))),
                  n_parseable_without_eos=sum(r["parseable"] and not r["terminated"] for r in rows))
    return result


def heavy_atom_standardized(left, reference):
    """Average within-size primary differences over common sizes, using reference-arm frequencies."""
    def group(rows):
        buckets = defaultdict(list)
        for r in rows:
            if r["heavy_bonds"]:
                buckets[r["heavy_atoms"]].append((r["double"] + r["triple"]) / r["heavy_bonds"])
        return buckets

    a, b = group(left), group(reference)
    shared = sorted(a.keys() & b.keys())
    n_ref = sum(len(b[k]) for k in shared)
    delta = sum(len(b[k]) * (np.mean(a[k]) - np.mean(b[k])) for k in shared)
    return {"difference": float(delta / n_ref) if n_ref else None,
            "n_common_sizes": len(shared), "n_left_retained": sum(len(a[k]) for k in shared),
            "n_reference_retained": n_ref}


def paired_contrast(left, reference, n_bootstrap, seed):
    """Bootstrap common attempt indices, re-estimating each arm's eligible denominator on each draw."""
    if not left or [r["sample_id"] for r in left] != [r["sample_id"] for r in reference]:
        raise ValueError("matched nonempty sample IDs required")
    a, b = metric_arrays(left), metric_arrays(reference)
    means_a, means_b = summarize(left), summarize(reference)
    rng = np.random.default_rng(seed)
    differences = {k: [] for k in a}
    for start in range(0, n_bootstrap, 32):
        index = rng.integers(len(left), size=(min(32, n_bootstrap - start), len(left)))
        for k in a:
            with np.errstate(invalid="ignore", divide="ignore"):
                av, bv = a[k][index], b[k][index]
                av = np.nansum(av, axis=1) / np.isfinite(av).sum(axis=1)
                bv = np.nansum(bv, axis=1) / np.isfinite(bv).sum(axis=1)
            delta = av - bv
            differences[k].extend(delta[np.isfinite(delta)].tolist())
    result = {}
    for k, samples in differences.items():
        estimate = None if means_a[k] is None or means_b[k] is None else means_a[k] - means_b[k]
        result[k] = {"difference": estimate, "ci95": np.quantile(samples, [0.025, 0.975]).tolist()
                     if samples else [None, None], "n_bootstrap_finite": len(samples)}
    result["heavy_atom_standardized"] = heavy_atom_standardized(left, reference)
    return result
