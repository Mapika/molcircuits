#!/usr/bin/env python3
"""Held-out matched within-molecule WSD controls for fragment alignment.

    python scripts/17_fragment_alignment_wsd_controls.py \
        --ckpt artifacts/tf_seed0/checkpoint.pt \
        --sae artifacts/saes_seed0/layer3_x16 \
        --data data --out artifacts/fragment_controls_seed0 --seed 0

This schema-v2 analysis is the matched control for stage 06's localization
claim.  It uses the same per-molecule coverage-weighted standardized difference
over atom tokens, averages fragment-containing molecules equally, and separates
coordinate selection from held-out evaluation.  The schema-v1 script remains a
distinct secondary analysis of pooled global atom-token discrimination.

For molecule b and coordinate j,

    d_bj = ((mu_in - mu_out) / sigma_all) * sqrt(p_b * (1 - p_b)).

The primary detector chooses ``argmax_j |mean_b d_bj|`` on selection molecules,
fixes that coordinate and its sign, and evaluates ``sign * mean_b d_bj`` on
evaluation molecules.  The secondary detector chooses
``argmax_j mean_b |d_bj|`` and evaluates ``mean_b |d_bj|`` at that same fixed
coordinate.  Neither detector can re-select on evaluation data.

The whole-pipeline null independently shuffles fragment membership within each
molecule on both halves while preserving per-molecule coverage, re-selects the
null coordinate/orientation on selection labels, and evaluates it unchanged on
the corresponding evaluation null labels.  Only null summaries are persisted.

The primary estimand cannot tell a coordinate that knows an atom's own symbol
from one that knows its chemical context: the ``atom_type_onehot`` arm, which
contains no model activations, matches or beats the SAE on it.  A second,
within-type estimand is therefore reported beside it.  For token type T it
contrasts in-fragment and out-of-fragment atoms of type T in the same molecule,
pooled over types with the Cochran-Mantel-Haenszel weight
``n_in,T * n_out,T / n_T``; equivalently it is the semipartial correlation of
the coordinate with the fragment label after removing token-type means from the
label (``within_type_wsd_coordinates``).  Any coordinate constant within token
type, the one-hot arm included, scores exactly zero.  Only molecules with a
token type on both sides of the match contribute, and the same
--min-molecules-per-half rule is applied to those counts.  Selection,
orientation and held-out evaluation follow the primary protocol, and the
type-matched null (a stratified within-type permutation) is its null.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os

import numpy as np
import torch

from molcircuits.analysis.fragment_controls import (
    WSD_SCHEMA_VERSION, aggregate_wsd_fragment_results,
    fit_selection_standardizer, heldout_within_type_detectors,
    heldout_wsd_detectors, heldout_wsd_pipeline_null,
    permute_fragment_masks_within_molecules,
    shift_fragment_masks_within_molecules, type_matched_fragment_masks,
    matched_arm_names, random_unit_projection, relu_topk, within_type_eligible_molecules,
    within_type_wsd_coordinates)
from molcircuits.analysis.wsd import basis_stats, mean_wsd_coordinates, to_basis
from molcircuits.atlas.core import deterministic_molecule_split
from molcircuits.chem.fragments import (atom_token_mask, build_masks, coverage,
                                    load_smarts_library, save_smarts_library,
                                    token_identity_array)
from molcircuits.downstream.features import load_stack
from molcircuits.model.hooks import HookManager, resid_post
from molcircuits.utils.manifest import Manifest, dump_json, file_sha256

ARTIFACT_NAME = f"fragment_alignment_controls_v{WSD_SCHEMA_VERSION}.json"


def _sample_encodable(pool, tokenizer, n_molecules: int, n_ctx: int,
                      seed: int):
    """Sample corpus rows, dropping rather than truncating invalid encodings."""
    rng = np.random.default_rng(seed)
    n_take = min(len(pool), n_molecules)
    source_indices = np.sort(rng.choice(len(pool), n_take, replace=False))
    smiles, kept_source, dropped = [], [], []
    for source_i in source_indices:
        smile = pool[int(source_i)]
        try:
            encoded = tokenizer.encode(smile)
        except ValueError:
            dropped.append({"source_index": int(source_i),
                            "reason": "tokenizer_failed"})
            continue
        if len(encoded) > n_ctx:
            dropped.append({"source_index": int(source_i),
                            "reason": "overlength",
                            "encoded_length": len(encoded)})
            continue
        smiles.append(smile)
        kept_source.append(int(source_i))
    return smiles, kept_source, dropped


@torch.no_grad()
def _collect_atom_residuals(model, tokenizer, smiles, population, layer: int,
                            n_ctx: int, device: str, batch: int):
    """Collect residual rows only at atom-token positions."""
    n_rows = int(population.sum())
    dense = np.empty((n_rows, model.cfg.d_model), dtype=np.float32)
    molecule_index = np.empty(n_rows, dtype=np.int32)
    token_index = np.empty(n_rows, dtype=np.int16)
    cursor = 0
    for start in range(0, len(smiles), batch):
        chunk = smiles[start:start + batch]
        encoded = [tokenizer.encode(smile) for smile in chunk]
        ids = [row + [tokenizer.pad_id] * (n_ctx - len(row)) for row in encoded]
        x = torch.tensor(ids, device=device)
        hooks = HookManager().capture(resid_post(layer))
        model(x, hooks=hooks)
        residual = hooks[resid_post(layer)].float()

        local_mask = population[start:start + len(chunk)]
        local_positions = np.argwhere(local_mask)
        selected = residual[
            torch.as_tensor(local_mask, device=device, dtype=torch.bool)]
        end = cursor + len(local_positions)
        dense[cursor:end] = selected.cpu().numpy()
        molecule_index[cursor:end] = start + local_positions[:, 0]
        token_index[cursor:end] = local_positions[:, 1]
        cursor = end
        if (start // batch) % 8 == 0:
            print(f"  residuals: {min(start + batch, len(smiles)):>6,} / "
                  f"{len(smiles):,} molecules", flush=True)
    if cursor != n_rows:
        raise RuntimeError(f"collected {cursor} atom rows, expected {n_rows}")
    return dense, molecule_index, token_index


@torch.no_grad()
def _transform_rows(dense: np.ndarray, row_mask: np.ndarray, out_dim: int,
                    transform, device: str, batch: int) -> np.ndarray:
    source_rows = np.flatnonzero(row_mask)
    out = np.empty((len(source_rows), out_dim), dtype=np.float32)
    for start in range(0, len(source_rows), batch):
        end = min(start + batch, len(source_rows))
        x = torch.as_tensor(
            np.ascontiguousarray(dense[source_rows[start:end]]), device=device)
        y = transform(x)
        if y.shape != (end - start, out_dim):
            raise RuntimeError(
                f"transform returned {tuple(y.shape)}, expected {(end-start, out_dim)}")
        out[start:end] = y.float().cpu().numpy()
    return out


def _packing(molecule_index: np.ndarray, row_mask: np.ndarray,
             molecule_indices: np.ndarray) -> dict:
    """Describe a compact molecule x atom-slot layout for selected flat rows."""
    row_molecules = np.asarray(molecule_index[row_mask], dtype=np.int64)
    molecule_indices = np.asarray(molecule_indices, dtype=np.int64)
    counts_all = np.bincount(molecule_index, minlength=int(molecule_index.max()) + 1)
    counts = counts_all[molecule_indices]
    expected = np.repeat(molecule_indices, counts)
    if not np.array_equal(row_molecules, expected):
        raise RuntimeError("atom residual rows are not grouped in molecule order")
    width = int(counts.max())
    local = np.repeat(np.arange(len(molecule_indices)), counts)
    slot = np.concatenate([np.arange(count) for count in counts])
    valid = np.zeros((len(molecule_indices), width), dtype=bool)
    valid[local, slot] = True
    return {"local": local, "slot": slot, "valid": valid,
            "n_rows": int(len(row_molecules)), "width": width}


def _pack(values: np.ndarray, packing: dict) -> np.ndarray:
    values = np.asarray(values)
    if len(values) != packing["n_rows"]:
        raise ValueError("value rows do not match compact molecule packing")
    out = np.zeros(packing["valid"].shape + values.shape[1:], dtype=values.dtype)
    out[packing["local"], packing["slot"]] = values
    return out


def _derived_seed(seed: int, fragment: str, split: str) -> int:
    payload = f"{int(seed)}\0{fragment}\0{split}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _release_device_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


#: Nulls the fragment screen is quoted against, weakest first. The uniform
#: permutation preserves only coverage; every basis clears it on every fragment,
#: which means it cannot separate bases and should not be read as evidence that
#: one detects the fragment. The other two hold more of the structure fixed, so
#: a coordinate has to know which atoms form the substructure rather than that a
#: contiguous run of atoms of the right kinds is present somewhere.
NULL_FAMILIES = ("permutation", "shift", "type_matched")

#: Recorded in the artifact so a reader can tell which null a number was quoted
#: against without consulting the code.
NULL_METHODS = {
    "permutation": "within-molecule whole-pipeline permutation "
                   "(preserves coverage only)",
    "shift": "within-molecule cyclic shift of the fragment footprint "
             "(preserves coverage and the pattern of runs and gaps)",
    "type_matched": "within-molecule resample matched on the token-identity "
                    "multiset (preserves coverage and atom-type composition)",
}

#: The type-matched resample keeps each molecule's in-fragment count per token
#: type, so for the within-type estimand it is the exact stratified permutation
#: null: the same draws as the primary type-matched null, rescored.
WITHIN_TYPE_NULL_METHOD = ("within-molecule within-token-type permutation "
                           "(the type-matched resample, scored within type)")

WITHIN_TYPE_ESTIMAND = {
    "name": "selection-oriented held-out within-token-type WSD",
    "formula": ("sum_i c_i*x_i / (sigma_all*sqrt(n*sum_i c_i^2)), "
                "c_i = y_i - n_in,T(i)/n_T(i)"),
    "interpretation": (
        "semipartial correlation of the coordinate with fragment membership "
        "after removing token-type means from the label; with one token type "
        "in the molecule it equals the primary WSD"),
    "token_type": "current-token identity, the code atom_type_onehot encodes",
    "combination_rule": (
        "per-type in-minus-out mean differences pooled with the "
        "Cochran-Mantel-Haenszel weight n_in,T*n_out,T/n_T; coverage factor "
        "sqrt(sum_T n_in,T*n_out,T/n_T / n) replaces sqrt(p*(1-p)); sigma_all "
        "is the primary estimand's whole-molecule SD"),
    "population": "atom tokens within each molecule",
    "eligible_molecules": (
        "molecules with at least one token type present both inside and "
        "outside the match; equal weight"),
    "fragment_eligibility": (
        "eligible molecules >= min_molecules_per_half in both halves; "
        "others listed in skipped_fragments_within_type"),
    "unit_of_split": "molecule",
    "selection_score": "abs(mean_molecule(signed within-type WSD))",
    "selection_rule": "argmax coordinate on selection molecules only",
    "orientation_rule": "sign(mean_molecule(signed within-type WSD)) on selection",
    "evaluation_score": (
        "selection orientation times mean signed within-type WSD at the same "
        "coordinate"),
    "evaluation_reselection": False,
    "null": ("whole pipeline; type-matched resample (pipeline_null_type_matched)"
             ", requires the type_matched null family"),
    "atom_type_onehot": (
        "exactly zero by construction: every one-hot column is constant within "
        "token type; its detector is flagged selection_signal_identically_zero"),
    "per_fragment_location": "fragments.<name>.arms.<arm>.within_type",
    "aggregate_location": "aggregate.within_type_oriented_wsd",
}


def _null_masks(family: str, mask, valid, token_ids, n_perm: int, seed: int):
    if family == "permutation":
        return permute_fragment_masks_within_molecules(mask, valid, n_perm, seed)
    if family == "shift":
        return shift_fragment_masks_within_molecules(mask, valid, n_perm, seed)
    if family == "type_matched":
        return type_matched_fragment_masks(mask, valid, token_ids, n_perm, seed)[0]
    raise ValueError(f"unknown null family {family!r}")


def _score_arm(name: str, build, fragment_rows: dict, fragment_masks: dict,
               selection_rows: np.ndarray, evaluation_rows: np.ndarray,
               selection_packing: dict, evaluation_packing: dict,
               n_perm: int, molecule_chunk: int, permutation_batch: int,
               device: str, token_ids: dict | None = None,
               null_families: tuple = ("permutation",)) -> None:
    """Select on one molecule half, then score fixed detectors on the other."""
    selection_state = {}
    print(f"\n[{name}] constructing selection representation", flush=True)
    rows = build(selection_rows)
    packed = _pack(rows, selection_packing)
    del rows
    basis = to_basis(packed, device)
    del packed
    stats = basis_stats(basis, selection_packing["valid"], molecule_chunk)
    print(f"  selection basis {tuple(basis.shape)}", flush=True)
    for fragment, row in fragment_rows.items():
        mask = fragment_masks[fragment]["selection"]
        signed, absolute = mean_wsd_coordinates(
            basis, mask, selection_packing["valid"], stats,
            chunk=molecule_chunk, mask_batch=1)
        selection_seed = row["permutation_seeds"]["selection"]
        nulls = {}
        within_type = _within_type_scored(row, token_ids)
        for family in null_families:
            null_masks = _null_masks(
                family, mask, selection_packing["valid"],
                None if token_ids is None else token_ids["selection"],
                n_perm, selection_seed)
            nulls[family] = mean_wsd_coordinates(
                basis, null_masks, selection_packing["valid"], stats,
                chunk=molecule_chunk, mask_batch=permutation_batch)
            if within_type and family == "type_matched":
                nulls["within_type"] = within_type_wsd_coordinates(
                    basis, null_masks, selection_packing["valid"],
                    token_ids["selection"], stats, chunk=molecule_chunk,
                    mask_batch=permutation_batch)[:2]
        selection_state[fragment] = {
            "signed": signed[0], "absolute": absolute[0], "nulls": nulls,
        }
        if within_type:
            wt_signed, wt_absolute, wt_n = within_type_wsd_coordinates(
                basis, mask, selection_packing["valid"],
                token_ids["selection"], stats, chunk=molecule_chunk,
                mask_batch=1)
            selection_state[fragment]["within_type"] = (
                wt_signed[0], wt_absolute[0], int(wt_n[0]))
        print(f"  select {fragment:<22}", flush=True)
    del basis, stats
    _release_device_memory(device)

    print(f"[{name}] constructing evaluation representation", flush=True)
    rows = build(evaluation_rows)
    packed = _pack(rows, evaluation_packing)
    del rows
    basis = to_basis(packed, device)
    del packed
    stats = basis_stats(basis, evaluation_packing["valid"], molecule_chunk)
    print(f"  evaluation basis {tuple(basis.shape)}", flush=True)
    for fragment, row in fragment_rows.items():
        mask = fragment_masks[fragment]["evaluation"]
        signed, absolute = mean_wsd_coordinates(
            basis, mask, evaluation_packing["valid"], stats,
            chunk=molecule_chunk, mask_batch=1)
        selected = selection_state[fragment]
        result = heldout_wsd_detectors(
            selected["signed"], selected["absolute"], signed[0], absolute[0])

        within_type = None
        if "within_type" in selected:
            wt_signed, wt_absolute, wt_n = within_type_wsd_coordinates(
                basis, mask, evaluation_packing["valid"],
                token_ids["evaluation"], stats, chunk=molecule_chunk,
                mask_batch=1)
            sel_wt_signed, sel_wt_absolute, sel_wt_n = selected["within_type"]
            eligibility = row["within_type_eligibility"]
            # The scorer and the eligibility rule that admitted the fragment
            # are separate code paths; they must agree on who is eligible.
            if (sel_wt_n, int(wt_n[0])) != (
                    eligibility["n_selection_molecules_eligible"],
                    eligibility["n_evaluation_molecules_eligible"]):
                raise RuntimeError(
                    f"{fragment}: within-type eligible counts disagree")
            within_type = heldout_within_type_detectors(
                sel_wt_signed, sel_wt_absolute, wt_signed[0], wt_absolute[0],
                sel_wt_n, int(wt_n[0]))

        evaluation_seed = row["permutation_seeds"]["evaluation"]
        for family in null_families:
            null_masks = _null_masks(
                family, mask, evaluation_packing["valid"],
                None if token_ids is None else token_ids["evaluation"],
                n_perm, evaluation_seed)
            null_signed, null_absolute = mean_wsd_coordinates(
                basis, null_masks, evaluation_packing["valid"], stats,
                chunk=molecule_chunk, mask_batch=permutation_batch)
            if within_type is not None and family == "type_matched":
                wt_null_signed, wt_null_absolute, _ = (
                    within_type_wsd_coordinates(
                        basis, null_masks, evaluation_packing["valid"],
                        token_ids["evaluation"], stats, chunk=molecule_chunk,
                        mask_batch=permutation_batch))
                sel_null_signed, sel_null_absolute = selected["nulls"][
                    "within_type"]
                wt_null = heldout_wsd_pipeline_null(
                    sel_null_signed, sel_null_absolute,
                    wt_null_signed, wt_null_absolute, within_type,
                    row["permutation_seeds"]["selection"], evaluation_seed,
                    method=WITHIN_TYPE_NULL_METHOD)
                within_type["oriented_detector"][
                    "pipeline_null_type_matched"] = wt_null["oriented_detector"]
                within_type["absolute_detector"][
                    "pipeline_null_type_matched"] = wt_null["absolute_detector"]
            sel_signed, sel_absolute = selected["nulls"][family]
            null = heldout_wsd_pipeline_null(
                sel_signed, sel_absolute, null_signed, null_absolute, result,
                row["permutation_seeds"]["selection"], evaluation_seed,
                method=NULL_METHODS[family])
            key = ("pipeline_null" if family == "permutation"
                   else f"pipeline_null_{family}")
            result["oriented_detector"][key] = null["oriented_detector"]
            result["absolute_detector"][key] = null["absolute_detector"]
        result["n_selection_molecules_matched"] = row[
            "n_selection_molecules_matched"]
        result["n_evaluation_molecules_matched"] = row[
            "n_evaluation_molecules_matched"]
        if within_type is not None:
            result["within_type"] = within_type
        row["arms"][name] = result
        primary = result["oriented_detector"].get(
            "evaluation_oriented_wsd", float("nan"))
        secondary = result["absolute_detector"].get(
            "evaluation_mean_absolute_wsd", float("nan"))
        coordinate = result["oriented_detector"].get("selected_coordinate")
        within = (within_type["oriented_detector"].get(
            "evaluation_oriented_wsd", float("nan"))
            if within_type is not None else float("nan"))
        print(f"  eval   {fragment:<22} coordinate={str(coordinate):>5} "
              f"oriented={primary:.3f} absolute={secondary:.3f} "
              f"within_type={within:.3f}", flush=True)
    del basis, stats
    _release_device_memory(device)
    del selection_state
    gc.collect()


def _within_type_scored(row: dict, token_ids: dict | None) -> bool:
    return bool(token_ids is not None and row.get(
        "within_type_eligibility", {}).get("scored", False))


def _arm_mean(fragment_rows: dict, arm: str, detector: str,
              score: str, estimand_key: str | None = None) -> tuple[int, float]:
    values = []
    for row in fragment_rows.values():
        result = row["arms"].get(arm, {})
        if estimand_key is not None:
            result = result.get(estimand_key, {})
        if result.get("status") == "ok":
            values.append(result[detector][score])
    return len(values), float(np.mean(values)) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sae", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--tokenizer", default=None,
                        help="default: <data>/tokenizer.json")
    parser.add_argument("--model-config", default=None,
                        help="legacy checkpoint geometry; embedded model_cfg wins")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--smarts", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-molecules", type=int, default=4000)
    parser.add_argument("--n-ctx", type=int, default=128)
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--selection-split-seed", type=int, default=None)
    parser.add_argument("--projection-seed", type=int, default=None)
    parser.add_argument("--permutation-seed", type=int, default=None)
    parser.add_argument("--analysis-seed", type=int, default=None,
                        help="Seeds everything that defines the SCREEN rather "
                             "than the model: which molecules are sampled, the "
                             "selection/evaluation split, the random "
                             "projection, the nulls and the bootstrap. Defaults "
                             "to --seed, which reproduces the historical "
                             "behaviour. Set it to a constant across "
                             "checkpoints to make the reported range a range "
                             "over training initializations rather than over "
                             "training initializations AND disjoint molecule "
                             "samples: at 4,000 molecules drawn per seed the "
                             "three historical screens overlapped by 3, 11 and "
                             "2 molecules, so they were not a common screen.")
    parser.add_argument("--null-families", nargs="+", default=list(NULL_FAMILIES),
                        choices=list(NULL_FAMILIES),
                        help="nulls each detector is quoted against. The uniform "
                             "permutation alone is too easy: it preserves "
                             "coverage and nothing else, and every basis clears "
                             "it on every fragment, so it cannot separate bases.")
    parser.add_argument("--n-perm", type=int, default=50)
    parser.add_argument("--permutation-batch", type=int, default=10)
    parser.add_argument("--molecule-chunk", type=int, default=256)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--min-molecules-per-half", type=int, default=15)
    parser.add_argument("--model-batch", type=int, default=256)
    parser.add_argument("--projection-batch", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.n_perm < 1:
        raise ValueError("--n-perm must be positive for the pipeline null")
    if args.permutation_batch < 1 or args.molecule_chunk < 1:
        raise ValueError("--permutation-batch and --molecule-chunk must be positive")
    if args.analysis_seed is None:
        args.analysis_seed = args.seed
    if args.selection_split_seed is None:
        args.selection_split_seed = 17000 + args.analysis_seed
    if args.projection_seed is None:
        args.projection_seed = 27000 + args.analysis_seed
    if args.bootstrap_seed is None:
        args.bootstrap_seed = 37000 + args.analysis_seed
    if args.permutation_seed is None:
        args.permutation_seed = 47000 + args.analysis_seed
    tokenizer_path = args.tokenizer or os.path.join(args.data, "tokenizer.json")

    torch.manual_seed(args.seed)
    model, tokenizer, sae, model_cfg, stack_metadata = load_stack(
        args.ckpt, args.sae, device=args.device,
        tokenizer_path=tokenizer_path, model_config_path=args.model_config,
        return_metadata=True)
    if args.n_ctx > model_cfg.n_ctx:
        raise ValueError(
            f"--n-ctx {args.n_ctx} exceeds checkpoint n_ctx {model_cfg.n_ctx}")
    d_model = int(model_cfg.d_model)
    d_sae, k_sae = int(sae.cfg.d_sae), int(sae.cfg.k)
    if sae.cfg.variant != "topk" or sae.cfg.d_in != d_model:
        raise ValueError("matched controls require a TopK SAE fitted to this model width")
    matched_arms = matched_arm_names(d_model, d_sae, k_sae)
    sae_arm, dense_arm, dense_topk_arm, random_arm = matched_arms
    wsd_arms = (*matched_arms, "atom_type_onehot")

    pool_path = os.path.join(args.data, f"{args.split}.smi")
    pool = open(pool_path).read().split()
    smiles, source_indices, dropped = _sample_encodable(
        pool, tokenizer, args.n_molecules, args.n_ctx, args.analysis_seed)
    if len(smiles) < 2:
        raise ValueError("fewer than two encodable molecules remain")
    selection_indices, evaluation_indices = deterministic_molecule_split(
        len(smiles), args.selection_fraction, args.selection_split_seed)
    molecule_is_selection = np.zeros(len(smiles), dtype=bool)
    molecule_is_selection[selection_indices] = True
    print(f"{len(smiles):,} encodable molecules: {len(selection_indices):,} "
          f"selection / {len(evaluation_indices):,} evaluation; "
          f"{len(dropped)} dropped")

    library = load_smarts_library(args.smarts)
    masks = build_masks(smiles, library, args.n_ctx, offset=1)
    population = atom_token_mask(smiles, args.n_ctx, offset=1)
    cov = coverage(masks, population)
    dense, molecule_index, token_index = _collect_atom_residuals(
        model, tokenizer, smiles, population, sae.cfg.layer, args.n_ctx,
        args.device, args.model_batch)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    selection_rows = molecule_is_selection[molecule_index]
    evaluation_rows = ~selection_rows
    selection_packing = _packing(
        molecule_index, selection_rows, selection_indices)
    evaluation_packing = _packing(
        molecule_index, evaluation_rows, evaluation_indices)
    mean, scale = fit_selection_standardizer(dense[selection_rows])
    mean_t = torch.as_tensor(mean, device=args.device)
    scale_t = torch.as_tensor(scale, device=args.device)

    # Integer token codes in the same compact layout as the masks, so the
    # atom-type-matched null can resample a fragment's composition. Code 0 is
    # reserved for positions _pack zero-fills, which are invalid anyway.
    token_strings = token_identity_array(smiles, args.n_ctx, offset=1)
    vocabulary = {t: i + 1 for i, t in
                  enumerate(sorted({t for t in token_strings.ravel() if t}))}
    token_codes = np.vectorize(lambda t: vocabulary.get(t, 0),
                               otypes=[np.int32])(token_strings)
    flat_codes = token_codes[molecule_index, token_index]
    token_id_packs = {
        "selection": _pack(flat_codes[selection_rows], selection_packing),
        "evaluation": _pack(flat_codes[evaluation_rows], evaluation_packing),
    }
    n_token_types = max(vocabulary.values()) + 1
    del token_strings

    fragment_rows, fragment_masks, skipped = {}, {}, {}
    within_type_eligibility, skipped_within_type = {}, {}
    for name, mask in masks.items():
        atom_mask = mask & population
        molecule_match = atom_mask.any(axis=1)
        n_selection_match = int(molecule_match[selection_indices].sum())
        n_evaluation_match = int(molecule_match[evaluation_indices].sum())
        eligible = within_type_eligible_molecules(
            atom_mask, population, token_codes)
        n_selection_eligible = int(eligible[selection_indices].sum())
        n_evaluation_eligible = int(eligible[evaluation_indices].sum())
        # Eligible molecules are a subset of matched ones, so a fragment the
        # primary estimand skips is always skipped here too.
        within_type_scored = (min(n_selection_eligible, n_evaluation_eligible)
                              >= args.min_molecules_per_half)
        within_type_eligibility[name] = {
            "n_selection_molecules_eligible": n_selection_eligible,
            "n_evaluation_molecules_eligible": n_evaluation_eligible,
            "min_molecules_per_half": args.min_molecules_per_half,
            "scored": within_type_scored,
        }
        if not within_type_scored:
            skipped_within_type[name] = {
                "reason": "below min_molecules_per_half",
                "n_selection_molecules_eligible": n_selection_eligible,
                "n_evaluation_molecules_eligible": n_evaluation_eligible,
                "n_selection_molecules_matched": n_selection_match,
                "n_evaluation_molecules_matched": n_evaluation_match,
                "min_molecules_per_half": args.min_molecules_per_half,
            }
        if min(n_selection_match, n_evaluation_match) < args.min_molecules_per_half:
            skipped[name] = {
                "reason": "below min_molecules_per_half",
                "n_selection_molecules_matched": n_selection_match,
                "n_evaluation_molecules_matched": n_evaluation_match,
                "min_molecules_per_half": args.min_molecules_per_half,
            }
            continue
        token_labels = atom_mask[molecule_index, token_index]
        fragment_masks[name] = {
            "selection": _pack(token_labels[selection_rows], selection_packing),
            "evaluation": _pack(token_labels[evaluation_rows], evaluation_packing),
        }
        fragment_rows[name] = {
            "coverage_among_atom_tokens": cov[name],
            "n_selection_molecules_matched": n_selection_match,
            "n_evaluation_molecules_matched": n_evaluation_match,
            "permutation_seeds": {
                "selection": _derived_seed(
                    args.permutation_seed, name, "selection"),
                "evaluation": _derived_seed(
                    args.permutation_seed, name, "evaluation"),
            },
            "within_type_eligibility": within_type_eligibility[name],
            "arms": {},
        }
    del token_codes
    if not fragment_rows:
        raise ValueError("no fragment meets --min-molecules-per-half")
    n_within_type = sum(row["within_type_eligibility"]["scored"]
                        for row in fragment_rows.values())
    print(f"{len(fragment_rows)} eligible fragments; {len(skipped)} skipped; "
          f"{n_within_type} scored within token type; "
          f"{len(dense):,} atom-token residuals")
    if n_within_type and "type_matched" not in args.null_families:
        print("  note: no type_matched null, so the within-type estimand is "
              "reported without a null", flush=True)

    os.makedirs(args.out, exist_ok=True)
    smarts_path = os.path.join(args.out, "smarts_library.json")
    save_smarts_library(library, smarts_path)
    artifact_path = os.path.join(args.out, ARTIFACT_NAME)
    sae_path = os.path.join(args.sae, "sae.pt")
    artifact = {
        "schema_version": WSD_SCHEMA_VERSION,
        "analysis": "heldout_fragment_alignment_within_molecule_wsd_controls",
        "status": "partial",
        "config": vars(args),
        "model_inputs": stack_metadata,
        "dictionary": {"d_model": d_model, "d_sae": d_sae, "k": k_sae},
        "input_sha256": {
            "checkpoint": file_sha256(args.ckpt),
            "sae": file_sha256(sae_path),
            "tokenizer": file_sha256(tokenizer_path),
            "corpus_split": file_sha256(pool_path),
        },
        "primary_estimand": {
            "name": "selection-oriented held-out within-molecule WSD",
            "formula": "((mu_in-mu_out)/sigma_all)*sqrt(p*(1-p))",
            "population": "atom tokens within each molecule",
            "eligible_molecules": "fragment-containing molecules; equal weight",
            "unit_of_split": "molecule",
            "selection_score": "abs(mean_molecule(signed_WSD))",
            "selection_rule": "argmax coordinate on selection molecules only",
            "orientation_rule": "sign(mean_molecule(signed_WSD)) on selection",
            "evaluation_score": (
                "selection orientation times mean signed WSD at the same coordinate"),
            "evaluation_reselection": False,
            "null": "whole pipeline; labels shuffled within molecule preserving coverage",
        },
        "within_type_estimand": WITHIN_TYPE_ESTIMAND,
        "secondary_estimand": {
            "name": "fixed-coordinate held-out mean absolute within-molecule WSD",
            "selection_score": "mean_molecule(abs(WSD))",
            "selection_rule": "argmax coordinate on selection molecules only",
            "evaluation_score": "mean absolute WSD at the same coordinate",
            "evaluation_reselection": False,
        },
        "distinction_from_schema_v1": (
            "v1 pools atom tokens across molecules and can measure molecule-level "
            "fragment context; v2 conditions contrasts within matched molecules"),
        "representations": {
            sae_arm: {
                "coordinates": d_sae, "nonlinearity": f"ReLU then TopK{k_sae}",
                "directions": "learned SAE encoder", "standardized_input": False,
            },
            "atom_type_onehot": {
                "nonlinearity": "identity",
                "directions": "one-hot current-token identity",
                "note": ("contains no model activations; upper-bounds what a "
                         "pure atom-type detector scores on this screen"),
            },
            dense_arm: {
                "coordinates": d_model, "nonlinearity": "identity",
                "directions": "residual-stream neuron basis",
            },
            dense_topk_arm: {
                "coordinates": d_model, "nonlinearity": f"ReLU then TopK{k_sae}",
                "directions": "residual-stream neuron basis",
                "standardizer_fit": "selection-molecule atom tokens only",
            },
            random_arm: {
                "coordinates": d_sae, "nonlinearity": f"ReLU then TopK{k_sae}",
                "directions": "Gaussian columns normalized to unit L2 norm",
                "projection_seed": args.projection_seed,
                "standardizer_fit": "selection-molecule atom tokens only",
            },
        },
        "standardizer": {"mean": mean.tolist(), "scale": scale.tolist(),
                         "fit_rows": int(selection_rows.sum())},
        "data": {
            "pool_path": os.path.realpath(pool_path),
            "n_pool": len(pool), "n_sampled_encodable": len(smiles),
            "sampled_source_indices": source_indices,
            "dropped_sampled_rows": dropped,
            "selection_molecule_indices": selection_indices.tolist(),
            "evaluation_molecule_indices": evaluation_indices.tolist(),
            "n_selection_atom_tokens": int(selection_rows.sum()),
            "n_evaluation_atom_tokens": int(evaluation_rows.sum()),
        },
        "fragments": fragment_rows,
        "skipped_fragments": skipped,
        "skipped_fragments_within_type": skipped_within_type,
        "completed_arms": [],
    }
    dump_json(artifact, artifact_path)

    def raw_dense(mask):
        return np.ascontiguousarray(dense[mask])

    def atom_type_onehot(mask):
        """Current-token identity as a basis, with no model in it at all."""
        codes = flat_codes[mask]
        out = np.zeros((len(codes), n_token_types), dtype=np.float32)
        out[np.arange(len(codes)), codes] = 1.0
        return out

    def dense_topk(mask):
        return _transform_rows(
            dense, mask, d_model,
            lambda x: relu_topk((x - mean_t) / scale_t, k_sae),
            args.device, args.projection_batch)

    def learned_sae(mask):
        return _transform_rows(
            dense, mask, d_sae, sae.encode, args.device, args.projection_batch)

    builders = {
        "atom_type_onehot": atom_type_onehot,
        dense_arm: raw_dense,
        dense_topk_arm: dense_topk,
        sae_arm: learned_sae,
    }
    execution_order = (
        "atom_type_onehot", dense_arm,
        dense_topk_arm, sae_arm)
    for arm in execution_order:
        _score_arm(
            arm, builders[arm], fragment_rows, fragment_masks,
            selection_rows, evaluation_rows,
            selection_packing, evaluation_packing,
            args.n_perm, args.molecule_chunk, args.permutation_batch,
            args.device, token_ids=token_id_packs,
            null_families=tuple(args.null_families))
        artifact["completed_arms"].append(arm)
        dump_json(artifact, artifact_path)

    sae.to("cpu")
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    projection = random_unit_projection(
        d_model, d_sae, args.projection_seed, device=args.device)
    projection_hash = hashlib.sha256(
        projection.detach().cpu().numpy().tobytes()).hexdigest()
    artifact["representations"][random_arm][
        "projection_float32_sha256"] = projection_hash

    def random_topk(mask):
        return _transform_rows(
            dense, mask, d_sae,
            lambda x: relu_topk(((x - mean_t) / scale_t) @ projection, k_sae),
            args.device, args.projection_batch)

    _score_arm(
        random_arm, random_topk,
        fragment_rows, fragment_masks, selection_rows, evaluation_rows,
        selection_packing, evaluation_packing,
        args.n_perm, args.molecule_chunk, args.permutation_batch, args.device,
        token_ids=token_id_packs, null_families=tuple(args.null_families))
    artifact["completed_arms"].append(random_arm)
    artifact["aggregate"] = aggregate_wsd_fragment_results(
        fragment_rows, arms=wsd_arms, reference=sae_arm,
        n_bootstrap=args.n_bootstrap, bootstrap_seed=args.bootstrap_seed)
    artifact["status"] = "complete"
    dump_json(artifact, artifact_path)

    Manifest().record(
        "fragment_alignment_within_molecule_wsd_controls_v2",
        {"schema_version": WSD_SCHEMA_VERSION,
         "n_fragments": len(fragment_rows),
         "aggregate": artifact["aggregate"]},
        config=vars(args), files=[artifact_path, smarts_path])
    print(f"\nwrote {artifact_path}")
    for arm in (sae_arm, dense_arm,
                dense_topk_arm, random_arm, "atom_type_onehot"):
        n, primary = _arm_mean(
            fragment_rows, arm, "oriented_detector", "evaluation_oriented_wsd")
        _, secondary = _arm_mean(
            fragment_rows, arm, "absolute_detector",
            "evaluation_mean_absolute_wsd")
        n_within, within = _arm_mean(
            fragment_rows, arm, "oriented_detector", "evaluation_oriented_wsd",
            estimand_key="within_type")
        sharing = artifact["aggregate"]["within_type_oriented_wsd"][
            "detector_sharing"][arm]["n_distinct_coordinates"]
        print(f"  {arm:<38} oriented={primary:.3f} absolute={secondary:.3f} "
              f"over {n} fragments; within_type={within:.3f} over {n_within} "
              f"({sharing} distinct coordinates)")


if __name__ == "__main__":
    main()
