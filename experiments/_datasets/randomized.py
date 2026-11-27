#!/usr/bin/env python3
"""Build a randomized-SMILES mirror of an existing canonical SMILES dataset.

Preserves the exact Murcko scaffold split, row ordering, and 64-token vocabulary
of the source dataset so canonical vs. randomized pretraining is a clean
controlled intervention.

Usage:
    python experiments/01_build_datasets.py randomized \
        --src runs/data \
        --out runs/data_rand
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger

from molcircuits.chem.tokenizer import SmilesTokenizer, tokenize
from molcircuits.utils.manifest import dump_json

SPLITS = ("train", "valid", "test")
SPLIT_IDS = {"train": 1, "valid": 2, "test": 3}

_VOCAB: dict[str, int] | None = None
_VOCAB_SET: set[str] | None = None
_BOS_ID: int = 1
_EOS_ID: int = 2
_UNK_ID: int = 3
_MAX_RETRIES: int = 5
_BASE_SEED: int = 2026


def _init_worker(vocab: dict[str, int], max_retries: int, base_seed: int) -> None:
    global _VOCAB, _VOCAB_SET, _BOS_ID, _EOS_ID, _UNK_ID, _MAX_RETRIES, _BASE_SEED
    RDLogger.DisableLog("rdApp.*")
    tok = SmilesTokenizer(vocab=vocab)
    _VOCAB = vocab
    _VOCAB_SET = set(vocab.keys())
    _BOS_ID = tok.bos_id
    _EOS_ID = tok.eos_id
    _UNK_ID = tok.unk_id
    _MAX_RETRIES = max_retries
    _BASE_SEED = base_seed


def _mol_seed(split_id: int, idx: int) -> int:
    # Deterministic non-zero 31-bit integer seed (RDKit treats randomSeed=0 as unseeded).
    s = (_BASE_SEED * 1_000_003 + split_id * 100_000_007 + idx * 2_654_435_761) & 0x7FFFFFFF
    return s if s != 0 else 1


def _try_encode(smi: str) -> list[int] | None:
    assert _VOCAB is not None and _VOCAB_SET is not None
    try:
        toks = tokenize(smi)
    except ValueError:
        return None
    for t in toks:
        if t not in _VOCAB_SET:
            return None
    ids = [_BOS_ID]
    vocab = _VOCAB
    ids.extend(vocab[t] for t in toks)
    ids.append(_EOS_ID)
    return ids


def _encode_fallback(smi: str) -> list[int]:
    assert _VOCAB is not None
    vocab = _VOCAB
    unk = _UNK_ID
    toks = tokenize(smi)
    ids = [_BOS_ID]
    ids.extend(vocab.get(t, unk) for t in toks)
    ids.append(_EOS_ID)
    return ids


def _process_subchunk(task: tuple[int, int, list[str]]) -> tuple[list[str], np.ndarray, dict[str, int]]:
    split_id, start_idx, smis = task
    out_smis: list[str] = []
    buf: list[int] = []

    n_total = len(smis)
    n_first_ok = 0
    n_retry_ok = 0
    n_fallback = 0
    n_parse_fail = 0
    n_changed = 0
    n_encode_skip = 0

    for offset, orig in enumerate(smis):
        idx = start_idx + offset
        mol = Chem.MolFromSmiles(orig)
        chosen: str | None = None
        chosen_ids: list[int] | None = None

        if mol is None:
            n_parse_fail += 1
        else:
            seed = _mol_seed(split_id, idx)
            cand = Chem.MolToRandomSmilesVect(mol, 1, randomSeed=seed, isomericSmiles=True)[0]
            ids = _try_encode(cand)
            if ids is not None:
                chosen = cand
                chosen_ids = ids
                n_first_ok += 1
            elif _MAX_RETRIES > 0:
                retry_seed = (seed ^ 0x5A5A5A5A) & 0x7FFFFFFF
                if retry_seed == 0:
                    retry_seed = 1
                cands = Chem.MolToRandomSmilesVect(
                    mol, _MAX_RETRIES, randomSeed=retry_seed, isomericSmiles=True
                )
                for cand2 in cands:
                    ids2 = _try_encode(cand2)
                    if ids2 is not None:
                        chosen = cand2
                        chosen_ids = ids2
                        n_retry_ok += 1
                        break

        if chosen is None or chosen_ids is None:
            chosen = orig
            n_fallback += 1
            try:
                chosen_ids = _encode_fallback(orig)
            except ValueError:
                n_encode_skip += 1
                chosen_ids = None

        if chosen != orig:
            n_changed += 1
        out_smis.append(chosen)
        if chosen_ids is not None:
            buf.extend(chosen_ids)

    arr = np.asarray(buf, dtype=np.uint16)
    stats = {
        "n_molecules": n_total,
        "n_first_ok": n_first_ok,
        "n_retry_ok": n_retry_ok,
        "n_fallback": n_fallback,
        "n_parse_fail": n_parse_fail,
        "n_changed": n_changed,
        "n_encode_skip": n_encode_skip,
        "n_tokens": int(arr.shape[0]),
    }
    return out_smis, arr, stats


def _iter_line_chunks(path: str, chunk_size: int, limit: int | None):
    buf: list[str] = []
    start_idx = 0
    emitted = 0
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            s = s.split()[0]
            buf.append(s)
            emitted += 1
            if len(buf) >= chunk_size:
                yield start_idx, buf
                start_idx += len(buf)
                buf = []
            if limit is not None and emitted >= limit:
                break
    if buf:
        yield start_idx, buf


def main() -> None:
    from molcircuits import paths as _paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=_paths.data())
    ap.add_argument("--out", default=os.path.join(_paths.root(), "data_rand"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument(
        "--workers",
        type=int,
        default=min(64, max(1, (os.cpu_count() or 8) - 4)),
    )
    ap.add_argument("--chunk", type=int, default=400_000,
                    help="Streaming outer chunk size (molecules)")
    ap.add_argument("--subchunk", type=int, default=10_000,
                    help="Per-worker task granularity (molecules)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional cap on molecules per split (for smoke testing)")
    ap.add_argument("--no-copy-canon-eval", action="store_true",
                    help="Skip copying canonical valid/test packed arrays")
    args = ap.parse_args()

    t_all = time.time()
    os.makedirs(args.out, exist_ok=True)

    # Copy tokenizer.json and split_manifest.json verbatim
    for fname in ("tokenizer.json", "split_manifest.json"):
        src_p = os.path.join(args.src, fname)
        dst_p = os.path.join(args.out, fname)
        shutil.copy2(src_p, dst_p)

    tok = SmilesTokenizer.load(os.path.join(args.out, "tokenizer.json"))
    print(f"Loaded tokenizer: vocab_size={tok.vocab_size}, workers={args.workers}", flush=True)

    molecule_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    rand_stats: dict[str, dict[str, float | int]] = {}

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(tok.vocab, args.max_retries, args.seed),
    ) as pool:
        for split in SPLITS:
            t0 = time.time()
            src_smi = os.path.join(args.src, f"{split}.smi")
            dst_smi = os.path.join(args.out, f"{split}.smi")
            dst_npy = os.path.join(args.out, f"{split}_packed.npy")
            split_id = SPLIT_IDS[split]

            agg = {
                "n_molecules": 0,
                "n_first_ok": 0,
                "n_retry_ok": 0,
                "n_fallback": 0,
                "n_parse_fail": 0,
                "n_changed": 0,
                "n_encode_skip": 0,
                "n_tokens": 0,
            }
            npy_parts: list[np.ndarray] = []

            with open(dst_smi, "w") as out_fh:
                for outer_start, outer_smis in _iter_line_chunks(src_smi, args.chunk, args.limit):
                    tasks = [
                        (split_id, outer_start + i, outer_smis[i : i + args.subchunk])
                        for i in range(0, len(outer_smis), args.subchunk)
                    ]
                    for out_smis, arr, st in pool.imap(_process_subchunk, tasks):
                        out_fh.write("\n".join(out_smis))
                        out_fh.write("\n")
                        npy_parts.append(arr)
                        for k, v in st.items():
                            agg[k] += v

                    elapsed = time.time() - t0
                    rate = agg["n_molecules"] / max(elapsed, 1e-6)
                    print(
                        f"[{split}] {agg['n_molecules']:,} mols | "
                        f"{rate:,.0f} mol/s | fallback={agg['n_fallback']:,} | "
                        f"changed={agg['n_changed']/max(agg['n_molecules'],1):.4f}",
                        flush=True,
                    )

            full_arr = np.concatenate(npy_parts) if npy_parts else np.zeros(0, dtype=np.uint16)
            del npy_parts
            np.save(dst_npy, full_arr)
            del full_arr

            n_mol = agg["n_molecules"]
            n_tok = agg["n_tokens"]
            dt = time.time() - t0
            molecule_counts[split] = n_mol
            token_counts[split] = n_tok
            rand_stats[split] = {
                **agg,
                "randomized_rate": (agg["n_first_ok"] + agg["n_retry_ok"]) / max(n_mol, 1),
                "retry_rate": agg["n_retry_ok"] / max(n_mol, 1),
                "fallback_rate": agg["n_fallback"] / max(n_mol, 1),
                "changed_rate": agg["n_changed"] / max(n_mol, 1),
                "mean_tokens_per_molecule": n_tok / max(n_mol, 1),
                "elapsed_sec": round(dt, 2),
            }
            print(
                f"Finished {split}: {n_mol:,} mols, {n_tok:,} tokens "
                f"(mean {n_tok/max(n_mol,1):.3f} tok/mol, "
                f"rand={rand_stats[split]['randomized_rate']:.6f}, "
                f"fallback={agg['n_fallback']:,}, {dt:.1f}s)",
                flush=True,
            )

    if not args.no_copy_canon_eval and args.limit is None:
        for split in ("valid", "test"):
            src_c = os.path.join(args.src, f"{split}_packed.npy")
            dst_c = os.path.join(args.out, f"{split}_canon_packed.npy")
            if os.path.exists(src_c):
                shutil.copy2(src_c, dst_c)
                print(f"Copied canonical eval array -> {dst_c}", flush=True)

    n_kept = sum(molecule_counts.values())
    payload = {
        "n_input": n_kept,
        "n_kept": n_kept,
        "molecule_counts": molecule_counts,
        "token_counts": token_counts,
        "vocab_size": tok.vocab_size,
        "filtered": True,
        "filter_source": "upstream (z20ll) + RDKit randomized SMILES",
        "source_dir": args.src,
        "seed": args.seed,
        "isomeric_smiles": True,
        "max_retries": args.max_retries,
        "limit_per_split": args.limit,
        "mean_tokens_per_molecule": sum(token_counts.values()) / max(n_kept, 1),
        "randomization": rand_stats,
        "total_elapsed_sec": round(time.time() - t_all, 2),
    }
    dump_json(payload, os.path.join(args.out, "dataset_stats.json"))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
