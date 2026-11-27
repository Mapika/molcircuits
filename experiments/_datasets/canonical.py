#!/usr/bin/env python3
"""Filter a SMILES corpus, assign deterministic scaffold splits, build a tokenizer
and pack each split into a token array.

    python scripts/01_build_dataset.py --input zinc20_leadlike.smi --out data/

Splitting is a pure function of the canonical Murcko scaffold, so this is
reproducible on any machine from the input file alone -- no split manifest to
lose. The exact counts are written to data/split_manifest.json anyway.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from molcircuits.chem.tokenizer import SmilesTokenizer
from molcircuits.train.data import (LeadLikeFilter, PackedDataset, iter_smiles,
                                scaffold_of, split_of, write_split_manifest)
from molcircuits.utils.manifest import Manifest, dump_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".smi/.txt, one SMILES per line")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--salt", default="molcircuits-v1")
    ap.add_argument("--no-filter", action="store_true",
                    help="skip lead-like filtering (input already filtered)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    filt = None if args.no_filter else LeadLikeFilter()

    buckets: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    n_in = n_kept = 0
    for smi in tqdm(iter_smiles(args.input), desc="filter+split"):
        if args.limit and n_in >= args.limit:
            break
        n_in += 1
        canon = filt(smi) if filt else smi
        if canon is None:
            continue
        scaf = scaffold_of(canon)
        if scaf is None:
            continue
        buckets[split_of(scaf, salt=args.salt)].append(canon)
        n_kept += 1

    tok = SmilesTokenizer.from_corpus(buckets["train"])
    tok.save(os.path.join(args.out, "tokenizer.json"))

    counts = {}
    for name, smis in buckets.items():
        with open(os.path.join(args.out, f"{name}.smi"), "w") as fh:
            fh.write("\n".join(smis))
        PackedDataset.build(smis, tok, args.n_ctx,
                            out_path=os.path.join(args.out, f"{name}_packed.npy"))
        counts[name] = len(smis)

    write_split_manifest(os.path.join(args.out, "split_manifest.json"),
                         counts, args.salt)

    tokens_per_split = {
        k: int(np.load(os.path.join(args.out, f"{k}_packed.npy")).shape[0])
        for k in buckets
    }
    payload = {"n_input": n_in, "n_kept": n_kept, "molecule_counts": counts,
               "token_counts": tokens_per_split, "vocab_size": tok.vocab_size,
               "filtered": not args.no_filter,
               "mean_tokens_per_molecule": (
                   sum(tokens_per_split.values()) / max(n_kept, 1))}
    dump_json(payload, os.path.join(args.out, "dataset_stats.json"))
    Manifest().record("dataset", payload, config=vars(args),
                      files=[os.path.join(args.out, "tokenizer.json")])
    print(payload)


if __name__ == "__main__":
    main()
