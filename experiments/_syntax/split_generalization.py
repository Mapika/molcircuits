"""Cross-entropy of trained checkpoints on seen and unseen scaffolds.

The corpus is split by Murcko scaffold, and the import keeps every ``stride``-th
row of each training shard. The rows in between are therefore molecules the
model never read, drawn from the same scaffolds it trained on. Scoring them next
to the validation and test splits separates two things that the validation loss
alone confounds:

* ``train_unseen``: new molecules, scaffolds on the training side of the split;
* ``valid`` / ``test``: new molecules, new scaffolds.

Every arm is scored on the same number of packed sequences, and the per-sequence
losses give a standard error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from molcircuits.chem.tokenizer import SmilesTokenizer
from molcircuits.model.transformer import ModelConfig, MolecularTransformer


def _shard_paths(hf_dir: str, split: str) -> list[str]:
    return sorted(glob.glob(os.path.join(hf_dir, split, "*.arrow")))


def _read_shard(path: str):
    import pyarrow as pa
    with pa.OSFile(path, "rb") as src:
        try:
            return pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            src.seek(0)
            return pa.ipc.open_file(src).read_all()


def unseen_train_ids(hf_dir: str, column: str, stride: int, offset: int,
                     n_molecules: int, tok: SmilesTokenizer, seed: int) -> np.ndarray:
    """Token ids of training-split rows that the strided import skipped."""
    if not 0 < offset < stride:
        raise SystemExit("offset must lie strictly between 0 and stride, "
                         "otherwise the rows are the imported ones")
    paths = _shard_paths(hf_dir, "train")
    per_shard = -(-n_molecules // len(paths))
    smis: list[str] = []
    for p in paths:
        col = _read_shard(p).column(column)
        idx = np.arange(offset, len(col), stride)
        # Spread the sample over the whole shard: shard order tracks molecule size.
        idx = idx[np.linspace(0, len(idx) - 1, min(per_shard, len(idx))).astype(int)]
        smis.extend(col.take(idx).to_pylist())
    rng = np.random.default_rng(seed)
    buf: list[int] = []
    for i in rng.permutation(len(smis)):
        try:
            buf.extend(tok.encode(smis[i]))
        except ValueError:
            continue
    return np.asarray(buf, dtype=np.int64)


@torch.no_grad()
def sequence_losses(model, ids: np.ndarray, n_ctx: int, n_seq: int, batch: int,
                    device: str) -> np.ndarray:
    n_seq = min(n_seq, (len(ids) - 1) // n_ctx)
    out = []
    for s in range(0, n_seq, batch):
        rows = range(s, min(s + batch, n_seq))
        x = torch.from_numpy(np.stack([ids[i * n_ctx:(i + 1) * n_ctx] for i in rows]
                                      ).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([ids[i * n_ctx + 1:(i + 1) * n_ctx + 1] for i in rows]
                                      ).astype(np.int64)).to(device)
        logits = model(x).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="none").reshape(len(rows), n_ctx)
        out.append(loss.mean(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--column", default="smiles")
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--offset", type=int, default=3)
    ap.add_argument("--n-sequences", type=int, default=25_600)
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = SmilesTokenizer.load(os.path.join(args.data, "tokenizer.json"))
    need = args.n_sequences * args.n_ctx + 1
    arms = {}
    for split in ("train", "valid", "test"):
        arr = np.load(os.path.join(args.data, f"{split}_packed.npy"), mmap_mode="r")
        arms["train_imported" if split == "train" else split] = np.asarray(arr[:need])
    # ~40 tokens per molecule; 1.3x head-room for the molecules that fail to encode.
    n_mol = int(need / 40 * 1.3)
    arms["train_unseen"] = unseen_train_ids(args.hf_dir, args.column, args.stride,
                                            args.offset, n_mol, tok, args.seed)
    for k, v in arms.items():
        print(f"{k}: {len(v):,} tokens", flush=True)

    results = {"config": vars(args), "runs": {}}
    for ckpt in args.ckpts:
        mcfg = ModelConfig.load(os.path.join(os.path.dirname(ckpt), "model_config.json"))
        model = MolecularTransformer(mcfg).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device)["model"])
        model.eval()
        row = {}
        for k, ids in arms.items():
            with torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16,
                                enabled=args.device != "cpu"):
                ls = sequence_losses(model, ids, args.n_ctx, args.n_sequences,
                                     args.batch, args.device)
            row[k] = {"ce": float(ls.mean()),
                      "se": float(ls.std(ddof=1) / np.sqrt(len(ls))),
                      "n_sequences": int(len(ls))}
        results["runs"][ckpt] = row
        print(ckpt, {k: round(v["ce"], 4) for k, v in row.items()}, flush=True)
        del model

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
