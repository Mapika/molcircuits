#!/usr/bin/env python3
"""Pretrain the molecular transformer.

    python scripts/02_train_transformer.py --data data --seed 0 \
        --out artifacts/transformer_seed0

Train several seeds. At this scale a packed run is a few hours, which turns
"we analyze one model, so findings may change with seed" from a limitation into
a measurement.
"""
from __future__ import annotations

import argparse
import os

import torch

from molcircuits.chem.tokenizer import SmilesTokenizer
from molcircuits.model.sampling import evaluate_generations, generate
from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.train.data import PackedDataset
from molcircuits.train.trainer import TrainConfig, train
from molcircuits.utils.manifest import Manifest, dump_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/transformer")
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--n-ctx", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=2048)
    ap.add_argument("--rope-theta", type=float, default=10_000.0)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--gen-samples", type=int, default=10_000)
    ap.add_argument("--gen-max-tokens", type=int, default=256)
    ap.add_argument("--gen-temperature", type=float, default=1.0)
    args = ap.parse_args()

    # Seed BEFORE the model is constructed. trainer.train() also seeds, but it
    # runs after __init__ has already drawn every weight from torch's default
    # generator -- which seeds itself from the OS on first use. The result was
    # that --seed selected the data order and nothing else: three runs at
    # --seed 0 produced three different models, and no run was reproducible.
    torch.manual_seed(args.seed)
    tok = SmilesTokenizer.load(os.path.join(args.data, "tokenizer.json"))
    mcfg = ModelConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                       n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff,
                       n_ctx=args.n_ctx, rope_theta=args.rope_theta)
    model = MolecularTransformer(mcfg)
    print(f"non-embedding parameters: {model.n_params(True):,}")
    print(f"total parameters:         {model.n_params(False):,}")

    tr = PackedDataset.load(os.path.join(args.data, "train_packed.npy"), args.n_ctx)
    va = PackedDataset.load(os.path.join(args.data, "valid_packed.npy"), args.n_ctx)

    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size,
                       grad_accum=args.grad_accum, muon_lr=args.muon_lr,
                       adam_lr=args.adam_lr, eval_every=args.eval_every,
                       seed=args.seed, dtype=args.dtype, compile=args.compile,
                       out_dir=args.out)
    os.makedirs(args.out, exist_ok=True)
    mcfg.save(os.path.join(args.out, "model_config.json"))

    man = Manifest()
    out = train(model, tr, va, tcfg, device=args.device, manifest=man)

    # Unconditional generation quality, with the settings recorded alongside.
    # A validity number without its max_tokens and temperature is meaningless:
    # the previous project had two figures for the same checkpoint (49.9% and
    # 100%) that differed only by generation budget.
    # Canonicalized with the local RDKit so a memorized molecule cannot count as
    # novel merely because the corpus spelling differs (3.7% of this corpus).
    # Done in parallel: single-core this is ~108 min at 34.9M molecules.
    from multiprocessing import Pool  # noqa: PLC0415

    from molcircuits.model.sampling import canonical_or_none  # noqa: PLC0415
    _raw = open(os.path.join(args.data, "train.smi")).read().split()
    with Pool(max(1, (os.cpu_count() or 8) - 4)) as _pool:
        train_canon = {c for c in _pool.map(canonical_or_none, _raw, 20000) if c}
    del _raw
    smis = generate(model, tok, args.gen_samples, max_tokens=args.gen_max_tokens,
                    temperature=args.gen_temperature, seed=args.seed,
                    device=args.device)
    gen = evaluate_generations(smis, train_canon, args.gen_max_tokens,
                               args.gen_temperature,
                               canonicalize_reference=False).as_dict()
    dump_json(gen, os.path.join(args.out, "generation_metrics.json"))
    man.record("generation", gen, config=vars(args))
    print("final:", out["final"])
    print("generation:", gen)


if __name__ == "__main__":
    main()
