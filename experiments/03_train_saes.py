#!/usr/bin/env python3
"""Train SAEs at every layer and expansion factor, and evaluate each one.

    python experiments/03_train_saes.py --ckpt artifacts/transformer/checkpoint.pt \
        --data data --expansions 4 8 16 --out artifacts/saes

Every dictionary is written with its realized L0, dead-feature fraction, and the
fraction of cross-entropy recovered against both a zero- and a mean-ablation
floor. Those are the numbers a reviewer needs to judge whether a dictionary is
any good, and they belong in the artifact, not only in a metrics dashboard.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from molcircuits.model.transformer import ModelConfig, MolecularTransformer
from molcircuits.sae.evaluate import evaluate_sae
from molcircuits.sae.sae import SAEConfig
from molcircuits.sae.train_sae import (SAETrainConfig, collect_activations,
                                   collect_activations_multi, train_sae)
from molcircuits.train.data import PackedDataset
from molcircuits.utils.manifest import Manifest, config_hash, dump_json, file_sha256


def completed_dictionary(directory, signature, layer, expansion):
    """Reuse only a fully evaluated dictionary with the same inputs and settings."""
    try:
        with open(directory / "eval_metrics.json") as handle:
            metrics = json.load(handle)
        if (metrics.get("run_signature") != signature or metrics.get("layer") != layer
                or metrics.get("expansion") != expansion):
            return None
        for name in ("sae.pt", "sae_config.json"):
            if metrics["dictionary_sha256"][name] != file_sha256(directory / name):
                return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-config", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts/saes")
    ap.add_argument("--expansions", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    # TopK by default, matching every committed dictionary. The pipeline used to
    # default to ReLU, so `run_all.sh` reproduced a configuration this project
    # does not use and the README calls degenerate: at the default l1_coeff the
    # penalty is inert (L0 723.0 against 723.9 at l1_coeff=0), so deeper ReLU
    # dictionaries train toward an identity map.
    ap.add_argument("--variant", default="topk", choices=["relu", "topk", "jumprelu"])
    # ReLU only, and in NORMALIZED loss units after the loss-scale fix: the
    # useful range is O(1), not 5e-4. Ignored by topk and jumprelu.
    ap.add_argument("--l1-coeff", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-activations", type=int, default=50_000_000)
    ap.add_argument("--sae-batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-ctx", type=int, default=256)
    # Activation collection is forward-only, so it can run a far wider batch than
    # training does; at 32 the stage was launch-latency bound with the GPU at
    # ~16% memory utilization.
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=50)
    # --batch-size drives activation *collection*, which wants to be wide.
    # It also drove the eval loader, and because --eval-batches counts batches
    # rather than tokens, raising collection from 32 to 256 multiplied the
    # fidelity evaluation by 8x (2.05M -> 16.38M tokens per dictionary) and
    # silently ate a 1.49x TF32 gain. Default None keeps them coupled so this
    # run's seeds stay comparable; set it explicitly to decouple.
    ap.add_argument("--eval-batch-size", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="reuse verified complete dictionaries, retraining any partial or stale one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg_path = args.model_config or os.path.join(
        os.path.dirname(args.ckpt), "model_config.json")
    mcfg = ModelConfig.load(cfg_path)
    model = MolecularTransformer(mcfg).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device)["model"])
    model.eval()

    # Reproducibility of the ACTIVATION SAMPLE, which is what every dictionary
    # in this stage is fitted to. `shuffle=True` without a generator draws from
    # the global RNG, which nothing here seeds, so two runs with the same
    # --seed collected different activations. Worse, the loader is re-iterated
    # once per layer and reshuffled each time, so every layer's dictionary was
    # fitted to a DIFFERENT token sample -- and the paper compares layers.
    torch.manual_seed(args.seed)
    sample_gen = torch.Generator().manual_seed(args.seed)

    tr = PackedDataset.load(os.path.join(args.data, "train_packed.npy"), args.n_ctx)
    va = PackedDataset.load(os.path.join(args.data, "valid_packed.npy"), args.n_ctx)
    train_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                              generator=sample_gen)
    eval_loader = DataLoader(va, batch_size=args.eval_batch_size or args.batch_size,
                             shuffle=False)

    layers = args.layers if args.layers is not None else list(range(mcfg.n_layers))
    man, all_rows = Manifest(), []

    # Each fit resets its own RNG. A completed layer therefore needs neither
    # another activation pass nor another fit when its exact inputs match.
    onepass = bool(os.environ.get("MOLCIRCUITS_SAE_ONEPASS") and
                   args.device.startswith("cuda") and len(layers) > 1)
    repo = Path(__file__).resolve().parents[1]
    sources = ["experiments/03_train_saes.py", "molcircuits/model/transformer.py",
               "molcircuits/model/hooks.py", "molcircuits/train/data.py",
               "molcircuits/sae/sae.py", "molcircuits/sae/train_sae.py", "molcircuits/sae/evaluate.py"]
    inputs = [args.ckpt, cfg_path, *(os.path.join(args.data, f"{split}_packed.npy")
                                    for split in ("train", "valid"))]
    settings = {k: v for k, v in vars(args).items()
                if k not in ("out", "resume", "layers", "expansions")}
    signature = config_hash({"settings": settings, "onepass": onepass, "torch": torch.__version__,
                             "no_graph": bool(os.environ.get("MOLCIRCUITS_SAE_NO_GRAPH")),
                             "source": [file_sha256(repo / f) for f in sources],
                             "inputs": [file_sha256(f) for f in inputs]})
    cached = {}
    if args.resume:
        for layer in layers:
            for exp in args.expansions:
                directory = Path(args.out) / f"layer{layer}_x{exp}"
                result = completed_dictionary(directory, signature, layer, exp)
                if result is not None:
                    cached[layer, exp] = result
    needed = [layer for layer in layers if any((layer, exp) not in cached
                                                for exp in args.expansions)]

    # MOLCIRCUITS_SAE_ONEPASS=1: one forward pass over the corpus caches every
    # layer at once (the capture hooks do not modify the stream, so each
    # layer's activations are the same values a dedicated pass computes). Only
    # the first layer's cache lives in VRAM (two don't fit); the rest are
    # staged in pinned host RAM in the identical GPU-cast float16 and uploaded
    # wholesale below. Measured: 11.9 min for the whole stage vs 27.3 min for
    # the default flow (baseline 33.4 min).
    #
    # It is OPT-IN because it CHANGES THE TRAINED DICTIONARIES at layers >= 1
    # relative to the committed seed-0 artifacts. In the default per-layer
    # flow, `_cache_plan` sees the previous layer's 51 GB cache still counted
    # against free VRAM (`torch.cuda.mem_get_info` cannot see blocks parked in
    # the caching allocator), so only the FIRST layer gets a float16 VRAM
    # cache and every later layer silently falls back to a float32 host cache.
    # The reference dictionaries for layers >= 1 were therefore trained on
    # float32 activations. ONEPASS caches every layer in float16, which is
    # uniform across layers -- arguably more principled, and its rounding is
    # ~2 orders below the measured FVU -- but it is a different training input,
    # and 24k steps amplify that into different (equally good) dictionaries:
    # summary metrics agree with the reference to <= 3.3e-4 in CE while the
    # weights land in a different basin. Keep it OFF for strict reproduction.
    #
    # Cross-entropy floors that don't depend on the SAE (clean, zero- and
    # mean-ablation) are computed once per layer and shared across expansions
    # -- they are bitwise what a per-dictionary recomputation yields.
    staged = None
    if onepass and needed:
        sample_gen.manual_seed(args.seed)
        staged = collect_activations_multi(model, train_loader, needed,
                                           args.n_activations, args.device,
                                           mcfg.d_model)
    ce_cache = {}

    for layer in layers:
        if layer not in needed:
            all_rows.extend(cached[layer, exp] for exp in args.expansions)
            print(f"layer {layer}: reused {len(args.expansions)} verified dictionaries")
            continue
        if staged is None:
            # Reset before every layer so all layers see the SAME tokens. Without
            # this a cross-layer difference is confounded with a change of sample.
            sample_gen.manual_seed(args.seed)
            acts = collect_activations(model, train_loader, layer,
                                       args.n_activations, args.device,
                                       mcfg.d_model)
        else:
            acts = staged.pop(layer)
            if acts.device.type != torch.device(args.device).type:
                acts = acts.to(args.device, non_blocking=False)
        print(f"layer {layer}: cached {len(acts):,} activations")
        for exp in args.expansions:
            if (layer, exp) in cached:
                all_rows.append(cached[layer, exp])
                print(f"layer {layer} x{exp}: reused verified dictionary")
                continue
            scfg = SAEConfig(d_in=mcfg.d_model, expansion=exp, variant=args.variant,
                             l1_coeff=args.l1_coeff, k=args.k, layer=layer)
            tcfg = SAETrainConfig(n_activations=args.n_activations,
                                  batch_size=args.sae_batch, lr=args.lr,
                                  seed=args.seed)
            out_dir = os.path.join(args.out, f"layer{layer}_x{exp}")
            sae, summary = train_sae(acts, scfg, tcfg, args.device, out_dir)
            metrics = evaluate_sae(model, sae, eval_loader, args.device, layer,
                                   args.eval_batches, ce_cache=ce_cache)
            metrics.update({"train_summary": summary, "seed": args.seed,
                            "run_signature": signature,
                            "activation_cache_dtype": str(acts.dtype),
                            "activation_cache_device": acts.device.type,
                            "dictionary_sha256": {
                                name: file_sha256(os.path.join(out_dir, name))
                                for name in ("sae.pt", "sae_config.json")}})
            dump_json(metrics, os.path.join(out_dir, "eval_metrics.json"))
            all_rows.append(metrics)
            print(f"  x{exp}: L0={metrics['l0']:.1f} dCE={metrics['delta_ce']:.4f} "
                  f"CE_rec(zero)={metrics['fraction_ce_recovered_vs_zero']:.3f} "
                  f"dead={metrics['dead_feature_fraction']:.3f}")
        del acts
        if args.device.startswith("cuda"):
            # Hand the cache back to the DRIVER, not merely to PyTorch's caching
            # allocator. `_cache_plan` sizes the next layer's cache from
            # `torch.cuda.mem_get_info`, which cannot see blocks parked in the
            # allocator, so without this every layer after the first saw ~46 GB
            # free against a ~48 GB requirement and silently fell back to a
            # float32 HOST cache. Activation precision then depended on how much
            # VRAM happened to be free: layer 0 trained on float16, layers 1-5 on
            # float32, and the same code on a smaller card would differ again.
            # Every layer now gets the float16 VRAM cache the docstring
            # describes, which also lets the CUDA-graph step engage throughout.
            torch.cuda.empty_cache()

    summary_path = dump_json(all_rows, os.path.join(args.out, "sae_summary.json"))
    checkpoint_files = [
        os.path.join(args.out, f"layer{row['layer']}_x{row['expansion']}", "sae.pt")
        for row in all_rows]
    man.record("saes", {"n_dictionaries": len(all_rows), "rows": all_rows},
               config=vars(args), files=[summary_path, *checkpoint_files])


if __name__ == "__main__":
    main()
