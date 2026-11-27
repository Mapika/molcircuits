"""Pretraining loop with a real validation pass.

The single most consequential omission in the previous project was that the
training loop never evaluated held-out loss, so no base cross-entropy existed and
every SAE Delta-CE in the paper was uninterpretable. Here validation runs on a
schedule, the final value is written to the metrics artifact, and the checkpoint
carries it.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.manifest import Manifest
from .muon import build_optimizers


@dataclass
class TrainConfig:
    steps: int = 60_000
    batch_size: int = 64
    grad_accum: int = 1
    muon_lr: float = 0.02
    adam_lr: float = 3e-4
    warmup: int = 3000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 2000
    eval_batches: int = 100
    ckpt_every: int = 10_000
    seed: int = 0
    dtype: str = "bfloat16"
    compile: bool = False
    out_dir: str = "artifacts/transformer"


def lr_scale(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str, max_batches: int = 100,
             ignore_index: int = -100) -> dict:
    """Held-out cross-entropy, in nats and as perplexity."""
    model.eval()
    # The per-batch losses are kept on the device and read back once at the end:
    # `tot += loss.item()` blocked the host on every batch, which serialized the
    # whole validation pass. The host-side sum is over the same float32 values in
    # the same order, so the result is bit-identical.
    losses, n = [], 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=ignore_index)
        losses.append(loss.detach()); n += 1
    model.train()
    tot = 0.0
    for v in (torch.stack(losses).tolist() if losses else []):
        tot += v
    if n == 0:
        # `tot / max(n, 1)` returned 0.0 here, i.e. CE 0 and perplexity 1 -- a
        # perfect model, reported silently. It happens whenever the validation
        # set is smaller than one batch, because the loader used
        # drop_last=True. The shipped split is 566,513 sequences so no released
        # number is affected, but a small custom split or a debug run got a
        # flawless score for having no data.
        raise ValueError(
            "validation produced no batches: the validation set is smaller "
            "than one batch. Reduce batch_size or enlarge the split.")
    ce = tot / n
    return {"val_ce": ce, "val_ppl": math.exp(min(ce, 20)), "n_batches": n}


def train(model, train_ds, valid_ds, cfg: TrainConfig, device: str = "cuda",
          manifest: Manifest | None = None) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    dtype = dict(float32=torch.float32, bfloat16=torch.bfloat16,
                 float16=torch.float16)[cfg.dtype]
    model = model.to(device)
    # Keep the uncompiled module: torch.compile wraps it in an OptimizedModule
    # whose state_dict keys are all prefixed "_orig_mod.", and a checkpoint saved
    # that way cannot be loaded by any of the analysis scripts -- a failure that
    # only surfaces after the full training run has finished.
    raw_model = model
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, num_workers=2, pin_memory=True)
    # drop_last=False on VALIDATION: dropping the tail silently discards data,
    # and when the whole set is shorter than one batch it discards all of it.
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False,
                              drop_last=False, num_workers=2)

    opt_muon, opt_adam, split_log = build_optimizers(
        model, cfg.muon_lr, cfg.adam_lr, cfg.weight_decay)
    opts = [o for o in (opt_muon, opt_adam) if o is not None]
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

    history, step, t0 = [], 0, time.time()
    it = iter(train_loader)
    while step < cfg.steps:
        for o in opts:
            o.zero_grad(set_to_none=True)
        acc_loss = []
        for _ in range(cfg.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_loader); x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=dtype,
                                enabled=(device != "cpu")):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)
                ) / cfg.grad_accum
            loss.backward()
            # Keep the loss on the device. `acc_loss += loss.item()` forced a
            # host/device synchronization on every single step, stalling the
            # dispatch of the next step behind the whole of this one, while the
            # value is only ever read at the 100-step logging points. The
            # micro-batch losses are still summed on the host, in the same order
            # and the same float64 arithmetic, so the logged number is unchanged.
            acc_loss.append(loss.detach())

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        s = lr_scale(step, cfg.warmup, cfg.steps)
        for o, bl in zip(opts, base_lrs):
            for g, b in zip(o.param_groups, bl):
                g["lr"] = b * s
            o.step()
        step += 1

        # The only points at which the running loss is read; everywhere else it
        # stays a device tensor and never blocks the host.
        report = step % 100 == 0 or step % cfg.eval_every == 0 or step == cfg.steps
        if report:
            train_ce = 0.0
            for v in torch.stack(acc_loss).tolist():
                train_ce += v

        if step % 100 == 0:
            history.append({"step": step, "train_ce": train_ce,
                            "lr_scale": s, "elapsed_s": time.time() - t0})
        if step % cfg.eval_every == 0 or step == cfg.steps:
            m = evaluate(model, valid_loader, device, cfg.eval_batches)
            history.append({"step": step, **m})
            print(f"step {step:>7} train_ce {train_ce:.4f} "
                  f"val_ce {m['val_ce']:.4f} ppl {m['val_ppl']:.3f}", flush=True)
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            # The checkpoint used to hold only model/cfg/step, where cfg is the
            # TRAIN config. Three consequences: a run could not be resumed (Muon
            # and Adam moments and the RNG state were gone), the last validation
            # value the module docstring promises was absent, and the model
            # GEOMETRY lived only in a sibling model_config.json -- loading the
            # state dict under a different rope_theta reports "All keys matched
            # successfully" and moves the logits by up to 3.09.
            torch.save({"model": raw_model.state_dict(),
                        "cfg": asdict(cfg),
                        "model_cfg": asdict(raw_model.cfg),
                        "step": step,
                        "val": history[-1] if history else None,
                        "opt_muon": opt_muon.state_dict(),
                        "opt_adam": opt_adam.state_dict(),
                        "rng": {"torch": torch.get_rng_state(),
                                "cuda": (torch.cuda.get_rng_state_all()
                                         if torch.cuda.is_available() else None)}},
                       os.path.join(cfg.out_dir, "checkpoint.pt"))

    final = evaluate(model, valid_loader, device, cfg.eval_batches * 2)
    out = {"final": final, "history": history, "optimizer_assignment": split_log,
           "n_params_non_embedding": raw_model.n_params(True) if hasattr(raw_model, "n_params") else None}
    with open(os.path.join(cfg.out_dir, "train_metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    if manifest:
        # Record the config and checksum the checkpoint. Previously this stage
        # logged only the final loss: no seed, no hyperparameters, and no hash of
        # the single most load-bearing artifact in the project.
        manifest.record("train", out["final"], config=asdict(cfg),
                        files=[os.path.join(cfg.out_dir, "checkpoint.pt")])
    return out
