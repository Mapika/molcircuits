"""Pooled residual statistics over content tokens.

Every difference-in-means direction and every RMS unit in this package is
computed over **content tokens only** -- positions strictly between BOS and EOS.
The reason is the one given in :func:`molcircuits.downstream.features.embed`: BOS and
EOS are attention sinks whose residual norms dwarf every chemical position, and
they carry the same content in every molecule.  A mean that includes them is
dominated by two constant positions, so a difference of two such means is mostly
the noise between two nearly identical vectors, and an RMS that includes them
overstates the scale that an intervention should be measured against.

These lived as private helpers inside the stage-14 script, which meant the one
analysis that needed them had to import a *script* by file path through
``importlib``.  Anything measuring a direction in the residual stream needs the
same pooling, and it has to be the same pooling or the numbers are not
comparable.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from ..model.hooks import HookManager, resid_post


def _content_batches(tok, smiles: Sequence[str], n_ctx: int, batch: int,
                     device) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(token_ids, content_mask)`` batches, skipping unencodable input.

    A molecule that fails to tokenise, or that has no interior once BOS and EOS
    are removed, is dropped rather than padded into the batch: it would
    contribute only masked positions and inflate nothing, but counting it would
    make the reported ``n`` disagree with the number of positions actually
    pooled.
    """
    for start in range(0, len(smiles), batch):
        ids, masks = [], []
        for s in smiles[start:start + batch]:
            try:
                encoded = tok.encode(s)[:n_ctx]
            except ValueError:
                continue
            if len(encoded) < 3:
                continue
            ids.append(encoded + [tok.pad_id] * (n_ctx - len(encoded)))
            row = [0] * n_ctx
            for p in range(1, len(encoded) - 1):
                row[p] = 1
            masks.append(row)
        if not ids:
            continue
        yield (torch.tensor(ids, device=device),
               torch.tensor(masks, device=device, dtype=torch.bool))


@torch.no_grad()
def content_mean(model, tok, smiles: Sequence[str], layer: int, device: str,
                 n_ctx: int = 128, batch: int = 128) -> tuple[torch.Tensor, int]:
    """Mean residual over content tokens, pooled across molecules.

    Accumulates in float64.  The sum runs over hundreds of thousands of
    positions, and a float32 accumulator loses enough precision there to matter
    for a difference of two means that are themselves close together.
    """
    total = torch.zeros(model.cfg.d_model, device=device, dtype=torch.float64)
    n = 0
    for x, mask in _content_batches(tok, smiles, n_ctx, batch, device):
        hooks = HookManager().capture(resid_post(layer))
        model(x, hooks=hooks)
        acts = hooks[resid_post(layer)].float()
        total += (acts * mask[..., None]).sum((0, 1)).double()
        n += int(mask.sum())
    return (total / max(n, 1)).float(), n


@torch.no_grad()
def content_residual_rms(model, tok, smiles: Sequence[str], layer: int,
                         device: str, n_ctx: int = 128,
                         batch: int = 128) -> float:
    """RMS per coordinate at content positions -- the unit alpha is measured in.

    Named for the pooling rather than called ``residual_rms`` because
    :func:`molcircuits.analysis.valence_intervene.residual_rms` already computes a
    per-position RMS from a token tensor.  The two answer different questions
    and silently swapping them would rescale every intervention strength.
    """
    square = torch.zeros((), device=device, dtype=torch.float64)
    n = 0
    for x, mask in _content_batches(tok, smiles, n_ctx, batch, device):
        hooks = HookManager().capture(resid_post(layer))
        model(x, hooks=hooks)
        acts = hooks[resid_post(layer)].float()
        square += ((acts ** 2) * mask[..., None]).sum().double()
        n += int(mask.sum()) * model.cfg.d_model
    return float((square / max(n, 1)).sqrt())


def difference_in_means_direction(model, tok, with_smiles: Sequence[str],
                                  without_smiles: Sequence[str], layer: int,
                                  device: str, n_ctx: int = 128,
                                  batch: int = 128) -> tuple[torch.Tensor, dict]:
    """Unit vector separating molecules that contain a fragment from those that do not.

    Returned normalised, with the pooled counts beside it, because the norm of
    the raw difference is not interpretable on its own -- how far apart two class
    means sit depends on how many positions went into each -- while the
    *direction* is exactly what gets injected.
    """
    mu_with, n_with = content_mean(model, tok, with_smiles, layer, device,
                                   n_ctx, batch)
    mu_without, n_without = content_mean(model, tok, without_smiles, layer,
                                         device, n_ctx, batch)
    delta = mu_with - mu_without
    norm = float(delta.norm())
    meta = {
        "n_molecules_with": len(with_smiles),
        "n_molecules_without": len(without_smiles),
        "n_positions_with": n_with,
        "n_positions_without": n_without,
        "raw_norm": norm,
    }
    return delta / max(norm, 1e-12), meta
