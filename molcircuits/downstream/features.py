"""Molecule representations for downstream transfer benchmarks.

Three representations, produced through one code path so a comparison between
them differs only in the representation:

* ``dense``  -- pooled residual stream at a layer (the transformer embedding)
* ``sae``    -- pooled SAE code at the same layer and the same positions
* ``ecfp``   -- Morgan fingerprint, the chemistry baseline

Pooling is over *real tokens only*. The model is pretrained on packed sequences
and never sees a pad token, so pooling across padding mixes in the model's
response to an input it was never trained on -- the same defect that inverted
`delta_ce` in the head-ablation sweep.

Max-pooling is the default because that is what the paper being reproduced used
("max-pooled SAE features"); mean is available for comparison. For a non-negative
SAE code the two differ substantially, so the choice is recorded in the artifact
rather than left implicit.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
import os

import numpy as np
import torch

from .. import paths
from ..chem.tokenizer import SmilesTokenizer
from ..model.hooks import HookManager, resid_post
from ..model.transformer import ModelConfig, MolecularTransformer
from ..sae.sae import SparseAutoencoder


def random_active_subset(train_variance, width, rng):
    """Draw distinct coordinates, filling from constant training columns only if necessary."""
    if not 0 < width <= len(train_variance):
        raise ValueError(f"Requested {width} coordinates from a {len(train_variance)}-wide dictionary")
    active = np.flatnonzero(train_variance > 0)
    selected = rng.choice(active, size=min(width, len(active)), replace=False)
    if len(selected) < width:
        inactive = np.flatnonzero(train_variance == 0)
        selected = np.concatenate([selected, rng.choice(inactive, width-len(selected), replace=False)])
    return selected


def load_stack(ckpt: str, sae_dir: str | None = None, device: str = "cuda", *,
               tokenizer_path: str | None = None,
               model_config_path: str | None = None,
               return_metadata: bool = False):
    """Load a checkpoint, tokenizer and optional SAE with explicit provenance.

    New checkpoints carry ``model_cfg`` and that embedded geometry is the
    authoritative description of the state dict.  Older checkpoints are still
    supported through a sibling ``model_config.json`` (or an explicitly passed
    ``model_config_path``).  The tokenizer must likewise come from an explicit
    path/environment setting, an embedded vocabulary, or a checkpoint-local
    ``tokenizer.json``; there is deliberately no machine-specific fallback.

    Set ``return_metadata`` to append a JSON-serialisable provenance mapping to
    the usual four return values.
    """
    resolved_ckpt = os.path.realpath(ckpt)
    payload = torch.load(resolved_ckpt, map_location=device)

    embedded_cfg = payload.get("model_cfg")
    if embedded_cfg is not None:
        mcfg = (embedded_cfg if isinstance(embedded_cfg, ModelConfig)
                else ModelConfig(**embedded_cfg))
        cfg_source = f"{resolved_ckpt}::model_cfg"
        # If the caller names a second source explicitly, disagreement is an
        # error rather than an arbitrary choice between two model geometries.
        if model_config_path is not None:
            declared = ModelConfig.load(model_config_path)
            if vars(declared) != vars(mcfg):
                raise ValueError(
                    "explicit model_config_path disagrees with checkpoint model_cfg"
                )
    else:
        cfg_path = (model_config_path or
                    os.path.join(os.path.dirname(resolved_ckpt),
                                 "model_config.json"))
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                "checkpoint has no embedded model_cfg and no model config was "
                f"found at {cfg_path!r}"
            )
        cfg_path = os.path.realpath(cfg_path)
        mcfg = ModelConfig.load(cfg_path)
        cfg_source = cfg_path

    model = MolecularTransformer(mcfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    tokenizer_path = tokenizer_path or paths.tokenizer()
    embedded_tok = payload.get("tokenizer_vocab", payload.get("tokenizer"))
    if tokenizer_path is not None:
        tokenizer_path = os.path.realpath(tokenizer_path)
        tok = SmilesTokenizer.load(tokenizer_path)
        tok_source = tokenizer_path
    elif isinstance(embedded_tok, Mapping):
        vocab = embedded_tok.get("vocab", embedded_tok)
        tok = SmilesTokenizer(vocab=dict(vocab))
        tok_source = f"{resolved_ckpt}::tokenizer_vocab"
    else:
        checkpoint_tok = os.path.join(os.path.dirname(resolved_ckpt),
                                      "tokenizer.json")
        if not os.path.isfile(checkpoint_tok):
            raise FileNotFoundError(
                "no tokenizer was configured: pass tokenizer_path, set "
                "MOLCIRCUITS_TOKENIZER, embed tokenizer_vocab in the checkpoint, "
                "or place tokenizer.json beside the checkpoint"
            )
        checkpoint_tok = os.path.realpath(checkpoint_tok)
        tok = SmilesTokenizer.load(checkpoint_tok)
        tok_source = checkpoint_tok
    if tok.vocab_size != mcfg.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary ({tok.vocab_size}) does not match checkpoint "
            f"model vocabulary ({mcfg.vocab_size})"
        )

    sae = None
    if sae_dir:
        sae = SparseAutoencoder.load(sae_dir, device).to(device).eval()
        if sae.cfg.d_in != mcfg.d_model:
            raise ValueError(
                f"SAE input width ({sae.cfg.d_in}) does not match model d_model "
                f"({mcfg.d_model})"
            )

    provenance = {
        "checkpoint": resolved_ckpt,
        "model_config_source": cfg_source,
        "tokenizer_source": tok_source,
        "sae_directory": os.path.realpath(sae_dir) if sae_dir else None,
    }
    loaded = (model, tok, sae, mcfg)
    return (*loaded, provenance) if return_metadata else loaded


def content_mask_row(n_encoded: int, n_ctx: int) -> list[int]:
    """Interior-token mask for one encoding: everything but BOS and EOS.

    CONTENT TOKENS ONLY: position 0 is BOS and the last real position is EOS,
    and both are attention sinks whose residual norm dwarfs every chemical
    position. Measured at layer 3: EOS 13,753 and BOS 2,414 against 312 for
    interior tokens, a 44x gap, so under max pooling 52.8% of the 512
    dimensions took their value from one of those two positions (chance would
    be 3.7%). The pooled vector was then nearly independent of the molecule:
    mean pairwise cosine between DIFFERENT molecules was 0.9997, and it stayed
    at 0.9997 when the model was fed a permuted vocabulary that raised its
    next-token loss from 1.42 to 11.16 nats. Excluding the two positions takes
    that cosine to 0.9276 and multiplies the median per-dimension coefficient
    of variation by 6.2.

    Shared by :func:`embed` and :func:`token_activations` so that a basis whose
    standardizer is fit here is fit over exactly the positions it is scored on.
    """
    row = [0] * n_ctx
    for p in range(1, min(n_encoded - 1, n_ctx)):
        row[p] = 1
    if not any(row):            # a 1-2 token encoding has no interior
        row[:n_encoded] = [1] * n_encoded
    return row


@torch.no_grad()
def token_activations(smiles, model, tok, layer: int = 3, n_ctx: int = 256,
                      device: str = "cuda", batch: int = 256):
    """Residual activations for every content token, as ``(n_tokens, d_model)``.

    Used to fit a per-neuron standardizer on corpus molecules rather than on any
    downstream endpoint, so that a transformed-dense arm carries no information
    from the task it is evaluated on.
    """
    out = []
    for s0 in range(0, len(smiles), batch):
        ids, mask = [], []
        for s in smiles[s0:s0 + batch]:
            try:
                e = tok.encode(s)
            except ValueError:
                continue
            if len(e) > n_ctx:
                continue
            ids.append(e + [tok.pad_id] * (n_ctx - len(e)))
            mask.append(content_mask_row(len(e), n_ctx))
        if not ids:
            continue
        x = torch.tensor(ids, device=device)
        m = torch.tensor(mask, device=device, dtype=torch.bool)
        h = HookManager().capture(resid_post(layer))
        model(x, hooks=h)
        a = h[resid_post(layer)].float()
        out.append(a[m].cpu().numpy())
    if not out:
        raise ValueError("no encodable molecules for standardizer fit")
    return np.concatenate(out)


@torch.no_grad()
def embed(smiles, model, tok, sae=None, layer: int = 3, pooling: str = "max",
          n_ctx: int = 256, device: str = "cuda", batch: int = 256,
          return_metadata: bool = False, transforms=None):
    """Pooled dense and optional SAE representations.

    Tokenization failures and encodings longer than ``n_ctx`` retain a zero
    placeholder row and are returned as drop indices.  In particular, an
    overlength molecule is never represented by a truncated learned vector
    while ECFP sees its full graph.  With ``return_metadata=True`` the third
    return value is a JSON-serialisable mapping with separate failure classes;
    otherwise it remains the combined index list for backwards compatibility.
    With ``pooling="both"``, the dense and SAE outputs are dictionaries keyed
    by ``"max"`` and ``"mean"``, reduced from one common forward pass.

    ``transforms`` maps an arm name to ``(d_out, fn)``, where ``fn`` maps token
    activations ``(n, d_model)`` to ``(n, d_out)``.  It exists so that a basis
    defined by a *token-level* nonlinearity can be pooled the same way the SAE
    code is.  This has to happen here rather than in the caller: ReLU--TopK is
    applied per token and does not commute with pooling, so transforming an
    already-pooled dense vector would measure a different object.  When
    ``transforms`` is given the return value gains a fourth element, a mapping
    from arm name to its pooled matrix; without it the signature is unchanged.
    """
    if n_ctx > model.cfg.n_ctx:
        raise ValueError(
            f"requested n_ctx={n_ctx} exceeds checkpoint n_ctx={model.cfg.n_ctx}"
        )
    if n_ctx < 2:
        raise ValueError("n_ctx must leave room for BOS and EOS")

    d_model = model.cfg.d_model
    poolings = ("max", "mean") if pooling == "both" else (pooling,)
    if pooling == "both" and transforms:
        raise ValueError("transformed arms require one pooling mode")
    dense = {how: np.zeros((len(smiles), d_model), dtype=np.float32) for how in poolings}
    codes = {how: np.zeros((len(smiles), sae.cfg.d_sae), dtype=np.float32)
             for how in poolings} if sae is not None else None
    extra = {name: np.zeros((len(smiles), d_out), dtype=np.float32)
             for name, (d_out, _) in (transforms or {}).items()}
    tokenizer_failed: list[int] = []
    overlength: list[int] = []
    valid_items: list[tuple[int, int, list[int]]] = []

    for j, s in enumerate(smiles):
        try:
            full = tok.encode(s)
        except ValueError:
            tokenizer_failed.append(j)
            continue
        if len(full) > n_ctx:
            overlength.append(j)
            continue
        valid_items.append((len(full), j, full))

    # Group similarly sized SMILES so causal forward passes pad only to each
    # batch's maximum length rather than n_ctx. Causal attention makes shorter
    # right-padding mathematically identical on every content token while
    # avoiding wasted compute on pad positions.
    valid_items.sort(key=lambda item: item[0])

    for s0 in range(0, len(valid_items), batch):
        chunk = valid_items[s0:s0 + batch]
        batch_len = max(length for length, _, _ in chunk)
        ids = [e + [tok.pad_id] * (batch_len - length) for length, _, e in chunk]
        # CONTENT TOKENS ONLY: position 0 is BOS and the last real position
        # is EOS, and both are attention sinks whose residual norm dwarfs
        # every chemical position. Measured at layer 3: EOS 13,753 and BOS
        # 2,414 against 312 for interior tokens, a 44x gap, so under max
        # pooling 52.8% of the 512 dimensions took their value from one of
        # those two positions (chance would be 3.7%). The pooled vector was
        # then nearly independent of the molecule: mean pairwise cosine
        # between DIFFERENT molecules was 0.9997, and it stayed at 0.9997
        # when the model was fed a permuted vocabulary that raised its
        # next-token loss from 1.42 to 11.16 nats. Excluding the two
        # positions takes that cosine to 0.9276 and multiplies the median
        # per-dimension coefficient of variation by 6.2.
        mask = [content_mask_row(length, batch_len) for length, _, _ in chunk]
        output_rows = [idx for _, idx, _ in chunk]
        x = torch.tensor(ids, device=device)
        m = torch.tensor(mask, device=device, dtype=torch.bool)

        h = HookManager().capture(resid_post(layer))
        model(x, hooks=h)
        a = h[resid_post(layer)].float()                      # (b, T, d_model)
        indices = np.asarray(output_rows, dtype=int)
        for how in poolings:
            dense[how][indices] = _pool(a, m, how).cpu().numpy()

        if sae is not None:
            # Encode only unmasked content tokens; sae.encode is strictly
            # position-independent, so skipping pad/BOS/EOS gives identical
            # pooled codes while saving substantial compute on wide dictionaries.
            flat_z = sae.encode(a[m])
            counts = m.sum(dim=1).tolist()
            pos = 0
            for row_idx, cnt in zip(output_rows, counts):
                seg = flat_z[pos:pos + cnt]
                pos += cnt
                if "max" in poolings:
                    codes["max"][row_idx] = seg.max(dim=0).values.cpu().numpy()
                if "mean" in poolings:
                    codes["mean"][row_idx] = seg.mean(dim=0).cpu().numpy()

        for name, (d_out, fn) in (transforms or {}).items():
            t = fn(a.reshape(-1, d_model)).reshape(a.shape[0], a.shape[1], -1)
            if t.shape[-1] != d_out:
                raise RuntimeError(
                    f"transform {name!r} returned width {t.shape[-1]}, expected {d_out}")
            extra[name][indices] = _pool(t, m, pooling).cpu().numpy()

    dropped = sorted(set(tokenizer_failed) | set(overlength))
    metadata = {
        "drop_policy": "drop_from_all_representations",
        "n_input": len(smiles),
        "n_ctx": int(n_ctx),
        "n_dropped": len(dropped),
        "dropped_indices": dropped,
        "n_tokenizer_failed": len(set(tokenizer_failed)),
        "tokenizer_failed_indices": sorted(set(tokenizer_failed)),
        "n_overlength": len(set(overlength)),
        "overlength_indices": sorted(set(overlength)),
    }
    if dropped:
        print(f"  embed: dropping {len(dropped)} of {len(smiles)} molecules "
              f"({metadata['n_tokenizer_failed']} tokenization failures, "
              f"{metadata['n_overlength']} over n_ctx={n_ctx})")
    meta = metadata if return_metadata else dropped
    if pooling != "both":
        dense = dense[pooling]
        codes = codes[pooling] if codes is not None else None
    if transforms:
        return dense, codes, extra, meta
    return dense, codes, meta


def match_split_occurrences(full_rows: Sequence, split_rows: Mapping[str, Sequence],
                            keep: Sequence[int]) -> dict[str, np.ndarray]:
    """Map split rows to feature rows without expanding duplicate keys.

    Split libraries commonly reset DataFrame indices.  A mapping from a SMILES
    string to *all* matching rows then turns ``k`` duplicates in the full table
    and ``k`` occurrences across the split into ``k**2`` selected rows.  This
    function instead builds one queue per row identity and consumes each source
    occurrence exactly once.  ``keep`` contains original positions that survived
    featurization; dropped occurrences are consumed but not returned.
    """
    full_rows = list(full_rows)
    keep_arr = np.asarray(keep, dtype=int)
    if keep_arr.ndim != 1:
        raise ValueError("keep must be one-dimensional")
    if len(np.unique(keep_arr)) != len(keep_arr):
        raise ValueError("keep contains duplicate source rows")
    if len(keep_arr) and (keep_arr.min() < 0 or keep_arr.max() >= len(full_rows)):
        raise IndexError("keep contains a source row outside full_rows")

    feature_row = np.full(len(full_rows), -1, dtype=int)
    feature_row[keep_arr] = np.arange(len(keep_arr), dtype=int)
    queues: dict[tuple, deque[int]] = defaultdict(deque)
    for source_i, row in enumerate(full_rows):
        queues[_identity_key(row)].append(source_i)

    selected: dict[str, np.ndarray] = {}
    assigned: set[int] = set()
    n_split_rows = 0
    for part, rows in split_rows.items():
        part_indices: list[int] = []
        for split_i, row in enumerate(rows):
            n_split_rows += 1
            key = _identity_key(row)
            if not queues[key]:
                raise ValueError(
                    f"split part {part!r} row {split_i} has no unused matching "
                    "row in the full dataset"
                )
            source_i = queues[key].popleft()
            feat_i = int(feature_row[source_i])
            if feat_i < 0:
                continue
            if feat_i in assigned:  # defensive: queue consumption should forbid it
                raise ValueError(f"feature row {feat_i} was assigned more than once")
            assigned.add(feat_i)
            part_indices.append(feat_i)
        selected[part] = np.asarray(part_indices, dtype=int)

    if n_split_rows != len(full_rows):
        raise ValueError(
            f"split contains {n_split_rows} rows but full dataset has "
            f"{len(full_rows)}; expected a complete partition"
        )
    missing = set(range(len(keep_arr))) - assigned
    if missing:
        raise ValueError(f"split omitted {len(missing)} featurized source rows")
    return selected


def _identity_key(row) -> tuple:
    """Hash-stable row identity, including a sentinel for NaN values."""
    if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
        row = (row,)
    return tuple(_identity_cell(value) for value in row)


def _identity_cell(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ("__molcircuits_nan__",)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _pool(t: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    w = mask[..., None]
    if how == "max":
        # nan_to_num(0.0) sets nan= positionally, leaving -inf to map to
        # -3.4e38; name the arguments so an all-masked row cannot produce that.
        return t.masked_fill(~w, float("-inf")).max(1).values.nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0)
    if how == "mean":
        return (t * w).sum(1) / w.sum(1).clamp_min(1)
    raise ValueError(f"unknown pooling {how!r}")


def ecfp(smiles, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Morgan fingerprints -- the baseline the SAE has to beat to matter."""
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        out[i] = np.frombuffer(gen.GetFingerprint(m).ToBitString().encode(),
                               "u1") - ord("0")
    return out
