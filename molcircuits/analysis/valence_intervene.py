"""Residual-stream intervention along the valence direction, with a null.

Three fixes over the previous version, all of which a reviewer asked for:

* **alpha is expressed in units of the residual-stream RMS at the intervention
  site.** An absolute offset along a unit vector is uninterpretable without the
  norm of what it is being added to.
* **A matched random-direction null is run by default**, not as an optional flag
  that was never passed. Directions are drawn uniformly on the unit sphere and
  orthogonalized against the probe direction, so the null isolates "movement of
  this size in an arbitrary direction".
* **Probability changes and argmax flip rates are reported alongside logit
  changes**, over the full vocabulary rather than only the three bond tokens.
  Describing a 0.05-logit change as a shift in probability mass overstates it by
  about two orders of magnitude.
"""
from __future__ import annotations

import numpy as np
import torch

from ..model.hooks import HookManager, resid_post
from .valence_suffix import CachedIntervention


def random_orthogonal_directions(w: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(w)
    V = rng.standard_normal((k, d)).astype(np.float32)
    V -= np.outer(V @ w, w)                      # remove the component along w
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return V


# Distinct entropy for the covariance-matched stream: seeding it with the bare
# `seed` would reuse the isotropic null's Gaussian draws and couple the two nulls.
_COV_STREAM = 0xC0F


def activation_covariance(acts: np.ndarray) -> np.ndarray:
    """Sample covariance (d, d) of row-vector activations, in float64."""
    X = np.asarray(acts, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    return X.T @ X / max(len(X) - 1, 1)


def covariance_matched_directions(w: np.ndarray, k: int, cov: np.ndarray,
                                  seed: int = 0,
                                  eig_rtol: float = 1e-10) -> np.ndarray:
    """`k` unit directions drawn from N(0, cov), orthogonal to `w`.

    The isotropic null is weak in a residual stream whose participation ratio is
    5-61 out of 512: a uniform direction puts almost all of its norm where the
    model reads nothing, so a fitted direction can beat it just by lying in the
    used subspace. Drawing from the activation covariance puts the null's energy
    where the activations' variance is, so beating it requires more than that.

    The covariance is projected with P = I - w w^T *before* the factorization,
    i.e. draws come from N(0, P cov P). That is the same distribution as drawing
    from N(0, cov) and projecting w out, but its eigenvectors with non-zero
    eigenvalue are orthogonal to w by construction, so orthogonality does not
    rest on a subtraction of two large numbers. Eigenvalues below
    ``eig_rtol * max`` -- rank deficiency (fewer positions than d_model, dead
    dimensions) and negative round-off -- are dropped instead of square-rooted.
    """
    w = np.asarray(w, dtype=np.float64)
    w = w / (np.linalg.norm(w) + 1e-12)
    C = np.asarray(cov, dtype=np.float64)
    if C.shape != (len(w), len(w)):
        raise ValueError(f"cov has shape {C.shape}, expected {(len(w), len(w))}")
    if not np.all(np.isfinite(C)):
        raise ValueError("cov contains non-finite entries")
    C = 0.5 * (C + C.T)
    total = float(np.trace(C))
    P = np.eye(len(w)) - np.outer(w, w)
    C = P @ C @ P
    lam, U = np.linalg.eigh(C)
    top = float(lam.max()) if len(lam) else 0.0
    # Relative to the total: when all variance lies along w, projection leaves
    # round-off that is positive but is not a direction to sample.
    if not (total > 0 and top > eig_rtol * total):
        raise ValueError("covariance has no variance orthogonal to w")
    keep = lam > eig_rtol * top
    U, s = U[:, keep], np.sqrt(lam[keep])

    rng = np.random.default_rng([int(seed), _COV_STREAM])
    V = (rng.standard_normal((k, int(keep.sum()))) * s) @ U.T
    V -= np.outer(V @ w, w)                      # round-off only; see above
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-300
    return V.astype(np.float32)


def variance_ratio(V: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Activation variance along each unit row of `V`, over the mean per axis.

    1 in expectation for an isotropic direction; the participation ratio's worth
    of concentration above that for a direction in the high-variance subspace.
    """
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    C = np.asarray(cov, dtype=np.float64)
    per_axis = np.trace(C) / C.shape[0]
    return np.einsum("kd,de,ke->k", V, C, V) / (per_axis + 1e-300)


def participation_ratio(cov: np.ndarray) -> float:
    lam = np.clip(np.linalg.eigvalsh(np.asarray(cov, dtype=np.float64)), 0, None)
    return float(lam.sum() ** 2 / ((lam ** 2).sum() + 1e-300))


@torch.no_grad()
def residual_rms(model, tokens: torch.Tensor, layer: int, device: str = "cpu",
                 pad_id: int | None = None,
                 positions: torch.Tensor | None = None) -> float:
    """RMS of the residual stream, over real positions only.

    Alpha is expressed in units of this quantity, so it has to describe the sites
    actually being intervened on. Averaging over a right-padded batch instead put
    84% of the weight on pad positions -- which the packed-pretrained model never
    saw and whose residual norm is much smaller -- understating the unit by
    ~1.5x, and by a different factor at every layer, so a layer sweep was not
    comparable. Pass ``positions`` to restrict to the intervention sites exactly.
    """
    h = HookManager().capture(resid_post(layer))
    tokens = tokens.to(device)
    model(tokens, hooks=h)
    a = h[resid_post(layer)].float()
    per_pos = a.pow(2).mean(-1).sqrt()                       # (B, T)

    if positions is not None and len(positions):
        p = positions.to(device)
        return float(per_pos[p[:, 0], p[:, 1]].mean())
    if pad_id is not None:
        m = tokens != pad_id
        if bool(m.any()):
            return float(per_pos[m].mean())
    return float(per_pos.mean())


@torch.no_grad()
def logits_from_resid(model, resid: torch.Tensor, layer: int,
                      b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Logits at (b, t), continuing from the residual stream AFTER `layer`.

    The intervention is added at ``resid_post(layer)``, so blocks 0..layer do
    not depend on it: over 257 directions x 12 alphas, the old code recomputed
    them 3084 times for a tensor that is bit-identical every time. At layer 4 of
    6 that is five sixths of the model, and it dominated the stage's runtime.

    Exact, not an approximation -- the same blocks, in the same order, on the
    same inputs. ``run_with_null`` asserts it against a full forward before
    using it, so a future change to ``MolecularTransformer.forward`` that this
    function fails to mirror is caught rather than silently plotted.

    The unembedding is applied only at the requested positions. RMSNorm is
    per-position, so ``ln_f(x)[b, t] == ln_f(x[b, t])`` and the (B, T, vocab)
    logit tensor the caller immediately indexed never has to exist.
    """
    x = resid
    cos, sin = model._rope_for(x.device, x.dtype)
    for blk in model.blocks[layer + 1:]:
        x = blk(x, cos, sin)
    x = model.ln_f(x)[b, t]
    w = model.embed.weight if model.lm_head is None else model.lm_head.weight
    return x @ w.T


@torch.no_grad()
def intervene(model, tokens: torch.Tensor, positions: torch.Tensor, layer: int,
              direction: np.ndarray, alphas, bond_token_ids: dict[str, int],
              device: str = "cpu", rms: float | None = None,
              base: torch.Tensor | None = None,
              resid: torch.Tensor | None = None,
              cached: CachedIntervention | None = None) -> list[dict]:
    """Add alpha * rms * w at `positions`; report logit, probability and flips.

    positions: (N, 2) tensor of (batch_index, token_index).
    alphas are multiples of the residual RMS, so alpha=1 injects a perturbation
    whose norm is that RMS. (Note this is the per-element RMS: a residual VECTOR
    has norm rms*sqrt(d_model), so alpha=1 is a few percent of the residual's
    own length, not comparable to it.)

    ``base`` and ``resid`` are the unperturbed logits at `positions` and the
    unperturbed residual stream after `layer`. Neither depends on the direction
    being tested, so a caller sweeping hundreds of directions computes them once
    and passes them in; without them this function re-derives both from
    `tokens`, which is what a direct caller or a test wants.
    """
    tokens = tokens.to(device)
    w = torch.tensor(direction, dtype=torch.float32, device=device)
    if rms is None:
        # Measured at the intervention sites. Defaulting to the whole padded
        # batch understates the unit by ~1.5x, and by a different factor per
        # layer, which makes a layer sweep incomparable.
        rms = residual_rms(model, tokens, layer, device, positions=positions)

    b, t = positions[:, 0].to(device), positions[:, 1].to(device)
    if cached is None:
        if resid is None:
            h = HookManager().capture(resid_post(layer))
            model(tokens, hooks=h)
            resid = h[resid_post(layer)]
        if base is None:
            base = logits_from_resid(model, resid, layer, b, t).float()
        w = w.to(resid.dtype)
    else:
        w = w.to(cached.batches[0].activations.dtype)
        if base is None:
            base = cached.logits(torch.zeros_like(w))
    base_probs = base.softmax(-1)
    base_arg = base.argmax(-1)

    rows = []
    for a in alphas:
        # Edit the cached residual directly instead of re-running the prefix
        # under a hook: the hook path recomputed blocks 0..layer to reproduce a
        # tensor already in hand. See logits_from_resid.
        delta = (a * rms) * w
        if cached is None:
            x = resid.clone()
            x[b, t] = x[b, t] + delta
            lg = logits_from_resid(model, x, layer, b, t).float()
        else:
            lg = cached.logits(delta)
        probs = lg.softmax(-1)
        row = {"alpha_rms_units": float(a), "alpha_absolute": float(a * rms),
               "n_positions": int(len(b)),
               "argmax_flip_rate": float((lg.argmax(-1) != base_arg).float().mean())}
        for name, tid in bond_token_ids.items():
            row[f"dlogit_{name}"] = float((lg[:, tid] - base[:, tid]).mean())
            row[f"dprob_{name}"] = float((probs[:, tid] - base_probs[:, tid]).mean())
        rows.append(row)
    return rows


@torch.no_grad()
def run_with_null(model, tokens, positions, layer, direction, alphas,
                  bond_token_ids, device="cpu", n_null: int = 32,
                  seed: int = 0, *, cache_prefixes: bool | None = None) -> dict:
    """Probe direction plus `n_null` matched random directions.

    Everything the probe direction is credited with should be read against the
    null distribution reported here.

    ``rms`` is measured at the INTERVENTION SITES, not over the whole padded
    batch. This call used to omit ``positions``, which is the exact failure
    ``residual_rms`` documents: a right-padded batch puts most of the weight on
    pad positions, whose residual norm is much smaller, so the unit came out
    ~1.5x too small and by a different factor at every layer. Since alpha is
    quoted in multiples of this unit, every effect size in the section depended
    on it.

    Set ``cache_prefixes=False`` to use the dense continuation on any device.
    CUDA defaults to reusing unedited attention prefixes across directions and
    strengths. Every decision in a molecule receives its edit simultaneously.
    """
    # Neither the residual stream before the injection site nor the unperturbed
    # logits depend on the direction being tested, so both are computed ONCE for
    # all n_null+1 calls below. With n_null=256 and 12 alphas that is the
    # difference between 3341 forward passes through the whole model and 3085
    # passes through the one block that follows the injection.
    tokens = tokens.to(device)
    positions = positions.to(device)
    rms = residual_rms(model, tokens, layer, device, positions=positions)
    if cache_prefixes is None:
        cache_prefixes = tokens.is_cuda
    cached = None
    resid = None
    if cache_prefixes:
        cached = CachedIntervention(model, tokens, positions, "post", layer, batch_size=128)
        full = cached.base_logits
        base = cached.logits(torch.zeros_like(cached.edit_activations[0]))
    else:
        b, t = positions[:, 0].to(device), positions[:, 1].to(device)
        h = HookManager().capture(resid_post(layer))
        full = model(tokens, hooks=h)[b, t].float()
        resid = h[resid_post(layer)]
        base = logits_from_resid(model, resid, layer, b, t).float()
    # The partial-forward path must agree with the whole model, or every number
    # below is measured against a baseline the model does not produce. fp32
    # reduction order differs between the two, so this is a tolerance and not an
    # equality -- but a real mismatch (a block skipped, ln_f dropped, the
    # untied head ignored) is orders of magnitude larger than 1e-3.
    gap = float((base - full).abs().max())
    if gap > 1e-3 * max(1.0, float(full.abs().max())):
        raise RuntimeError(
            f"intervention continuation disagrees with the full model by {gap:.3e}")

    real = intervene(model, tokens, positions, layer, direction, alphas,
                     bond_token_ids, device, rms, base, resid, cached)

    nulls = []
    for v in random_orthogonal_directions(direction, n_null, seed):
        nulls.append(intervene(model, tokens, positions, layer, v, alphas,
                               bond_token_ids, device, rms, base, resid, cached))

    keys = [k for k in real[0] if k.startswith(("dlogit_", "dprob_", "argmax"))]
    null_summary = []
    for j, a in enumerate(alphas):
        entry = {"alpha_rms_units": float(a)}
        for k in keys:
            vals = np.array([n[j][k] for n in nulls], dtype=float)
            entry[f"{k}_null_mean"] = float(vals.mean())
            entry[f"{k}_null_std"] = float(vals.std())
            entry[f"{k}_null_p95_abs"] = float(np.percentile(np.abs(vals), 95))
            obs = real[j][k]
            entry[f"{k}_z_vs_null"] = float(
                (obs - vals.mean()) / (vals.std() + 1e-12))
        null_summary.append(entry)

    return {"residual_rms": rms, "n_null_directions": n_null,
            "alphas_in_rms_units": [float(a) for a in alphas],
            "probe_direction": real, "null": null_summary,
            "note": ("alpha is a multiple of the residual-stream RMS at the "
                     "intervention site; null directions are unit vectors "
                     "orthogonal to the probe direction")}
