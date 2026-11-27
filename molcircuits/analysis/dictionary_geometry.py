"""Why amplifying a dictionary feature need not steer, when a difference-in-means does.

Both are single directions in the same residual stream, injected at the same
norm, so the difference in what they do is a fact about geometry rather than
about the size of the intervention.  If the dictionary had isolated a fragment,
the decoder column of that fragment's best detector would point close to the
fragment's difference-in-means direction.  This module measures that angle and
the two things that would explain a large one:

* how well the detector actually detects -- its atlas AUROC; and
* how much of the difference-in-means direction the dictionary can express at
  all, as the norm recovered by projecting onto the decoder span.

The second is worth stating carefully, because it is easy to misread as a
positive result.  With more dictionary features than residual dimensions the
decoder columns span the whole space, so the recovered fraction is ~1 *by
construction* and shows only that no direction is unreachable in principle.  It
is the contrast that carries the finding: the dictionary can represent the
steering direction perfectly as a combination of many features while no single
detector is close to it.  A recovered fraction well below 1 would instead mean
the dictionary is rank-deficient in the relevant subspace, which is a different
and more basic failure -- so the number is reported rather than assumed.

Everything here is cheap: activations only, no generation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from ..model.hooks import HookManager, resid_post


def decoder_directions(sae) -> torch.Tensor:
    """Row-normalised decoder matrix, shape ``(d_sae, d_in)``.

    Normalised because only the *direction* of a decoder column is comparable
    with a unit steering direction; the column's length is absorbed into the
    code value at encode time and says nothing about alignment.
    """
    weights = sae.W_dec.detach().float()
    return weights / weights.norm(dim=1, keepdim=True).clamp_min(1e-9)


def _unit(direction: torch.Tensor) -> torch.Tensor:
    return direction / direction.norm().clamp_min(1e-12)


def direction_alignment(sae, direction: torch.Tensor,
                        feature: int | None = None) -> dict:
    """Cosine between a residual-stream direction and the dictionary's columns.

    Absolute cosines throughout: a decoder column that points anti-parallel to
    the steering direction is just as aligned as one pointing along it, since
    the sign is a convention of which side of the fragment contrast was
    subtracted from which.

    ``random_baseline`` is ``1/sqrt(d)``, the expected cosine between two
    independent directions.  Without it a cosine of 0.05 in 512 dimensions reads
    as "small but present" when it is in fact indistinguishable from chance.
    """
    unit = _unit(direction.detach().float())
    normalized = decoder_directions(sae)
    cosines = (normalized @ unit).abs()
    best_index = int(cosines.argmax())
    d_in = int(unit.shape[0])
    out = {
        "best_cosine_any_feature": float(cosines[best_index]),
        "best_cosine_feature": best_index,
        "random_baseline": float(1.0 / np.sqrt(d_in)),
        "d_in": d_in,
        "n_features": int(normalized.shape[0]),
    }
    if feature is not None:
        out["feature"] = int(feature)
        out["cosine_detector"] = float(cosines[int(feature)])
        out["detector_over_baseline"] = (out["cosine_detector"]
                                         * float(np.sqrt(d_in)))
    return out


def dictionary_span_recovery(sae, direction: torch.Tensor) -> dict:
    """Fraction of a unit direction's norm that lies in the decoder span.

    Computed from an orthonormal basis of the decoder columns rather than by
    solving a least-squares problem, because the dictionary is heavily
    overcomplete and the coefficient vector is not unique -- while the projection
    onto the span, which is what "can the dictionary express this at all" asks,
    is.

    The basis comes from an SVD with a rank tolerance, not from a QR
    factorisation.  Reduced QR of the wide ``(d_in, d_sae)`` decoder returns
    ``d_in`` orthonormal columns whenever there are more features than
    dimensions, *whatever the actual rank* -- so it reports a rank-deficient
    dictionary as spanning everything, which is exactly the failure this
    function exists to detect.
    """
    unit = _unit(direction.detach().float())
    weights = sae.W_dec.detach().float()
    left, singular, _ = torch.linalg.svd(weights.T, full_matrices=False)
    # Standard numerical rank cutoff, scaled by the largest singular value so
    # the threshold is relative to the dictionary's own magnitude.
    if singular.numel() == 0 or float(singular[0]) == 0.0:
        return {"recovered_norm": 0.0, "span_rank": 0, "spans_direction": False}
    tolerance = float(singular[0]) * max(weights.shape) * torch.finfo(
        singular.dtype).eps
    keep = int((singular > tolerance).sum())
    basis = left[:, :keep]
    recovered = float((basis.T @ unit).norm())
    return {
        "recovered_norm": recovered,
        "span_rank": keep,
        # ~1.0 whenever the dictionary is overcomplete and full rank; see the
        # module docstring for why that is the expected value and not a result.
        "spans_direction": bool(recovered > 0.99),
    }


def fragment_geometry(sae, direction: torch.Tensor, feature: int | None = None,
                      auroc: float | None = None) -> dict:
    """Alignment, span recovery and detection quality for one fragment."""
    out = {**direction_alignment(sae, direction, feature),
           **dictionary_span_recovery(sae, direction)}
    if auroc is not None:
        out["detector_auroc"] = float(auroc)
    return out


def alignment_versus_detection(rows: Sequence[Mapping],
                               alignment_key: str = "cosine_detector",
                               auroc_key: str = "detector_auroc") -> dict:
    """Does a better detector point more nearly along the steering direction?

    A null here is the informative outcome: it says detection quality and
    steering geometry are unrelated properties of a feature, so the dictionary
    finding a fragment is no evidence that amplifying it will produce the
    fragment.  Reported as Spearman because the sample is small and neither
    quantity is expected to be linear in the other.
    """
    pairs = [(float(r[auroc_key]), float(r[alignment_key])) for r in rows
             if r.get(auroc_key) is not None
             and r.get(alignment_key) is not None
             and np.isfinite(r[auroc_key]) and np.isfinite(r[alignment_key])]
    if len(pairs) < 4:
        # Fewer than four points cannot distinguish any monotone relation from
        # chance; returning a statistic would invite quoting one.
        return {"n": len(pairs), "spearman": None, "pvalue": None}
    from scipy.stats import spearmanr

    auroc, alignment = zip(*pairs)
    result = spearmanr(auroc, alignment)
    return {"n": len(pairs),
            "spearman": float(result.statistic),
            "pvalue": float(result.pvalue)}


@torch.no_grad()
def nonpad_residuals_by_layer(model, tok, smiles, layers, n_ctx, device, batch=64):
    """{layer: (N, d_model)} over real (non-pad) token positions."""
    out = {L: [] for L in layers}
    for s0 in range(0, len(smiles), batch):
        chunk = smiles[s0:s0 + batch]
        ids, lens = [], []
        for s in chunk:
            e = tok.encode(s)[:n_ctx]
            ids.append(e + [tok.pad_id] * (n_ctx - len(e)))
            lens.append(len(e))
        x = torch.tensor(ids, device=device)
        h = HookManager().capture(*[resid_post(L) for L in layers])
        model(x, hooks=h)
        for L in layers:
            a = h[resid_post(L)].float()
            for bi, ln in enumerate(lens):
                out[L].append(a[bi, :ln].cpu())
    return {L: torch.cat(v) for L, v in out.items()}


@torch.no_grad()
def feature_firing_rates(sae, activations, device, batch=8192):
    """Fraction of sampled non-padding positions with a positive SAE code."""
    if not len(activations):
        raise ValueError("empty token sample for dictionary firing rates")
    fired = torch.zeros(sae.cfg.d_sae, device=device)
    for block in activations.split(batch):
        fired += (sae.encode(block.to(device)) > 0).sum(0)
    return (fired / len(activations)).cpu().numpy()
