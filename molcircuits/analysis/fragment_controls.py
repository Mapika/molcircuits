"""Matched controls for held-out, coordinate-level fragment alignment.

Schema v1 is the historical pooled atom-token AUROC screen.  Schema v2 uses the
same within-molecule standardized difference as stage 06, but fixes a coordinate
and (for its primary signed detector) an orientation on selection molecules and
evaluates that exact detector on other molecules.  The schemas remain explicit
because pooled discrimination and within-molecule localization are different
estimands even when they use the same molecule split and representations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .fragment_families import family_map
from .wsd import basis_stats

SCHEMA_VERSION = 1
WSD_SCHEMA_VERSION = 2

def matched_arm_names(d_model: int, d_sae: int, k: int) -> tuple[str, ...]:
    """Canonical names for controls with the checkpoint's width and sparsity."""
    if not 0 < k <= min(d_model, d_sae):
        raise ValueError("TopK must fit both the dense and sparse bases")
    return (f"learned_sae_topk{k}_{d_sae}", f"dense_neurons_{d_model}",
            f"dense_standardized_topk{k}_{d_model}", f"random_relu_topk{k}_{d_sae}")


ARMS = matched_arm_names(512, 8192, 32)

#: Arms of the schema-v2 within-molecule WSD screen. The extra arm is not a
#: representation of the model at all: a one-hot over current token identity,
#: carrying atom type and nothing else. It is the explicit form of what the
#: composition-matched null removes, and it states in the same units how much of
#: a fragment score is reachable without encoding substructure membership.
WSD_ARMS = (*ARMS, "atom_type_onehot")


def fit_selection_standardizer(selection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a per-neuron affine standardizer on selection rows only.

    Population moments (``ddof=0``) are used because this is a deterministic
    transform rather than an estimate reported with uncertainty. Near-constant
    coordinates receive a scale floor tied to the typical neuron scale.
    """
    selection = np.asarray(selection)
    if selection.ndim != 2 or not len(selection):
        raise ValueError("selection activations must be a non-empty 2-D array")
    mean = selection.mean(axis=0, dtype=np.float64)
    scale = selection.std(axis=0, dtype=np.float64)
    floor = max(1e-8, 1e-6 * float(np.mean(scale)))
    scale = np.maximum(scale, floor)
    return mean.astype(np.float32), scale.astype(np.float32)


def relu_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    """The same ReLU-then-TopK nonlinearity used by the learned SAE."""
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("x must have a non-empty coordinate axis")
    if k < 1:
        raise ValueError("k must be positive")
    z = F.relu(x)
    k = min(int(k), z.shape[-1])
    threshold = z.topk(k, dim=-1).values[..., -1:]
    return z * (z >= threshold)


def random_unit_projection(d_in: int, d_out: int, seed: int,
                           device: str | torch.device = "cpu") -> torch.Tensor:
    """Draw reproducible Gaussian directions and normalize every output column.

    Sampling happens on CPU so the same seed identifies the same matrix on CPU
    and CUDA. Unit column norms make the TopK competition insensitive to random
    variation in direction length.
    """
    if d_in < 1 or d_out < 1:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    weights = torch.randn(d_in, d_out, generator=generator, dtype=torch.float32)
    weights /= weights.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return weights.to(device)


def rank_coordinates(activations: np.ndarray, chunk: int = 256) -> dict:
    """Midrank each coordinate once for repeated fragment AUROC calculations.

    Ranks are float32 and computed in coordinate chunks, so peak temporary
    memory is ``O(n_rows * chunk)`` rather than ``O(n_rows * n_coordinates)``
    beyond the rank matrix itself. A coordinate is dead only when it is
    constant; varying all-negative dense neurons remain evaluable.
    """
    from scipy.stats import rankdata  # noqa: PLC0415

    acts = np.asarray(activations)
    if acts.ndim != 2 or not len(acts) or not acts.shape[1]:
        raise ValueError("activations must be a non-empty 2-D array")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    n_rows, n_coordinates = acts.shape
    ranks = np.empty((n_rows, n_coordinates), dtype=np.float32)
    for start in range(0, n_coordinates, chunk):
        end = min(start + chunk, n_coordinates)
        block = acts[:, start:end]
        if not np.isfinite(block).all():
            raise ValueError("activations contain non-finite values")
        ranks[:, start:end] = rankdata(
            block, method="average", axis=0)
    dead = acts.max(axis=0) == acts.min(axis=0)
    return {"ranks": ranks, "dead": dead, "n": int(n_rows)}


def coordinate_aurocs(ranks: dict, labels: np.ndarray) -> np.ndarray:
    """Mann-Whitney AUROC for every coordinate from a midrank transform."""
    labels = np.asarray(labels, dtype=bool)
    if labels.shape != (ranks["n"],):
        raise ValueError("label count does not match ranked activation rows")
    n_positive = int(labels.sum())
    n_negative = int(ranks["n"] - n_positive)
    if n_positive == 0 or n_negative == 0:
        return np.full(ranks["ranks"].shape[1], np.nan, dtype=np.float64)
    rank_sum = ranks["ranks"][labels].sum(axis=0, dtype=np.float64)
    auc = ((rank_sum - n_positive * (n_positive + 1) / 2.0)
           / (n_positive * n_negative))
    auc[np.asarray(ranks["dead"], dtype=bool)] = np.nan
    return auc


def heldout_coordinate_alignment(selection_ranks: dict, evaluation_ranks: dict,
                                 selection_labels: np.ndarray,
                                 evaluation_labels: np.ndarray) -> dict:
    """Select a sign-invariant detector, then evaluate it without leakage.

    For coordinate ``j`` on the selection molecules, let ``A_j`` be token-level
    AUROC for fragment atoms versus all other atom tokens. Its sign-invariant
    alignment is ``0.5 + abs(A_j - 0.5)``. We select the maximizing coordinate
    and fix its orientation from the sign of ``A_j - 0.5``. On evaluation
    molecules we report the AUROC of that *same coordinate in that fixed
    orientation*: ``A_eval`` for a positive selection orientation and
    ``1 - A_eval`` for a negative one. No absolute value or re-selection is
    applied after evaluation labels are observed. Multiplying any coordinate by
    -1 therefore leaves the full statistic unchanged, while a detector whose
    direction reverses out of sample is correctly allowed to score below 0.5.
    """
    selection_labels = np.asarray(selection_labels, dtype=bool)
    evaluation_labels = np.asarray(evaluation_labels, dtype=bool)
    if selection_ranks["ranks"].shape[1] != evaluation_ranks["ranks"].shape[1]:
        raise ValueError("selection and evaluation coordinate counts differ")
    selection_auc = coordinate_aurocs(selection_ranks, selection_labels)
    if not np.isfinite(selection_auc).any():
        return {
            "status": "insufficient_selection_variation",
            "selected_coordinate": None,
            "n_coordinates": int(selection_ranks["ranks"].shape[1]),
            "n_selection_positions": int(selection_ranks["n"]),
            "n_evaluation_positions": int(evaluation_ranks["n"]),
        }
    alignment = 0.5 + np.abs(selection_auc - 0.5)
    selected = int(np.nanargmax(alignment))
    orientation = 1 if selection_auc[selected] >= 0.5 else -1

    evaluation_auc = coordinate_aurocs(evaluation_ranks, evaluation_labels)
    raw_evaluation = float(evaluation_auc[selected])
    if not np.isfinite(raw_evaluation):
        status = "selected_coordinate_not_evaluable"
        heldout = None
        raw_evaluation = None
        if bool(np.asarray(evaluation_ranks["dead"])[selected]):
            unavailable_reason = "selected_coordinate_constant_on_evaluation"
        elif not evaluation_labels.any() or evaluation_labels.all():
            unavailable_reason = "evaluation_labels_have_one_class"
        else:
            unavailable_reason = "evaluation_auroc_not_finite"
    else:
        status = "ok"
        heldout = raw_evaluation if orientation > 0 else 1.0 - raw_evaluation
        unavailable_reason = None

    return {
        "status": status,
        "selected_coordinate": selected,
        "orientation": "positive" if orientation > 0 else "negative",
        "orientation_multiplier": orientation,
        "selection_raw_auroc": float(selection_auc[selected]),
        "selection_alignment": float(alignment[selected]),
        # JSON null is deliberate for an unavailable held-out estimand.  A NaN
        # is neither portable JSON nor a numerical result and must never be
        # mistaken for chance performance (0.5).
        "evaluation_raw_auroc": raw_evaluation,
        "evaluation_alignment": heldout,
        "evaluation_unavailable_reason": unavailable_reason,
        "n_coordinates": int(selection_ranks["ranks"].shape[1]),
        "n_selection_positions": int(selection_ranks["n"]),
        "n_evaluation_positions": int(evaluation_ranks["n"]),
        "n_selection_positive_positions": int(selection_labels.sum()),
        "n_evaluation_positive_positions": int(evaluation_labels.sum()),
    }


def _coordinate_vectors(*vectors: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(vector, dtype=np.float64) for vector in vectors)
    if not arrays or arrays[0].ndim != 1 or not len(arrays[0]):
        raise ValueError("coordinate scores must be non-empty 1-D arrays")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("coordinate score arrays must have the same shape")
    return arrays


def heldout_wsd_detectors(selection_signed: np.ndarray,
                          selection_absolute: np.ndarray,
                          evaluation_signed: np.ndarray,
                          evaluation_absolute: np.ndarray) -> dict:
    """Select within-molecule WSD detectors and score fixed held-out columns.

    The primary detector selects ``argmax(abs(mean_b d_bj))`` on selection
    molecules and fixes the sign of that mean.  Evaluation reports that sign
    times ``mean_b d_bj`` for the same coordinate, with no re-selection or
    evaluation-side absolute value.  The secondary detector mirrors stage 06's
    absolute statistic: select ``argmax(mean_b abs(d_bj))`` and evaluate
    ``mean_b abs(d_bj)`` for that same coordinate.
    """
    sel_signed, sel_absolute, eval_signed, eval_absolute = _coordinate_vectors(
        selection_signed, selection_absolute,
        evaluation_signed, evaluation_absolute)
    n_coordinates = int(len(sel_signed))
    if not np.isfinite(sel_signed).any() or not np.isfinite(sel_absolute).any():
        return {
            "status": "insufficient_selection_variation",
            "n_coordinates": n_coordinates,
            "oriented_detector": {"selected_coordinate": None},
            "absolute_detector": {"selected_coordinate": None},
        }

    oriented_coordinate = int(np.nanargmax(np.abs(sel_signed)))
    orientation = 1 if sel_signed[oriented_coordinate] >= 0 else -1
    absolute_coordinate = int(np.nanargmax(sel_absolute))
    raw_evaluation = float(eval_signed[oriented_coordinate])
    absolute_evaluation = float(eval_absolute[absolute_coordinate])
    status = ("ok" if np.isfinite(raw_evaluation)
              and np.isfinite(absolute_evaluation)
              else "selected_coordinate_not_evaluable")
    return {
        "status": status,
        "n_coordinates": n_coordinates,
        "oriented_detector": {
            "selected_coordinate": oriented_coordinate,
            "orientation": "positive" if orientation > 0 else "negative",
            "orientation_multiplier": orientation,
            "selection_mean_signed_wsd": float(sel_signed[oriented_coordinate]),
            "selection_alignment": float(abs(sel_signed[oriented_coordinate])),
            "evaluation_mean_signed_wsd": raw_evaluation,
            "evaluation_oriented_wsd": float(orientation * raw_evaluation),
        },
        "absolute_detector": {
            "selected_coordinate": absolute_coordinate,
            "selection_mean_absolute_wsd": float(
                sel_absolute[absolute_coordinate]),
            "evaluation_mean_absolute_wsd": absolute_evaluation,
        },
    }


def _pipeline_null_summary(values: list[float], requested: int,
                           observed: float, selection_seed: int,
                           evaluation_seed: int,
                           method: str = "within-molecule whole-pipeline permutation") -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    mean = float(finite.mean()) if len(finite) else float("nan")
    p95 = float(np.percentile(finite, 95)) if len(finite) else float("nan")
    return {
        "method": method,
        "n_perm_requested": int(requested),
        "n_perm_valid": int(len(finite)),
        "selection_seed": int(selection_seed),
        "evaluation_seed": int(evaluation_seed),
        "mean": mean,
        "p95": p95,
        "observed_exceeds_p95": bool(
            len(finite) and np.isfinite(observed) and observed > p95),
    }


def heldout_wsd_pipeline_null(selection_signed: np.ndarray,
                              selection_absolute: np.ndarray,
                              evaluation_signed: np.ndarray,
                              evaluation_absolute: np.ndarray,
                              observed: dict, selection_seed: int,
                              evaluation_seed: int,
                              method: str = "within-molecule whole-pipeline permutation") -> dict:
    """Score null detectors selected on one split on independent null labels.

    Each row is one whole-pipeline permutation and each column a coordinate.
    The null coordinate and signed orientation are selected from that row's
    selection scores, then applied unchanged to the corresponding evaluation
    row.  No draw-level values are returned, keeping result artifacts compact.
    """
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (
        selection_signed, selection_absolute,
        evaluation_signed, evaluation_absolute))
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("pipeline-null scores must be 2-D")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("pipeline-null score arrays must have the same shape")
    oriented_values, absolute_values = [], []
    for sel_signed, sel_absolute, eval_signed, eval_absolute in zip(*arrays):
        if np.isfinite(sel_signed).any():
            coordinate = int(np.nanargmax(np.abs(sel_signed)))
            if np.isfinite(eval_signed[coordinate]):
                orientation = 1 if sel_signed[coordinate] >= 0 else -1
                oriented_values.append(float(orientation * eval_signed[coordinate]))
        if np.isfinite(sel_absolute).any():
            coordinate = int(np.nanargmax(sel_absolute))
            if np.isfinite(eval_absolute[coordinate]):
                absolute_values.append(float(eval_absolute[coordinate]))
    oriented_observed = observed.get("oriented_detector", {}).get(
        "evaluation_oriented_wsd", np.nan)
    absolute_observed = observed.get("absolute_detector", {}).get(
        "evaluation_mean_absolute_wsd", np.nan)
    return {
        "oriented_detector": _pipeline_null_summary(
            oriented_values, arrays[0].shape[0], float(oriented_observed),
            selection_seed, evaluation_seed, method),
        "absolute_detector": _pipeline_null_summary(
            absolute_values, arrays[0].shape[0], float(absolute_observed),
            selection_seed, evaluation_seed, method),
    }


def permute_fragment_masks_within_molecules(fragment_mask: np.ndarray,
                                             valid_mask: np.ndarray,
                                             n_perm: int,
                                             seed: int) -> np.ndarray:
    """Shuffle fragment membership within each molecule, preserving coverage."""
    fragment_mask = np.asarray(fragment_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if fragment_mask.shape != valid_mask.shape or fragment_mask.ndim != 2:
        raise ValueError("fragment_mask and valid_mask must share shape (B, T)")
    if n_perm < 0:
        raise ValueError("n_perm must be non-negative")
    rng = np.random.default_rng(seed)
    shuffled = np.zeros((fragment_mask.shape[0], n_perm,
                         fragment_mask.shape[1]), dtype=bool)
    for molecule in range(fragment_mask.shape[0]):
        valid = np.flatnonzero(valid_mask[molecule])
        n_positive = int((fragment_mask[molecule] & valid_mask[molecule]).sum())
        if n_positive:
            for draw in range(n_perm):
                selected = rng.choice(valid, size=n_positive, replace=False)
                shuffled[molecule, draw, selected] = True
    return shuffled


def shift_fragment_masks_within_molecules(fragment_mask: np.ndarray,
                                          valid_mask: np.ndarray,
                                          n_perm: int,
                                          seed: int) -> np.ndarray:
    """Relocate each fragment footprint without changing its shape.

    The uniform permutation null preserves how *many* positions a fragment
    covers and nothing else. Real SMARTS matches are not scattered: they fall in
    runs, because the atoms of a substructure are usually written consecutively.
    A coordinate that responds to any contiguous chemical context therefore beats
    the uniform null without detecting the fragment, which is one reason all four
    bases clear it on every fragment.

    Here the mask over valid positions is rotated by a random offset, so the
    exact pattern of runs and gaps survives and only its location moves. What
    remains discriminable is fragment identity rather than fragment shape.
    """
    fragment_mask = np.asarray(fragment_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if fragment_mask.shape != valid_mask.shape or fragment_mask.ndim != 2:
        raise ValueError("fragment_mask and valid_mask must share shape (B, T)")
    if n_perm < 0:
        raise ValueError("n_perm must be non-negative")
    rng = np.random.default_rng(seed)
    shifted = np.zeros((fragment_mask.shape[0], n_perm,
                        fragment_mask.shape[1]), dtype=bool)
    for molecule in range(fragment_mask.shape[0]):
        valid = np.flatnonzero(valid_mask[molecule])
        if len(valid) < 2:
            continue
        pattern = fragment_mask[molecule, valid]
        if not pattern.any():
            continue
        for draw in range(n_perm):
            offset = int(rng.integers(1, len(valid)))
            shifted[molecule, draw, valid] = np.roll(pattern, offset)
    return shifted


def type_matched_fragment_masks(fragment_mask: np.ndarray,
                                valid_mask: np.ndarray,
                                token_ids: np.ndarray,
                                n_perm: int,
                                seed: int) -> tuple[np.ndarray, dict]:
    """Resample a fragment's positions holding its token-type multiset fixed.

    The strongest cheap explanation for a fragment detector is that it is an
    element detector: an ester match is mostly oxygens, so a coordinate that
    responds to oxygen scores well on it against any null that does not control
    composition. This null draws, for each token type in the real match, the
    same number of positions of that same type elsewhere in the molecule. A
    coordinate that only knows atom types is then at chance, and only
    information about which *particular* atoms form the substructure survives.

    Molecules without enough same-type positions to fill the draw take as many
    as exist; the shortfall is returned rather than absorbed, because a null
    that quietly shrinks its positive set is easier to beat than the one that
    was specified.
    """
    fragment_mask = np.asarray(fragment_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    token_ids = np.asarray(token_ids)
    if fragment_mask.shape != valid_mask.shape or fragment_mask.ndim != 2:
        raise ValueError("fragment_mask and valid_mask must share shape (B, T)")
    if token_ids.shape != fragment_mask.shape:
        raise ValueError("token_ids must share shape with fragment_mask")
    if n_perm < 0:
        raise ValueError("n_perm must be non-negative")
    rng = np.random.default_rng(seed)
    out = np.zeros((fragment_mask.shape[0], n_perm, fragment_mask.shape[1]),
                   dtype=bool)
    n_requested = n_short = 0
    for molecule in range(fragment_mask.shape[0]):
        positive = fragment_mask[molecule] & valid_mask[molecule]
        if not positive.any():
            continue
        types, counts = np.unique(token_ids[molecule][positive],
                                  return_counts=True)
        for draw in range(n_perm):
            for tid, need in zip(types, counts, strict=True):
                pool = np.flatnonzero(valid_mask[molecule]
                                      & (token_ids[molecule] == tid))
                n_requested += int(need)
                take = min(int(need), len(pool))
                n_short += int(need) - take
                if take:
                    out[molecule, draw,
                        rng.choice(pool, size=take, replace=False)] = True
    return out, {"positions_requested": int(n_requested),
                 "positions_short": int(n_short),
                 "shortfall_fraction": (n_short / n_requested) if n_requested else 0.0}


def within_type_eligible_molecules(fragment_mask: np.ndarray,
                                   valid_mask: np.ndarray,
                                   token_ids: np.ndarray) -> np.ndarray:
    """Molecules where some token type occurs both inside and outside a match.

    Only these molecules carry a within-type contrast. A molecule whose
    fragment atoms are all of types absent from the rest of it (toluene for
    benzene: every ``c`` is in the ring) says nothing about context beyond
    token identity, however strongly a coordinate separates it.
    """
    fragment_mask = np.asarray(fragment_mask, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    if fragment_mask.shape != valid_mask.shape or fragment_mask.ndim != 2:
        raise ValueError("fragment_mask and valid_mask must share shape (B, T)")
    if token_ids.shape != fragment_mask.shape:
        raise ValueError("token_ids must share shape with fragment_mask")
    if not valid_mask.any():
        return np.zeros(fragment_mask.shape[0], dtype=bool)
    if token_ids[valid_mask].min() < 0:
        raise ValueError("token_ids must be non-negative at valid positions")
    n_types = int(token_ids[valid_mask].max()) + 1
    inside = np.zeros((fragment_mask.shape[0], n_types), dtype=bool)
    outside = np.zeros_like(inside)
    positive = fragment_mask & valid_mask
    negative = valid_mask & ~fragment_mask
    rows, cols = np.nonzero(positive)
    inside[rows, token_ids[rows, cols]] = True
    rows, cols = np.nonzero(negative)
    outside[rows, token_ids[rows, cols]] = True
    return (inside & outside).any(axis=1)


def _as_tensor(a, device: str, dtype) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(a)).to(device=device, dtype=dtype)


def within_type_wsd_coordinates(acts, masks, valid_mask, token_ids,
                                stats: dict | None = None, chunk: int = 256,
                                mask_batch: int = 25
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Molecule-mean within-token-type WSD for each mask and coordinate.

    The primary WSD is the within-molecule point-biserial correlation of a
    coordinate ``x`` with the fragment label ``y``. A coordinate that encodes
    only the current token's identity scores well on it whenever a fragment is
    enriched for some atom type, so that score cannot separate "knows the atom
    symbol" from "knows the atom's chemical context". This estimand removes
    token identity from the label before correlating:

        c_i  = y_i - p_T(i),   p_T = n_in,T / n_T  (share of type-T atoms in the match)
        d_bj = sum_i c_i x_ij / (sigma_bj * sqrt(n_b * sum_i c_i**2))

    over the molecule's ``n_b`` atom tokens, with ``sigma_bj`` the same
    within-molecule SD of coordinate ``j`` the primary estimand divides by.
    This is the semipartial correlation of ``x`` with ``y`` given token type.
    Written per type, its numerator is ``sum_T w_T (mu_in,T - mu_out,T)`` with
    the Cochran-Mantel-Haenszel weight ``w_T = n_in,T * n_out,T / n_T``, and
    ``sqrt(sum_T w_T / n_b)`` replaces the primary ``sqrt(p(1-p))``: the
    in-type versus out-of-type mean differences are pooled across token types
    weighted by how much contrast each type offers. Types with atoms only
    inside or only outside the match have ``c = 0`` and drop out. With a single
    token type in the molecule this is exactly the primary WSD.

    ``sigma`` is the whole-molecule SD rather than the within-type residual SD
    so the two estimands share units and a stratum of two atoms never sets the
    scale. The numerator is computed on type-demeaned activations, which is the
    same sum mathematically but makes it exactly zero, not merely rounded to
    zero, for a 0/1 coordinate that is constant within each token type -- the
    ``atom_type_onehot`` arm -- and zero up to float32 rounding of the type
    mean for any other type-constant coordinate.

    A molecule contributes only if some token type has atoms both inside and
    outside the match (see :func:`within_type_eligible_molecules`); eligible
    molecules are weighted equally. Returns mean signed and mean absolute
    scores with shape ``(M, K)`` (NaN where no molecule is eligible) and the
    eligible-molecule count per mask.
    """
    masks3d = np.asarray(masks, dtype=bool)
    if masks3d.ndim == 2:
        masks3d = masks3d[:, None, :]
    if masks3d.ndim != 3:
        raise ValueError("masks must have shape (B, T) or (B, M, T)")
    B, T, K = acts.shape
    if tuple(masks3d.shape[::2]) != (B, T):
        raise ValueError("mask molecule/token axes do not match activations")
    valid = np.asarray(valid_mask, dtype=bool)
    ids = np.asarray(token_ids, dtype=np.int64)
    if valid.shape != (B, T) or ids.shape != (B, T):
        raise ValueError("valid_mask and token_ids must match activations")
    if chunk < 1 or mask_batch < 1:
        raise ValueError("chunk and mask_batch must be positive")
    st = basis_stats(acts, valid, chunk) if stats is None else stats
    dev = st["device"]
    M = masks3d.shape[1]
    acc_signed = torch.zeros((M, K), dtype=torch.float64, device=dev)
    acc_absolute = torch.zeros((M, K), dtype=torch.float64, device=dev)
    count = torch.zeros(M, dtype=torch.float64, device=dev)

    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        chunk_valid = valid[s:e]
        if not chunk_valid.any():
            continue
        # Types are re-indexed per chunk so the (b, types, K) temporary scales
        # with the types present, not with the vocabulary.
        types = np.unique(ids[s:e][chunk_valid])
        local = np.searchsorted(types, ids[s:e]).clip(0, len(types) - 1)
        local = np.where(chunk_valid, local, 0)
        b = e - s
        a = _as_tensor(acts[s:e], dev, torch.float32)
        v = _as_tensor(chunk_valid, dev, torch.float32)
        local_t = torch.as_tensor(local, device=dev)
        onehot = F.one_hot(local_t, len(types)).float() * v[..., None]   # (b,T,V)
        n_type = onehot.sum(1)                                          # (b,V)
        type_mean = torch.bmm(onehot.transpose(1, 2), a) / n_type.clamp_min(
            1.0)[..., None]                                             # (b,V,K)
        rows = torch.arange(b, device=dev)[:, None]
        residual = a - type_mean[rows, local_t]                         # (b,T,K)
        del type_mean
        sigma = st["sigma"][s:e]                                        # (b,1,K)
        n_all = st["n_all"][s:e]                                        # (b,1,1)

        for m0 in range(0, M, mask_batch):
            m1 = min(m0 + mask_batch, M)
            pos = _as_tensor(masks3d[s:e, m0:m1], dev, torch.float32) * v[:, None, :]
            n_in = torch.bmm(pos, onehot)                               # (b,m,V)
            eligible = ((n_in > 0) & (n_in < n_type[:, None, :])).any(-1)
            share = n_in / n_type[:, None, :].clamp_min(1.0)
            share_at = torch.gather(
                share, 2, local_t[:, None, :].expand(-1, m1 - m0, -1))
            c = (pos - share_at) * v[:, None, :]                        # (b,m,T)
            numerator = torch.bmm(c, residual)                          # (b,m,K)
            label_ss = (c * c).sum(-1, keepdim=True)                    # (b,m,1)
            denominator = sigma * torch.sqrt(n_all * label_ss)
            d = torch.where(eligible[..., None] & (denominator > 0),
                            numerator / denominator.clamp_min(1e-30),
                            torch.zeros((), device=dev))
            acc_signed[m0:m1] += d.sum(0, dtype=torch.float64)
            acc_absolute[m0:m1] += d.abs().sum(0, dtype=torch.float64)
            count[m0:m1] += eligible.sum(0, dtype=torch.float64)

    missing = (count <= 0)[:, None]
    mean_signed = torch.where(
        missing, torch.full_like(acc_signed, float("nan")),
        acc_signed / count.clamp_min(1.0)[:, None])
    mean_absolute = torch.where(
        missing, torch.full_like(acc_absolute, float("nan")),
        acc_absolute / count.clamp_min(1.0)[:, None])
    return (mean_signed.cpu().numpy(), mean_absolute.cpu().numpy(),
            count.cpu().numpy().astype(np.int64))


def heldout_within_type_detectors(selection_signed: np.ndarray,
                                  selection_absolute: np.ndarray,
                                  evaluation_signed: np.ndarray,
                                  evaluation_absolute: np.ndarray,
                                  n_selection_eligible: int,
                                  n_evaluation_eligible: int) -> dict:
    """Held-out within-type detectors, selected exactly as the primary ones.

    The selection rule and the fixed evaluation are those of
    :func:`heldout_wsd_detectors`, applied to within-type scores. A basis whose
    every coordinate is constant within token type scores exactly zero on
    selection; its detector is then an arbitrary column (the first), which is
    flagged so that it is not read as a detector shared across fragments.
    """
    result = heldout_wsd_detectors(selection_signed, selection_absolute,
                                   evaluation_signed, evaluation_absolute)
    signed = np.asarray(selection_signed, dtype=np.float64)
    finite = signed[np.isfinite(signed)]
    result["selection_signal_identically_zero"] = bool(
        len(finite) and not np.any(finite))
    result["n_selection_molecules_eligible"] = int(n_selection_eligible)
    result["n_evaluation_molecules_eligible"] = int(n_evaluation_eligible)
    return result


def detector_sharing(fragments: dict, arms, detector: str = "oriented_detector",
                     estimand_key: str | None = None) -> dict:
    """How many distinct coordinates serve the fragments, and which are reused.

    A screen that reports one detector per fragment reads as many detectors;
    when one coordinate is selected for several fragments it is one detector
    scored several times, and those fragment scores are not independent
    evidence about the basis. Fragments whose selection signal is identically
    zero (the one-hot arm under the within-type estimand) select an arbitrary
    column and are counted separately rather than as sharing it.
    """
    out = {}
    for arm in arms:
        by_coordinate: dict[int, list[str]] = {}
        n_zero = 0
        for name in sorted(fragments):
            result = fragments[name].get("arms", {}).get(arm)
            if result is None:
                continue
            if estimand_key is not None:
                result = result.get(estimand_key)
                if result is None:
                    continue
            if result.get("selection_signal_identically_zero"):
                n_zero += 1
                continue
            coordinate = result.get(detector, {}).get("selected_coordinate")
            if coordinate is None:
                continue
            by_coordinate.setdefault(int(coordinate), []).append(name)
        shared = sorted(
            ({"coordinate": coordinate, "n_fragments": len(names),
              "fragments": names}
             for coordinate, names in by_coordinate.items() if len(names) > 1),
            key=lambda row: (-row["n_fragments"], row["coordinate"]))
        out[arm] = {
            "n_fragments_with_selected_coordinate": int(sum(
                len(names) for names in by_coordinate.values())),
            "n_distinct_coordinates": len(by_coordinate),
            "n_fragments_without_selection_signal": int(n_zero),
            "shared_coordinates": shared,
        }
    return out


def fragment_family_cluster_bootstrap(differences: dict[str, float],
                                      n_bootstrap: int, seed: int) -> dict:
    """Cluster-bootstrap a mean contrast by resampling SMARTS families.

    Every sampled family contributes all of its fragments, matching the stage-15
    estimator. This preserves within-family dependence instead of pretending,
    for example, that pyridine and pyrimidine are independent targets.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    names = sorted(differences)
    if not names:
        return {"n_fragments": 0, "n_families": 0,
                "n_bootstrap": int(n_bootstrap), "seed": int(seed),
                "ci95": [float("nan"), float("nan")]}
    families = family_map(names)
    family_names = sorted(set(families.values()))
    by_family = {family: [name for name in names if families[name] == family]
                 for family in family_names}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        selected = rng.integers(0, len(family_names), size=len(family_names))
        values = [differences[name]
                  for index in selected
                  for name in by_family[family_names[int(index)]]]
        draws[draw] = np.mean(values)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "method": "fragment-family cluster bootstrap",
        "n_fragments": len(names),
        "n_families": len(family_names),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "ci95": [float(low), float(high)],
    }


def aggregate_fragment_results(fragments: dict, arms=ARMS,
                               reference: str = "learned_sae_topk32_8192",
                               n_bootstrap: int = 10_000,
                               bootstrap_seed: int = 0) -> dict:
    """Equal-fragment summaries and paired differences for held-out scores."""
    summaries, values = {}, {}
    for arm in arms:
        rows = {
            name: float(row["arms"][arm]["evaluation_alignment"])
            for name, row in fragments.items()
            if arm in row.get("arms", {})
            and row["arms"][arm].get("status") == "ok"
            and np.isfinite(row["arms"][arm].get("evaluation_alignment", np.nan))
        }
        values[arm] = rows
        a = np.asarray(list(rows.values()), dtype=float)
        summaries[arm] = {
            "status": "ok" if len(a) else "unavailable",
            "unavailable_reason": (None if len(a)
                                     else "no finite held-out fragment scores"),
            "n_fragments": int(len(a)),
            "mean_evaluation_alignment": float(a.mean()) if len(a) else None,
            "median_evaluation_alignment": (float(np.median(a))
                                              if len(a) else None),
            "mean_excess_over_chance": (float((a - 0.5).mean())
                                         if len(a) else None),
        }

    contrasts = {}
    for arm in arms:
        if arm == reference:
            continue
        common = sorted(set(values.get(reference, {})) & set(values.get(arm, {})))
        by_fragment = {name: values[reference][name] - values[arm][name]
                       for name in common}
        diff = np.asarray(list(by_fragment.values()), dtype=float)
        if len(diff):
            bootstrap = fragment_family_cluster_bootstrap(
                by_fragment, n_bootstrap, bootstrap_seed)
        else:
            bootstrap = {
                "status": "unavailable",
                "unavailable_reason": "no paired held-out fragment scores",
                "method": "fragment-family cluster bootstrap",
                "n_fragments": 0,
                "n_families": 0,
                "n_bootstrap": int(n_bootstrap),
                "seed": int(bootstrap_seed),
                "ci95": [None, None],
            }
        contrasts[f"{reference}_minus_{arm}"] = {
            "status": "ok" if len(diff) else "unavailable",
            "unavailable_reason": (None if len(diff)
                                     else "no paired held-out fragment scores"),
            "n_fragments": int(len(diff)),
            "mean_difference": float(diff.mean()) if len(diff) else None,
            "median_difference": (float(np.median(diff))
                                  if len(diff) else None),
            "reference_wins": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "fragment_family_cluster_bootstrap": bootstrap,
        }
    return {
        "arms": summaries,
        "paired_contrasts": contrasts,
        "inference": {
            "cluster_unit": "fragment family",
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "family_assignments": family_map(fragments),
        },
    }


def _estimand_result(row: dict, arm: str, estimand_key: str | None) -> dict:
    result = row.get("arms", {}).get(arm, {})
    if estimand_key is not None:
        result = result.get(estimand_key, {})
    return result


def _aggregate_wsd_metric(fragments: dict, detector: str, score: str,
                          arms, reference: str, n_bootstrap: int,
                          bootstrap_seed: int,
                          estimand_key: str | None = None,
                          null_key: str = "pipeline_null") -> dict:
    summaries, values = {}, {}
    for arm in arms:
        results = {name: _estimand_result(row, arm, estimand_key)
                   for name, row in fragments.items()}
        rows = {
            name: float(result[detector][score])
            for name, result in results.items()
            if result.get("status") == "ok"
            and np.isfinite(result.get(detector, {}).get(score, np.nan))
        }
        values[arm] = rows
        array = np.asarray(list(rows.values()), dtype=float)
        # An empty arm is JSON null, not NaN: the within-type estimand can
        # legitimately have no eligible fragment, and NaN is not portable JSON.
        summaries[arm] = {
            "status": "ok" if len(array) else "unavailable",
            "n_fragments": int(len(array)),
            "mean": float(array.mean()) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "n_exceeds_pipeline_null_p95": int(sum(
                bool(results[name][detector].get(
                    null_key, {}).get("observed_exceeds_p95", False))
                for name in rows)),
        }

    contrasts = {}
    for arm in arms:
        if arm == reference:
            continue
        common = sorted(set(values.get(reference, {})) & set(values.get(arm, {})))
        by_fragment = {name: values[reference][name] - values[arm][name]
                       for name in common}
        difference = np.asarray(list(by_fragment.values()), dtype=float)
        if len(difference):
            bootstrap = fragment_family_cluster_bootstrap(
                by_fragment, n_bootstrap, bootstrap_seed)
        else:
            bootstrap = {
                "status": "unavailable",
                "unavailable_reason": "no paired held-out fragment scores",
                "method": "fragment-family cluster bootstrap",
                "n_fragments": 0, "n_families": 0,
                "n_bootstrap": int(n_bootstrap), "seed": int(bootstrap_seed),
                "ci95": [None, None],
            }
        contrasts[f"{reference}_minus_{arm}"] = {
            "status": "ok" if len(difference) else "unavailable",
            "n_fragments": int(len(difference)),
            "mean_difference": (float(difference.mean())
                                if len(difference) else None),
            "median_difference": (float(np.median(difference))
                                  if len(difference) else None),
            "reference_wins": int((difference > 0).sum()),
            "ties": int((difference == 0).sum()),
            "fragment_family_cluster_bootstrap": bootstrap,
        }
    return {"arms": summaries, "paired_contrasts": contrasts,
            "pipeline_null_key": null_key,
            "detector_sharing": detector_sharing(
                fragments, arms, detector, estimand_key)}


def aggregate_wsd_fragment_results(
        fragments: dict, arms=WSD_ARMS,
        reference: str = "learned_sae_topk32_8192",
        n_bootstrap: int = 10_000, bootstrap_seed: int = 0) -> dict:
    """Aggregate schema-v2 held-out WSD estimands.

    ``within_type_oriented_wsd`` reads each arm's ``within_type`` result (see
    :func:`within_type_wsd_coordinates`) and counts null exceedances against
    the type-matched null, the exact stratified permutation null for that
    estimand. Fragments without a within-type result are absent from it, so
    its fragment set can be smaller than the primary one.
    """
    return {
        "primary_oriented_wsd": _aggregate_wsd_metric(
            fragments, "oriented_detector", "evaluation_oriented_wsd",
            arms, reference, n_bootstrap, bootstrap_seed),
        "secondary_mean_absolute_wsd": _aggregate_wsd_metric(
            fragments, "absolute_detector", "evaluation_mean_absolute_wsd",
            arms, reference, n_bootstrap, bootstrap_seed),
        "within_type_oriented_wsd": _aggregate_wsd_metric(
            fragments, "oriented_detector", "evaluation_oriented_wsd",
            arms, reference, n_bootstrap, bootstrap_seed,
            estimand_key="within_type",
            null_key="pipeline_null_type_matched"),
        "inference": {
            "cluster_unit": "fragment family",
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_seed": int(bootstrap_seed),
            "family_assignments": family_map(fragments),
        },
    }
