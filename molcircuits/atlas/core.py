"""Per-feature statistics: what each dictionary feature actually detects.

Everything the qualitative feature analyses need comes from three primitives:

* ``FeatureAtlas``  -- one streaming pass over a corpus giving, per feature, its
  firing rate, activation moments, and the top-activating token positions.
* ``fragment_auroc`` -- how well a feature's activation separates tokens inside a
  SMARTS match from tokens outside it. Unlike WSD this is scale-free and needs no
  sign convention, so a non-negative dictionary and a zero-centred dense basis are
  directly comparable.
* ``ablation_delta_nll`` -- the causal counterpart: zero a feature through the
  SAE and measure the change in next-token loss.  It supports a position mask so
  the edit can be made at the same token whose activation defined the detector.
* ``fragment_ablation_tensors`` -- spell out the temporal estimand.  A detector
  read at fragment token ``t`` is ablated at ``t`` and scored on the logit at
  ``t`` predicting ``t+1``; predicting the fragment token itself is a separate
  ``t-1`` estimand.

lmkit spread these across ``descriptors.py``, ``atlas_utils.py`` and
``attribution.py`` (~1,300 lines, much of it one-off inline code and a
hand-rolled running-stats class). The compression here is not stylistic: keeping
the top-k merge on the device, and computing AUROC from a rank sum rather than by
sorting per feature, is what makes an 8192-feature atlas a single pass instead of
a loop over features.
"""
from __future__ import annotations

import numpy as np
import torch

from ..model.hooks import HookManager, resid_post


# These strings are persisted in atlas artifacts.  Do not reuse one for a new
# alignment: the point is that an old result remains identifiable rather than
# acquiring corrected semantics merely because the reader was upgraded.
LEGACY_GLOBAL_PREDICT_FRAGMENT = "legacy_detector_t_global_ablation_predict_t"
DETECTOR_T_NEXT_TOKEN = "detector_t_local_ablation_predict_t_plus_1"
PREDICT_FRAGMENT_FROM_T_MINUS_1 = (
    "detector_t_minus_1_local_ablation_predict_fragment_t"
)


def fragment_ablation_tensors(
        tokens: torch.Tensor, fragment_mask: torch.Tensor, pad_id: int,
        estimand: str = DETECTOR_T_NEXT_TOKEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return targets, edit mask and score mask for an explicit causal estimand.

    ``tokens[:, p]`` produces a logit at position ``p`` for the target token at
    ``p + 1``.  ``fragment_mask[:, t]`` marks the encoded fragment token itself.

    ``DETECTOR_T_NEXT_TOKEN`` is the causal counterpart to an atlas detector
    selected from activations at fragment token ``t``: zero that feature only at
    the matched ``t`` positions and score the logits at those positions, whose
    targets are ``t + 1``.

    ``PREDICT_FRAGMENT_FROM_T_MINUS_1`` asks a different, also valid question:
    select a detector from activations at ``t - 1``, zero it only there, and
    score its prediction of the fragment token at ``t``.  The caller is
    responsible for selecting the feature on those ``t - 1`` activations; using
    a feature selected at ``t`` would recreate the temporal mismatch this helper
    exists to prevent.

    The edit and score masks are deliberately returned separately even though
    they coincide for both current estimands.  That makes their equality an
    auditable part of the design rather than an accidental coupling inside the
    ablation implementation.
    """
    if tokens.shape != fragment_mask.shape:
        raise ValueError(
            "tokens and fragment_mask must have the same (batch, sequence) shape; "
            f"got {tuple(tokens.shape)} and {tuple(fragment_mask.shape)}")
    if tokens.ndim != 2:
        raise ValueError(f"expected 2-D token batch, got shape {tuple(tokens.shape)}")

    targets = torch.full_like(tokens, pad_id)
    targets[:, :-1] = tokens[:, 1:]
    fragment_mask = fragment_mask.to(device=tokens.device, dtype=torch.bool)

    if estimand == DETECTOR_T_NEXT_TOKEN:
        positions = fragment_mask.clone()
    elif estimand == PREDICT_FRAGMENT_FROM_T_MINUS_1:
        positions = torch.zeros_like(fragment_mask)
        positions[:, :-1] = fragment_mask[:, 1:]
    else:
        raise ValueError(
            f"unknown estimand {estimand!r}; expected {DETECTOR_T_NEXT_TOKEN!r} "
            f"or {PREDICT_FRAGMENT_FROM_T_MINUS_1!r}")

    # A final-column logit has no next-token target in this batch.  This is
    # normally already false for a content-token fragment mask, but enforcing it
    # here keeps edit and score counts equal for arbitrary callers and tests.
    positions &= targets != pad_id
    return targets, positions.clone(), positions


class FeatureAtlas:
    """Streaming per-feature statistics with top-k activating positions.

    The top-k is merged on the device: each batch's own top-k is concatenated
    with the running top-k and re-reduced, which is O(K) memory per feature
    instead of materialising every activation, and avoids a per-feature Python
    heap over d_sae=8192 features.
    """

    def __init__(self, d_sae: int, k: int = 16, device: str = "cuda"):
        self.d_sae, self.k, self.device = d_sae, k, device
        self.top_vals = torch.full((k, d_sae), float("-inf"), device=device)
        self.top_pos = torch.full((k, d_sae), -1, dtype=torch.long, device=device)
        self.n_fired = torch.zeros(d_sae, device=device, dtype=torch.float64)
        self.sum_act = torch.zeros(d_sae, device=device, dtype=torch.float64)
        self.sum_sq = torch.zeros(d_sae, device=device, dtype=torch.float64)
        self.n_tokens = 0

    @torch.no_grad()
    def update(self, z: torch.Tensor, global_ids: torch.Tensor) -> None:
        """z: (N, d_sae) activations. global_ids: (N,) position identifiers."""
        self.n_tokens += z.shape[0]
        self.n_fired += (z > 0).sum(0, dtype=torch.float64)
        self.sum_act += z.sum(0, dtype=torch.float64)
        self.sum_sq += (z.double() ** 2).sum(0)

        k = min(self.k, z.shape[0])
        vals, idx = z.topk(k, dim=0)                       # (k, d_sae)
        pos = global_ids[idx]
        cat_v = torch.cat([self.top_vals, vals], 0)
        cat_p = torch.cat([self.top_pos, pos], 0)
        sel = cat_v.topk(self.k, dim=0)
        self.top_vals = sel.values
        self.top_pos = cat_p.gather(0, sel.indices)

    def summary(self) -> dict:
        n = max(self.n_tokens, 1)
        rate = (self.n_fired / n).cpu().numpy()
        mean = (self.sum_act / n).cpu().numpy()
        var = (self.sum_sq / n).cpu().numpy() - mean ** 2
        return {
            "n_tokens": int(self.n_tokens),
            "firing_rate": rate,
            "mean_activation": mean,
            "std_activation": np.sqrt(np.maximum(var, 0.0)),
            "top_values": self.top_vals.cpu().numpy(),
            "top_positions": self.top_pos.cpu().numpy(),
            "dead": rate == 0.0,
        }


@torch.no_grad()
def build_atlas(model, sae, tok, smiles, layer: int, n_ctx: int = 128,
                k: int = 16, device: str = "cuda", batch: int = 64):
    """One pass over ``smiles``; returns (FeatureAtlas, positions index).

    ``positions`` maps each global id back to (molecule_index, token_index),
    where token_index indexes the ENCODED sequence -- position 0 is BOS. Use
    ``token_index - 1`` to index ``token_spans(smiles)``; the rest of the
    codebase spells this as ``offset=1``. Returned explicitly as
    ``bos_offset`` so a caller cannot silently render the wrong substring.
    """
    atlas = FeatureAtlas(sae.cfg.d_sae, k, device)
    positions: list[tuple[int, int]] = []
    for s0 in range(0, len(smiles), batch):
        chunk = smiles[s0:s0 + batch]
        ids, keep = [], []
        for j, s in enumerate(chunk):
            try:
                e = tok.encode(s)[:n_ctx]
            except ValueError:
                continue
            ids.append(e + [tok.pad_id] * (n_ctx - len(e)))
            keep.append((s0 + j, len(e)))
        if not ids:
            continue
        x = torch.tensor(ids, device=device)
        h = HookManager().capture(resid_post(layer))
        model(x, hooks=h)
        a = h[resid_post(layer)].float()
        z = sae.encode(a.reshape(-1, sae.cfg.d_in))

        gid, flat_keep = [], []
        for row, (mi, ln) in enumerate(keep):
            for t in range(ln):
                flat_keep.append(row * n_ctx + t)
                gid.append(len(positions))
                positions.append((mi, t))
        sel = torch.tensor(flat_keep, device=device)
        atlas.update(z[sel], torch.tensor(gid, device=device))
    return atlas, {"positions": positions, "bos_offset": 1}


def fragment_auroc(acts: np.ndarray, in_fragment: np.ndarray,
                   chunk: int = 1024) -> np.ndarray:
    """AUROC per feature for separating in-fragment from out-of-fragment tokens.

    acts:         (N, K) activations at N token positions
    in_fragment:  (N,) bool
    returns:      (K,) AUROC in [0, 1]

    Computed from the Mann-Whitney rank sum with MIDRANKS. Ordinal ranks are only
    valid when there are no ties, and a sparse code is ~99.6% exact zeros -- with
    `argsort` each tied zero gets a distinct rank set by its row position, so the
    statistic encodes where a token sits in the corpus. Measured effect before
    the fix: the null sat anywhere from 0.02 to 0.52 instead of 0.5, dead features
    returned 1.0 or 0.0 depending on layout, and the best-feature argmax flipped
    for 9 of 33 fragments.

    Chunked over the feature axis: ranking is O(N x K) in memory, which at
    10k molecules x 8192 features would otherwise be ~50 GB.
    """
    return auroc_from_ranks(rank_transform(acts, chunk), in_fragment)


def rank_transform(acts: np.ndarray, chunk: int = 1024) -> dict:
    """Midrank-transform the activations once, for reuse across many masks.

    The ranking does not depend on which tokens are in the fragment, so it is
    hoisted out of the mask loop: a permutation null over 30 fragments x 20 draws
    would otherwise re-sort the same (N, K) array 600 times. Exactly the hoist
    that ``wsd.basis_stats`` performs for the same reason.
    """
    from scipy.stats import rankdata  # noqa: PLC0415

    N, K = acts.shape
    ranks = np.empty((N, K), dtype=np.float32)
    for s in range(0, K, chunk):
        e = min(s + chunk, K)
        ranks[:, s:e] = rankdata(acts[:, s:e], method="average", axis=0)
    return {"ranks": ranks, "dead": acts.max(0) <= 0, "n": N}


def sparse_rank_transform(acts: np.ndarray) -> dict:
    """Exact midranks for sparse SAE codes, stored as offsets from each zero tie."""
    from scipy.sparse import csc_matrix
    from scipy.stats import rankdata

    columns = csc_matrix(acts).astype(np.float64, copy=False)
    if not np.isfinite(columns.data).all():
        return rank_transform(acts)
    n, width = columns.shape
    base = np.empty(width, dtype=np.float64)
    dead = np.ones(width, dtype=bool)
    for feature in range(width):
        begin, end = columns.indptr[feature:feature + 2]
        values = columns.data[begin:end]
        zeros = n - len(values)
        positive = values > 0
        base[feature] = np.count_nonzero(values < 0) + (zeros + 1) / 2
        if len(values):
            dead[feature] = not positive.any()
            # Half-integers and their sums are exact in float64 at these sizes.
            values[:] = rankdata(values, method="average") + zeros * positive - base[feature]
    return {"ranks": columns.tocsr(), "columns": columns, "zero_rank": base,
            "dead": dead, "n": n}


def auroc_from_ranks(rt: dict, in_fragment: np.ndarray) -> np.ndarray:
    """AUROC per feature from a precomputed midrank transform."""
    pos = int(in_fragment.sum())
    neg = rt["n"] - pos
    if pos == 0 or neg == 0:
        return np.full(rt["ranks"].shape[1], np.nan, dtype=np.float64)
    if "zero_rank" in rt:
        hits = rt["ranks"][in_fragment]
        rank_sum = pos * rt["zero_rank"] + np.bincount(
            hits.indices, weights=hits.data, minlength=hits.shape[1])
    else:
        rank_sum = rt["ranks"][in_fragment].sum(0, dtype=np.float64)
    out = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    # A feature that never fires carries no signal; 0.5 is the honest value, but
    # NaN is honester still -- it cannot be silently averaged into a summary.
    out[rt["dead"]] = np.nan
    return out


def auroc_feature_from_ranks(rt: dict, in_fragment: np.ndarray,
                             feature: int) -> float:
    """AUROC for one fixed feature in an already ranked evaluation population."""
    in_fragment = np.asarray(in_fragment, dtype=bool)
    if in_fragment.shape != (rt["n"],):
        raise ValueError(
            f"label shape {in_fragment.shape} does not match {rt['n']} ranks")
    pos = int(in_fragment.sum())
    neg = rt["n"] - pos
    if pos == 0 or neg == 0 or bool(rt["dead"][feature]):
        return float("nan")
    if "zero_rank" in rt:
        columns = rt["columns"]
        begin, end = columns.indptr[feature:feature + 2]
        selected = in_fragment[columns.indices[begin:end]]
        rank_sum = pos * rt["zero_rank"][feature] + columns.data[begin:end][selected].sum()
    else:
        rank_sum = float(rt["ranks"][in_fragment, feature].sum(dtype=np.float64))
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def deterministic_molecule_split(n_molecules: int, selection_fraction: float,
                                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically partition molecule indices for selection and evaluation."""
    if n_molecules < 2:
        raise ValueError("a selection/evaluation split needs at least two molecules")
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be strictly between 0 and 1")
    n_selection = min(n_molecules - 1,
                      max(1, int(np.floor(n_molecules * selection_fraction))))
    order = np.random.default_rng(seed).permutation(n_molecules)
    # Sorting does not change membership and makes artifact indices compact and
    # easy to compare across reruns.
    return np.sort(order[:n_selection]), np.sort(order[n_selection:])


def _permute_within_blocks(labels: np.ndarray, blocks: dict[int, list[int]],
                           rng: np.random.Generator) -> np.ndarray:
    out = np.asarray(labels, dtype=bool).copy()
    for idxs in blocks.values():
        sub = out[idxs]
        if sub.any() and not sub.all():
            out[idxs] = rng.permutation(sub)
    return out


def heldout_detector_screen(selection_rt: dict, evaluation_rt: dict,
                            selection_labels: np.ndarray,
                            evaluation_labels: np.ndarray,
                            selection_blocks: dict[int, list[int]],
                            evaluation_blocks: dict[int, list[int]],
                            n_perm: int = 20, seed: int = 0) -> dict:
    """Select a feature on one molecule split and evaluate it on the other.

    The returned ``best_feature`` is *only* the selection-split argmax.
    ``evaluation_auroc`` reads that fixed coordinate in the held-out split; it
    never re-selects on evaluation data.  The pipeline null repeats the entire
    operation: permute labels within molecules on both splits, select the null
    winner on selection, and score that winner on held-out labels.
    """
    selection_labels = np.asarray(selection_labels, dtype=bool)
    evaluation_labels = np.asarray(evaluation_labels, dtype=bool)
    if selection_labels.shape != (selection_rt["n"],):
        raise ValueError("selection label count does not match selection ranks")
    if evaluation_labels.shape != (evaluation_rt["n"],):
        raise ValueError("evaluation label count does not match evaluation ranks")
    if n_perm < 0:
        raise ValueError("n_perm must be non-negative")

    selection_aurocs = auroc_from_ranks(selection_rt, selection_labels)
    if not np.isfinite(selection_aurocs).any():
        return {
            "status": "insufficient_selection_class_variation",
            "best_feature": None,
            "n_selection_positions": int(selection_rt["n"]),
            "n_evaluation_positions": int(evaluation_rt["n"]),
        }
    best = int(np.nanargmax(selection_aurocs))
    evaluation_auroc = auroc_feature_from_ranks(
        evaluation_rt, evaluation_labels, best)
    if not np.isfinite(evaluation_auroc):
        return {
            "status": "selected_feature_not_evaluable",
            "best_feature": best,
            "selection_auroc": float(selection_aurocs[best]),
            "evaluation_auroc": float("nan"),
            "n_selection_positions": int(selection_rt["n"]),
            "n_evaluation_positions": int(evaluation_rt["n"]),
        }

    # Permutations leave homogeneous molecules untouched. Identify those once.
    selection_blocks = {k: np.asarray(v) for k, v in selection_blocks.items()
                        if selection_labels[v].any() and not selection_labels[v].all()}
    evaluation_blocks = {k: np.asarray(v) for k, v in evaluation_blocks.items()
                         if evaluation_labels[v].any() and not evaluation_labels[v].all()}
    pipeline_null, fixed_feature_null, selection_null = [], [], []
    for p in range(n_perm):
        select_perm = _permute_within_blocks(
            selection_labels, selection_blocks,
            np.random.default_rng(seed + 2 * p))
        eval_perm = _permute_within_blocks(
            evaluation_labels, evaluation_blocks,
            np.random.default_rng(seed + 2 * p + 1))
        null_selection_aurocs = auroc_from_ranks(selection_rt, select_perm)
        if not np.isfinite(null_selection_aurocs).any():
            continue
        null_best = int(np.nanargmax(null_selection_aurocs))
        selection_null.append(float(null_selection_aurocs[null_best]))
        pipeline_null.append(auroc_feature_from_ranks(
            evaluation_rt, eval_perm, null_best))
        fixed_feature_null.append(auroc_feature_from_ranks(
            evaluation_rt, eval_perm, best))

    def summary(values):
        a = np.asarray(values, dtype=float)
        a = a[np.isfinite(a)]
        if not len(a):
            return {"mean": float("nan"), "p95": float("nan"), "n": 0}
        return {"mean": float(a.mean()),
                "p95": float(np.percentile(a, 95)), "n": int(len(a))}

    pipeline_summary = summary(pipeline_null)
    return {
        "status": "ok",
        "best_feature": best,
        "selection_auroc": float(selection_aurocs[best]),
        "evaluation_auroc": float(evaluation_auroc),
        "evaluation_oracle_best_auroc": float(
            np.nanmax(auroc_from_ranks(evaluation_rt, evaluation_labels))),
        "selection_null_best": summary(selection_null),
        "evaluation_pipeline_null": pipeline_summary,
        "evaluation_fixed_feature_null": summary(fixed_feature_null),
        "exceeds_evaluation_pipeline_null_p95": bool(
            pipeline_summary["n"] > 0
            and evaluation_auroc > pipeline_summary["p95"]),
        "n_perm_requested": int(n_perm),
        "n_selection_positions": int(selection_rt["n"]),
        "n_evaluation_positions": int(evaluation_rt["n"]),
        "n_selection_positive_positions": int(selection_labels.sum()),
        "n_evaluation_positive_positions": int(evaluation_labels.sum()),
    }


@torch.no_grad()
def ablation_delta_nll(model, sae, tokens: torch.Tensor, targets: torch.Tensor,
                       features, layer: int, ignore_index: int = 0,
                       device: str = "cuda",
                       score_mask: torch.Tensor | None = None,
                       ablation_mask: torch.Tensor | None = None,
                       estimand: str = "unspecified_feature_ablation") -> dict:
    """Change in next-token loss when the given SAE features are zeroed.

    The baseline passes through the *same* SAE encode/decode path with nothing
    zeroed, so the number isolates the feature's contribution rather than the
    dictionary's reconstruction error.

    ``ignore_index`` defaults to 0 (the pad id), not -100: this model is
    pretrained on packed sequences and never sees a pad token, so scoring a
    right-padded batch unmasked measures out-of-distribution behaviour -- the
    same defect that inverted the sign of the head-ablation delta_ce.

    ``score_mask`` is (B, T) and restricts scoring to the positions the question
    is about. ``ablation_mask`` independently restricts where the selected
    features are zeroed.  A missing ablation mask deliberately retains the old,
    global-ablation API semantics for existing callers and artifacts.  A local
    detector estimand must pass both masks explicitly; use
    :func:`fragment_ablation_tensors` to construct a temporally aligned pair.

    WITHOUT A SCORE MASK THIS FUNCTION ANSWERS A DIFFERENT QUESTION. Whole-batch
    CE mixes the selected fragment positions with every off-fragment consequence
    of a global feature ablation; it is not a diluted estimate of the local
    quantity.  On the atlas batch morpholine's global whole-batch cost was +0.145
    while the separately defined fragment-position score was +0.0011, and five
    fragments including benzene and phenol changed sign. Ten of 31 flipped sign
    overall, and the rank correlation moved from rho=-0.28 (P=0.12) to rho=-0.39
    (P=0.03). Restricting only the score still did not fix the separate temporal
    error in which a detector observed at token ``t`` was compared with the
    ``t-1`` logit that had already predicted it.

    The CE is token-weighted over the scored positions, so it does not depend on
    how the batch was grouped.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    tokens = tokens.to(device)
    targets = targets.to(device)
    if targets.shape != tokens.shape:
        raise ValueError(
            "tokens and targets must have the same (batch, sequence) shape; "
            f"got {tuple(tokens.shape)} and {tuple(targets.shape)}")
    keep = (targets != ignore_index)
    if score_mask is not None:
        if score_mask.shape != tokens.shape:
            raise ValueError(
                "score_mask must match tokens; "
                f"got {tuple(score_mask.shape)} and {tuple(tokens.shape)}")
        keep = keep & score_mask.to(device).bool()
    n_scored = int(keep.sum())

    local_mask = None
    if ablation_mask is not None:
        if ablation_mask.shape != tokens.shape:
            raise ValueError(
                "ablation_mask must match tokens; "
                f"got {tuple(ablation_mask.shape)} and {tuple(tokens.shape)}")
        local_mask = ablation_mask.to(device).bool()

    feature_idx = torch.as_tensor(features, device=device, dtype=torch.long).reshape(-1)
    if feature_idx.numel() == 0:
        raise ValueError("features must contain at least one feature index")

    def run(mask_feats):
        def edit(t):
            shape = t.shape
            z = sae.encode(t.reshape(-1, sae.cfg.d_in)).reshape(
                *shape[:-1], sae.cfg.d_sae)
            if mask_feats is not None:
                if local_mask is None:
                    z[..., mask_feats] = 0.0
                else:
                    selected = z.index_select(-1, mask_feats)
                    selected = selected.masked_fill(local_mask[..., None], 0.0)
                    z.index_copy_(-1, mask_feats, selected)
            return sae.decode(z.reshape(-1, sae.cfg.d_sae)).reshape(shape)
        hooks = HookManager().edit(resid_post(layer), edit)
        lg = model(tokens, hooks=hooks)
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(),
                               targets.reshape(-1), reduction="none")
        loss = loss.reshape(targets.shape) * keep
        return float(loss.sum() / max(n_scored, 1))

    base = run(None)
    abl = run(feature_idx)
    scope = "global" if local_mask is None else "position_local"
    n_ablated = int(tokens.numel() if local_mask is None else local_mask.sum())
    return {"ce_sae_baseline": base, "ce_ablated": abl, "delta_nll": abl - base,
            "n_features_ablated": int(feature_idx.numel()),
            "n_positions_scored": n_scored,
            "n_positions_ablated": n_ablated,
            "ablation_scope": scope, "estimand": estimand}
