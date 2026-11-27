"""Linear probes for remaining valence capacity, with the controls that make the
accuracy figure mean something.

Remaining valence is a deterministic function of the visible prefix, so a probe
recovering it at 99% says nothing on its own. This module always reports, next to
the probe:

* **majority-class accuracy** and the label histogram,
* a **control task** in the sense of Hewitt & Liang (2019) -- the same probe fit
  against a random but token-consistent relabelling -- and the resulting
  **selectivity** (probe accuracy minus control accuracy),
* a **token-identity baseline**: multinomial regression on the one-hot current
  token alone. If that also reaches ~99%, the result is about SMILES, not about
  the model.

Splits are **grouped by molecule**. The previous implementation split at token
level, so positions from one molecule appeared on both sides.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit


class _TorchLogReg:
    """Multinomial logistic regression on the GPU, matching sklearn's objective.

    ``LogisticRegression`` minimises ``0.5*||W||^2 + C * sum_i CE_i`` with an
    unpenalised intercept; the same objective is optimised here with LBFGS and
    strong-Wolfe line search. Stage 05 fits thirteen of these over ~160k x 512
    matrices, which is minutes of CPU per run.

    This is not bit-identical to sklearn -- a different optimiser reaching the
    same convex optimum agrees to a few 1e-4 in accuracy, which is far below the
    effects being reported, but it *is* a numerical change and the fallback
    stays available via ``device="cpu"``.
    """

    def __init__(self, coef, intercept):
        self.coef_, self.intercept_ = coef, intercept

    def decision(self, X):
        return np.asarray(X, dtype=np.float32) @ self.coef_.T + self.intercept_

    def predict(self, X):
        return self.decision(X).argmax(1)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())


def _fit_torch(X, y, C=1.0, max_iter=1000, device="cuda"):
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    Xt = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32, device=device)
    yt = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.long, device=device)
    K = int(yt.max().item()) + 1
    W = torch.zeros(Xt.shape[1], K, device=device, requires_grad=True)
    b = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=10,
                            tolerance_grad=1e-7, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = 0.5 * (W * W).sum() + C * F.cross_entropy(Xt @ W + b, yt,
                                                         reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    return _TorchLogReg(W.detach().T.cpu().numpy(), b.detach().cpu().numpy())


def _fit_eval(X, y, groups, seed=0, test_size=0.3, max_iter=1000, C=1.0,
              device=None):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    if device is None:
        try:
            import torch  # noqa: PLC0415
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if device.startswith("cuda"):
        clf = _fit_torch(X[tr], y[tr], C=C, max_iter=max_iter, device=device)
    else:
        clf = LogisticRegression(max_iter=max_iter, C=C)  # multinomial by default
        clf.fit(X[tr], y[tr])
    return clf, float(clf.score(X[te], y[te])), tr, te


def bootstrap_ci(y_true, y_pred, groups, n_boot=1000, seed=0, ci=95):
    """Molecule-level bootstrap. Resampling tokens would understate uncertainty
    because positions within a molecule are strongly correlated."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        stats.append(float((y_true[idx] == y_pred[idx]).mean()))
    lo, hi = np.percentile(stats, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(np.mean(stats)), float(lo), float(hi)


def control_labels(token_ids: np.ndarray, n_classes: int, seed: int = 0) -> np.ndarray:
    """Hewitt-Liang control task: a fixed random label per token *type*.

    A probe can only fit this by memorizing token identity, so the gap between
    real and control accuracy (selectivity) measures how much of the probe's
    success reflects structure beyond the current token.
    """
    rng = np.random.default_rng(seed)
    table = {t: int(rng.integers(n_classes)) for t in np.unique(token_ids)}
    return np.array([table[t] for t in token_ids], dtype=int)


def within_type_shuffled_labels(token_ids: np.ndarray, labels: np.ndarray,
                                seed: int = 0, blocks: np.ndarray | None = None) -> np.ndarray:
    """Permute the true labels among positions sharing a current-token identity.

    The Hewitt--Liang control assigns each token type one random label, which a
    probe fits by memorizing 64 type-to-label pairs; here that reaches 0.999 and
    selectivity collapses to nothing, so the control measures the probe's
    capacity rather than the representation's.

    This control instead keeps the empirical distribution of capacity *given the
    current token* exactly as it is --- same marginal, same class imbalance, same
    per-type label multiset --- and destroys only the part that depends on
    context. A probe reading token identity alone is therefore unaffected by the
    shuffle, and any accuracy above it is information the residual stream holds
    about the prefix rather than about the token being scored.

    ``blocks`` must be supplied whenever accuracy is measured on a split of these
    rows, and must identify which side of that split each row falls on. Shuffling
    across the whole array first and splitting afterwards lets labels cross the
    boundary, so the per-token label distribution is preserved globally but not
    on the train and test populations where accuracy is actually computed --- and
    a token-only predictor is then no longer unaffected, which silently inflates
    the measured selectivity. Permuting inside each (token identity, split) cell
    keeps the invariant where it is used.
    """
    token_ids = np.asarray(token_ids)
    labels = np.asarray(labels)
    if token_ids.shape != labels.shape:
        raise ValueError("token_ids and labels must have the same shape")
    if blocks is None:
        blocks = np.zeros_like(token_ids)
    blocks = np.asarray(blocks)
    if blocks.shape != labels.shape:
        raise ValueError("blocks must have the same shape as labels")
    rng = np.random.default_rng(seed)
    out = labels.copy()
    for t in np.unique(token_ids):
        for b in np.unique(blocks):
            idx = np.flatnonzero((token_ids == t) & (blocks == b))
            if len(idx) > 1:
                out[idx] = labels[rng.permutation(idx)]
    return out


def run_probe(acts: np.ndarray, labels: np.ndarray, groups: np.ndarray,
              token_ids: np.ndarray, n_classes: int = 5, seed: int = 0,
              n_boot: int = 500) -> dict:
    """Fit the probe and all three baselines on the same grouped split."""
    clf, acc, tr, te = _fit_eval(acts, labels, groups, seed=seed)
    pred = clf.predict(acts[te])
    mean, lo, hi = bootstrap_ci(labels[te], pred, groups[te], n_boot=n_boot, seed=seed)

    counts = np.bincount(labels, minlength=n_classes)
    majority = float(counts.max() / counts.sum())

    ctrl_y = control_labels(token_ids, n_classes, seed=seed)
    _, ctrl_acc, _, _ = _fit_eval(acts, ctrl_y, groups, seed=seed)

    onehot = np.zeros((len(token_ids), int(token_ids.max()) + 1), dtype=np.float32)
    onehot[np.arange(len(token_ids)), token_ids] = 1.0
    _, tok_acc, _, _ = _fit_eval(onehot, labels, groups, seed=seed)

    # The control that the type-level one fails to be: same representation, same
    # per-token label distribution, context removed. The shuffle is confined to
    # each side of the split _fit_eval will construct, so the invariant holds on
    # the population the accuracy is measured on rather than only in aggregate.
    split_block = np.zeros(len(labels), dtype=int)
    split_block[te] = 1
    within_y = within_type_shuffled_labels(token_ids, labels, seed=seed,
                                           blocks=split_block)
    _, within_acc, _, _ = _fit_eval(acts, within_y, groups, seed=seed)

    return {
        "accuracy": acc,
        "accuracy_ci95": [lo, hi],
        "accuracy_bootstrap_mean": mean,
        "majority_class_accuracy": majority,
        "control_task_accuracy": ctrl_acc,
        "selectivity": acc - ctrl_acc,
        "within_type_shuffled_accuracy": within_acc,
        "within_type_selectivity": acc - within_acc,
        "token_identity_baseline_accuracy": tok_acc,
        "label_histogram": counts.tolist(),
        "confusion_matrix": confusion_matrix(
            labels[te], pred, labels=list(range(n_classes))).tolist(),
        "n_train_positions": int(len(tr)),
        "n_test_positions": int(len(te)),
        "n_molecules": int(len(np.unique(groups))),
        "split": "GroupShuffleSplit by molecule",
    }


def capacity_direction(acts: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                       seed: int = 0, alpha: float = 1.0
                       ) -> tuple[np.ndarray, dict]:
    """Unit-norm direction from a ridge regression on remaining capacity itself.

    ``binary_direction`` needs a threshold, and at bond-decision positions there
    is no good one: remaining capacity is 0/1/2/3/4 with frequencies
    0.001/0.056/0.110/0.811/0.022, so 81% of the population sits in a single
    class. Every cut is lopsided -- ``>=2`` gives 0.942 positive and ``>=3``
    gives 0.833, and there is nothing in between. That is a property of the
    chemistry (most bond decisions follow a carbon that has written one bond),
    not a parameter that was chosen badly, so it cannot be fixed by picking a
    different threshold.

    Regressing the integer label directly sidesteps the question. The resulting
    direction is the one along which decoded remaining capacity increases, fit
    on the whole ordinal range rather than on one arbitrary cut of it, and it
    uses every class instead of collapsing four of them together.

    Reported alongside the binary direction rather than replacing it: the paper
    quotes the binary probe's accuracy, and the two directions answer different
    questions.
    """
    from sklearn.linear_model import Ridge  # noqa: PLC0415

    y = labels.astype(np.float64)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(acts, y, groups))
    reg = Ridge(alpha=alpha)
    reg.fit(acts[tr], y[tr])
    w = reg.coef_
    w = w / (np.linalg.norm(w) + 1e-12)

    pred = reg.predict(acts[te])
    ss_res = float(((y[te] - pred) ** 2).sum())
    ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
    return w.astype(np.float32), {
        "threshold": None,
        "objective": "ridge regression on remaining capacity (0-4)",
        "alpha": float(alpha),
        "label_histogram": {int(v): int(c) for v, c in
                            zip(*np.unique(labels, return_counts=True))},
        "n_positions": int(len(y)),
        "heldout_r2": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "heldout_rmse": float(np.sqrt(((y[te] - pred) ** 2).mean())),
        # Rounding the prediction to the nearest class makes this comparable to
        # the binary probe's accuracy, which is otherwise on a different scale.
        "heldout_accuracy_rounded": float(
            (np.clip(np.rint(pred), 0, 4) == y[te]).mean()),
    }


def binary_direction(acts: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                     threshold: int = 2, seed: int = 0, *,
                     max_iter: int = 1000) -> tuple[np.ndarray, dict]:
    """Unit-norm direction from a binary probe on 1[remaining >= threshold].

    This is the vector the intervention uses. It is fit *separately* from the
    multinomial probe above; the previous paper described the intervention
    direction as coming from the multinomial probe, which was not what the code
    did. Reporting them as two objects prevents that confusion recurring.
    """
    y = (labels >= threshold).astype(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(acts, y, groups))
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    clf.fit(acts[tr], y[tr])
    w = clf.coef_[0]
    w = w / (np.linalg.norm(w) + 1e-12)
    meta = {
        "threshold": threshold,
        "objective": f"binary logistic regression on 1[remaining >= {threshold}]",
        "class_weight": "balanced",
        "positive_rate": float(y.mean()),
        "n_positions": int(len(y)),
        "heldout_accuracy": float(clf.score(acts[te], y[te])),
    }
    return w.astype(np.float32), meta
