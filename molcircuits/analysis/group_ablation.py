"""Statistics and control constructions for the group-ablation experiment.

``scripts/28_group_ablation.py`` asks whether detector AUROC ranks the causal
cost of removing a feature group. Across fragments, AUROC and cost both rise
with the amount of activation the group removes (Spearman +0.70 to +0.81 on the
committed seeds), so a raw rank correlation between them can be carried
entirely by that dose. The helpers here make the confound testable:

* :func:`partial_spearman` and :func:`spearman_with_bootstrap` report a rank
  correlation after linearly removing the ranks of nuisance covariates, with a
  bootstrap interval over fragments.
* :func:`active_control_threshold` picks off-fragment positions at which the
  group removes a dose comparable to its dose at the fragment, so a
  "costs less away from the fragment" contrast is not trivially explained by
  the group being silent there.
* :func:`dose_scales` sets the common removed-norm target of the dose-matched
  ablation.
* :func:`covariance_factor` and :func:`sample_directions` draw random
  directions with the residual stream's covariance rather than isotropically.

Everything here is pure numpy/torch on arrays, so it is unit-testable without
a model.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata
from scipy.stats import t as student_t


def _as_covariate_matrix(covariates, n: int) -> np.ndarray:
    if covariates is None:
        return np.zeros((n, 0))
    c = np.asarray(covariates, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    if c.shape[0] != n:
        raise ValueError(f"covariates have {c.shape[0]} rows for {n} observations")
    return c


def _partial_rank_r(x: np.ndarray, y: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Partial correlation of rank(x) and rank(y) given rank(c), batched.

    ``x`` and ``y`` are (B, n) and ``c`` is (B, n, k). Each row is ranked on its
    own, so tied draws from a bootstrap get average ranks exactly as a
    stand-alone Spearman would give them. Rows whose residuals have no variance
    come back NaN rather than an arbitrary number.
    """
    rx = rankdata(x, axis=-1)
    ry = rankdata(y, axis=-1)
    design = np.ones(x.shape + (1,))
    if c.shape[-1]:
        design = np.concatenate([design, rankdata(c, axis=-2)], axis=-1)
    # Projection onto the column space of [1, ranks(c)]; pinv keeps a draw in
    # which a covariate happens to be constant from failing the whole batch.
    hat = design @ np.linalg.pinv(design)
    ex = rx - np.einsum("bij,bj->bi", hat, rx)
    ey = ry - np.einsum("bij,bj->bi", hat, ry)
    num = (ex * ey).sum(-1)
    den = np.sqrt((ex * ex).sum(-1) * (ey * ey).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~(den > 1e-12 * np.maximum(1.0, np.abs(num)))] = np.nan
    return np.clip(r, -1.0, 1.0)


def partial_spearman(x, y, covariates=None) -> dict:
    """Spearman correlation of ``x`` and ``y`` controlling ``covariates``.

    All variables are rank-transformed; the ranks of ``x`` and ``y`` are each
    regressed on an intercept plus the covariate ranks, and the residuals are
    correlated. With no covariates this is exactly Spearman's rho. The P value
    is the usual t approximation with ``n - 2 - k`` degrees of freedom.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length")
    n = len(x)
    c = _as_covariate_matrix(covariates, n)
    k = c.shape[1]
    df = n - 2 - k
    if df < 1:
        return {"rho": float("nan"), "p": float("nan"), "n": n, "n_covariates": k}
    r = float(_partial_rank_r(x[None], y[None], c[None])[0])
    if not np.isfinite(r):
        p = float("nan")
    elif abs(r) >= 1.0:
        p = 0.0
    else:
        tstat = r * np.sqrt(df / (1.0 - r * r))
        p = float(2.0 * student_t.sf(abs(tstat), df))
    return {"rho": r, "p": p, "n": n, "n_covariates": k}


def spearman_with_bootstrap(x, y, covariates=None, n_boot: int = 10_000,
                            seed: int = 0, level: float = 0.95,
                            chunk: int = 2_000) -> dict:
    """:func:`partial_spearman` plus a percentile bootstrap over observations.

    Observations (fragments) are resampled with replacement. The draws depend
    only on ``seed`` and ``n``, so every statistic computed on the same
    fragments with the same seed uses the same resamples and their intervals
    are directly comparable. Degenerate draws are dropped and counted.
    """
    out = partial_spearman(x, y, covariates)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    c = _as_covariate_matrix(covariates, n)
    out.update({"n_boot": int(n_boot), "seed": int(seed), "level": float(level),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot_valid": 0})
    if n_boot < 1 or n - 2 - c.shape[1] < 1:
        return out
    rng = np.random.default_rng(seed)
    draws = []
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, n, size=(min(chunk, n_boot - s), n))
        draws.append(_partial_rank_r(x[idx], y[idx], c[idx]))
    draws = np.concatenate(draws)
    ok = draws[np.isfinite(draws)]
    out["n_boot_valid"] = int(len(ok))
    if len(ok):
        a = (1.0 - level) / 2.0
        out["ci_low"], out["ci_high"] = (float(v) for v in
                                         np.quantile(ok, [a, 1.0 - a]))
    return out


def active_control_threshold(control_doses, fragment_mean_dose: float,
                             max_fallback_positions: int | None = None) -> dict:
    """Dose threshold that makes off-fragment positions comparable to on-fragment.

    ``control_doses`` is the group's per-position removed-norm fraction
    (``|u_p| / |x_p|``) at every non-fragment atom position. Positions where
    the group is silent (dose 0) are never eligible. Among the active ones,
    sorted by dose, the rule keeps the LARGEST top-k whose mean dose is still
    at least ``fragment_mean_dose``: as many positions as possible, at a mean
    dose no lower than the fragment's. If even the single highest-dose control
    position falls short, the top ``max_fallback_positions`` (by default all
    active positions) are kept and ``reached_target`` is false; the achieved
    ratio is reported either way, so a shortfall is visible rather than hidden.

    Selecting the highest-dose control positions favours contexts in which the
    group fires strongly, which may themselves be fragment-like; that is the
    price of matching dose, and the unrestricted control is kept beside it.
    """
    d = np.asarray(control_doses, dtype=np.float64).reshape(-1)
    active = np.sort(d[d > 0])[::-1]
    out = {"rule": ("largest top-k of active non-fragment atom positions, by "
                    "removed-norm fraction, whose mean is >= the fragment-position "
                    "mean; else the top max_fallback_positions"),
           "fragment_mean_dose": float(fragment_mean_dose),
           "n_control_positions": int(len(d)), "n_active": int(len(active)),
           "threshold": None, "n_selected": 0, "selected_mean_dose": float("nan"),
           "dose_ratio_to_fragment": float("nan"), "reached_target": False}
    if not len(active):
        return out
    prefix = np.cumsum(active) / np.arange(1, len(active) + 1)
    # The selection is applied as a threshold, so k may only end a block of tied
    # doses; otherwise ">= threshold" would pull in more tied positions than k
    # and drag the mean below the target the prefix was checked against.
    boundary = np.r_[active[:-1] > active[1:], True]
    ok = np.flatnonzero((prefix >= fragment_mean_dose) & boundary)
    if len(ok):
        k = int(ok[-1]) + 1
        out["reached_target"] = True
    else:
        k = len(active) if max_fallback_positions is None else max(
            1, min(len(active), int(max_fallback_positions)))
    thr = float(active[k - 1])
    sel = active[active >= thr]
    out.update({"threshold": thr, "n_selected": int(len(sel)),
                "selected_mean_dose": float(sel.mean())})
    if fragment_mean_dose > 0:
        out["dose_ratio_to_fragment"] = float(sel.mean() / fragment_mean_dose)
    return out


def dose_scales(natural_doses: dict, target="auto") -> dict:
    """Common dose target and per-fragment scale for a dose-matched ablation.

    ``natural_doses`` maps fragment -> mean ``|u_p| / |x_p|`` when the whole
    group contribution ``u`` is removed. Removing ``alpha * u`` instead removes
    ``alpha`` times that fraction exactly, since the decoder is linear, so
    ``alpha = target / natural`` puts every fragment at the same dose.

    ``target="auto"`` takes the minimum natural dose over fragments: every
    fragment then gets ``alpha <= 1``, i.e. a partial removal of something the
    model actually carries, and none is extrapolated past its own full
    ablation. A numeric target is used as given; fragments that need
    ``alpha > 1`` are flagged ``extrapolated``. Fragments whose group never
    fires (natural dose 0) cannot be dosed and get ``alpha = None``.
    """
    finite = {k: float(v) for k, v in natural_doses.items()
              if v is not None and np.isfinite(v) and v > 0}
    if target == "auto" or target is None:
        rule = "auto: minimum natural dose over fragments (alpha <= 1 for all)"
        tgt = min(finite.values()) if finite else float("nan")
    else:
        rule = "fixed fraction from the command line"
        tgt = float(target)
    alphas = {}
    for k, v in natural_doses.items():
        if k in finite and np.isfinite(tgt):
            a = tgt / finite[k]
            alphas[k] = {"alpha": float(a), "extrapolated": bool(a > 1.0 + 1e-9)}
        else:
            alphas[k] = {"alpha": None, "extrapolated": False}
    return {"target": float(tgt), "rule": rule, "fragments": alphas}


def covariance_factor(acts, eps: float = 0.0) -> tuple[torch.Tensor, dict]:
    """Symmetric square root of the centred covariance of ``acts`` (n, d).

    ``g @ L`` with ``g ~ N(0, I)`` then has covariance ``Sigma``. Also returns
    the spectrum summary, since the effective dimensionality is the reason an
    isotropic direction is a weak control.
    """
    a = torch.as_tensor(acts).double()
    if a.ndim != 2 or a.shape[0] < 2:
        raise ValueError("need an (n >= 2, d) activation matrix")
    a = a - a.mean(0, keepdim=True)
    cov = a.T @ a / (a.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(0.0) + eps
    root = (evecs * evals.sqrt()) @ evecs.T
    lam = evals.cpu().numpy()
    tot = float(lam.sum())
    info = {"n_positions": int(a.shape[0]), "d": int(a.shape[1]),
            "participation_ratio": float(tot * tot / max(float((lam ** 2).sum()), 1e-300)),
            "top1_variance_fraction": float(lam.max() / tot) if tot > 0 else float("nan"),
            "top10_variance_fraction": (float(np.sort(lam)[::-1][:10].sum() / tot)
                                        if tot > 0 else float("nan"))}
    return root.float(), info


def sample_directions(norms: torch.Tensor, d: int, generator: torch.Generator,
                      factor: torch.Tensor | None = None) -> torch.Tensor:
    """One random direction per position, rescaled to the given per-position norm.

    ``factor=None`` draws isotropically; otherwise each draw is ``g @ factor``,
    i.e. Gaussian with covariance ``factor @ factor.T``. Draws happen on the CPU
    generator so a run is reproducible across devices.
    """
    g = torch.randn(int(norms.shape[0]), d, generator=generator)
    v = g.to(norms.device)
    if factor is not None:
        v = v @ factor.to(norms.device, v.dtype)
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12) * norms[:, None]
