"""Is an intervention effect *valence-shaped*, or just a common-mode shift?

Pushing a genuine remaining-capacity direction should reallocate probability
between bond orders: suppress the single bond and raise the double and triple.
A direction that merely rescales the whole next-token distribution moves all
three the same way, which looks like a large effect in any single number but
encodes nothing about valence.  The distinction is the substance of the valence
section, so it is written down here as a predicate rather than re-derived by eye
from a table of deltas.

    valence-shaped at alpha > 0  <=>  dprob_- < 0,  dprob_= > 0,  dprob_# > 0

Four signatures are checked, and they are not redundant -- a rescaling passes
some of them:

* **sign pattern** -- the pattern above, with negative alphas folded onto the
  positive convention so that a direction is not credited twice for one effect.
* **common mode** -- all three moving together, which is the failure the sign
  pattern is there to catch.
* **reallocation** -- ``dprob_-`` opposite in sign to ``dprob_= + dprob_#``, so
  mass moves *between* bond orders rather than into or out of them jointly.
* **sign reversal and monotonicity** in alpha -- a real direction should reverse
  when pushed the other way and scale with how hard it is pushed.  A
  thresholding artefact typically does neither.

Every function takes the already-persisted ``intervention`` rows, so this module
imports nothing from torch and can be exercised on a fixture.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

#: Persisted per-bond-order probability deltas, in single/double/triple order.
BOND_PROB_KEYS = ("dprob_-", "dprob_=", "dprob_#")

#: Persisted per-bond-order logit deltas.
BOND_LOGIT_KEYS = ("dlogit_-", "dlogit_=", "dlogit_#")

#: The signature a remaining-capacity direction should produce at alpha > 0.
VALENCE_SHAPED_SIGNS = (-1, 1, 1)

#: A ``|f(+a) + f(-a)|`` this small, relative to the larger of the two, counts
#: as the effect reversing with the sign of alpha.  Loose on purpose: an exactly
#: antisymmetric response is not expected from a nonlinear model, and the claim
#: being tested is directional, not quantitative.
REVERSAL_TOLERANCE = 0.35


def _signs(values: Sequence[float]) -> tuple[int, ...]:
    """Sign of each delta, with zero counted as negative.

    Ties are pushed to -1 rather than dropped so that the three-way pattern
    always has three entries; an exact zero in a float probability delta means
    "no measured movement", which is not evidence of the positive direction the
    valence-shaped pattern requires.
    """
    return tuple(1 if v > 0 else -1 for v in values)


def fold_to_positive_alpha(row: Mapping, alpha: float,
                           keys: Sequence[str] = BOND_PROB_KEYS) -> list[float]:
    """Deltas expressed in the alpha > 0 convention.

    A direction that reverses correctly produces mirror-image deltas at -alpha.
    Folding lets both halves of the sweep test the same hypothesis instead of
    the negative half appearing to contradict it.
    """
    values = [float(row[k]) for k in keys]
    return [-v for v in values] if alpha < 0 else values


def shape_signs(row: Mapping, alpha: float) -> dict:
    """Classify one (alpha, row) cell's sign pattern."""
    folded = fold_to_positive_alpha(row, alpha)
    got = _signs(folded)
    n_matching = sum(g == w for g, w in zip(got, VALENCE_SHAPED_SIGNS))
    return {
        "alpha": float(alpha),
        "signs": got,
        "n_matching": int(n_matching),
        "valence_shaped": got == VALENCE_SHAPED_SIGNS,
        "common_mode": len(set(got)) == 1,
    }


def reallocates(row: Mapping) -> bool:
    """Whether single-bond mass moves opposite to double-plus-triple.

    Double and triple are summed rather than tested separately because the
    claim is about mass leaving one side of the bond-order split for the other;
    a triple-bond delta that is tiny and noisy should not be able to veto that.
    """
    single = float(row[BOND_PROB_KEYS[0]])
    higher = float(row[BOND_PROB_KEYS[1]]) + float(row[BOND_PROB_KEYS[2]])
    return (single > 0) != (higher > 0)


def sign_reverses(rows_by_alpha: Mapping[float, Mapping], key: str,
                  tolerance: float = REVERSAL_TOLERANCE) -> bool:
    """Whether ``key`` flips sign between matched +/- alpha pairs.

    Returns ``False`` when the sweep contains no matched pair.  The predecessor
    of this function used a bare :func:`all` over the pairs it found, so a sweep
    with only positive alphas reported reversal vacuously -- an absence of
    evidence rendered as evidence.
    """
    pairs = [(a, -a) for a in rows_by_alpha
             if a > 0 and -a in rows_by_alpha]
    if not pairs:
        return False
    for pos, neg in pairs:
        up = float(rows_by_alpha[pos][key])
        down = float(rows_by_alpha[neg][key])
        scale = max(abs(up), abs(down), 1e-30)
        if abs(up + down) >= tolerance * scale:
            return False
    return True


def monotone(rows_by_alpha: Mapping[float, Mapping], key: str) -> bool:
    """Whether ``key`` is non-decreasing or non-increasing across alpha."""
    values = [float(rows_by_alpha[a][key]) for a in sorted(rows_by_alpha)]
    if len(values) < 2:
        return False
    ups = all(b >= a for a, b in zip(values, values[1:]))
    downs = all(b <= a for a, b in zip(values, values[1:]))
    return ups or downs


def linear_response(rows_by_alpha: Mapping[float, Mapping], key: str) -> dict:
    """Slope per RMS unit and R^2 of ``key`` regressed on alpha.

    A direction whose effect is a smooth function of how hard it is pushed
    should be close to linear over a modest range; a thresholding artefact is
    typically not, which is what the R^2 is here to expose.
    """
    alphas = sorted(rows_by_alpha)
    if len(alphas) < 2:
        return {"slope": float("nan"), "r2": float("nan"), "n_alpha": len(alphas)}
    x = np.asarray(alphas, dtype=float)
    y = np.asarray([float(rows_by_alpha[a][key]) for a in alphas], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = ((y - (slope * x + intercept)) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return {
        "slope": float(slope),
        "r2": float(1.0 - residual / max(total, 1e-30)),
        "n_alpha": len(alphas),
    }


def summarize_direction(rows_by_alpha: Mapping[float, Mapping]) -> dict:
    """Every signature at once, for one fitted direction in one seed.

    ``n_alpha`` is reported beside each count because "valence-shaped at 2 of 2
    strengths" and "at 2 of 12" are not the same claim, and a bare count invites
    reading the first as the second.
    """
    cells = [shape_signs(rows_by_alpha[a], a) for a in sorted(rows_by_alpha)]
    n = len(cells)
    return {
        "n_alpha": n,
        "n_valence_shaped": sum(c["valence_shaped"] for c in cells),
        "n_common_mode": sum(c["common_mode"] for c in cells),
        "n_reallocating": sum(reallocates(rows_by_alpha[a]) for a in rows_by_alpha),
        "per_alpha": cells,
        "single_bond": {
            "sign_reverses": sign_reverses(rows_by_alpha, BOND_PROB_KEYS[0]),
            "monotone": monotone(rows_by_alpha, BOND_PROB_KEYS[0]),
            **linear_response(rows_by_alpha, BOND_PROB_KEYS[0]),
        },
    }


def rows_by_alpha(intervention: Mapping, arm: str = "probe_direction",
                  alpha_key: str = "alpha_rms_units") -> dict[float, dict]:
    """Index one arm of a persisted ``intervention`` payload by strength.

    Alpha is keyed in RMS units of the residual stream.  The earlier sweep
    expressed it as a multiple of the target's own code values, which put the
    nominal ``alpha=1`` several times the residual norm and made every effect at
    that strength a description of a destroyed forward pass; keeping the unit in
    the key name is what stops the two being compared by accident.
    """
    return {float(r[alpha_key]): dict(r) for r in intervention[arm]}
