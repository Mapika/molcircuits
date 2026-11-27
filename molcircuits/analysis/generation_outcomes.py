"""Generation outcomes whose denominator cannot hide invalid samples.

For molecular steering, ``contains the requested fragment`` is only a success
when the generated string is also a valid molecule.  Reporting hits divided by
valid generations conditions on an intervention-dependent post-treatment
variable: an arm can look better simply by destroying most of its samples.

The primary endpoint here is therefore the *joint* rate

    P(valid and contains fragment) = n_hits / n_attempted.

The conditional hit rate and validity are retained as secondary diagnostics.
"""
from __future__ import annotations

import math
from typing import Iterable


def wilson_interval(
        n_success: int, n_total: int, z: float = 1.96,
) -> tuple[float | None, float | None]:
    """Wilson score interval from integer counts.

    An interval is undefined when there is no denominator.  JSON ``null`` is
    used for that state rather than a non-standard IEEE NaN token.
    """
    if n_total <= 0:
        return None, None
    p = n_success / n_total
    d = 1.0 + z * z / n_total
    c = (p + z * z / (2 * n_total)) / d
    h = z * math.sqrt(p * (1 - p) / n_total + z * z / (4 * n_total**2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fragment_outcomes(smiles: Iterable[str], smarts: str) -> dict:
    """Count fragment hits, validity, and joint success over all attempts."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"invalid SMARTS: {smarts!r}")

    samples = list(smiles)
    n_attempted = len(samples)
    n_valid = 0
    n_hits = 0
    for value in samples:
        mol = Chem.MolFromSmiles(value) if value else None
        if mol is None:
            continue
        n_valid += 1
        n_hits += int(mol.HasSubstructMatch(query))

    conditional = n_hits / n_valid if n_valid else None
    joint = n_hits / n_attempted if n_attempted else None
    validity = n_valid / n_attempted if n_attempted else None
    return {
        "n_attempted": n_attempted,
        "n_valid": n_valid,
        "n_fragment_hits": n_hits,
        "validity": validity,
        "validity_ci95": list(wilson_interval(n_valid, n_attempted)),
        "fragment_rate_given_valid": conditional,
        "fragment_rate_given_valid_ci95": list(wilson_interval(n_hits, n_valid)),
        "fragment_success_rate": joint,
        "fragment_success_ci95": list(wilson_interval(n_hits, n_attempted)),
    }


def normalized_outcome(row: dict, n_attempted: int | None = None) -> dict:
    """Read new rows and losslessly upgrade legacy conditional-rate rows.

    Older artifacts store ``fragment_rate = n_hits / n_valid`` and ``n_valid``.
    Their integer hit count is recoverable up to floating-point roundoff.  This
    adapter lets corrected figures use the existing generation runs without
    pretending that the old field had the new denominator.
    """
    attempted = int(row.get("n_attempted", n_attempted or 0))
    valid = int(row.get("n_valid", 0))
    if "n_fragment_hits" in row:
        hits = int(row["n_fragment_hits"])
    elif valid == 0:
        # With no valid molecules the exact count is known to be zero even
        # when the historical conditional rate was serialized as NaN.
        hits = 0
    else:
        legacy = row.get("fragment_rate_given_valid", row.get("fragment_rate"))
        try:
            conditional = float(legacy)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "a finite legacy conditional rate is required when n_valid > 0"
            ) from exc
        if not math.isfinite(conditional) or not 0.0 <= conditional <= 1.0:
            raise ValueError(
                "a finite legacy conditional rate in [0, 1] is required "
                "when n_valid > 0"
            )
        hits = int(round(conditional * valid))
    if attempted <= 0:
        raise ValueError("n_attempted is required to normalize a steering outcome")
    if not (0 <= hits <= valid <= attempted):
        raise ValueError(f"inconsistent outcome counts: {hits=}, {valid=}, {attempted=}")
    return {
        "n_attempted": attempted,
        "n_valid": valid,
        "n_fragment_hits": hits,
        "validity": valid / attempted,
        "validity_ci95": list(wilson_interval(valid, attempted)),
        "fragment_rate_given_valid": hits / valid if valid else None,
        "fragment_rate_given_valid_ci95": list(wilson_interval(hits, valid)),
        "fragment_success_rate": hits / attempted,
        "fragment_success_ci95": list(wilson_interval(hits, attempted)),
    }
