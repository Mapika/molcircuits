"""Remaining valence capacity as a per-token label.

At an atom token, "remaining capacity" is the additional bond order the atom can
still accept given its element, charge, and the bonds already written *to the
left* in the string. It is a deterministic function of the visible prefix, which
is exactly why a probe recovering it is evidence about accessibility rather than
about computation -- see the control-task baselines in
``analysis/valence_probe.py``.

Labels are clipped to 0..4.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

MAX_LABEL = 4

# Default (neutral) valences for the organic subset.
#
# A multi-valence table -- S: (2, 4, 6), P: (3, 5), halogens (1, 3, 5, 7), with
# the atom taking the smallest allowed valence covering the bonds it has -- was
# tried here and REVERTED. It is the standard SMILES rule and it is wrong for
# this codebase, on measurement:
#
#   * it gained nothing. `semantics="final"` already agreed with RDKit's
#     implicit-hydrogen count on 92,550 of 92,550 corpus atoms, sulfones
#     included: `CS(=O)(=O)C` gives S = 0 under both tables, and 0 is correct
#     (the S carries 1+2+2+1 = 6 bond orders, not 4). The prefix labels the
#     probe is fitted to did not move for a single atom.
#   * it broke aromatic sulfur. Aromatic ring bonds are charged 1.5 each, so a
#     thiophene S carries 3.0 and gets promoted from valence 2 to 4, reporting
#     1 remaining where RDKit says 0. That is 379 new disagreements, all of
#     them aromatic S.
#
# The clamp in `_remaining` is what makes the single table sufficient. If this
# is revisited, the multi-valence lookup must use integer bond counts, not the
# 1.5-weighted aromatic ledger, and
# tests/test_chem.py::test_final_semantics_matches_rdkit_implicit_hydrogens
# must be extended to aromatic S first -- it did not cover it, which is why the
# regression passed.
DEFAULT_VALENCE = {
    "B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2,
    "F": 1, "Cl": 1, "Br": 1, "I": 1,
}
BOND_ORDER = {"-": 1, "=": 2, "#": 3, "$": 4, ":": 1}

_BRACKET_RE = re.compile(r"\[(?P<iso>\d+)?(?P<sym>[A-Za-z][a-z]?)(?P<rest>[^\]]*)\]")


def _raw_symbol(tok: str) -> str | None:
    """Element symbol exactly as written (case preserved -> aromaticity)."""
    if tok.startswith("["):
        m = _BRACKET_RE.match(tok)
        return m.group("sym") if m else None
    if tok[:1].isalpha():
        return tok
    return None


def _element_of(tok: str) -> str | None:
    sym = _raw_symbol(tok)
    return sym.capitalize() if sym else None


def _is_aromatic_token(tok: str) -> bool:
    sym = _raw_symbol(tok)
    return bool(sym) and sym[0].islower()


def _charge_of(tok: str) -> int:
    if not tok.startswith("["):
        return 0
    m = _BRACKET_RE.match(tok)
    if not m:
        return 0
    rest = m.group("rest")
    q = 0
    for sign, val in (("+", 1), ("-", -1)):
        for mm in re.finditer(re.escape(sign) + r"(\d*)", rest):
            q += val * (int(mm.group(1)) if mm.group(1) else 1)
    return q


def _explicit_h(tok: str) -> int:
    if not tok.startswith("["):
        return 0
    m = _BRACKET_RE.match(tok)
    if not m:
        return 0
    mm = re.search(r"H(\d*)", m.group("rest"))
    if not mm:
        return 0
    return int(mm.group(1)) if mm.group(1) else 1


def is_atom_token(tok: str) -> bool:
    return _element_of(tok) is not None


@dataclass(frozen=True)
class ValenceLabel:
    token_idx: int
    element: str
    remaining: int
    aromatic: bool


def _remaining(el: str, charge: int, spent_bonds: float, expl_h: int):
    """Remaining bond capacity for one atom, or None for elements we do not model.

    Shared by the write-time snapshot and the completed-molecule path so the two
    semantics differ only in *when* they are evaluated, never in the arithmetic.
    """
    base = DEFAULT_VALENCE.get(el)
    if base is None:
        return None
    cap = base + charge if el in ("N", "P") else base - abs(charge)
    if el in ("O", "S") and charge > 0:
        cap = base + charge
    return int(max(0, min(MAX_LABEL, round(cap - spent_bonds - expl_h))))


def valence_labels(tokens: Sequence[str], offset: int = 0,
                   semantics: str = "prefix",
                   bond_decisions: list | None = None) -> list[ValenceLabel]:
    """Remaining capacity at every atom token, from the left-context only.

    This walks the string the way the model does, tracking ring-bond commitments
    and branch structure, so the label at position i depends on tokens < i only.

    ``semantics="prefix"`` (default) is that quantity, snapshotted when the atom
    token is written. ``semantics="final"`` instead reports capacity in the
    *completed* molecule -- exactly RDKit's implicit-hydrogen count -- which is
    what this function returned before the snapshot was added, and which makes a
    probe target that depends on tokens the model has not emitted yet. It is kept
    only so the two can be compared; it is not a left-context label and must not
    be used to argue that a probe is reading a running valence ledger.
    """
    if semantics not in ("prefix", "final"):
        raise ValueError(f"semantics must be 'prefix' or 'final', got {semantics!r}")
    used: dict[int, float] = {}
    charges: dict[int, int] = {}
    elements: dict[int, str] = {}
    arom: dict[int, bool] = {}
    exp_h: dict[int, int] = {}

    prev_atom: int | None = None
    branch_stack: list[int | None] = []
    ring_open: dict[str, tuple[int, float]] = {}
    pending_bond: float | None = None
    order: list[int] = []
    at_write: dict[int, int] = {}

    for i, tok in enumerate(tokens):
        if tok == "(":
            branch_stack.append(prev_atom)
            continue
        if tok == ")":
            prev_atom = branch_stack.pop() if branch_stack else None
            continue
        if tok in BOND_ORDER:
            # Snapshot the governing atom's capacity AS IT STANDS HERE, before
            # this bond is charged to it. `used[prev_atom]` already includes any
            # bonds written inside branches that have since closed, which is the
            # whole point: reading the atom's birth-time label instead gives the
            # carbon in C(=O)=O a capacity of 4 at the second bond decision when
            # 2 is correct, and the error grows with every branch.
            if (bond_decisions is not None and tok in ("-", "=", "#")
                    and i > 0 and prev_atom is not None):
                cap = _remaining(elements[prev_atom], charges[prev_atom],
                                 used.get(prev_atom, 0.0), exp_h[prev_atom])
                if cap is not None:
                    bond_decisions.append((i - 1 + offset, cap))
            pending_bond = BOND_ORDER[tok]
            continue
        if tok == ".":
            prev_atom, pending_bond = None, None
            continue
        if tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit()):
            lab = tok[1:] if tok.startswith("%") else tok
            b = pending_bond          # may be None: order not yet determined
            pending_bond = None
            if lab in ring_open:
                j, bj = ring_open.pop(lab)
                explicit = [x for x in (b, bj) if x is not None]
                if explicit:
                    order_ = max(explicit)
                else:
                    # An unannotated ring bond between two aromatic atoms is
                    # aromatic (order 1.5), otherwise single.
                    both_arom = arom.get(j, False) and arom.get(prev_atom, False)
                    order_ = 1.5 if both_arom else 1.0
                if prev_atom is not None:
                    used[prev_atom] = used.get(prev_atom, 0.0) + order_
                used[j] = used.get(j, 0.0) + order_
            elif prev_atom is not None:
                ring_open[lab] = (prev_atom, b)
            continue

        el = _element_of(tok)
        if el is None:
            continue

        idx = i
        order.append(idx)
        elements[idx] = el
        charges[idx] = _charge_of(tok)
        exp_h[idx] = _explicit_h(tok)
        arom[idx] = _is_aromatic_token(tok)
        used.setdefault(idx, 0.0)

        if prev_atom is not None:
            b = pending_bond
            if b is None:
                b = 1.5 if (arom[idx] and arom.get(prev_atom, False)) else 1.0
            used[idx] += b
            used[prev_atom] = used.get(prev_atom, 0.0) + b
        # Snapshot the label HERE, while `used[idx]` still reflects only what has
        # been written to the left. Reading `used[idx]` after the scan instead
        # lets bonds from later tokens -- the next atom, and any ring closure
        # back to this one -- count against it, which made 85% of labels a
        # function of the *completed* molecule rather than the visible prefix.
        at_write[idx] = _remaining(elements[idx], charges[idx],
                                   used.get(idx, 0.0), exp_h[idx])
        pending_bond = None
        prev_atom = idx

    out: list[ValenceLabel] = []
    for idx in order:
        el = elements[idx]
        if semantics == "prefix":
            remaining = at_write.get(idx)
        else:
            remaining = _remaining(el, charges[idx], used.get(idx, 0.0), exp_h[idx])
        if remaining is None:
            continue
        out.append(ValenceLabel(
            token_idx=idx + offset, element=el, remaining=remaining, aromatic=arom[idx]
        ))
    return out


def bond_decision_labels(tokens: Sequence[str],
                         offset: int = 0) -> list[tuple[int, int]]:
    """(prediction position, remaining capacity of the atom being extended).

    The intervention acts at bond-decision positions, but the probe direction was
    fit at *atom* positions -- only 11.6% of them coincide, and 79.6% of the
    injection sites were `(` tokens. A direction validated on one population and
    applied to another is not a matched intervention, so the near-null result it
    produced was uninterpretable either way.

    The governing atom at a bond decision is the most recent atom written, and
    the quantity that matters there is its remaining capacity *at that point* --
    the same prefix-snapshot semantics as ``valence_labels``.

    This used to look up the governing atom's label from ``valence_labels``,
    which is its capacity at the moment the atom itself was WRITTEN. Those two
    quantities differ as soon as the atom has been extended in between: in
    ``C(=O)=O`` the carbon is labelled 4 at the second bond decision, where 2 is
    correct, because the ``=O`` inside the closed branch is not counted. The
    label is now taken from the same running ledger ``valence_labels`` keeps, so
    the two cannot drift apart.
    """
    out: list[tuple[int, int]] = []
    valence_labels(tokens, offset=offset, bond_decisions=out)
    return out


def bond_decision_positions(tokens: Sequence[str], offset: int = 0) -> list[int]:
    """Prediction positions whose target is an explicit bond token.

    These are where a valence-tracking mechanism, if the model has one, would
    have to act. Interventions are restricted to them.
    """
    return [
        i - 1 + offset
        for i, tok in enumerate(tokens)
        if tok in ("-", "=", "#") and i > 0
    ]


def unlabelled_bond_decisions(tokens: Sequence[str], offset: int = 0) -> list[int]:
    """Bond-decision positions that ``bond_decision_labels`` cannot label.

    The two functions can disagree: the labeller skips a decision whose
    governing atom is an element outside ``DEFAULT_VALENCE`` (``[Si]`` and
    ``[Sn]`` are both in the tokenizer vocabulary and neither is modelled), or
    whose governing atom is undetermined. ``bond_decision_positions`` has no
    such guard, so any caller that index-aligns the two lists misaligns
    silently.

        >>> unlabelled_bond_decisions(tokenize("C[Si](C)=C"))
        [4]

    207 molecules of ``valid.smi`` contain ``[Si]``/``[Sn]``; none currently
    governs a bond decision, so this returns [] on the corpus today. It is here
    so that stops being luck.
    """
    labelled = {p for p, _ in bond_decision_labels(tokens, offset=offset)}
    return [p for p in bond_decision_positions(tokens, offset=offset)
            if p not in labelled]


def canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None
