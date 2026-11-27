"""Classify why a generated SMILES string fails, by grammatical category.

Overall validity is the wrong endpoint for a head-ablation experiment. Removing
any head the model relies on lowers validity, so a validity drop at an ablated
ring-pointer head says only that the head mattered, not that it mattered *to
rings*. That is the same confound `RelSpec` (eq:rel-spec) removes at the margin
level, and it has to be removed again at the generation level: the endpoint must
separate ring failures from branch failures from chemistry failures, so a
ring-specific effect is distinguishable from general damage.

Categories are mutually exclusive and assigned in this order:

    truncated        no EOS emitted -- the string was cut at max_tokens
    ring_unclosed    a ring label opened and never closed
    ring_unopened    a ring digit closed a label that was never opened
    ring_illegal     a self-bond or a duplicate ring closure bond
    paren_unclosed   '(' with no matching ')'
    paren_unopened   ')' with an empty branch stack
    parse            RDKit cannot read the string even unsanitized
    valence          reads as a graph, rejected on valence
    aromaticity      reads as a graph, rejected on kekulization/aromaticity
    other            sanitization failed for some other reason
    ok               parses and sanitizes

`truncated` is separated first and deliberately. A perturbation that suppresses
EOS produces strings with open rings and open branches for a reason that has
nothing to do with ring tracking, and scoring those as ring failures would
manufacture the result this experiment is meant to test. Rates are therefore
reported over terminated sequences, with the termination rate beside them.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .tokenizer import tokenize
from .valence import is_atom_token

SYNTAX_CATEGORIES = ("ring_unclosed", "ring_unopened", "ring_illegal",
                     "paren_unclosed", "paren_unopened")
CHEMISTRY_CATEGORIES = ("parse", "valence", "aromaticity", "other")
CATEGORIES = ("ok", "truncated", *SYNTAX_CATEGORIES, *CHEMISTRY_CATEGORIES)


def _is_ring_token(tok: str) -> bool:
    return tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit())


def _ring_label(tok: str) -> str:
    return tok[1:] if tok.startswith("%") else tok


def scan_grammar(tokens: Sequence[str]) -> dict[str, int]:
    """Count unmatched ring labels and parentheses, without any chemistry.

    Mirrors `events.find_events`'s pairing rule -- one stack per ring label, push
    on empty and pop otherwise -- but keeps what that function discards, which is
    exactly the failure information this experiment needs.
    """
    ring_open: dict[str, list[int]] = {}
    paren_stack: list[int] = []
    ring_unopened = paren_unopened = 0

    for i, tok in enumerate(tokens):
        if _is_ring_token(tok):
            stack = ring_open.setdefault(_ring_label(tok), [])
            if stack:
                stack.pop()
            else:
                stack.append(i)
        elif tok == "(":
            paren_stack.append(i)
        elif tok == ")":
            if paren_stack:
                paren_stack.pop()
            else:
                paren_unopened += 1

    # A label left on its stack was opened and never closed. `ring_unopened`
    # cannot be produced by this scanner (an unmatched closer is indistinguishable
    # from an opener until the string ends), so it stays 0 and is kept only so the
    # category set matches what a stricter parser would report.
    ring_unclosed = sum(len(s) for s in ring_open.values())
    return {
        "ring_unclosed": ring_unclosed,
        "ring_unopened": ring_unopened,
        "ring_illegal": scan_illegal_ring_bonds(tokens),
        "paren_unclosed": len(paren_stack),
        "paren_unopened": paren_unopened,
        "n_ring_tokens": sum(1 for t in tokens if _is_ring_token(t)),
        "n_paren_tokens": sum(1 for t in tokens if t in ("(", ")")),
        "n_tokens": len(tokens),
    }


def scan_illegal_ring_bonds(tokens) -> int:
    """Ring closures that are balanced but not legal bonds.

    Counting digit occurrences is not enough to say a ring closed correctly.
    ``C11`` and ``C1C1`` both have an even count for label 1, so a count-based
    scanner calls them balanced; the first closes a ring onto the atom that
    opened it, and the second re-bonds two atoms that the chain already bonded.
    RDKit rejects both, and before this check they fell through to ``parse``,
    which put a *ring* failure into the chemistry bucket and biased the
    ring-specific endpoint downward.

    Traversal mirrors :mod:`molcircuits.chem.valence`: atoms are numbered in
    emission order, ``(`` pushes the current atom and ``)`` pops it, so the
    partner of a ring closure is the atom that opened the label.
    """
    bonds: set[frozenset] = set()
    ring_at: dict[str, int] = {}
    prev_atom: int | None = None
    branch_stack: list[int | None] = []
    atom_i = -1
    illegal = 0
    for tok in tokens:
        if tok == "(":
            branch_stack.append(prev_atom)
        elif tok == ")":
            prev_atom = branch_stack.pop() if branch_stack else None
        elif _is_ring_token(tok):
            label = _ring_label(tok)
            if prev_atom is None:
                continue
            if label in ring_at:
                opener = ring_at.pop(label)
                bond = frozenset((opener, prev_atom))
                if opener == prev_atom or bond in bonds:
                    illegal += 1
                else:
                    bonds.add(bond)
            else:
                ring_at[label] = prev_atom
        elif is_atom_token(tok):
            atom_i += 1
            if prev_atom is not None:
                bonds.add(frozenset((prev_atom, atom_i)))
            prev_atom = atom_i
    return illegal


def classify(smiles: str, terminated: bool = True) -> str:
    """Single mutually exclusive failure category for one generated string."""
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")

    if not terminated:
        return "truncated"
    if not smiles:
        return "parse"

    g = scan_grammar(tokenize(smiles))
    for cat in SYNTAX_CATEGORIES:
        if g[cat]:
            return cat

    # sanitize=False separates "cannot be read as a graph at all" from "reads as
    # a graph but is not a legal molecule". Relying on RDKit's log text instead
    # would couple the category to a message string that changes between releases.
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return "parse"
    try:
        Chem.SanitizeMol(m)
    except Exception as exc:  # noqa: BLE001 -- RDKit raises several types here
        msg = str(exc).lower()
        if "valence" in msg:
            return "valence"
        if "kekulize" in msg or "aromatic" in msg:
            return "aromaticity"
        return "other"
    return "ok"


@dataclass
class ErrorProfile:
    """Category counts over one arm's generations, plus the rates that matter."""

    n: int = 0
    n_terminated: int = 0
    counts: Counter = field(default_factory=Counter)

    @property
    def termination_rate(self) -> float:
        return self.n_terminated / self.n if self.n else float("nan")

    def rate(self, category: str) -> float:
        """Category rate over *terminated* sequences.

        Truncated strings are excluded from the denominator rather than counted
        as successes or failures: they are a different event, and pooling them
        lets a change in EOS behaviour masquerade as a change in grammar.
        """
        return (self.counts[category] / self.n_terminated
                if self.n_terminated else float("nan"))

    @property
    def validity(self) -> float:
        return self.rate("ok")

    @property
    def ring_error_rate(self) -> float:
        return sum(self.counts[c] for c in ("ring_unclosed", "ring_unopened", "ring_illegal")) \
            / self.n_terminated if self.n_terminated else float("nan")

    @property
    def branch_error_rate(self) -> float:
        return sum(self.counts[c] for c in ("paren_unclosed", "paren_unopened")) \
            / self.n_terminated if self.n_terminated else float("nan")

    @property
    def chemistry_error_rate(self) -> float:
        return sum(self.counts[c] for c in CHEMISTRY_CATEGORIES) \
            / self.n_terminated if self.n_terminated else float("nan")

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "n_terminated": self.n_terminated,
            "termination_rate": self.termination_rate,
            "validity": self.validity,
            "ring_error_rate": self.ring_error_rate,
            "branch_error_rate": self.branch_error_rate,
            "chemistry_error_rate": self.chemistry_error_rate,
            "counts": dict(self.counts),
        }


def profile(smiles: Sequence[str], terminated: Sequence[bool]) -> ErrorProfile:
    p = ErrorProfile(n=len(smiles), n_terminated=int(sum(terminated)))
    for s, t in zip(smiles, terminated):
        p.counts[classify(s, bool(t))] += 1
    return p
