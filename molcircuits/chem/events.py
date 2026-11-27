"""Ring-closure and branch-closure events in a tokenized SMILES string.

An *event* is a position where the grammar fully determines a matching earlier
token: a ring digit that closes an earlier one, or a ')' that closes an earlier
'('. These are the ground truth for the pointer analyses.

Fixed relative to the previous implementation
---------------------------------------------
The old scanner called the ring-id reader at every index with no skip-ahead, so a
two-digit label ``%10`` fired three times -- once correctly as "10", then twice
more as phantom "1" and "0" events. Here the tokenizer already emits ``%10`` as a
single token, and the scanner advances by whole tokens, so the failure mode is
structurally impossible. ``test_events.py`` pins it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Event:
    """A closing token and the opening token it resolves.

    Attributes
    ----------
    open_idx, close_idx : token indices of the opener and the closer
    predict_idx : the position whose *next-token* prediction is the closer, i.e.
        ``close_idx - 1``. All margin/ablation measurements happen here.
    kind : "ring" or "branch"
    span : close_idx - open_idx, the syntactic distance being bridged
    depth : branch nesting depth at the closer (branch events only)
    label : the ring identifier, for ring events
    """
    open_idx: int
    close_idx: int
    predict_idx: int
    kind: str
    span: int
    depth: int = 0
    label: str = ""


def _is_ring_token(tok: str) -> bool:
    return tok.isdigit() or (tok.startswith("%") and tok[1:].isdigit())


def _ring_label(tok: str) -> str:
    return tok[1:] if tok.startswith("%") else tok


def find_events(tokens: Sequence[str], offset: int = 0) -> list[Event]:
    """Extract all ring and branch events from a token sequence.

    ``offset`` is added to every index, so callers that prepend a BOS token can
    pass ``offset=1`` and get indices into the model's input sequence directly.

    Unclosed rings and unbalanced parentheses are dropped rather than raising:
    the model is expected to be fed only valid strings, but generated samples may
    not be, and an analysis pass should not crash on them.
    """
    events: list[Event] = []
    ring_open: dict[str, list[int]] = {}
    paren_stack: list[int] = []

    for i, tok in enumerate(tokens):
        if _is_ring_token(tok):
            lab = _ring_label(tok)
            stack = ring_open.setdefault(lab, [])
            if stack:
                j = stack.pop()
                events.append(Event(
                    open_idx=j + offset,
                    close_idx=i + offset,
                    predict_idx=i - 1 + offset,
                    kind="ring",
                    span=i - j,
                    label=lab,
                ))
            else:
                stack.append(i)
        elif tok == "(":
            paren_stack.append(i)
        elif tok == ")":
            if paren_stack:
                j = paren_stack.pop()
                events.append(Event(
                    open_idx=j + offset,
                    close_idx=i + offset,
                    predict_idx=i - 1 + offset,
                    kind="branch",
                    span=i - j,
                    depth=len(paren_stack) + 1,
                ))
    return events


def control_positions(
    tokens: Sequence[str],
    events: Sequence[Event],
    offset: int = 0,
    rng=None,
    n: int | None = None,
) -> list[int]:
    """Non-event prediction positions from the same molecule.

    These are the matched controls for the event-specificity contrast. Without
    them a head that is globally important for next-token prediction is
    indistinguishable from one that matters specifically at ring closures.
    """
    taken = {e.predict_idx for e in events}
    cand = [i + offset for i in range(len(tokens) - 1) if i + offset not in taken]
    if n is not None and rng is not None and len(cand) > n:
        idx = rng.choice(len(cand), size=n, replace=False)
        cand = [cand[i] for i in sorted(idx)]
    return cand
