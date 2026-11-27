"""Atom-level SMILES tokenizer.

Deliberately regex-based rather than learned-subword: every token is a chemically
meaningful unit, which is what makes token-level analyses (ring events, valence
labels, SMARTS spans) well defined in the first place.

`SmilesTokenizer.to_hf()` exports a `PreTrainedTokenizerFast` so checkpoints are
loadable by anything HuggingFace-shaped, including TransformerLens.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# Order matters: two-character elements and bracket atoms must be tried before
# their single-character prefixes, and %nn before a bare digit.
SMILES_PATTERN = (
    r"(\[[^\]]+\]"          # bracket atom, e.g. [nH], [C@@H], [O-]
    r"|%\d{2}"              # two-digit ring closure
    r"|Br|Cl"               # two-char elements that are not bracket atoms
    r"|[BCNOSPFIbcnosp]"    # organic subset
    r"|@@|@"                # chirality outside brackets (rare but legal)
    r"|[=#$:/\\+\-.()]"     # bonds, branches, disconnection, charges
    r"|\d)"                 # single-digit ring closure
)
_TOKEN_RE = re.compile(SMILES_PATTERN)

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]


def tokenize(smiles: str) -> list[str]:
    """Split a SMILES string into atom-level tokens.

    Raises ValueError if any character is unconsumed, so silent corruption of a
    molecule into <unk> soup is impossible at dataset-build time.
    """
    toks, pos = [], 0
    for m in _TOKEN_RE.finditer(smiles):
        if m.start() != pos:
            raise ValueError(
                f"unparsed characters {smiles[pos:m.start()]!r} at {pos} in {smiles!r}"
            )
        toks.append(m.group(0))
        pos = m.end()
    if pos != len(smiles):
        raise ValueError(f"trailing characters {smiles[pos:]!r} in {smiles!r}")
    return toks


def token_spans(smiles: str) -> list[tuple[int, int]]:
    """Character span of each token; needed to map RDKit atom indices to tokens."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(smiles)]


@dataclass
class SmilesTokenizer:
    vocab: dict[str, int] = field(default_factory=dict)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_corpus(cls, smiles_iter: Iterable[str]) -> "SmilesTokenizer":
        seen: set[str] = set()
        for s in smiles_iter:
            seen.update(tokenize(s))
        vocab = {t: i for i, t in enumerate(SPECIALS)}
        for t in sorted(seen):
            vocab.setdefault(t, len(vocab))
        return cls(vocab=vocab)

    @classmethod
    def load(cls, path: str) -> "SmilesTokenizer":
        with open(path) as fh:
            return cls(vocab=json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.vocab, fh, indent=2, sort_keys=False)

    # -- properties -----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.vocab[PAD]

    @property
    def bos_id(self) -> int:
        return self.vocab[BOS]

    @property
    def eos_id(self) -> int:
        return self.vocab[EOS]

    @property
    def unk_id(self) -> int:
        return self.vocab[UNK]

    # -- encode / decode ------------------------------------------------------
    def encode(self, smiles: str, add_special: bool = True) -> list[int]:
        ids = [self.vocab.get(t, self.unk_id) for t in tokenize(smiles)]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        inv = {i: t for t, i in self.vocab.items()}
        return "".join(inv[i] for i in ids if inv.get(i) not in SPECIALS)

    # -- interop --------------------------------------------------------------
    def to_hf(self):
        """Wrap as a PreTrainedTokenizerFast (WordLevel + regex pre-tokenizer)."""
        from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: PLC0415
        from tokenizers import Regex  # noqa: PLC0415
        from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

        tok = Tokenizer(models.WordLevel(vocab=self.vocab, unk_token=UNK))
        tok.pre_tokenizer = pre_tokenizers.Split(
            Regex(SMILES_PATTERN), behavior="isolated"
        )
        return PreTrainedTokenizerFast(
            tokenizer_object=tok,
            pad_token=PAD, bos_token=BOS, eos_token=EOS, unk_token=UNK,
        )
