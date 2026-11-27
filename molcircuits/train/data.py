"""Corpus construction, deterministic scaffold split, and sequence packing.

Two things here materially change the economics and the credibility of the runs.

**Packing.** Lead-like SMILES are ~40 tokens. Padding each to a 256-token context
wastes ~85% of the compute budget. ``PackedDataset`` concatenates molecules into
full-length sequences separated by EOS, which cuts the pretraining cost by
roughly 6x. That is the difference between "a day per seed" and "a few hours per
seed", which is what makes multi-seed replication affordable.

**Deterministic split.** The train/val/test assignment is a BLAKE2b hash of the
canonical Murcko scaffold SMILES -- not of the molecule -- so the split is a pure
function of the corpus and is reproducible on any machine with no manifest to
lose. ``split_of`` is the single source of truth.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset

from ..chem.tokenizer import SmilesTokenizer

SPLITS = ("train", "valid", "test")
DEFAULT_FRACTIONS = (0.70, 0.10, 0.20)


def scaffold_of(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m), isomericSmiles=False)
    except Exception:
        return None


def split_of(scaffold: str, fractions=DEFAULT_FRACTIONS, salt: str = "molcircuits-v1") -> str:
    """Deterministic bucket for a canonical scaffold SMILES.

    The hash key is the *scaffold*, so every molecule sharing a scaffold lands in
    the same split. Empty scaffolds (acyclic molecules) hash on the empty string
    and therefore all land together, which is the conservative choice.
    """
    h = hashlib.blake2b((salt + "|" + scaffold).encode(), digest_size=8).digest()
    u = int.from_bytes(h, "big") / float(1 << 64)
    a, b, _ = fractions
    return "train" if u < a else ("valid" if u < a + b else "test")


# ---------------------------------------------------------------- filtering
@dataclass
class LeadLikeFilter:
    mw: tuple[float, float] = (150.0, 500.0)
    logp: tuple[float, float] = (-1.0, 5.0)
    hbd_max: int = 5
    hba_max: int = 10
    tpsa: tuple[float, float] = (20.0, 140.0)
    rotb_max: int = 10
    rings: tuple[int, int] = (1, 6)
    qed_min: float = 0.4
    drop_pains: bool = True

    def __call__(self, smiles: str) -> str | None:
        from rdkit import Chem, RDLogger  # noqa: PLC0415
        from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors  # noqa: PLC0415
        from rdkit.Chem import FilterCatalog  # noqa: PLC0415
        RDLogger.DisableLog("rdApp.*")

        m = Chem.MolFromSmiles(smiles)
        if m is None:
            return None
        # strip salts: keep the largest organic fragment
        frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        m = max(frags, key=lambda f: f.GetNumHeavyAtoms())

        if not (self.mw[0] <= Descriptors.MolWt(m) <= self.mw[1]):
            return None
        if not (self.logp[0] <= Crippen.MolLogP(m) <= self.logp[1]):
            return None
        if rdMolDescriptors.CalcNumHBD(m) > self.hbd_max:
            return None
        if rdMolDescriptors.CalcNumHBA(m) > self.hba_max:
            return None
        if not (self.tpsa[0] <= rdMolDescriptors.CalcTPSA(m) <= self.tpsa[1]):
            return None
        if rdMolDescriptors.CalcNumRotatableBonds(m) > self.rotb_max:
            return None
        if not (self.rings[0] <= rdMolDescriptors.CalcNumRings(m) <= self.rings[1]):
            return None
        if QED.qed(m) < self.qed_min:
            return None
        if self.drop_pains:
            params = FilterCatalog.FilterCatalogParams()
            params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
            if FilterCatalog.FilterCatalog(params).HasMatch(m):
                return None
        return Chem.MolToSmiles(m)


# ---------------------------------------------------------------- datasets
class PackedDataset(Dataset):
    """Pre-tokenized corpus packed into contiguous n_ctx-length sequences."""

    def __init__(self, ids: np.ndarray, n_ctx: int):
        self.ids = ids
        self.n_ctx = n_ctx
        self.n = (len(ids) - 1) // n_ctx

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = i * self.n_ctx
        x = torch.from_numpy(self.ids[s:s + self.n_ctx].astype(np.int64))
        y = torch.from_numpy(self.ids[s + 1:s + 1 + self.n_ctx].astype(np.int64))
        return x, y

    @classmethod
    def build(cls, smiles: Iterable[str], tok: SmilesTokenizer, n_ctx: int,
              out_path: str | None = None) -> "PackedDataset":
        buf: list[int] = []
        for s in smiles:
            try:
                buf.extend(tok.encode(s))
            except ValueError:
                continue
        arr = np.asarray(buf, dtype=np.uint16 if tok.vocab_size < 65536 else np.int32)
        if out_path:
            np.save(out_path, arr)
        return cls(arr, n_ctx)

    @classmethod
    def load(cls, path: str, n_ctx: int) -> "PackedDataset":
        return cls(np.load(path), n_ctx)


class MoleculeDataset(Dataset):
    """One molecule per row, right-padded. Used by analyses that need token
    positions to line up with a specific SMILES string (all of them)."""

    def __init__(self, smiles: list[str], tok: SmilesTokenizer, n_ctx: int):
        self.smiles, self.tok, self.n_ctx = smiles, tok, n_ctx

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, i: int):
        ids = self.tok.encode(self.smiles[i])[: self.n_ctx]
        pad = self.n_ctx - len(ids)
        x = torch.tensor(ids + [self.tok.pad_id] * pad, dtype=torch.long)
        mask = torch.tensor([1] * len(ids) + [0] * pad, dtype=torch.bool)
        return x, mask, i


def iter_smiles(path: str) -> Iterator[str]:
    """Stream SMILES from .smi/.txt (one per line, first whitespace field)."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                yield line.split()[0]


def write_split_manifest(path: str, counts: dict[str, int], salt: str,
                         fractions=DEFAULT_FRACTIONS) -> None:
    """Persist exact split counts next to the data.

    The previous project shipped approximate counts and an unresolved TODO about
    whether the hash key was the scaffold or the whole molecule. Writing this
    file makes that question answerable from the artifact.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"hash_key": "canonical Murcko scaffold SMILES",
                   "salt": salt, "fractions": list(fractions),
                   "counts": counts}, fh, indent=2)
