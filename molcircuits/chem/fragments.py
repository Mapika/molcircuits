"""Map SMARTS substructure matches onto SMILES token positions.

The mapping follows the atom order of the input SMILES as parsed by RDKit.

``load_smarts_library`` reads a JSON name -> SMARTS mapping and is committed with
the repo, so the fragment definitions behind every reported number are part of
the artifact rather than an uncommitted YAML file.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger

from .tokenizer import tokenize
from .valence import is_atom_token

RDLogger.DisableLog("rdApp.*")

DEFAULT_LIBRARY = {
    "amide": "[NX3][CX3](=[OX1])",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ether": "[OD2]([#6])[#6]",
    "alcohol": "[OX2H][CX4]",
    "primary_amine": "[NX3;H2;!$([NX3][CX3]=[OX1])]",
    "nitrile": "[NX1]#[CX2]",
    # RDKit sanitization normalizes neutral hypervalent nitro to the
    # charge-separated form, so [NX3](=O)=O alone can never match anything --
    # it scored 0 of 5000 molecules while still reporting WSD numbers from an
    # all-zero mask. The charge-separated pattern is therefore the only one that
    # fires on this corpus.
    #
    # It matches the WHOLE GROUP, N and both oxygens. The recursive form it
    # replaces, [$(...),$(...)], matches only the central nitrogen by
    # construction, so "nitro" was a ONE-ATOM fragment: a single [N+] token,
    # which is close to unique in this vocabulary. That is why it scored the
    # atlas's only AUROC of 1.000, and it made the atoms-per-match confound
    # (Spearman -0.72 between AUROC and match size) partly an artifact of the
    # library rather than a property of the model. Every other entry here marks
    # its whole group; this one now does too.
    "nitro": "[NX3+](=[OX1])[OX1-]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "thioether": "[SX2]([#6])[#6]",
    "trifluoromethyl": "[CX4](F)(F)F",
    "fluoro_aromatic": "[F][c]",
    "fluoro_aliphatic": "[F][CX4]",
    "chloro_aromatic": "[Cl][c]",
    "chloro_aliphatic": "[Cl][CX4]",
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "pyrimidine": "c1cncnc1",
    "imidazole": "c1cnc[nH]1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "piperidine": "C1CCNCC1",
    "piperazine": "C1CNCCN1",
    "morpholine": "C1COCCN1",
    "cyclohexane": "C1CCCCC1",
    "phenol": "[OX2H][c]",
    "aniline": "[NX3H2][c]",
    "tert_butyl": "[CX4]([CH3])([CH3])[CH3]",
    "isopropyl": "[CH1X4]([CH3])[CH3]",
    "alkene": "[CX3]=[CX3]",
}


def load_smarts_library(path: str | None = None) -> dict[str, str]:
    if path is None or not os.path.exists(path):
        return dict(DEFAULT_LIBRARY)
    with open(path) as fh:
        return json.load(fh)


def save_smarts_library(lib: dict[str, str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(lib, fh, indent=2, sort_keys=True)


def atom_token_positions(smiles: str) -> list[int]:
    """Token indices that correspond to atoms, in string order."""
    return [i for i, t in enumerate(tokenize(smiles)) if is_atom_token(t)]


def atom_index_to_token(smiles: str, mol: Chem.Mol | None = None) -> dict[int, int]:
    """RDKit atom index -> token index, for this exact SMILES string.

    The mapping is the IDENTITY over atom order: RDKit's SMILES parser numbers
    atoms in the order they appear in the string it parsed, so atom *i* is the
    *i*-th atom token of ``smiles``.

    This used to round-trip through ``MolToSmiles`` and read
    ``_smilesAtomOutputOrder``, falling back to ``range(N)`` inside a bare
    ``except``. Two things were wrong with that, and they cancelled:

    * ``_smilesAtomOutputOrder`` is a COMPUTED property, absent from
      ``GetPropsAsDict(includePrivate=True)`` unless ``includeComputed=True`` is
      also passed. It therefore raised ``KeyError`` for every molecule -- 0 of
      500 corpus molecules took the documented path -- and the fallback did all
      the work.
    * the fallback is the correct mapping, and the documented mechanism is not.
      ``_smilesAtomOutputOrder`` describes the order of the CANONICAL output
      string, not of ``smiles`` as written. For ``OCC`` it is ``[2, 1, 0]``,
      which maps atom 0 (the oxygen) onto a carbon token. About 4% of this
      corpus is non-canonical, so anyone "fixing" the KeyError would silently
      corrupt those molecules' fragment masks.

    The round trip and the except are gone so that cannot happen.
    """
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    atom_toks = atom_token_positions(smiles)
    return {a: atom_toks[a] for a in range(mol.GetNumAtoms())
            if a < len(atom_toks)}


def _mark_fragment_matches(mask: np.ndarray, mol: Chem.Mol, patt: Chem.Mol,
                           a2t: dict[int, int], offset: int) -> None:
    for match in mol.GetSubstructMatches(patt):
        for a in match:
            t = a2t.get(int(a))
            if t is not None and 0 <= t + offset < len(mask):
                mask[t + offset] = True


def fragment_token_mask(smiles: str, smarts: str, n_tokens: int,
                        offset: int = 0) -> np.ndarray:
    """Boolean mask over token positions covered by any match of `smarts`."""
    mask = np.zeros(n_tokens, dtype=bool)
    mol = Chem.MolFromSmiles(smiles)
    patt = Chem.MolFromSmarts(smarts)
    if mol is not None and patt is not None:
        _mark_fragment_matches(mask, mol, patt, atom_index_to_token(smiles, mol), offset)
    return mask


def build_masks(smiles_list: Sequence[str], library: dict[str, str], n_tokens: int,
                offset: int = 1) -> dict[str, np.ndarray]:
    """(B, T) mask per fragment for a batch. `offset=1` accounts for BOS."""
    out = {name: np.zeros((len(smiles_list), n_tokens), dtype=bool)
           for name in library}
    if not len(smiles_list):
        return out
    patterns = {}
    for name, smarts in library.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            patterns[name] = patt
    if not patterns:
        return out
    for b, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        a2t = atom_index_to_token(smi, mol)
        for name, patt in patterns.items():
            _mark_fragment_matches(out[name][b], mol, patt, a2t, offset)
    return out


def valid_token_mask(smiles_list: Sequence[str], n_tokens: int,
                     offset: int = 1) -> np.ndarray:
    """(B, T) mask of real (non-pad) token positions, atoms and syntax alike.

    NOT the right population for a fragment-specificity score -- use
    :func:`atom_token_mask` for that. See its docstring for why.
    """
    m = np.zeros((len(smiles_list), n_tokens), dtype=bool)
    for b, smi in enumerate(smiles_list):
        n = len(tokenize(smi))
        m[b, offset:min(offset + n, n_tokens)] = True
    return m


def atom_token_mask(smiles_list: Sequence[str], n_tokens: int,
                    offset: int = 1) -> np.ndarray:
    """(B, T) mask of ATOM token positions only.

    This is the population a fragment-specificity statistic has to be scored
    over, and scoring over :func:`valid_token_mask` instead is a confound rather
    than a nuisance.

    A SMARTS match marks atoms, so ``build_masks`` produces a positive set that
    is entirely atoms. If the negative set is every other real token, it
    contains the bonds, branch parentheses and ring digits as well -- and then
    any feature that merely distinguishes atoms from punctuation separates the
    two sets perfectly, for every fragment, without encoding any chemistry.

    Measured on 2,000 corpus molecules, a synthetic feature that is 1 on atom
    tokens and 0 elsewhere scores WSD 0.14-0.29 and clears its own
    within-molecule permutation null for 8 of 8 fragments tested -- a range that
    overlaps what real dictionaries score. The permutation null does not catch
    it, because shuffling fragment membership across *all* real tokens moves the
    positive set onto punctuation and drops the score. Restricting the
    population here takes the same synthetic feature to exactly 0.000.
    """
    m = np.zeros((len(smiles_list), n_tokens), dtype=bool)
    for b, smi in enumerate(smiles_list):
        for t in atom_token_positions(smi):
            if 0 <= t + offset < n_tokens:
                m[b, t + offset] = True
    return m


def token_identity_array(smiles_list: Sequence[str], n_tokens: int,
                         offset: int = 1) -> np.ndarray:
    """(B, T) array of token strings, in the same layout as the mask helpers.

    Needed by the atom-type-matched null, which has to know what kind of token
    each position holds in order to resample a fragment's composition rather
    than only its size. Positions outside a molecule's encoding hold ``""``.
    """
    out = np.full((len(smiles_list), n_tokens), "", dtype=object)
    for b, smi in enumerate(smiles_list):
        for t, tokenval in enumerate(tokenize(smi)):
            if 0 <= t + offset < n_tokens:
                out[b, t + offset] = tokenval
    return out


def coverage(masks: dict[str, np.ndarray], valid: np.ndarray) -> dict[str, float]:
    return {k: float((v & valid).sum() / max(valid.sum(), 1)) for k, v in masks.items()}


def iter_library(library: dict[str, str]) -> Iterable[tuple[str, str]]:
    return sorted(library.items())
