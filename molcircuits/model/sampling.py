"""Autoregressive sampling with optional activation edits, plus generation metrics.

`generate` takes the same HookManager as everything else, so a steered run and
its alpha=0 control differ only in the callback, never in the code path.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..chem.tokenizer import SmilesTokenizer


@torch.no_grad()
def generate(model, tok: SmilesTokenizer, n: int, max_tokens: int = 128,
             temperature: float = 1.0, top_k: int | None = None,
             hooks=None, seed: int = 0, batch_size: int = 512,
             device: str = "cpu", meta: dict | None = None,
             use_kv_cache: bool | None = None) -> list[str]:
    """Sample ``n`` molecules. ``meta``, if given, is filled with per-sequence
    termination facts.

    Set ``use_kv_cache=False`` for a full-prefix reference. With hooks,
    explicitly enabling the cache at the context limit requires edits that act
    independently at each token. Attention-pattern hooks need the full prefix.

    ``meta["terminated"]`` is a bool per returned sequence: did it emit EOS, or
    was it cut off at ``max_tokens``? Without this, an unterminated sequence is
    indistinguishable from a finished one in the returned strings, and a
    perturbation that stops the model from ever emitting EOS is scored as
    "invalid chemistry" when it is really "not finished". Under residual-stream
    edits large enough to matter, that is the common case rather than the edge
    case, so validity has to be read next to the termination rate.
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    out: list[str] = []
    finished: list[bool] = []
    n_layers = model.cfg.n_layers
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        seq = torch.full((b, 1), tok.bos_id, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        # Per-layer KV cache: without it each sampled token re-runs the entire
        # prefix, making generation O(T^2) in the number of tokens. Falls back to
        # the uncached path when the context would overflow, and when a hook
        # captures attention patterns (which need the full square).
        # The final call predicts from positions 0 through max_tokens - 1.
        # A window exactly as long as the generation horizon therefore fits.
        within_window = max_tokens < model.cfg.n_ctx or (
            max_tokens == model.cfg.n_ctx and (hooks is None or use_kv_cache is True))
        use_cache = use_kv_cache is not False and within_window and not (
            hooks is not None and any(
                hooks.wants(f"blocks.{i}.attn.hook_pattern") for i in range(n_layers)))
        cache = [{} for _ in range(n_layers)] if use_cache else None
        step_in = seq
        for step in range(max_tokens):
            logits = model(step_in if use_cache else seq[:, -model.cfg.n_ctx:],
                           hooks=hooks, cache=cache)[:, -1]
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = logits.softmax(-1)
                nxt = torch.multinomial(probs, 1, generator=g)
            nxt = torch.where(done[:, None], torch.full_like(nxt, tok.pad_id), nxt)
            seq = torch.cat([seq, nxt], dim=1)
            step_in = nxt
            done |= nxt.squeeze(1) == tok.eos_id
            # `bool(done.all())` is a device sync, and at one per step it is a
            # measurable fraction of a long sweep. Checking every 8 steps is
            # OUTPUT-NEUTRAL, not an approximation: once every row is done, each
            # further step forces `nxt` to pad for every row, so the extra
            # tokens decode to nothing and the strings are unchanged.
            if step % 8 == 7 and bool(done.all()):
                break
        for row in seq.tolist():
            out.append(tok.decode(row))
        finished.extend(bool(v) for v in done.tolist())
    if meta is not None:
        meta["terminated"] = finished[:n]
        meta["n_terminated"] = int(sum(finished[:n]))
        meta["termination_rate"] = meta["n_terminated"] / max(len(out[:n]), 1)
        meta["max_tokens"] = max_tokens
    return out[:n]


@dataclass
class GenerationMetrics:
    n: int
    validity: float
    uniqueness: float
    novelty: float
    n_scaffolds: int
    mean_qed: float
    max_tokens: int
    temperature: float

    def as_dict(self) -> dict:
        return self.__dict__


def canonical_or_none(smiles: str):
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(m) if m is not None else None


def evaluate_generations(smiles: list[str], train_canon: set[str] | None = None,
                         max_tokens: int = 128, temperature: float = 1.0,
                         canonicalize_reference: bool = True
                         ) -> GenerationMetrics:
    """Validity / uniqueness / novelty / scaffold count.

    ``max_tokens`` and ``temperature`` are recorded in the result because the old
    project produced two different baseline validity figures (49.9% and 100%)
    that differed only by generation budget; any validity number without its
    settings attached is uninterpretable.
    """
    from rdkit import Chem, RDLogger  # noqa: PLC0415
    from rdkit.Chem import QED  # noqa: PLC0415
    from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")

    # Canonicalize the reference side as well. The generated side goes through
    # MolToSmiles, so comparing it against raw corpus strings counts a perfectly
    # memorized molecule as novel whenever the corpus spelling is not RDKit
    # canonical -- 3.8% of this corpus, biased one way.
    if train_canon and canonicalize_reference:
        train_canon = {c for c in (canonical_or_none(s) for s in train_canon) if c}
    canon, scaffolds, qeds = [], set(), []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        canon.append(Chem.MolToSmiles(m))
        try:
            scaffolds.add(Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
            qeds.append(QED.qed(m))
        except Exception:
            pass
    n = max(len(smiles), 1)
    uniq = set(canon)
    novel = uniq - train_canon if train_canon else uniq
    return GenerationMetrics(
        n=len(smiles),
        validity=len(canon) / n,
        uniqueness=len(uniq) / max(len(canon), 1),
        novelty=len(novel) / max(len(uniq), 1),
        n_scaffolds=len(scaffolds),
        mean_qed=float(sum(qeds) / len(qeds)) if qeds else float("nan"),
        max_tokens=max_tokens,
        temperature=temperature,
    )
