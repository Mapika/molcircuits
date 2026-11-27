"""Ring / branch pointer analysis.

What changed relative to the previous implementation
----------------------------------------------------
1. **The margin is computed correctly.** The old code closed over the baseline
   gold logit when computing the ablated margin, so ``delta_margin`` reduced
   algebraically to ``best_other_abl - best_other_base`` and the ablated gold
   logit never entered. ``margin_at`` here takes one logit tensor and uses it for
   both terms; ``test_pointers.py`` pins the identity.
2. **The ablation type is a parameter**, not an implicit choice, and both zero and
   mean ablation are reported.
3. **All heads are swept**, so any single head can be quoted as a percentile of
   the full distribution rather than against ten sampled heads.
4. **Event and control effects are reported separately** as well as differenced.
   The difference (previously called "Causal ES") is a specificity index; a head
   with a large control term is globally important, not event-specific, and the
   summary makes that visible instead of hiding it inside one number.
"""
from __future__ import annotations

import numpy as np
import torch

from ..chem.events import Event, control_positions, find_events
from ..chem.tokenizer import tokenize
from ..model.hooks import HookManager, attn_pattern


def margin_at(logits: torch.Tensor, targets: torch.Tensor,
              positions: torch.Tensor) -> torch.Tensor:
    """logit[gold] - max_{v != gold} logit[v], at the given (batch, pos) pairs.

    positions: (N, 2) long tensor of (batch_index, token_index).
    """
    b, t = positions[:, 0], positions[:, 1]
    lg = logits[b, t]                                   # (N, V)
    gold = targets[b, t]                                # (N,)
    gold_logit = lg.gather(1, gold[:, None]).squeeze(1)
    masked = lg.scatter(1, gold[:, None], torch.finfo(lg.dtype).min)
    return gold_logit - masked.max(dim=-1).values


def _positions_tensor(pairs: list[tuple[int, int]], device) -> torch.Tensor:
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.long, device=device)
    return torch.tensor(pairs, dtype=torch.long, device=device)


def build_event_index(smiles: list[str], tok, n_ctx: int, kind: str,
                      max_events: int | None = None, seed: int = 0,
                      n_controls_per_mol: int = 4):
    """Collect events and matched control positions across a molecule batch.

    Events are sampled *at random* when capped, not truncated to the first N in
    corpus order, so results are not conditioned on early batches.
    """
    rng = np.random.default_rng(seed)
    ev_pairs, ev_meta, per_mol_tokens = [], [], {}
    for bi, smi in enumerate(smiles):
        toks = tokenize(smi)
        all_evs = find_events(toks, offset=1)
        evs = [e for e in all_evs if e.kind == kind]
        evs = [e for e in evs if e.close_idx < n_ctx and e.predict_idx >= 0]
        if not evs:
            continue
        per_mol_tokens[bi] = (toks, all_evs)
        for e in evs:
            ev_pairs.append((bi, e.predict_idx))
            ev_meta.append(e)

    # Cap events BEFORE drawing controls. Drawing them first and capping after
    # made the control population a sample of every molecule while the events
    # were a sample of a subset: a third of controls then came from molecules
    # with no surviving event, events were weighted by event count while
    # controls were weighted per molecule, and the ring and branch analyses
    # shared one generic control pool. Every relative-specificity number derived
    # from that comparison was of an unmatched contrast.
    if max_events is not None and len(ev_pairs) > max_events:
        idx = rng.choice(len(ev_pairs), size=max_events, replace=False)
        ev_pairs = [ev_pairs[i] for i in idx]
        ev_meta = [ev_meta[i] for i in idx]

    # Controls are drawn per SURVIVING event, from that event's own molecule, so
    # the control population matches the event population in molecule membership
    # and is weighted the same way. Excluding *every* closure event from the pool
    # (not just this kind's) keeps the strongest non-target pointer events out of
    # the baseline: passing only same-kind events made 11% of ring controls
    # branch closures and 8% of branch controls ring closures.
    events_per_mol: dict[int, int] = {}
    for bi, _ in ev_pairs:
        events_per_mol[bi] = events_per_mol.get(bi, 0) + 1
    ctrl_pairs = []
    for bi, n_events in sorted(events_per_mol.items()):
        toks, all_evs = per_mol_tokens[bi]
        want = n_controls_per_mol * n_events
        drawn = control_positions(toks, all_evs, offset=1, rng=rng, n=want)
        ctrl_pairs.extend((bi, p) for p in drawn if p < n_ctx - 1)
    return ev_pairs, ev_meta, ctrl_pairs


@torch.no_grad()
def pointer_mass(model, tokens: torch.Tensor, events: list[Event],
                 pairs: list[tuple[int, int]], device: str = "cpu") -> np.ndarray:
    """Attention mass from each closer onto its own opener, for every head.

    Returns an (n_layers, n_heads) array in [0, 1].
    """
    cfg = model.cfg
    out = np.zeros((cfg.n_layers, cfg.n_heads))
    counts = 0
    hooks = HookManager().capture(*[attn_pattern(l) for l in range(cfg.n_layers)])
    model(tokens.to(device), hooks=hooks)
    for l in range(cfg.n_layers):
        pat = hooks[attn_pattern(l)]                     # (B, H, T, T)
        for (bi, _), e in zip(pairs, events):
            # Query at predict_idx, not close_idx. Every causal measurement in
            # this module is made at predict_idx = close_idx - 1, and attention
            # is causally masked, so attention *at* close_idx cannot influence
            # the logits whose margin `head_effects` reports -- pairing the two
            # was comparing quantities that cannot be causally linked. The JAX
            # reference queries i_pred for the same reason.
            out[l] += pat[bi, :, e.predict_idx, e.open_idx].float().cpu().numpy()
    counts = max(len(events), 1)
    return out / counts


@torch.no_grad()
def mean_head_activations(model, tokens: torch.Tensor, device: str = "cpu",
                          pad_id: int | None = None) -> dict[int, torch.Tensor]:
    """Per-layer mean attention-head output, shaped (n_heads, d_head).

    This is the replacement value that ``mode="mean"`` ablation substitutes for a
    head. Padding positions are excluded: they are most of a right-padded batch,
    and averaging over them would drag the mean toward the model's response to
    ``<pad>`` -- which would quietly make mean ablation behave like zero ablation
    and destroy the contrast the two modes exist to draw.
    """
    names = [f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)]
    hooks = HookManager().capture(*names)
    tokens = tokens.to(device)
    model(tokens, hooks=hooks)

    mask = torch.ones_like(tokens, dtype=torch.bool) if pad_id is None \
        else tokens != pad_id
    w = mask[:, None, :, None].float()               # (B, 1, T, 1)
    denom = w.sum(dim=(0, 2)).clamp_min(1.0)         # (1, 1)
    return {l: (hooks[n].float() * w).sum(dim=(0, 2)) / denom
            for l, n in enumerate(names)}            # (n_heads, d_head)


@torch.no_grad()
def head_effects(model, tokens: torch.Tensor, targets: torch.Tensor,
                 event_pairs, control_pairs, device: str = "cpu",
                 mode: str = "zero", mean_z: dict | None = None,
                 ignore_index: int = -100) -> list[dict]:
    """Ablate every head in turn; report event, control and difference.

    ``ignore_index`` must be the pad id. Molecules are right-padded to n_ctx here,
    but the model is pretrained on *packed* sequences and never sees a pad token,
    so scoring them is scoring pure out-of-distribution behaviour. At n_ctx=256
    with ~40-token molecules that is 84% of positions, which is enough to invert
    the sign of ``delta_ce`` -- the very statistic that is supposed to reveal when
    a head is globally important rather than event-specific.
    """
    ev = _positions_tensor(event_pairs, device)
    ct = _positions_tensor(control_pairs, device)
    tokens, targets = tokens.to(device), targets.to(device)

    base_logits = model(tokens)
    base_ev = margin_at(base_logits, targets, ev).mean().item() if len(ev) else float("nan")
    base_ct = margin_at(base_logits, targets, ct).mean().item() if len(ct) else float("nan")

    rows = []
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            mz = mean_z.get(l) if mean_z else None
            with model.ablate_head(l, h, mode=mode, mean_z=mz):
                lg = model(tokens)
                a_ev = margin_at(lg, targets, ev).mean().item() if len(ev) else float("nan")
                a_ct = margin_at(lg, targets, ct).mean().item() if len(ct) else float("nan")
                ce = torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.size(-1)).float(), targets.reshape(-1),
                    ignore_index=ignore_index).item()
            # Raw drops are differenced for continuity, but the *relative* form
            # is the one to rank on. Baseline margins differ ~2.3x between event
            # and control populations (10.6 vs 4.6), so a head that scales all
            # margins down by a common fraction scores positive on the raw
            # difference purely from that gap. Empirically the raw index
            # correlates 0.76 with delta_ce and 0.02 with pointer_mass -- it
            # ranks global damage, which is exactly what this module exists to
            # avoid conflating with event specificity.
            rel_ev = (base_ev - a_ev) / base_ev if base_ev else float("nan")
            rel_ct = (base_ct - a_ct) / base_ct if base_ct else float("nan")
            rows.append({
                "layer": l, "head": h, "mode": mode,
                "delta_margin_event": base_ev - a_ev,
                "delta_margin_control": base_ct - a_ct,
                "delta_margin_specificity": (base_ev - a_ev) - (base_ct - a_ct),
                "baseline_margin_event": base_ev,
                "baseline_margin_control": base_ct,
                "relative_drop_event": rel_ev,
                "relative_drop_control": rel_ct,
                "relative_specificity": rel_ev - rel_ct,
                "ce_ablated": ce,
                # Counts, so a caller pooling several batches can weight by
                # them. Averaging per-batch means (and per-batch RATIOS) makes
                # the result depend on batch boundaries: at 1000+24 batching a
                # top head's relative specificity read +0.283 against a true
                # +0.232, and 40 of 48 heads changed rank.
                "n_events": int(len(ev)),
                "n_controls": int(len(ct)),
                "ablated_margin_event": a_ev,
                "ablated_margin_control": a_ct,
                "n_scored_tokens": int((targets != ignore_index).sum()),
            })
    base_ce = torch.nn.functional.cross_entropy(
        base_logits.reshape(-1, base_logits.size(-1)).float(),
        targets.reshape(-1), ignore_index=ignore_index).item()
    for r in rows:
        r["delta_ce"] = r["ce_ablated"] - base_ce
    return rows


def summarize(rows: list[dict], key: str = "delta_margin_specificity") -> dict:
    """Percentile position of every head within the full 48-head distribution."""
    vals = np.array([r[key] for r in rows], dtype=float)
    order = vals.argsort()
    pct = np.empty_like(vals)
    pct[order] = np.linspace(0, 100, len(vals))
    for r, p in zip(rows, pct):
        r[f"{key}_percentile"] = float(p)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
            "n_heads": len(vals)}


def span_curve(rows_by_event: list[tuple[Event, float]], bins=(2, 4, 6, 11, 23, 64)) -> list[dict]:
    """Bin an event-level quantity by opener-closer span."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [v for e, v in rows_by_event if lo <= e.span < hi]
        out.append({"span_lo": lo, "span_hi": hi, "n": len(sel),
                    "mean": float(np.mean(sel)) if sel else float("nan")})
    return out
