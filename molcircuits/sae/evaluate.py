"""SAE fidelity evaluation: L0, CE recovered, dead features, generation quality.

``fraction_ce_recovered`` is the field-standard metric the previous paper omitted:

    (CE_ablate - CE_sae) / (CE_ablate - CE_clean)

with the ablation floor computed both ways (zero and mean), because they give
materially different denominators and quoting one without saying which is how a
weak dictionary looks strong.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..model.hooks import HookManager, resid_post


@torch.no_grad()
def _ce_with_edit(model, loader, device, layer, edit_fn, max_batches,
                  ignore_index: int = -100):
    """Token-weighted cross-entropy under an optional residual-stream edit.

    Two defects this signature exists to prevent, both of the same shape as the
    FVU one in ``evaluate_sae``:

    * it used to average per-BATCH means, so the result depended on batch
      grouping. On 100 real sequences the same data gave 0.55497 at batch 50
      and 0.56500 at batch 33 -- a 0.01 swing, about fifteen times the
      ``delta_ce`` this function is used to report. Full batches hid it, and a
      ragged tail would not.
    * it passed no ``ignore_index``, so on a right-padded loader the padding
      dominates. Measured on a one-molecule-per-row loader, ``ce_clean`` rose
      to 12.98 while the zero-ablation "floor" stayed at 4.16, and
      ``fraction_ce_recovered`` came out ABOVE 1 -- the metric inverts.
      ``atlas.core.ablation_delta_nll`` already defaults this to the pad id.

    Sums and one division; pass ``ignore_index=tok.pad_id`` for any padded
    loader.
    """
    tot = torch.zeros((), dtype=torch.float64, device=device)
    n_tok = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        hooks = HookManager()
        if edit_fn is not None:
            hooks.edit(resid_post(layer), edit_fn)
        logits = model(x, hooks=hooks if edit_fn is not None else None)
        flat_y = y.reshape(-1)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               flat_y, ignore_index=ignore_index,
                               reduction="sum").double()
        n_tok += int((flat_y != ignore_index).sum())
    return float(tot / max(n_tok, 1))


@torch.no_grad()
def evaluate_sae(model, sae, loader: DataLoader, device: str = "cpu",
                 layer: int | None = None, max_batches: int = 50,
                 ignore_index: int = -100, ce_cache: dict | None = None) -> dict:
    """Reconstruction fidelity and sparsity at one layer.

    Pass ``ignore_index=tok.pad_id`` for a right-padded loader: without it the
    padding dominates the cross-entropy and ``fraction_ce_recovered`` inverts
    past 1. The packed loader this is normally called with has no padding, so
    the default is the torch sentinel.

    ``ce_cache``: the clean, zero-ablation and mean-ablation cross-entropies do
    not depend on the SAE at all -- clean depends on nothing evaluated here,
    and the two floors only on the layer (the mean-ablation vector is a
    function of the model and the scored batches alone). Pass one dict across
    calls that share a loader, ``max_batches`` and ``ignore_index`` and each of
    those forward passes runs once instead of once per dictionary; the cached
    values are bitwise the ones a fresh pass computes. With no dict every call
    recomputes everything, exactly as before.
    """
    layer = sae.cfg.layer if layer is None else layer
    model.eval(); sae.eval()
    if ce_cache is None:
        ce_cache = {}

    # --- one pass with the SAE spliced into the residual stream, scoring BOTH
    # the sae-condition cross-entropy AND reconstruction fidelity.
    #
    # The fidelity statistics are computed on the hook's *pre-edit* activation,
    # which is bitwise the activation a clean forward produces at this layer
    # (nothing upstream of the edit differs), so folding the old separate
    # capture-only pass into this one changes no number -- it removes one full
    # forward pass over the scored batches per dictionary.
    #
    # FVU is accumulated as SUMS and divided ONCE, not computed per batch and
    # averaged. Two things were wrong with the per-batch form: a mean of ratios
    # is not the ratio of means, and the denominator used each batch's OWN mean,
    # so the variance a batch is scored against shrank with batch size. The same
    # activations could be made to yield almost any FVU by regrouping them.
    # Totals are exact in one pass: SS_tot = sum||a||^2 - N*||mean||^2.
    l0s, fired = [], torch.zeros(sae.cfg.d_sae, device=device)
    n_tok = 0
    ss_res = torch.zeros((), dtype=torch.float64, device=device)
    sum_sq = torch.zeros((), dtype=torch.float64, device=device)
    sum_a = torch.zeros(sae.cfg.d_in, dtype=torch.float64, device=device)
    ce_tot = torch.zeros((), dtype=torch.float64, device=device)
    n_tok_ce = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        box = {}

        def edit(t):
            a = t.reshape(-1, sae.cfg.d_in)
            xhat, z = sae(a)
            box["a"], box["xhat"], box["z"] = a, xhat, z
            return xhat.reshape(t.shape)

        hooks = HookManager().edit(resid_post(layer), edit)
        logits = model(x, hooks=hooks)
        flat_y = y.reshape(-1)
        ce_tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  flat_y, ignore_index=ignore_index,
                                  reduction="sum").double()
        n_tok_ce += int((flat_y != ignore_index).sum())

        a, xhat, z = box["a"], box["xhat"], box["z"]
        l0s.append((z > 0).float().sum(-1).mean().item())
        a64 = a.double()
        ss_res += (a64 - xhat.double()).pow(2).sum()
        sum_sq += a64.pow(2).sum()
        sum_a += a64.sum(0)
        n_tok += a.shape[0]
        fired += (z > 0).any(0).float()
    ce_sae = float(ce_tot / max(n_tok_ce, 1))

    mean_a = sum_a / max(n_tok, 1)
    ss_tot = sum_sq - n_tok * mean_a.pow(2).sum()
    fvu_total = float(ss_res / ss_tot.clamp_min(1e-12))
    # The mean-ablation replacement, computed ONCE over everything scored rather
    # than per batch. Taking it from the batch under evaluation leaks that
    # batch's own composition -- other molecules, and tokens to the right of the
    # position being predicted -- into what is supposed to be an information-free
    # baseline, which makes mean ablation look milder than it is.
    mean_vec = mean_a.float()

    # --- cross-entropy floors (sae-independent, hence cacheable)
    if "clean" not in ce_cache:
        ce_cache["clean"] = _ce_with_edit(model, loader, device, layer, None,
                                          max_batches, ignore_index)
    ce_clean = ce_cache["clean"]
    if ("zero", layer) not in ce_cache:
        ce_cache[("zero", layer)] = _ce_with_edit(
            model, loader, device, layer, lambda t: torch.zeros_like(t),
            max_batches, ignore_index)
    ce_zero = ce_cache[("zero", layer)]
    if ("mean", layer) not in ce_cache:
        ce_cache[("mean", layer)] = _ce_with_edit(
            model, loader, device, layer,
            lambda t: mean_vec.to(t.dtype).expand_as(t), max_batches,
            ignore_index)
    ce_mean = ce_cache[("mean", layer)]

    def recovered(floor: float) -> float:
        denom = floor - ce_clean
        return float((floor - ce_sae) / denom) if abs(denom) > 1e-9 else float("nan")

    return {
        "layer": layer,
        "variant": sae.cfg.variant,
        "expansion": sae.cfg.expansion,
        "d_sae": sae.cfg.d_sae,
        "l1_coeff": sae.cfg.l1_coeff,
        "k": sae.cfg.k,
        "l0": sum(l0s) / max(len(l0s), 1),
        "fvu": fvu_total,
        "n_tokens_scored": int(n_tok),
        "dead_feature_fraction": float((fired == 0).float().mean()),
        "ce_clean": ce_clean,
        "ce_sae": ce_sae,
        "ce_zero_ablate": ce_zero,
        "ce_mean_ablate": ce_mean,
        "delta_ce": ce_sae - ce_clean,
        "fraction_ce_recovered_vs_zero": recovered(ce_zero),
        "fraction_ce_recovered_vs_mean": recovered(ce_mean),
    }


def sae_edit_fn(sae, alpha: float = 0.0, add_latent: torch.Tensor | None = None):
    """Residual-stream edit that always passes through the SAE.

    With ``alpha=0`` and ``add_latent=None`` this is the reconstruction-only
    control: identical code path to a steered run, differing solely in alpha.
    Using it as the steering baseline is what separates the steering effect from
    reconstruction error.
    """
    def fn(t: torch.Tensor) -> torch.Tensor:
        shape = t.shape
        flat = t.reshape(-1, sae.cfg.d_in)
        z = sae.encode(flat)
        if add_latent is not None and alpha != 0.0:
            z = z + alpha * add_latent.to(z.dtype).to(z.device)
        return sae.decode(z).reshape(shape)
    return fn
