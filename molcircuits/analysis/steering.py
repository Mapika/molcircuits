"""SAE feature steering, with the controls that were missing.

ALPHA IS IN UNITS OF THE RESIDUAL-STREAM RMS, matching the valence intervention
in ``valence_intervene.intervene``. This is not cosmetic. Scaling alpha by the
target's own max-pooled top-k code values -- the obvious parameterization, and
the one used before -- makes the physical size of the perturbation depend on the
dictionary: for a TopK dictionary those values run to ~80 in normalized units,
the decoder columns are unit-norm and ``decode`` multiplies by ``input_scale``,
so ``alpha=1`` injected a vector of norm ~6,000 onto a residual of norm ~890.
The whole sweep {0.5, 1, 2} was then 3x to 14x the norm of the residual stream:
not steering but overwriting, and a sweep whose every point destroys the model
cannot distinguish a feature that carries chemistry from one that does not.
Worse, the same alpha means a different perturbation under a different
dictionary, so no two dictionaries' sweeps are comparable.

Here the injection is rescaled so that the residual-space perturbation has norm
exactly ``alpha * rms``. Two consequences worth having: alpha is comparable with
the valence experiment and across dictionaries, and the steered and random arms
are matched in perturbation norm *exactly* rather than to within a few percent.

Four design faults in the previous experiment, all fixed here:

* **The baseline went through a different code path.** Steered samples passed the
  residual stream through an SAE encode-decode round trip; the baseline used the
  unmodified transformer. The difference therefore confounded steering with
  reconstruction error, which at delta-CE up to 0.10 is not negligible against a
  ~0.007 similarity shift. Here the baseline is ``alpha=0`` through the *same*
  hook, so the only difference between arms is the coefficient.
* **No random-feature control.** ``run_sweep`` always includes one: the same
  number of features at matched activation magnitude, drawn at random.
* **Different sampling seeds per arm.** All arms share a seed here.
* **Validity conflated with truncation.** A perturbation this size stops the
  model emitting EOS at all, and a sequence cut off at ``max_tokens`` is scored
  invalid because its rings and branches are still open. Every arm now records
  its termination rate beside its validity, and ``max_tokens`` defaults to 256
  rather than 128 -- unsteered generation averages 39 tokens, so the cap only
  ever binds on perturbed runs, which is exactly where it silently became the
  measurement.

Targets should be drawn from the training distribution. Steering toward molecules
excluded by the corpus filter (e.g. below the 150 Da cutoff) tests nothing.
"""
from __future__ import annotations

import numpy as np
import torch

from ..model.hooks import HookManager, resid_post
from ..model.sampling import generate
from ..sae.evaluate import sae_edit_fn


def rms_scaled_latent(sae, z: torch.Tensor, alpha: float, rms: float) -> torch.Tensor:
    """Rescale a latent so its decoded perturbation has residual-space norm alpha*rms.

    ``decode(encode(x) + z') - decode(encode(x))`` is exactly
    ``(z' @ W_dec) * input_scale``, so the scale factor is closed-form and the
    perturbation norm is exact rather than approximate. Returns a zero latent
    when ``z`` decodes to nothing, which happens only if the target activated no
    features at all.
    """
    d = (z.float() @ sae.W_dec.float().detach()) * sae.input_scale.float()
    n = float(d.norm().detach())
    if n <= 0 or alpha == 0.0:
        return torch.zeros_like(z)
    return z * (alpha * rms / n)


def perturbation_norm(sae, z: torch.Tensor) -> float:
    """Residual-space norm of the perturbation ``z`` would inject, unscaled."""
    return float(((z.float() @ sae.W_dec.float().detach())
                   * sae.input_scale.float()).norm().detach())


def tanimoto_stats(smiles: list[str], target: str | None = None,
                   n_bits: int = 2048, radius: int = 2,
                   meta: dict | None = None) -> dict:
    from rdkit import Chem, DataStructs, RDLogger  # noqa: PLC0415
    from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415
    RDLogger.DisableLog("rdApp.*")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps, n_valid = [], 0
    for s in smiles:
        m = Chem.MolFromSmiles(s) if s else None
        if m is None:
            continue
        n_valid += 1
        fps.append(gen.GetFingerprint(m))

    out = {"n": len(smiles), "n_valid": n_valid,
           "validity": n_valid / max(len(smiles), 1)}
    if meta is not None:
        # Validity alone cannot separate "wrote nonsense" from "never finished".
        out["termination_rate"] = meta.get("termination_rate")
        out["n_terminated"] = meta.get("n_terminated")
    if len(fps) > 1:
        sims = []
        for i in range(min(len(fps), 500)):
            sims.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:i + 200]))
        out["pairwise_tanimoto_mean"] = float(np.mean(sims)) if sims else float("nan")
    if target:
        tm = Chem.MolFromSmiles(target)
        if tm is not None and fps:
            ts = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(tm), fps)
            out["target_tanimoto_mean"] = float(np.mean(ts))
            out["target_tanimoto_max"] = float(np.max(ts))
    return out


@torch.no_grad()
def reference_latent(model, sae, tok, target: str, layer: int, top_k: int = 5,
                     device: str = "cpu") -> torch.Tensor:
    """Max-pooled SAE code of the target, truncated to its top-k features.

    Pooling is over CONTENT tokens only -- position 0 is BOS and the last real
    position is EOS, and both are attention sinks whose residual norm dwarfs
    every chemical position (the same defect ``downstream.features.embed``
    documents).

    Including them does not merely add noise here, it destroys the experiment.
    The features that fire hardest at BOS and EOS are the same features for
    every molecule, so a max over all positions returns them every time:
    measured on the 50 steering targets at layer 3, pooling over all positions
    gives ONE distinct top-5 feature set across all 50 targets -- features
    {1131, 2668, 5368, 5470, 7908} for every one of them, mean pairwise overlap
    5.00 of 5. "Steering toward the target" was then injecting the same
    direction regardless of the target, which is why 50 targets produced 2
    distinct output sets while 50 matched RANDOM features produced 50.

    Restricted to content tokens the same 50 targets give 41 distinct top-5
    sets drawing on 74 distinct features, with mean pairwise overlap 2.10 of 5.
    """
    enc = tok.encode(target)
    ids = torch.tensor([enc], dtype=torch.long, device=device)
    h = HookManager().capture(resid_post(layer))
    model(ids, hooks=h)
    z = sae.encode(h[resid_post(layer)].reshape(-1, sae.cfg.d_in))
    if z.shape[0] > 2:              # a 1-2 token encoding has no interior
        z = z[1:-1]
    z = z.max(0).values
    if top_k and top_k < z.numel():
        keep = z.topk(top_k).indices
        masked = torch.zeros_like(z)
        masked[keep] = z[keep]
        z = masked
    return z


def matched_random_latent(z_ref: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Same number of active features, same magnitudes, random identities."""
    # The generator is pinned to CPU so the drawn feature identities are the same
    # on GPU and CPU; the indices are then moved to z_ref's device rather than
    # dragging z_ref back to the host, which mismatched devices when z_ref is cuda.
    g = torch.Generator(device="cpu").manual_seed(seed)
    active = (z_ref > 0).nonzero().squeeze(-1)
    out = torch.zeros_like(z_ref)
    if len(active) == 0:
        return out
    idx = torch.randperm(z_ref.numel(), generator=g)[: len(active)].to(out.device)
    out[idx] = z_ref[active].to(out.dtype)
    return out


def rate_matched_random_latent(z_ref: torch.Tensor, rates, seed: int = 0,
                               log_tol: float = 0.25) -> torch.Tensor:
    """Random identities matched to the reference features on firing rate.

    :func:`matched_random_latent` matches count and magnitude but draws
    identities uniformly over the dictionary. That is not a neutral draw here:
    most coordinates fire very rarely, so a uniform control is overwhelmingly
    made of near-dead features, and the two arms differ in how often their
    features fire in ordinary contexts as well as in which features they are.
    Injecting a coordinate the model almost never sees active is a different
    intervention from injecting one it sees often, so a null obtained against
    the uniform control leaves that difference uncontrolled.

    Each active reference feature is therefore replaced by one drawn uniformly
    from the coordinates whose log firing rate is within ``log_tol`` of its own,
    excluding the reference set and anything already drawn. Features with no
    such partner fall back to the closest available rate, which is recorded by
    the caller rather than passed over silently.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    active = (z_ref > 0).nonzero().squeeze(-1)
    out = torch.zeros_like(z_ref)
    if len(active) == 0:
        return out
    r = torch.as_tensor(rates, dtype=torch.float64, device="cpu")
    if r.numel() != z_ref.numel():
        raise ValueError(
            f"firing-rate table has {r.numel()} entries for a {z_ref.numel()}-d code")
    logr = torch.log10(r.clamp_min(1e-12))
    taken = set(active.cpu().tolist())
    chosen = []
    for i in active.cpu().tolist():
        d = (logr - logr[i]).abs()
        d[list(taken)] = float("inf")
        pool = (d <= log_tol).nonzero().squeeze(-1)
        if len(pool) == 0:                       # no partner in band: nearest
            pick = int(d.argmin())
        else:
            pick = int(pool[torch.randint(len(pool), (1,), generator=g)])
        taken.add(pick)
        chosen.append(pick)
    idx = torch.tensor(chosen, device=out.device)
    out[idx] = z_ref[active].to(out.dtype)
    return out


def run_sweep(model, sae, tok, target: str, layer: int,
              alphas=(0.0, 1.0, 2.0, 4.0, 8.0, 16.0),
              n_samples: int = 200, top_k: int = 5, seed: int = 0,
              device: str = "cpu", max_tokens: int = 256,
              control_seed: int | None = None, rms: float = 1.0,
              firing_rates=None) -> dict:
    """Steered, alpha=0 and random-feature arms under one shared sampling seed.

    ``alphas`` are multiples of ``rms``, the residual-stream RMS at this layer,
    so alpha=1 injects a perturbation of norm rms. Note that a residual VECTOR
    has norm rms*sqrt(d_model), so alpha=1 is about 4% of the residual's own
    length at d_model=512; the sweep runs to 16 for that reason. ``rms`` must be
    measured once by the caller on real token positions and passed in --
    defaulting it to 1.0 would silently reproduce the absolute-magnitude
    parameterization this function exists to replace.

    ``control_seed`` must vary per target. With a single constant seed
    ``matched_random_latent`` returns the same permutation prefix every call, so
    all targets shared one set of random features -- an n=1 control whose
    across-target spread reflected sampling noise only, and which could not
    detect that those particular features were unusually destructive or benign.
    The *sampling* seed stays shared across arms, which is what makes the arms
    comparable; only the feature draw varies.
    """
    z_ref = reference_latent(model, sae, tok, target, layer, top_k, device)
    cseed = seed if control_seed is None else control_seed
    z_rand = matched_random_latent(z_ref, seed=cseed).to(device)
    z_rate = (rate_matched_random_latent(z_ref, firing_rates, seed=cseed).to(device)
              if firing_rates is not None else None)

    def run(z, a):
        # Rescaled per alpha: both arms then carry the SAME perturbation norm,
        # alpha*rms, so the comparison is between feature identities and not
        # between perturbation sizes.
        z_a = rms_scaled_latent(sae, z, a, rms)
        hooks = HookManager().edit(resid_post(layer), sae_edit_fn(sae, 1.0, z_a))
        m: dict = {}
        smi = generate(model, tok, n_samples, max_tokens=max_tokens, hooks=hooks,
                       seed=seed, device=device, meta=m, use_kv_cache=True)
        s = tanimoto_stats(smi, target, meta=m)
        s["alpha_rms_units"] = float(a)
        s["perturbation_norm"] = float(a * rms)
        return s

    arms = {}
    for a in alphas:
        arms[f"steered_alpha={a}"] = run(z_ref, a)
    for a in alphas:
        if a == 0.0:
            continue
        arms[f"random_feature_alpha={a}"] = run(z_rand, a)
    if z_rate is not None:
        for a in alphas:
            if a == 0.0:
                continue
            arms[f"rate_matched_alpha={a}"] = run(z_rate, a)

    m0: dict = {}
    unsteered = generate(model, tok, n_samples, max_tokens=max_tokens, hooks=None,
                         seed=seed, device=device, meta=m0)
    arms["no_sae_at_all"] = tanimoto_stats(unsteered, target, meta=m0)

    base = arms["steered_alpha=0.0"]
    summary = {}
    for k, v in arms.items():
        if "target_tanimoto_mean" in v and "target_tanimoto_mean" in base:
            summary[k] = {
                "delta_target_tanimoto_vs_alpha0":
                    v["target_tanimoto_mean"] - base["target_tanimoto_mean"],
                "validity": v["validity"],
                "termination_rate": v.get("termination_rate"),
            }
    # Preserve the actual decoded inputs for downstream geometry analyses:
    # regenerating a rate-matched draw on another corpus can change its identity.
    latent_updates = {}
    for arm, latent in (("target", z_ref), ("random", z_rand), ("rate_matched", z_rate)):
        if latent is not None:
            active = (latent > 0).nonzero().flatten()
            latent_updates[arm] = {"indices": active.tolist(), "values": latent[active].tolist()}
    return {"target": target, "layer": layer, "latent_updates": latent_updates, "top_k": top_k, "seed": seed,
            "rms": float(rms), "max_tokens": int(max_tokens),
            "unscaled_perturbation_norm_ref": perturbation_norm(sae, z_ref),
            "unscaled_perturbation_norm_rand": perturbation_norm(sae, z_rand),
            "control_seed": seed if control_seed is None else control_seed,
            "random_control_features":
                sorted(int(i) for i in (z_rand > 0).nonzero().flatten().tolist()),
            "arms": arms, "summary_vs_alpha0_baseline": summary,
            "note": ("the alpha=0 arm passes through the same SAE encode/decode "
                     "path as the steered arms and is the correct baseline; "
                     "'no_sae_at_all' is shown only to quantify reconstruction cost")}
