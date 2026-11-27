"""Fragment specificity (WSD) and a basis comparison that is actually fair.

The problem with the previous comparison
----------------------------------------
WSD is a *signed* statistic averaged over molecules:

    (mu_in - mu_out) / sigma_all * sqrt(p(1-p))

A ReLU dictionary code is non-negative, so for a feature that fires on a fragment
the per-molecule difference has a consistent sign and survives averaging. A dense
residual coordinate is roughly zero-centred: the same fragment drives it positive
in some molecules and negative in others, and the average cancels. The dense
column therefore came out at 0.00 for every fragment while the SAE column did
not -- a difference in sign symmetry, not in interpretability.

Two changes make the bases comparable:

* ``signed=False`` scores ``|WSD|``, which does not punish a basis for being
  zero-centred. Both are reported.
* every basis is scored through the *same* function, and a per-molecule
  **permutation null** (fragment labels shuffled within each molecule) gives an
  absolute reference, so a value of 0.2 can be read as strong or weak.

Cost
----
The original form of this module was unusable at realistic dictionary widths: it
projected to ~17 hours for 34 fragments. Four things were wrong, none of which
changes any number it produces.

* It cast to float64 and materialised ``a * pos``, ``a * neg``, ``a * val`` and
  ``a**2 * val`` in full -- four ``(B, T, d_sae)`` temporaries, 16.8 GB *each* at
  2000 molecules x 128 tokens x 8192 features.
* ``n_all``, ``sum_all`` and hence ``mu_all``, ``var``, ``sigma`` do not depend on
  the fragment mask. They are properties of the basis, so ``basis_stats``
  computes them once and every permutation reuses them. Likewise ``neg`` is the
  complement of ``pos`` within ``valid``, so ``sum_out == sum_all - sum_in``
  exactly and only one masked reduction per mask is needed, not three.
* The permutation null ran one mask at a time, re-reading the whole array 25
  times. Stacked into a single batched matmul it is read once.
* Numpy's ``matmul`` on stacked ``(b, M, T) @ (b, T, K)`` loops per batch element
  rather than dispatching a batched GEMM, achieving ~215 MB/s. The reduction now
  runs through ``torch.bmm``, on the GPU when one is available -- the bases total
  ~12 GB and stay resident, so no array crosses PCIe more than once.

``best_detector`` reduces to a mean over molecules then a max over features, so
the per-molecule WSD never has to exist in full; the mean is accumulated inside
the chunk loop.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 1e-12
_CHUNK = 512


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _t(a, device: str, dtype=torch.float32) -> torch.Tensor:
    """Numpy or torch -> torch tensor on ``device``, without a needless copy."""
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(a)).to(device=device, dtype=dtype)


def to_basis(acts, device: str | None = None) -> torch.Tensor:
    """Park a basis on the compute device so repeated scoring never re-uploads."""
    return _t(acts, device or _device(), torch.float32)


def basis_stats(acts, valid_mask, chunk: int = _CHUNK) -> dict:
    """Fragment-independent moments of one basis. Computed once, reused per mask.

    ``var = E[x^2] - E[x]^2`` is a cancelling difference and is the one place
    here where float32 would cost real accuracy, so the moments accumulate in
    float64; everything downstream is float32.
    """
    dev = acts.device.type if isinstance(acts, torch.Tensor) else _device()
    B, T, K = acts.shape
    v_np = valid_mask
    n_all = _t(v_np.sum(1), dev, torch.float64)[:, None, None]        # (B,1,1)

    sum_all = torch.empty((B, 1, K), dtype=torch.float64, device=dev)
    sumsq = torch.empty((B, 1, K), dtype=torch.float64, device=dev)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        a = _t(acts[s:e], dev, torch.float64)
        w = _t(v_np[s:e], dev, torch.float64)[:, None, :]              # (b,1,T)
        sum_all[s:e] = torch.bmm(w, a)
        sumsq[s:e] = torch.bmm(w, a * a)

    mu_all = sum_all / n_all.clamp_min(1)
    var = sumsq / n_all.clamp_min(1) - mu_all**2
    sigma = var.clamp_min(0.0).sqrt() + EPS
    return {"n_all": n_all.float(), "sum_all": sum_all.float(),
            "sigma": sigma.float(), "device": dev}


def _chunk_wsd(a, pos, st, s, e, signed: bool):
    """WSD for one molecule-chunk and M masks -> (b, M, K) float32, plus the
    (b, M) mask of molecules that actually contain the fragment."""
    sum_in = torch.bmm(pos.float(), a)                                 # (b,M,K)
    n_in = pos.sum(2, dtype=torch.float32)[..., None]                  # (b,M,1)
    n_all = st["n_all"][s:e]
    mu_in = sum_in / n_in.clamp_min(1)
    mu_out = (st["sum_all"][s:e] - sum_in) / (n_all - n_in).clamp_min(1)
    p = n_in / n_all.clamp_min(1)
    weight = (p * (1 - p)).clamp(0.0, 0.25).sqrt()
    d = torch.nan_to_num((mu_in - mu_out) / st["sigma"][s:e]) * weight
    if not signed:
        d = d.abs()
    return d, n_in[:, :, 0] > 0


def _mean_multi_pair(acts, masks3d, valid_mask, st,
                     chunk: int = _CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Mean signed and mean-absolute WSD for M masks and every coordinate.

    A molecule with no fragment match contributes nothing: the naive form marks
    it NaN and leans on ``nanmean``; here such rows are zeroed and dropped from
    the denominator, which is the same equal-molecule average.  Accumulating the
    signed and absolute forms together avoids reading a potentially very wide
    basis twice.
    """
    dev = st["device"]
    B, T, K = acts.shape
    M = masks3d.shape[1]
    acc_signed = torch.zeros((M, K), dtype=torch.float64, device=dev)
    acc_absolute = torch.zeros((M, K), dtype=torch.float64, device=dev)
    cnt = torch.zeros((M, 1), dtype=torch.float64, device=dev)

    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        a = _t(acts[s:e], dev, torch.float32)
        pos = _t(masks3d[s:e], dev, torch.bool) & _t(
            valid_mask[s:e], dev, torch.bool)[:, None, :]
        d, ok = _chunk_wsd(a, pos, st, s, e, signed=True)
        d = d * ok[..., None]
        acc_signed += d.sum(0, dtype=torch.float64)
        acc_absolute += d.abs().sum(0, dtype=torch.float64)
        cnt += ok.sum(0, dtype=torch.float64)[:, None]

    missing = cnt <= 0
    mean_signed = torch.where(
        missing, torch.full_like(acc_signed, float("nan")),
        acc_signed / cnt.clamp_min(1.0))
    mean_absolute = torch.where(
        missing, torch.full_like(acc_absolute, float("nan")),
        acc_absolute / cnt.clamp_min(1.0))
    return mean_signed.cpu().numpy(), mean_absolute.cpu().numpy()


def mean_wsd_coordinates(acts, masks, valid_mask, stats: dict | None = None,
                         chunk: int = _CHUNK,
                         mask_batch: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Return molecule-mean signed and absolute WSD for each mask/coordinate.

    ``masks`` may be ``(B, T)`` or ``(B, M, T)``.  Molecules without a match
    are excluded exactly as in :func:`best_detector`; matched molecules are
    weighted equally rather than by their number of atom tokens.  Mask batching
    bounds the temporary ``(molecule_chunk, mask_batch, coordinates)`` tensor,
    which matters for the 8,192-coordinate controls and their permutation nulls.
    """
    masks3d = np.asarray(masks, dtype=bool)
    if masks3d.ndim == 2:
        masks3d = masks3d[:, None, :]
    if masks3d.ndim != 3:
        raise ValueError("masks must have shape (B, T) or (B, M, T)")
    if tuple(masks3d.shape[::2]) != (acts.shape[0], acts.shape[1]):
        raise ValueError("mask molecule/token axes do not match activations")
    if np.asarray(valid_mask).shape != (acts.shape[0], acts.shape[1]):
        raise ValueError("valid_mask shape does not match activations")
    if chunk < 1 or mask_batch < 1:
        raise ValueError("chunk and mask_batch must be positive")
    st = basis_stats(acts, valid_mask, chunk) if stats is None else stats
    signed_rows, absolute_rows = [], []
    for start in range(0, masks3d.shape[1], mask_batch):
        signed, absolute = _mean_multi_pair(
            acts, masks3d[:, start:start + mask_batch], valid_mask, st, chunk)
        signed_rows.append(signed)
        absolute_rows.append(absolute)
    return np.concatenate(signed_rows), np.concatenate(absolute_rows)


def _best_multi(acts, masks3d, valid_mask, signed, st,
                chunk: int = _CHUNK) -> np.ndarray:
    """Best-detector score for M masks at once, without holding ``(B,M,K)``."""
    mean_signed, mean_absolute = _mean_multi_pair(
        acts, masks3d, valid_mask, st, chunk)
    mean = mean_signed if signed else mean_absolute
    all_missing = np.isnan(mean).all(axis=1)
    out = np.max(np.nan_to_num(mean, nan=-np.inf), axis=1)
    out[all_missing] = np.nan
    return out


def wsd(acts, frag_mask, valid_mask, signed: bool = True,
        stats: dict | None = None, chunk: int = _CHUNK) -> np.ndarray:
    """Coverage-weighted standardized contrast, per (molecule, feature).

    acts:       (B, T, K) activations
    frag_mask:  (B, T) True where the fragment matches
    valid_mask: (B, T) the POPULATION the contrast is taken over. Pass
                ``fragments.atom_token_mask``, not ``valid_token_mask``.
    stats:      optional output of ``basis_stats`` for this same ``acts``
    returns:    (B, K)

    The population is not a detail. A SMARTS match marks atoms, so if the
    negative set is every real token it also contains bonds, branch parentheses
    and ring digits, and a feature that only separates atoms from punctuation
    scores as a detector for every fragment in the library -- clearing its own
    permutation null while carrying no chemistry, because shuffling fragment
    membership across all real tokens moves the positives onto punctuation.
    Measured: WSD 0.14-0.36 for such a feature, against 0.309-0.341 for a real
    neuron basis. Restricted to atoms it scores 0.000. See
    ``fragments.atom_token_mask`` and
    ``tests/test_chem.py::test_fragment_specificity_rejects_an_atom_vs_punctuation_detector``.
    """
    st = basis_stats(acts, valid_mask, chunk) if stats is None else stats
    dev = st["device"]
    B, T, K = acts.shape
    out = np.empty((B, K), dtype=np.float32)
    m3 = frag_mask[:, None, :]
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        a = _t(acts[s:e], dev, torch.float32)
        pos = _t(m3[s:e], dev, torch.bool) & _t(
            valid_mask[s:e], dev, torch.bool)[:, None, :]
        d, ok = _chunk_wsd(a, pos, st, s, e, signed)
        d = d[:, 0, :].masked_fill(~ok[:, 0, None], float("nan"))
        out[s:e] = d.cpu().numpy()
    return out


def best_detector(acts, frag_mask, valid_mask, signed: bool = True,
                  stats: dict | None = None) -> float:
    """Max over features of the mean WSD -- the 'best detector' in this basis."""
    st = basis_stats(acts, valid_mask) if stats is None else stats
    return float(_best_multi(acts, frag_mask[:, None, :], valid_mask, signed, st)[0])


def permutation_null(acts, frag_mask, valid_mask, n_perm: int = 50,
                     signed: bool = True, seed: int = 0,
                     stats: dict | None = None, batch: int = 25) -> dict:
    """Shuffle fragment membership *within* each molecule, preserving coverage."""
    rng = np.random.default_rng(seed)
    st = basis_stats(acts, valid_mask) if stats is None else stats

    shuffled = np.zeros((frag_mask.shape[0], n_perm, frag_mask.shape[1]), bool)
    for i in range(n_perm):
        for b in range(frag_mask.shape[0]):
            valid_idx = np.where(valid_mask[b])[0]
            k = int(frag_mask[b].sum())
            if k and len(valid_idx) >= k:
                shuffled[b, i, rng.choice(valid_idx, size=k, replace=False)] = True

    vals = np.concatenate([
        _best_multi(acts, shuffled[:, s:s + batch], valid_mask, signed, st)
        for s in range(0, n_perm, batch)])
    return {"null_mean": float(np.nanmean(vals)), "null_std": float(np.nanstd(vals)),
            "null_p95": float(np.nanpercentile(vals, 95)), "n_perm": n_perm}


# ------------------------------------------------------------------ bases
def dense_basis(acts: np.ndarray) -> np.ndarray:
    return acts


def pca_basis(acts: np.ndarray, n_components: int | None = None, seed: int = 0):
    from sklearn.decomposition import PCA  # noqa: PLC0415
    B, T, K = acts.shape
    flat = acts.reshape(-1, K)
    n = n_components or K
    p = PCA(n_components=min(n, K), random_state=seed).fit(flat)
    return p.transform(flat).reshape(B, T, -1), p


def random_rotation_basis(acts, seed: int = 0):
    """A single Haar-random orthogonal rotation. Use several seeds."""
    rng = np.random.default_rng(seed)
    K = acts.shape[-1]
    Q, _ = np.linalg.qr(rng.standard_normal((K, K)))
    if isinstance(acts, torch.Tensor):
        return acts @ _t(Q, acts.device.type, torch.float32)
    return acts @ Q


def compare_bases(dense, sae_acts, frag_mask, valid_mask,
                  rotation_seeds=(0, 1, 2, 3, 4), signed: bool = True,
                  n_perm: int = 25, seed: int = 0,
                  cache: dict | None = None) -> dict:
    """Score every basis identically and report the rotation spread.

    The random-rotation entry carries a genuine standard deviation across seeds.
    A baseline reported as a single fixed rotation with a +/- 0.00 interval is
    not a baseline.

    ``cache`` carries the derived bases, their device residency and their
    ``basis_stats`` across calls. Nothing in it depends on the fragment mask or
    on ``signed``, so without it the PCA fit, the Haar rotations and every moment
    are recomputed for all 34 fragments x 2 sign conventions.
    """
    if cache is None:
        cache = {}
    if "bases" not in cache:
        dev = _device()
        d_t = to_basis(dense, dev)
        sae_t = to_basis(sae_acts, dev)
        bases = {"dense": d_t, "sae": sae_t,
                 "pca_full": to_basis(pca_basis(dense, None, seed)[0], dev),
                 **{f"rot{s}": random_rotation_basis(d_t, s) for s in rotation_seeds}}
        # Dimension-matched SAE arms. "best detector" is a MAX over columns, and
        # the SAE has 8192 of them against 512 for every other basis here -- a
        # max over 16x more draws is larger in expectation whatever the columns
        # mean, so the headline SAE-vs-dense gap is not a like-for-like
        # comparison without this. Two matched arms, because either one alone is
        # attackable: `sae_var512` keeps the highest-variance columns (the
        # informative subset, and the analogue of the downstream `sae_top512`
        # control), `sae_rand512` draws them uniformly (no selection at all, so
        # it cannot be accused of picking the winners).
        d_model = d_t.shape[-1]
        if sae_t.shape[-1] > d_model:
            flat = sae_t.reshape(-1, sae_t.shape[-1])
            var = flat.var(0)
            keep_var = torch.topk(var, d_model).indices
            g = torch.Generator(device="cpu").manual_seed(seed)
            keep_rnd = torch.randperm(sae_t.shape[-1], generator=g)[:d_model]
            bases["sae_var512"] = sae_t[..., keep_var]
            bases["sae_rand512"] = sae_t[..., keep_rnd.to(sae_t.device)]
        cache["bases"] = bases
        cache["stats"] = {k: basis_stats(v, valid_mask) for k, v in bases.items()}
    bases, stats = cache["bases"], cache["stats"]

    def best(name):
        return best_detector(bases[name], frag_mask, valid_mask, signed,
                             stats[name])

    out = {"signed": signed}
    out["dense"] = {"best": best("dense")}
    out["sae"] = {"best": best("sae")}
    out["pca_full"] = {"best": best("pca_full")}
    for arm in ("sae_var512", "sae_rand512"):
        if arm in bases:
            out[arm] = {"best": best(arm), "n_columns": int(bases[arm].shape[-1])}

    rot = [best(f"rot{s}") for s in rotation_seeds]
    out["random_rotation"] = {"best_mean": float(np.mean(rot)),
                              "best_std": float(np.std(rot)),
                              "per_seed": [float(r) for r in rot],
                              "n_seeds": len(rotation_seeds)}
    out["permutation_null_sae"] = permutation_null(
        bases["sae"], frag_mask, valid_mask, n_perm, signed, seed, stats["sae"])
    out["permutation_null_dense"] = permutation_null(
        bases["dense"], frag_mask, valid_mask, n_perm, signed, seed, stats["dense"])
    return out
