"""MoleculeACE transfer benchmark: 30 ChEMBL potency endpoints, activity cliffs.

Van Tilborg, Alenicheva & Grisoni (2022), *Exposing the limitations of molecular
machine learning with activity cliffs*, JCIM. The benchmark ships 30 curated
endpoints with a **predefined train/test split** and a per-molecule ``cliff_mol``
flag, so a comparison here is against published numbers rather than against a
split we chose ourselves. Both are used verbatim; nothing is re-split.

Why this module exists separately from :mod:`molcircuits.downstream.features`: the
representations are shared with every other downstream benchmark, but the
protocol -- which split, which label column, which head, which metric -- is
specific to MoleculeACE and belongs next to the benchmark, not next to the
featurizer.

**The pooling confound.** The original JAX implementation
(``lmkit/experiments/downstream_eval.py``, lines 116-122) pooled the two learned
representations differently::

    dense_pooled = masked_dense.sum(axis=1) / seq_lengths   # MEAN
    sae_pooled   = masked_sae.max(axis=1)                   # MAX

That is not a like-for-like comparison. Max over a non-negative sparse code keeps
"this feature fired somewhere in the molecule", which is exactly the substructure
signal a potency model wants; mean over a roughly zero-centred dense residual
stream averages that signal away. Any dense-vs-SAE gap measured on that diagonal
confounds the representation with the pooling operator. This module therefore
runs the full 2x2 -- ``{dense, sae} x {mean, max}`` -- so the two effects can be
separated, and labels the paper's diagonal explicitly rather than reproducing it
silently.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np

RAW_BASE = ("https://raw.githubusercontent.com/molML/MoleculeACE/main/"
            "MoleculeACE/Data/benchmark_data")

#: The 30 benchmark endpoints, exactly the files at the top level of
#: ``MoleculeACE/Data/benchmark_data`` (``raw/`` and ``old/`` are earlier
#: curations and are *not* the benchmark).
ENDPOINTS = (
    "CHEMBL1862_Ki", "CHEMBL1871_Ki", "CHEMBL2034_Ki", "CHEMBL2047_EC50",
    "CHEMBL204_Ki", "CHEMBL2147_Ki", "CHEMBL214_Ki", "CHEMBL218_EC50",
    "CHEMBL219_Ki", "CHEMBL228_Ki", "CHEMBL231_Ki", "CHEMBL233_Ki",
    "CHEMBL234_Ki", "CHEMBL235_EC50", "CHEMBL236_Ki", "CHEMBL237_EC50",
    "CHEMBL237_Ki", "CHEMBL238_Ki", "CHEMBL239_EC50", "CHEMBL244_Ki",
    "CHEMBL262_Ki", "CHEMBL264_Ki", "CHEMBL2835_Ki", "CHEMBL287_Ki",
    "CHEMBL2971_Ki", "CHEMBL3979_EC50", "CHEMBL4005_Ki", "CHEMBL4203_Ki",
    "CHEMBL4616_EC50", "CHEMBL4792_Ki",
)

#: pEC50 / pKi. The sibling column ``y`` is the same quantity shifted by a
#: constant 9 (it is -log10 of the value in nM), so RMSE is identical either way;
#: this one is the scale the published baselines are reported on.
TARGET_COLUMN = "y [pEC50/pKi]"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass
class Endpoint:
    """One benchmark target with the split and cliff flags as shipped."""
    name: str
    smiles: list[str]
    y: np.ndarray            # (N,) pEC50 / pKi
    train: np.ndarray        # (N,) bool, from the shipped `split` column
    cliff: np.ndarray        # (N,) bool, from the shipped `cliff_mol` column

    @property
    def test(self) -> np.ndarray:
        return ~self.train

    def drop(self, idx) -> "Endpoint":
        """Remove molecules by index, from features and labels together."""
        if not len(idx):
            return self
        keep = np.ones(len(self.smiles), bool)
        keep[np.asarray(list(idx), int)] = False
        return Endpoint(self.name, [s for s, k in zip(self.smiles, keep) if k],
                        self.y[keep], self.train[keep], self.cliff[keep])


def download(root: str, endpoints=ENDPOINTS) -> None:
    """Fetch any missing benchmark CSVs from the MoleculeACE repository."""
    os.makedirs(root, exist_ok=True)
    for name in endpoints:
        dst = os.path.join(root, f"{name}.csv")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        urllib.request.urlretrieve(f"{RAW_BASE}/{name}.csv", dst)


def load_endpoints(root: str, endpoints=ENDPOINTS,
                   download_missing: bool = True) -> list[Endpoint]:
    import pandas as pd  # noqa: PLC0415

    if download_missing:
        download(root, endpoints)
    out = []
    for name in endpoints:
        path = os.path.join(root, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"missing {path}; pass --download or fetch {RAW_BASE}/{name}.csv")
        df = pd.read_csv(path)
        missing = {"smiles", TARGET_COLUMN, "cliff_mol", "split"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}")
        out.append(Endpoint(
            name=name,
            smiles=df["smiles"].tolist(),
            y=df[TARGET_COLUMN].to_numpy(np.float64),
            train=(df["split"].to_numpy() == "train"),
            cliff=(df["cliff_mol"].to_numpy().astype(int) == 1),
        ))
    return out


# --------------------------------------------------------------------------- #
# prediction head
# --------------------------------------------------------------------------- #
#: Hyperparameters copied from the original implementation's
#: ``_fit_xgb_regressor`` so the head is not a new degree of freedom. Shared by
#: every representation -- that is the whole point of the comparison.
XGB_PARAMS = dict(
    n_estimators=2000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    random_state=42,
    tree_method="hist",
)


def fit_predict(X: np.ndarray, ep: Endpoint, seed: int = 0, n_jobs: int = -1,
                val_frac: float = 0.2, early_stopping_rounds: int = 50,
                params: dict | None = None, *,
                device: str = "cpu") -> tuple[np.ndarray, int]:
    """Train on the shipped train split, predict the shipped test split.

    Mirrors the original protocol: early-stop on a held-out validation set, then
    refit on train+validation with the selected number of trees. MoleculeACE
    ships only train/test, so the validation set is carved out of *train* with
    the run seed; the test split is never touched, and the refit sees the whole
    train split, so nothing is spent.

    ``device`` selects the tree-building backend: ``"cpu"`` or ``"cuda"``. It is
    keyword-only and defaults to CPU deliberately. ``hist`` on GPU and ``hist``
    on CPU are not bit-identical -- they sum gradients in a different order and
    can break split ties differently -- so a set of runs being compared to each
    other must all use the same backend. The device is recorded in the returned
    metadata by the callers, so a mixed set is detectable after the fact rather
    than silently averaged.
    """
    from xgboost import XGBRegressor  # noqa: PLC0415

    P = dict(XGB_PARAMS, n_jobs=n_jobs, device=device)
    if params:
        P.update(params)

    tr = np.flatnonzero(ep.train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    n_val = max(1, int(round(val_frac * len(perm))))
    val, sub = perm[:n_val], perm[n_val:]

    early = XGBRegressor(**P, early_stopping_rounds=early_stopping_rounds)
    early.fit(X[sub], ep.y[sub], eval_set=[(X[val], ep.y[val])], verbose=False)
    # `best_iteration` is 0-indexed; the original took it as a tree count, and
    # one tree out of hundreds does not move an RMSE, so the quirk is inherited
    # rather than silently "fixed" into a different protocol.
    n_best = max(1, int(getattr(early, "best_iteration", P["n_estimators"])))

    final = XGBRegressor(**dict(P, n_estimators=n_best))
    final.fit(X[tr], ep.y[tr], verbose=False)
    return final.predict(X[ep.test]), n_best


#: Alpha grid for the ridge head, searched on the same carved validation fold the
#: tree head early-stops on, so neither head gets a tuning budget the other lacks.
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def fit_predict_head(X: np.ndarray, ep: Endpoint, head: str = "xgb",
                     seed: int = 0, n_jobs: int = -1, val_frac: float = 0.2,
                     early_stopping_rounds: int = 50, *,
                     device: str = "cpu") -> tuple[np.ndarray, int]:
    """Same protocol as `fit_predict`, under a selectable prediction head.

    Why this exists: the dimension control concludes that retaining *selected*
    sparse axes matters "jointly with a tree head that can split on those axes".
    That clause is an explanation, and with only a tree head in the pipeline it is
    an untested one. A ridge head is linear and fully rotation-equivariant, so any
    advantage that is really about axis alignment must shrink under it; an MLP is
    nonlinear but not axis-aligned, which separates "needs a nonlinearity" from
    "needs the axes". Running all three turns the clause into a measurement.

    Every head sees the identical split, the identical seed and a validation fold
    carved the same way, so the head is the only thing that varies.
    """
    if head == "xgb":
        return fit_predict(X, ep, seed=seed, n_jobs=n_jobs, val_frac=val_frac,
                           early_stopping_rounds=early_stopping_rounds,
                           device=device)

    from sklearn.linear_model import Ridge  # noqa: PLC0415
    from sklearn.neural_network import MLPRegressor  # noqa: PLC0415
    from sklearn.preprocessing import MaxAbsScaler  # noqa: PLC0415

    tr = np.flatnonzero(ep.train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr)
    n_val = max(1, int(round(val_frac * len(perm))))
    val, sub = perm[:n_val], perm[n_val:]

    # MaxAbs, not StandardScaler. Dividing by a per-column standard deviation is
    # the wrong transform for a code that is ~99.6% exact zeros: a coordinate
    # that fires on a handful of molecules has a tiny sigma, so standardizing
    # turns its rare activations into enormous values and the optimizer chases
    # them. That is what produced the first version's MLP fits -- RMSE above the
    # train-mean baseline, with convergence warnings throughout. MaxAbs scales
    # each column by its largest magnitude, so it normalizes range without
    # inflating rare features, and it is applied identically to every arm so the
    # preprocessing is not itself a difference between representations.
    sc = MaxAbsScaler().fit(X[sub])
    Xs, Xv = sc.transform(X[sub]), sc.transform(X[val])

    if head == "ridge":
        best_a, best_e = RIDGE_ALPHAS[0], float("inf")
        for a in RIDGE_ALPHAS:
            e = rmse(Ridge(alpha=a).fit(Xs, ep.y[sub]).predict(Xv), ep.y[val])
            if e < best_e:
                best_a, best_e = a, e
        sc_full = MaxAbsScaler().fit(X[tr])
        final = Ridge(alpha=best_a).fit(sc_full.transform(X[tr]), ep.y[tr])
        return final.predict(sc_full.transform(X[ep.test])), int(
            RIDGE_ALPHAS.index(best_a))

    if head == "mlp":
        # Early stopping decides the refit budget, exactly as the tree head's
        # best_iteration does. The first version refit at the early-stopped
        # iteration count on MORE data, which under-trained every fit; here the
        # refit early-stops on its own carved fraction instead, so it is allowed
        # to converge. max_iter is a ceiling, not a target.
        kw = dict(hidden_layer_sizes=(256, 64), alpha=1e-3,
                  learning_rate_init=1e-3, max_iter=2000, tol=1e-5,
                  early_stopping=True, n_iter_no_change=30,
                  validation_fraction=max(0.1, val_frac / 2),
                  random_state=seed)
        mlp = MLPRegressor(**kw).fit(Xs, ep.y[sub])
        sc_full = MaxAbsScaler().fit(X[tr])
        final = MLPRegressor(**kw).fit(sc_full.transform(X[tr]), ep.y[tr])
        return (final.predict(sc_full.transform(X[ep.test])),
                int(getattr(final, "n_iter_", getattr(mlp, "n_iter_", 0))))

    raise ValueError(f"unknown head {head!r}; expected xgb, ridge or mlp")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    if len(pred) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(true, float)) ** 2)))


def score(pred: np.ndarray, ep: Endpoint) -> dict:
    """Overall, cliff-only and non-cliff RMSE on the shipped test split.

    MoleculeACE's whole argument is that mean error hides the failure mode, so
    the cliff subset is a headline number here, not an appendix one.
    """
    y = ep.y[ep.test]
    cl = ep.cliff[ep.test]
    r_all, r_cliff = rmse(pred, y), rmse(pred[cl], y[cl])
    return {
        "rmse": r_all,
        "rmse_cliff": r_cliff,
        "rmse_noncliff": rmse(pred[~cl], y[~cl]),
        "rmse_ratio": r_cliff / r_all if r_all > 0 else float("nan"),
        "n_test": int(cl.size),
        "n_test_cliff": int(cl.sum()),
    }


def aggregate(per_endpoint: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Mean over endpoints, unweighted -- one number per endpoint, as published.

    Also reports the pooled (molecule-weighted) RMSE, because an unweighted mean
    over endpoints of unequal size is a different quantity and the two can
    disagree; the paper's headline is the unweighted mean.
    """
    def is_score(v) -> bool:
        return isinstance(v, dict) and "rmse" in v

    reps = sorted({r for e in per_endpoint.values()
                   for r, v in e.items() if is_score(v)})
    out = {}
    for rep in reps:
        rows = [e[rep] for e in per_endpoint.values() if is_score(e.get(rep))]
        agg = {"n_endpoints": len(rows)}
        for key in ("rmse", "rmse_cliff", "rmse_noncliff", "rmse_ratio"):
            vals = np.array([r[key] for r in rows], float)
            agg[f"mean_{key}"] = float(np.nanmean(vals))
            agg[f"std_{key}"] = float(np.nanstd(vals))
        # pooled: sum of squared error over all test molecules of all endpoints
        for key, n_key in (("rmse", "n_test"), ("rmse_cliff", "n_test_cliff")):
            sse = sum(r[key] ** 2 * r[n_key] for r in rows if r[n_key])
            n = sum(r[n_key] for r in rows)
            agg[f"pooled_{key}"] = float(np.sqrt(sse / n)) if n else float("nan")
        out[rep] = agg
    return out


def paired_contrasts(per_endpoint: dict[str, dict], pairs, seed: int = 0,
                     n_boot: int = 20000) -> dict[str, dict]:
    """Per-endpoint paired differences with a bootstrap CI over endpoints.

    Two representations see the identical 30 endpoints, splits and head, so the
    endpoint is the natural pairing unit and a difference of unweighted means is
    a weaker statement than it looks. A paired CI plus a win count says whether a
    gap of 0.05 RMSE is a consistent effect or two or three endpoints dragging an
    average around.
    """
    rng = np.random.default_rng(seed)
    names = list(per_endpoint)
    out = {}
    for a, b in pairs:
        entry = {"n_endpoints": len(names)}
        for key in ("rmse", "rmse_cliff"):
            d = np.array([per_endpoint[n][a][key] - per_endpoint[n][b][key]
                          for n in names], float)
            boot = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
            entry[f"mean_{key}_diff"] = float(d.mean())
            entry[f"{key}_ci95"] = [float(np.percentile(boot, 2.5)),
                                    float(np.percentile(boot, 97.5))]
            entry[f"n_{key}_first_worse"] = int((d > 0).sum())
        out[f"{a}_minus_{b}"] = entry
    return out
