"""Reuse exact MoleculeACE head fits and Morgan fingerprints across dictionary runs."""
from __future__ import annotations

import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import zipfile

import numpy as np

from . import moleculeace
from .features import ecfp
from ..utils.manifest import file_sha256


def _digest(settings, arrays=()):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for arr in arrays:
        a = np.ascontiguousarray(arr)
        digest.update(json.dumps([a.dtype.str, a.shape]).encode())
        digest.update(memoryview(a).cast("B"))
    return digest.hexdigest()


def _stored(directory, key, compute):
    """Only one worker computes a key. Atomic replacement also survives worker death."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + ".npz")
    with (directory / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    if str(saved["lookup_key"]) == key:
                        return {name: saved[name] for name in saved.files if name != "lookup_key"}, True
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass
        result = compute()
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            tmp = Path(handle.name)
            try:
                np.savez(handle, lookup_key=key, **result)
                handle.flush()
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return result, False


@lru_cache(maxsize=8)
def _fit_runtime(device):
    import xgboost
    import sklearn
    hardware = platform.machine()
    if device.startswith("cuda"):
        import torch
        prop = torch.cuda.get_device_properties(torch.device(device))
        hardware = [prop.name, prop.major, prop.minor, prop.multi_processor_count, prop.total_memory]
        driver = Path("/proc/driver/nvidia/version")
        if driver.is_file():
            hardware.append(driver.read_text())
    else:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            hardware = [line for line in cpuinfo.read_text().split("\n\n")[0].splitlines()
                        if line.startswith(("model name", "flags"))]
    return {"device": device, "hardware": hardware, "numpy": np.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__,
            "xgboost_binary": file_sha256(xgboost.build_info()["libxgboost"]),
            "implementation": file_sha256(moleculeace.__file__)}


class MoleculeACECache:
    def __init__(self, directory, device):
        self.directory = Path(directory) if directory else None
        # Resolve the GPU once in the model process. The XGBoost workers need
        # no extra torch CUDA contexts of their own to read a completed fit.
        self.runtime = _fit_runtime(device) if self.directory else None

    def fit(self, X, ep, seed, n_jobs, val_frac, early_stop, device):
        def compute():
            pred, trees = moleculeace.fit_predict(X, ep, seed, n_jobs, val_frac,
                                                  early_stop, device=device)
            return {"pred": pred, "trees": np.asarray(trees, dtype=np.int64)}
        if self.directory is None:
            result, hit = compute(), False
        else:
            # The actual matrix fixes checkpoint, layer, features and pooling.
            # Labels and split rows are hashed in order. Cliff labels enter
            # only the scoring step, which is repeated for the current assay.
            settings = [self.runtime, moleculeace.XGB_PARAMS, seed, n_jobs, val_frac, early_stop]
            key = _digest(settings, (X, ep.y, ep.train))
            result, hit = _stored(self.directory / "fits" / key[:2], key, compute)
        return result["pred"], int(result["trees"]), hit

    def fingerprints(self, smiles):
        if self.directory is None:
            return ecfp(smiles)
        from rdkit import rdBase
        key = _digest([rdBase.rdkitVersion, file_sha256(ecfp.__code__.co_filename),
                       2048, 2, smiles])
        def compute():
            return {"bits": np.packbits(ecfp(smiles), axis=1)}
        result, _ = _stored(self.directory / "fingerprints", key, compute)
        return np.unpackbits(result["bits"], axis=1)
