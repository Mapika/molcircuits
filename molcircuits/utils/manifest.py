"""Run manifest: the discipline that would have saved the previous paper.

Every script writes its numbers to ``artifacts/`` as JSON *and* appends a record
here containing the git SHA, library versions, config hash, seed and file
checksums. Two consequences:

1. Nothing lives only in a plotting notebook or a metrics dashboard. If the
   figure and the table disagree, there is a third artifact that settles it.
2. ``artifacts/`` is committed. The previous project gitignored ``results/``,
   ``models/`` and ``dump/``, and lost every number in the paper when the disk
   went. Small JSON is cheap; regenerating a paper is not.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .. import paths


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _dirty():
    """True/False when git answers, None when it does not.

    Returning False on failure emitted a positive assertion -- "the tree was
    clean" -- from a call that never ran. Every record in this project carried
    git_dirty: false while git_sha was "unknown".
    """
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
    except Exception:
        return None


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(cfg: Any) -> str:
    d = cfg if isinstance(cfg, dict) else getattr(cfg, "__dict__", str(cfg))
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def source_tree_hash(root: str | os.PathLike | None = None) -> tuple[str, int]:
    """Hash the runnable source even when the checkout has no ``.git``.

    Artifact provenance must not silently collapse to ``git_sha=unknown``.  The
    hash covers code, shell runners, project metadata and the resolved lockfile;
    generated artifacts, caches and manuscript binaries are deliberately out of
    scope.  Relative paths are included so moving/renaming a module changes the
    identity as it should.
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    candidates = [base / "pyproject.toml", base / "uv.lock",
                  base / ".python-version"]
    for folder in (base / "molcircuits", base / "experiments", base / "tests"):
        if folder.exists():
            candidates.extend(p for p in folder.rglob("*")
                              if p.is_file() and p.suffix in {".py", ".sh"})
    h = hashlib.sha256()
    files = sorted({p.resolve() for p in candidates if p.exists()})
    for path in files:
        rel = path.relative_to(base.resolve()).as_posix().encode()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest(), len(files)


# Capture provenance when this module is imported, before a long-running stage
# begins.  Hashing only in ``record`` mislabeled a run whenever another process
# edited an unrelated source file while the run was in flight.  We retain the
# end-of-run hash below as an explicit drift diagnostic rather than silently
# substituting it for the source identity the process actually started with.
_PROCESS_SOURCE_TREE_HASH, _PROCESS_SOURCE_TREE_FILES = source_tree_hash()


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "numpy", "rdkit", "sklearn", "xgboost", "transformers"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "absent"
    try:
        import torch
        out["cuda_runtime"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        out["cuda_device"] = (torch.cuda.get_device_name(0)
                              if torch.cuda.is_available() else None)
        out["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except Exception:
        pass
    return out


class Manifest:
    def __init__(self, path: str | None = None):
        self.path = path or paths.manifest()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def record(self, stage: str, payload: dict, config: Any = None,
               files: list[str] | None = None) -> None:
        record_hash, record_files = source_tree_hash()
        entry = {
            "stage": stage,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_sha": _git_sha(),
            "git_dirty": _dirty(),
            "source_tree_sha256": _PROCESS_SOURCE_TREE_HASH,
            "source_tree_n_files": _PROCESS_SOURCE_TREE_FILES,
            "source_tree_sha256_at_record": record_hash,
            "source_tree_n_files_at_record": record_files,
            "source_tree_changed_during_process": (
                record_hash != _PROCESS_SOURCE_TREE_HASH
                or record_files != _PROCESS_SOURCE_TREE_FILES),
            "versions": _versions(),
            "payload": payload,
        }
        if config is not None:
            entry["config_hash"] = config_hash(config)
            entry["config"] = config if isinstance(config, dict) else getattr(
                config, "__dict__", str(config))
        if files:
            entry["files"] = {f: file_sha256(f) for f in files if os.path.exists(f)}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def dump_json(obj: Any, path: str) -> str:
    """Atomically replace a JSON artifact after its full payload reaches disk.

    Long stages checkpoint into the same filename that consumers later read.
    Writing directly to that file exposes a truncated-but-present artifact to a
    concurrent summarizer (or after interruption).  A sibling temporary file
    plus ``os.replace`` makes every visible version a complete JSON document.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    destination_mode = (stat.S_IMODE(os.stat(path).st_mode)
                        if os.path.exists(path) else 0o644)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory,
                prefix=f".{os.path.basename(path)}.", suffix=".tmp",
                delete=False) as fh:
            temporary = fh.name
            # RFC 8259 JSON has no NaN or Infinity literals.  Refuse them
            # before replacing the prior artifact so every visible file is
            # portable to strict parsers and JavaScript.
            json.dump(obj, fh, indent=2, default=str, allow_nan=False)
            fh.write("\n")
            fh.flush()
            # NamedTemporaryFile starts at 0600. Preserve an existing
            # artifact's mode, or use the repository's readable JSON default,
            # before atomically replacing it.
            os.fchmod(fh.fileno(), destination_mode)
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise
    return path
