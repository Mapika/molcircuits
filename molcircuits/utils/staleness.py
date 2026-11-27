"""Which artifacts predate the code that produced them.

A result JSON on disk is only evidence about the code that wrote it.  Once a
stage's source has been edited, its existing outputs describe the previous
implementation, and nothing about the file says so -- the number still parses,
still has a seed, still looks like a result.  The previous round of corrections
in this project turned up exactly that failure: figures were drawn from outputs
that predated fixes to the code paths behind them.

The check is a modification-time comparison, made meaningful by taking the
closure.  An artifact is stale when it is older than the newest file among the
stage script *and every ``molcircuits`` module it transitively imports*, because
editing a helper deep in the package invalidates outputs just as surely as
editing the script does, while leaving the script's own mtime untouched.

Its limits are worth being explicit about, since a green report is meant to be
trusted: mtimes move when a file is touched without being changed, so this
over-reports rather than under-reports, which is the safe direction; and it
cannot see edits to data or configuration that are not Python imports.  For a
content-addressed answer the manifest written by
:class:`molcircuits.utils.manifest.Manifest` records a hash of the whole source
tree per run -- this is the cheap check that runs before a rebuild, not a
replacement for that record.
"""
from __future__ import annotations

import ast
import os
from collections.abc import Iterable, Mapping, Sequence

from .. import paths

#: Stage script -> artifact patterns it writes, relative to the artifacts root.
#: ``{seed}`` expands over the seeds passed to :func:`stale_artifacts`; a pattern
#: without it is checked once.
STAGES: dict[str, tuple[str, tuple[str, ...]]] = {
    "01 datasets": ("experiments/01_build_datasets.py",
                    ("dataset_stats.json",)),
    "02 transformer": ("experiments/02_train_transformer.py",
                       ("tf_seed{seed}/train_metrics.json",)),
    "03 saes": ("experiments/03_train_saes.py",
                ("saes_seed{seed}/sae_summary.json",
                 "saes_seed{seed}/dictionary_stats.json")),
    "04 graph invariance": ("experiments/04_graph_invariance.py",
                            ("graph_invariance/summary_graph_invariance.json",)),
    "05 chemical hierarchy": ("experiments/05_chemical_hierarchy.py",
                              ("chemical_hierarchy/summary_chemical_hierarchy.json",)),
    "06 syntax and valence": ("experiments/06_syntax_and_valence.py",
                              ("pointers_seed{seed}/pointer_results.json",
                               "ring_patching_seed{seed}/ring_patching.json",
                               "valence_patching_seed{seed}/valence_patching.json")),
    "07 sparse and downstream": ("experiments/07_sparse_and_downstream.py",
                                 ("fragment_controls_seed{seed}/fragment_alignment_controls_v1.json",
                                  "moleculeace_seed{seed}/moleculeace_results.json",
                                  "tdc_seed{seed}/tdc_results.json")),
    "08 steering diagnostics": ("experiments/08_steering_diagnostics.py",
                                ("steering/summary_diagnostics.json",)),
    "09 constructive steering": ("experiments/09_constructive_steering.py",
                                 ("steering/summary_constructive_steering.json",)),
    "10 external models": ("experiments/10_external_models.py",
                           ("external_models/summary_external_models.json",)),
}


def repo_root() -> str:
    """Directory containing the ``molcircuits`` package and the ``experiments`` tree.

    Derived from the package location rather than hard-coded, so a second
    checkout checks itself instead of silently reporting on the first one.
    Three levels up from ``molcircuits/utils/staleness.py``: ``utils`` ->
    ``molcircuits`` -> the directory holding both the package and ``experiments``.
    """
    here = os.path.abspath(__file__)
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def module_file(module: str, repo: str) -> str | None:
    """Source file backing a dotted module name, or ``None`` if absent."""
    stem = os.path.join(repo, module.replace(".", os.sep))
    if os.path.exists(stem + ".py"):
        return stem + ".py"
    package_init = os.path.join(stem, "__init__.py")
    return package_init if os.path.exists(package_init) else None


def _imported_modules(node: ast.AST, base: str) -> list[str]:
    """Dotted names a single import statement refers to.

    ``from x import y`` yields both ``x`` and ``x.y``: ``y`` may be a submodule
    or may be a name defined inside ``x``, and resolving which would need an
    import.  Emitting both and discarding whichever has no file is cheaper and
    cannot miss a real dependency.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level:
        parts = base.split(".")
        root = ".".join(parts[:len(parts) - node.level + 1])
        head = f"{root}.{node.module}" if node.module else root
        return [head] + [f"{head}.{alias.name}" for alias in node.names]
    if node.module:
        return [node.module] + [f"{node.module}.{alias.name}"
                                for alias in node.names]
    return []


def import_closure(script: str, repo: str | None = None,
                   package: str = "molcircuits") -> set[str]:
    """Every ``package`` source file reachable from ``script`` by import."""
    repo = repo or repo_root()
    start = script if os.path.isabs(script) else os.path.join(repo, script)
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen or not os.path.exists(current):
            continue
        seen.add(current)
        try:
            tree = ast.parse(open(current).read())
        except (SyntaxError, UnicodeDecodeError):
            # A file that will not parse cannot contribute imports, but it is
            # still an input whose mtime counts -- so it stays in the set.
            continue
        base = os.path.relpath(os.path.dirname(current), repo).replace(os.sep, ".")
        for node in ast.walk(tree):
            for module in _imported_modules(node, base):
                if module.split(".")[0] != package:
                    continue
                resolved = module_file(module, repo)
                if resolved:
                    stack.append(resolved)
    return seen


def newest_input(files: Iterable[str]) -> tuple[str | None, float]:
    """The most recently modified file in a set, and its mtime."""
    existing = [f for f in files if os.path.exists(f)]
    if not existing:
        return None, 0.0
    newest = max(existing, key=os.path.getmtime)
    return newest, os.path.getmtime(newest)


def stale_artifacts(stages: Mapping[str, tuple[str, Sequence[str]]] | None = None,
                    repo: str | None = None,
                    artifacts: str | None = None,
                    seeds: Sequence[int] = (0, 1, 2)) -> list[dict]:
    """One record per artifact that exists, saying whether its code is newer.

    Artifacts that do not exist are omitted rather than reported as stale: a
    stage that has not run yet is a different problem from one whose output is
    out of date, and conflating them makes the stale count useless as a gate.
    """
    stages = STAGES if stages is None else stages
    repo = repo or repo_root()
    artifacts = artifacts or paths.artifacts()
    records = []
    for name, (script, patterns) in stages.items():
        closure = import_closure(script, repo)
        newest, code_mtime = newest_input(closure)
        for pattern in patterns:
            expanded = ([pattern.format(seed=s) for s in seeds]
                        if "{seed}" in pattern else [pattern])
            for relative in expanded:
                target = os.path.join(artifacts, relative)
                if not os.path.exists(target):
                    continue
                artifact_mtime = os.path.getmtime(target)
                records.append({
                    "stage": name,
                    "script": script,
                    "artifact": relative,
                    "artifact_mtime": artifact_mtime,
                    "code_mtime": code_mtime,
                    "newest_input": (os.path.relpath(newest, repo)
                                     if newest else None),
                    "n_source_files": len(closure),
                    "stale": bool(artifact_mtime < code_mtime),
                })
    return records


def summarize(records: Sequence[Mapping]) -> dict:
    """Counts and the offending inputs, grouped by stage."""
    stale = [r for r in records if r["stale"]]
    by_stage: dict[str, dict] = {}
    for record in stale:
        entry = by_stage.setdefault(
            record["stage"],
            {"newest_input": record["newest_input"], "artifacts": []})
        entry["artifacts"].append(record["artifact"])
    return {
        "n_checked": len(records),
        "n_stale": len(stale),
        "clean": not stale,
        "by_stage": by_stage,
    }
