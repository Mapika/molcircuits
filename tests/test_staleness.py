"""An artifact older than any module its stage imports is stale."""
import os

from molcircuits.utils import staleness


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _fixture_repo(tmp_path):
    """A miniature repo: a stage script importing a helper two levels deep."""
    repo = str(tmp_path / "repo")
    _write(f"{repo}/molcircuits/__init__.py", "")
    _write(f"{repo}/molcircuits/deep/__init__.py", "")
    _write(f"{repo}/molcircuits/deep/helper.py", "VALUE = 1\n")
    _write(f"{repo}/molcircuits/middle.py",
           "from molcircuits.deep.helper import VALUE\n")
    _write(f"{repo}/scripts/stage.py",
           "import json\nfrom molcircuits.middle import VALUE\n")
    return repo


def _touch(path, when):
    os.utime(path, (when, when))


def test_repo_root_contains_the_package_and_the_scripts():
    """The bug this guards: two dirname() calls returned the package directory,
    so every stage script resolved to a path that does not exist."""
    root = staleness.repo_root()
    assert os.path.isdir(os.path.join(root, "molcircuits"))
    assert os.path.isdir(os.path.join(root, "experiments"))


def test_closure_reaches_transitively_imported_modules(tmp_path):
    """Editing a helper deep in the package invalidates outputs just as surely
    as editing the script, while leaving the script's own mtime untouched."""
    repo = _fixture_repo(tmp_path)
    closure = staleness.import_closure("scripts/stage.py", repo)
    relative = {os.path.relpath(p, repo) for p in closure}
    assert relative == {"scripts/stage.py", "molcircuits/middle.py",
                        "molcircuits/deep/helper.py"}


def test_closure_ignores_third_party_imports(tmp_path):
    repo = _fixture_repo(tmp_path)
    closure = staleness.import_closure("scripts/stage.py", repo)
    assert not any("json" in os.path.basename(p) for p in closure)


def test_module_file_resolves_packages_through_their_init(tmp_path):
    repo = _fixture_repo(tmp_path)
    assert staleness.module_file("molcircuits.deep", repo).endswith("deep/__init__.py")
    assert staleness.module_file("molcircuits.middle", repo).endswith("middle.py")
    assert staleness.module_file("molcircuits.absent", repo) is None


def test_a_deep_helper_edit_marks_the_artifact_stale(tmp_path):
    repo = _fixture_repo(tmp_path)
    artifacts = str(tmp_path / "artifacts")
    _write(f"{artifacts}/out_seed0/result.json", "{}")

    for name in ("scripts/stage.py", "molcircuits/middle.py",
                 "molcircuits/deep/helper.py"):
        _touch(f"{repo}/{name}", 1_000)
    _touch(f"{artifacts}/out_seed0/result.json", 2_000)

    stages = {"stage": ("scripts/stage.py", ("out_seed{seed}/result.json",))}
    fresh = staleness.stale_artifacts(stages, repo, artifacts, seeds=(0,))
    assert len(fresh) == 1
    assert not fresh[0]["stale"]

    # Touch only the deepest helper, well after the artifact was written.
    _touch(f"{repo}/molcircuits/deep/helper.py", 3_000)
    stale = staleness.stale_artifacts(stages, repo, artifacts, seeds=(0,))
    assert stale[0]["stale"]
    assert stale[0]["newest_input"] == "molcircuits/deep/helper.py"


def test_missing_artifacts_are_omitted_not_reported_stale(tmp_path):
    """A stage that has not run is a different problem from one whose output is
    out of date; conflating them makes the count useless as a gate."""
    repo = _fixture_repo(tmp_path)
    stages = {"stage": ("scripts/stage.py", ("never_ran_seed{seed}/result.json",))}
    records = staleness.stale_artifacts(stages, repo, str(tmp_path / "empty"),
                                        seeds=(0, 1, 2))
    assert records == []


def test_patterns_without_a_seed_are_checked_once(tmp_path):
    repo = _fixture_repo(tmp_path)
    artifacts = str(tmp_path / "artifacts")
    _write(f"{artifacts}/single/result.json", "{}")
    stages = {"stage": ("scripts/stage.py", ("single/result.json",))}
    records = staleness.stale_artifacts(stages, repo, artifacts, seeds=(0, 1, 2))
    assert len(records) == 1


def test_summary_groups_stale_outputs_by_stage(tmp_path):
    repo = _fixture_repo(tmp_path)
    artifacts = str(tmp_path / "artifacts")
    for seed in (0, 1):
        _write(f"{artifacts}/out_seed{seed}/result.json", "{}")
        _touch(f"{artifacts}/out_seed{seed}/result.json", 3_000)
    # Every source older than the artifacts, then one helper edited after them.
    for name in ("scripts/stage.py", "molcircuits/middle.py",
                 "molcircuits/deep/helper.py"):
        _touch(f"{repo}/{name}", 1_000)
    _touch(f"{repo}/molcircuits/deep/helper.py", 5_000)

    stages = {"stage": ("scripts/stage.py", ("out_seed{seed}/result.json",))}
    summary = staleness.summarize(
        staleness.stale_artifacts(stages, repo, artifacts, seeds=(0, 1)))
    assert summary["n_checked"] == 2
    assert summary["n_stale"] == 2
    assert not summary["clean"]
    assert summary["by_stage"]["stage"]["newest_input"] == "molcircuits/deep/helper.py"


def test_every_declared_stage_script_exists():
    """A stage map that has drifted from the scripts tree silently checks
    nothing: the closure of a missing file is empty and its mtime is 0."""
    root = staleness.repo_root()
    missing = [script for script, _ in staleness.STAGES.values()
               if not os.path.exists(os.path.join(root, script))]
    assert missing == []
