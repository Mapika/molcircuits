from __future__ import annotations

import json
import stat

import pytest

from molcircuits.utils import manifest as manifest_module
from molcircuits.utils.manifest import Manifest, dump_json, source_tree_hash


def test_source_tree_hash_is_deterministic_and_nonempty():
    first = source_tree_hash()
    second = source_tree_hash()
    assert first == second
    assert len(first[0]) == 64
    assert first[1] > 10


def test_manifest_uses_explicit_environment_path(tmp_path, monkeypatch):
    path = tmp_path / "corrected" / "manifest.jsonl"
    output = tmp_path / "result.json"
    output.write_text("{}")
    monkeypatch.setenv("MOLCIRCUITS_MANIFEST", str(path))
    Manifest().record("test", {"ok": True}, config={"seed": 1},
                      files=[str(output)])
    row = json.loads(path.read_text())
    assert row["stage"] == "test"
    assert row["source_tree_sha256"] != "unknown"
    assert row["source_tree_sha256_at_record"] != "unknown"
    assert isinstance(row["source_tree_changed_during_process"], bool)
    assert str(output) in row["files"]


def test_dump_json_atomically_preserves_previous_artifact_on_failure(
        tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_text('{"status": "complete"}\n')

    def interrupted_dump(_obj, fh, **_kwargs):
        fh.write('{"status": "part')
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(manifest_module.json, "dump", interrupted_dump)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        dump_json({"status": "partial"}, str(target))

    assert json.loads(target.read_text()) == {"status": "complete"}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_dump_json_rejects_nonfinite_values_without_replacing_artifact(
        tmp_path, nonfinite):
    target = tmp_path / "result.json"
    target.write_text('{"status": "complete"}\n')

    with pytest.raises(ValueError, match="Out of range float values"):
        dump_json({"value": nonfinite}, str(target))

    assert json.loads(target.read_text()) == {"status": "complete"}
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def test_dump_json_writes_readable_mode_and_preserves_existing_mode(tmp_path):
    target = tmp_path / "result.json"
    dump_json({"status": "new"}, str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o644

    target.chmod(0o640)
    dump_json({"status": "replaced"}, str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
