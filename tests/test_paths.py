"""Root resolution must be lazy, independently overridable, and relative where it matters."""
import os

from molcircuits import paths


def test_roots_derive_from_a_single_variable(monkeypatch):
    monkeypatch.setenv("MOLCIRCUITS_ROOT", "/scratch/run")
    monkeypatch.delenv("MOLCIRCUITS_ARTIFACTS", raising=False)
    monkeypatch.delenv("MOLCIRCUITS_DATA", raising=False)
    assert paths.root() == "/scratch/run"
    assert paths.artifacts() == "/scratch/run/artifacts"
    assert paths.data() == "/scratch/run/data"


def test_each_root_overrides_independently(monkeypatch):
    monkeypatch.setenv("MOLCIRCUITS_ROOT", "/scratch/run")
    monkeypatch.setenv("MOLCIRCUITS_ARTIFACTS", "/scratch/rerun/artifacts")
    assert paths.artifacts() == "/scratch/rerun/artifacts"
    assert paths.data() == "/scratch/run/data"


def test_resolution_is_lazy_not_import_time(monkeypatch):
    monkeypatch.setenv("MOLCIRCUITS_ARTIFACTS", "/first")
    assert paths.artifacts() == "/first"
    monkeypatch.setenv("MOLCIRCUITS_ARTIFACTS", "/second")
    assert paths.artifacts() == "/second"


def test_manifest_default_stays_relative(monkeypatch):
    monkeypatch.delenv("MOLCIRCUITS_MANIFEST", raising=False)
    assert not os.path.isabs(paths.manifest())


def test_tokenizer_is_none_when_unset(monkeypatch):
    monkeypatch.delenv("MOLCIRCUITS_TOKENIZER", raising=False)
    assert paths.tokenizer() is None


def test_describe_reports_every_root(monkeypatch):
    monkeypatch.setenv("MOLCIRCUITS_ROOT", "/scratch/run")
    monkeypatch.delenv("MOLCIRCUITS_ARTIFACTS", raising=False)
    described = paths.describe()
    assert set(described) == {"root", "artifacts", "data", "tokenizer", "manifest"}
    assert described["artifacts"] == "/scratch/run/artifacts"
