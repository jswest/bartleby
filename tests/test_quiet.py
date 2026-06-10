"""Tests for offline-mode gating in `bartleby.lib.quiet` (issue #88).

Offline mode must only switch on once *every* model the run needs is cached,
so docling's lazily-downloaded layout/table models can still fetch on the first
scanned PDF instead of hard-failing.
"""

from __future__ import annotations

import os

import pytest

from bartleby.ingest.resolve import _required_hf_models
from bartleby.lib import quiet
from bartleby.lib.consts import DOCLING_HF_REPOS, EMBEDDING_MODEL


def _cache_model(root, repo_id: str) -> None:
    """Materialise a non-empty HF-cache snapshot for ``repo_id`` under ``root``."""
    folder = "models--" + repo_id.replace("/", "--")
    snap = root / folder / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    # A leftover offline flag from the real environment would mask setdefault.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    return cache


def test_model_cached_detects_presence(hf_cache):
    assert quiet._model_cached(EMBEDDING_MODEL) is False
    _cache_model(hf_cache, EMBEDDING_MODEL)
    assert quiet._model_cached(EMBEDDING_MODEL) is True


def test_offline_not_set_when_required_model_missing(hf_cache, monkeypatch):
    # Only the embedding model is cached; a docling model is still missing.
    _cache_model(hf_cache, EMBEDDING_MODEL)
    required = (EMBEDDING_MODEL, *DOCLING_HF_REPOS)
    quiet.setup_quiet_third_party(verbose=True, required_models=required)
    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_set_when_all_required_cached(hf_cache, monkeypatch):
    required = (EMBEDDING_MODEL, *DOCLING_HF_REPOS)
    for repo in required:
        _cache_model(hf_cache, repo)
    quiet.setup_quiet_third_party(verbose=True, required_models=required)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_offline_not_set_without_declared_models(hf_cache):
    # Even a fully-populated cache stays online when the caller declares nothing
    # — we never guess that the cache is complete.
    _cache_model(hf_cache, EMBEDDING_MODEL)
    quiet.setup_quiet_third_party(verbose=True, required_models=())
    assert "HF_HUB_OFFLINE" not in os.environ


def test_explicit_offline_env_is_respected(hf_cache, monkeypatch):
    # setdefault must not clobber a user's explicit choice.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    required = (EMBEDDING_MODEL, *DOCLING_HF_REPOS)
    for repo in required:
        _cache_model(hf_cache, repo)
    quiet.setup_quiet_third_party(verbose=True, required_models=required)
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_offline_blocked_detection(monkeypatch):
    err = RuntimeError(
        "Cannot find an appropriate cached snapshot folder ... and outgoing "
        "traffic has been disabled."
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert quiet.offline_blocked(err) is True
    # Same error but offline not active → not our doing.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    assert quiet.offline_blocked(err) is False
    # Unrelated error while offline → not flagged.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert quiet.offline_blocked(ValueError("bad page")) is False


def test_required_hf_models_includes_docling_only_when_active():
    assert _required_hf_models("pdfplumber", "sec2md") == (EMBEDDING_MODEL,)
    pdf_docling = _required_hf_models("docling", "sec2md")
    assert set(pdf_docling) == {EMBEDDING_MODEL, *DOCLING_HF_REPOS}
    html_docling = _required_hf_models("pdfplumber", "docling")
    assert set(html_docling) == {EMBEDDING_MODEL, *DOCLING_HF_REPOS}
