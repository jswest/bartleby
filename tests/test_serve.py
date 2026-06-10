"""Tests for `bartleby serve` — focused on the sync logic, not the dev server.

The actual `npm run dev` exec is out of scope here; we verify that the sync
symlinks ``src/`` from a checkout (and copies it from an installed package),
copies top-level files, preserves ``node_modules``, and removes stale entries.
"""

from __future__ import annotations

import os
import time

import pytest

from bartleby.commands import serve


def test_sync_symlinks_src_copies_files_preserves_node_modules(tmp_path):
    src = tmp_path / "web"
    dst = tmp_path / "serve"
    (src / "src").mkdir(parents=True)
    (src / "src" / "a.js").write_text("console.log('a');")
    (src / "package.json").write_text("{}")
    (src / "svelte.config.js").write_text("// config")

    # Pre-populate dst with node_modules + a stale top-level file and an old
    # src/ as a real directory to confirm it gets replaced with a symlink.
    (dst / "node_modules" / "left-pad").mkdir(parents=True)
    (dst / "node_modules" / "left-pad" / "index.js").write_text("// pinned")
    (dst / "src").mkdir(parents=True)
    (dst / "src" / "stale.js").write_text("// to be deleted")
    (dst / "stale-config.json").write_text("// stale top-level")

    serve._sync_web(src, dst)

    # src/ is now a symlink resolving back to the repo's src.
    assert (dst / "src").is_symlink()
    assert (dst / "src").resolve() == (src / "src").resolve()
    # Editing through the symlink edits the original — live HMR semantics.
    assert (dst / "src" / "a.js").read_text() == "console.log('a');"

    # Top-level files are real copies, not symlinks.
    assert (dst / "package.json").exists() and not (dst / "package.json").is_symlink()
    assert (dst / "svelte.config.js").exists()

    # Stale top-level entry from a previous run is gone.
    assert not (dst / "stale-config.json").exists()

    # node_modules survives the sync.
    assert (dst / "node_modules" / "left-pad" / "index.js").read_text() == "// pinned"


def test_sync_replaces_existing_symlink_idempotently(tmp_path):
    src = tmp_path / "web"
    dst = tmp_path / "serve"
    (src / "src").mkdir(parents=True)
    (src / "src" / "a.js").write_text("a")
    (src / "package.json").write_text("{}")

    serve._sync_web(src, dst)
    serve._sync_web(src, dst)  # Second call must not raise.

    assert (dst / "src").is_symlink()
    assert (dst / "src" / "a.js").read_text() == "a"


def test_sync_copies_src_when_symlink_disabled(tmp_path):
    """Installed-package path: src/ is copied, not symlinked, because the
    packaged source is read-only and there's nothing to live-edit."""
    src = tmp_path / "web"
    dst = tmp_path / "serve"
    (src / "src").mkdir(parents=True)
    (src / "src" / "a.js").write_text("console.log('a');")
    (src / "package.json").write_text("{}")

    serve._sync_web(src, dst, symlink_src=False)

    # src/ is a real copied directory, not a symlink.
    assert (dst / "src").is_dir()
    assert not (dst / "src").is_symlink()
    assert (dst / "src" / "a.js").read_text() == "console.log('a');"

    # Editing the copy does not touch the (read-only) packaged source.
    (dst / "src" / "a.js").write_text("// edited")
    assert (src / "src" / "a.js").read_text() == "console.log('a');"


def test_sync_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        serve._sync_web(tmp_path / "no-such-dir", tmp_path / "out")


def test_require_node_exits_when_missing(monkeypatch):
    monkeypatch.setattr(serve.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as exc:
        serve._require_node()
    assert exc.value.code == 1


def test_needs_install_true_when_no_marker(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    assert serve._needs_install(tmp_path) is True


def test_needs_install_false_when_marker_newer(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    marker = tmp_path / "node_modules" / ".package-lock.json"
    marker.parent.mkdir()
    marker.write_text("{}")
    os.utime(marker, (time.time() + 10, time.time() + 10))
    assert serve._needs_install(tmp_path) is False


def test_needs_install_true_when_package_json_newer(tmp_path):
    pkg = tmp_path / "package.json"
    pkg.write_text("{}")
    marker = tmp_path / "node_modules" / ".package-lock.json"
    marker.parent.mkdir()
    marker.write_text("{}")
    os.utime(pkg, (time.time() + 10, time.time() + 10))
    assert serve._needs_install(tmp_path) is True


def test_web_src_is_packaged_under_bartleby():
    # The UI now ships as package data at bartleby/web — it resolves the same
    # from a checkout or an installed wheel.
    assert serve.WEB_SRC.name == "web"
    assert serve.WEB_SRC.parent.name == "bartleby"
    assert (serve.WEB_SRC / "package.json").exists()
    assert (serve.WEB_SRC / "src").is_dir()


def test_is_source_checkout_true_in_repo():
    # The test suite runs from a checkout, so the repo root has pyproject.toml.
    assert serve._is_source_checkout() is True


def test_is_source_checkout_false_when_no_pyproject(monkeypatch, tmp_path):
    # Simulate an installed wheel: bartleby/web under site-packages, no
    # pyproject.toml two levels up.
    fake_web = tmp_path / "site-packages" / "bartleby" / "web"
    fake_web.mkdir(parents=True)
    monkeypatch.setattr(serve, "WEB_SRC", fake_web)
    assert serve._is_source_checkout() is False


def test_override_project_exports_env_when_db_exists(monkeypatch, tmp_path):
    # A real DB under the projects dir → BARTLEBY_PROJECT is exported and the
    # persisted active project (config.yaml) is never read or written.
    db = tmp_path / "projects" / "acme" / "bartleby.db"
    db.parent.mkdir(parents=True)
    db.write_text("")
    monkeypatch.setattr(serve, "project_db_path", lambda name: db)
    # setenv (not delenv) so monkeypatch tracks the var and teardown removes the
    # value _override_project writes directly into os.environ — no leak.
    monkeypatch.setenv("BARTLEBY_PROJECT", "")

    serve._override_project("acme")

    assert os.environ["BARTLEBY_PROJECT"] == "acme"


def test_override_project_exits_when_db_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        serve, "project_db_path", lambda name: tmp_path / "nope" / "bartleby.db"
    )
    monkeypatch.delenv("BARTLEBY_PROJECT", raising=False)

    with pytest.raises(SystemExit) as exc:
        serve._override_project("nope")

    assert exc.value.code == 1
    assert "BARTLEBY_PROJECT" not in os.environ
