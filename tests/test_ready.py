"""Tests for the `bartleby ready` skill installer."""

from __future__ import annotations

import json

import pytest

import bartleby.commands.ready as ready


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "skills" / "bartleby"


def _marker(dest):
    return json.loads((dest / ready.MARKER_NAME).read_text())


def test_fresh_install_copies_skill_and_writes_marker(dest):
    ready.main(dest=dest)

    assert (dest / "SKILL.md").is_file()
    assert (dest / "README.md").is_file()
    marker = _marker(dest)
    assert marker["version"]
    assert marker["hash"] == ready._hash_dir(ready._source_dir())


def test_rerun_when_up_to_date_is_a_noop(dest, monkeypatch):
    ready.main(dest=dest)

    calls = []
    monkeypatch.setattr(ready, "_install", lambda *a, **k: calls.append(1))
    ready.main(dest=dest)

    assert calls == []  # nothing reinstalled when already current


def test_force_reinstalls_even_when_up_to_date(dest, monkeypatch):
    ready.main(dest=dest)

    calls = []
    real_install = ready._install

    def spy(*a, **k):
        calls.append(1)
        real_install(*a, **k)

    monkeypatch.setattr(ready, "_install", spy)
    ready.main(dest=dest, force=True)

    assert calls == [1]  # --force rewrites despite being up to date
    assert (dest / "SKILL.md").is_file()


def test_drift_is_repaired_on_plain_rerun(dest):
    ready.main(dest=dest)
    (dest / "SKILL.md").write_text("hand-edited, now stale")

    ready.main(dest=dest)  # no --force needed; drift alone triggers a refresh

    assert ready._hash_dir(dest) == ready._hash_dir(ready._source_dir())


def test_stale_marker_refreshes_without_reinstall(dest, monkeypatch):
    ready.main(dest=dest)
    # Simulate a content-identical version bump: the skill files are byte-for-byte
    # the same, but the marker was written by an older bartleby.
    marker = _marker(dest)
    (dest / ready.MARKER_NAME).write_text(
        json.dumps({"version": "0.0.1", "hash": marker["hash"]}) + "\n"
    )

    calls = []
    monkeypatch.setattr(ready, "_install", lambda *a, **k: calls.append(1))
    ready.main(dest=dest)

    assert calls == []  # marker-only refresh, never a full reinstall
    assert _marker(dest)["version"] == ready.__version__


def test_check_agrees_with_default_after_refresh(dest, monkeypatch):
    ready.main(dest=dest)
    marker = _marker(dest)
    (dest / ready.MARKER_NAME).write_text(
        json.dumps({"version": "0.0.1", "hash": marker["hash"]}) + "\n"
    )

    messages = []
    for name in ("complete", "big", "warn"):
        monkeypatch.setattr(ready.console, name, lambda m, _m=messages: _m.append(m))

    ready.main(dest=dest)  # default path heals the stamp to the running version
    assert messages == [
        f"Updated skill marker v0.0.1 → v{ready.__version__} at {dest}"
    ]

    messages.clear()
    ready.main(dest=dest, check=True)  # read-only; now reports the healed version

    assert messages == [f"Skill is up to date (v{ready.__version__}) at {dest}."]


def test_check_up_to_date_returns_without_error(dest):
    ready.main(dest=dest)
    ready.main(dest=dest, check=True)  # does not raise


def test_check_reports_drift_and_exits_nonzero(dest):
    ready.main(dest=dest)
    (dest / "SKILL.md").write_text("drifted")

    with pytest.raises(SystemExit) as exc:
        ready.main(dest=dest, check=True)
    assert exc.value.code == 1


def test_check_when_not_installed_exits_nonzero(dest):
    with pytest.raises(SystemExit) as exc:
        ready.main(dest=dest, check=True)
    assert exc.value.code == 1


def test_check_writes_nothing(dest):
    with pytest.raises(SystemExit):
        ready.main(dest=dest, check=True)
    assert not dest.exists()


def test_dest_override_installs_there(tmp_path):
    custom = tmp_path / "elsewhere" / "bartleby"
    ready.main(dest=custom)
    assert (custom / "SKILL.md").is_file()


def test_refuses_to_overwrite_unrelated_directory(dest):
    dest.mkdir(parents=True)
    bystander = dest / "important.txt"
    bystander.write_text("not a skill")

    with pytest.raises(SystemExit) as exc:
        ready.main(dest=dest)
    assert exc.value.code == 1
    assert bystander.exists()  # left untouched
