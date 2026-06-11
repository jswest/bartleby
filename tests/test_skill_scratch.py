"""Skill scratch directory: created mode 700, never world-readable /tmp (issue #13)."""

from __future__ import annotations

import stat

import bartleby.config as config
from bartleby.skill_scripts import list_documents
from tests._skill_fixtures import project_env  # noqa: F401


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_ensure_scratch_dir_creates_mode_700(tmp_path):
    d = config.ensure_scratch_dir()
    assert d == tmp_path / "tmp"
    assert d.is_dir()
    assert _mode(d) == 0o700


def test_ensure_scratch_dir_idempotent_and_tightens_perms(tmp_path):
    loose = tmp_path / "tmp"
    loose.mkdir()
    loose.chmod(0o777)  # simulate a pre-existing world-readable dir

    config.ensure_scratch_dir()
    config.ensure_scratch_dir()  # second call must not raise

    assert _mode(loose) == 0o700


def test_runner_creates_scratch_dir(project_env, capsys):
    """Any skill invocation must materialize the scratch dir before the agent writes."""
    assert not config.scratch_dir().exists()

    list_documents.main(["--project", project_env])
    capsys.readouterr()  # drain the JSON payload

    assert config.scratch_dir().is_dir()
    assert _mode(config.scratch_dir()) == 0o700
