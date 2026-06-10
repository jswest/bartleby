"""Behavioral tests for the /ship guard hook (`.claude/hooks/guard-main-write.sh`).

The hook reads a PreToolUse(Bash) event as JSON on stdin and exits 2 to block or
0 to allow. We drive it as a subprocess against a throwaway git repo so the
current-branch fallback (bare push, commit) has a real HEAD to read.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard-main-write.sh"

BLOCK = 2
ALLOW = 0


def _run(command: str, cwd: Path) -> int:
    payload = json.dumps({"tool_input": {"command": command}, "cwd": str(cwd)})
    proc = subprocess.run(
        ["bash", str(HOOK)], input=payload, capture_output=True, text=True
    )
    return proc.returncode


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo_on_main(tmp_path: Path) -> Path:
    """A repo with one commit, a `feature` branch, currently checked out on main."""
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t.t")
    _git(tmp_path, "config", "user.name", "t")
    _git(tmp_path, "commit", "--allow-empty", "-m", "init")
    _git(tmp_path, "branch", "feature")
    return tmp_path


# (command, expected) evaluated with HEAD on `main`.
ON_MAIN = [
    # Explicit non-main refspec → allowed even though HEAD is main (the point).
    ("git push -u origin omnibus/v0.8.7", ALLOW),
    ("git push origin issue/248-foo", ALLOW),
    ("git push origin HEAD:issue/248-foo", ALLOW),
    ("git push -o ci.skip origin feature", ALLOW),  # option-value not read as remote
    ("git push origin --delete feature", ALLOW),
    # Refspec-less push → upstream-resolving → branch fallback → blocked on main.
    ("git push", BLOCK),
    ("git push origin", BLOCK),
    ("git push --all origin", BLOCK),
    ("git push --mirror origin", BLOCK),
    # Explicit main destination, in every spelling → blocked.
    ("git push origin main", BLOCK),
    ("git push origin HEAD:main", BLOCK),
    ("git push origin +main", BLOCK),
    ("git push origin refs/heads/main", BLOCK),
    ("git push origin develop:main", BLOCK),
    ("git push origin --delete main", BLOCK),
    ("git push -o ci.skip origin main", BLOCK),
    # commit stays current-branch-gated.
    ("git commit -m wip", BLOCK),
    ("git -C . commit -m wip", BLOCK),
    # Not a gated git write.
    ("echo git push origin main", ALLOW),
    ("git log --grep commit", ALLOW),
    ("git pull --ff-only origin main", ALLOW),
]


@pytest.mark.parametrize("command, expected", ON_MAIN)
def test_on_main(repo_on_main: Path, command: str, expected: int) -> None:
    assert _run(command, repo_on_main) == expected


# (command, expected) evaluated with HEAD on `feature`.
ON_FEATURE = [
    ("git push", ALLOW),  # fallback: not on main
    ("git push origin feature", ALLOW),
    ("git commit -m wip", ALLOW),
    # An explicit main push is blocked regardless of which branch is checked out.
    ("git push origin main", BLOCK),
    ("git push origin HEAD:main", BLOCK),
]


@pytest.mark.parametrize("command, expected", ON_FEATURE)
def test_on_feature(repo_on_main: Path, command: str, expected: int) -> None:
    _git(repo_on_main, "checkout", "feature")
    assert _run(command, repo_on_main) == expected


def test_cd_into_worktree_then_push_main_is_blocked(repo_on_main: Path) -> None:
    """A leading `cd` moves the effective dir; an explicit main push still blocks."""
    assert _run(f"cd {repo_on_main} && git push origin main", repo_on_main.parent) == BLOCK
