"""Drift gate for the vendored, drawer-pressed skills (`ship`, `ultraship`).

These two skill dirs under `.claude/skills/` are *pressed* from a local skill
drawer by `signet`, not hand-authored here. This test fails if the checked-in
copy diverges from the drawer — so a hand-edit in the repo (or a drawer that
advanced without a re-press) is caught at the normal `uv run pytest` gate.

It **skips** when there's no drawer to compare against — `signet` not installed,
or the drawer doesn't carry the skill. That's the clone / CI / teammate case:
the only machine that can be out of sync with the drawer is one that *has* it, so
skip-if-absent never penalizes a checkout without the drawer.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# `signet press --check` exit codes: 0 = in sync, 1 = drift, 2 = skill/drawer
# absent (resolve failure) or bad args.
IN_SYNC = 0
DRIFT = 1
NO_DRAWER = 2

PRESSED_SKILLS = ["ship", "ultraship"]


@pytest.mark.parametrize("skill", PRESSED_SKILLS)
def test_pressed_skill_matches_drawer(skill: str) -> None:
    if shutil.which("signet") is None:
        pytest.skip("signet not installed — no drawer to compare against")

    proc = subprocess.run(
        ["signet", "press", skill, "--into", str(REPO_ROOT), "--check"],
        capture_output=True,
        text=True,
    )

    if proc.returncode == NO_DRAWER:
        pytest.skip(f"drawer has no '{skill}' skill here:\n{proc.stderr.strip()}")

    assert proc.returncode == IN_SYNC, (
        f"vendored '{skill}' skill has drifted from the drawer — re-press it "
        f"(`signet press {skill} --into .`) or sync the drawer.\n"
        f"{proc.stdout.strip()}\n{proc.stderr.strip()}"
    )
