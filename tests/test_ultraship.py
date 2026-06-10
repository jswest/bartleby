"""Tests for the /ultraship manifest core (scripts/ultraship.py).

The script lives under scripts/ (not an installed package), so it is loaded by
path, the same way test_release.py loads the release helper. Only the pure
functions — parsing, validation, and wave derivation — are exercised; the
agentic orchestration lives in the skill prose, not here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ULTRASHIP_PATH = Path(__file__).resolve().parent.parent / "scripts" / "ultraship.py"
_spec = importlib.util.spec_from_file_location("ultraship", _ULTRASHIP_PATH)
ultraship = importlib.util.module_from_spec(_spec)
# Register before exec: the module's dataclasses resolve their __module__ via
# sys.modules when their string annotations are introspected.
sys.modules["ultraship"] = ultraship
_spec.loader.exec_module(ultraship)


def _body(sub_issues: str, *, goal: str = "the bundle does X, checkably", notes: str = "") -> str:
    notes_block = (
        f"\n<!-- director-notes:start -->\n{notes}\n<!-- director-notes:end -->\n"
        if notes
        else ""
    )
    return (
        f"## Summary\n\nSome prose.\n\n**goal:** {goal}\n{notes_block}\n"
        f"### Sub-issues\n{sub_issues}\n"
    )


_TWO_INDEPENDENT = """\
- #10 — first thing.
  - **objective:** do the first thing
  - **touches:** `a.py`
  - **depends-on:** —
- #20 — second thing.
  - **objective:** do the second thing
  - **touches:** `b.py`
  - **depends-on:** —
"""


# --- parsing ---------------------------------------------------------------

def test_parse_basic_manifest():
    m = ultraship.parse_manifest(_body(_TWO_INDEPENDENT))
    assert m.goal == "the bundle does X, checkably"
    assert [i.number for i in m.sub_issues] == [10, 20]
    first = m.sub_issues[0]
    assert first.label == "first thing."
    assert first.objective == "do the first thing"
    assert first.touches == ["a.py"]
    assert first.depends_on == []


def test_parse_label_keeps_markdown_and_em_dash():
    body = _body(
        "- #236 — *(bug)* PER_WORKER_GB undercounts RSS.\n"
        "  - **objective:** track measured peak RSS\n"
        "  - **touches:** `bartleby/scribe/sizing.py`, `bartleby/scribe/workers.py`\n"
        "  - **depends-on:** —\n"
    )
    issue = ultraship.parse_manifest(body).sub_issues[0]
    assert issue.label == "*(bug)* PER_WORKER_GB undercounts RSS."
    assert issue.touches == ["bartleby/scribe/sizing.py", "bartleby/scribe/workers.py"]


def test_parse_depends_on_multiple():
    body = _body(
        "- #30 — third.\n"
        "  - **objective:** o\n"
        "  - **touches:** `c.py`\n"
        "  - **depends-on:** #10, #20\n"
    )
    assert ultraship.parse_manifest(body).sub_issues[0].depends_on == [10, 20]


def test_depends_on_space_separated_keeps_every_edge():
    # A separator typo must not silently drop an ordering edge (review #2).
    body = _body(
        "- #30 — third.\n"
        "  - **objective:** o\n"
        "  - **touches:** `c.py`\n"
        "  - **depends-on:** #10 and #20\n"
    )
    assert ultraship.parse_manifest(body).sub_issues[0].depends_on == [10, 20]


def test_depends_on_dedupes():
    body = _body(
        "- #30 — third.\n"
        "  - **objective:** o\n"
        "  - **touches:** `c.py`\n"
        "  - **depends-on:** #10, #10\n"
    )
    assert ultraship.parse_manifest(body).sub_issues[0].depends_on == [10]


def test_parse_tolerates_plain_and_bold_field_forms():
    body = _body(
        "- #5 — plain fields.\n"
        "  - objective: lowercase no bold\n"
        "  - touches: a.py, b.py\n"
        "  - depends-on: none\n"
    )
    issue = ultraship.parse_manifest(body).sub_issues[0]
    assert issue.objective == "lowercase no bold"
    assert issue.touches == ["a.py", "b.py"]
    assert issue.depends_on == []


def test_parse_director_notes_block():
    m = ultraship.parse_manifest(_body(_TWO_INDEPENDENT, notes="X out of scope.\nPrefer Y."))
    assert m.director_notes == "X out of scope.\nPrefer Y."


def test_missing_field_is_none_not_empty():
    # A wholly-absent depends-on line is None (structural error); a present "—"
    # is [] (valid). This distinction is what validation hangs on.
    body = _body(
        "- #7 — no deps line.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
    )
    issue = ultraship.parse_manifest(body).sub_issues[0]
    assert issue.depends_on is None


def test_goal_not_confused_with_objective():
    # The goal scan must ignore the Sub-issues section entirely.
    body = _body(_TWO_INDEPENDENT, goal="REAL GOAL")
    assert ultraship.parse_manifest(body).goal == "REAL GOAL"


# --- validation ------------------------------------------------------------

def test_valid_manifest_has_no_problems():
    m = ultraship.parse_manifest(_body(_TWO_INDEPENDENT))
    assert ultraship.validate_manifest(m) == []


def test_missing_goal_is_a_problem():
    body = f"### Sub-issues\n{_TWO_INDEPENDENT}\n"  # no goal line
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("goal" in p for p in problems)


def test_no_sub_issues_is_a_problem():
    body = "**goal:** something\n\nNo section here.\n"
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("no sub-issues" in p.lower() for p in problems)


def test_missing_required_field_is_a_problem():
    body = _body(
        "- #7 — incomplete.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"  # no depends-on
    )
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("depends-on" in p for p in problems)


def test_empty_objective_is_a_problem():
    body = _body(
        "- #7 — blank objective.\n"
        "  - **objective:**\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
    )
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("objective" in p for p in problems)


def test_unresolved_dependency_is_a_problem():
    body = _body(
        "- #10 — solo.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** #999\n"
    )
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("#999" in p for p in problems)


def test_duplicate_sub_issue_is_a_problem():
    body = _body(
        "- #10 — once.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
        "- #10 — twice.\n"
        "  - **objective:** o\n"
        "  - **touches:** `b.py`\n"
        "  - **depends-on:** —\n"
    )
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("more than once" in p for p in problems)


def test_dependency_cycle_is_a_problem():
    body = _body(
        "- #10 — a.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** #20\n"
        "- #20 — b.\n"
        "  - **objective:** o\n"
        "  - **touches:** `b.py`\n"
        "  - **depends-on:** #10\n"
    )
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("cycle" in p for p in problems)


# --- strictness: unrecognized content fails loud, never silently drops ------

def test_nested_header_is_flagged_and_does_not_corrupt_previous_issue():
    # An indented sub-issue header (a nesting typo) must not vanish silently nor
    # have its fields overwrite the previous issue (review #1).
    body = _body(
        "- #10 — first.\n"
        "  - **objective:** o1\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
        "  - #20 — nested by mistake.\n"
        "    - **objective:** o2\n"
        "    - **touches:** `b.py`\n"
        "    - **depends-on:** —\n"
    )
    m = ultraship.parse_manifest(body)
    # #10 keeps its own objective; #20's fields did not bleed into it.
    assert m.sub_issues[0].objective == "o1"
    assert m.sub_issues[0].touches == ["a.py"]
    problems = ultraship.validate_manifest(m)
    assert any("#20" in p or "unrecognized" in p for p in problems)


def test_star_bullet_header_is_flagged():
    # A `*`/`+`-bulleted header isn't the canonical `- #N` form (review #3).
    body = _body(
        "- #10 — ok.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
        "* #20 — star bullet.\n"
        "  - **objective:** o\n"
        "  - **touches:** `b.py`\n"
        "  - **depends-on:** —\n"
    )
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("unrecognized" in p for p in problems)


def test_duplicate_field_within_issue_is_flagged():
    # Review #7: a repeated field line must not silently last-write-win.
    body = _body(
        "- #10 — dup field.\n"
        "  - **objective:** first\n"
        "  - **objective:** second\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
    )
    problems = ultraship.validate_manifest(ultraship.parse_manifest(body))
    assert any("more than once" in p for p in problems)


def test_code_fence_in_section_does_not_drop_following_issue():
    # A `#` line inside a code fence must not truncate the section (review #4).
    body = _body(
        "- #10 — before fence.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
        "```\n"
        "# this is code, not a heading\n"
        "```\n"
        "- #20 — after fence.\n"
        "  - **objective:** o\n"
        "  - **touches:** `b.py`\n"
        "  - **depends-on:** —\n"
    )
    m = ultraship.parse_manifest(body)
    assert [i.number for i in m.sub_issues] == [10, 20]
    assert ultraship.validate_manifest(m) == []


def test_goal_in_director_notes_does_not_shadow_real_header():
    # Review #5: a `goal:` mention inside the notes prose must not win over the
    # real `**goal:**` header, even when it appears first.
    body = (
        "## Summary\n\n"
        "<!-- director-notes:start -->\n"
        "goal: BOGUS from notes prose\n"
        "<!-- director-notes:end -->\n\n"
        "**goal:** REAL GOAL\n\n"
        "### Sub-issues\n" + _TWO_INDEPENDENT
    )
    assert ultraship.parse_manifest(body).goal == "REAL GOAL"


# --- wave derivation -------------------------------------------------------

def test_independent_issues_share_one_wave():
    m = ultraship.parse_manifest(_body(_TWO_INDEPENDENT))
    assert ultraship.derive_waves(m) == [[10, 20]]


def test_depends_on_forces_a_later_wave():
    body = _body(
        "- #10 — base.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
        "- #20 — needs base.\n"
        "  - **objective:** o\n"
        "  - **touches:** `b.py`\n"
        "  - **depends-on:** #10\n"
    )
    assert ultraship.derive_waves(ultraship.parse_manifest(body)) == [[10], [20]]


def test_touches_overlap_serializes_independent_issues():
    # No dependency edge, but both touch a.py -> they cannot share a wave.
    body = _body(
        "- #10 — first writer.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`, `shared.py`\n"
        "  - **depends-on:** —\n"
        "- #20 — second writer.\n"
        "  - **objective:** o\n"
        "  - **touches:** `shared.py`\n"
        "  - **depends-on:** —\n"
    )
    assert ultraship.derive_waves(ultraship.parse_manifest(body)) == [[10], [20]]


def test_mixed_dag_waves():
    body = _body(
        "- #10 — root.\n"
        "  - **objective:** o\n"
        "  - **touches:** `a.py`\n"
        "  - **depends-on:** —\n"
        "- #20 — independent.\n"
        "  - **objective:** o\n"
        "  - **touches:** `b.py`\n"
        "  - **depends-on:** —\n"
        "- #30 — after root.\n"
        "  - **objective:** o\n"
        "  - **touches:** `c.py`\n"
        "  - **depends-on:** #10\n"
    )
    waves = ultraship.derive_waves(ultraship.parse_manifest(body))
    assert waves[0] == [10, 20]
    assert waves[1] == [30]


# --- CLI -------------------------------------------------------------------

def test_cli_validate_ok(tmp_path, capsys):
    body_file = tmp_path / "body.md"
    body_file.write_text(_body(_TWO_INDEPENDENT))
    rc = ultraship.main(["validate", str(body_file)])
    assert rc == 0
    assert "ok" in capsys.readouterr().out


def test_cli_validate_reports_and_exits_nonzero(tmp_path, capsys):
    body_file = tmp_path / "body.md"
    body_file.write_text("### Sub-issues\n(nothing)\n")  # no goal, no issues
    rc = ultraship.main(["validate", str(body_file)])
    assert rc == 1
    assert "not well-formed" in capsys.readouterr().err


def test_cli_plan_json(tmp_path, capsys):
    body_file = tmp_path / "body.md"
    body_file.write_text(_body(_TWO_INDEPENDENT))
    rc = ultraship.main(["plan", str(body_file), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["waves"] == [[10, 20]]
    assert payload["goal"] == "the bundle does X, checkably"
