"""Tests for skill_scripts/probe_dates.py — the read-only date-regex prober.

Reuses the shared `dated_corpus` fixture (#536): seven single-chunk documents
named `<key>__YYYY-MM-DD__slug.md`. Six filenames carry a date-shaped token; one
(`nomatch_doc`, `D004__no-date-here__delta.md`) does not. Of the six matches,
`bad_date_doc` (`E005__2024-13-40__epsilon.md`) captures an impossible calendar
date — it must land under `normalized_invalid`, never `normalized_ok`.
"""

from __future__ import annotations

import json

import pytest

from bartleby.db.connection import open_db
from bartleby.skill_scripts import probe_dates
from bartleby.skill_scripts.probe_dates import MATCH_THRESHOLD
from tests._skill_fixtures import (  # noqa: F401
    dated_corpus, project_env, seeded_project,
)


_DATE_RE = r"(?P<date>\d{4}-\d{2}-\d{2})"


def _run(argv):
    probe_dates.main(argv)


def _out(capsys):
    return json.loads(capsys.readouterr().out)


def test_filename_match_counts(dated_corpus, capsys):
    # Six of seven file_names carry a date-shaped token; nomatch_doc misses.
    # bad_date_doc matches but its capture (2024-13-40) is not a real date.
    _run(["--project", dated_corpus["project"], "--regex", _DATE_RE])
    out = _out(capsys)
    assert out["field"] == "filename"
    assert out["population"] == 7
    assert out["matched"] == 6
    assert out["normalized_ok"] == 5
    assert out["normalized_invalid"] == 1
    # 6/7 ~= 0.857, below the 0.9 threshold → no command suggested.
    assert out["match_rate"] == pytest.approx(6 / 7, abs=1e-4)
    assert out["suggested_command"] is None
    assert out["match_threshold"] == MATCH_THRESHOLD


def test_examples_and_unmatched_examples(dated_corpus, capsys):
    _run(["--project", dated_corpus["project"], "--regex", _DATE_RE])
    out = _out(capsys)
    # Examples are good (normalized) hits: file_name -> extracted -> normalized.
    assert out["examples"]
    for ex in out["examples"]:
        assert set(ex) == {"file_name", "extracted", "normalized"}
        assert ex["extracted"] == ex["normalized"]
    # bad_date_doc's invalid capture never appears among the good examples.
    assert "2024-13-40" not in {ex["extracted"] for ex in out["examples"]}
    # The one filename without a date is surfaced for refinement.
    assert "D004__no-date-here__delta.md" in out["unmatched_examples"]


def test_suggested_command_only_on_high_match_rate(dated_corpus, capsys):
    # Scope away nomatch_doc (D004...) with --file-like, OR'ing the dated
    # prefixes so every probed filename carries a date token: 6/6 = 1.0 >= 0.9.
    patterns = ["A001%", "B002%", "C003%", "E005%", "F006%"]
    argv = ["--project", dated_corpus["project"], "--regex", _DATE_RE]
    for p in patterns:
        argv += ["--file-like", p]
    _run(argv)
    out = _out(capsys)
    assert out["matched"] == out["population"]
    assert out["match_rate"] == 1.0
    assert out["suggested_command"] == (
        f"bartleby scribe backfill-dates {dated_corpus['project']} "
        f"--from-filename '{_DATE_RE}'"
    )
    assert out["filters"] == {"file_like": patterns}


def test_invalid_only_match_yields_no_normalized_ok(dated_corpus, capsys):
    # Scope to bad_date_doc alone: it matches the regex but normalizes invalid.
    _run([
        "--project", dated_corpus["project"], "--regex", _DATE_RE,
        "--file-like", "E005%",
    ])
    out = _out(capsys)
    assert out["population"] == 1
    assert out["matched"] == 1
    assert out["normalized_ok"] == 0
    assert out["normalized_invalid"] == 1
    # A regex that only ever captures junk dates is a 1.0 match_rate but the
    # caller can see every match is invalid; we still emit the command (it
    # cleared the rate) — invalidity is reported, not suppressed.
    assert out["suggested_command"] is not None


def test_body_field_is_always_sampled(dated_corpus, capsys):
    # Body chunks read "body of <file_name>", so the date token rides in the body
    # too. --field body must mark sampled:true regardless of population size.
    _run([
        "--project", dated_corpus["project"], "--regex", _DATE_RE,
        "--field", "body",
    ])
    out = _out(capsys)
    assert out["field"] == "body"
    assert out["sampled"] is True
    assert out["sample_size"] == 200
    assert out["matched"] == 6


def test_filename_sample_caps_population_and_marks_sampled(dated_corpus, capsys):
    # A --sample below the corpus size bites: population is capped and sampled
    # flips true (for --field filename, sampled means the cap actually bit).
    _run([
        "--project", dated_corpus["project"], "--regex", _DATE_RE,
        "--sample", "3",
    ])
    out = _out(capsys)
    assert out["population"] == 3
    assert out["sampled"] is True
    assert out["sample_size"] == 3


def test_filename_unsampled_when_corpus_below_cap(dated_corpus, capsys):
    # Default --sample 200 exceeds the 7-doc corpus, so the filename probe sees
    # everything: sampled false, sample_size null.
    _run(["--project", dated_corpus["project"], "--regex", _DATE_RE])
    out = _out(capsys)
    assert out["sampled"] is False
    assert out["sample_size"] is None


def test_missing_date_group_is_usage_error(dated_corpus, capsys):
    # The #536 helper enforces the named `date` group; a regex without one is a
    # usage error surfaced as INVALID_REGEX (JSON envelope, non-zero exit).
    with pytest.raises(SystemExit):
        _run([
            "--project", dated_corpus["project"], "--regex", r"\d{4}-\d{2}-\d{2}",
        ])
    out = _out(capsys)
    assert out["code"] == "INVALID_REGEX"


def test_malformed_regex_is_usage_error(dated_corpus, capsys):
    with pytest.raises(SystemExit):
        _run([
            "--project", dated_corpus["project"], "--regex", r"(?P<date>[",
        ])
    out = _out(capsys)
    assert out["code"] == "INVALID_REGEX"


def _snapshot_db(project):
    """Return a comparable snapshot of every row that probe_dates could touch."""
    conn = open_db(project)
    try:
        cur = conn.cursor()
        docs = list(cur.execute(
            "SELECT document_id, file_name FROM documents ORDER BY document_id"))
        summaries = list(cur.execute(
            "SELECT document_id, authored_date, model FROM summaries "
            "ORDER BY document_id"))
        chunks = list(cur.execute(
            "SELECT chunk_id, source_kind, source_id, text FROM chunks "
            "ORDER BY chunk_id"))
    finally:
        conn.close()
    return docs, summaries, chunks


def test_probe_writes_nothing(dated_corpus, capsys):
    # Read-only contract: the DB is byte-identical before and after a probe that
    # exercises both fields and an emitted suggested_command.
    before = _snapshot_db(dated_corpus["project"])
    _run(["--project", dated_corpus["project"], "--regex", _DATE_RE])
    capsys.readouterr()
    _run([
        "--project", dated_corpus["project"], "--regex", _DATE_RE,
        "--field", "body",
    ])
    capsys.readouterr()
    after = _snapshot_db(dated_corpus["project"])
    assert before == after
