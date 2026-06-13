"""Tests for `bartleby scribe backfill-dates` (#536).

Covers the backfill command (filename → authored_date, stub creation, dry-run,
overwrite, invalid-date counting, idempotency, #254 sections) and the
honest-stub read side (describe_corpus / list_documents / read_document).
"""

from __future__ import annotations

import json

import pytest

from bartleby.commands import backfill
from bartleby.db.connection import open_db
from bartleby.ingest.summarize import (
    FilenameDateError,
    compile_filename_date_regex,
    extract_filename_date,
)
from bartleby.skill_scripts import describe_corpus, list_documents, read_document
from tests._skill_fixtures import (  # noqa: F401
    dated_corpus,
    project_env,
    seeded_project,
)

ISO_RE = r"(?P<date>\d{4}-\d{2}-\d{2})"


def _date(project, document_id):
    conn = open_db(project)
    try:
        row = conn.cursor().execute(
            "SELECT authored_date, model FROM summaries WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return row if row else None
    finally:
        conn.close()


def _summary_count(project, document_id):
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM summaries WHERE document_id = ?",
            (document_id,),
        ).fetchone()[0]
    finally:
        conn.close()


def _capture_console(monkeypatch):
    """Collect ``console.info`` lines (the report rows). The shared Rich Console
    binds its stream at import time, so capsys/capfd miss it — the repo's idiom
    is to monkeypatch the console functions directly."""
    lines: list[str] = []
    monkeypatch.setattr(backfill.console, "info", lambda m: lines.append(m))
    monkeypatch.setattr(backfill.console, "big", lambda m: lines.append(m))
    monkeypatch.setattr(backfill.console, "complete", lambda m: lines.append(m))
    return lines


# ---- helper: compile/extract ------------------------------------------------

def test_compile_requires_named_date_group():
    with pytest.raises(FilenameDateError):
        compile_filename_date_regex(r"\d{4}-\d{2}-\d{2}")


def test_compile_rejects_bad_regex():
    with pytest.raises(FilenameDateError):
        compile_filename_date_regex(r"(?P<date>")


def test_extract_returns_raw_substring_or_none():
    c = compile_filename_date_regex(ISO_RE)
    assert extract_filename_date(c, "X__2021-03-15__y.md") == "2021-03-15"
    assert extract_filename_date(c, "no-date.md") is None


# ---- command: writes & counts ----------------------------------------------

def test_backfill_updates_existing_summary_and_inserts_stub(dated_corpus):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)

    # real summary got its NULL date filled
    assert _date(dated_corpus["project"], dated_corpus["summary_doc"]) == (
        "2021-03-15", "test")
    # no-summary doc got a backfill stub
    assert _date(dated_corpus["project"], dated_corpus["stub_doc"]) == (
        "2022-07-04", "backfill")


def test_backfill_leaves_dated_summary_unless_overwrite(dated_corpus):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    assert _date(dated_corpus["project"], dated_corpus["dated_doc"])[0] == "2099-12-31"

    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE,
                  overwrite=True)
    assert _date(dated_corpus["project"], dated_corpus["dated_doc"])[0] == "2023-01-09"


def test_backfill_skips_unmatched(dated_corpus):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    # nomatch doc has no summary and stays summary-less
    assert _summary_count(dated_corpus["project"], dated_corpus["nomatch_doc"]) == 0


def test_backfill_counts_invalid_date_never_writes(dated_corpus, monkeypatch):
    lines = _capture_console(monkeypatch)
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    # bad_date_doc matched the regex but 2024-13-40 is not a real date → no row
    assert _summary_count(dated_corpus["project"], dated_corpus["bad_date_doc"]) == 0
    assert any("invalid date:      1" in ln for ln in lines)


def test_backfill_dates_sections_too(dated_corpus):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    assert _date(dated_corpus["project"], dated_corpus["parent_doc"]) == (
        "2020-05-20", "backfill")
    # the #254 section shares the parent file_name, gets the same date via a stub
    assert _date(dated_corpus["project"], dated_corpus["section_doc"]) == (
        "2020-05-20", "backfill")


def test_backfill_dry_run_mutates_nothing(dated_corpus, monkeypatch):
    lines = _capture_console(monkeypatch)
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE,
                  dry_run=True)
    assert _date(dated_corpus["project"], dated_corpus["summary_doc"]) == (None, "test")
    assert _summary_count(dated_corpus["project"], dated_corpus["stub_doc"]) == 0
    # stub_doc + parent_doc + section_doc (dated_doc already has a date → skipped)
    assert any("would insert stub:  3" in ln for ln in lines)
    assert any("would update date:  1" in ln for ln in lines)  # summary_doc


def test_backfill_is_idempotent(dated_corpus):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    before = _summary_count(dated_corpus["project"], dated_corpus["stub_doc"])
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    after = _summary_count(dated_corpus["project"], dated_corpus["stub_doc"])
    assert before == after == 1
    assert _date(dated_corpus["project"], dated_corpus["stub_doc"]) == (
        "2022-07-04", "backfill")


def test_backfill_bad_regex_exits_nonzero(dated_corpus, capsys):
    with pytest.raises(SystemExit) as exc:
        backfill.main(project=dated_corpus["project"],
                      from_filename=r"\d{4}")  # no named group
    assert exc.value.code != 0


# ---- honest-stub read side --------------------------------------------------

def test_stub_does_not_inflate_summary_coverage(dated_corpus, capsys):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    capsys.readouterr()
    describe_corpus.main(["--project", dated_corpus["project"]])
    out = json.loads(capsys.readouterr().out)
    # Only the two real summaries (summary_doc, dated_doc) count as summarized,
    # not the backfill stubs.
    assert out["summary_coverage"]["summarized"] == 2
    # but the date block counts every dated doc, stubs included
    assert out["authored_date"]["dated_document_count"] >= 4


def test_list_documents_has_summary_false_for_stub(dated_corpus, capsys):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    capsys.readouterr()
    list_documents.main(["--project", dated_corpus["project"], "--limit", "100"])
    out = json.loads(capsys.readouterr().out)
    by_id = {d["id"]: d for d in out["documents"]}
    stub = by_id[dated_corpus["stub_doc"]]
    assert stub["has_summary"] is False
    assert stub["authored_date"] == "2022-07-04"
    assert stub["title"] is None
    # a real summary still reads has_summary=true
    assert by_id[dated_corpus["summary_doc"]]["has_summary"] is True


def test_read_document_suppresses_stub_summary_but_keeps_date(dated_corpus, capsys):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    capsys.readouterr()
    read_document.main([
        "--project", dated_corpus["project"],
        "--document", str(dated_corpus["stub_doc"]),
        "--summary",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["summary"] is None
    assert out["document"]["authored_date"] == "2022-07-04"


def test_read_document_exposes_authored_date_for_real_summary(dated_corpus, capsys):
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    capsys.readouterr()
    read_document.main([
        "--project", dated_corpus["project"],
        "--document", str(dated_corpus["summary_doc"]),
        "--summary",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["summary"] == "real summary text"
    assert out["document"]["authored_date"] == "2021-03-15"


def test_backfill_does_not_reclassify_container_as_unsummarized(dated_corpus, capsys):
    # A backfill stub on a #254 *container* must not shift it into the
    # unsummarized tally: the container still owes no real summary, so the
    # container exclusion keys off the same `model != 'backfill'` sentinel as
    # summary_coverage. (Regression: a bare `NOT IN (summaries)` saw the stub and
    # *deflated* coverage — the opposite of the honest-stub goal.)
    def _unsummarized():
        capsys.readouterr()
        describe_corpus.main(["--project", dated_corpus["project"]])
        return json.loads(capsys.readouterr().out)["summary_coverage"]["unsummarized"]

    before = _unsummarized()
    backfill.main(project=dated_corpus["project"], from_filename=ISO_RE)
    assert _unsummarized() == before
