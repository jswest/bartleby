"""Smoke test for skill/scripts/list_documents.py."""

from __future__ import annotations

import json

import pytest

from bartleby.skill_scripts import list_documents
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


def _run(capsys, argv):
    with pytest.raises(SystemExit) as exc:
        list_documents.main(argv)
    return exc.value.code, capsys.readouterr()


def test_list_documents_happy_path(seeded_project, capsys):
    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 2
    by_name = {d["file_name"]: d for d in out["documents"]}
    assert by_name["alpha.pdf"]["has_summary"] is True
    assert by_name["alpha.pdf"]["image_count"] == 0
    assert by_name["alpha.pdf"]["title"] == "Alpha"
    assert by_name["alpha.pdf"]["description"] == "Test summary of alpha document."
    assert by_name["alpha.pdf"]["authored_date"] is None
    assert by_name["alpha.pdf"]["created_at"] is not None
    assert by_name["beta.txt"]["has_summary"] is False
    assert by_name["beta.txt"]["image_count"] == 0
    # Unsummarized doc reports title/description/authored_date as null.
    assert by_name["beta.txt"]["title"] is None
    assert by_name["beta.txt"]["description"] is None
    assert by_name["beta.txt"]["authored_date"] is None


def test_list_documents_brief_projects_three_fields(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"], "--brief",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["documents"]
    for d in out["documents"]:
        assert set(d) == {"id", "file_name", "title"}
    assert out["total"] == 2  # envelope unchanged


def test_list_documents_brief_and_verbose_are_mutually_exclusive(seeded_project, capsys):
    code, captured = _run(capsys, [
        "--project", seeded_project["project"], "--brief", "--verbose",
    ])
    # argparse mutually-exclusive-group error → JSON usage envelope (issue #402).
    assert code == 1
    assert json.loads(captured.out)["code"] == "USAGE_ERROR"


def test_list_documents_verbose_includes_chunk_count(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"], "--verbose",
    ])
    out = json.loads(capsys.readouterr().out)
    by_name = {d["file_name"]: d for d in out["documents"]}
    assert by_name["alpha.pdf"]["chunk_count"] == 4
    assert by_name["beta.txt"]["chunk_count"] == 2


def test_list_documents_surfaces_authored_date(seeded_project, capsys):
    from bartleby.db.connection import open_db
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "UPDATE summaries SET authored_date = ? WHERE document_id = ?",
            ("2024-03-15", seeded_project["doc_a"]),
        )
    finally:
        conn.close()

    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    by_name = {d["file_name"]: d for d in out["documents"]}
    assert by_name["alpha.pdf"]["authored_date"] == "2024-03-15"


def test_list_documents_reports_image_count(seeded_project, capsys):
    from bartleby.db.connection import open_db
    from tests._skill_fixtures import seed_image
    conn = open_db(seeded_project["project"])
    try:
        seed_image(conn, seeded_project["doc_a"],
                   file_hash="img-a-1", file_path="images/a1.jpg")
        seed_image(conn, seeded_project["doc_a"],
                   file_hash="img-a-2", file_path="images/a2.jpg",
                   image_index_on_page=2)
    finally:
        conn.close()

    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    by_name = {d["file_name"]: d for d in out["documents"]}
    assert by_name["alpha.pdf"]["image_count"] == 2
    assert by_name["beta.txt"]["image_count"] == 0


def test_list_documents_limit_and_offset(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"],
        "--limit", "1", "--offset", "1",
    ])
    out = json.loads(capsys.readouterr().out)
    assert len(out["documents"]) == 1
    assert out["total"] == 2


def _set_authored_date(project, document_id, value):
    from bartleby.db.connection import open_db
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "UPDATE summaries SET authored_date = ? WHERE document_id = ?",
            (value, document_id),
        )
    finally:
        conn.close()


def test_list_documents_date_bound_excludes_null_dated(seeded_project, capsys):
    # alpha is dated; beta has no summary at all (undated).
    _set_authored_date(seeded_project["project"], seeded_project["doc_a"],
                       "2024-03-15")
    list_documents.main([
        "--project", seeded_project["project"], "--authored-after", "2024-01-01",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["file_name"] for d in out["documents"]] == ["alpha.pdf"]
    assert out["total"] == 1
    # beta is dropped purely for being undated — that must be reported.
    assert out["filters"]["excluded_null_dated"] == 1


def test_list_documents_authored_before(seeded_project, capsys):
    _set_authored_date(seeded_project["project"], seeded_project["doc_a"],
                       "2024-03-15")
    list_documents.main([
        "--project", seeded_project["project"], "--authored-before", "2023-12-31",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["documents"] == []
    assert out["total"] == 0
    assert out["filters"]["excluded_null_dated"] == 1


def test_list_documents_date_range_both_bounds(seeded_project, capsys):
    _set_authored_date(seeded_project["project"], seeded_project["doc_a"],
                       "2024-03-15")
    list_documents.main([
        "--project", seeded_project["project"],
        "--authored-after", "2024-01-01", "--authored-before", "2024-12-31",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["file_name"] for d in out["documents"]] == ["alpha.pdf"]
    assert out["filters"]["excluded_null_dated"] == 1


def test_list_documents_include_nulls_keeps_undated(seeded_project, capsys):
    _set_authored_date(seeded_project["project"], seeded_project["doc_a"],
                       "2024-03-15")
    list_documents.main([
        "--project", seeded_project["project"],
        "--authored-after", "2024-01-01", "--include-nulls",
    ])
    out = json.loads(capsys.readouterr().out)
    by_name = {d["file_name"] for d in out["documents"]}
    assert by_name == {"alpha.pdf", "beta.txt"}
    assert out["total"] == 2
    # Nothing was excluded: the undated doc rode along (date bound active, so the
    # filters echo is present and reports zero).
    assert out["filters"]["excluded_null_dated"] == 0


def test_list_documents_file_like_single_pattern(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"], "--file-like", "alpha%",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["file_name"] for d in out["documents"]] == ["alpha.pdf"]
    assert out["total"] == 1
    assert out["filters"]["file_like"] == ["alpha%"]


def test_list_documents_file_like_repeated_ors(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"],
        "--file-like", "alpha%", "--file-like", "%.txt",
    ])
    out = json.loads(capsys.readouterr().out)
    assert {d["file_name"] for d in out["documents"]} == {"alpha.pdf", "beta.txt"}
    assert out["total"] == 2
    assert out["filters"]["file_like"] == ["alpha%", "%.txt"]


def test_list_documents_file_like_ands_with_date_bound(seeded_project, capsys):
    # alpha is dated and matches alpha%; beta matches the pattern's sibling but
    # the date bound + pattern both have to hold, so only alpha survives.
    _set_authored_date(seeded_project["project"], seeded_project["doc_a"],
                       "2024-03-15")
    list_documents.main([
        "--project", seeded_project["project"],
        "--file-like", "%.pdf", "--authored-after", "2024-01-01",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["file_name"] for d in out["documents"]] == ["alpha.pdf"]
    assert out["total"] == 1
    assert out["filters"]["file_like"] == ["%.pdf"]


def test_list_documents_file_like_no_match_empties(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"], "--file-like", "zzz%",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["documents"] == []
    assert out["total"] == 0
    assert out["filters"]["file_like"] == ["zzz%"]


def test_list_documents_in_documents_scopes_to_ids(seeded_project, capsys):
    # --in-documents restricts the listing to the named ids (the #111 scope
    # convention shared with search/scan).
    list_documents.main([
        "--project", seeded_project["project"],
        "--in-documents", str(seeded_project["doc_a"]),
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["id"] for d in out["documents"]] == [seeded_project["doc_a"]]
    assert out["total"] == 1
    assert out["filters"]["in_documents"] == [seeded_project["doc_a"]]


def test_list_documents_unscoped_default_unchanged(seeded_project, capsys):
    # Default (no --in-documents) still returns the whole corpus.
    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 2
    assert {d["file_name"] for d in out["documents"]} == {"alpha.pdf", "beta.txt"}


def test_list_documents_no_filter_omits_filters_echo(seeded_project, capsys):
    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    # No scope filter active → no filters echo at all (uniform contract).
    assert "filters" not in out


def test_list_documents_invalid_date_raises(seeded_project, capsys):
    code, captured = _run(capsys, [
        "--project", seeded_project["project"], "--authored-after", "2024",
    ])
    assert code == 1
    err = json.loads(captured.out)
    assert err["code"] == "INVALID_DATE"


def _set_title(project, document_id, title):
    from bartleby.db.connection import open_db
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "UPDATE summaries SET title = ? WHERE document_id = ?",
            (title, document_id),
        )
    finally:
        conn.close()


def _add_summary(project, document_id, *, title, authored_date=None):
    from bartleby.db.connection import open_db
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO summaries "
            "(document_id, title, description, text, model, authored_date) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, title, "desc", "body", "test", authored_date),
        )
    finally:
        conn.close()


def test_list_documents_default_sort_is_id(seeded_project, capsys):
    # Re-title alpha so it sorts *after* beta alphabetically; the default (id)
    # order must ignore that and stay in ingest order.
    _set_title(seeded_project["project"], seeded_project["doc_a"], "Zzz")
    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    assert [d["id"] for d in out["documents"]] == [
        seeded_project["doc_a"], seeded_project["doc_b"],
    ]


def test_list_documents_sort_title_is_alphabetical(seeded_project, capsys):
    # alpha re-titled "Zzz"; beta is unsummarized so it falls back to file_name
    # "beta.txt". Alphabetical → beta before alpha, the reverse of id order.
    _set_title(seeded_project["project"], seeded_project["doc_a"], "Zzz")
    list_documents.main([
        "--project", seeded_project["project"], "--sort", "title",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["file_name"] for d in out["documents"]] == ["beta.txt", "alpha.pdf"]


def test_list_documents_sort_date_newest_first(seeded_project, capsys):
    # Two dated docs in opposite id/date order: alpha (lower id) is older.
    _set_authored_date(seeded_project["project"], seeded_project["doc_a"],
                       "2024-03-15")
    _add_summary(seeded_project["project"], seeded_project["doc_b"],
                 title="Beta", authored_date="2025-01-01")
    list_documents.main([
        "--project", seeded_project["project"], "--sort", "date",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["file_name"] for d in out["documents"]] == ["beta.txt", "alpha.pdf"]


def test_list_documents_sort_date_puts_undated_last(seeded_project, capsys):
    # alpha dated, beta undated (no summary at all) → undated sorts last.
    _set_authored_date(seeded_project["project"], seeded_project["doc_a"],
                       "2024-03-15")
    list_documents.main([
        "--project", seeded_project["project"], "--sort", "date",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [d["file_name"] for d in out["documents"]] == ["alpha.pdf", "beta.txt"]


def test_list_documents_sort_rejects_unknown_value(seeded_project, capsys):
    code, captured = _run(capsys, [
        "--project", seeded_project["project"], "--sort", "bogus",
    ])
    # argparse choices error → JSON usage envelope (issue #402).
    assert code == 1
    assert json.loads(captured.out)["code"] == "USAGE_ERROR"


@pytest.mark.parametrize("bad", [
    ["--limit", "0"],     # positive_int rejects < 1
    ["--limit", "-5"],
    ["--offset", "-1"],   # nonneg_int rejects < 0
])
def test_list_documents_out_of_range_pagination_rejected(seeded_project, capsys, bad):
    # The shared positive/non-negative validators (issue #403) reject out-of-range
    # --limit/--offset at parse time, so the JSON envelope is emitted before any
    # query runs.
    code, captured = _run(capsys, ["--project", seeded_project["project"], *bad])
    assert code == 1
    assert json.loads(captured.out)["code"] == "USAGE_ERROR"


def test_list_documents_date_filter_composes_with_tag(seeded_project, capsys):
    from bartleby.db.connection import open_db
    project = seeded_project["project"]
    _set_authored_date(project, seeded_project["doc_a"], "2024-03-15")
    conn = open_db(project)
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES (?, ?)",
                    ("ch", "chapter docs"))
        tag_id = conn.last_insert_rowid()
        # Tag the *undated* doc only, so a date bound + this tag yields nothing.
        cur.execute("INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
                    (seeded_project["doc_b"], tag_id))
    finally:
        conn.close()

    list_documents.main([
        "--project", project, "--tag", "ch", "--authored-after", "2024-01-01",
    ])
    out = json.loads(capsys.readouterr().out)
    # beta carries the tag but is undated → excluded by the bound, counted once.
    assert out["documents"] == []
    assert out["total"] == 0
    assert out["filters"]["excluded_null_dated"] == 1


# ---------- --returning projection (issue #419) ----------


def test_list_documents_returning_projects_exact_fields(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"],
        "--returning", "document_id,file_name,title",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 2  # envelope unchanged
    for d in out["documents"]:
        assert list(d.keys()) == ["document_id", "file_name", "title"]
    by_id = {d["document_id"]: d for d in out["documents"]}
    assert by_id[seeded_project["doc_a"]]["file_name"] == "alpha.pdf"


def test_list_documents_returning_document_id_aliases_id(seeded_project, capsys):
    """document_id and id are both selectable and carry the same value."""
    list_documents.main([
        "--project", seeded_project["project"],
        "--returning", "id,document_id",
    ])
    out = json.loads(capsys.readouterr().out)
    for d in out["documents"]:
        assert d["id"] == d["document_id"]


def test_list_documents_returning_overrides_brief(seeded_project, capsys):
    list_documents.main([
        "--project", seeded_project["project"],
        "--returning", "document_id", "--brief",
    ])
    out = json.loads(capsys.readouterr().out)
    for d in out["documents"]:
        assert set(d) == {"document_id"}


def test_list_documents_returning_can_pull_verbose_fields(seeded_project, capsys):
    """--returning reaches verbose-tier columns without --verbose."""
    list_documents.main([
        "--project", seeded_project["project"],
        "--returning", "document_id,chunk_count,token_count",
    ])
    out = json.loads(capsys.readouterr().out)
    by_id = {d["document_id"]: d for d in out["documents"]}
    assert by_id[seeded_project["doc_a"]]["chunk_count"] == 4


def test_list_documents_returning_unknown_field_errors(seeded_project, capsys):
    code, captured = _run(capsys, [
        "--project", seeded_project["project"],
        "--returning", "document_id,bogus",
    ])
    assert code == 1
    out = json.loads(captured.out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"
    assert "document_id" in out["valid_fields"]


def test_list_documents_returning_unknown_field_errors_on_zero_documents(
    seeded_project, capsys
):
    """A typo'd --returning must error even when the filter matches no documents,
    rather than coming back as a silent empty list."""
    code, captured = _run(capsys, [
        "--project", seeded_project["project"],
        "--file-like", "zzzznomatchzzz",
        "--returning", "document_id,bogus",
    ])
    assert code == 1
    out = json.loads(captured.out)
    assert out["code"] == "UNKNOWN_RETURNING_FIELD"


def test_list_documents_default_projection_unchanged(seeded_project, capsys):
    list_documents.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    # No document_id leak into the default row.
    assert set(out["documents"][0]) == {
        "id", "file_name", "title", "description", "authored_date",
        "created_at", "has_summary", "image_count",
    }
