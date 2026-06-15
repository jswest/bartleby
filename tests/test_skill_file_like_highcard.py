"""High-cardinality --file-like regression test for issue #632.

Before the fix, a broad --file-like glob that matched >32,766 documents
crashed search, scan, AND list_documents with "too many SQL variables" because
the matched ids were materialized as an IN (?, ?, ...) bind list that exceeded
SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (32,766 on this build).

The fix materializes the match set into a temp table (_scope_file_like) inside
SQLite and has all three consumers join against it instead.

This test creates 33,000 documents so the file-like match set exceeds the cap,
then asserts that all three commands return (not crash) the correct results.
"""

from __future__ import annotations

import json

import pytest

import bartleby.project
from bartleby.db.chunks import ChunkInput, insert_document_chunks
from bartleby.db.connection import open_db
from bartleby.skill_scripts import list_documents, scan, search
from tests._skill_fixtures import _emb, mock_embed, project_env  # noqa: F401


# One more than SQLITE_MAX_VARIABLE_NUMBER (32,766) so a naive IN-list would crash.
_HIGHCARD_COUNT = 33_000
_PATTERN = "hc_doc_%"
_MARKER = "highcard unique marker phrase"


@pytest.fixture
def highcard_project(project_env):  # noqa: F811
    """A project with 33,000 matching documents and 1 non-matching document.

    All 33k documents share a filename prefix ``hc_doc_`` and contain a unique
    marker phrase. One extra document (``other_doc.txt``) does NOT match the
    pattern, to verify the scope excludes it.
    """
    conn = open_db(project_env)
    try:
        cur = conn.cursor()

        # Bulk-insert 33k documents without chunks — we only need them for the
        # scope/filter path, not for FTS matches. One chunk per doc keeps the
        # test honest for scan (FTS must find something) without being slow.
        rows = [
            (f"h{i:07d}", f"hc_doc_{i:07d}.txt", f"/tmp/hc_doc_{i:07d}.txt", 1, 50)
            for i in range(_HIGHCARD_COUNT)
        ]
        cur.executemany(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        # Fetch the first inserted id to compute the range.
        first_id = conn.last_insert_rowid() - _HIGHCARD_COUNT + 1
        doc_ids = list(range(first_id, first_id + _HIGHCARD_COUNT))

        # One chunk per doc — content contains the marker so scan can find it.
        insert_document_chunks(conn, doc_ids[0], [
            ChunkInput(
                text=_MARKER,
                embedding=_emb(0.0),
                chunk_index=0,
            ),
        ])

        # One document that does NOT match the pattern — control for exclusion.
        cur.execute(
            "INSERT INTO documents (file_hash, file_name, file_path, page_count, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("other_hash", "other_doc.txt", "/tmp/other_doc.txt", 1, 20),
        )
        other_id = conn.last_insert_rowid()
    finally:
        conn.close()

    return {
        "project": project_env,
        "doc_ids": doc_ids,
        "other_id": other_id,
    }


def _run_list(project, extra_args):
    list_documents.main(["--project", project, *extra_args])


def _run_scan(project, extra_args):
    scan.main(["--project", project, *extra_args])


def _run_search(project, extra_args):
    search.main(["--project", project, *extra_args])


def test_file_like_highcard_list_documents(highcard_project, capsys):
    """list_documents --file-like with >32,766 matches must not crash."""
    _run_list(highcard_project["project"], ["--file-like", _PATTERN, "--limit", "10"])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == _HIGHCARD_COUNT
    assert len(out["documents"]) == 10
    assert out["filters"]["file_like"] == [_PATTERN]


def test_file_like_highcard_scan(highcard_project, capsys):
    """scan --file-like with >32,766 matches must not crash."""
    _run_scan(
        highcard_project["project"],
        [_MARKER, "--file-like", _PATTERN, "--limit", "5"],
    )
    out = json.loads(capsys.readouterr().out)
    # Exactly one doc has the marker chunk — the scope must not widen.
    assert out["total"] == 1
    assert out["filters"]["file_like"] == [_PATTERN]


def test_file_like_highcard_search(highcard_project, capsys):
    """search --file-like with >32,766 matches must not crash."""
    _run_search(
        highcard_project["project"],
        [_MARKER, "--file-like", _PATTERN, "--limit", "5", "--full-text"],
    )
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["file_like"] == [_PATTERN]
    # Exactly one doc has the marker chunk; the non-matching doc must be absent.
    assert len(out["results"]) >= 1
    result_source_ids = [r["source_id"] for r in out["results"]]
    assert f"document:{highcard_project['other_id']}" not in result_source_ids


def test_file_like_highcard_excludes_nonmatching(highcard_project, capsys):
    """The non-matching document must not appear when --file-like is active."""
    _run_list(
        highcard_project["project"],
        ["--file-like", _PATTERN, "--limit", str(_HIGHCARD_COUNT + 1)],
    )
    out = json.loads(capsys.readouterr().out)
    returned_ids = [int(d["id"].split(":")[1]) for d in out["documents"]]
    assert highcard_project["other_id"] not in returned_ids
