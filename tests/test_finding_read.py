"""Tests for `bartleby finding read` (#649).

Covers: default Markdown output (title/meta/footnotes), live chunk citation
rendering with †, dangling citation rendering with ‡, external citation
rendering with §, --json falling through to read_finding JSON, and --render
not crashing.
"""

from __future__ import annotations

import json
import sys

import pytest

from bartleby.commands import finding as finding_cmd
from bartleby.db.connection import open_db
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seeded_project,
    seed_finding,
    unprefix,
)


def _active_session_id(project):
    from bartleby.session import ensure_active_session
    return ensure_active_session(project)


def _doc_chunk_ids(conn, doc_id, limit=2):
    return [
        r[0] for r in conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id=? ORDER BY chunk_index LIMIT ?",
            (doc_id, limit),
        )
    ]


# ---------------------------------------------------------------------------
# Default Markdown output
# ---------------------------------------------------------------------------

def test_read_emits_title_h1(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim.[^chunk:{cited[0]}]"
        finding_id, _ = seed_finding(
            conn, session_id, title="My Title", description="hook", body=body,
            cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out
    assert "# My Title" in out


def test_read_emits_meta_subtitle(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim.[^chunk:{cited[0]}]"
        finding_id, _ = seed_finding(
            conn, session_id, title="Meta Test", description="d", body=body,
            cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out
    # session_name and saved appear in the metadata subtitle
    assert "session:" in out
    assert "saved:" in out


def test_read_emits_description(seeded_project, capsys):
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim.[^chunk:{cited[0]}]"
        finding_id, _ = seed_finding(
            conn, session_id, title="T", description="The unique description text here.",
            body=body, cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out
    assert "The unique description text here." in out


# ---------------------------------------------------------------------------
# Live chunk citation → † footnote
# ---------------------------------------------------------------------------

def test_read_live_citation_renders_dagger(seeded_project, capsys):
    """A resolved chunk citation renders as [^N] inline + '† file' in the footnote block."""
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim[^chunk:{cited[0]}]."
        finding_id, _ = seed_finding(
            conn, session_id, title="T", description="d", body=body,
            cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out

    # The raw [^chunk:N] marker is gone; a sequential footnote reference appears.
    assert "[^chunk:" not in out
    assert "[^1]" in out
    # The footnote definition carries the dagger and file name.
    assert "[^1]: †" in out
    assert "alpha.pdf" in out


def test_read_live_citation_with_page(seeded_project, capsys, tmp_path):
    """A chunk with a page_number renders 'file · p.N'."""
    from bartleby.db.chunks import ChunkInput, insert_document_chunks
    from bartleby.db.schema import EMBEDDING_DIM

    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        emb = [0.01 * i for i in range(EMBEDDING_DIM)]
        insert_document_chunks(conn, seeded_project["doc_a"], [
            ChunkInput(text="paged chunk", embedding=emb, chunk_index=99,
                       section_heading=None, page_number=42, content_type="text"),
        ])
        cited = [conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id=? AND chunk_index=99",
            (seeded_project["doc_a"],),
        ).fetchone()[0]]
        body = f"Paged claim[^chunk:{cited[0]}]."
        finding_id, _ = seed_finding(
            conn, session_id, title="T", description="d", body=body,
            cited_chunk_ids=cited,
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out
    assert "· p.42" in out


# ---------------------------------------------------------------------------
# Dangling citation → ‡ footnote
# ---------------------------------------------------------------------------

def test_read_dangling_citation_renders_double_dagger(seeded_project, capsys):
    """A dangling [^chunk:N] (source gone) renders as ‡ in the footnote block."""
    from bartleby.db.chunks import delete_chunks_for

    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim[^chunk:{cited[0]}]."
        finding_id, _ = seed_finding(
            conn, session_id, title="T", description="d", body=body,
            cited_chunk_ids=cited[:1],
        )
        delete_chunks_for(conn, "document", seeded_project["doc_a"])
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out

    assert "[^1]" in out
    assert "[^1]: ‡ source no longer available" in out
    # dagger should NOT appear (it's dangling, not live)
    assert "† alpha.pdf" not in out


# ---------------------------------------------------------------------------
# External citation → § footnote
# ---------------------------------------------------------------------------

def test_read_external_url_citation_renders_section(seeded_project, capsys):
    """An external [^url:…] citation renders § in the footnote block."""
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = (
            f"Claim[^chunk:{cited[0]}]. See also[^url:https://example.com/report]."
        )
        finding_id, _ = seed_finding(
            conn, session_id, title="T", description="d", body=body,
            cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out

    # External marker is rewritten away; the footnote def carries §
    assert "[^url:" not in out
    assert "§" in out
    assert "https://example.com/report" in out


def test_read_external_doc_citation_renders_section(seeded_project, capsys):
    """An external [^doc:…] citation renders § in the footnote block."""
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim[^chunk:{cited[0]}]. From[^doc:WHO 2023 report]."
        finding_id, _ = seed_finding(
            conn, session_id, title="T", description="d", body=body,
            cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out

    assert "[^doc:" not in out
    assert "§" in out
    assert "WHO 2023 report" in out


# ---------------------------------------------------------------------------
# Footnote numbering — sequential across chunk + external
# ---------------------------------------------------------------------------

def test_read_footnote_numbering_is_sequential(seeded_project, capsys):
    """Footnote numbers run 1…N across both chunk and external citations."""
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"], limit=2)
        body = (
            f"First[^chunk:{cited[0]}] and second[^chunk:{cited[1]}]"
            f" and ext[^url:https://example.com]."
        )
        finding_id, _ = seed_finding(
            conn, session_id, title="T", description="d", body=body,
            cited_chunk_ids=cited,
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=False)
    out = capsys.readouterr().out

    assert "[^1]" in out
    assert "[^2]" in out
    assert "[^3]" in out
    # definitions present
    assert "[^1]:" in out
    assert "[^2]:" in out
    assert "[^3]:" in out


# ---------------------------------------------------------------------------
# --json flag
# ---------------------------------------------------------------------------

def test_read_json_flag_emits_skill_json(seeded_project, capsys):
    """--json emits the read_finding skill JSON (same shape as the skill script)."""
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim[^chunk:{cited[0]}]."
        finding_id, _ = seed_finding(
            conn, session_id, title="JSON Test", description="d", body=body,
            cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    finding_cmd.read(finding_id=finding_id, project=project, json_out=True, render=False)
    out_json = json.loads(capsys.readouterr().out)

    assert out_json["title"] == "JSON Test"
    assert out_json["finding_id"] == f"finding:{finding_id}"
    assert "citations" in out_json
    assert "dangling_citations" in out_json
    assert "external_citations" in out_json


# ---------------------------------------------------------------------------
# --render flag
# ---------------------------------------------------------------------------

def test_read_render_flag_does_not_crash(seeded_project, capsys):
    """--render pretty-prints via rich without raising."""
    project = seeded_project["project"]
    session_id = _active_session_id(project)
    conn = open_db(project)
    try:
        cited = _doc_chunk_ids(conn, seeded_project["doc_a"])
        body = f"Claim[^chunk:{cited[0]}]."
        finding_id, _ = seed_finding(
            conn, session_id, title="Render Test", description="d", body=body,
            cited_chunk_ids=cited[:1],
        )
    finally:
        conn.close()

    # Should not raise; rich may or may not write to captured stdout in test
    # environment, so we just assert no exception.
    finding_cmd.read(finding_id=finding_id, project=project, json_out=False, render=True)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_read_missing_finding_exits_1(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        finding_cmd.read(finding_id=999999, project=seeded_project["project"],
                         json_out=False, render=False)
    assert exc.value.code == 1


def test_read_no_active_project_exits_1(capsys, project_env):
    """No project and no active project fails cleanly."""
    with pytest.raises(SystemExit) as exc:
        finding_cmd.read(finding_id=1, project=None, json_out=False, render=False)
    assert exc.value.code == 1
