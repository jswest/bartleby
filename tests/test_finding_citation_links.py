"""Tests for [^finding:N] citation markers — issue #654.

Finding-to-finding links: a ``[^finding:<int>]`` marker in a finding body is a
valid unidirectional reference to another finding. No new table; the body text
is the storage. Validated at save/edit time (target must exist); rendered as a
link in the web view. Tests cover:

- pure helpers: extraction, no-false-positive on chunk/external markers
- save_finding accepts ``[^finding:N]`` alongside mandatory chunk citations
- save_finding rejects ``[^finding:N]`` when the referenced finding is missing
- [^finding:N] with a non-integer ref raises MALFORMED_CITATION
- read_finding echoes dangling_finding_links when a linked finding is deleted
- [^document:N] is still rejected (WRONG_CITATION_TYPE is unchanged for that)
- [^finding:N] (formerly rejected as WRONG_CITATION_TYPE) is now accepted
"""

from __future__ import annotations

import json

import pytest

from bartleby.db.connection import open_db
from bartleby.skill_scripts import read_finding, save_finding
from bartleby.skill_scripts._common import (
    SkillError,
    extract_finding_citations,
    reject_malformed_internal_citations,
    reject_wrong_typed_citations,
)
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seed_finding,
    seeded_project,
    unprefix,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_extract_finding_citations_basic():
    body = "See also finding[^finding:7] and finding[^finding:42]."
    assert extract_finding_citations(body) == [7, 42]


def test_extract_finding_citations_dedupes():
    body = "[^finding:3] then [^finding:3] again."
    assert extract_finding_citations(body) == [3]


def test_extract_finding_citations_ignores_chunk_and_external():
    body = "Corpus[^chunk:10]. Web[^url:https://a.test]. Finding[^finding:5]."
    assert extract_finding_citations(body) == [5]


def test_extract_finding_citations_empty_when_none():
    body = "Corpus[^chunk:10]. Web[^url:https://a.test]. No finding links."
    assert extract_finding_citations(body) == []


def test_finding_scheme_not_rejected_by_wrong_type_check():
    """`[^finding:N]` must pass ``reject_wrong_typed_citations`` — it is now valid."""
    # Should not raise
    reject_wrong_typed_citations("Claim[^chunk:1] and link[^finding:7].")


def test_document_scheme_still_rejected_by_wrong_type_check():
    """`[^document:N]` must still raise ``WRONG_CITATION_TYPE``."""
    with pytest.raises(SkillError) as exc:
        reject_wrong_typed_citations("Bad[^document:42].")
    assert exc.value.code == "WRONG_CITATION_TYPE"


def test_malformed_finding_ref_rejected():
    """`[^finding:abc]` (non-integer ref) raises ``MALFORMED_CITATION``."""
    with pytest.raises(SkillError) as exc:
        reject_malformed_internal_citations("Bad[^finding:abc].")
    assert exc.value.code == "MALFORMED_CITATION"


# ---------------------------------------------------------------------------
# Integration: save_finding
# ---------------------------------------------------------------------------

def _first_chunk_id(seeded_project) -> int:
    conn = open_db(seeded_project["project"])
    try:
        return conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 1",
            (seeded_project["doc_a"],),
        ).fetchone()[0]
    finally:
        conn.close()


def _make_existing_finding(seeded_project) -> int:
    """Insert a finding directly and return its finding_id."""
    project = seeded_project["project"]
    conn = open_db(project)
    try:
        from bartleby.session import ensure_active_session
        session_id = ensure_active_session(project)
        finding_id, _ = seed_finding(
            conn, session_id=session_id, title="Linked target",
            description="A finding that other findings can link to",
        )
        return finding_id
    finally:
        conn.close()


def test_save_finding_accepts_finding_link(seeded_project, tmp_path, capsys):
    """``[^finding:N]`` alongside a chunk citation is accepted and the body
    is stored verbatim."""
    chunk_id = _first_chunk_id(seeded_project)
    target_id = _make_existing_finding(seeded_project)

    body_text = (
        f"Based on corpus evidence[^chunk:{chunk_id}], "
        f"see also finding[^finding:{target_id}] for context."
    )
    body_file = tmp_path / "f.md"
    body_file.write_text(body_text, encoding="utf-8")

    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "Finding with link",
        "--description", "A finding that links to another finding.",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)

    assert out.get("code") is None, f"Unexpected error: {out}"
    finding_id = unprefix(out["finding_id"])
    assert finding_id >= 1
    # The chunk citation lands normally.
    assert [c["chunk_id"] for c in out["citations"]] == [f"chunk:{chunk_id}"]
    # The body is stored verbatim, including the [^finding:N] marker.
    assert out["body"] == body_text


def test_save_finding_rejects_unknown_finding_link(seeded_project, tmp_path, capsys):
    """``[^finding:99999]`` referencing a non-existent finding raises
    ``UNKNOWN_FINDING_LINKS``."""
    chunk_id = _first_chunk_id(seeded_project)
    body_file = tmp_path / "f.md"
    body_file.write_text(
        f"Claim[^chunk:{chunk_id}] and bad link[^finding:99999].",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "bad finding link",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "UNKNOWN_FINDING_LINKS"
    assert out["unknown_finding_ids"] == [99999]


def test_save_finding_rejects_malformed_finding_ref(seeded_project, tmp_path, capsys):
    """``[^finding:abc]`` (non-integer ref) raises ``MALFORMED_CITATION``."""
    chunk_id = _first_chunk_id(seeded_project)
    body_file = tmp_path / "f.md"
    body_file.write_text(
        f"Claim[^chunk:{chunk_id}] and broken link[^finding:abc].",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "bad finding ref",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MALFORMED_CITATION"
    assert "[^finding:abc]" in out["malformed_markers"]


def test_save_finding_still_rejects_document_link(seeded_project, tmp_path, capsys):
    """``[^document:N]`` is still rejected as WRONG_CITATION_TYPE (unchanged)."""
    chunk_id = _first_chunk_id(seeded_project)
    body_file = tmp_path / "f.md"
    body_file.write_text(
        f"Claim[^chunk:{chunk_id}] bad doc link[^document:5].",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        save_finding.main([
            "--project", seeded_project["project"],
            "--title", "bad document link",
            "--description", "x",
            "--body-file", str(body_file),
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "WRONG_CITATION_TYPE"


# ---------------------------------------------------------------------------
# Integration: read_finding — dangling_finding_links
# ---------------------------------------------------------------------------

def test_read_finding_dangling_finding_link(seeded_project, tmp_path, capsys):
    """After the referenced finding is deleted, ``dangling_finding_links``
    surfaces the orphaned id in ``read_finding`` output."""
    project = seeded_project["project"]
    chunk_id = _first_chunk_id(seeded_project)

    # Save a finding that links to the target finding.
    target_id = _make_existing_finding(seeded_project)
    body_text = (
        f"Claim[^chunk:{chunk_id}] links to finding[^finding:{target_id}]."
    )
    body_file = tmp_path / "f.md"
    body_file.write_text(body_text, encoding="utf-8")
    save_finding.main([
        "--project", project,
        "--title", "linker",
        "--description", "x",
        "--body-file", str(body_file),
    ])
    out = json.loads(capsys.readouterr().out)
    linker_id = out["finding_id"]  # type-tagged, e.g. "finding:2"

    # Verify the finding link reads cleanly before deletion.
    read_finding.main(["--project", project, "--finding-id", linker_id])
    before = json.loads(capsys.readouterr().out)
    assert before["dangling_finding_links"] == []

    # Delete the target finding and confirm the link is now dangling.
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "DELETE FROM findings WHERE finding_id = ?", (target_id,)
        )
    finally:
        conn.close()

    read_finding.main(["--project", project, "--finding-id", linker_id])
    after = json.loads(capsys.readouterr().out)
    assert after["dangling_finding_links"] == [f"finding:{target_id}"]
    # The body is left verbatim — the marker text is provenance.
    assert after["body"] == body_text


def test_read_finding_no_dangling_finding_links_when_all_resolve(
    seeded_project, tmp_path, capsys
):
    """When all ``[^finding:N]`` targets exist, ``dangling_finding_links`` is
    empty."""
    project = seeded_project["project"]
    chunk_id = _first_chunk_id(seeded_project)
    target_id = _make_existing_finding(seeded_project)

    body_file = tmp_path / "f.md"
    body_file.write_text(
        f"Claim[^chunk:{chunk_id}] see[^finding:{target_id}].", encoding="utf-8"
    )
    save_finding.main([
        "--project", project, "--title", "linker2", "--description", "x",
        "--body-file", str(body_file),
    ])
    linker_id = json.loads(capsys.readouterr().out)["finding_id"]

    read_finding.main(["--project", project, "--finding-id", linker_id])
    out = json.loads(capsys.readouterr().out)
    assert out["dangling_finding_links"] == []
