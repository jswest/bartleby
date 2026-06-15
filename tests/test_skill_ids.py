"""Type-tagged entity ids — the #624 interface layer.

Covers the dedicated ``_ids`` helpers and the cross-cutting behaviors the hard
cutover introduces: prefixed output shape, prefixed-only input flags (bare ints
and wrong-typed ids both rejected), and the type-tagged ``[^chunk:N]`` citation
marker (with bare ``[^N]`` and wrong-type ``[^document:N]`` / ``[^finding:N]``
rejected). Storage stays integer — these are all interface-layer assertions.
"""

from __future__ import annotations

import json

import pytest

from bartleby.db.connection import open_db
from bartleby.skill_scripts import (
    read_chunks,
    read_finding,
    save_finding,
    search,
)
from bartleby.skill_scripts import _ids
from tests._skill_fixtures import (  # noqa: F401
    mock_embed,
    project_env,
    seeded_project,
)


# ---- _ids unit tests ------------------------------------------------------

def test_format_id_and_none_passthrough():
    assert _ids.format_id("chunk", 15837) == "chunk:15837"
    assert _ids.format_id("document", 204) == "document:204"
    assert _ids.format_id("chunk", None) is None


def test_parse_id_accepts_prefixed():
    assert _ids.parse_id("chunk:15", "chunk") == 15
    assert _ids.parse_id("document:204", "document") == 204


def test_parse_id_rejects_bare_int():
    with pytest.raises(Exception) as exc:
        _ids.parse_id("15", "chunk")
    assert "bare ids are no longer accepted" in str(exc.value)


def test_parse_id_rejects_wrong_type():
    with pytest.raises(Exception) as exc:
        _ids.parse_id("document:15", "chunk")
    assert "document id" in str(exc.value) and "chunk id" in str(exc.value)


def test_parse_id_rejects_non_integer_body():
    with pytest.raises(Exception):
        _ids.parse_id("chunk:abc", "chunk")


def test_prefixed_int_list_requires_all_prefixed():
    parse = _ids.prefixed_int_list("document")
    assert parse("document:1,document:2") == [1, 2]
    with pytest.raises(Exception):
        parse("document:1,2")          # one bare element fails the whole list
    with pytest.raises(Exception):
        parse("chunk:1,document:2")    # one wrong-typed element fails


def test_format_output_ids_walks_scalars_lists_and_none():
    out = _ids.format_output_ids({
        "chunk_id": 5,
        "document_id": None,
        "chunk_ids": [1, 2],
        "missing_count": 3,                # not an id field → untouched
        "nested": {"finding_id": 9, "tag_id": 7},
        "rows": [{"chunk_id": 8}],
    })
    assert out == {
        "chunk_id": "chunk:5",
        "document_id": None,
        "chunk_ids": ["chunk:1", "chunk:2"],
        "missing_count": 3,
        "nested": {"finding_id": "finding:9", "tag_id": "tag:7"},
        "rows": [{"chunk_id": "chunk:8"}],
    }


def test_format_source_id_prefixes_by_kind():
    assert _ids.format_source_id("document", 3) == "document:3"
    assert _ids.format_source_id("summary", 4) == "summary:4"
    assert _ids.format_source_id("finding", 5) == "finding:5"
    assert _ids.format_source_id("image", 6) == "image:6"
    assert _ids.format_source_id("document", None) is None


# ---- prefixed-only input flags --------------------------------------------

def _doc_a(seeded_project):
    return seeded_project["doc_a"]


def test_read_chunks_bare_document_id_rejected(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        read_chunks.main([
            "--project", seeded_project["project"],
            "--document-id", str(_doc_a(seeded_project)),   # bare int
        ])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


def test_read_chunks_wrong_type_on_chunks_flag_rejected(seeded_project, capsys):
    # A document:-typed value handed to the chunk-only --chunks flag must error
    # loudly, not look up the colliding row.
    with pytest.raises(SystemExit) as exc:
        read_chunks.main([
            "--project", seeded_project["project"],
            "--chunks", f"document:{_doc_a(seeded_project)}",
        ])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


def test_search_bare_in_documents_rejected(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        search.main([
            "--project", seeded_project["project"], "alpha",
            "--in-documents", str(_doc_a(seeded_project)),
        ])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


def test_read_finding_wrong_type_on_finding_id_rejected(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        read_finding.main([
            "--project", seeded_project["project"],
            "--finding-id", "chunk:1",      # wrong type for --finding-id
        ])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)["code"] == "USAGE_ERROR"


# ---- prefixed output shape ------------------------------------------------

def test_read_chunks_output_ids_are_prefixed(seeded_project, capsys):
    conn = open_db(seeded_project["project"])
    try:
        cids = [r[0] for r in conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT 2",
            (_doc_a(seeded_project),),
        )]
    finally:
        conn.close()

    read_chunks.main([
        "--project", seeded_project["project"],
        "--chunks", f"chunk:{cids[0]},chunk:{cids[1]}",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["requested"] == [f"chunk:{cids[0]}", f"chunk:{cids[1]}"]
    assert out["missing"] == []
    for ch in out["chunks"]:
        assert ch["chunk_id"].startswith("chunk:")
        # source_id is a document-kind chunk → prefixed by kind.
        assert ch["source_id"] == f"document:{_doc_a(seeded_project)}"


# ---- type-tagged citation markers -----------------------------------------

def _cited_ids(seeded_project, n=1):
    conn = open_db(seeded_project["project"])
    try:
        return [r[0] for r in conn.cursor().execute(
            "SELECT chunk_id FROM chunks WHERE source_kind='document' "
            "AND source_id = ? ORDER BY chunk_index LIMIT ?",
            (seeded_project["doc_a"], n),
        )]
    finally:
        conn.close()


def _save(seeded_project, tmp_path, body):
    body_file = tmp_path / "f.md"
    body_file.write_text(body, encoding="utf-8")
    save_finding.main([
        "--project", seeded_project["project"],
        "--title", "t", "--description", "d",
        "--body-file", str(body_file),
    ])


def test_chunk_marker_accepted_and_stored(seeded_project, tmp_path, capsys):
    (cid,) = _cited_ids(seeded_project, 1)
    _save(seeded_project, tmp_path, f"Claim[^chunk:{cid}].")
    out = json.loads(capsys.readouterr().out)
    assert out["finding_id"].startswith("finding:")
    assert [c["chunk_id"] for c in out["citations"]] == [f"chunk:{cid}"]

    # Storage stays integer: the finding_citations row is a bare int.
    finding_id = int(out["finding_id"].split(":")[1])
    conn = open_db(seeded_project["project"])
    try:
        stored = [r[0] for r in conn.cursor().execute(
            "SELECT chunk_id FROM finding_citations WHERE finding_id = ?",
            (finding_id,),
        )]
    finally:
        conn.close()
    assert stored == [cid]


def test_bare_caret_marker_rejected(seeded_project, tmp_path, capsys):
    (cid,) = _cited_ids(seeded_project, 1)
    with pytest.raises(SystemExit):
        _save(seeded_project, tmp_path, f"Claim[^{cid}].")   # bare [^N]
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MALFORMED_CITATION"


def test_document_typed_marker_rejected(seeded_project, tmp_path, capsys):
    (cid,) = _cited_ids(seeded_project, 1)
    # A real chunk marker plus a wrong-type [^document:N] marker.
    with pytest.raises(SystemExit):
        _save(seeded_project, tmp_path,
              f"Good[^chunk:{cid}]. Bad[^document:{cid}].")
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "WRONG_CITATION_TYPE"
    assert any("document" in m for m in out["wrong_type_markers"])


def test_finding_typed_marker_rejected(seeded_project, tmp_path, capsys):
    (cid,) = _cited_ids(seeded_project, 1)
    with pytest.raises(SystemExit):
        _save(seeded_project, tmp_path,
              f"Good[^chunk:{cid}]. Bad[^finding:1].")
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "WRONG_CITATION_TYPE"


def test_external_markers_still_ride_alongside_chunk(seeded_project, tmp_path, capsys):
    (cid,) = _cited_ids(seeded_project, 1)
    _save(seeded_project, tmp_path,
          f"Corpus[^chunk:{cid}]. Web[^url:https://example.gov/r]. "
          "Dataset[^doc:H-2024-0148].")
    out = json.loads(capsys.readouterr().out)
    schemes = {e["scheme"] for e in out["external_citations"]}
    assert schemes == {"url", "doc"}
    assert [c["chunk_id"] for c in out["citations"]] == [f"chunk:{cid}"]
