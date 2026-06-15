"""Tests for the `extract` skill script (value-bearing tag extraction)."""

from __future__ import annotations

import json

import pytest

from bartleby.db.chunks import ChunkInput, insert_document_chunks
from bartleby.db.connection import open_db
from bartleby.skill_scripts import add_tag, extract, read_tags
from tests._skill_fixtures import _emb, project_env  # noqa: F401


# ---------- helpers ----------


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch):
    """add_tag runs a similarity check that embeds; stub it (no model load)."""
    from bartleby.db.schema import EMBEDDING_DIM

    def _stub(texts):
        return [[0.0] * EMBEDDING_DIM for _ in texts]

    # find_similar_tag imports embed_texts lazily from its source module (#371).
    monkeypatch.setattr("bartleby.ingest.embed.embed_texts", _stub)


def _doc(conn, file_hash, file_name) -> int:
    conn.cursor().execute(
        "INSERT INTO documents (file_hash, file_name, file_path) VALUES (?, ?, ?)",
        (file_hash, file_name, f"/tmp/{file_name}"),
    )
    return conn.last_insert_rowid()


def _chunks(conn, document_id, texts) -> list[int]:
    base = conn.cursor().execute(
        "SELECT COALESCE(MAX(chunk_index) + 1, 0) FROM chunks "
        "WHERE source_kind = 'document' AND source_id = ?",
        (document_id,),
    ).fetchone()[0]
    return insert_document_chunks(conn, document_id, [
        ChunkInput(text=t, embedding=_emb(0.1 * i), chunk_index=base + i)
        for i, t in enumerate(texts)
    ])


@pytest.fixture
def corpus(project_env):
    """Two documents with revenue-ish chunk text and a value-tag created via add_tag."""
    conn = open_db(project_env)
    try:
        doc_a = _doc(conn, "ha", "a.pdf")
        doc_b = _doc(conn, "hb", "b.pdf")
        a_chunks = _chunks(conn, doc_a, [
            "Intro with no number here.",
            "Total revenue was $1,234,567 for the year.",
            "Costs were ($5,000) net of adjustments.",
        ])
        b_chunks = _chunks(conn, doc_b, [
            "Revenue: 42 units reported.",
            "Nothing numeric in this passage.",
        ])
    finally:
        conn.close()

    add_tag.main([
        "--project", project_env, "--name", "revenue",
        "--description", "Reported revenue figure per document.",
        "--value-type", "number",
        "--pattern", r"(?i)revenue:?\s*(?:was\s+)?\$?(?P<value>[\d,]+)",
    ])

    return {
        "project": project_env,
        "doc_a": doc_a, "doc_b": doc_b,
        "a_chunks": a_chunks, "b_chunks": b_chunks,
    }


def _stored_value(project, document_id, tag_name="revenue"):
    conn = open_db(project)
    try:
        row = conn.cursor().execute(
            "SELECT dt.value, dt.chunk_id FROM document_tags dt "
            "JOIN tags t USING (tag_id) "
            "WHERE dt.document_id = ? AND t.name = ?",
            (document_id, tag_name),
        ).fetchone()
    finally:
        conn.close()
    return row


# ---------- add_tag value-tag creation ----------


def test_add_tag_creates_value_tag(corpus, capsys):
    capsys.readouterr()
    read_tags.main(["--project", corpus["project"]])
    out = json.loads(capsys.readouterr().out)
    tag = next(t for t in out["tags"] if t["name"] == "revenue")
    assert tag["value_type"] == "number"
    assert "revenue" in tag["pattern"]


def test_add_tag_rejects_pattern_without_value_group(project_env, capsys):
    with pytest.raises(SystemExit):
        add_tag.main([
            "--project", project_env, "--name", "x", "--description", "no group",
            "--value-type", "string", "--pattern", r"\d+",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INVALID_PATTERN"


def test_add_tag_rejects_value_type_without_pattern(project_env, capsys):
    with pytest.raises(SystemExit):
        add_tag.main([
            "--project", project_env, "--name", "x", "--description", "lonely type",
            "--value-type", "number",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INCOMPLETE_VALUE_TAG"


def test_add_tag_rejects_pattern_without_value_type(project_env, capsys):
    with pytest.raises(SystemExit):
        add_tag.main([
            "--project", project_env, "--name", "x", "--description", "lonely pat",
            "--pattern", r"(?P<value>\d+)",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INCOMPLETE_VALUE_TAG"


# ---------- extract ----------


def test_extract_stores_normalized_values(corpus, capsys):
    extract.main([
        "--project", corpus["project"], "--tag", "revenue",
        "--chunks", ",".join(f"chunk:{c}" for c in corpus["a_chunks"] + corpus["b_chunks"]),
    ])
    out = json.loads(capsys.readouterr().out)

    stored = {s["document_id"]: s for s in out["stored"]}
    # doc_a: "$1,234,567" → stripped of $ and commas.
    assert stored[f"document:{corpus['doc_a']}"]["value"] == "1234567"
    assert stored[f"document:{corpus['doc_a']}"]["chunk_id"] == f"chunk:{corpus['a_chunks'][1]}"
    # doc_b: "42".
    assert stored[f"document:{corpus['doc_b']}"]["value"] == "42"

    # Non-matching chunks land in no_match, never fabricated.
    assert f"chunk:{corpus['a_chunks'][0]}" in out["no_match"]
    assert f"chunk:{corpus['b_chunks'][1]}" in out["no_match"]

    # Persisted to document_tags.value + chunk_id.
    assert _stored_value(corpus["project"], corpus["doc_a"]) == (
        "1234567", corpus["a_chunks"][1],
    )


def test_extract_conflict_stores_neither(corpus, capsys):
    """Two chunks of one doc yielding distinct values → conflict, store neither."""
    conn = open_db(corpus["project"])
    try:
        extra = _chunks(conn, corpus["doc_a"], [
            "Later note: revenue was $999 in a footnote.",
        ])
    finally:
        conn.close()

    extract.main([
        "--project", corpus["project"], "--tag", "revenue",
        "--chunks", f"chunk:{corpus['a_chunks'][1]},chunk:{extra[0]}",
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["stored"] == []
    assert len(out["conflicts"]) == 1
    conflict = out["conflicts"][0]
    assert conflict["document_id"] == f"document:{corpus['doc_a']}"
    assert {v["value"] for v in conflict["values"]} == {"1234567", "999"}
    # Nothing persisted.
    assert _stored_value(corpus["project"], corpus["doc_a"]) is None


def test_extract_same_value_twice_is_not_a_conflict(corpus, capsys):
    conn = open_db(corpus["project"])
    try:
        dup = _chunks(conn, corpus["doc_a"], [
            "Restated: revenue was $1,234,567 confirmed.",
        ])
    finally:
        conn.close()

    extract.main([
        "--project", corpus["project"], "--tag", "revenue",
        "--chunks", f"chunk:{corpus['a_chunks'][1]},chunk:{dup[0]}",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["conflicts"] == []
    assert len(out["stored"]) == 1
    # First matching chunk is the stored anchor.
    assert out["stored"][0]["chunk_id"] == f"chunk:{corpus['a_chunks'][1]}"


def test_extract_reports_missing_chunk_ids(corpus, capsys):
    extract.main([
        "--project", corpus["project"], "--tag", "revenue",
        "--chunks", f"chunk:{corpus['a_chunks'][1]},chunk:999999",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["missing"] == ["chunk:999999"]
    assert len(out["stored"]) == 1


def test_extract_rejects_boolean_tag(corpus, capsys):
    add_tag.main([
        "--project", corpus["project"], "--name", "plain",
        "--description", "an ordinary boolean tag",
    ])
    capsys.readouterr()
    with pytest.raises(SystemExit):
        extract.main([
            "--project", corpus["project"], "--tag", "plain",
            "--chunks", f"chunk:{corpus['a_chunks'][1]}",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NOT_A_VALUE_TAG"


def test_extract_rerun_overwrites_value(corpus, capsys):
    extract.main([
        "--project", corpus["project"], "--tag", "revenue",
        "--chunks", f"chunk:{corpus['a_chunks'][1]}",
    ])
    capsys.readouterr()
    # New chunk with a different revenue figure; re-running over just it
    # overwrites the stored value (one value per (tag, document)).
    conn = open_db(corpus["project"])
    try:
        new = _chunks(conn, corpus["doc_a"], ["revenue was $77 now."])
    finally:
        conn.close()
    extract.main([
        "--project", corpus["project"], "--tag", "revenue",
        "--chunks", f"chunk:{new[0]}",
    ])
    json.loads(capsys.readouterr().out)
    assert _stored_value(corpus["project"], corpus["doc_a"]) == ("77", new[0])


def test_extract_cast_error_skips_and_reports(project_env, capsys):
    """A 'number' tag capturing non-numeric text → cast_errors, not stored."""
    conn = open_db(project_env)
    try:
        doc = _doc(conn, "h", "n.pdf")
        chunks = _chunks(conn, doc, ["code is ABC in the ledger"])
    finally:
        conn.close()
    add_tag.main([
        "--project", project_env, "--name", "amount",
        "--description", "a numeric amount",
        "--value-type", "number", "--pattern", r"code is (?P<value>\w+)",
    ])
    capsys.readouterr()
    extract.main([
        "--project", project_env, "--tag", "amount",
        "--chunks", f"chunk:{chunks[0]}",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["stored"] == []
    assert len(out["cast_errors"]) == 1
    assert out["cast_errors"][0]["captured"] == "ABC"
