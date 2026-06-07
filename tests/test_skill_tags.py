"""Tests for the tag-* skill scripts and tag-filter integration."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from bartleby.db.connection import open_db
from bartleby.db.schema import EMBEDDING_DIM
from bartleby.skill_scripts import (
    add_tag,
    assign_tag,
    delete_tag,
    list_documents,
    merge_tags,
    read_tags,
    rename_tag,
    tag as tag_script,
    unassign_tag,
)
from bartleby.skill_scripts import _tags as tags_helpers
from tests._skill_fixtures import project_env, seeded_project  # noqa: F401


# ---------- fixtures ----------


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch):
    """Make BGE embeddings deterministic without loading the real model.

    We map description text → vector via a tiny hash so that similar inputs
    produce similar vectors (just enough to drive the conflict check).
    """
    def _stub(texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            # Bag-of-words → fingerprint vector. Same words → same vector,
            # which is what the similarity check needs.
            vec = [0.0] * EMBEDDING_DIM
            for word in t.lower().split():
                idx = (sum(ord(c) for c in word)) % EMBEDDING_DIM
                vec[idx] += 1.0
            # L2 normalize so dot product = cosine.
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out
    monkeypatch.setattr(tags_helpers, "embed_texts", _stub)


@pytest.fixture
def stub_classifier(monkeypatch):
    """Replace the real provider with a recorded classification stub.

    The fixture returns a list onto which tests append the classifier's
    intended verdict for the next call. Each call pops the head — order
    matters and a mismatch ("classifier called more times than expected")
    surfaces as IndexError, which is what we want.
    """
    verdicts: list = []

    class _StubProvider:
        name = "stub"

        def classify(self, prompt, *, model, schema, temperature=0.0):
            v = verdicts.pop(0)
            return schema.model_validate(v)

        def summarize(self, *a, **k):  # protocol completeness
            raise NotImplementedError

        def analyze_image(self, *a, **k):
            raise NotImplementedError

    def _resolve():
        return _StubProvider(), "stub-model", 0.0

    monkeypatch.setattr(tags_helpers, "resolve_classifier", _resolve)
    return verdicts


# ---------- read_tags ----------


def test_read_tags_empty(seeded_project, capsys):
    read_tags.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    assert out == {"tags": []}


def test_read_tags_reports_doc_count(seeded_project, capsys):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tags (name, description) VALUES (?, ?)",
            ("ch", "Central Hudson rate-case filings"),
        )
        tag_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
    finally:
        conn.close()

    read_tags.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    assert out["tags"] == [{
        "tag_id": tag_id, "name": "ch",
        "description": "Central Hudson rate-case filings", "doc_count": 1,
    }]


# ---------- add_tag ----------


def test_add_tag_creates(seeded_project, capsys):
    add_tag.main([
        "--project", seeded_project["project"],
        "--name", "ch",
        "--description", "Central Hudson rate-case filings",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "created"
    assert out["tag"]["name"] == "ch"


def test_add_tag_detects_normalized_name_conflict(seeded_project, capsys):
    add_tag.main([
        "--project", seeded_project["project"],
        "--name", "NYSEG",
        "--description", "NYSEG filings and exhibits",
    ])
    capsys.readouterr()
    add_tag.main([
        "--project", seeded_project["project"],
        "--name", "nyseg",
        "--description", "An entirely different description here.",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "conflict"
    assert out["similar_to"]["name"] == "NYSEG"
    assert out["similar_to"]["similarity"] == 1.0


def test_add_tag_detects_similar_description(seeded_project, capsys):
    add_tag.main([
        "--project", seeded_project["project"],
        "--name", "ch",
        "--description", "Central Hudson rate-case filings",
    ])
    capsys.readouterr()
    add_tag.main([
        "--project", seeded_project["project"],
        "--name", "centralhudson",
        "--description", "Central Hudson rate-case filings",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "conflict"
    assert out["similar_to"]["similarity"] >= tags_helpers.SIMILARITY_THRESHOLD


# ---------- delete_tag / rename_tag / merge_tags ----------


def test_delete_tag_cascades_assignments(seeded_project, capsys):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO tags (name, description) VALUES (?, ?)", ("ch", "d"),
        )
        tag_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
    finally:
        conn.close()

    delete_tag.main([
        "--project", seeded_project["project"], "--name", "ch",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "deleted"
    assert out["removed_assignments"] == 1


def test_rename_tag_refuses_existing_target(seeded_project, capsys):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        cur.execute("INSERT INTO tags (name, description) VALUES ('b', 'd2')")
    finally:
        conn.close()

    with pytest.raises(SystemExit):
        rename_tag.main([
            "--project", seeded_project["project"], "--old", "a", "--new", "b",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TAG_EXISTS"


def test_merge_tags_moves_assignments_and_deletes_source(seeded_project, capsys):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute("INSERT INTO tags (name, description) VALUES ('b', 'd2')")
        b_id = conn.last_insert_rowid()
        # doc_a tagged a; doc_b tagged a + b (the latter forces collision on merge)
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?), (?, ?), (?, ?)",
            (seeded_project["doc_a"], a_id,
             seeded_project["doc_b"], a_id,
             seeded_project["doc_b"], b_id),
        )
    finally:
        conn.close()

    merge_tags.main([
        "--project", seeded_project["project"], "--from", "a", "--to", "b",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "merged"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Source gone.
        assert cur.execute(
            "SELECT COUNT(*) FROM tags WHERE name = 'a'"
        ).fetchone()[0] == 0
        # b now has both docs (PK collision absorbed by OR IGNORE).
        assigned = sorted(
            row[0] for row in cur.execute(
                "SELECT document_id FROM document_tags WHERE tag_id = ?",
                (b_id,),
            )
        )
        assert assigned == sorted([
            seeded_project["doc_a"], seeded_project["doc_b"],
        ])
    finally:
        conn.close()


# ---------- tag (classification) ----------


def test_tag_full_vocab_assigns(seeded_project, capsys, stub_classifier):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute("INSERT INTO tags (name, description) VALUES ('b', 'd2')")
        b_id = conn.last_insert_rowid()
    finally:
        conn.close()

    # doc_a has a summary; doc_b doesn't. Only one classifier call expected.
    stub_classifier.append({"tag_ids": [a_id, b_id]})

    tag_script.main(["--project", seeded_project["project"], "--all"])
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "full-vocab"
    assigned = next(c for c in out["classified"]
                    if c["document_id"] == seeded_project["doc_a"])
    assert sorted(assigned["assigned_tag_ids"]) == sorted([a_id, b_id])
    assert any(s["reason"] == "no_summary" for s in out["skipped"])


def test_tag_full_vocab_skips_already_tagged(
    seeded_project, capsys, stub_classifier
):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], a_id),
        )
    finally:
        conn.close()

    # Should not call classifier at all (doc_a already tagged, doc_b has no
    # summary). The empty `verdicts` list will pop-from-empty if anything is
    # called.
    tag_script.main(["--project", seeded_project["project"], "--all"])
    out = json.loads(capsys.readouterr().out)
    assert out["classified"] == []
    reasons = sorted(s["reason"] for s in out["skipped"])
    assert reasons == ["already_tagged", "no_summary"]


def test_tag_single_tag_force_can_unassign(
    seeded_project, capsys, stub_classifier
):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], a_id),
        )
    finally:
        conn.close()

    # Classifier says: no longer applies. With --force, we remove the assignment.
    stub_classifier.append({"applies": False})
    tag_script.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]),
        "--tag", "a", "--force",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["classified"][0]["applies"] is False

    conn = open_db(seeded_project["project"])
    try:
        n = conn.cursor().execute(
            "SELECT COUNT(*) FROM document_tags WHERE document_id = ? AND tag_id = ?",
            (seeded_project["doc_a"], a_id),
        ).fetchone()[0]
        assert n == 0
    finally:
        conn.close()


def test_tag_one_failure_does_not_abort_run(seeded_project, capsys, monkeypatch):
    """A classifier error on one document records it in `failed` and the rest
    of the sweep still runs (issue #36)."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # doc_b also gets a summary so both documents reach the classifier.
        cur.execute(
            "INSERT INTO summaries (document_id, title, description, text, model) "
            "VALUES (?, ?, ?, ?, ?)",
            (seeded_project["doc_b"], "Beta", "Test summary of beta.",
             "A summary of beta.", "test"),
        )
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
    finally:
        conn.close()

    class _RaisingProvider:
        """Raises on the alpha document, succeeds on every other."""
        name = "stub"

        def classify(self, prompt, *, model, schema, temperature=0.0):
            if "alpha" in prompt:
                raise RuntimeError(
                    "Ollama returned an empty response for _TagApplies."
                )
            return schema.model_validate({"applies": True})

        def summarize(self, *a, **k):
            raise NotImplementedError

        def analyze_image(self, *a, **k):
            raise NotImplementedError

    monkeypatch.setattr(
        tags_helpers, "resolve_classifier",
        lambda: (_RaisingProvider(), "stub-model", 0.0),
    )

    tag_script.main([
        "--project", seeded_project["project"], "--all", "--tag", "a",
    ])
    out = json.loads(capsys.readouterr().out)

    classified_ids = [c["document_id"] for c in out["classified"]]
    failed_ids = [f["document_id"] for f in out["failed"]]
    assert seeded_project["doc_b"] in classified_ids  # the good doc still ran
    assert failed_ids == [seeded_project["doc_a"]]     # the bad doc is recorded
    assert "RuntimeError" in out["failed"][0]["error"]


def test_tag_retries_transient_failure_once(seeded_project, capsys, monkeypatch):
    """An empty/transient classifier error is retried once before being
    recorded as failed (issue #36)."""
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO tags (name, description) VALUES ('a', 'd1')"
        )
    finally:
        conn.close()

    calls = {"n": 0}

    class _FlakyProvider:
        """Fails the first call, succeeds on the retry."""
        name = "stub"

        def classify(self, prompt, *, model, schema, temperature=0.0):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError(
                    "Ollama returned an empty response for _TagApplies."
                )
            return schema.model_validate({"applies": True})

        def summarize(self, *a, **k):
            raise NotImplementedError

        def analyze_image(self, *a, **k):
            raise NotImplementedError

    monkeypatch.setattr(
        tags_helpers, "resolve_classifier",
        lambda: (_FlakyProvider(), "stub-model", 0.0),
    )

    tag_script.main([
        "--project", seeded_project["project"],
        "--document", str(seeded_project["doc_a"]), "--tag", "a",
    ])
    out = json.loads(capsys.readouterr().out)

    assert calls["n"] == 2          # one failure, then a successful retry
    assert out["failed"] == []
    assert out["classified"][0]["applies"] is True


def test_tag_rejects_when_no_vocabulary(seeded_project, capsys):
    with pytest.raises(SystemExit):
        tag_script.main([
            "--project", seeded_project["project"], "--all",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] in ("EMPTY_VOCABULARY", "NO_PROVIDER")


# ---------- list_documents --tag ----------


def test_list_documents_tag_filter(seeded_project, capsys):
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('ch', 'd')")
        tag_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
    finally:
        conn.close()

    list_documents.main([
        "--project", seeded_project["project"], "--tag", "ch",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["total"] == 1
    assert out["documents"][0]["id"] == seeded_project["doc_a"]


def test_list_documents_tag_filter_unknown_tag(seeded_project, capsys):
    with pytest.raises(SystemExit):
        list_documents.main([
            "--project", seeded_project["project"], "--tag", "nonexistent",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TAG_NOT_FOUND"


# ---------- search --tag ----------


def test_search_tag_filter_restricts_results(seeded_project, capsys, monkeypatch):
    """A `--tag` filter narrows search to chunks of tagged documents only."""
    import struct
    from bartleby.skill_scripts import search as search_script

    monkeypatch.setattr(
        search_script, "_embed_query",
        lambda q: struct.pack(f"{EMBEDDING_DIM}f", *[0.001] * EMBEDDING_DIM),
    )

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('ch', 'd')")
        tag_id = conn.last_insert_rowid()
        # Tag only doc_a — doc_b's "farewell" chunk must not show up.
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
    finally:
        conn.close()

    search_script.main([
        "--project", seeded_project["project"],
        "--full-text", "alpha", "--tag", "ch",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["filters"]["tags"] == ["ch"]
    assert all(
        r["source_kind"] != "finding" for r in out["results"]
    )  # findings dropped under --tag
    assert all(
        r["source_id"] == seeded_project["doc_a"] or r["source_kind"] != "document"
        for r in out["results"]
    )


# ---------- assign_tag / unassign_tag (manual, no LLM) ----------


def _seed_tag(project, name="bad_ocr", description="d") -> int:
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO tags (name, description) VALUES (?, ?)",
            (name, description),
        )
        return conn.last_insert_rowid()
    finally:
        conn.close()


def _assignment_count(project, document_id, tag_id) -> int:
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT COUNT(*) FROM document_tags "
            "WHERE document_id = ? AND tag_id = ?",
            (document_id, tag_id),
        ).fetchone()[0]
    finally:
        conn.close()


def test_assign_tag_creates_assignment(seeded_project, capsys):
    tag_id = _seed_tag(seeded_project["project"])
    assign_tag.main([
        "--project", seeded_project["project"],
        "--documents", str(seeded_project["doc_a"]), "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out == {
        "tag_id": tag_id, "tag": "bad_ocr",
        "assigned": [
            {"document_id": seeded_project["doc_a"], "file_name": "alpha.pdf"},
        ],
        "not_found": [],
    }
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 1


def test_assign_tag_batch_assigns_all(seeded_project, capsys):
    """One process start tags several documents (the issue #68 case)."""
    tag_id = _seed_tag(seeded_project["project"])
    assign_tag.main([
        "--project", seeded_project["project"],
        "--documents", f"{seeded_project['doc_a']},{seeded_project['doc_b']}",
        "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [a["document_id"] for a in out["assigned"]] == [
        seeded_project["doc_a"], seeded_project["doc_b"],
    ]
    assert out["not_found"] == []
    for doc in ("doc_a", "doc_b"):
        assert _assignment_count(
            seeded_project["project"], seeded_project[doc], tag_id,
        ) == 1


def test_assign_tag_is_idempotent(seeded_project, capsys):
    tag_id = _seed_tag(seeded_project["project"])
    for _ in range(2):
        assign_tag.main([
            "--project", seeded_project["project"],
            "--documents", str(seeded_project["doc_a"]), "--tag", "bad_ocr",
        ])
        out = json.loads(capsys.readouterr().out)
        assert out["assigned"][0]["document_id"] == seeded_project["doc_a"]
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 1


def test_assign_tag_dedups_repeated_ids(seeded_project, capsys):
    tag_id = _seed_tag(seeded_project["project"])
    assign_tag.main([
        "--project", seeded_project["project"],
        "--documents", f"{seeded_project['doc_a']},{seeded_project['doc_a']}",
        "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert len(out["assigned"]) == 1
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 1


def test_assign_tag_reports_not_found_without_aborting(seeded_project, capsys):
    """A bad id lands in `not_found`; the valid ids are still assigned."""
    tag_id = _seed_tag(seeded_project["project"])
    assign_tag.main([
        "--project", seeded_project["project"],
        "--documents", f"{seeded_project['doc_a']},999999", "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [a["document_id"] for a in out["assigned"]] == [seeded_project["doc_a"]]
    assert out["not_found"] == [999999]
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 1


def test_assign_tag_unknown_tag(seeded_project, capsys):
    with pytest.raises(SystemExit):
        assign_tag.main([
            "--project", seeded_project["project"],
            "--documents", str(seeded_project["doc_a"]), "--tag", "nope",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TAG_NOT_FOUND"


def test_unassign_tag_removes_assignment(seeded_project, capsys):
    tag_id = _seed_tag(seeded_project["project"])
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
    finally:
        conn.close()

    unassign_tag.main([
        "--project", seeded_project["project"],
        "--documents", str(seeded_project["doc_a"]), "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [u["document_id"] for u in out["unassigned"]] == [seeded_project["doc_a"]]
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 0


def test_unassign_tag_batch_removes_all(seeded_project, capsys):
    tag_id = _seed_tag(seeded_project["project"])
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().executemany(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            [(seeded_project["doc_a"], tag_id), (seeded_project["doc_b"], tag_id)],
        )
    finally:
        conn.close()

    unassign_tag.main([
        "--project", seeded_project["project"],
        "--documents", f"{seeded_project['doc_a']},{seeded_project['doc_b']}",
        "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert len(out["unassigned"]) == 2
    for doc in ("doc_a", "doc_b"):
        assert _assignment_count(
            seeded_project["project"], seeded_project[doc], tag_id,
        ) == 0


def test_unassign_tag_absent_is_noop(seeded_project, capsys):
    tag_id = _seed_tag(seeded_project["project"])
    # No prior assignment; unassign should succeed and leave the tag intact.
    unassign_tag.main([
        "--project", seeded_project["project"],
        "--documents", str(seeded_project["doc_a"]), "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [u["document_id"] for u in out["unassigned"]] == [seeded_project["doc_a"]]
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 0
    # The tag itself must still exist — unassign does not cascade like delete_tag.
    conn = open_db(seeded_project["project"])
    try:
        assert tags_helpers.get_tag_by_name(conn, "bad_ocr") is not None
    finally:
        conn.close()


def test_unassign_tag_reports_not_found_without_aborting(seeded_project, capsys):
    tag_id = _seed_tag(seeded_project["project"])
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
    finally:
        conn.close()

    unassign_tag.main([
        "--project", seeded_project["project"],
        "--documents", f"{seeded_project['doc_a']},999999", "--tag", "bad_ocr",
    ])
    out = json.loads(capsys.readouterr().out)
    assert [u["document_id"] for u in out["unassigned"]] == [seeded_project["doc_a"]]
    assert out["not_found"] == [999999]
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 0


def test_unassign_tag_unknown_tag(seeded_project, capsys):
    with pytest.raises(SystemExit):
        unassign_tag.main([
            "--project", seeded_project["project"],
            "--documents", str(seeded_project["doc_a"]), "--tag", "nope",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TAG_NOT_FOUND"
