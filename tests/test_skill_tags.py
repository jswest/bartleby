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
from bartleby import skill_runner
from bartleby.skill_runner import SkillError
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
    # find_similar_tag imports embed_texts lazily from its source module
    # (#371), so patch it there rather than on _tags.
    monkeypatch.setattr("bartleby.ingest.embed.embed_texts", _stub)


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
    assert out.pop("run")["session_id"]  # every result echoes the run (#547)
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
        "value_type": None, "pattern": None,
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


def test_add_tag_rejects_whitespace_name(seeded_project, capsys):
    # A whitespace-only name has no alphanumeric content; it falls through to
    # the normalize_name guard and reports EMPTY_NORMALIZED_NAME.
    with pytest.raises(SystemExit):
        add_tag.main([
            "--project", seeded_project["project"],
            "--name", "   ", "--description", "a real description",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_NORMALIZED_NAME"


def test_add_tag_rejects_punctuation_only_name(seeded_project, capsys):
    # A punctuation-only name normalizes to "" — same EMPTY_NORMALIZED_NAME
    # envelope. (Leading-dash values like "---" are eaten by argparse before
    # work() runs, so use non-flag punctuation to exercise the normalize guard.)
    with pytest.raises(SystemExit):
        add_tag.main([
            "--project", seeded_project["project"],
            "--name", "%%%", "--description", "a real description",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_NORMALIZED_NAME"


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
        "--project", seeded_project["project"], "--tag", "ch",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "deleted"
    assert out["removed_assignments"] == 1


def test_delete_tag_unknown_tag(seeded_project, capsys):
    with pytest.raises(SystemExit) as exc:
        delete_tag.main([
            "--project", seeded_project["project"], "--tag", "nonexistent",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TAG_NOT_FOUND"


def test_rename_tag_renames_and_preserves_assignment(seeded_project, capsys):
    # The happy path: a fresh-named target renames in place. The tag_id and the
    # document_tags row both survive (rename only touches tags.name), and the
    # envelope reports the new name alongside the preserved old one.
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('ch', 'd1')")
        tag_id = conn.last_insert_rowid()
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            (seeded_project["doc_a"], tag_id),
        )
    finally:
        conn.close()

    rename_tag.main([
        "--project", seeded_project["project"],
        "--old", "ch", "--new", "central-hudson",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out.pop("run")["session_id"]  # every result echoes the run (#547)
    assert out == {
        "status": "renamed",
        "tag_id": tag_id,
        "old_name": "ch",
        "new_name": "central-hudson",
    }

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # The UPDATE landed under the same tag_id — no new row created.
        assert cur.execute(
            "SELECT name FROM tags WHERE tag_id = ?", (tag_id,)
        ).fetchone()[0] == "central-hudson"
        assert cur.execute(
            "SELECT COUNT(*) FROM tags WHERE name = 'ch'"
        ).fetchone()[0] == 0
        # The assignment rides along untouched — same tag_id, same document.
        assert cur.execute(
            "SELECT COUNT(*) FROM document_tags "
            "WHERE document_id = ? AND tag_id = ?",
            (seeded_project["doc_a"], tag_id),
        ).fetchone()[0] == 1
    finally:
        conn.close()


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


def test_rename_tag_refuses_normalized_equal_target(seeded_project, capsys):
    # add_tag catches "NYSEG" ≡ "ny-seg" via its normalized leg; rename_tag
    # must too — renaming "ny-seg" onto a normalized-equal "NYSEG" is refused,
    # not silently duplicated.
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('ny-seg', 'd1')")
        cur.execute("INSERT INTO tags (name, description) VALUES ('NYSEG', 'd2')")
    finally:
        conn.close()

    with pytest.raises(SystemExit):
        rename_tag.main([
            "--project", seeded_project["project"],
            "--old", "ny-seg", "--new", "NYSEG",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TAG_EXISTS"


def test_rename_tag_allows_case_punctuation_self_rename(seeded_project, capsys):
    # A normalized-equal self-rename (same tag_id) must pass the collision
    # guard — the tag_id != target check lets "ny-seg" → "NYSEG" through.
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO tags (name, description) VALUES ('ny-seg', 'd1')"
        )
    finally:
        conn.close()

    rename_tag.main([
        "--project", seeded_project["project"],
        "--old", "ny-seg", "--new", "NYSEG",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "renamed"
    assert out["new_name"] == "NYSEG"

    conn = open_db(seeded_project["project"])
    try:
        assert conn.cursor().execute(
            "SELECT name FROM tags WHERE name = 'NYSEG'"
        ).fetchone()[0] == "NYSEG"
    finally:
        conn.close()


def test_rename_tag_rejects_whitespace_new_name(seeded_project, capsys):
    # A whitespace-only --new normalizes to "" and is refused via the
    # normalize_name guard's EMPTY_NORMALIZED_NAME envelope.
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO tags (name, description) VALUES ('a', 'd1')"
        )
    finally:
        conn.close()

    with pytest.raises(SystemExit):
        rename_tag.main([
            "--project", seeded_project["project"],
            "--old", "a", "--new", "   ",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "EMPTY_NORMALIZED_NAME"


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
        "--project", seeded_project["project"], "--from", "a", "--into", "b",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "merged"
    assert out["from"]["name"] == "a"
    assert out["into"]["name"] == "b"
    # Partial overlap: doc_a is new on b, doc_b already on b.
    assert out["inserted"] == 1
    assert out["already_present"] == 1

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


def test_merge_tags_disjoint_reports_all_inserted(seeded_project, capsys):
    """No overlap: every source assignment is newly inserted onto the dest."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute("INSERT INTO tags (name, description) VALUES ('b', 'd2')")
        # a tags doc_a and doc_b; b tags nothing → no collision on merge.
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?), (?, ?)",
            (seeded_project["doc_a"], a_id, seeded_project["doc_b"], a_id),
        )
    finally:
        conn.close()

    merge_tags.main([
        "--project", seeded_project["project"], "--from", "a", "--into", "b",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["inserted"] == 2
    assert out["already_present"] == 0


def test_merge_tags_full_overlap_reports_none_inserted(seeded_project, capsys):
    """Full overlap: dest already carries every source doc → nothing inserted."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute("INSERT INTO tags (name, description) VALUES ('b', 'd2')")
        b_id = conn.last_insert_rowid()
        # Both tags cover doc_a and doc_b — a's assignments are all duplicates.
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) "
            "VALUES (?, ?), (?, ?), (?, ?), (?, ?)",
            (seeded_project["doc_a"], a_id, seeded_project["doc_b"], a_id,
             seeded_project["doc_a"], b_id, seeded_project["doc_b"], b_id),
        )
    finally:
        conn.close()

    merge_tags.main([
        "--project", seeded_project["project"], "--from", "a", "--into", "b",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["inserted"] == 0
    assert out["already_present"] == 2


def test_merge_tags_self_merge_rejected(seeded_project, capsys):
    # --from == --into is refused before any tag lookup or mutation.
    with pytest.raises(SystemExit) as exc:
        merge_tags.main([
            "--project", seeded_project["project"],
            "--from", "a", "--into", "a",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "SELF_MERGE"


def test_merge_tags_unknown_source_tag(seeded_project, capsys):
    # The destination exists but the source doesn't → TAG_NOT_FOUND.
    _seed_tag(seeded_project["project"], name="dst", description="d")
    with pytest.raises(SystemExit) as exc:
        merge_tags.main([
            "--project", seeded_project["project"],
            "--from", "nope", "--into", "dst",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "TAG_NOT_FOUND"


def _fail_on_sql(monkeypatch, needle: str) -> None:
    """Make every connection the runner opens raise on the first SQL statement
    whose text contains ``needle`` (issue #340 atomicity injection).

    Installs an apsw exec-trace on the connection ``skill_runner`` opens, so the
    failure fires *inside* ``work()`` — under the runner's transaction wrap —
    rather than before it. A raising exec-trace aborts that one statement and
    propagates, exercising rollback of whatever the same ``work()`` already
    wrote."""
    real_open_db = skill_runner.open_db

    def _patched(project):
        conn = real_open_db(project)

        def _trace(cursor, sql, bindings):
            if needle in sql:
                raise RuntimeError(f"injected failure on: {needle}")
            return True

        conn.set_exec_trace(_trace)
        return conn

    monkeypatch.setattr(skill_runner, "open_db", _patched)


def test_merge_tags_failure_mid_write_leaves_both_tags_untouched(
    seeded_project, capsys, monkeypatch
):
    """A failure between copying assignments and deleting the source tag rolls
    the whole merge back: no assignments copied onto the destination, and the
    source tag (with its assignments) survives (issue #340). Without the
    transaction wrap the copy would commit and the source tag would linger with
    its assignments already duplicated onto the destination."""
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tags (name, description) VALUES ('a', 'd1')")
        a_id = conn.last_insert_rowid()
        cur.execute("INSERT INTO tags (name, description) VALUES ('b', 'd2')")
        b_id = conn.last_insert_rowid()
        # 'a' tags doc_a + doc_b; 'b' tags nothing. A clean merge would copy two.
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?), (?, ?)",
            (seeded_project["doc_a"], a_id, seeded_project["doc_b"], a_id),
        )
    finally:
        conn.close()

    # Fail the source-tag DELETE — the second write, after the copy INSERT.
    _fail_on_sql(monkeypatch, "DELETE FROM tags")

    with pytest.raises(SystemExit) as exc:
        merge_tags.main([
            "--project", seeded_project["project"], "--from", "a", "--into", "b",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Source tag survives with its two assignments — the copy rolled back.
        assert cur.execute(
            "SELECT COUNT(*) FROM tags WHERE name = 'a'"
        ).fetchone()[0] == 1
        assert cur.execute(
            "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (a_id,)
        ).fetchone()[0] == 2
        # Destination carries nothing — no assignments were copied.
        assert cur.execute(
            "SELECT COUNT(*) FROM document_tags WHERE tag_id = ?", (b_id,)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_assign_tag_failure_mid_batch_assigns_nothing(
    seeded_project, capsys, monkeypatch
):
    """A failure partway through a multi-document assign rolls back the whole
    batch — neither document ends up tagged (issue #340). The per-document loop
    writes doc_a, then the injected failure fires on doc_b's insert; the runner
    transaction unwinds doc_a too."""
    tag_id = _seed_tag(seeded_project["project"])

    calls = {"n": 0}
    real_assign = tags_helpers.assign

    def _assign(conn, document_id, tag_ids):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure on second document")
        return real_assign(conn, document_id, tag_ids)

    monkeypatch.setattr("bartleby.skill_scripts.assign_tag.assign", _assign)

    with pytest.raises(SystemExit) as exc:
        assign_tag.main([
            "--project", seeded_project["project"],
            "--documents", f"{seeded_project['doc_a']},{seeded_project['doc_b']}",
            "--tag", "bad_ocr",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    # Neither document is tagged — the first write rolled back with the second.
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 0
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_b"], tag_id,
    ) == 0


def test_unassign_tag_failure_mid_batch_removes_nothing(
    seeded_project, capsys, monkeypatch
):
    """A failure partway through a multi-document unassign rolls back the whole
    batch — both assignments survive (issue #340)."""
    tag_id = _seed_tag(seeded_project["project"])
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().executemany(
            "INSERT INTO document_tags (document_id, tag_id) VALUES (?, ?)",
            [(seeded_project["doc_a"], tag_id), (seeded_project["doc_b"], tag_id)],
        )
    finally:
        conn.close()

    calls = {"n": 0}
    real_unassign = tags_helpers.unassign

    def _unassign(conn, document_id, tag_id_):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure on second document")
        return real_unassign(conn, document_id, tag_id_)

    monkeypatch.setattr("bartleby.skill_scripts.unassign_tag.unassign", _unassign)

    with pytest.raises(SystemExit) as exc:
        unassign_tag.main([
            "--project", seeded_project["project"],
            "--documents", f"{seeded_project['doc_a']},{seeded_project['doc_b']}",
            "--tag", "bad_ocr",
        ])
    assert exc.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "INTERNAL_ERROR"

    # Both assignments survive — the first delete rolled back with the second.
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_a"], tag_id,
    ) == 1
    assert _assignment_count(
        seeded_project["project"], seeded_project["doc_b"], tag_id,
    ) == 1


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


def test_resolve_classifier_no_provider_names_config_command(monkeypatch):
    """The NO_PROVIDER error points at `bartleby config`, not the stale
    `bartleby ready` (issue #329)."""
    monkeypatch.setattr(tags_helpers, "load_config", lambda: {})
    with pytest.raises(SkillError) as exc_info:
        tags_helpers.resolve_classifier()
    err = exc_info.value
    assert err.code == "NO_PROVIDER"
    assert "bartleby config" in err.message
    assert "bartleby ready" not in err.message


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
    assert out.pop("run")["session_id"]  # every result echoes the run (#547)
    assert out == {
        "tag_id": tag_id, "tag": "bad_ocr",
        "value": None, "chunk_id": None,
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


# ---------- value-bearing tags ----------


def _seed_value_tag(
    project, name="revenue", value_type="number",
    pattern=r"(?P<value>\d+)", description="revenue figure",
) -> int:
    conn = open_db(project)
    try:
        conn.cursor().execute(
            "INSERT INTO tags (name, description, value_type, pattern) "
            "VALUES (?, ?, ?, ?)",
            (name, description, value_type, pattern),
        )
        return conn.last_insert_rowid()
    finally:
        conn.close()


def _doc_value(project, document_id, tag_id):
    conn = open_db(project)
    try:
        return conn.cursor().execute(
            "SELECT value, chunk_id FROM document_tags "
            "WHERE document_id = ? AND tag_id = ?",
            (document_id, tag_id),
        ).fetchone()
    finally:
        conn.close()


def test_add_tag_creates_value_tag(seeded_project, capsys):
    add_tag.main([
        "--project", seeded_project["project"],
        "--name", "amount", "--description", "a dollar amount",
        "--value-type", "number", "--pattern", r"\$(?P<value>[\d,]+)",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "created"
    assert out["tag"]["value_type"] == "number"
    assert out["tag"]["pattern"] == r"\$(?P<value>[\d,]+)"


def test_read_tags_surfaces_and_scopes_value_tags(seeded_project, capsys):
    _seed_tag(seeded_project["project"], name="boolcat", description="a category")
    _seed_value_tag(seeded_project["project"])

    read_tags.main(["--project", seeded_project["project"]])
    out = json.loads(capsys.readouterr().out)
    by_name = {t["name"]: t for t in out["tags"]}
    assert by_name["boolcat"]["value_type"] is None
    assert by_name["revenue"]["value_type"] == "number"
    assert by_name["revenue"]["pattern"] == r"(?P<value>\d+)"

    read_tags.main(["--project", seeded_project["project"], "--boolean-only"])
    out = json.loads(capsys.readouterr().out)
    names = {t["name"] for t in out["tags"]}
    assert "boolcat" in names and "revenue" not in names


def test_assign_tag_manual_value_casts_and_stores(seeded_project, capsys):
    tag_id = _seed_value_tag(seeded_project["project"])
    assign_tag.main([
        "--project", seeded_project["project"],
        "--documents", str(seeded_project["doc_a"]), "--tag", "revenue",
        "--value", "$1,234",
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == "1234"  # normalized
    assert out["chunk_id"] is None
    assert _doc_value(seeded_project["project"], seeded_project["doc_a"], tag_id) == (
        "1234", None,
    )


def test_assign_tag_value_rejected_on_boolean_tag(seeded_project, capsys):
    _seed_tag(seeded_project["project"])
    with pytest.raises(SystemExit):
        assign_tag.main([
            "--project", seeded_project["project"],
            "--documents", str(seeded_project["doc_a"]), "--tag", "bad_ocr",
            "--value", "x",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "NOT_A_VALUE_TAG"


def test_assign_tag_chunk_requires_value(seeded_project, capsys):
    _seed_value_tag(seeded_project["project"])
    with pytest.raises(SystemExit):
        assign_tag.main([
            "--project", seeded_project["project"],
            "--documents", str(seeded_project["doc_a"]), "--tag", "revenue",
            "--chunk", "1",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "CHUNK_WITHOUT_VALUE"


def test_merge_value_tags_keeps_target_reports_dropped(seeded_project, capsys):
    src = _seed_value_tag(
        seeded_project["project"], name="rev_src", pattern=r"(?P<value>\d+)",
    )
    dst = _seed_value_tag(
        seeded_project["project"], name="rev_dst", pattern=r"(?P<value>\d+)",
    )
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Overlap doc: both carry a distinct value (target kept, source dropped).
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id, value) VALUES (?, ?, ?)",
            (seeded_project["doc_a"], dst, "100"),
        )
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id, value) VALUES (?, ?, ?)",
            (seeded_project["doc_a"], src, "999"),
        )
        # Source-only doc: its value rides along.
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id, value) VALUES (?, ?, ?)",
            (seeded_project["doc_b"], src, "200"),
        )
    finally:
        conn.close()

    merge_tags.main([
        "--project", seeded_project["project"],
        "--from", "rev_src", "--into", "rev_dst",
    ])
    out = json.loads(capsys.readouterr().out)

    assert out["value_collisions"] == [
        {"document_id": seeded_project["doc_a"], "kept": "100", "dropped": "999"},
    ]
    # Target kept its value on the overlap doc.
    assert _doc_value(
        seeded_project["project"], seeded_project["doc_a"], dst,
    ) == ("100", None)
    # Source-only value carried over to the target.
    assert _doc_value(
        seeded_project["project"], seeded_project["doc_b"], dst,
    ) == ("200", None)


def test_merge_value_tags_carries_value_onto_null_destination(
    seeded_project, capsys,
):
    """A plain (value-NULL) destination row absorbs the source's value+anchor.

    Regression: the destination had a value-tag assignment with no extracted
    value while the source held one for the same doc. The old INSERT OR IGNORE
    kept the NULL destination row and then deleted the source — losing the
    value silently. The merge must carry the source value onto the destination.
    """
    src = _seed_value_tag(
        seeded_project["project"], name="rev_src", pattern=r"(?P<value>\d+)",
    )
    dst = _seed_value_tag(
        seeded_project["project"], name="rev_dst", pattern=r"(?P<value>\d+)",
    )
    conn = open_db(seeded_project["project"])
    try:
        cur = conn.cursor()
        # Destination: a plain assignment (value NULL, no chunk anchor).
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id, value) VALUES (?, ?, ?)",
            (seeded_project["doc_a"], dst, None),
        )
        # Source: holds an extracted value + chunk anchor for the same doc.
        cur.execute(
            "INSERT INTO document_tags (document_id, tag_id, value, chunk_id) "
            "VALUES (?, ?, ?, ?)",
            (seeded_project["doc_a"], src, "777", 1),
        )
    finally:
        conn.close()

    merge_tags.main([
        "--project", seeded_project["project"],
        "--from", "rev_src", "--into", "rev_dst",
    ])
    out = json.loads(capsys.readouterr().out)

    # Not a kept/dropped collision — the destination had no competing value.
    assert out["value_collisions"] == []
    # The source's value + anchor were carried onto the destination, not lost.
    assert _doc_value(
        seeded_project["project"], seeded_project["doc_a"], dst,
    ) == ("777", 1)


def test_merge_value_tag_into_boolean_tag_refused(seeded_project, capsys):
    """value-tag → boolean tag is refused before any mutation (MIXED_TYPE_MERGE)."""
    src = _seed_value_tag(
        seeded_project["project"], name="rev_src", pattern=r"(?P<value>\d+)",
    )
    _seed_tag(seeded_project["project"], name="boolcat", description="a category")
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO document_tags (document_id, tag_id, value, chunk_id) "
            "VALUES (?, ?, ?, ?)",
            (seeded_project["doc_a"], src, "500", 1),
        )
    finally:
        conn.close()

    with pytest.raises(SystemExit):
        merge_tags.main([
            "--project", seeded_project["project"],
            "--from", "rev_src", "--into", "boolcat",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MIXED_TYPE_MERGE"
    # Nothing mutated: source tag and its value row both survive.
    assert _doc_value(
        seeded_project["project"], seeded_project["doc_a"], src,
    ) == ("500", 1)
    conn = open_db(seeded_project["project"])
    try:
        assert tags_helpers.get_tag_by_name(conn, "rev_src") is not None
    finally:
        conn.close()


def test_merge_boolean_tag_into_value_tag_refused(seeded_project, capsys):
    """boolean tag → value-tag is refused too (MIXED_TYPE_MERGE)."""
    _seed_tag(seeded_project["project"], name="boolcat", description="a category")
    _seed_value_tag(
        seeded_project["project"], name="rev_dst", pattern=r"(?P<value>\d+)",
    )
    with pytest.raises(SystemExit):
        merge_tags.main([
            "--project", seeded_project["project"],
            "--from", "boolcat", "--into", "rev_dst",
        ])
    out = json.loads(capsys.readouterr().out)
    assert out["code"] == "MIXED_TYPE_MERGE"


def test_list_documents_surfaces_value_chip(seeded_project, capsys):
    tag_id = _seed_value_tag(seeded_project["project"])
    conn = open_db(seeded_project["project"])
    try:
        conn.cursor().execute(
            "INSERT INTO document_tags (document_id, tag_id, value, chunk_id) "
            "VALUES (?, ?, ?, ?)",
            (seeded_project["doc_a"], tag_id, "4242", None),
        )
    finally:
        conn.close()

    list_documents.main([
        "--project", seeded_project["project"], "--tag", "revenue",
    ])
    out = json.loads(capsys.readouterr().out)
    doc = next(d for d in out["documents"] if d["id"] == seeded_project["doc_a"])
    assert doc["tag_values"]["revenue"] == {"value": "4242", "chunk_id": None}
