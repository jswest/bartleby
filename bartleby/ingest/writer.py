"""The single-writer drain: one owner for every ingest write.

A :class:`Writer` holds a project's WAL connection and is the *only* path that
persists a parsed document, its image captions, and its summary. Each unit
commits in its own transaction, so a failure in one (an expensive VLM caption,
say) never rolls back another (the parse that produced the text chunks before
it). That per-unit atomicity is what makes ingest restartable: a run that dies
mid-captioning leaves the parse durable, and the next run resumes only the
*missing* units — see :mod:`bartleby.commands.scribe`, which reads this Writer's
state queries to compute what's left.

All chunk / FTS5 / sqlite-vec writes go through the typed ``bartleby.db.chunks``
helpers — never raw INSERTs — keeping the polymorphic-chunks invariant at one
chokepoint.

The parse pool (#165) parses many documents at once across worker *processes*,
but every one of them only parses — none touch the database. Each parsed result
returns to the main process, which drains them through this Writer one at a
time. So the Writer stays the connection's single owner on a single thread
(apsw connections aren't thread-safe), and the concurrency lives entirely
upstream in parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import apsw

from bartleby import __version__
from bartleby.db.chunks import (
    ChunkInput,
    delete_chunks_for,
    insert_document_chunks,
    insert_image_chunks,
    insert_summary_chunks,
)
from bartleby.db.schema import SCHEMA_VERSION
from bartleby.ingest.summarize import SummaryResult


# A unit that fails this many times stops being retried — it's recorded in
# ``failed_ingests`` and surfaced, not attempted again, so a deterministically
# broken doc/image can't loop forever (and can't silently read as "done").
MAX_INGEST_ATTEMPTS = 3


# -------------------- producer → writer payloads --------------------


@dataclass
class ParsedImage:
    """An image extracted, scaled, and archived during parse — not yet captioned.

    Pure metadata: the prepared JPEG lives on disk at ``archive_path``, and the
    caption stage reloads it there (and OCRs it) when it runs. The parse result
    therefore crosses the parse-pool queue cheaply — no image bytes ride along.
    """
    hash: str
    archive_path: Path
    width: int
    height: int
    page_number: int | None
    image_index_on_page: int


@dataclass
class ParsedDocument:
    """The product of parsing one source file — one atomic write unit."""
    file_hash: str
    file_name: str
    archive_path: Path
    page_count: int | None
    token_count: int
    document_chunks: list[ChunkInput]
    images: list[ParsedImage]


@dataclass
class ImageCaption:
    """The product of captioning one image row — one atomic write unit."""
    image_id: int
    analysis_json: str
    analysis_model: str
    chunks: list[ChunkInput]


# -------------------- state-query results --------------------


@dataclass
class PendingImage:
    """An uncaptioned image row linked to a document — a resume work item."""
    image_id: int
    file_hash: str
    file_path: str
    width: int
    height: int
    page_number: int | None


@dataclass
class FailedUnit:
    file_hash: str
    file_name: str
    stage: str
    error: str
    attempts: int
    last_attempt: str

    @property
    def capped(self) -> bool:
        return self.attempts >= MAX_INGEST_ATTEMPTS


class Writer:
    """Sole owner of a project's write connection (the single-writer drain)."""

    def __init__(self, conn: apsw.Connection):
        self.conn = conn
        # The open ingest run, set by begin_run(); every persist_* below stamps
        # its document / summary / chunk rows with it. None until a run opens
        # (and for skill-side writes that never call begin_run), which is why
        # ingest_run_id is nullable.
        self.run_id: int | None = None

    # ---- run provenance: stamp each unit with its producing invocation ----

    def begin_run(self, config: dict) -> int:
        """Open an ingest run, recording its resolved config snapshot.

        ``config`` is the run's resolved configuration with secrets already
        stripped (see :func:`bartleby.config.redact_config`) — it is stored
        verbatim, so never hand this method anything containing an API key.
        Serialized with sorted keys so a later run's drift check is a plain
        string compare. Returns (and stores as ``self.run_id``) the new run id.
        """
        config_json = json.dumps(config, sort_keys=True)
        with self.conn:
            self.conn.cursor().execute(
                "INSERT INTO ingests "
                "(config_json, bartleby_version, schema_version) "
                "VALUES (?, ?, ?)",
                (config_json, __version__, SCHEMA_VERSION),
            )
            self.run_id = self.conn.last_insert_rowid()
        return self.run_id

    def finish_run(self) -> None:
        """Mark the open run finished. No-op if no run was opened."""
        if self.run_id is None:
            return
        with self.conn:
            self.conn.cursor().execute(
                "UPDATE ingests SET finished_at = CURRENT_TIMESTAMP "
                "WHERE run_id = ?",
                (self.run_id,),
            )

    def latest_config(self) -> dict | None:
        """The resolved config snapshot of the most recent prior run, or None.

        Call this *before* :meth:`begin_run` so the current run isn't compared
        against itself; the caller diffs it against the new snapshot to warn on
        config drift across a resume/re-run.
        """
        row = self.conn.cursor().execute(
            "SELECT config_json FROM ingests ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        return json.loads(row[0]) if row else None

    # ---- state: resume by what's missing ----

    def document_id_for(self, file_hash: str) -> int | None:
        """The document_id for an already-parsed file, or None.

        Parse commits atomically (:meth:`persist_parse`), so a ``documents``
        row existing means the parse landed in full — there is no half-written
        parse to disambiguate.
        """
        row = self.conn.cursor().execute(
            "SELECT document_id FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return row[0] if row else None

    def uncaptioned_images(self, document_id: int) -> list[PendingImage]:
        """Image rows linked to ``document_id`` whose caption hasn't landed.

        ``analysis_json IS NULL`` is the uncaptioned marker: parse records the
        row with a null analysis, :meth:`persist_caption` fills it. Grouped by
        image so a figure repeated across pages of one document is captioned
        once, not once per appearance.
        """
        rows = self.conn.cursor().execute(
            "SELECT i.image_id, i.file_hash, i.file_path, i.width, i.height, "
            "       MIN(di.page_number) "
            "FROM images i "
            "JOIN document_images di ON di.image_id = i.image_id "
            "WHERE di.document_id = ? AND i.analysis_json IS NULL "
            "GROUP BY i.image_id "
            "ORDER BY i.image_id",
            (document_id,),
        ).fetchall()
        return [PendingImage(*r) for r in rows]

    def summary_exists(self, document_id: int) -> bool:
        return self.conn.cursor().execute(
            "SELECT 1 FROM summaries WHERE document_id = ?", (document_id,)
        ).fetchone() is not None

    def summarizable_chunk_count(self, document_id: int) -> int:
        """Indexed chunks attributable to a document: its own document chunks
        plus image chunks joined through ``document_images``.

        Zero means there's nothing real to summarize (an image-only doc whose
        images carry no OCR/description, say) — the summary stage skips such a
        doc rather than feed the model trace garbage (issue #80), and it counts
        as summary-complete.
        """
        return self.conn.cursor().execute(
            "SELECT "
            "  (SELECT COUNT(*) FROM chunks "
            "   WHERE source_kind = 'document' AND source_id = ?) "
            "+ (SELECT COUNT(*) FROM chunks c "
            "     JOIN document_images di ON di.image_id = c.source_id "
            "   WHERE c.source_kind = 'image' AND di.document_id = ?)",
            (document_id, document_id),
        ).fetchone()[0]

    def summary_input(self, document_id: int) -> str:
        """Interleave document and image chunks in source order for the summarizer.

        Either set may be empty (a text-only doc has no image chunks; an
        image-only doc has no document chunks). We build from whatever indexed
        chunks exist so the summarizer always sees real, indexed content —
        never raw trace text, which makes the model confabulate (issue #80).
        Callers gate on :meth:`summarizable_chunk_count` > 0, so this never
        returns empty in practice.
        """
        cur = self.conn.cursor()
        img_rows = cur.execute(
            "SELECT di.page_number, c.chunk_index, c.text "
            "FROM chunks c "
            "JOIN document_images di ON di.image_id = c.source_id "
            "WHERE c.source_kind = 'image' AND di.document_id = ?",
            (document_id,),
        ).fetchall()
        doc_rows = cur.execute(
            "SELECT page_number, chunk_index, text FROM chunks "
            "WHERE source_kind = 'document' AND source_id = ?",
            (document_id,),
        ).fetchall()

        KIND_DOC, KIND_IMG = 0, 1
        entries = (
            [(p, KIND_DOC, ci, t) for p, ci, t in doc_rows]
            + [(p, KIND_IMG, ci, t) for p, ci, t in img_rows]
        )
        entries.sort(key=lambda r: (r[0] if r[0] is not None else -1, r[1], r[2]))

        parts: list[str] = []
        for page_number, kind, _chunk_index, text in entries:
            if kind == KIND_IMG:
                label = (
                    f"[Image on page {page_number}]"
                    if page_number is not None else "[Image]"
                )
                parts.append(f"{label}\n{text}")
            else:
                parts.append(text)
        return "\n\n".join(parts)

    # ---- writes: one transaction per unit ----

    def persist_parse(self, parsed: ParsedDocument) -> int:
        """Commit a parsed document — row, text chunks, and image rows — atomically.

        Image rows land *uncaptioned* (``analysis_json`` NULL); captioning is a
        separate unit. Image dedup is by byte-hash: an image already recorded
        by another document is reused, only the ``document_images`` join is
        added. Returns the new document_id.
        """
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO documents "
                "(file_hash, file_name, file_path, page_count, token_count, "
                " ingest_run_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (parsed.file_hash, parsed.file_name, str(parsed.archive_path),
                 parsed.page_count, parsed.token_count, self.run_id),
            )
            document_id = self.conn.last_insert_rowid()
            if parsed.document_chunks:
                insert_document_chunks(
                    self.conn, document_id, parsed.document_chunks,
                    ingest_run_id=self.run_id,
                )
            for img in parsed.images:
                existing = cur.execute(
                    "SELECT image_id FROM images WHERE file_hash = ?", (img.hash,)
                ).fetchone()
                if existing is not None:
                    image_id = existing[0]
                else:
                    cur.execute(
                        "INSERT INTO images "
                        "(file_hash, file_path, width, height, "
                        " analysis_json, analysis_model) "
                        "VALUES (?, ?, ?, ?, NULL, NULL)",
                        (img.hash, str(img.archive_path), img.width, img.height),
                    )
                    image_id = self.conn.last_insert_rowid()
                # OR IGNORE: the same (doc, image, page, index) tuple can replay
                # across runs; a no-op is fine rather than a constraint crash.
                cur.execute(
                    "INSERT OR IGNORE INTO document_images "
                    "(document_id, image_id, page_number, image_index_on_page) "
                    "VALUES (?, ?, ?, ?)",
                    (document_id, image_id, img.page_number, img.image_index_on_page),
                )
        return document_id

    def persist_caption(self, caption: ImageCaption) -> None:
        """Fill an image row's caption (analysis + chunks) atomically.

        The ``analysis_json IS NULL`` guard makes this idempotent: a caption
        already applied (a shared image captioned via a sibling document) is a
        no-op rather than a double-write.
        """
        with self.conn:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE images SET analysis_json = ?, analysis_model = ? "
                "WHERE image_id = ? AND analysis_json IS NULL",
                (caption.analysis_json, caption.analysis_model, caption.image_id),
            )
            if caption.chunks:
                insert_image_chunks(
                    self.conn, caption.image_id, caption.chunks,
                    ingest_run_id=self.run_id,
                )

    def persist_summary(
        self,
        document_id: int,
        summary: SummaryResult,
        summary_chunks: list[ChunkInput],
    ) -> int:
        """Replace a document's summary (row + chunks) atomically."""
        with self.conn:
            cur = self.conn.cursor()
            prior = cur.execute(
                "SELECT summary_id FROM summaries WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if prior:
                delete_chunks_for(self.conn, "summary", prior[0])
                cur.execute(
                    "DELETE FROM summaries WHERE summary_id = ?", (prior[0],)
                )
            cur.execute(
                "INSERT INTO summaries "
                "(document_id, title, description, text, model, authored_date, "
                " ingest_run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, summary.title, summary.description,
                 summary.text, summary.model, summary.authored_date,
                 self.run_id),
            )
            summary_id = self.conn.last_insert_rowid()
            if summary_chunks:
                insert_summary_chunks(
                    self.conn, summary_id, summary_chunks,
                    ingest_run_id=self.run_id,
                )
        return summary_id

    # ---- failure tracking: cap retries, stay visible ----

    def attempts(self, file_hash: str, stage: str) -> int:
        """How many times this unit has already failed (0 if never)."""
        row = self.conn.cursor().execute(
            "SELECT attempts FROM failed_ingests "
            "WHERE file_hash = ? AND stage = ?",
            (file_hash, stage),
        ).fetchone()
        return row[0] if row else 0

    def is_capped(self, file_hash: str, stage: str) -> bool:
        """True once a unit has failed MAX_INGEST_ATTEMPTS times — stop retrying."""
        return self.attempts(file_hash, stage) >= MAX_INGEST_ATTEMPTS

    def record_failure(
        self, file_hash: str, file_name: str, stage: str, error: object,
    ) -> None:
        """Record (or bump the attempt count of) a failed unit."""
        with self.conn:
            self.conn.cursor().execute(
                "INSERT INTO failed_ingests "
                "(file_hash, file_name, stage, error, attempts, last_attempt) "
                "VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP) "
                "ON CONFLICT(file_hash, stage) DO UPDATE SET "
                "  file_name = excluded.file_name, "
                "  error = excluded.error, "
                "  attempts = attempts + 1, "
                "  last_attempt = CURRENT_TIMESTAMP",
                (file_hash, file_name, stage, str(error)),
            )

    def clear_failure(self, file_hash: str, stage: str) -> None:
        """Drop a unit's failure record — called when the unit finally succeeds."""
        with self.conn:
            self.conn.cursor().execute(
                "DELETE FROM failed_ingests WHERE file_hash = ? AND stage = ?",
                (file_hash, stage),
            )

    def failures(self) -> list[FailedUnit]:
        """Every still-unresolved failed unit, oldest attempt first."""
        rows = self.conn.cursor().execute(
            "SELECT file_hash, file_name, stage, error, attempts, last_attempt "
            "FROM failed_ingests ORDER BY last_attempt, file_name"
        ).fetchall()
        return [FailedUnit(*r) for r in rows]
