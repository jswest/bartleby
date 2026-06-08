"""Bartleby SQLite schema.

The DDL string here is the canonical schema; run it via ``init_db`` in
``bartleby.db.connection``. See ``ARCHITECTURE.md`` for the polymorphic-chunks
invariant and the rest of the project's load-bearing rules.
"""

# Held at 8 across the #169 concurrent-ingestion omnibus. The `ingests` table
# and the `ingest_run_id` columns below are additive work landing under that
# omnibus (issue #171); the single v8→v9 bump + a consolidated
# `_upgrade_v8_to_v9` ships once when #169 releases, rather than each sub-issue
# bumping in turn. Until then fresh DBs carry these structures at v8 and an
# existing v8 corpus must re-ingest to gain them.
SCHEMA_VERSION = 8

EMBEDDING_DIM = 768

ALLOWED_SOURCE_KINDS = ("document", "summary", "finding", "image")

DDL = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE ingests (
    run_id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    bartleby_version TEXT,
    schema_version INTEGER NOT NULL
);

CREATE TABLE documents (
    document_id INTEGER PRIMARY KEY,
    file_hash TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    page_count INTEGER,
    token_count INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingest_run_id INTEGER REFERENCES ingests(run_id)
);

CREATE TABLE summaries (
    summary_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(document_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    text TEXT NOT NULL,
    model TEXT NOT NULL,
    authored_date TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingest_run_id INTEGER REFERENCES ingests(run_id)
);

CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    memory_enabled INTEGER NOT NULL DEFAULT 1,
    model TEXT,
    harness TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT
);

CREATE TABLE findings (
    finding_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE images (
    image_id INTEGER PRIMARY KEY,
    file_hash TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    analysis_json TEXT,
    analysis_model TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_images (
    document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    image_id INTEGER NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    page_number INTEGER,
    image_index_on_page INTEGER,
    PRIMARY KEY (document_id, image_id, page_number, image_index_on_page)
);

CREATE INDEX idx_document_images_image ON document_images(image_id);

CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('document', 'summary', 'finding', 'image')),
    source_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    section_heading TEXT,
    page_number INTEGER,
    content_type TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ingest_run_id INTEGER REFERENCES ingests(run_id),
    UNIQUE (source_kind, source_id, chunk_index)
);

CREATE INDEX idx_chunks_source ON chunks(source_kind, source_id);

CREATE TABLE finding_citations (
    finding_id INTEGER NOT NULL REFERENCES findings(finding_id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    PRIMARY KEY (finding_id, chunk_id)
);

CREATE TABLE audit_logs (
    audit_log_id INTEGER PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT,
    result_summary TEXT,
    duration_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_audit_logs_session ON audit_logs(session_id, created_at);

CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_tags (
    document_id INTEGER NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(tag_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, tag_id)
);

CREATE INDEX idx_document_tags_tag ON document_tags(tag_id);

CREATE TABLE failed_ingests (
    file_hash TEXT NOT NULL,
    file_name TEXT NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('parse', 'caption', 'summary')),
    error TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    last_attempt TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (file_hash, stage)
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    text,
    section_heading,
    content='chunks',
    content_rowid='chunk_id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[768]
);
"""
