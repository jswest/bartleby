"""Bartleby v1 SQLite schema.

The DDL string here is the canonical schema for v1 (see SPEC.md §3).
Run it via ``init_db`` in ``bartleby.db.connection``.
"""

SCHEMA_VERSION = 2

EMBEDDING_DIM = 768

ALLOWED_SOURCE_KINDS = ("document", "summary", "finding")

DDL = """
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE documents (
    document_id INTEGER PRIMARY KEY,
    file_hash TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    page_count INTEGER,
    token_count INTEGER,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE summaries (
    summary_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL UNIQUE REFERENCES documents(document_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    text TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    session_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    memory_enabled INTEGER NOT NULL DEFAULT 1,
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

CREATE TABLE chunks (
    chunk_id INTEGER PRIMARY KEY,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('document', 'summary', 'finding')),
    source_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    section_heading TEXT,
    content_type TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
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
