CREATE TABLE IF NOT EXISTS documents (
  document_id TEXT PRIMARY KEY,
  origin_file_path TEXT NOT NULL,
  pages_count INTEGER
);

CREATE TABLE IF NOT EXISTS pages (
  page_id TEXT PRIMARY KEY,
  body TEXT,
  document_id TEXT NOT NULL REFERENCES documents(document_id),
  page_number INTEGER NOT NULL,
  UNIQUE(document_id, page_number)
);

CREATE TABLE IF NOT EXISTS summaries (
  summary_id TEXT PRIMARY KEY,
  body TEXT,
  page_id TEXT REFERENCES pages(page_id)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  page_id TEXT REFERENCES pages(page_id),
  summary_id TEXT REFERENCES summaries(summary_id),
  CHECK ((page_id IS NOT NULL) <> (summary_id IS NOT NULL))
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
  body,
  chunk_id UNINDEXED,
  content='',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
  chunk_id TEXT,
  embedding float[768]
);

CREATE INDEX IF NOT EXISTS idx_pages_document_ids ON pages(document_id);
CREATE INDEX IF NOT EXISTS idx_summaries_page_ids ON summaries(page_id);

CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks BEGIN
  INSERT INTO fts_chunks(body, chunk_id)
  VALUES (new.body, new.chunk_id);
END;
CREATE INDEX IF NOT EXISTS idx_chunks_page_order ON chunks(page_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_summary_order ON chunks(summary_id, chunk_index);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_page_order
  ON chunks(page_id, chunk_index)
  WHERE page_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_summary_order
  ON chunks(summary_id, chunk_index)
  WHERE summary_id IS NOT NULL;