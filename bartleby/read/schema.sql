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
  document_id TEXT PRIMARY KEY REFERENCES documents(document_id),
  title TEXT NOT NULL,
  subtitle TEXT,
  body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  body TEXT NOT NULL,
  chunk_index INTEGER NOT NULL,
  page_id TEXT NOT NULL REFERENCES pages(page_id),
  section_heading TEXT,
  content_type TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
  body,
  section_heading,
  chunk_id UNINDEXED,
  tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
  chunk_id TEXT,
  embedding float[768]
);

CREATE INDEX IF NOT EXISTS idx_pages_document_ids ON pages(document_id);

CREATE TRIGGER IF NOT EXISTS chunks_after_insert AFTER INSERT ON chunks BEGIN
  INSERT INTO fts_chunks(body, section_heading, chunk_id)
  VALUES (new.body, new.section_heading, new.chunk_id);
END;
CREATE INDEX IF NOT EXISTS idx_chunks_page_order ON chunks(page_id, chunk_index);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_page_order
  ON chunks(page_id, chunk_index);