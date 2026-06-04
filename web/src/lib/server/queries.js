import { getDb } from './db.js';

// A document's assigned tags as a JSON array, ordered by name. Correlates on the
// outer `d.document_id`, so any SELECT using this must alias documents as `d`.
// Empty (the common case — most documents are untagged) yields '[]'.
const TAGS_JSON_SUBQUERY = `
  (SELECT json_group_array(json_object('tag_id', tag_id, 'name', name, 'description', description))
   FROM (SELECT t.tag_id, t.name, t.description
         FROM document_tags dt JOIN tags t USING (tag_id)
         WHERE dt.document_id = d.document_id
         ORDER BY t.name COLLATE NOCASE)) AS tags_json`;

export function listFindings() {
  const { db } = getDb();
  return db.prepare(`
    SELECT f.finding_id, f.title, f.description, f.created_at,
           s.name AS session_name,
           (SELECT COUNT(*) FROM finding_citations fc WHERE fc.finding_id = f.finding_id) AS citation_count
    FROM findings f
    JOIN sessions s USING (session_id)
    ORDER BY f.created_at DESC
  `).all();
}

export function getFinding(findingId) {
  const { db } = getDb();
  const finding = db.prepare(`
    SELECT f.finding_id, f.title, f.description, f.body, f.created_at,
           s.name AS session_name
    FROM findings f
    JOIN sessions s USING (session_id)
    WHERE f.finding_id = ?
  `).get(findingId);
  if (!finding) return null;
  finding.citations = getCitations(findingId);
  return finding;
}

// Resolve each cited chunk to {document_id, file_name, page_number, source_kind}
// so the UI can render a link straight to the archived PDF at the right page.
// Image chunks pull their page from document_images (image can appear on
// different pages in different documents — take the lowest-doc, lowest-page).
function getCitations(findingId) {
  const { db } = getDb();
  const rows = db.prepare(`
    SELECT fc.chunk_id, c.source_kind, c.source_id, c.page_number, c.text
    FROM finding_citations fc
    JOIN chunks c USING (chunk_id)
    WHERE fc.finding_id = ?
    ORDER BY fc.chunk_id
  `).all(findingId);

  return rows.map(r => ({
    chunk_id: r.chunk_id,
    source_kind: r.source_kind,
    text: r.text,
    ...resolveSource(r.source_kind, r.source_id, r.page_number)
  }));
}

function resolveSource(kind, sourceId, pageNumber) {
  const { db } = getDb();
  if (kind === 'document') {
    const doc = db.prepare(
      'SELECT document_id, file_name FROM documents WHERE document_id = ?'
    ).get(sourceId);
    return doc
      ? { document_id: doc.document_id, file_name: doc.file_name, page_number: pageNumber }
      : { document_id: null, file_name: null, page_number: pageNumber };
  }
  if (kind === 'summary') {
    const row = db.prepare(`
      SELECT d.document_id, d.file_name FROM summaries s
      JOIN documents d USING (document_id) WHERE s.summary_id = ?
    `).get(sourceId);
    return row
      ? { document_id: row.document_id, file_name: row.file_name, page_number: null }
      : { document_id: null, file_name: null, page_number: null };
  }
  if (kind === 'image') {
    const row = db.prepare(`
      SELECT di.document_id, di.page_number, d.file_name
      FROM document_images di
      JOIN documents d ON d.document_id = di.document_id
      WHERE di.image_id = ?
      ORDER BY di.document_id, di.page_number
      LIMIT 1
    `).get(sourceId);
    return row
      ? { document_id: row.document_id, file_name: row.file_name, page_number: row.page_number }
      : { document_id: null, file_name: null, page_number: null };
  }
  if (kind !== 'finding') {
    // 'finding' citations are expected to resolve to no linkable source;
    // anything else means the Python side grew a new source_kind we don't
    // know about yet. Loud at dev time so the drift gets caught.
    console.warn(`[bartleby/serve] Unknown chunk source_kind '${kind}' — citation will render unlinked.`);
  }
  return { document_id: null, file_name: null, page_number: null };
}

export function getDocumentFilePath(documentId) {
  const { db } = getDb();
  const row = db.prepare(
    'SELECT file_path FROM documents WHERE document_id = ?'
  ).get(documentId);
  return row ? row.file_path : null;
}

// Documents list — LEFT JOIN summaries so unsummarized docs still show up
// (rendered with the file name as a fallback title).
export function listDocuments() {
  const { db } = getDb();
  const rows = db.prepare(`
    SELECT d.document_id, d.file_name, d.page_count, d.created_at,
           s.title, s.description, ${TAGS_JSON_SUBQUERY}
    FROM documents d
    LEFT JOIN summaries s USING (document_id)
    ORDER BY COALESCE(s.title, d.file_name) COLLATE NOCASE
  `).all();
  return rows.map(withTags);
}

export function getDocument(documentId) {
  const { db } = getDb();
  const row = db.prepare(`
    SELECT d.document_id, d.file_name, d.page_count, d.created_at,
           s.title, s.description, s.text AS summary_text, s.model, ${TAGS_JSON_SUBQUERY}
    FROM documents d
    LEFT JOIN summaries s USING (document_id)
    WHERE d.document_id = ?
  `).get(documentId);
  return row ? withTags(row) : row;
}

// Replace the raw tags_json string from TAGS_JSON_SUBQUERY with a parsed `tags`
// array of {tag_id, name, description}.
function withTags(row) {
  const { tags_json, ...rest } = row;
  return { ...rest, tags: JSON.parse(tags_json) };
}

// Tag vocabulary, with the document count for each — used to populate the
// search filter. Tags with no documents are dropped (nothing to filter to).
export function listTags() {
  const { db } = getDb();
  return db.prepare(`
    SELECT t.name, COUNT(dt.document_id) AS document_count
    FROM tags t
    JOIN document_tags dt USING (tag_id)
    GROUP BY t.tag_id
    ORDER BY t.name COLLATE NOCASE
  `).all();
}

// The full tag vocabulary for the /tags index — includes the description and,
// unlike listTags() above, keeps tags with zero documents (a vocabulary entry
// is worth showing even before anything is assigned to it). Mirrors the
// read_tags skill's SQL.
export function listAllTags() {
  const { db } = getDb();
  return db.prepare(`
    SELECT t.tag_id, t.name, t.description, COALESCE(dt.n, 0) AS document_count
    FROM tags t
    LEFT JOIN (SELECT tag_id, COUNT(*) AS n FROM document_tags GROUP BY tag_id) dt
      USING (tag_id)
    ORDER BY t.name COLLATE NOCASE
  `).all();
}

// A single tag plus the documents carrying it (same shape as listDocuments,
// tags included). Returns null when the tag id is unknown.
export function getTag(tagId) {
  const { db } = getDb();
  const tag = db.prepare(
    'SELECT tag_id, name, description FROM tags WHERE tag_id = ?'
  ).get(tagId);
  if (!tag) return null;
  const rows = db.prepare(`
    SELECT d.document_id, d.file_name, d.page_count, d.created_at,
           s.title, s.description, ${TAGS_JSON_SUBQUERY}
    FROM document_tags dt
    JOIN documents d USING (document_id)
    LEFT JOIN summaries s USING (document_id)
    WHERE dt.tag_id = ?
    ORDER BY COALESCE(s.title, d.file_name) COLLATE NOCASE
  `).all(tagId);
  tag.documents = rows.map(withTags);
  return tag;
}

export function getCounts() {
  const { db } = getDb();
  const documents = db.prepare('SELECT COUNT(*) AS n FROM documents').get().n;
  const findings = db.prepare('SELECT COUNT(*) AS n FROM findings').get().n;
  return { documents, findings };
}
