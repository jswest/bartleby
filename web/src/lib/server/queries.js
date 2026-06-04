import { getDb } from './db.js';

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
  return db.prepare(`
    SELECT d.document_id, d.file_name, d.page_count, d.created_at,
           s.title, s.description
    FROM documents d
    LEFT JOIN summaries s USING (document_id)
    ORDER BY COALESCE(s.title, d.file_name) COLLATE NOCASE
  `).all();
}

export function getDocument(documentId) {
  const { db } = getDb();
  return db.prepare(`
    SELECT d.document_id, d.file_name, d.page_count, d.created_at,
           s.title, s.description, s.text AS summary_text, s.model
    FROM documents d
    LEFT JOIN summaries s USING (document_id)
    WHERE d.document_id = ?
  `).get(documentId);
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

export function getCounts() {
  const { db } = getDb();
  const documents = db.prepare('SELECT COUNT(*) AS n FROM documents').get().n;
  const findings = db.prepare('SELECT COUNT(*) AS n FROM findings').get().n;
  return { documents, findings };
}
