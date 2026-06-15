import { getDb } from './db.js';
import { clampSnippet } from '$lib/format.js';

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
           s.name AS session_name, s.model,
           (s.run_key IS NOT NULL AND s.model IS NOT NULL) AS model_set_by_llm,
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
           s.name AS session_name, s.model,
           (s.run_key IS NOT NULL AND s.model IS NOT NULL) AS model_set_by_llm
    FROM findings f
    JOIN sessions s USING (session_id)
    WHERE f.finding_id = ?
  `).get(findingId);
  if (!finding) return null;
  finding.citations = getCitations(findingId);
  return finding;
}

// Resolve each cited chunk to {document_id, file_name, page_number} so the UI
// can render a link straight to the archived PDF at the right page.
// Image chunks pull their page from document_images (image can appear on
// different pages in different documents — take the lowest-doc, lowest-page).
function getCitations(findingId) {
  const { db } = getDb();
  const rows = db.prepare(`
    SELECT fc.chunk_id, c.source_kind, c.source_id, c.page_number
    FROM finding_citations fc
    JOIN chunks c USING (chunk_id)
    WHERE fc.finding_id = ?
    ORDER BY fc.chunk_id
  `).all(findingId);

  const resolved = resolveSources(rows);
  return rows.map((r, i) => ({
    chunk_id: r.chunk_id,
    ...resolved[i]
  }));
}

// Batch-resolve cited chunks ({source_kind, source_id, page_number}) to
// {document_id, file_name, page_number} — one IN(...) query per source_kind
// rather than a row-at-a-time lookup, mirroring titleMeta/tagsByDocument.
// Returns an array aligned 1:1 with `items`. A 'finding' (or unrecognized)
// kind resolves to the null source — findings link to their own page; an
// unknown kind also warns once per occurrence so Python-side drift gets caught.
function resolveSources(items) {
  const { db } = getDb();

  // Distinct non-null source_ids cited under `kind`, for its IN(...) query.
  const idsOf = (kind) => [...new Set(
    items.filter((it) => it.source_kind === kind && it.source_id != null)
         .map((it) => it.source_id)
  )];

  // Run `buildSql(placeholders)` over `ids` and key the rows by `row.key`. For
  // kinds with several rows per id (images), the first row wins — callers order
  // the query so that's the intended pick.
  const loadMap = (ids, buildSql) => {
    const map = new Map();
    if (ids.length === 0) return map;
    const ph = ids.map(() => '?').join(',');
    for (const row of db.prepare(buildSql(ph)).all(...ids)) {
      if (!map.has(row.key)) map.set(row.key, row);
    }
    return map;
  };

  const docs = loadMap(idsOf('document'), (ph) =>
    `SELECT document_id AS key, document_id, file_name
     FROM documents WHERE document_id IN (${ph})`);
  const summaries = loadMap(idsOf('summary'), (ph) =>
    `SELECT s.summary_id AS key, d.document_id, d.file_name
     FROM summaries s JOIN documents d USING (document_id)
     WHERE s.summary_id IN (${ph})`);
  // An image can appear on different pages in different documents — order so the
  // lowest-doc, lowest-page row is first (loadMap keeps the first per image_id),
  // matching the per-row ORDER BY ... LIMIT 1 this batches.
  const images = loadMap(idsOf('image'), (ph) =>
    `SELECT di.image_id AS key, di.document_id, di.page_number, d.file_name
     FROM document_images di JOIN documents d ON d.document_id = di.document_id
     WHERE di.image_id IN (${ph})
     ORDER BY di.image_id, di.document_id, di.page_number`);

  return items.map((it) => {
    if (it.source_kind === 'document') {
      const doc = docs.get(it.source_id);
      return { document_id: doc?.document_id ?? null, file_name: doc?.file_name ?? null, page_number: it.page_number };
    }
    if (it.source_kind === 'summary') {
      const row = summaries.get(it.source_id);
      return { document_id: row?.document_id ?? null, file_name: row?.file_name ?? null, page_number: null };
    }
    if (it.source_kind === 'image') {
      const row = images.get(it.source_id);
      return row
        ? { document_id: row.document_id, file_name: row.file_name, page_number: row.page_number }
        : { document_id: null, file_name: null, page_number: null };
    }
    if (it.source_kind !== 'finding') {
      console.warn(`[bartleby/serve] Unknown chunk source_kind '${it.source_kind}' — citation will render unlinked.`);
    }
    return { document_id: null, file_name: null, page_number: null };
  });
}

// Single-item resolveSources, for callers with one source to resolve.
function resolveSource(kind, sourceId, pageNumber) {
  return resolveSources([{ source_kind: kind, source_id: sourceId, page_number: pageNumber }])[0];
}

// Batch-load `title, description` rows from `table`, keyed by `keyCol`. Used to
// enrich search/scan hits, which the skill returns with filenames/ids only —
// the human-facing title/description live in `summaries` (by document_id) and
// `findings` (by finding_id).
function titleMeta(db, table, keyCol, ids) {
  const uniq = [...new Set(ids.filter((id) => id != null))];
  const out = new Map();
  if (uniq.length === 0) return out;
  const ph = uniq.map(() => '?').join(',');
  for (const row of db.prepare(
    `SELECT ${keyCol}, title, description FROM ${table} WHERE ${keyCol} IN (${ph})`
  ).all(...uniq)) {
    out.set(row[keyCol], row);
  }
  return out;
}

// The document detail route (summary + PDF viewer), opened at the cited page
// via ?page= — the [id] route seeds its viewer iframe from that param.
function documentHref(documentId, pageNumber) {
  return `/documents/${documentId}${pageNumber ? `?page=${pageNumber}` : ''}`;
}

// Enrich ranked `search` hits with a display title, description, and a single
// destination href: the document detail page at the cited page for
// document-backed hits, the finding page for findings. Keeps the result
// component free of source_kind branching. (resolveSources already maps every
// kind → {document_id, file_name, page_number}; we layer title/description on top.)
export function enrichHits(hits) {
  const { db } = getDb();
  const resolved = resolveSources(hits);
  const summaries = titleMeta(db, 'summaries', 'document_id', resolved.map((r) => r.document_id));
  const findings = titleMeta(db, 'findings', 'finding_id',
    hits.filter((h) => h.source_kind === 'finding').map((h) => h.source_id));

  return hits.map((h, i) => {
    if (h.source_kind === 'finding') {
      const f = findings.get(h.source_id);
      return {
        ...h,
        title: f?.title ?? h.source_name,
        description: f?.description ?? null,
        file_name: null,
        page_number: null,
        href: `/findings/${h.source_id}`
      };
    }
    const r = resolved[i];
    const meta = r.document_id != null ? summaries.get(r.document_id) : null;
    const href = r.document_id != null ? documentHref(r.document_id, r.page_number) : null;
    return {
      ...h,
      text: clampSnippet(h.text),
      title: meta?.title ?? null,
      description: meta?.description ?? null,
      file_name: r.file_name ?? h.file_name ?? null,
      page_number: r.page_number,
      href
    };
  });
}

// Same enrichment for `scan` matches, which are documents-only and already
// carry document_id / file_name / page_number directly.
export function enrichScanMatches(matches) {
  const { db } = getDb();
  const summaries = titleMeta(db, 'summaries', 'document_id', matches.map((m) => m.document_id));
  return matches.map((m) => {
    const meta = summaries.get(m.document_id);
    return {
      ...m,
      text: clampSnippet(m.text),
      title: meta?.title ?? null,
      description: meta?.description ?? null,
      href: documentHref(m.document_id, m.page_number)
    };
  });
}

export function getDocumentFilePath(documentId) {
  const { db } = getDb();
  const row = db.prepare(
    'SELECT file_path FROM documents WHERE document_id = ?'
  ).get(documentId);
  return row ? row.file_path : null;
}

// How many neighbor chunks to show on each side of the target in the /chunks
// view — the surrounding context, rendered muted. Clamped at source boundaries.
const CHUNK_NEIGHBORS = 2;

// A single chunk plus its neighbor window — the chunks sharing its
// (source_kind, source_id), within ±CHUNK_NEIGHBORS of its chunk_index, ordered
// by chunk_index with the target flagged. Layers on a "back to source" link and
// a display label: for a document/summary/image chunk that's the document detail
// page (via resolveSource, same mapping as citations/search); for a finding
// chunk it's the finding page. Returns null for an unknown chunk id.
export function getChunk(chunkId) {
  const { db } = getDb();
  const target = db.prepare(`
    SELECT chunk_id, source_kind, source_id, chunk_index, text,
           section_heading, page_number, content_type
    FROM chunks WHERE chunk_id = ?
  `).get(chunkId);
  if (!target) return null;

  const neighbors = db.prepare(`
    SELECT chunk_id, chunk_index, text, section_heading, page_number, content_type
    FROM chunks
    WHERE source_kind = ? AND source_id = ? AND chunk_index BETWEEN ? AND ?
    ORDER BY chunk_index
  `).all(
    target.source_kind, target.source_id,
    target.chunk_index - CHUNK_NEIGHBORS, target.chunk_index + CHUNK_NEIGHBORS
  ).map((c) => ({ ...c, is_target: c.chunk_id === target.chunk_id }));

  // Resolve the source identity + back link. Findings don't resolve to a
  // document (resolveSource returns nulls), so they link to their own page.
  let title = null;
  let file_name = null;
  let back_href = null;
  if (target.source_kind === 'finding') {
    title = titleMeta(db, 'findings', 'finding_id', [target.source_id]).get(target.source_id)?.title ?? null;
    back_href = `/findings/${target.source_id}`;
  } else {
    const source = resolveSource(target.source_kind, target.source_id, target.page_number);
    file_name = source.file_name;
    if (source.document_id != null) {
      title = titleMeta(db, 'summaries', 'document_id', [source.document_id]).get(source.document_id)?.title ?? null;
      back_href = documentHref(source.document_id, source.page_number);
    }
  }

  return { ...target, neighbors, title, file_name, back_href };
}

// Tags for a set of documents, as a Map<document_id, tags[]>. The /documents
// list delegates enumeration + filtering to the list_documents skill (one
// source of truth for the date/null semantics), but the skill is agent-shaped
// and returns no tags — so we re-attach them here in one batched query, the
// same pattern titleMeta() uses to enrich search hits. Documents with no tags
// are simply absent from the map (callers default to []).
export function tagsByDocument(documentIds) {
  const { db } = getDb();
  const uniq = [...new Set(documentIds.filter((id) => id != null))];
  const out = new Map();
  if (uniq.length === 0) return out;
  const ph = uniq.map(() => '?').join(',');
  const rows = db.prepare(`
    SELECT dt.document_id, t.tag_id, t.name, t.description
    FROM document_tags dt JOIN tags t USING (tag_id)
    WHERE dt.document_id IN (${ph})
    ORDER BY t.name COLLATE NOCASE
  `).all(...uniq);
  for (const r of rows) {
    if (!out.has(r.document_id)) out.set(r.document_id, []);
    out.get(r.document_id).push({ tag_id: r.tag_id, name: r.name, description: r.description });
  }
  return out;
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

// Tag vocabulary for the search/documents filter — the listAllTags() rows that
// carry at least one document (nothing to filter to otherwise). Same ordering;
// the filter forms read only name + document_count.
export function listTags() {
  return listAllTags().filter((t) => t.document_count > 0);
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

// A single tag plus the documents carrying it, each in the shape DocumentList
// expects (document_id, file_name, page_count, title, description, tags).
// Returns null when the tag id is unknown.
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
