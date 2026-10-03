// finding_annotations access for serve (#689/#734). The DDL is pinned in #689
// and owned by bartleby/db/schema.py; this is the web twin of
// bartleby/db/annotations.py. Reads go through the shared read handle;
// insert/delete are the ONLY writes in the web app and use getWritableDb —
// see docs/decisions/GH-0689-serve-first-write-path-0001.md.
import { getDb, getWritableDb } from './db.js';
import { locateAnchor, occurrenceIndex } from '../annotations.js';

const COLUMNS = `annotation_id, finding_id, body, anchor_exact, anchor_prefix,
  anchor_suffix, chunk_id, is_human_author, session_id, created_at`;

// A validation failure the endpoints turn into a JSON error response.
export class AnnotationError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

// Layer the anchor's state against the finding's CURRENT body onto a row:
// `anchor_found` (true for a whole-finding note, as in the Python helper) and
// `anchor_occurrence` — which occurrence of `anchor_exact` the anchor picks
// out, so the page can highlight the same one in the rendered text.
function withAnchorState(row, findingBody) {
  if (row.anchor_exact == null) return { ...row, anchor_found: true, anchor_occurrence: null };
  const hit = locateAnchor(findingBody, row.anchor_exact, row.anchor_prefix, row.anchor_suffix);
  return {
    ...row,
    anchor_found: hit !== null,
    anchor_occurrence: hit ? occurrenceIndex(findingBody, row.anchor_exact, hit[0]) : null,
  };
}

// Oldest first. `findingBody` is passed in by callers that already loaded it.
export function listAnnotations(findingId, findingBody) {
  const { db } = getDb();
  return db.prepare(`SELECT ${COLUMNS} FROM finding_annotations
                     WHERE finding_id = ? ORDER BY annotation_id`)
    .all(findingId)
    .map((r) => withAnchorState(r, findingBody));
}

// Human-authored insert: is_human_author = 1, session_id = NULL (a browser has
// no session). Validates finding (404), anchor (400), chunk (400) in one
// transaction with the insert. `anchor` is {exact, prefix?, suffix?} or null.
export function insertAnnotation({ findingId, body, anchor, chunkId }) {
  const { db } = getWritableDb();
  return db.transaction(() => {
    const finding = db.prepare('SELECT body FROM findings WHERE finding_id = ?').get(findingId);
    if (!finding) throw new AnnotationError(404, 'not_found', `finding ${findingId} does not exist`);
    if (anchor && !locateAnchor(finding.body, anchor.exact, anchor.prefix, anchor.suffix)) {
      throw new AnnotationError(400, 'anchor_not_found', 'anchor does not match the finding body');
    }
    if (chunkId != null && !db.prepare('SELECT 1 FROM chunks WHERE chunk_id = ?').get(chunkId)) {
      throw new AnnotationError(400, 'invalid', `chunk ${chunkId} does not exist`);
    }
    const { lastInsertRowid } = db.prepare(`
      INSERT INTO finding_annotations
        (finding_id, body, anchor_exact, anchor_prefix, anchor_suffix, chunk_id, is_human_author, session_id)
      VALUES (?, ?, ?, ?, ?, ?, 1, NULL)
    `).run(findingId, body, anchor?.exact ?? null, anchor?.prefix ?? null, anchor?.suffix ?? null, chunkId ?? null);
    const row = db.prepare(`SELECT ${COLUMNS} FROM finding_annotations WHERE annotation_id = ?`).get(lastInsertRowid);
    return withAnchorState(row, finding.body);
  })();
}

// Scoped to the finding in the URL so /findings/1/annotations/9 can't delete
// a note on another finding. True if a row was removed.
export function deleteAnnotation(findingId, annotationId) {
  const { db } = getWritableDb();
  return db.prepare('DELETE FROM finding_annotations WHERE annotation_id = ? AND finding_id = ?')
    .run(annotationId, findingId).changes > 0;
}
