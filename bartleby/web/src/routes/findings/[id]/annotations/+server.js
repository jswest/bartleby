import { json } from '@sveltejs/kit';
import { parseIdParam } from '$lib/server/params.js';
import { bareId } from '$lib/server/ids.js';
import { insertAnnotation, AnnotationError } from '$lib/server/annotations.js';

// POST /findings/[id]/annotations — add a human note (#734). One of serve's
// two write endpoints (with DELETE in [annotation_id]/); see
// docs/decisions/GH-0689-serve-first-write-path-0001.md.
// Body: {body, anchor?: {exact, prefix?, suffix?}, chunk_id?} → 201 + the row.
// Errors are JSON {error, code}, like the skill scripts.
export async function POST({ params, request }) {
  const findingId = parseIdParam(params.id, 'finding');
  try {
    const input = await request.json().catch(() => {
      throw new AnnotationError(400, 'invalid', 'request body must be JSON');
    });
    const row = insertAnnotation({ findingId, ...validate(input) });
    return json(row, { status: 201 });
  } catch (e) {
    if (e instanceof AnnotationError) return json({ error: e.message, code: e.code }, { status: e.status });
    throw e;
  }
}

const isText = (v) => typeof v === 'string';
const optionalText = (v) => v == null || isText(v);

function validate(input) {
  const bad = (msg) => new AnnotationError(400, 'invalid', msg);
  if (!input || typeof input !== 'object') throw bad('body must be a JSON object');
  const { body, anchor, chunk_id } = input;
  if (!isText(body) || !body.trim()) throw bad('note body must be a non-empty string');
  if (anchor != null) {
    if (typeof anchor !== 'object' || !isText(anchor.exact) || !anchor.exact) {
      throw bad('anchor.exact must be a non-empty string');
    }
    if (!optionalText(anchor.prefix) || !optionalText(anchor.suffix)) {
      throw bad('anchor.prefix/anchor.suffix must be strings');
    }
  }
  // A bare int or a type-tagged `chunk:N` (#624) — not some other type's id.
  const chunkId = chunk_id == null ? null
    : isText(chunk_id) && !/^\s*(chunk:)?\d+\s*$/.test(chunk_id) ? NaN
    : bareId(chunk_id);
  if (Number.isNaN(chunkId)) throw bad('chunk_id must be an integer or chunk:<N>');
  return {
    body: body.trim(),
    anchor: anchor ? { exact: anchor.exact, prefix: anchor.prefix || null, suffix: anchor.suffix || null } : null,
    chunkId,
  };
}
