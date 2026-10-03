import { json } from '@sveltejs/kit';
import { parseIdParam } from '$lib/server/params.js';
import { deleteAnnotation } from '$lib/server/annotations.js';

// DELETE /findings/[id]/annotations/[annotation_id] — remove a note (#734).
// Any note, human or agent: the web user owns the corpus. 204, or 404 JSON.
export function DELETE({ params }) {
  const findingId = parseIdParam(params.id, 'finding');
  const annotationId = parseIdParam(params.annotation_id, 'annotation');
  if (!deleteAnnotation(findingId, annotationId)) {
    return json({ error: `annotation ${annotationId} not found on finding ${findingId}`, code: 'not_found' }, { status: 404 });
  }
  return new Response(null, { status: 204 });
}
