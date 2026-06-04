import { error } from '@sveltejs/kit';
import { getDocument } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';

export function load({ params }) {
  const documentId = parseIdParam(params.id);
  const document = getDocument(documentId);
  if (!document) throw error(404, 'Not found');
  return { document };
}
