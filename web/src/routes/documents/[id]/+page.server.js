import { error } from '@sveltejs/kit';
import { getDocument } from '$lib/server/queries.js';

export function load({ params }) {
  const documentId = Number(params.id);
  if (!Number.isInteger(documentId)) throw error(404, 'Not found');
  const document = getDocument(documentId);
  if (!document) throw error(404, 'Not found');
  return { document };
}
