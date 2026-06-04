import { error } from '@sveltejs/kit';

// Parse an integer route param (e.g. /documents/[id]); 404 on anything that
// isn't a whole number. Shared by the document/finding/tag/file loaders.
export function parseIdParam(raw) {
  const id = Number(raw);
  if (!Number.isInteger(id)) throw error(404, 'Not found');
  return id;
}
