import { error } from '@sveltejs/kit';
import { getChunk } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';

export function load({ params }) {
  const chunkId = parseIdParam(params.id);
  const chunk = getChunk(chunkId);
  if (!chunk) throw error(404, 'Not found');
  return { chunk };
}
