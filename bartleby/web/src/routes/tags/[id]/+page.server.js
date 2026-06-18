import { error } from '@sveltejs/kit';
import { getTag } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';

export function load({ params }) {
  const tagId = parseIdParam(params.id, 'tag');
  const tag = getTag(tagId);
  if (!tag) throw error(404, 'Not found');
  return { tag };
}
