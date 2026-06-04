import { error } from '@sveltejs/kit';
import { getTag } from '$lib/server/queries.js';

export function load({ params }) {
  const tagId = Number(params.id);
  if (!Number.isInteger(tagId)) throw error(404, 'Not found');
  const tag = getTag(tagId);
  if (!tag) throw error(404, 'Not found');
  return { tag };
}
