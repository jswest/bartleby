import { error } from '@sveltejs/kit';
import { getFinding } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';

export function load({ params }) {
  const findingId = parseIdParam(params.id, 'finding');
  const finding = getFinding(findingId);
  if (!finding) throw error(404, 'Not found');
  return { finding };
}
