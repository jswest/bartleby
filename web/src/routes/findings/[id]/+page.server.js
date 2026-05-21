import { error } from '@sveltejs/kit';
import { getFinding } from '$lib/server/queries.js';

export function load({ params }) {
  const findingId = Number(params.id);
  if (!Number.isInteger(findingId)) throw error(404, 'Not found');
  const finding = getFinding(findingId);
  if (!finding) throw error(404, 'Not found');
  return { finding };
}
