import { getCounts } from '$lib/server/queries.js';

export function load() {
  return { counts: getCounts() };
}
