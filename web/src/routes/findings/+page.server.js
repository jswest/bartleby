import { listFindings } from '$lib/server/queries.js';

export function load() {
  return { findings: listFindings() };
}
