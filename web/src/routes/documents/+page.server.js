import { listDocuments } from '$lib/server/queries.js';

export function load() {
  return { documents: listDocuments() };
}
