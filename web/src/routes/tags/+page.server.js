import { listAllTags } from '$lib/server/queries.js';

export function load() {
  return { tags: listAllTags() };
}
