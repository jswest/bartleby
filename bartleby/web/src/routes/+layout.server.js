import { getDb } from '$lib/server/db.js';

export function load() {
  const { project } = getDb();
  return { project };
}
