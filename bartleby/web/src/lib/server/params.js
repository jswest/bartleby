import { error } from '@sveltejs/kit';

// Parse an integer route param (e.g. /documents/[id]); 404 on anything that
// isn't a whole number. Shared by the document/finding/tag/file loaders.
export function parseIdParam(raw) {
  const id = Number(raw);
  if (!Number.isInteger(id)) throw error(404, 'Not found');
  return id;
}

// Parse an integer query param into [lo, hi], falling back when absent or junk.
// Shared by the search and documents loaders for limit/offset/context bounds.
export function clampInt(raw, lo, hi, fallback) {
  const n = parseInt(raw ?? '', 10);
  if (Number.isNaN(n)) return fallback;
  return Math.min(hi, Math.max(lo, n));
}
