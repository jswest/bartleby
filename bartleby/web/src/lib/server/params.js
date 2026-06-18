import { error } from '@sveltejs/kit';
import { bareId } from './ids.js';

// Parse an integer route param (e.g. /documents/[id]); 404 on anything that
// isn't a whole number or a prefixed id of the expected type.
// `expectedType` is optional; when provided and the raw value carries a
// recognised prefix, the prefix must match — a wrong type (e.g. a finding id
// handed to the documents route) 404s rather than silently looking up the
// colliding row. Bare (non-prefixed) values skip the type check and are
// accepted as-is (tolerant of DB-sourced bare ints in URLs).
export function parseIdParam(raw, expectedType) {
  const str = typeof raw === 'string' ? raw.trim() : String(raw ?? '');

  if (expectedType && str.includes(':') && !str.startsWith(`${expectedType}:`)) {
    throw error(404, 'Not found');
  }

  const id = bareId(str);
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
