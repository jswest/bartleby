import { error } from '@sveltejs/kit';
import fs from 'node:fs';
import path from 'node:path';
import { getDocumentFilePath } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';
import { MIME_BY_EXT } from '$lib/server/mime.js';

export function GET({ params }) {
  const documentId = parseIdParam(params.document_id, 'document');

  const filePath = getDocumentFilePath(documentId);
  if (!filePath || !fs.existsSync(filePath)) throw error(404, 'File missing');

  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_BY_EXT[ext] ?? 'application/octet-stream';

  // Buffer rather than stream: a streamed Response throws an unhandled
  // ERR_INVALID_STATE (ReadableStream already closed) when a client aborts
  // mid-flight — iframe reloads, early-closing probes — crashing the whole Vite
  // dev server. These are small local evidence docs, so buffering is simpler and
  // bulletproof. The manual Content-Length is dropped; SvelteKit drains the
  // buffer as a chunked response, which browsers and the iframe handle fine.
  const data = fs.readFileSync(filePath);
  const headers = {
    'Content-Type': contentType,
    'Content-Disposition': 'inline',
    // Don't let the browser MIME-sniff a text or octet-stream file into runnable HTML.
    'X-Content-Type-Options': 'nosniff'
  };
  // Source documents are evidence to read, never to run. Sandbox HTML so its
  // scripts can't execute — whether embedded in our iframe or opened directly.
  // PDFs/images/text don't script, so they're left untouched.
  if (ext === '.html' || ext === '.htm') {
    headers['Content-Security-Policy'] = 'sandbox';
  }
  return new Response(data, { headers });
}
