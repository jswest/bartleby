import { error } from '@sveltejs/kit';
import fs from 'node:fs';
import path from 'node:path';
import { getDocumentFilePath } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';

// Extensions that the ingest pipeline accepts (see bartleby/ingest/chunk.py).
// Anything else falls back to octet-stream — the browser will offer download.
const MIME_BY_EXT = {
  '.pdf': 'application/pdf',
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  // Browsers download text/markdown — serve as plain text so the iframe shows it.
  '.md': 'text/plain; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.tiff': 'image/tiff',
  '.tif': 'image/tiff'
};

export function GET({ params }) {
  const documentId = parseIdParam(params.document_id);

  const filePath = getDocumentFilePath(documentId);
  if (!filePath || !fs.existsSync(filePath)) throw error(404, 'File missing');

  const ext = path.extname(filePath).toLowerCase();
  const contentType = MIME_BY_EXT[ext] ?? 'application/octet-stream';

  const stat = fs.statSync(filePath);
  const stream = fs.createReadStream(filePath);
  const headers = {
    'Content-Type': contentType,
    'Content-Length': String(stat.size),
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
  return new Response(stream, { headers });
}
