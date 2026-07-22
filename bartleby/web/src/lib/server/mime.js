// Extensions the ingest pipeline accepts (see bartleby/ingest/chunk.py).
// Anything else falls back to octet-stream — the browser will offer download.
// Shared by the live /files/[document_id] route (Content-Type header) and the
// standalone HTML export (routes/findings/[id]/export.html/+server.js —
// GH-0690, embedded sources' data: URI MIME types) so the two stay in sync.
export const MIME_BY_EXT = {
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
