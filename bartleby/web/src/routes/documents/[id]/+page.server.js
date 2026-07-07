import { error } from '@sveltejs/kit';
import fs from 'node:fs';
import { getDocument, getDocumentFilePath } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';

export function load({ params }) {
  const documentId = parseIdParam(params.id, 'document');
  const document = getDocument(documentId);
  if (!document) throw error(404, 'Not found');

  // Markdown and plain-text sources render inline in the viewer pane (the
  // iframe would only show raw text, or the browser's unstyled text/plain view),
  // so read the file here for SSR — no client fetch, no flash. Everything else
  // keeps the iframe and leaves this null.
  let sourceText = null;
  if (/\.(md|markdown|txt|text|log)$/i.test(document.file_name ?? '')) {
    const filePath = getDocumentFilePath(documentId);
    if (filePath && fs.existsSync(filePath)) {
      sourceText = fs.readFileSync(filePath, 'utf8');
    }
  }
  return { document, sourceText };
}
