import { error } from '@sveltejs/kit';
import fs from 'node:fs';
import { getDocument, getDocumentFilePath } from '$lib/server/queries.js';
import { parseIdParam } from '$lib/server/params.js';

export function load({ params }) {
  const documentId = parseIdParam(params.id, 'document');
  const document = getDocument(documentId);
  if (!document) throw error(404, 'Not found');

  // Markdown sources render to HTML in the viewer pane (the iframe would only
  // show raw text), so read the file here for SSR — no client fetch, no flash.
  // Everything else keeps the iframe and leaves this null.
  let sourceMarkdown = null;
  if (/\.md$/i.test(document.file_name ?? '')) {
    const filePath = getDocumentFilePath(documentId);
    if (filePath && fs.existsSync(filePath)) {
      sourceMarkdown = fs.readFileSync(filePath, 'utf8');
    }
  }
  return { document, sourceMarkdown };
}
