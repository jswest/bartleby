import { error } from '@sveltejs/kit';
import fs from 'node:fs';
import { getDocumentFilePath } from '$lib/server/queries.js';

// Stream the archived file behind a document so the browser opens it inline
// with #page=N honored. We don't sniff content type — pdfplumber archives PDFs
// and that's what's clickable. If someone ingests something else we'll deal
// with it then.
export function GET({ params }) {
  const documentId = Number(params.document_id);
  if (!Number.isInteger(documentId)) throw error(404, 'Not found');

  const filePath = getDocumentFilePath(documentId);
  if (!filePath || !fs.existsSync(filePath)) throw error(404, 'File missing');

  const stat = fs.statSync(filePath);
  const stream = fs.createReadStream(filePath);
  return new Response(stream, {
    headers: {
      'Content-Type': 'application/pdf',
      'Content-Length': String(stat.size),
      'Content-Disposition': 'inline'
    }
  });
}
