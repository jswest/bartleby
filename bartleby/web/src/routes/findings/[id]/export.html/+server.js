import { error } from '@sveltejs/kit';
import { buildFindingExportHtml } from '$lib/server/exportHtml.js';
import { parseIdParam } from '$lib/server/params.js';

// "Save as HTML" (GH-0690): a single self-contained file reproducing the
// /findings/[id] view, with every cited source and font embedded so it opens
// with no server, no network, and no Bartleby install. See
// bartleby/web/src/lib/server/exportHtml.js for the assembly and
// docs/decisions/GH-0690-finding-html-export-0001.md for the design.
export function GET({ params }) {
  const findingId = parseIdParam(params.id, 'finding');
  const built = buildFindingExportHtml(findingId);
  if (!built) throw error(404, 'Not found');

  return new Response(built.html, {
    headers: {
      'Content-Type': 'text/html; charset=utf-8',
      'Content-Disposition': `attachment; filename="${built.filename}"`
    }
  });
}
