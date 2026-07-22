// The sandboxed-iframe document shell for inline (markdown/text) source
// rendering — shared by SourceViewer.svelte (the live viewer pane) and the
// standalone HTML export (routes/findings/[id]/export.html/+server.js —
// GH-0690), so the exported file's embedded sources are styled identically to
// the live app without a second copy of this CSS to drift out of sync.

// File-type dispatch by extension — also shared by both, so a cited file
// renders through the same branch live and in the export.
export const RE_MARKDOWN = /\.(md|markdown)$/i;
export const RE_TEXT = /\.(txt|text|log)$/i;
export const RE_PDF = /\.pdf$/i;

// The iframe is its own document and can't reach app.css or its CSS variables,
// so the source styling is inlined. Neutral and compact — readable evidence,
// not a pixel-match to the prose pane.
export const VIEWER_CSS = `
    body { margin: 0; padding: 1.25rem 1.5rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 0.9rem; line-height: 1.6; color: #1a1a1a; background: #fff; overflow-wrap: anywhere; }
    h1, h2, h3, h4 { line-height: 1.25; margin: 1.4em 0 0.5em; }
    h1 { font-size: 1.5rem; } h2 { font-size: 1.25rem; } h3 { font-size: 1.05rem; }
    p, ul, ol, blockquote, table, pre { margin: 0 0 1em; }
    a { color: #0645ad; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85em; background: #f2f2f2; padding: 0.1em 0.3em; border-radius: 3px; }
    pre { background: #f6f6f6; padding: 0.8em 1em; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; overflow-wrap: anywhere; }
    pre code { background: none; padding: 0; }
    blockquote { border-left: 3px solid #ddd; padding-left: 1em; color: #555; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 0.4em 0.6em; text-align: left; }
    th { background: #f6f6f6; }
    img { max-width: 100%; }
    hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
  `;

// Wrap `inner` (already-rendered/escaped HTML) in a minimal standalone
// document for a sandboxed iframe: `default-src 'none'` blocks every outbound
// request (a tracking-pixel <img> in untrusted source markdown, which
// DOMPurify keeps, would otherwise beacon out on view); `style-src
// 'unsafe-inline'` only enables this inlined stylesheet.
export function wrapViewerDocument(inner) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'" />
<style>${VIEWER_CSS}</style>
</head>
<body>${inner}</body>
</html>`;
}
