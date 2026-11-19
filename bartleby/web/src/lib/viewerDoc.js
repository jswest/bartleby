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
    pre { background: #f6f6f6; padding: 0.8em 1em; border-radius: 4px; white-space: pre-wrap; overflow-wrap: anywhere; }
    pre code { background: none; padding: 0; }
    blockquote { border-left: 3px solid #ddd; padding-left: 1em; color: #555; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 0.4em 0.6em; text-align: left; }
    th { background: #f6f6f6; }
    img { max-width: 100%; }
    hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
    html { scroll-padding-top: 3rem; }
    mark.cite { background: #ffecb9; color: inherit; text-decoration: underline 1px #ffbd08; text-underline-offset: 2px; }
    .cite-anchor { display: inline-block; vertical-align: top; scroll-initial-target: nearest; }
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

// ===== Cited-passage highlight (GH-0754) ==================================
// The iframe is sandbox="" with a default-src 'none' CSP, so nothing can
// scroll it after load. Instead the highlight is baked into the srcdoc: the
// cited chunk's text is wrapped in <mark class="cite">, and a zero-size
// inline-block .cite-anchor at its start carries `scroll-initial-target`, so
// the browser opens the document already scrolled there (Chromium 133+; others
// just open at the top). It has to be an atomic inline box: Chromium ignores
// scroll-initial-target on a plain inline <mark>. It also scrolls only the
// nearest scroll container, which is why `pre` above sets no overflow-x (that
// made every <pre> one, so a .txt source never scrolled; pre-wrap and
// overflow-wrap already keep its lines inside the pane).

const WORD = /[\p{L}\p{N}]+/gu;
const wordKey = (words) => words.map((m) => m[0].toLowerCase()).join(" ");

// [from, to) of `passage` in `text`, or null. Chunk text isn't byte-identical
// to the source (CRLF is folded at ingest, and line breaks and spacing
// differ), so both sides are compared as word sequences, ignoring whitespace
// and punctuation. The .txt chunker cuts at character offsets, so a chunk's
// first and last words may be fragments: the search isn't padded at the ends,
// so the first word can match a word's tail and the last word its head.
// Docling prefixes each chunk with its heading path, one line per heading,
// and those headings aren't contiguous in the source, so if the whole chunk
// doesn't match, leading lines are dropped one at a time (at most six, one
// per heading level).
function locatePassage(text, passage) {
  const hay = [...text.matchAll(WORD)];
  const hayKey = wordKey(hay);
  const starts = []; // offset of each hay word in hayKey
  let pos = 0;
  for (const m of hay) {
    starts.push(pos);
    pos += m[0].toLowerCase().length + 1;
  }
  // hayKey offset → text offset, via the word it falls in.
  const toText = (at) => {
    const i = starts.findLastIndex((st) => st <= at);
    return hay[i].index + (at - starts[i]);
  };
  const lines = passage.split("\n");
  for (let drop = 0; drop < lines.length && drop <= 6; drop++) {
    const need = wordKey([...lines.slice(drop).join("\n").matchAll(WORD)]);
    if (!need) break;
    const at = hayKey.indexOf(need);
    if (at >= 0) return [toText(at), toText(at + need.length - 1) + 1];
  }
  return null;
}

// Wrap the first occurrence of the passage in rendered `html` (the srcdoc
// body) in <mark class="cite">s. `passageHtml` is the passage rendered the same
// way as the source, so the match runs rendered text to rendered text: a
// docling markdown chunk keeps `**`/`[text](url)` syntax that only rendering
// strips. The span may cross elements, so each text node it touches gets its
// own <mark>, and whitespace-only nodes are skipped (a <mark> between table
// rows would be foster-parented out of the table).
// Returns `html` unchanged when the passage isn't found. Browser-only.
export function markPassage(html, passageHtml) {
  const tpl = document.createElement("template");
  tpl.innerHTML = passageHtml;
  const passage = tpl.content.textContent;
  tpl.innerHTML = html;
  const walker = document.createTreeWalker(tpl.content, NodeFilter.SHOW_TEXT);
  const nodes = [];
  let text = "";
  while (walker.nextNode()) {
    nodes.push({ node: walker.currentNode, start: text.length });
    text += walker.currentNode.data;
  }
  const span = locatePassage(text, passage);
  if (!span) return html;
  const [from, to] = span;
  let anchored = false;
  for (const { node, start } of nodes) {
    const s = Math.max(from - start, 0);
    const e = Math.min(to - start, node.data.length);
    if (s >= e || !/\S/.test(node.data.slice(s, e))) continue;
    const hit = s > 0 ? node.splitText(s) : node;
    if (e - s < hit.data.length) hit.splitText(e - s);
    const mark = document.createElement("mark");
    mark.className = "cite";
    hit.replaceWith(mark);
    mark.append(hit);
    if (!anchored) {
      const anchor = document.createElement("span");
      anchor.className = "cite-anchor";
      mark.prepend(anchor);
      anchored = true;
    }
  }
  return tpl.innerHTML;
}
