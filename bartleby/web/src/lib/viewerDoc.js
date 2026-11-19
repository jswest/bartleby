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

// A root's text nodes, each with its offset into their concatenation. Plain
// (the default), `text` is exactly the root's textContent / Range.toString(),
// which the finding page's annotation offsets depend on. With `forMatch`, the
// passage matcher's view instead: text inside raw-text elements (<style>,
// <textarea>, <title>, …) is skipped, since a <mark> there re-parses as
// literal text, and a "\n" is put between nodes split by a <br> or a block
// boundary, so `one<br>two` reads as two words rather than "onetwo". That
// "\n" belongs to no node, so no mark ever covers it.
const RAW_TEXT = new Set(["script", "style", "textarea", "title", "noscript", "xmp", "iframe", "noembed", "noframes", "plaintext"]);
const BLOCK = /^(address|article|aside|blockquote|caption|dd|details|div|dl|dt|figcaption|figure|footer|h[1-6]|header|hr|li|main|nav|ol|p|pre|section|summary|table|tbody|td|tfoot|th|thead|tr|ul)$/;
export function textNodes(root, { forMatch = false } = {}) {
  const walker = forMatch
    ? document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
        acceptNode: (n) => (RAW_TEXT.has(n.localName) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
      })
    : document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const blockOf = (n) => {
    while ((n = n.parentNode) && n !== root && !BLOCK.test(n.localName));
    return n;
  };
  const nodes = [];
  let text = "";
  let gap = false;
  let block = null;
  while (walker.nextNode()) {
    const n = walker.currentNode;
    if (n.nodeType !== Node.TEXT_NODE) {
      if (n.localName === "br") gap = true;
      continue;
    }
    if (forMatch) {
      const b = blockOf(n);
      if (gap || b !== block) text += text && "\n";
      gap = false;
      block = b;
    }
    nodes.push({ node: n, start: text.length });
    text += n.data;
  }
  return { nodes, text };
}

// [from, to) of `passage` in `text`, or null. Chunk text isn't byte-identical
// to the source (CRLF is folded at ingest, and line breaks and spacing
// differ), so both sides are compared as word sequences, ignoring whitespace,
// punctuation and case. That's one regex over `text`, built from the passage's
// words, so nothing is materialised per source word and the offsets are the
// source's own (lowercasing first would shift them: `İ` → `i̇`). The .txt
// chunker cuts at character offsets, so a chunk's first and last words may be
// fragments: the regex isn't anchored at its ends, so the first word can
// match a word's tail and the last word its head. Docling prefixes each chunk
// with its heading path, one line per heading, and those headings aren't
// contiguous in the source, so if the whole chunk doesn't match, leading lines
// are dropped one at a time (at most six, one per heading level). Every
// attempt needs at least MIN_WORDS words and at least half the chunk's words:
// a shorter needle recurs by chance, the first occurrence wins, and a wrong
// highlight is worse than none.
const WORD = /[\p{L}\p{N}]+/gu;
const MIN_WORDS = 5;
export function locatePassage(text, passage) {
  const lines = passage.split("\n").map((l) => l.match(WORD)).filter(Boolean);
  const min = Math.max(MIN_WORDS, Math.ceil(lines.flat().length / 2));
  for (let drop = 0; drop < lines.length && drop <= 6; drop++) {
    const words = lines.slice(drop).flat();
    if (words.length < min) break;
    let m;
    try {
      m = new RegExp(words.join("[^\\p{L}\\p{N}]+"), "iu").exec(text);
    } catch {
      return null; // a very long passage can overflow V8's regex compiler
    }
    if (m) return [m.index, m.index + m[0].length];
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
  const passage = textNodes(tpl.content, { forMatch: true }).text;
  tpl.innerHTML = html;
  const { nodes, text } = textNodes(tpl.content, { forMatch: true });
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
