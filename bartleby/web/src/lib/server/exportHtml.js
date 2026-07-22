// Builds the standalone "Save as HTML" export of one finding (GH-0690): a
// single self-contained file — no network, no server — that reproduces the
// live /findings/[id] view: the two-column split, drop-capped serif prose,
// Tufte-style margin notes, and every cited source embedded inline (fonts as
// base64 @font-face, PDFs/images as base64 data: URIs, .txt/.md rendered the
// same way the live SourceViewer does). See
// docs/decisions/GH-0690-finding-html-export-0001.md for the resolved open
// questions this implements (PDF fallback download link, inert
// cross-finding/external notes, image-chunk embedding) and why the CSS below
// is a hand-copied snapshot of app.css rather than a shared import.
import fs from 'node:fs';
import path from 'node:path';
import { Marked } from 'marked';
import DOMPurify from 'isomorphic-dompurify';
import { getFinding, getDocumentFilePath, getImageFilePath } from './queries.js';
import { MIME_BY_EXT, IMAGE_EXTS } from './mime.js';
import { substituteCitations } from '../citations.js';
import { escapeHtml, slugify } from '../format.js';
import { wrapViewerDocument, RE_MARKDOWN, RE_TEXT, RE_PDF } from '../viewerDoc.js';

// ---------------------------------------------------------------------------
// Markdown rendering — an ISOLATED Marked instance, not the app's global
// `marked` singleton. That singleton gets its DOMPurify-sanitize + target/rel
// hook registered once, at module-load time, by routes/+layout.svelte's
// module-context script (see the comment there) — a page component this
// server-only export route never imports. Depending on whether some page
// happened to render first in the same process (registering the hook) would
// make this route's sanitization order-dependent; a private instance with the
// same hook sanitizes unconditionally instead.
const exportMarked = new Marked();
exportMarked.use({
  hooks: {
    postprocess: (html) =>
      DOMPurify.sanitize(html).replace(/<a /g, '<a target="_blank" rel="noopener noreferrer" ')
  }
});

// ---------------------------------------------------------------------------
// Fonts. `bartleby serve` always runs the vite DEV server out of
// ~/.bartleby/serve (commands/serve.py's _sync_web copies/symlinks this app's
// `static/` tree verbatim alongside `src/`, then `os.chdir`s there before
// exec) — so `static/fonts/` is a stable process.cwd()-relative path in the
// one way this app is actually run.
const FONTS_DIR = path.join(process.cwd(), 'static', 'fonts');

// Mirrors fonts.css's six @font-face declarations exactly (family, weight
// range, style) — only the `src` changes, from a `/fonts/...` URL to a base64
// `data:` URI so the export carries the font data itself.
const FONT_FACES = [
  { family: 'Doto', file: 'Doto-Variable.woff2', weight: '100 900', style: 'normal' },
  { family: 'Iosevka Term', file: 'IosevkaTerm-Regular.woff2', weight: '400', style: 'normal' },
  { family: 'Iosevka Term', file: 'IosevkaTerm-Medium.woff2', weight: '500', style: 'normal' },
  { family: 'Iosevka Term', file: 'IosevkaTerm-Bold.woff2', weight: '600 900', style: 'normal' },
  { family: 'Source Serif 4', file: 'SourceSerif4-Variable.woff2', weight: '200 900', style: 'normal' },
  { family: 'Source Serif 4', file: 'SourceSerif4-Italic-Variable.woff2', weight: '200 900', style: 'italic' }
];

function buildFontFaceCss() {
  return FONT_FACES.map(({ family, file, weight, style }) => {
    const data = fs.readFileSync(path.join(FONTS_DIR, file));
    const dataUri = `data:font/woff2;base64,${data.toString('base64')}`;
    return `@font-face { font-family: "${family}"; src: url("${dataUri}") format("woff2"); font-weight: ${weight}; font-style: ${style}; font-display: swap; }`;
  }).join('\n');
}

// ---------------------------------------------------------------------------
// CSS — a hand-curated snapshot of the app.css/SourceViewer.svelte rules the
// ledger view actually uses (tokens, base typography, .surface/.markdown-body,
// the R4 split/ledger/margin-note/cite-ref treatment, the R3 dagger bookmark,
// and SourceViewer's placeholder states), plus a handful of `.export-*` rules
// for pieces that only exist in the export (the PDF fallback link, the
// image-chunk viewer). It is NOT generated from app.css at request time, so a
// future visual change there won't automatically reach old or new exports —
// an accepted, documented trade-off (see the decision doc) given fonts must
// already be captured as a base64 snapshot for the same reason.
const EXPORT_CSS = `
:root {
  --font-display: "Doto", ui-monospace, SFMono-Regular, Menlo, monospace;
  --font-sans: "Iosevka Term", ui-monospace, SFMono-Regular, Menlo, monospace;
  --font-serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --font-mono: "Iosevka Term", ui-monospace, SFMono-Regular, Menlo, monospace;
  --text-display-floor: 1.25rem;
  --color-off: #698879;
  --color-off-light: #dbfaeb;
  --color-token: #ffecb9;
  --color-token-dark: #ffbd08;
  --color-token-dark-rgb: 255, 189, 8;
  --color-off-light-rgb: 219, 250, 235;
  --color-rule: #e3e1d5;
  --color-bg-soft: #f6f5ef;
  --color-shell: #14171a;
  --color-shell-text: #e8eae6;
  --color-text: #222;
  --color-text-soft: #444;
  --color-surface: #fff;
  --color-surface-card: #fafaf6;
  --color-link: #2f5a44;
  --color-token-text: #9a6700;
  --color-danger-text: #8a2b22;
  --space-2xs: 0.25rem;
  --space-3xs: 0.125rem;
  --space-xs: 0.375rem;
  --space-sm: 0.5rem;
  --space-md: 0.75rem;
  --space-lg: 1rem;
  --space-xl: 1.5rem;
  --space-2xl: 2rem;
  --text-2xs: 0.72rem;
  --text-xs: 0.8rem;
  --text-sm: 0.9rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.25rem;
  --text-2xl: 1.5rem;
  --text-3xl: 1.75rem;
  --tracking-wide: 0.05em;
  --shadow-card: 0 0 0 1px rgba(0, 0, 0, 0.04);
  --unit: 18px;
  --pixel-bookmark: url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%3E%3Cpath%20d%3D%22M6%202h12v2H6zM4%204h2v18H4zm14%200h2v18h-2zm-2%2016h2v2h-2zm-2-2h2v2h-2zm-8%202h2v2H6zm2-2h2v2H8zm2-2h4v2h-4z%22%2F%3E%3C%2Fsvg%3E");
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-size: var(--unit);
  color: var(--color-shell-text);
  background: var(--color-shell);
  padding: var(--space-2xl) var(--space-lg);
}
h1, h2, h3, h4, h5 { font-family: var(--font-sans); line-height: 1.2; }
li, p { font-family: var(--font-serif); line-height: 1.55; font-feature-settings: "onum" 1; }
.surface, .report { color: var(--color-text); }
.surface {
  background: var(--color-surface-card);
  border: 1px solid var(--color-rule);
  border-radius: 0;
  padding: var(--space-lg);
  box-shadow: var(--shadow-card);
}
.surface--finding { background: var(--color-off-light); border-color: var(--color-off); }
.surface--finding .meta { color: var(--color-link); font-weight: 400; }
p.meta, .meta { font-family: var(--font-sans); font-size: var(--text-xs); font-weight: 200; color: var(--color-off); }
.finding-id { font-family: var(--font-display); font-weight: 400; }

/* --- markdown-body (shared with /findings, /documents) --- */
.markdown-body { font-family: var(--font-serif); font-feature-settings: "onum" 1; line-height: 1.6; overflow-wrap: break-word; }
.markdown-body table { font-feature-settings: "lnum" 1; }
.markdown-body > * + * { margin-top: var(--space-lg); }
.markdown-body h1 { font-size: var(--text-2xl); font-weight: 700; margin-top: var(--space-2xl); padding-bottom: var(--space-xs); border-bottom: 1px solid var(--color-rule); }
.markdown-body h2 { font-size: var(--text-xl); font-weight: 600; margin-top: var(--space-xl); }
.markdown-body h3 { font-size: var(--text-lg); font-weight: 600; margin-top: var(--space-xl); }
.markdown-body h4, .markdown-body h5, .markdown-body h6 { font-size: var(--text-base); font-weight: 600; color: var(--color-off); margin-top: var(--space-lg); }
.markdown-body p { margin: 0; }
.markdown-body * + p { margin-top: var(--space-lg); }
.markdown-body strong { font-weight: 700; }
.markdown-body em { font-style: italic; }
.markdown-body ul, .markdown-body ol { padding-left: var(--space-xl); }
.markdown-body li { margin-top: var(--space-xs); }
.markdown-body li:first-child { margin-top: 0; }
.markdown-body li > p { margin-top: var(--space-xs); }
.markdown-body li > ul, .markdown-body li > ol { margin-top: var(--space-xs); }
.markdown-body a { color: var(--color-link); font-weight: 700; text-decoration: none; }
.markdown-body a:hover { text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 2px; }
.markdown-body blockquote { padding: var(--space-2xs) var(--space-lg); border-left: 3px solid var(--color-off); color: var(--color-text-soft); font-style: italic; }
.markdown-body code { font-family: var(--font-mono); font-size: 0.9em; background: var(--color-bg-soft); padding: 0.1em 0.35em; border-radius: 0; }
.markdown-body pre { font-family: var(--font-mono); font-size: var(--text-sm); background: var(--color-bg-soft); padding: var(--space-md) var(--space-lg); border-radius: 0; overflow-x: auto; line-height: 1.5; }
.markdown-body pre code { background: transparent; padding: 0; font-size: inherit; }
.markdown-body hr { border: 0; border-top: 1px solid var(--color-rule); margin: var(--space-2xl) 0; }
.markdown-body table { display: block; width: max-content; max-width: 100%; overflow-x: auto; border-collapse: collapse; font-family: var(--font-sans); font-size: var(--text-sm); }
.markdown-body th, .markdown-body td { background-color: var(--color-surface); border: 1px solid var(--color-rule); padding: var(--space-2xs) var(--space-sm); text-align: left; }
.markdown-body th { background: var(--color-bg-soft); font-weight: 600; }
.markdown-body img { max-width: 100%; height: auto; }

/* --- two-column split + R4 ledger treatment (#593) --- */
.split { display: grid; grid-template-columns: minmax(0, 54rem) minmax(0, 1fr); gap: var(--space-xl); align-items: start; }
.split .report { min-width: 0; }
.split .viewer { position: sticky; top: var(--space-lg); height: calc(100vh - 4rem); }
.split .viewer iframe { width: 100%; height: 100%; border: 1px solid var(--color-rule); }
.ledger { --ledger-prose-width: 36rem; --ledger-notes-width: 14rem; }
.ledger-hed { font-family: var(--font-sans); font-weight: 700; letter-spacing: -0.01em; max-width: var(--ledger-prose-width); }
.ledger .meta { max-width: var(--ledger-prose-width); }
.ledger-dek { font-family: var(--font-sans); font-weight: 400; font-size: var(--text-base); font-feature-settings: "lnum" 1; font-style: normal; color: var(--color-text-soft); margin-bottom: var(--space-lg); max-width: var(--ledger-prose-width); }
.ledger-column { display: grid; grid-template-columns: minmax(0, var(--ledger-prose-width)) minmax(0, var(--ledger-notes-width)); gap: var(--space-xl); align-items: start; margin: var(--space-xl) 0; }
.ledger-column .body { margin: 0; }
.drop-cap-body > p:first-of-type::first-letter { font-family: var(--font-display); font-weight: 500; float: left; font-size: 3.1em; line-height: 0.78; margin: 0.02em 0.06em 0 0; color: var(--color-text); }

/* --- margin-note citation gutter --- */
.cite-notes { position: relative; padding-left: var(--space-xl); }
.cite-notes .margin-note { left: 0; right: 0; }
.margin-note { font-family: var(--font-sans); font-size: var(--text-2xs); line-height: 1.4; }
.margin-note__head { margin: 0; text-transform: uppercase; letter-spacing: var(--tracking-wide); font-weight: 500; }
.margin-note__dagger { font-weight: 700; }
.margin-note__dagger::before {
  content: "";
  display: inline-block;
  width: 9px;
  height: 9px;
  margin-right: 0.25em;
  vertical-align: -1px;
  background-color: currentColor;
  image-rendering: pixelated;
  -webkit-mask: var(--pixel-bookmark) center / contain no-repeat;
  mask: var(--pixel-bookmark) center / contain no-repeat;
}
.margin-note__body { margin: var(--space-3xs) 0 0; font-feature-settings: "lnum" 1; word-break: break-word; }
.margin-note__body--doc { font-style: italic; color: var(--color-text-soft); }
.margin-note__source-row { display: flex; align-items: baseline; gap: var(--space-3xs); }
.margin-note__link { display: inline; padding: 0; border: 0; background: none; font: inherit; text-align: left; cursor: pointer; color: var(--color-token-text); text-decoration: underline; text-underline-offset: 2px; text-decoration-color: color-mix(in srgb, var(--color-token-text) 35%, transparent); }
.margin-note__link:hover, .margin-note__link.active { text-decoration-color: var(--color-token-text); }
.margin-note--source .margin-note__head { color: var(--color-token-text); }
.margin-note--gone .margin-note__head { color: var(--color-danger-text); }
.margin-note--gone .margin-note__body { color: var(--color-danger-text); font-style: italic; }
.margin-note--finding .margin-note__head { color: var(--color-token-text); }

/* --- inline dagger anchors --- */
.cite-ref { font-family: var(--font-sans); font-weight: 700; font-size: 0.7em; cursor: pointer; padding-left: 0.05em; }
.cite-ref--source { color: var(--color-token-text); }
.cite-ref--gone { color: var(--color-danger-text); }
.cite-ref--finding { color: var(--color-token-text); }
.cite-ref:hover { text-decoration: underline; }

/* --- viewer pane placeholder (mirrors SourceViewer.svelte) --- */
.placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: var(--space-xs); height: 100%; min-height: 8rem; padding: var(--space-2xl); color: var(--color-off); font-family: var(--font-sans); text-align: center; }
.placeholder__icon { font-size: var(--text-2xl); line-height: 1; opacity: 0.45; margin-bottom: var(--space-xs); }
.placeholder__msg { font-size: var(--text-sm); color: var(--color-off); font-family: var(--font-sans); }
.placeholder__detail { font-size: var(--text-xs); color: var(--color-off); opacity: 0.7; font-family: var(--font-sans); max-width: 22ch; line-height: 1.4; }

/* --- export-only additions: PDF fallback link, image-chunk viewer --- */
.export-pdf-wrap { display: flex; flex-direction: column; height: 100%; gap: var(--space-sm); }
.export-pdf-wrap iframe { flex: 1; min-height: 0; }
.export-fallback { font-family: var(--font-sans); font-size: var(--text-xs); }
.export-fallback a { color: var(--color-off-light); }
.export-image { max-width: 100%; max-height: 100%; display: block; margin: 0 auto; background: var(--color-surface); }

@media (max-width: 56rem) {
  .split { grid-template-columns: minmax(0, 1fr); }
  .split .viewer { height: auto; }
  .ledger-column { grid-template-columns: minmax(0, 1fr); }
  .cite-notes { display: flex; flex-direction: column; gap: var(--space-lg); border-top: 1px solid var(--color-rule); padding-left: 0; padding-top: var(--space-lg); margin-top: var(--space-md); height: auto !important; }
  .cite-notes .margin-note { position: static !important; top: auto !important; }
}
`;

// ---------------------------------------------------------------------------
// Source classification + embedding. RE_MARKDOWN/RE_TEXT/RE_PDF/IMAGE_EXTS are
// shared with SourceViewer.svelte's own dispatch ($lib/viewerDoc.js,
// $lib/server/mime.js), so a cited file renders through the same branch live
// and in the export.

// One citation → one embed key + page. `key` dedupes: several citations
// against the same document (at different pages) share one embedded payload,
// so a PDF's base64 data isn't repeated per-citation (see the module docs on
// why this is necessary — an "accepted, massive" file is still not the same
// as an accidentally-quadratic one).
function classifyCitation(c) {
  if (c.source_kind === 'image') {
    return { embedKind: 'image', key: `image:${c.source_id}` };
  }
  if (c.document_id == null) return null;
  const name = c.file_name ?? '';
  if (RE_PDF.test(name)) return { embedKind: 'pdf', key: `doc:${c.document_id}`, page: c.page_number ?? null };
  if (RE_MARKDOWN.test(name)) return { embedKind: 'markdown', key: `doc:${c.document_id}` };
  if (RE_TEXT.test(name)) return { embedKind: 'text', key: `doc:${c.document_id}` };
  const ext = path.extname(name).toLowerCase();
  if (IMAGE_EXTS.has(ext)) return { embedKind: 'image', key: `doc:${c.document_id}` };
  return { embedKind: 'raw', key: `doc:${c.document_id}` };
}

// Reads the file for one classified citation and returns its embed payload
// (or null when the archived file is missing on disk). Image-kind citations
// (source_kind='image') embed the archived .jpg itself, keyed by image_id —
// not the parent document's PDF (GH-0690 resolved question #3).
function buildEmbedPayload(classified, c) {
  const isImageChunk = classified.embedKind === 'image' && c.source_kind === 'image';
  const filePath = isImageChunk ? getImageFilePath(c.source_id) : getDocumentFilePath(c.document_id);
  if (!filePath || !fs.existsSync(filePath)) return null;

  // For an image-chunk citation, `c.file_name` is the PARENT document's name
  // (resolveSources() in queries.js maps image chunks to their parent doc for
  // the live citationUrl/label — see its comment); the archived jpg on disk
  // has its own name, which is what's actually being embedded here.
  const fileName = isImageChunk ? path.basename(filePath) : (c.file_name ?? path.basename(filePath));
  const data = fs.readFileSync(filePath);

  if (classified.embedKind === 'pdf') {
    return { kind: 'pdf', dataUri: `data:application/pdf;base64,${data.toString('base64')}`, fileName };
  }
  if (classified.embedKind === 'text') {
    return { kind: 'inline', srcdoc: wrapViewerDocument(`<pre>${escapeHtml(data.toString('utf-8'))}</pre>`), fileName };
  }
  if (classified.embedKind === 'markdown') {
    return { kind: 'inline', srcdoc: wrapViewerDocument(exportMarked.parse(data.toString('utf-8'))), fileName };
  }
  const mime = MIME_BY_EXT[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream';
  if (classified.embedKind === 'image') {
    return { kind: 'image', dataUri: `data:${mime};base64,${data.toString('base64')}`, fileName };
  }
  // 'raw' (html/htm, or anything the ingest pipeline didn't recognize): a
  // sandboxed iframe on the raw bytes, same as the live "everything else"
  // branch — no script execution (sandbox=""), whatever the browser can render.
  return { kind: 'raw', dataUri: `data:${mime};base64,${data.toString('base64')}`, fileName };
}

// Only an http(s) URL becomes a real, clickable href — this file is meant to
// be emailed/archived and opened by third parties, so a note carrying e.g.
// `[^url:javascript:alert(1)]` must not become an active link. (The live page
// has this same pre-existing gap in its own externalNote() rendering; left
// alone there — this export-only guard doesn't change live behavior.)
const RE_HTTP_URL = /^https?:\/\//i;

// ---------------------------------------------------------------------------
// Margin-note HTML. Head label is always "source" / "missing source" — the
// live template (+page.svelte) uses that generic label regardless of dagger
// kind, so this matches. Deliberate departures from the live markup (GH-0690
// resolved questions): a [^finding:N] note renders inert (no link — the other
// finding isn't in this file); a resolved chunk note drops the "open chunk"
// icon link (/chunks/N doesn't exist standalone either); a resolved chunk
// note whose archived file was missing at export time (`embeddedChunkIds`
// doesn't have it) renders inert with a visible hint instead of a dead,
// still-clickable button — same ¶/"source" glyph and head (it did resolve
// against the DB; only the file embed failed), just non-interactive body text.
function renderMarginNoteHtml(note, embeddedChunkIds) {
  const headLabel = note.gone ? 'missing source' : 'source';
  const kindClass = note.gone ? 'gone' : note.finding ? 'finding' : 'source';
  let bodyHtml;
  if (note.gone) {
    bodyHtml = `<p class="margin-note__body">no longer available</p>`;
  } else if (note.finding) {
    bodyHtml = `<p class="margin-note__body">${escapeHtml(note.label)}</p>`;
  } else if (note.external === 'url') {
    bodyHtml = RE_HTTP_URL.test(note.href)
      ? `<p class="margin-note__body"><a href="${escapeHtml(note.href)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(note.title)}">${escapeHtml(note.label)}</a></p>`
      : `<p class="margin-note__body">${escapeHtml(note.label)}</p>`;
  } else if (note.external === 'doc') {
    bodyHtml = `<p class="margin-note__body margin-note__body--doc" title="${escapeHtml(note.title)}">${escapeHtml(note.label)}</p>`;
  } else if (embeddedChunkIds.has(note.chunkId)) {
    bodyHtml = `<span class="margin-note__body margin-note__source-row"><button type="button" class="margin-note__link" data-chunk-id="${note.chunkId}" title="${escapeHtml(note.title)}">${escapeHtml(note.label)}</button></span>`;
  } else {
    bodyHtml = `<p class="margin-note__body margin-note__body--doc" title="${escapeHtml(note.title)}">${escapeHtml(note.label)} — source file missing at export time</p>`;
  }
  return `<div class="margin-note margin-note--${kindClass}" data-note="${note.n}">
    <p class="margin-note__head"><span class="margin-note__dagger">${note.dagger}${note.n}</span> ${headLabel}</p>
    ${bodyHtml}
  </div>`;
}

// A finding's meta line: "#N · session · model (Set by LLM) · created_at" —
// mirrors +page.svelte's <p class="meta"> markup.
function buildMetaLine(finding) {
  const modelMeta = finding.model
    ? `${finding.model}${finding.model_set_by_llm ? ' (Set by LLM)' : ''}`
    : null;
  const bits = [finding.session_name, modelMeta, finding.created_at].filter(Boolean);
  return `<span class="finding-id">#${finding.finding_id}</span> · ${escapeHtml(bits.join(' · '))}`;
}

// Escapes `<` out of a JSON payload before it goes inside a <script> tag, so a
// literal "</script" substring in embedded data (an ingest filename, a cited
// document's text) can't prematurely close the tag. The HTML tokenizer looks
// for that literal text regardless of the script's `type`, so this is needed
// even though these are `type="application/json"` (inert, never executed).
function jsonScriptSafe(value) {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}

// The margin-note layout (Tufte dagger-alignment) + viewer-pane click wiring,
// re-expressed in vanilla JS — a standalone equivalent of +page.svelte's
// layoutNotes()/activate()/citationUrl(), since the export carries no Svelte
// runtime. Embedded sources are pre-built server-side (BARTLEBY_CITATIONS maps
// a chunk_id to an embed key + page; BARTLEBY_SOURCES holds one payload per
// unique embed key), so activation here is a pure client-side lookup + DOM
// swap — no fetch, no network.
const RUNTIME_JS = `
(function () {
  var CITATIONS = JSON.parse(document.getElementById('bartleby-citations').textContent);
  var SOURCES = JSON.parse(document.getElementById('bartleby-sources').textContent);
  var container = document.getElementById('report');
  var citesAside = document.getElementById('cite-notes');
  var viewer = document.getElementById('viewer');

  function showPlaceholder() {
    viewer.innerHTML = '';
    var div = document.createElement('div');
    div.className = 'placeholder';
    var icon = document.createElement('span');
    icon.className = 'placeholder__icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '\\u25E7';
    var msg = document.createElement('p');
    msg.className = 'placeholder__msg';
    msg.textContent = 'No source selected';
    var detail = document.createElement('p');
    detail.className = 'placeholder__detail';
    detail.textContent = 'Click an inline citation to view the source here.';
    div.appendChild(icon);
    div.appendChild(msg);
    div.appendChild(detail);
    viewer.appendChild(div);
  }

  function showPdf(dataUri, page, fileName) {
    viewer.innerHTML = '';
    var wrap = document.createElement('div');
    wrap.className = 'export-pdf-wrap';
    var iframe = document.createElement('iframe');
    iframe.title = 'Source document';
    iframe.src = dataUri + (page ? ('#page=' + page + '&navpanes=0') : '#navpanes=0');
    wrap.appendChild(iframe);
    var dl = document.createElement('div');
    dl.className = 'export-fallback';
    var a = document.createElement('a');
    a.href = dataUri;
    a.download = fileName || 'source.pdf';
    a.textContent = 'Download ' + (fileName || 'this PDF') + ' (if the viewer above stays blank)';
    dl.appendChild(a);
    wrap.appendChild(dl);
    viewer.appendChild(wrap);
  }

  function showInline(srcdoc, title) {
    viewer.innerHTML = '';
    var iframe = document.createElement('iframe');
    iframe.title = title || 'Source';
    iframe.setAttribute('sandbox', '');
    iframe.srcdoc = srcdoc;
    viewer.appendChild(iframe);
  }

  function showImage(dataUri, fileName) {
    viewer.innerHTML = '';
    var img = document.createElement('img');
    img.className = 'export-image';
    img.src = dataUri;
    img.alt = fileName || 'Source image';
    viewer.appendChild(img);
  }

  function showRaw(dataUri, fileName) {
    viewer.innerHTML = '';
    var iframe = document.createElement('iframe');
    iframe.title = fileName || 'Source document';
    iframe.setAttribute('sandbox', '');
    iframe.src = dataUri;
    viewer.appendChild(iframe);
  }

  function activate(chunkId) {
    var meta = CITATIONS[chunkId];
    var src = meta && SOURCES[meta.key];
    if (!src) { showPlaceholder(); return; }
    if (src.kind === 'pdf') showPdf(src.dataUri, meta.page, src.fileName);
    else if (src.kind === 'inline') showInline(src.srcdoc, src.fileName);
    else if (src.kind === 'image') showImage(src.dataUri, src.fileName);
    else if (src.kind === 'raw') showRaw(src.dataUri, src.fileName);
    else { showPlaceholder(); return; }

    var buttons = container.querySelectorAll('.margin-note__link[data-chunk-id]');
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].classList.toggle('active', buttons[i].dataset.chunkId === String(chunkId));
    }
  }

  container.addEventListener('click', function (e) {
    var ref = e.target.closest('.cite-ref');
    if (ref) {
      e.preventDefault();
      var note = citesAside && citesAside.querySelector('.margin-note[data-note="' + ref.dataset.note + '"]');
      if (note) note.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      var btn = note && note.querySelector('[data-chunk-id]');
      if (btn) activate(btn.dataset.chunkId);
      return;
    }
    var link = e.target.closest('.margin-note__link[data-chunk-id]');
    if (link) {
      e.preventDefault();
      activate(link.dataset.chunkId);
    }
  });

  var NOTE_GAP = 8;
  function layoutNotes() {
    if (!citesAside) return;
    var narrow = window.matchMedia('(max-width: 56rem)').matches;
    var noteEls = Array.prototype.slice.call(citesAside.querySelectorAll('.margin-note'));
    if (narrow) {
      noteEls.forEach(function (el) { el.style.position = ''; el.style.top = ''; });
      citesAside.style.height = '';
      return;
    }
    if (noteEls.length === 0) return;
    var asideTop = citesAside.getBoundingClientRect().top + window.scrollY;
    var items = noteEls.map(function (el) {
      var n = el.dataset.note;
      var ref = container.querySelector('.cite-ref[data-note="' + n + '"]');
      var top = 0;
      if (ref) {
        top = ref.getBoundingClientRect().top + window.scrollY - asideTop;
        if (top < 0) top = 0;
      }
      el.style.position = 'absolute';
      el.style.top = top + 'px';
      return { el: el, top: top };
    });
    var runningBottom = 0;
    items.forEach(function (item) {
      var height = item.el.offsetHeight;
      if (item.top < runningBottom) item.top = runningBottom;
      item.el.style.top = item.top + 'px';
      runningBottom = item.top + height + NOTE_GAP;
    });
    citesAside.style.height = (runningBottom - NOTE_GAP) + 'px';
  }

  window.addEventListener('resize', layoutNotes);
  if (typeof ResizeObserver !== 'undefined') {
    var body = container.querySelector('.body');
    if (body) new ResizeObserver(layoutNotes).observe(body);
  }
  showPlaceholder();
  layoutNotes();
})();
`;

function renderDocument({ title, metaLine, description, bodyHtml, notesHtml, citationMetaJson, sourcesJson }) {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width" />
<title>${escapeHtml(title)} — Bartleby finding export</title>
<style>${buildFontFaceCss()}
${EXPORT_CSS}</style>
</head>
<body>
<div class="split">
  <article class="report surface surface--finding ledger" id="report">
    <h1 class="ledger-hed">${escapeHtml(title)}</h1>
    <p class="meta">${metaLine}</p>
    ${description ? `<p class="ledger-dek">${escapeHtml(description)}</p>` : ''}
    <div class="ledger-column">
      <div class="body markdown-body drop-cap-body">${bodyHtml}</div>
      ${notesHtml ? `<aside class="cite-notes" id="cite-notes" aria-label="Citations">${notesHtml}</aside>` : ''}
    </div>
  </article>
  <aside class="viewer" id="viewer"></aside>
</div>
<script type="application/json" id="bartleby-citations">${citationMetaJson}</script>
<script type="application/json" id="bartleby-sources">${sourcesJson}</script>
<script>${RUNTIME_JS}</script>
</body>
</html>
`;
}

// Assembles the standalone export for one finding. Returns null when the
// finding doesn't exist (the route 404s); returns `{ html, filename }`
// otherwise.
export function buildFindingExportHtml(findingId) {
  const finding = getFinding(findingId);
  if (!finding) return null;

  const byId = new Map(finding.citations.map((c) => [c.chunk_id, c]));
  const { markdown, notes } = substituteCitations(finding.body, byId, null);
  const bodyHtml = exportMarked.parse(markdown);

  // One embed payload per unique document/image id — `sources` is set
  // unconditionally (including `null`, when the archived file is missing),
  // so a repeat citation against the same key is a single lookup rather than
  // a re-stat-and-retry. `embeddedChunkIds` (only the chunks whose payload
  // came back non-null) gates both the runtime's CITATIONS wiring and the
  // margin note's clickability — a chunk that resolved in the DB but whose
  // file is gone renders inert, not a dead-but-clickable button.
  const sources = new Map();
  const embeddedChunkIds = new Set();
  const citationMeta = {};
  for (const note of notes) {
    if (note.gone || note.finding || note.external) continue;
    const c = byId.get(note.chunkId);
    const classified = c && classifyCitation(c);
    if (!classified) continue;
    if (!sources.has(classified.key)) {
      sources.set(classified.key, buildEmbedPayload(classified, c));
    }
    if (sources.get(classified.key)) {
      embeddedChunkIds.add(note.chunkId);
      citationMeta[note.chunkId] = { key: classified.key, page: classified.page ?? null };
    }
  }

  const html = renderDocument({
    title: finding.title,
    metaLine: buildMetaLine(finding),
    description: finding.description,
    bodyHtml,
    notesHtml: notes.map((note) => renderMarginNoteHtml(note, embeddedChunkIds)).join('\n'),
    citationMetaJson: jsonScriptSafe(citationMeta),
    sourcesJson: jsonScriptSafe(
      Object.fromEntries([...sources].filter(([, payload]) => payload != null))
    )
  });

  return { html, filename: `${slugify(finding.title)}.html` };
}
