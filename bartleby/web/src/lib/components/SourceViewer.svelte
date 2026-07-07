<script>
  import { marked } from "marked";

  // The right-hand source pane, shared by /findings/[id] and /documents/[id].
  // Dispatches on file type: markdown renders to a sandboxed iframe (below),
  // plain text renders literally into the same sandboxed iframe (never through
  // `marked` — see docs/decisions/GH-0680), PDFs keep the browser's native
  // viewer, everything else is a sandboxed raw-file iframe. `src` is the /files
  // URL; `sourceText` is pre-loaded source text (SSR path) — when absent for a
  // markdown/text file, the text is fetched from `src`. With no `src` (and no
  // sourceText) the empty placeholder shows.
  export let fileName = null;
  export let src = null;
  export let sourceText = null;

  $: isMarkdown = /\.(md|markdown)$/i.test(fileName ?? "");
  // Plain text stays literal — never routed through marked. marked runs with
  // breaks: false (see +layout.svelte), so single newlines would collapse into
  // run-on paragraphs and stray #/*/_ chars would be misread as markdown.
  $: isText = /\.(txt|text|log)$/i.test(fileName ?? "");
  // PDFs need the native viewer (and #page= jumps), which a sandbox would hobble;
  // every other raw file is sandboxed so an ingested HTML doc can't run scripts.
  $: isPdf = /\.pdf$/i.test(fileName ?? "");
  // Markdown and plain text both render inline (into the iframe below);
  // everything else falls through to the raw-file iframe branch.
  $: isInline = isMarkdown || isText;

  let text = null;
  let errorMsg = null;
  let loading = false;

  // Resolve the source text: use the pre-loaded prop, else fetch `src`. Only
  // the inputs are referenced, so this never re-runs on its own writes to `text`.
  $: resolveText(isInline, sourceText, src);

  async function resolveText(active, preloaded, url) {
    errorMsg = null;
    if (!active || (preloaded == null && !url)) {
      text = null;
      return;
    }
    if (preloaded != null) {
      text = preloaded;
      return;
    }
    loading = true;
    text = null;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      text = await res.text();
    } catch (e) {
      errorMsg = e.message ?? String(e);
    } finally {
      loading = false;
    }
  }

  // marked.parse flows through the global DOMPurify hook (+layout.svelte), so the
  // HTML is already script-free. We still render it inside a sandboxed iframe
  // carrying `default-src 'none'`: the sandbox blocks scripts as defense in
  // depth, and the CSP is the *only* control that stops passive remote
  // subresource loads — a tracking-pixel `<img>` in untrusted source markdown,
  // which DOMPurify keeps, would otherwise beacon the host's IP on view.
  // `default-src 'none'` ⇒ zero outbound requests; `style-src 'unsafe-inline'`
  // only enables the inlined stylesheet below.
  $: srcdoc =
    text == null
      ? null
      : isMarkdown
        ? wrapDocument(marked.parse(text))
        : wrapDocument(`<pre>${escapeHtml(text)}</pre>`);

  function escapeHtml(raw) {
    return raw.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function wrapDocument(inner) {
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

  // The iframe is its own document and can't reach app.css or its CSS variables,
  // so the source styling is inlined. Neutral and compact — readable evidence,
  // not a pixel-match to the prose pane.
  const VIEWER_CSS = `
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
</script>

{#if isInline}
  {#if errorMsg}
    <div class="placeholder placeholder--error">
      <span class="placeholder__icon" aria-hidden="true">⚠</span>
      <p class="placeholder__msg">Source unavailable</p>
      <p class="placeholder__detail">{errorMsg}</p>
    </div>
  {:else if loading}
    <div class="placeholder">
      <p class="placeholder__msg">Loading source…</p>
    </div>
  {:else if srcdoc}
    <iframe title={isMarkdown ? "Rendered markdown source" : "Plain text source"} {srcdoc} sandbox=""></iframe>
  {/if}
{:else if src}
  {#key src}
    <iframe title="Source document" {src} sandbox={isPdf ? undefined : ""}></iframe>
  {/key}
{:else}
  <div class="placeholder">
    <span class="placeholder__icon" aria-hidden="true">◧</span>
    <p class="placeholder__msg">No source selected</p>
    <p class="placeholder__detail">Click an inline citation to view the source here.</p>
  </div>
{/if}

<style>
  /* B5 — SourceViewer empty/error placeholder.
     The viewer panel sits on the dark shell (not a paper card), so these states
     use shell-chrome tokens throughout. The placeholder sizes to its own content
     (no fixed height here — the .split .viewer container owns the height) and is
     centred vertically within whatever space the viewer gives it. */
  .placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: var(--space-xs);
    height: 100%;
    min-height: 8rem;
    padding: var(--space-2xl);
    color: var(--color-shell-text-soft);
    font-family: var(--font-sans);
    text-align: center;
  }
  .placeholder__icon {
    font-size: var(--text-2xl);
    line-height: 1;
    opacity: 0.45;
    margin-bottom: var(--space-xs);
  }
  .placeholder__msg {
    font-size: var(--text-sm);
    color: var(--color-shell-text-soft);
    font-family: var(--font-sans);
  }
  .placeholder__detail {
    font-size: var(--text-xs);
    color: var(--color-shell-text-soft);
    opacity: 0.7;
    font-family: var(--font-sans);
    max-width: 22ch;
    line-height: 1.4;
  }
  .placeholder--error .placeholder__icon {
    color: var(--color-token-dark);
    opacity: 0.8;
  }
  .placeholder--error .placeholder__msg {
    color: var(--color-shell-text);
  }
  .placeholder--error .placeholder__detail {
    color: var(--color-shell-text-soft);
    font-family: var(--font-mono);
    font-size: var(--text-2xs);
  }
</style>
