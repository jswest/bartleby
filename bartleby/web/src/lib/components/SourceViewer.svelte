<script>
  import { marked } from "marked";

  // The right-hand source pane, shared by /findings/[id] and /documents/[id].
  // Dispatches on file type: markdown renders to a sandboxed iframe (below),
  // PDFs keep the browser's native viewer, everything else is a sandboxed
  // raw-file iframe. `src` is the /files URL; `markdown` is pre-loaded source
  // text (SSR path) — when absent for a markdown file, the text is fetched from
  // `src`. With no `src` (and no markdown) the empty placeholder shows.
  export let fileName = null;
  export let src = null;
  export let markdown = null;

  $: isMarkdown = /\.(md|markdown)$/i.test(fileName ?? "");
  // PDFs need the native viewer (and #page= jumps), which a sandbox would hobble;
  // every other raw file is sandboxed so an ingested HTML doc can't run scripts.
  $: isPdf = /\.pdf$/i.test(fileName ?? "");

  let text = null;
  let errorMsg = null;
  let loading = false;

  // Resolve the markdown text: use the pre-loaded prop, else fetch `src`. Only
  // the inputs are referenced, so this never re-runs on its own writes to `text`.
  $: resolveMarkdown(isMarkdown, markdown, src);

  async function resolveMarkdown(active, md, url) {
    errorMsg = null;
    if (!active || (md == null && !url)) {
      text = null;
      return;
    }
    if (md != null) {
      text = md;
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
  $: srcdoc = text == null ? null : wrapMarkdown(marked.parse(text));

  function wrapMarkdown(inner) {
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
    pre { background: #f6f6f6; padding: 0.8em 1em; border-radius: 4px; overflow-x: auto; }
    pre code { background: none; padding: 0; }
    blockquote { border-left: 3px solid #ddd; padding-left: 1em; color: #555; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 0.4em 0.6em; text-align: left; }
    th { background: #f6f6f6; }
    img { max-width: 100%; }
    hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
  `;
</script>

{#if isMarkdown}
  {#if errorMsg}
    <p class="status">Couldn't load source: {errorMsg}</p>
  {:else if loading}
    <p class="status">Loading source…</p>
  {:else if srcdoc}
    <iframe title="Rendered markdown source" {srcdoc} sandbox=""></iframe>
  {/if}
{:else if src}
  {#key src}
    <iframe title="Source document" {src} sandbox={isPdf ? undefined : ""}></iframe>
  {/key}
{:else}
  <p class="empty">Click an inline citation to view the source here.</p>
{/if}

<style>
  .status {
    padding: var(--space-lg);
  }
</style>
