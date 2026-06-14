<script>
  import { onMount } from "svelte";
  import { goto } from "$app/navigation";
  import { marked } from "marked";
  import Button from "$lib/components/Button.svelte";
  import { stripExt } from "$lib/format.js";
  import { CHUNK_ICON } from "$lib/icons.js";
  export let data;

  $: byId = new Map(data.finding.citations.map((c) => [c.chunk_id, c]));

  // The run's model for the meta line. Rendered inline among the · -separated
  // fields with the qualifier in parens — deliberately distinct from the index's
  // "model · Set by LLM" muted line (#547). null when no model was recorded.
  $: modelMeta = data.finding.model
    ? `${data.finding.model}${data.finding.model_set_by_llm ? " (Set by LLM)" : ""}`
    : null;

  // Substitute each citation marker in the markdown source before marked parses.
  // Two kinds: digit [^N] → a corpus-chunk chip (resolves against byId); and
  // [^url:…]/[^doc:…] → a visibly-distinct external-citation chip. External
  // markers are matched first so the digit pass can't see their alpha schemes.
  // Marked passes inline HTML through, so chips sit inside the same <p> as the
  // surrounding prose. Reactive on `active` so .active bakes into the HTML.
  $: bodyHtml = marked.parse(
    data.finding.body
      .replace(/\[\^([A-Za-z]+):([^\]]+)\]/g, (match, scheme, ref) =>
        renderExternal(scheme.toLowerCase(), ref.trim()),
      )
      .replace(/\[\^(\d+)\]/g, (match, idStr) => {
        const c = byId.get(Number(idStr));
        return c ? renderChip(c, c === active) : renderTombstone(Number(idStr));
      }),
  );

  // An external (URL / external-dataset doc) citation that rides alongside the
  // corpus-chunk citations. No DB row backs it — it's parsed straight from the
  // body marker. Rendered as a distinct chip: a url scheme becomes a real link
  // (new tab via the layout's marked hook), a doc scheme a muted ref label.
  // Unknown schemes are dropped (left to the well-formedness check at save time).
  function renderExternal(scheme, ref) {
    if (scheme === "url") {
      return `<a class="cite-external cite-external--url" href="${esc(ref)}" title="${esc(`external source · ${ref}`)}">${esc(ref)}</a>`;
    }
    if (scheme === "doc") {
      return `<span class="cite-external cite-external--doc" title="${esc(`external dataset doc · ${ref}`)}">${esc(ref)}</span>`;
    }
    return esc(`[^${scheme}:${ref}]`);
  }

  // An unresolved [^N] marker: the cited source has been removed (deleted, or
  // rebuilt under new chunk ids by an edit/merge), so only the id survives. We
  // can't say what it was or whether it was truly deleted — render a muted
  // tombstone that keeps the marker present rather than leaking raw [^N] text.
  function renderTombstone(chunkId) {
    return `<span class="cite-gone" title="${esc(`chunk ${chunkId} · cited source no longer available`)}">cited source no longer available</span>`;
  }

  function renderChip(c, isActive) {
    const name = stripExt(c.file_name);
    const label =
      name && c.page_number ? `${name} p.${c.page_number}`
      : name ? name
      : c.page_number ? `p.${c.page_number}`
      : `chunk ${c.chunk_id}`;
    const title = [
      c.file_name,
      c.page_number && `page ${c.page_number}`,
      `chunk ${c.chunk_id}`,
    ].filter(Boolean).join(" · ");
    const cls = `cite-chip${isActive ? " active" : ""}`;
    const chip = `<button type="button" class="${cls}" data-chunk-id="${c.chunk_id}" title="${esc(title)}">${esc(label)}</button>`;
    // A sibling jump button → /chunks/<id>. A <button> (not <a>) so it routes
    // in-tab via goto; the layout's marked hook rewrites every <a> to a new tab.
    const jump = `<button type="button" class="cite-jump" data-chunk-id="${c.chunk_id}" title="Open chunk ${c.chunk_id} in context" aria-label="Open chunk ${c.chunk_id} in context">${CHUNK_ICON}</button>`;
    return `<span class="cite">${chip}${jump}</span>`;
  }

  function esc(s) {
    return s.replace(/[&<>"]/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
    ));
  }

  let active = null;
  $: activeUrl = active ? citationUrl(active) : null;
  // Sandbox the viewer for non-PDF sources so an ingested HTML document can't run
  // its scripts; PDFs keep the native viewer + #page= jumps. Unknown → sandboxed.
  $: activeIsPdf = /\.pdf$/i.test(active?.file_name ?? "");

  function citationUrl(c) {
    if (c.document_id == null) return null;
    const page = c.page_number ? `#page=${c.page_number}` : "";
    return `/files/${c.document_id}${page}`;
  }

  // Chips are static HTML inside {@html}, so click delegation on the container
  // is simpler than wiring each as a Svelte component.
  let container;
  onMount(() => {
    container.addEventListener("click", (e) => {
      const jump = e.target.closest(".cite-jump");
      if (jump) {
        e.preventDefault();
        goto(`/chunks/${jump.dataset.chunkId}`);
        return;
      }
      const btn = e.target.closest(".cite-chip");
      if (!btn) return;
      e.preventDefault();
      const c = byId.get(Number(btn.dataset.chunkId));
      if (c) active = c;
    });
  });

  let copied = false;
  async function copyMarkdown() {
    await navigator.clipboard.writeText(data.finding.body);
    copied = true;
    setTimeout(() => (copied = false), 1500);
  }

  function downloadMarkdown() {
    const blob = new Blob([data.finding.body], { type: "text/markdown" });
    const name = data.finding.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.md`;
    a.click();
  }
</script>

<div class="split">
  <article class="report surface surface--finding" bind:this={container}>
    <h1>{data.finding.title}</h1>
    <p class="meta">
      <span class="finding-id">#{data.finding.finding_id}</span> · {data.finding.session_name}{#if modelMeta} · {modelMeta}{/if} · {data.finding.created_at}
    </p>
    <p class="desc">{data.finding.description}</p>

    <div class="toolbar">
      <Button size="sm" type="button" on:click={copyMarkdown}>
        {copied ? "Copied" : "Copy as Markdown"}
      </Button>
      <Button size="sm" type="button" on:click={downloadMarkdown}>Download .md</Button>
    </div>

    <div class="body markdown-body">
      {@html bodyHtml}
    </div>
  </article>

  <aside class="viewer">
    {#if activeUrl}
      {#key activeUrl}
        <iframe title="Source document" src={activeUrl} sandbox={activeIsPdf ? undefined : ""}></iframe>
      {/key}
    {:else}
      <p class="empty">
        Click an inline citation to view the source PDF here.
      </p>
    {/if}
  </aside>
</div>

<style>
  .toolbar {
    display: flex;
    gap: var(--space-sm);
    margin-top: var(--space-md);
  }
  /* Inline citation chips are emitted as raw HTML strings inside {@html}
     (see renderChip), so they sit outside Svelte's scoping and must be styled
     :global. Padding/size use em so the chip tracks the surrounding prose. */
  :global(.cite-chip) {
    display: inline-block;
    vertical-align: baseline;
    max-width: 20ch;
    padding: 0 0.4em;
    margin: 0 0.15em;
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    border-radius: var(--radius-sm);
    font-family: var(--font-sans);
    font-size: 0.8em;
    line-height: 1.4;
    color: var(--color-off);
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  :global(.cite-chip:hover),
  :global(.cite-chip.active) {
    background: var(--color-off);
    color: var(--color-surface);
    border-color: var(--color-off);
  }
  /* Tombstone for a [^N] whose cited source is gone. Raw {@html}, hence :global.
     Muted + italic so it reads as an absence, not a live citation chip. */
  :global(.cite-gone) {
    font-style: italic;
    font-size: 0.85em;
    color: var(--color-token-dark);
    cursor: help;
  }
  /* External (URL / external-dataset doc) citations. Distinct from the corpus
     chunk chips: a dashed border + accent tint signals "not from the corpus."
     Raw {@html}, hence :global. */
  :global(.cite-external) {
    display: inline-block;
    vertical-align: baseline;
    max-width: 24ch;
    padding: 0 0.4em;
    margin: 0 0.15em;
    border: 1px dashed var(--color-link);
    border-radius: var(--radius-sm);
    font-family: var(--font-sans);
    font-size: 0.8em;
    line-height: 1.4;
    color: var(--color-link);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  :global(a.cite-external--url) {
    text-decoration: none;
    cursor: pointer;
  }
  :global(a.cite-external--url:hover) {
    background: var(--color-link);
    color: var(--color-surface);
  }
  :global(.cite-external--doc) {
    font-style: italic;
    cursor: help;
  }
  /* The chip and its jump button travel together as one inline unit, so a line
     break never splits them. Also raw {@html}, hence :global. */
  :global(.cite) {
    white-space: nowrap;
  }
  :global(.cite-jump) {
    display: inline-flex;
    align-items: center;
    vertical-align: baseline;
    margin: 0 0.15em 0 0.1em;
    padding: 0;
    background: none;
    border: none;
    color: var(--color-off);
    cursor: pointer;
  }
  :global(.cite-jump:hover) {
    color: var(--color-link);
  }
  :global(.cite-jump .icon) {
    position: relative;
    top: 1px;
  }
</style>
