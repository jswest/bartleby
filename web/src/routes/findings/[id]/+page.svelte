<script>
  import { onMount } from "svelte";
  import { marked } from "marked";
  export let data;

  $: byId = new Map(data.finding.citations.map((c) => [c.chunk_id, c]));

  // Substitute each [^N] marker in the markdown source with an inline <button>
  // before marked parses. Marked passes inline HTML through, so the chip ends
  // up inside the same <p> as the surrounding prose — preserves reading flow.
  // Reactive on `active` so the .active class is baked into the HTML on click.
  $: bodyHtml = marked.parse(
    data.finding.body.replace(/\[\^(\d+)\]/g, (match, idStr) => {
      const c = byId.get(Number(idStr));
      return c ? renderChip(c, c === active) : match;
    }),
  );

  function renderChip(c, isActive) {
    const name = c.file_name ? c.file_name.replace(/\.pdf$/i, "") : null;
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
    return `<button type="button" class="${cls}" data-chunk-id="${c.chunk_id}" title="${esc(title)}">${esc(label)}</button>`;
  }

  function esc(s) {
    return s.replace(/[&<>"]/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
    ));
  }

  let active = null;
  $: activeUrl = active ? citationUrl(active) : null;

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
  <article class="report" bind:this={container}>
    <h1>{data.finding.title}</h1>
    <p class="meta">
      <span class="finding-id">#{data.finding.finding_id}</span> · {data.finding.session_name} · {data.finding.created_at}
    </p>
    <p class="desc">{data.finding.description}</p>

    <div class="toolbar">
      <button type="button" on:click={copyMarkdown}>
        {copied ? "Copied" : "Copy as Markdown"}
      </button>
      <button type="button" on:click={downloadMarkdown}>Download .md</button>
    </div>

    <div class="body markdown-body">
      {@html bodyHtml}
    </div>
  </article>

  <aside class="viewer">
    {#if activeUrl}
      {#key activeUrl}
        <iframe title="Source document" src={activeUrl}></iframe>
      {/key}
    {:else}
      <p class="placeholder">
        Click an inline citation to view the source PDF here.
      </p>
    {/if}
  </aside>
</div>

<style>
  article.report {
    background-color: var(--color-off-light);
    border: 1px solid var(--color-off);
    box-sizing: border-box;
    padding: 1rem;
  }
  .toolbar {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  .toolbar button {
    font-family: var(--font-sans);
    font-size: 0.8rem;
    padding: 0.3rem 0.7rem;
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    border-radius: 3px;
    color: var(--color-off);
    cursor: pointer;
  }
  .toolbar button:hover {
    background: var(--color-off);
    color: #fff;
    border-color: var(--color-off);
  }
  .placeholder {
    color: var(--color-off);
    font-style: italic;
  }
  :global(.cite-chip) {
    display: inline-block;
    vertical-align: baseline;
    max-width: 20ch;
    padding: 0 0.4em;
    margin: 0 0.15em;
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    border-radius: 3px;
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
    color: #fff;
    border-color: var(--color-off);
  }
</style>
