<script>
  import { marked } from "marked";
  export let data;
  $: bodyHtml = marked.parse(data.finding.body);

  // Currently-shown citation in the right pane. Initially nothing.
  let active = null;

  function citationUrl(c) {
    if (c.document_id == null) return null;
    const page = c.page_number ? `#page=${c.page_number}` : "";
    return `/files/${c.document_id}${page}`;
  }

  function citationLabel(c) {
    const parts = [];
    if (c.file_name) parts.push(c.file_name);
    if (c.page_number) parts.push(`p.${c.page_number}`);
    parts.push(`chunk ${c.chunk_id}`);
    return parts.join(" · ");
  }

  function showCitation(c) {
    if (c.document_id == null) return;
    active = c;
  }

  $: activeUrl = active ? citationUrl(active) : null;

  function isActive(c) {
    return (
      active != null &&
      c.chunk_id === active.chunk_id &&
      c.document_id === active.document_id &&
      c.page_number === active.page_number
    );
  }
</script>

<div class="split">
  <article class="report">
    <h1>{data.finding.title}</h1>
    <p class="meta">
      {data.finding.session_name} · {data.finding.created_at}
    </p>
    <p class="desc">{data.finding.description}</p>

    <div class="body">
      {@html bodyHtml}
    </div>

    <h2>Citations</h2>
    {#if data.finding.citations.length === 0}
      <p>No citations.</p>
    {:else}
      <ul class="citations">
        {#each data.finding.citations as c}
          <li class:active={isActive(c)}>
            {#if citationUrl(c)}
              <button
                type="button"
                class="cite-link"
                on:click={() => showCitation(c)}
              >
                {citationLabel(c)}
              </button>
            {:else}
              <span>{citationLabel(c)}</span>
            {/if}
            <div class="kind">{c.source_kind}</div>
            <pre class="snippet">{c.text}</pre>
          </li>
        {/each}
      </ul>
    {/if}
  </article>

  <aside class="viewer">
    {#if activeUrl}
      {#key activeUrl}
        <iframe title="Source document" src={activeUrl}></iframe>
      {/key}
    {:else}
      <p class="placeholder">Click a citation to view the source PDF here.</p>
    {/if}
  </aside>
</div>

<style>
  .split {
    display: grid;
    grid-template-columns: minmax(0, 40rem) minmax(0, 1fr);
    gap: 1.5rem;
    align-items: start;
  }
  .report {
    min-width: 0;
  }
  .viewer {
    position: sticky;
    top: 1rem;
    height: calc(100vh - 2rem);
  }
  .viewer iframe {
    width: 100%;
    height: 100%;
    border: 1px solid #ccc;
  }
  .placeholder {
    color: var(--color-off);
    font-style: italic;
  }
  .desc {
    font-style: italic;
  }
  .body {
    margin: 1.5rem 0;
  }
  :global(.body p) {
    margin-bottom: 1.5rem;
  }
  :global(.body h1) {
    font-size: 1.5rem;
    font-weight: 900;
  }
  :global(.body h2) {
    font-size: 1.5rem;
    font-weight: 200;
  }
  :global(.body ul),
  :global(.body ol) {
    margin-left: 1rem;
    margin-bottom: 1.5rem;
  }
  .citations {
    list-style: none;
    padding: 0;
  }
  .citations li {
    margin-bottom: 1rem;
    padding: 0.5rem;
    border-left: 2px solid #ccc;
  }
  .citations li.active {
    border-left-color: var(--color-off);
  }
  .cite-link {
    background: none;
    border: 0;
    padding: 0;
    font: inherit;
    color: inherit;
    text-decoration: underline;
    cursor: pointer;
  }
  .kind {
    font-size: 0.75rem;
    color: #888;
  }
  .snippet {
    margin: 0.25rem 0 0;
    padding: 0.5rem;
    background: #f6f6f6;
    white-space: pre-wrap;
    font-size: 0.85rem;
  }
</style>
