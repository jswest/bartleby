<script>
  import ChunkExpander from "./ChunkExpander.svelte";

  export let match;

  // Scan is documents-only and exposes document_id directly, so both links
  // always resolve.
  $: fileHref = `/files/${match.document_id}${match.page_number ? `#page=${match.page_number}` : ""}`;
  $: truncated = match.text_length > match.text.length;
</script>

<li class="match">
  <div class="head">
    <h3 class="title"><a href={`/documents/${match.document_id}`}>{match.file_name}</a></h3>
  </div>

  <p class="meta">
    {#if match.page_number}p.{match.page_number} · {/if}
    {#if match.section_heading}{match.section_heading} · {/if}
    {#if match.content_type}{match.content_type} · {/if}
    chunk {match.chunk_id} · {match.text_length} chars
  </p>

  <p class="snippet">{match.text}{#if truncated}<span class="trunc">…</span>{/if}</p>

  <div class="actions">
    <a class="open" href={fileHref} target="_blank" rel="noopener noreferrer">
      Open source{match.page_number ? ` p.${match.page_number}` : ""}
    </a>
    {#if truncated}
      <ChunkExpander chunkId={match.chunk_id} />
    {/if}
  </div>
</li>

<style>
  .match {
    list-style: none;
    padding: 0.8rem 1rem;
    margin-bottom: 0.85rem;
    background: var(--color-off-light);
    border: 1px solid var(--color-off);
    border-radius: 4px;
  }
  .title {
    font-size: 1rem;
    font-weight: 600;
    min-width: 0;
  }
  .title a {
    color: inherit;
    text-decoration: none;
  }
  .title a:hover {
    color: var(--color-off);
    text-decoration: underline;
  }
  .meta {
    font-family: var(--font-sans);
    font-size: 0.78rem;
    color: var(--color-off);
    margin-top: 0.2rem;
  }
  .snippet {
    margin-top: 0.5rem;
    font-family: var(--font-serif);
    line-height: 1.5;
  }
  .trunc {
    color: var(--color-off);
  }
  .actions {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 0.6rem;
  }
  .open {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-token-dark);
    text-decoration: none;
  }
  .open:hover {
    text-decoration: underline;
  }
</style>
