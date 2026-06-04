<script>
  import ChunkExpander from "./ChunkExpander.svelte";

  export let hit;

  // Where the result's *title* links, when there's a natural detail page.
  $: detailHref =
    hit.source_kind === "document" ? `/documents/${hit.source_id}`
    : hit.source_kind === "finding" ? `/findings/${hit.source_id}`
    : null;

  // Deep link to the archived source file at the cited page. Only document
  // chunks carry a usable document_id (it *is* source_id) here; summaries and
  // images don't expose theirs in the search output, so they get no file link.
  $: fileHref =
    hit.source_kind === "document"
      ? `/files/${hit.source_id}${hit.page_number ? `#page=${hit.page_number}` : ""}`
      : null;

  $: scorePct = Math.round((hit.normalized_score ?? 0) * 100);

  function stripExt(name) {
    return name ? name.replace(/\.[^.]+$/, "") : name;
  }
</script>

<li class="result">
  <div class="head">
    <span class="rank">{hit.rank}</span>
    <h3 class="title">
      {#if detailHref}
        <a href={detailHref}>{hit.source_name}</a>
      {:else}
        {hit.source_name}
      {/if}
    </h3>
    <span class="score" title="normalized score (top hit = 100)">{scorePct}</span>
  </div>

  <p class="meta">
    <span class="kind">{hit.source_kind}</span>
    {#if hit.file_name && hit.file_name !== hit.source_name} · {hit.file_name}{/if}
    {#if hit.page_number} · p.{hit.page_number}{/if}
    {#if hit.section_heading} · {hit.section_heading}{/if}
    {#if hit.content_type} · {hit.content_type}{/if}
    · chunk {hit.chunk_id}
  </p>

  {#if hit.context_before?.length}
    <p class="ctx before">…{hit.context_before.map((c) => c.text).join(" ")}</p>
  {/if}
  <p class="snippet">{hit.text}</p>
  {#if hit.context_after?.length}
    <p class="ctx after">{hit.context_after.map((c) => c.text).join(" ")}…</p>
  {/if}

  <div class="actions">
    {#if fileHref}
      <a class="open" href={fileHref} target="_blank" rel="noopener noreferrer">
        Open source{hit.page_number ? ` p.${hit.page_number}` : ""}
      </a>
    {/if}
    {#if hit.image_file_path}
      <span class="img-path" title={hit.image_file_path}>image: {stripExt(hit.file_name) ?? "embedded"}</span>
    {/if}
    <ChunkExpander chunkId={hit.chunk_id} />
  </div>
</li>

<style>
  .result {
    list-style: none;
    padding: 0.9rem 1rem;
    margin-bottom: 1rem;
    background: var(--color-off-light);
    border: 1px solid var(--color-off);
    border-radius: 4px;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
  }
  .rank {
    font-family: var(--font-display);
    color: var(--color-off);
    font-size: 0.9rem;
  }
  .title {
    flex: 1;
    font-size: 1.05rem;
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
  .score {
    font-family: var(--font-display);
    font-size: 0.85rem;
    color: var(--color-off);
    background: #fff;
    border: 1px solid var(--color-off);
    border-radius: 3px;
    padding: 0 0.4rem;
  }
  .meta {
    font-family: var(--font-sans);
    font-size: 0.78rem;
    color: var(--color-off);
    margin-top: 0.2rem;
  }
  .kind {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600;
  }
  .snippet {
    margin-top: 0.5rem;
    font-family: var(--font-serif);
    line-height: 1.5;
  }
  .ctx {
    font-family: var(--font-serif);
    font-size: 0.85rem;
    color: #777;
    font-style: italic;
  }
  .ctx.before {
    margin-top: 0.5rem;
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
  .img-path {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    color: var(--color-off);
  }
</style>
