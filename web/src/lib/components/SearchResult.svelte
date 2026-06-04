<script>
  import { stripExt } from "$lib/format.js";

  export let hit;

  // Enriched server-side: hit.title (summary/finding title), hit.description,
  // hit.file_name (the originating file, shown as a subtitle), hit.href (the
  // document detail page at the cited page, or the finding page).
  $: title = hit.title ?? stripExt(hit.file_name) ?? hit.source_name;
  $: scorePct = Math.round((hit.normalized_score ?? 0) * 100);
</script>

<li class="result">
  <div class="head">
    <h3 class="title">
      {#if hit.href}
        <a href={hit.href}>
          {title}{#if hit.page_number}<span class="page-hint"> p.{hit.page_number}</span>{/if}
        </a>
      {:else}
        {title}
      {/if}
    </h3>
    <span class="score" title="normalized score (top hit = 100)">{scorePct}</span>
  </div>

  {#if hit.file_name}
    <p class="filename">{hit.file_name}</p>
  {/if}
  {#if hit.description}
    <p class="desc">{hit.description}</p>
  {/if}

  <p class="meta">
    <span class="kind">{hit.source_kind}</span>
    {#if hit.section_heading} · {hit.section_heading}{/if}
    {#if hit.content_type} · {hit.content_type}{/if}
    · chunk {hit.chunk_id}
  </p>

  <p class="snippet">{hit.text}</p>
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
  }
  .title .page-hint {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-token-dark);
    white-space: nowrap;
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
  .filename {
    font-family: var(--font-sans);
    font-size: 0.78rem;
    color: var(--color-off);
    margin-top: 0.15rem;
  }
  .desc {
    font-family: var(--font-serif);
    font-size: 0.9rem;
    color: #444;
    margin-top: 0.3rem;
  }
  .meta {
    font-family: var(--font-sans);
    font-size: 0.72rem;
    color: var(--color-off);
    margin-top: 0.4rem;
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
</style>
