<script>
  import { marked } from "marked";
  import { stripExt, isMarkdownChunk } from "$lib/format.js";
  export let data;

  $: chunk = data.chunk;
  // The source label: a summary/finding title if we have one, else the file name
  // sans extension, else a bare "<kind> #<id>" so there's always something.
  $: sourceLabel =
    chunk.title ?? stripExt(chunk.file_name) ?? `${chunk.source_kind} #${chunk.source_id}`;

  // Render a chunk's text as markdown when it's markdown-authored, else literal —
  // same rule the search cards use (file_name/source_kind are shared across the
  // window; content_type varies per chunk, e.g. a lone sec_table table).
  function html(c) {
    return isMarkdownChunk({
      source_kind: chunk.source_kind,
      content_type: c.content_type,
      file_name: chunk.file_name,
    })
      ? marked.parse(c.text)
      : null;
  }
</script>

<article class="report surface">
  <p class="meta">
    {#if chunk.back_href}
      <a href={chunk.back_href}>← {sourceLabel}</a>
    {:else}
      {sourceLabel}
    {/if}
  </p>
  <h1>Chunk {chunk.chunk_id}</h1>
  <p class="meta">
    <span class="kind">{chunk.source_kind}</span>
    {#if chunk.section_heading} · {chunk.section_heading}{/if}
    {#if chunk.page_number} · p.{chunk.page_number}{/if}
    {#if chunk.content_type} · {chunk.content_type}{/if}
  </p>

  <div class="stream">
    {#each chunk.neighbors as c (c.chunk_id)}
      {@const rendered = html(c)}
      <section class="chunk" class:target={c.is_target}>
        <p class="chunk-label">
          {#if c.is_target}
            chunk {c.chunk_id}
          {:else}
            <a href="/chunks/{c.chunk_id}">chunk {c.chunk_id}</a>
          {/if}
          {#if c.section_heading} · {c.section_heading}{/if}
          {#if c.page_number} · p.{c.page_number}{/if}
        </p>
        {#if rendered}
          <div class="text markdown-body">{@html rendered}</div>
        {:else}
          <p class="text">{c.text}</p>
        {/if}
      </section>
    {/each}
  </div>
</article>

<style>
  /* A single reading column — the chunk view is text, no PDF viewer. */
  .report {
    max-width: 48rem;
    margin: 0 auto;
  }
  .stream {
    margin-top: var(--space-2xl);
    display: flex;
    flex-direction: column;
    gap: var(--space-xl);
  }
  /* Context chunks are muted (gray); the target rides a token-amber rail and
     reads at full text contrast so the eye lands on it. */
  .chunk {
    padding-left: var(--space-lg);
    border-left: 3px solid transparent;
    color: var(--color-off);
  }
  .chunk.target {
    border-left-color: var(--color-token-dark);
    color: var(--color-text);
  }
  .chunk-label {
    font-family: var(--font-sans);
    font-size: var(--text-xs);
    color: var(--color-off);
    margin-bottom: var(--space-xs);
  }
  .text {
    font-family: var(--font-serif);
    line-height: 1.6;
    /* Inherit the chunk's muted/full color set above. */
    color: inherit;
  }
</style>
