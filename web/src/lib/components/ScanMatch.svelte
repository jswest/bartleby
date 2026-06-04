<script>
  export let match;

  function stripExt(name) {
    return name ? name.replace(/\.[^.]+$/, "") : name;
  }

  // Enriched server-side: match.title (summary title), match.description,
  // match.href (the document detail page at the cited page). Scan is documents-only.
  $: title = match.title ?? stripExt(match.file_name);
  $: truncated = match.text_length > match.text.length;
</script>

<li class="match">
  <h3 class="title">
    <a href={match.href}>
      {title}{#if match.page_number}<span class="page-hint"> p.{match.page_number}</span>{/if}
    </a>
  </h3>

  <p class="filename">{match.file_name}</p>
  {#if match.description}
    <p class="desc">{match.description}</p>
  {/if}

  <p class="meta">
    {#if match.section_heading}{match.section_heading} · {/if}
    {#if match.content_type}{match.content_type} · {/if}
    chunk {match.chunk_id} · {match.text_length} chars
  </p>

  <p class="snippet">{match.text}{#if truncated}<span class="trunc">…</span>{/if}</p>
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
  }
  .title .page-hint {
    font-family: var(--font-sans);
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-token-dark);
    white-space: nowrap;
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
  .snippet {
    margin-top: 0.5rem;
    font-family: var(--font-serif);
    line-height: 1.5;
  }
  .trunc {
    color: var(--color-off);
  }
</style>
