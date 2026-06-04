<script>
  import { navigating, page } from "$app/stores";
  import SearchForm from "$lib/components/SearchForm.svelte";
  import SearchResult from "$lib/components/SearchResult.svelte";
  import ScanMatch from "$lib/components/ScanMatch.svelte";
  import Pagination from "$lib/components/Pagination.svelte";
  import { pluralize } from "$lib/format.js";

  export let data;

  $: ({ params, available, result, error } = data);
  // Any in-flight navigation here is a search/scan submit — show progress
  // while the subprocess (and, for semantic search, the BGE model) runs.
  $: busy = !!$navigating;

  // Preserve the current query while only moving the scan offset.
  function scanHref(offset) {
    const sp = new URLSearchParams($page.url.searchParams);
    sp.set("offset", String(offset));
    return `/search?${sp.toString()}`;
  }
</script>

<h1>Search</h1>

<SearchForm {params} availableTags={available.tags} />

{#if busy}
  <div class="status busy" aria-live="polite">
    Searching… <span class="hint">semantic queries load the embedding model and can take a few seconds.</span>
  </div>
{/if}

<div class="results" class:dim={busy}>
  {#if error}
    <div class="status error">
      <strong>Search failed.</strong> {error.message}
      {#if error.code}<span class="code">({error.code})</span>{/if}
    </div>
  {:else if !params.q}
    <p class="empty">Enter a query to search the corpus.</p>
  {:else if params.mode === "scan"}
    {#if result.matches.length === 0}
      <p class="empty">No chunks match “{result.query}”.</p>
    {:else}
      <p class="summary">
        {pluralize(result.total, "match", "matches")} for the
        {result.match_mode} “{result.query}”.
      </p>
      <Pagination
        offset={result.offset}
        limit={result.limit}
        total={result.total}
        buildHref={scanHref}
      />
      <ul class="list">
        {#each result.matches as m (m.chunk_id)}
          <ScanMatch match={m} />
        {/each}
      </ul>
      <Pagination
        offset={result.offset}
        limit={result.limit}
        total={result.total}
        buildHref={scanHref}
      />
    {/if}
  {:else}
    {#if result.results.length === 0}
      <p class="empty">No results for “{result.query}”.</p>
    {:else}
      <p class="summary">
        {pluralize(result.results.length, "result")}
        for “{result.query}” across {result.source_kinds.join(", ")}.
        {#if result.memory_excluded}<span class="note">Findings excluded (memory off).</span>{/if}
      </p>
      <ul class="list">
        {#each result.results as h (h.chunk_id)}
          <SearchResult hit={h} />
        {/each}
      </ul>
    {/if}
  {/if}
</div>

<style>
  .results {
    max-width: 48rem;
    transition: opacity 0.15s;
  }
  .results.dim {
    opacity: 0.45;
  }
  .list {
    padding: 0;
    margin: 0;
  }
  .summary {
    font-family: var(--font-sans);
    font-size: 0.85rem;
    color: var(--color-off);
    margin-bottom: 1rem;
  }
  .note,
  .code {
    color: var(--color-token-dark);
  }
  .status {
    font-family: var(--font-sans);
    font-size: 0.9rem;
    padding: 0.6rem 0.85rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    max-width: 48rem;
  }
  .status.busy {
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    color: var(--color-off);
  }
  .status.busy .hint {
    color: var(--color-off);
    opacity: 0.75;
    font-size: 0.8rem;
  }
  .status.error {
    background: #fdecea;
    border: 1px solid #e0a9a3;
    color: #8a2b22;
  }
  .empty {
    color: var(--color-off);
    font-style: italic;
  }
</style>
