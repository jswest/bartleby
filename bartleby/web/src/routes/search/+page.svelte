<script>
  import { navigating, page } from "$app/stores";
  import SearchForm from "$lib/components/SearchForm.svelte";
  import ResultCard from "$lib/components/ResultCard.svelte";
  import StatusBanner from "$lib/components/StatusBanner.svelte";
  import Pagination from "$lib/components/Pagination.svelte";
  import WaitingIndicator from "$lib/components/WaitingIndicator.svelte";
  import { pluralize } from "$lib/format.js";
  import { isSearchNavigation } from "$lib/navigation.js";

  export let data;

  $: ({ params, available, result, error } = data);
  // Show "Searching…" only for a genuine search/scan submission. Clicking a
  // result or pressing Back also triggers $navigating while on this page, but
  // those go elsewhere and are owned by the global layout "Loading…" indicator.
  // The shared isSearchNavigation predicate keeps the two in lockstep.
  $: busy = isSearchNavigation($navigating);

  // Preserve the current query while only moving the scan offset.
  function scanHref(offset) {
    const sp = new URLSearchParams($page.url.searchParams);
    sp.set("offset", String(offset));
    return `/search?${sp.toString()}`;
  }
</script>

<div class="search">
<h1>Search</h1>

<SearchForm {params} availableTags={available.tags} />

{#if busy}
  <StatusBanner variant="busy">
    <WaitingIndicator label="Searching…" active={true} />
    <span class="hint">semantic queries load the embedding model and can take a few seconds.</span>
  </StatusBanner>
{/if}

<div class="results" class:dim={busy}>
  {#if error}
    <StatusBanner variant="error">
      <strong>Search failed.</strong> {error.message}
      {#if error.code}<span class="code">({error.code})</span>{/if}
    </StatusBanner>
  {:else if !params.q}
    <p class="empty empty--terminal">
      <WaitingIndicator label="awaiting query" active={false} />
    </p>
  {:else if params.mode === "scan"}
    {#if result.matches.length === 0}
      <p class="empty empty--terminal">no chunks match “{result.query}”</p>
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
          <ResultCard item={m} variant="scan" query={result.query} />
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
      <p class="empty empty--terminal">no results for “{result.query}”</p>
    {:else}
      <p class="summary">
        {pluralize(result.results.length, "result")}
        for “{result.query}” across {result.source_kinds.join(", ")}.
        {#if result.memory_excluded}<span class="note-inline">Findings excluded (memory off).</span>{/if}
      </p>
      <ul class="list">
        {#each result.results as h (h.chunk_id)}
          <ResultCard item={h} variant="search" query={result.query} />
        {/each}
      </ul>
    {/if}
  {/if}
</div>
</div>

<style>
  .search {
    max-width: var(--width-content);
    margin-inline: auto;
  }
  .results {
    transition: opacity 0.15s;
  }
  .results.dim {
    opacity: 0.45;
  }
  .list {
    padding: 0;
    margin: 0;
  }
  /* Inline amber-text accent inside the result summary. */
  .note-inline {
    color: var(--color-token-text);
  }
  /* The secondary line inside the busy banner (slot content is styled here,
     by the caller's scope, not inside StatusBanner). */
  .hint {
    color: var(--color-off);
    opacity: 0.75;
    font-size: var(--text-xs);
  }
</style>
