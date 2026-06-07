<script>
  export let offset;
  export let limit;
  export let total;
  export let buildHref; // (offset:number) => string

  $: from = total === 0 ? 0 : offset + 1;
  $: to = Math.min(offset + limit, total);
  $: hasPrev = offset > 0;
  $: hasNext = offset + limit < total;
  $: prevOffset = Math.max(0, offset - limit);
  $: nextOffset = offset + limit;
</script>

<nav class="pager">
  {#if hasPrev}
    <a href={buildHref(prevOffset)}>← Prev</a>
  {:else}
    <span class="disabled">← Prev</span>
  {/if}

  <span class="range">{from}–{to} of {total}</span>

  {#if hasNext}
    <a href={buildHref(nextOffset)}>Next →</a>
  {:else}
    <span class="disabled">Next →</span>
  {/if}
</nav>

<style>
  .pager {
    display: flex;
    align-items: center;
    gap: var(--space-lg);
    margin: var(--space-lg) 0;
    font-family: var(--font-sans);
    font-size: var(--text-sm);
  }
  .pager a {
    color: var(--color-token-dark);
    font-weight: 600;
    text-decoration: none;
  }
  .pager a:hover {
    text-decoration: underline;
  }
  .disabled {
    color: var(--color-rule);
  }
  .range {
    color: var(--color-off);
  }
</style>
