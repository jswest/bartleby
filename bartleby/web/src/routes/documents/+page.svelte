<script>
  import { page } from "$app/stores";
  import DocumentList from "$lib/components/DocumentList.svelte";
  import Pagination from "$lib/components/Pagination.svelte";
  import { pluralize } from "$lib/format.js";

  export let data;

  $: ({ params, availableTags, documents, total, excludedNullDated, error } = data);
  $: tagChecked = (name) => params.tags.includes(name);

  // Preserve all active filters while only moving the page offset.
  function pageHref(offset) {
    const sp = new URLSearchParams($page.url.searchParams);
    sp.set("offset", String(offset));
    return `/documents?${sp.toString()}`;
  }

  // One-click "show the undated docs the date bound is hiding" — flips the
  // include-nulls flag and returns to page 1.
  $: showUndatedHref = (() => {
    const sp = new URLSearchParams($page.url.searchParams);
    sp.set("nulls", "1");
    sp.delete("offset");
    return `/documents?${sp.toString()}`;
  })();
</script>

<h1>Documents</h1>

<form class="filters" method="GET" action="/documents">
  <fieldset>
    <legend>Dates</legend>
    <label class="inline">
      after <input type="date" name="after" value={params.after} />
    </label>
    <label class="inline">
      before <input type="date" name="before" value={params.before} />
    </label>
    <label class="check">
      <input type="checkbox" name="nulls" value="1" checked={params.includeNulls} />
      include undated
    </label>
  </fieldset>

  {#if availableTags.length}
    <fieldset>
      <legend>Tags</legend>
      <div class="checks">
        {#each availableTags as t}
          <label class="check">
            <input type="checkbox" name="tag" value={t.name} checked={tagChecked(t.name)} />
            {t.name} <span class="tag-count">{t.document_count}</span>
          </label>
        {/each}
      </div>
    </fieldset>
  {/if}

  <fieldset>
    <legend>Order</legend>
    <label class="inline">
      <select name="sort">
        <option value="title" selected={params.sort === "title"}>title (A–Z)</option>
        <option value="date" selected={params.sort === "date"}>newest first</option>
        <option value="id" selected={params.sort === "id"}>ingest order</option>
      </select>
    </label>
    <label class="inline">
      per page
      <input type="number" name="limit" min="1" max="500" value={params.limit} />
    </label>
    <button type="submit">Apply</button>
  </fieldset>
</form>

{#if error}
  <div class="status error">
    <strong>Could not list documents.</strong> {error.message}
    {#if error.code}<span class="code">({error.code})</span>{/if}
  </div>
{:else}
  {#if excludedNullDated > 0}
    <p class="note">
      {pluralize(excludedNullDated, "undated document")} hidden by the date filter.
      <a href={showUndatedHref}>Show {excludedNullDated === 1 ? "it" : "them"}</a>.
    </p>
  {/if}

  {#if documents.length === 0}
    <p class="empty">No documents match these filters.</p>
  {:else}
    <p class="summary">{pluralize(total, "document")}.</p>
    <Pagination offset={params.offset} limit={params.limit} {total} buildHref={pageHref} />
    <DocumentList {documents} />
    <Pagination offset={params.offset} limit={params.limit} {total} buildHref={pageHref} />
  {/if}
{/if}

<style>
  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin: 1rem 0 1.5rem;
    max-width: 48rem;
    font-family: var(--font-sans);
  }
  fieldset {
    border: 1px solid var(--color-rule);
    border-radius: 4px;
    padding: 0.5rem 0.75rem;
  }
  legend {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-off);
    padding: 0 0.35rem;
  }
  .checks {
    display: flex;
    flex-wrap: wrap;
    gap: 0.25rem 0.9rem;
    max-width: 28rem;
  }
  .check,
  .inline {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.85rem;
    color: #333;
  }
  .inline + .inline,
  .inline + button,
  .check + .check {
    margin-left: 0.9rem;
  }
  .tag-count {
    color: var(--color-off);
    font-size: 0.75rem;
  }
  input[type="date"],
  select {
    padding: 0.2rem 0.3rem;
    border: 1px solid var(--color-rule);
    border-radius: 3px;
    font-family: var(--font-sans);
    font-size: 0.85rem;
  }
  input[type="number"] {
    width: 4rem;
    padding: 0.2rem 0.3rem;
    border: 1px solid var(--color-rule);
    border-radius: 3px;
    font-family: var(--font-sans);
  }
  button[type="submit"] {
    padding: 0.3rem 1rem;
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    border-radius: 4px;
    color: var(--color-off);
    font-family: var(--font-sans);
    font-weight: 600;
    cursor: pointer;
  }
  button[type="submit"]:hover {
    background: var(--color-off);
    color: #fff;
    border-color: var(--color-off);
  }
  .summary {
    font-family: var(--font-sans);
    font-size: 0.85rem;
    color: var(--color-off);
    margin-bottom: 0.5rem;
  }
  .note {
    font-family: var(--font-sans);
    font-size: 0.85rem;
    color: var(--color-token-dark);
    margin-bottom: 1rem;
  }
  .note a {
    color: var(--color-token-dark);
  }
  .status.error {
    font-family: var(--font-sans);
    font-size: 0.9rem;
    padding: 0.6rem 0.85rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    max-width: 48rem;
    background: #fdecea;
    border: 1px solid #e0a9a3;
    color: #8a2b22;
  }
  .code {
    color: var(--color-token-dark);
  }
  .empty {
    color: var(--color-off);
    font-style: italic;
  }
</style>
