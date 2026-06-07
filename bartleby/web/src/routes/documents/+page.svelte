<script>
  import { page } from "$app/stores";
  import DocumentList from "$lib/components/DocumentList.svelte";
  import Pagination from "$lib/components/Pagination.svelte";
  import StatusBanner from "$lib/components/StatusBanner.svelte";
  import Button from "$lib/components/Button.svelte";
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
    <Button type="submit">Apply</Button>
  </fieldset>
</form>

{#if error}
  <StatusBanner variant="error">
    <strong>Could not list documents.</strong> {error.message}
    {#if error.code}<span class="code">({error.code})</span>{/if}
  </StatusBanner>
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
