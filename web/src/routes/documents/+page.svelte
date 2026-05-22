<script>
  export let data;

  // Strip a trailing .pdf so the file name reads as a title when no summary exists.
  function stripExt(name) {
    return name.replace(/\.[^.]+$/, "");
  }
</script>

<h1>Documents</h1>

{#if data.documents.length === 0}
  <p class="empty">No documents ingested yet.</p>
{:else}
  <ul class="entity-list">
    {#each data.documents as d}
      <li>
        <h2>
          <a href="/documents/{d.document_id}">
            {d.title ?? stripExt(d.file_name)}
          </a>
        </h2>
        <p class="meta">
          {d.file_name}{d.page_count ? ` · ${d.page_count} page${d.page_count === 1 ? "" : "s"}` : ""}
          {#if !d.title} · <em>no summary</em>{/if}
        </p>
        {#if d.description}
          <p class="desc">{d.description}</p>
        {/if}
      </li>
    {/each}
  </ul>
{/if}
