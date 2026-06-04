<script>
  import { stripExt } from "$lib/format.js";
  // Shared document listing — used by /documents and /tags/[id]. Each document
  // carries a `tags` array (possibly empty); untagged docs render no chips.
  export let documents;
</script>

<ul class="entity-list">
  {#each documents as d}
    <li class="entity">
      <h2>
        <a href="/documents/{d.document_id}">
          {d.title ?? stripExt(d.file_name)}
        </a>
      </h2>
      <p class="meta">
        {d.file_name}{d.page_count ? ` · ${d.page_count} page${d.page_count === 1 ? "" : "s"}` : ""}
        {#if !d.title} · <em>no summary</em>{/if}
      </p>
      {#if d.tags.length}
        <div class="tags">
          {#each d.tags as t}
            <span class="tag" title={t.description}>{t.name}</span>
          {/each}
        </div>
      {/if}
      {#if d.description}
        <p class="desc">{d.description}</p>
      {/if}
    </li>
  {/each}
</ul>
