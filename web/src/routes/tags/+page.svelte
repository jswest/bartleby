<script>
  import { pluralize } from "$lib/format.js";

  export let data;
</script>

<h1>Tags</h1>

{#if data.tags.length === 0}
  <p class="empty">No tags created yet.</p>
{:else}
  <ul class="entity-list">
    {#each data.tags as t}
      <li class="entity">
        <h2><a href="/tags/{t.tag_id}">{t.name}</a></h2>
        <p class="meta">
          {pluralize(t.document_count, "document")}
        </p>
        {#if t.description}
          <p class="desc">{t.description}</p>
        {/if}
      </li>
    {/each}
  </ul>
{/if}
