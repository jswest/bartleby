<script>
  export let data;
</script>

<h1>Findings</h1>

{#if data.findings.length === 0}
  <p class="empty">No findings yet.</p>
{:else}
  <ul class="entity-list">
    {#each data.findings as f}
      <li class="entity">
        <h2><a href="/findings/{f.finding_id}">{f.title}</a></h2>
        <p class="meta">
          {f.session_name} · {f.citation_count} citation{f.citation_count === 1
            ? ""
            : "s"} · {f.created_at}
        </p>
        <p class="desc">{f.description}</p>
      </li>
    {/each}
  </ul>
{/if}