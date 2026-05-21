<script>
  export let data;
</script>

<h1>Findings</h1>

{#if data.findings.length === 0}
  <p>No findings yet.</p>
{:else}
  <ul class="findings">
    {#each data.findings as f}
      <li>
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

<style>
  ul.findings {
    max-width: 40rem;
    list-style: none;
    padding: 0;
  }
  .findings h2 {
    font-size: 1rem;
  }
  .findings li {
    margin-bottom: 1.25rem;
  }
  .findings a {
    font-size: 1rem;
  }
  .desc {
    margin: 0.25rem 0 0;
  }
</style>
