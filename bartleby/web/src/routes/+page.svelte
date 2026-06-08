<script>
  import { pluralize, stripExt } from "$lib/format.js";
  import StatusBanner from "$lib/components/StatusBanner.svelte";
  export let data;

  $: ({ project, counts, corpus, error } = data);

  const fmt = (n) => (n ?? 0).toLocaleString("en-US");

  // Tallest bar in the year histogram, floored at 1 so an empty corpus doesn't
  // divide by zero.
  $: maxYear = corpus
    ? Math.max(1, ...corpus.documents_by_year.map((y) => y.document_count))
    : 1;
</script>

<section class="home">
  <h1 class="display">Bartleby</h1>
  <p class="tagline">
    The agent's notebook for <span class="project">{project}</span>.
  </p>

  <!-- The two cards teach the app's colour signal up front: the mint card is
       agent output (findings), the warm card is source material (documents) —
       the same split that holds across every list and detail page. -->
  <ul class="cards">
    <li>
      <a class="surface surface--finding surface--interactive" href="/findings">
        <h2>Findings</h2>
        <p class="count">{fmt(counts.findings)}</p>
        <p class="hint">Saved research notes with inline citations.</p>
      </a>
    </li>
    <li>
      <a class="surface surface--interactive" href="/documents">
        <h2>Documents</h2>
        <p class="count">{fmt(counts.documents)}</p>
        <p class="hint">Ingested source material with summaries.</p>
      </a>
    </li>
  </ul>

  {#if error}
    <StatusBanner variant="error">
      Corpus overview unavailable: {error.message}
      {#if error.code}<span class="code">({error.code})</span>{/if}
    </StatusBanner>
  {:else if corpus}
    <div class="overview">
      <div class="stat-strip">
        <div class="stat">
          <span class="stat-value">{fmt(corpus.chunk_count)}</span>
          <span class="stat-label">chunks</span>
        </div>
        <div class="stat">
          <span class="stat-value">{fmt(corpus.token_count)}</span>
          <span class="stat-label">tokens</span>
        </div>
        <div class="stat">
          <span class="stat-value stat-value--text">
            {#if corpus.authored_date.min}
              {corpus.authored_date.min} – {corpus.authored_date.max}
            {:else}
              no dated documents
            {/if}
          </span>
          <span class="stat-label">
            {#if corpus.authored_date.undated_document_count}
              {fmt(corpus.authored_date.undated_document_count)} undated
            {:else}
              all dated
            {/if}
          </span>
        </div>
        <div class="stat">
          <span class="stat-value">{fmt(corpus.summary_coverage.summarized)}</span>
          <span class="stat-label">
            summarized{#if corpus.summary_coverage.unsummarized} · {fmt(corpus.summary_coverage.unsummarized)} without{/if}
          </span>
        </div>
      </div>

      <div class="panels">
        {#if corpus.documents_by_year.length}
          <section class="panel surface">
            <h3>Documents by year</h3>
            <ul class="histogram">
              {#each corpus.documents_by_year as y}
                <li>
                  <span class="bar-label">{y.year}</span>
                  <span class="bar-track">
                    <span class="bar" style="width: {(y.document_count / maxYear) * 100}%"></span>
                  </span>
                  <span class="bar-count">{fmt(y.document_count)}</span>
                </li>
              {/each}
            </ul>
          </section>
        {/if}

        {#if corpus.content_mix.length}
          <section class="panel surface">
            <h3>Content mix</h3>
            <ul class="rows">
              {#each corpus.content_mix as c}
                <li>
                  <span>{c.content_type ?? "text"}</span>
                  <span class="muted">{fmt(c.chunk_count)}</span>
                </li>
              {/each}
            </ul>
          </section>
        {/if}

        {#if corpus.tags.length}
          <section class="panel surface">
            <h3>Tags</h3>
            <div class="tags">
              {#each corpus.tags as t}
                {#if t.tag_id}
                  <a class="tag" href="/tags/{t.tag_id}">
                    {t.name} <span class="muted">{fmt(t.document_count)}</span>
                  </a>
                {:else}
                  <span class="tag">{t.name} <span class="muted">{fmt(t.document_count)}</span></span>
                {/if}
              {/each}
            </div>
          </section>
        {/if}

        {#if corpus.largest_documents.length}
          <section class="panel surface">
            <h3>Largest documents</h3>
            <ul class="rows">
              {#each corpus.largest_documents as d}
                <li>
                  <a href="/documents/{d.id}" title={d.title ?? stripExt(d.file_name)}>{d.title ?? stripExt(d.file_name)}</a>
                  <span class="muted">{d.token_count == null ? "—" : pluralize(d.token_count, "token")}</span>
                </li>
              {/each}
            </ul>
          </section>
        {/if}
      </div>
    </div>
  {/if}
</section>

<style>
  /* The corpus dashboard is intentionally wider than the text-column width
     (--width-content); it's a grid of panels, not a reading column. */
  .home {
    max-width: 52rem;
  }
  h1.display {
    font-family: var(--font-display);
    font-size: var(--text-5xl);
    margin-bottom: var(--space-2xs);
  }
  .tagline {
    color: var(--color-off);
    margin-bottom: var(--space-2xl);
  }
  .project {
    font-family: var(--font-display);
  }
  ul.cards {
    list-style: none;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
    gap: var(--space-lg);
  }
  /* The card box (border, fill, radius, shadow, hover) is the shared .surface
     primitive; only the link reset and inner type live here. */
  ul.cards a {
    display: block;
    text-decoration: none;
    color: inherit;
  }
  ul.cards h2 {
    font-size: var(--text-base);
    color: var(--color-off);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
    margin-bottom: var(--space-sm);
  }
  .count {
    font-family: var(--font-display);
    font-size: var(--text-4xl);
    line-height: 1;
    margin-bottom: var(--space-sm);
  }
  .hint {
    font-size: var(--text-sm);
    color: var(--color-off);
  }

  .overview {
    margin-top: var(--space-4xl);
    font-family: var(--font-sans);
  }
  /* The data panels inherit the base serif `li`; force sans across the whole
     dashboard so the rows/histograms read as data, not prose. */
  .overview li {
    font-family: var(--font-sans);
  }

  /* The status strip: one value-over-label unit per stat, framed by a rule top
     and bottom so it reads as its own band between the cards and the panels. */
  .stat-strip {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-lg) var(--space-3xl);
    padding: var(--space-lg) 0 var(--space-xl);
    border-top: 1px solid var(--color-rule);
    border-bottom: 1px solid var(--color-rule);
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: var(--space-3xs);
  }
  .stat-value {
    font-family: var(--font-display);
    font-size: var(--text-xl);
    line-height: 1;
    color: var(--color-text);
  }
  /* The authored-date range is text, not a metric — render it in readable sans
     instead of the pixel display face, which mangles a long date string. */
  .stat-value--text {
    font-family: var(--font-sans);
    font-size: var(--text-base);
    font-weight: 600;
  }
  .stat-label {
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
    color: var(--color-off);
  }

  /* Four panels in a 2×2 grid — auto-fit stranded the fourth in a lonely third
     column at this width. Each panel is a .surface (set in the markup). */
  .panels {
    margin-top: var(--space-2xl);
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: var(--space-xl);
    /* Each panel is only as tall as its content — without this the short
       panels stretch to match a tall neighbour, leaving big empty surfaces. */
    align-items: start;
  }
  .panel h3 {
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
    color: var(--color-off);
    margin-bottom: var(--space-md);
  }
  .rows {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: var(--text-sm);
  }
  .rows li {
    display: flex;
    justify-content: space-between;
    gap: var(--space-lg);
    padding: var(--space-2xs) 0;
  }
  .rows a {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--color-link);
    text-decoration: none;
  }
  .rows a:hover {
    text-decoration: underline;
  }
  .rows .muted {
    flex-shrink: 0;
  }

  .histogram {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: var(--text-sm);
  }
  .histogram li {
    display: grid;
    grid-template-columns: 3rem 1fr 2.5rem;
    align-items: center;
    gap: var(--space-sm);
    padding: var(--space-3xs) 0;
  }
  .bar-track {
    background: var(--color-rule);
    border-radius: 2px;
    height: 0.7rem;
  }
  .bar {
    display: block;
    height: 100%;
    background: var(--color-token);
    border-radius: 2px;
  }
  .bar-count {
    text-align: right;
    color: var(--color-off);
  }
</style>
