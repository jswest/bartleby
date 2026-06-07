<script>
  import { pluralize, stripExt } from "$lib/format.js";
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

  <ul class="cards">
    <li>
      <a href="/findings">
        <h2>Findings</h2>
        <p class="count">{fmt(counts.findings)}</p>
        <p class="hint">Saved research notes with inline citations.</p>
      </a>
    </li>
    <li>
      <a href="/documents">
        <h2>Documents</h2>
        <p class="count">{fmt(counts.documents)}</p>
        <p class="hint">Ingested source material with summaries.</p>
      </a>
    </li>
  </ul>

  {#if error}
    <p class="overview-error">
      Corpus overview unavailable: {error.message}
      {#if error.code}<span class="code">({error.code})</span>{/if}
    </p>
  {:else if corpus}
    <div class="overview">
      <div class="stat-strip">
        <span><strong>{fmt(corpus.chunk_count)}</strong> chunks</span>
        <span><strong>{fmt(corpus.token_count)}</strong> tokens</span>
        <span class="dates">
          {#if corpus.authored_date.min}
            {corpus.authored_date.min} – {corpus.authored_date.max}
          {:else}
            no dated documents
          {/if}
          · <span class="muted">{fmt(corpus.authored_date.undated_document_count)} undated</span>
        </span>
        <span>
          <strong>{fmt(corpus.summary_coverage.summarized)}</strong> summarized
          {#if corpus.summary_coverage.unsummarized}
            · <span class="muted">{fmt(corpus.summary_coverage.unsummarized)} without</span>
          {/if}
        </span>
      </div>

      <div class="panels">
        {#if corpus.documents_by_year.length}
          <section class="panel">
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
          <section class="panel">
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
          <section class="panel">
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
          <section class="panel">
            <h3>Largest documents</h3>
            <ul class="rows">
              {#each corpus.largest_documents as d}
                <li>
                  <a href="/documents/{d.id}">{d.title ?? stripExt(d.file_name)}</a>
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
  .home {
    max-width: 52rem;
  }
  h1.display {
    font-family: var(--font-display);
    font-size: 3rem;
    margin-bottom: 0.25rem;
  }
  .tagline {
    color: var(--color-off);
    margin-bottom: 2rem;
  }
  .project {
    font-family: var(--font-display);
  }
  ul.cards {
    list-style: none;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr));
    gap: 1rem;
  }
  ul.cards a {
    display: block;
    padding: 1.25rem;
    border: 1px solid #d6d4c6;
    border-radius: 6px;
    text-decoration: none;
    color: inherit;
    background: #fafaf6;
    transition: border-color 0.1s, background 0.1s;
  }
  ul.cards a:hover {
    border-color: var(--color-off);
    background: #fff;
  }
  ul.cards h2 {
    font-size: 1rem;
    color: var(--color-off);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }
  .count {
    font-family: var(--font-display);
    font-size: 2.5rem;
    line-height: 1;
    margin-bottom: 0.5rem;
  }
  .hint {
    font-size: 0.9rem;
    color: var(--color-off);
  }

  .overview-error {
    margin-top: 2rem;
    font-family: var(--font-sans);
    font-size: 0.9rem;
    color: #8a2b22;
  }
  .overview {
    margin-top: 2.5rem;
    font-family: var(--font-sans);
  }
  .stat-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--color-rule);
    font-size: 0.9rem;
    color: #333;
  }
  .stat-strip strong {
    font-family: var(--font-display);
  }
  .muted {
    color: var(--color-off);
  }
  .code {
    color: var(--color-token-dark);
  }

  .panels {
    margin-top: 1.5rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr));
    gap: 1.5rem 2rem;
  }
  .panel h3 {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-off);
    margin-bottom: 0.75rem;
  }
  .rows {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: 0.9rem;
  }
  .rows li {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.2rem 0;
  }
  .rows a {
    color: var(--color-token-dark);
    text-decoration: none;
  }
  .rows a:hover {
    text-decoration: underline;
  }

  .histogram {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: 0.85rem;
  }
  .histogram li {
    display: grid;
    grid-template-columns: 3rem 1fr 2.5rem;
    align-items: center;
    gap: 0.5rem;
    padding: 0.15rem 0;
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

  .tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  .tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border: 1px solid var(--color-rule);
    border-radius: 3px;
    font-size: 0.8rem;
    color: inherit;
    text-decoration: none;
  }
  a.tag:hover {
    border-color: var(--color-off);
  }
</style>
