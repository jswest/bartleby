<script>
  export let data;
</script>

<section class="home">
  <h1 class="display">Bartleby</h1>
  <p class="tagline">
    The agent's notebook for <span class="project">{data.project}</span>.
  </p>

  <ul class="cards">
    <li>
      <a href="/findings">
        <h2>Findings</h2>
        <p class="count">{data.counts.findings}</p>
        <p class="hint">Saved research notes with inline citations.</p>
      </a>
    </li>
    <li>
      <a href="/documents">
        <h2>Documents</h2>
        <p class="count">{data.counts.documents}</p>
        <p class="hint">Ingested source material with summaries.</p>
      </a>
    </li>
  </ul>
</section>

<style>
  .home {
    max-width: 44rem;
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
</style>
