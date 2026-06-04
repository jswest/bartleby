<script>
  export let params;
  export let availableTags = [];

  // Must mirror DEFAULT_KINDS in +page.server.js — the set checked on first
  // load and whenever no kind is explicitly selected.
  const DEFAULT_KINDS = ["documents", "findings", "images"];
  const ALL_KINDS = ["documents", "summaries", "findings", "images"];

  // Local copy of the mode so the form can show/hide mode-specific fields
  // before submit. Everything else submits straight from the inputs.
  let mode = params.mode;

  $: kindChecked = (k) =>
    params.kinds.length ? params.kinds.includes(k) : DEFAULT_KINDS.includes(k);
  $: tagChecked = (name) => params.tags.includes(name);
</script>

<form class="search-form" method="GET" action="/search">
  <div class="mode" role="radiogroup" aria-label="Search mode">
    <label class:active={mode === "search"}>
      <input type="radio" name="mode" value="search" bind:group={mode} />
      Search <span class="mode-hint">ranked</span>
    </label>
    <label class:active={mode === "scan"}>
      <input type="radio" name="mode" value="scan" bind:group={mode} />
      Scan <span class="mode-hint">enumerate</span>
    </label>
  </div>

  <div class="query-row">
    <input
      class="query"
      type="search"
      name="q"
      value={params.q}
      placeholder={mode === "scan"
        ? "Phrase to enumerate across documents…"
        : "Search documents, findings, images…"}
      autocomplete="off"
      autofocus
    />
    <button type="submit">{mode === "scan" ? "Scan" : "Search"}</button>
  </div>

  <div class="filters">
    {#if mode === "search"}
      <fieldset>
        <legend>Sources</legend>
        <div class="checks">
          {#each ALL_KINDS as k}
            <label class="check">
              <input type="checkbox" name="kind" value={k} checked={kindChecked(k)} />
              {k}
            </label>
          {/each}
        </div>
      </fieldset>

      <fieldset>
        <legend>Context</legend>
        <label class="inline">
          <input type="number" name="context" min="0" max="5" value={params.context} />
          neighbor chunks
        </label>
      </fieldset>
    {:else}
      <fieldset>
        <legend>Matching</legend>
        <label class="check">
          <input type="checkbox" name="terms" value="1" checked={params.matchTerms} />
          terms (AND) instead of exact phrase
        </label>
      </fieldset>
    {/if}

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
      <legend>Scope</legend>
      <label class="inline">
        <input
          class="docs"
          type="text"
          name="docs"
          value={params.docs}
          placeholder="document ids, e.g. 3,5,9"
        />
      </label>
      <label class="inline">
        limit
        <input type="number" name="limit" min="1" max="500" value={params.limit} />
      </label>
    </fieldset>
  </div>
</form>

<style>
  .search-form {
    margin-bottom: 1.5rem;
    font-family: var(--font-sans);
  }
  .mode {
    display: inline-flex;
    border: 1px solid var(--color-off);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.75rem;
  }
  .mode label {
    padding: 0.35rem 0.9rem;
    cursor: pointer;
    color: var(--color-off);
    background: #fff;
    font-size: 0.9rem;
  }
  .mode label.active {
    background: var(--color-off);
    color: var(--color-off-light);
  }
  .mode input {
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }
  .mode-hint {
    opacity: 0.6;
    font-size: 0.75rem;
  }
  .query-row {
    display: flex;
    gap: 0.5rem;
    max-width: 44rem;
  }
  .query {
    flex: 1;
    padding: 0.55rem 0.75rem;
    font-size: 1rem;
    font-family: var(--font-serif);
    border: 1px solid var(--color-off);
    border-radius: 4px;
  }
  button[type="submit"] {
    padding: 0.55rem 1.2rem;
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
  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 0.85rem;
    max-width: 44rem;
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
  }
  .check,
  .inline {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.85rem;
    color: #333;
  }
  .inline + .inline {
    margin-left: 0.9rem;
  }
  .tag-count {
    color: var(--color-off);
    font-size: 0.75rem;
  }
  input[type="number"] {
    width: 3.5rem;
    padding: 0.2rem 0.3rem;
    border: 1px solid var(--color-rule);
    border-radius: 3px;
    font-family: var(--font-sans);
  }
  .docs {
    width: 12rem;
    padding: 0.2rem 0.4rem;
    border: 1px solid var(--color-rule);
    border-radius: 3px;
    font-family: var(--font-sans);
  }
</style>
