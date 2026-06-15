<script>
  import { formatDateRange, pluralize, stripExt } from "$lib/format.js";
  import StatusBanner from "$lib/components/StatusBanner.svelte";
  import { FINDINGS_ICON, DOCUMENTS_ICON } from "$lib/icons.js";
  export let data;

  $: ({ project, counts, corpus, error } = data);

  const fmt = (n) => (n ?? 0).toLocaleString("en-US");

  // Count-up action for the LED readout numerals (#591): on mount, ramp the
  // displayed figure from 0 → target so the panel reads as a screen powering
  // on. Honours prefers-reduced-motion (and SSR / no-rAF) by snapping straight
  // to the final value, so the number is always correct without JS.
  function countUp(node, target) {
    const final = Number(target) || 0;
    const reduce =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce || typeof requestAnimationFrame === "undefined" || final === 0) {
      node.textContent = fmt(final);
      return;
    }
    const duration = 700;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / duration);
      // Ease-out so it decelerates into the final reading like a settling dial.
      const eased = 1 - Math.pow(1 - t, 3);
      node.textContent = fmt(Math.round(final * eased));
      if (t < 1) requestAnimationFrame(tick);
    };
    node.textContent = fmt(0);
    requestAnimationFrame(tick);
  }

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
        <h2><span class="card-eyebrow-icon">{@html FINDINGS_ICON}</span>Findings</h2>
        <p class="count">{fmt(counts.findings)}</p>
        <p class="hint">Saved research notes with inline citations.</p>
      </a>
    </li>
    <li>
      <a class="surface surface--interactive" href="/documents">
        <h2><span class="card-eyebrow-icon">{@html DOCUMENTS_ICON}</span>Documents</h2>
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
          <span class="stat-value" use:countUp={corpus.chunk_count}>{fmt(corpus.chunk_count)}</span>
          <span class="stat-label">chunks</span>
        </div>
        <div class="stat">
          <span class="stat-value" use:countUp={corpus.token_count}>{fmt(corpus.token_count)}</span>
          <span class="stat-label">tokens</span>
        </div>
        <div class="stat">
          <span class="stat-value stat-value--text">
            {#if formatDateRange(corpus.authored_date)}
              {formatDateRange(corpus.authored_date)}
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
          <span class="stat-value" use:countUp={corpus.summary_coverage.summarized}>{fmt(corpus.summary_coverage.summarized)}</span>
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
    /* R3 (#592): a Pixelarticon eyebrow rides before the card label, inheriting
       the sage --color-off the heading already carries. */
    display: flex;
    align-items: center;
    gap: var(--space-xs);
  }
  .card-eyebrow-icon {
    display: inline-flex;
  }
  .count {
    font-family: var(--font-display);
    font-size: var(--text-4xl);
    line-height: 1;
    margin-bottom: var(--space-sm);
    /* The dot-matrix count reads as an amber readout even on the paper card —
       the one apparatus accent that carries onto the artifact. */
    color: var(--color-token-dark);
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

  /* ──────────────────────────────────────────────────────────────────────
     R2 · Dot-matrix LED readout panel (#591)
     Grows R1's seeded stat band into a glowing dark "screen" distinct from
     the paper cards: a recessed near-black readout with amber Doto numerals
     (soft glow), a green dot-matrix date readout, muted-green caps labels,
     framing rules, an inset shadow, and a faint scanline overlay. Numerals
     count up on load. All selectors here are owned by this issue; the only
     shared tokens consumed are R1's :root vars (read-only).
     ────────────────────────────────────────────────────────────────────── */

  /* Local readout palette, derived from R1 tokens — kept here so the panel
     can be tuned without touching the foundation :root. */
  .stat-strip {
    --led-amber: var(--color-token-dark);
    --led-green: var(--color-off-light);
    --led-glow-amber: rgba(255, 189, 8, 0.45);
    --led-glow-green: rgba(219, 250, 235, 0.3);

    display: flex;
    flex-wrap: wrap;
    gap: var(--space-lg) var(--space-3xl);
    padding: var(--space-xl);
    position: relative;
    /* A touch darker than the shell so the readout reads as a recessed screen
       rather than a raised card; the inset shadow seals the "behind glass" feel. */
    background:
      linear-gradient(180deg, rgba(0, 0, 0, 0.25), rgba(0, 0, 0, 0)) ,
      var(--color-shell);
    border-top: 1px solid var(--color-shell-rule);
    border-bottom: 1px solid var(--color-shell-rule);
    box-shadow:
      inset 0 1px 0 rgba(0, 0, 0, 0.5),
      inset 0 12px 24px -12px rgba(0, 0, 0, 0.7);
    overflow: hidden;
  }
  /* Faint horizontal scanlines drawn over the panel — a repeating 1px dark
     line every 3px. Sits above the background but below the figures (the
     readout content stays at the default stacking, this layer is z-index:0
     and pointer-events:none so it never intercepts clicks). */
  .stat-strip::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
      to bottom,
      rgba(0, 0, 0, 0.18) 0,
      rgba(0, 0, 0, 0.18) 1px,
      transparent 1px,
      transparent 3px
    );
    z-index: 0;
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: var(--space-3xs);
    position: relative;
    z-index: 1; /* float the figures above the scanline overlay */
  }
  .stat-value {
    font-family: var(--font-display);
    /* Honour the dot-matrix size floor (--text-display-floor: 1.25rem) — bumped
       above R1's --text-xl so the LED figures read crisp, never muddy. */
    font-size: var(--text-2xl);
    line-height: 1;
    /* Amber dot-matrix readout with a soft LED glow. */
    color: var(--led-amber);
    text-shadow: 0 0 6px var(--led-glow-amber);
    font-variant-numeric: tabular-nums;
  }
  /* The authored-date range is text, not a metric, but on the LED panel it
     becomes the green dot-matrix readout from the prototype — Doto, sized at
     the display floor, glowing green where amber marks the metrics. */
  .stat-value--text {
    font-family: var(--font-display);
    font-size: var(--text-display-floor);
    line-height: 1;
    color: var(--led-green);
    text-shadow: 0 0 6px var(--led-glow-green);
    /* The range is a single token visually — keep it on one line and let it
       drive its own column width rather than wrapping mid-date. */
    white-space: nowrap;
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
