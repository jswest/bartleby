<script>
  import { pluralize } from "$lib/format.js";

  export let data;

  // Heroes are the two most-recent findings — listFindings() already orders
  // created_at DESC, so they're just the first two rows (no extra query).
  $: findings = data.findings;
  $: heroes = findings.slice(0, 2);

  // Citation bars are scaled to the largest count on the page; floor the
  // divisor at 1 so an all-zero corpus doesn't divide by zero.
  $: maxCites = Math.max(1, ...findings.map((f) => f.citation_count));

  // Index view: a scannable table by default, or the same findings grouped by
  // the research session that produced them. Client-side only.
  let mode = "table";

  // Group findings by session, preserving first-appearance order. Because
  // `findings` is newest-first, sessions sort newest-first too, and each
  // group's items stay newest-first within it.
  $: groups = (() => {
    const order = [];
    const bySession = new Map();
    for (const f of findings) {
      if (!bySession.has(f.session_name)) {
        bySession.set(f.session_name, []);
        order.push(f.session_name);
      }
      bySession.get(f.session_name).push(f);
    }
    return order.map((name) => {
      const items = bySession.get(name);
      const total = items.reduce((sum, f) => sum + f.citation_count, 0);
      const lo = day(items[items.length - 1].created_at);
      const hi = day(items[0].created_at);
      // All findings in one session share its run, so the model is the first
      // item's — the axis the multi-model comparison cares about (#547).
      const model = modelLabel(items[0]);
      return { name, items, total, model, span: lo === hi ? hi : `${lo} – ${hi}` };
    });
  })();

  // created_at carries a time component ("2026-06-11 20:43:44"); the list only
  // wants the day.
  function day(ts) {
    return (ts ?? "").slice(0, 10);
  }

  // The model behind a finding's run, or null when none was recorded. A model
  // the agent self-reported (model_set_by_llm, #547) is tagged "Set by LLM" —
  // it's a claim, not a verified fact. SQLite returns the boolean as 0/1.
  function modelLabel(f) {
    if (!f.model) return null;
    return f.model_set_by_llm ? `${f.model} · Set by LLM` : f.model;
  }

  function pct(count) {
    return Math.round((count / maxCites) * 100);
  }
</script>

<div class="findings">
  <div class="findings-head">
    <h1>Findings</h1>
    {#if findings.length}
      <p class="summary">
        {pluralize(findings.length, "finding")} · {pluralize(groups.length, "session")}
      </p>
    {/if}
  </div>

  {#if findings.length === 0}
    <p class="empty">No findings yet.</p>
  {:else}
    <p class="band-label">Most recent</p>
    <div class="heroes">
      {#each heroes as f}
        <article class="hero surface surface--finding">
          <span class="hero-badge">{pluralize(f.citation_count, "citation")}</span>
          <h2><a href="/findings/{f.finding_id}">{f.title}</a></h2>
          <p class="desc">{f.description}</p>
          <p class="meta">
            <span class="finding-id">#{f.finding_id}</span> · {f.session_name} · {day(f.created_at)}
          </p>
        </article>
      {/each}
    </div>

    <div class="index-toggle" role="group" aria-label="Index view">
      <span class="index-label">Index</span>
      <button type="button" class:on={mode === "table"} on:click={() => (mode = "table")}>
        Table
      </button>
      <button type="button" class:on={mode === "grouped"} on:click={() => (mode = "grouped")}>
        Grouped by session
      </button>
    </div>

    {#if mode === "table"}
      <table class="findings-table">
        <thead>
          <tr>
            <th class="num">#</th>
            <th>Finding</th>
            <th>Session</th>
            <th class="num">Citations</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {#each findings as f}
            <tr>
              <td class="id">{f.finding_id}</td>
              <td>
                <a class="t-title" href="/findings/{f.finding_id}">{f.title}</a>
                <span class="t-desc">{f.description}</span>
              </td>
              <td class="t-sess">
                {f.session_name}
                {#if modelLabel(f)}<span class="t-model">{modelLabel(f)}</span>{/if}
              </td>
              <td>
                <span class="cbar" class:zero={f.citation_count === 0}>
                  <span class="track"><span class="fill" style="width:{pct(f.citation_count)}%"></span></span>
                  <span class="n">{f.citation_count}</span>
                </span>
              </td>
              <td class="t-date">{day(f.created_at)}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    {:else}
      <div class="grouped">
        {#each groups as g}
          <section class="group">
            <div class="group-head">
              <span class="group-name">{g.name}</span>
              {#if g.model}<span class="group-model">{g.model}</span>{/if}
              <span class="group-meta">
                {pluralize(g.items.length, "finding")} · {pluralize(g.total, "citation")} · {g.span}
              </span>
            </div>
            {#each g.items as f}
              <div class="grow">
                <span class="g-id">#{f.finding_id}</span>
                <div>
                  <a class="g-title" href="/findings/{f.finding_id}">{f.title}</a>
                  <span class="g-desc">{f.description}</span>
                </div>
                <span class="g-chip" class:zero={f.citation_count === 0}>{f.citation_count}</span>
              </div>
            {/each}
          </section>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<style>
  /* The findings page runs wider than the reading column so the index table has
     room for its columns and the heroes can sit two-up. 72rem is a structural
     width, not a token (the design system reserves --width-content for prose). */
  .findings {
    max-width: 72rem;
  }

  .findings-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-lg);
    margin-bottom: var(--space-lg);
  }
  .findings-head .summary {
    margin: 0;
  }

  /* A small eyebrow framing the spotlight band (shared visual idiom with the
     index toggle's label). */
  .band-label {
    font-family: var(--font-sans);
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
    color: var(--color-off);
    margin-bottom: var(--space-sm);
  }

  /* Heroes — the two most-recent findings, spotlighted above the index. The
     mint .surface--finding base reads as "agent output" (same signal as the
     detail report); .hero only enlarges them and lays them side by side. They
     also reappear in the index below: a spotlight over a complete list, by
     design — not a de-dupe bug. */
  .heroes {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--space-lg);
    margin-bottom: var(--space-xl);
  }
  .hero {
    padding: var(--space-xl);
  }
  .hero h2 {
    font-size: var(--text-2xl);
    line-height: 1.2;
    margin: var(--space-sm) 0;
  }
  .hero h2 a {
    color: var(--color-text);
    text-decoration: none;
  }
  .hero h2 a:hover {
    color: var(--color-link);
  }
  .hero .desc {
    font-size: var(--text-base);
  }
  .hero .meta {
    margin-top: var(--space-md);
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    font-family: var(--font-display);
    font-size: var(--text-2xs);
    font-weight: 700;
    color: var(--color-link);
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    border-radius: var(--radius-pill);
    padding: var(--space-3xs) var(--space-sm);
  }

  /* Index toggle */
  .index-toggle {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    margin-bottom: var(--space-lg);
    font-family: var(--font-sans);
  }
  .index-label {
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
    color: var(--color-off);
  }
  .index-toggle button {
    font-family: var(--font-sans);
    font-size: var(--text-xs);
    font-weight: 600;
    color: var(--color-off);
    background: transparent;
    border: 1px solid var(--color-rule);
    border-radius: var(--radius-pill);
    padding: var(--space-2xs) var(--space-md);
    cursor: pointer;
  }
  .index-toggle button:hover {
    border-color: var(--color-off);
  }
  .index-toggle button.on {
    background: var(--color-off);
    color: var(--color-off-light);
    border-color: var(--color-off);
  }

  /* Table index — a dense, scannable TOC. */
  .findings-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-sans);
  }
  .findings-table thead th {
    text-align: left;
    font-size: var(--text-2xs);
    text-transform: uppercase;
    letter-spacing: var(--tracking-wide);
    color: var(--color-off);
    font-weight: 600;
    padding: 0 var(--space-md) var(--space-sm);
    border-bottom: 2px solid var(--color-off);
  }
  .findings-table thead th.num {
    text-align: right;
  }
  .findings-table tbody tr {
    border-bottom: 1px solid var(--color-rule);
  }
  .findings-table tbody tr:hover {
    background: var(--color-off-light);
  }
  .findings-table td {
    padding: var(--space-md);
    vertical-align: top;
  }
  .findings-table td.id {
    font-family: var(--font-display);
    color: var(--color-off);
    font-size: var(--text-xs);
    text-align: right;
    white-space: nowrap;
  }
  /* A finding row in either index view: a sans title over a serif description.
     Shared by the table (.t-*) and grouped (.g-*) views so the idiom stays one
     definition rather than two copies that can drift. */
  .t-title,
  .g-title {
    font-family: var(--font-sans);
    font-size: var(--text-sm);
    font-weight: 600;
    color: var(--color-text);
    text-decoration: none;
    line-height: 1.3;
  }
  .t-title:hover,
  .g-title:hover {
    color: var(--color-link);
  }
  .t-desc,
  .g-desc {
    display: block;
    font-family: var(--font-serif);
    font-size: var(--text-xs);
    color: var(--color-text-soft);
    margin-top: var(--space-3xs);
    line-height: 1.4;
  }
  .findings-table td.t-sess {
    font-size: var(--text-xs);
    color: var(--color-link);
    white-space: nowrap;
  }
  /* The run's model under the session name — a muted second line. */
  .t-model {
    display: block;
    font-size: var(--text-2xs);
    color: var(--color-off);
    margin-top: var(--space-3xs);
  }
  .findings-table td.t-date {
    font-size: var(--text-xs);
    color: var(--color-off);
    white-space: nowrap;
  }

  /* Citation weight bar. The fill MUST be block — an inline span ignores
     `width`, silently rendering every bar at 0px. Scaled to the max count on
     the page; a low non-zero count keeps a visible sliver via min-width, and a
     zero count drops the sliver and mutes the number. */
  .cbar {
    display: flex;
    align-items: center;
    gap: var(--space-xs);
  }
  .cbar .track {
    width: 3.75rem;
    height: 6px;
    background: var(--color-rule);
    border-radius: var(--radius-pill);
    overflow: hidden;
  }
  .cbar .fill {
    display: block;
    height: 100%;
    min-width: 2px;
    background: var(--color-token-dark);
    border-radius: var(--radius-pill);
  }
  .cbar .n {
    font-family: var(--font-display);
    font-size: var(--text-xs);
    color: var(--color-text-soft);
    min-width: 1.6rem;
  }
  .cbar.zero .fill {
    min-width: 0;
  }
  .cbar.zero .n {
    color: var(--color-off);
  }

  /* Grouped index — the same findings, sectioned by research session. */
  .group {
    margin-bottom: var(--space-2xl);
  }
  .group-head {
    display: flex;
    align-items: baseline;
    gap: var(--space-md);
    padding-bottom: var(--space-xs);
    border-bottom: 2px solid var(--color-off);
    margin-bottom: var(--space-md);
  }
  .group-name {
    font-family: var(--font-display);
    font-size: var(--text-lg);
    font-weight: 700;
    color: var(--color-off);
  }
  .group-model {
    font-family: var(--font-sans);
    font-size: var(--text-2xs);
    color: var(--color-link);
  }
  .group-meta {
    font-family: var(--font-sans);
    font-size: var(--text-2xs);
    color: var(--color-off);
  }
  .grow {
    display: grid;
    grid-template-columns: 2.5rem 1fr auto;
    gap: var(--space-md);
    align-items: baseline;
    padding: var(--space-sm);
    border-radius: var(--radius-md);
  }
  .grow:hover {
    background: var(--color-off-light);
  }
  .g-id {
    font-family: var(--font-display);
    font-size: var(--text-2xs);
    color: var(--color-off);
    text-align: right;
  }
  .g-chip {
    font-family: var(--font-display);
    font-size: var(--text-2xs);
    font-weight: 600;
    color: var(--color-link);
    background: var(--color-token);
    border: 1px solid var(--color-token-dark);
    border-radius: var(--radius-pill);
    padding: var(--space-3xs) var(--space-sm);
    white-space: nowrap;
  }
  .g-chip.zero {
    background: transparent;
    border-color: var(--color-rule);
    color: var(--color-off);
  }

  @media (max-width: 48rem) {
    .heroes {
      grid-template-columns: 1fr;
    }
  }
</style>
