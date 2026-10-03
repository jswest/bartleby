<script>
  import { onMount, onDestroy, afterUpdate } from "svelte";
  import { invalidateAll } from "$app/navigation";
  import { marked } from "marked";
  import Button from "$lib/components/Button.svelte";
  import SourceViewer from "$lib/components/SourceViewer.svelte";
  import AnnotationNote from "$lib/components/AnnotationNote.svelte";
  import { substituteCitations } from "$lib/citations.js";
  import { anchorFor, nthIndexOf, occurrenceIndex } from "$lib/annotations.js";
  import { CHUNK_ICON } from "$lib/icons.js";
  import { slugify } from "$lib/format.js";

  export let data;

  $: byId = new Map(data.finding.citations.map((c) => [c.chunk_id, c]));

  // The run's model for the meta line. Rendered inline among the · -separated
  // fields with the qualifier in parens — deliberately distinct from the index's
  // "model · Set by LLM" muted line (#547). null when no model was recorded.
  $: modelMeta = data.finding.model
    ? `${data.finding.model}${data.finding.model_set_by_llm ? " (Set by LLM)" : ""}`
    : null;

  // ===== R4 ledger treatment (#593) =================================
  // Citations leave the prose and move into a right-hand gutter as margin
  // notes. Each marker in the body becomes an inline glyph anchor (¶/§/†/‡ +
  // ordinal); the matching note carries the same glyph+ordinal in the gutter.
  // The ordinal is the tie-back — daggers repeat, the number disambiguates.
  //
  // One type-tagged marker grammar (issue #624): [^<scheme>:<ref>].
  //   [^chunk:N]         corpus-chunk citation. Resolves against byId →
  //                        resolved  → ¶ "source" note (amber, filename, jumps)
  //                        unresolved → ‡ "no longer available" note (danger)
  //   [^finding:N]       finding-to-finding link (#654) → § note with a link
  //                        to /findings/N. No DB row — validated at save time,
  //                        rendered from body text on read.
  //   [^url:…]/[^doc:…]   external citation → † "source" note (link / ref)
  // Glyph mapping (supersedes GH-0654): ¶ corpus chunk (resolved), ‡ chunk
  // gone, § finding-to-finding link, † external url/doc — the glyph alone
  // tells the kind apart; the ordinal stays the tie-back.
  // Mirrors the backend grammar in skill_scripts/_common.py: `chunk` and
  // `finding` are internal schemes; `url`/`doc` are external; any other scheme
  // (incl. `document`) is dropped, not rendered as a citation.
  //
  // renderBody is a PURE function returning { html, notes } in one ordered pass,
  // so the gutter order matches reading order. Reactive on `active` so the
  // .active state bakes into the rendered HTML. It must NOT assign `notes` as a
  // side effect: a reactive computation that mutates another reactive variable
  // creates an update cycle Svelte cannot order, and the `tick()` relayout below
  // then re-enters the flush forever, freezing the tab (#631).
  $: rendered = renderBody(data.finding.body, byId, active);
  $: bodyHtml = rendered.html;
  $: notes = rendered.notes;

  // Marker substitution + note-building is shared with the HTML export
  // (GH-0690) via $lib/citations.js; only the final marked.parse() pass (this
  // app's global, DOMPurify-hooked `marked` singleton — see +layout.svelte)
  // stays here.
  function renderBody(body, byId, active) {
    const { markdown, notes } = substituteCitations(body, byId, active);
    return { html: marked.parse(markdown), notes };
  }

  let active = null;
  $: activeUrl = active ? citationUrl(active) : null;

  function citationUrl(c) {
    if (c.document_id == null) return null;
    const page = c.page_number ? `#page=${c.page_number}&navpanes=0` : "#navpanes=0";
    return `/files/${c.document_id}${page}`;
  }

  function activate(chunkId) {
    const c = byId.get(chunkId);
    if (c) active = c;
  }

  // Clicking an inline dagger scrolls its gutter note into view and activates
  // it (loads the source in the viewer). Delegated on the report container
  // since the markers are static {@html}.
  let container;
  let citesAside;
  let bodyEl;

  // ===== Annotations (#689/#734) =====================================
  // Human/agent notes layered on the finding — commentary, NOT citations, so
  // they get their own ✎ kind in the gutter and never touch the body markdown.
  // An anchored note whose quote still locates sits in the gutter beside a
  // highlight of its span; whole-finding notes and stale anchors (the finding
  // was edited since) render as a block below the body.
  const isPlaced = (a) => a.anchor_exact != null && a.anchor_found;
  $: placed = data.annotations.filter(isPlaced);
  $: unplaced = data.annotations.filter((a) => !isPlaced(a));

  // annotation_id → DOM Range over its span in the rendered body. Mutated (never
  // reassigned) in afterUpdate, so it is deliberately non-reactive.
  const annotationRanges = new Map();

  // The body's text nodes with their offsets into the concatenated text —
  // the same string Range.toString() yields over the body.
  function bodyText() {
    const walker = document.createTreeWalker(bodyEl, NodeFilter.SHOW_TEXT);
    const nodes = [];
    let text = "";
    while (walker.nextNode()) {
      nodes.push({ node: walker.currentNode, start: text.length });
      text += walker.currentNode.data;
    }
    return { nodes, text };
  }

  function rangeAt(nodes, from, to) {
    const point = (pos, isEnd) => {
      const hit = nodes.find(({ node, start }) =>
        isEnd ? pos <= start + node.data.length : pos < start + node.data.length);
      return [hit.node, pos - hit.start];
    };
    const range = document.createRange();
    range.setStart(...point(from, false));
    range.setEnd(...point(to, true));
    return range;
  }

  // Highlight each placed note's span by searching the RENDERED text (never by
  // editing the markdown — see app.css on <mark> mid-parse) and painting it with
  // the CSS Custom Highlight API, so the {@html} body DOM is never mutated. The
  // occurrence index from the raw anchor picks the same repeat here. An
  // agent-written quote may carry inline markdown (`**`, backticks) the
  // rendered text lacks, so retry with those stripped.
  function highlightAnnotations() {
    annotationRanges.clear();
    if (!bodyEl) return;
    const { nodes, text } = bodyText();
    for (const a of placed) {
      for (const needle of [a.anchor_exact, a.anchor_exact.replace(/[*`]/g, "")]) {
        if (!needle) continue;
        let at = nthIndexOf(text, needle, a.anchor_occurrence);
        if (at === -1) at = text.indexOf(needle);
        if (at === -1) continue;
        annotationRanges.set(a.annotation_id, rangeAt(nodes, at, at + needle.length));
        break;
      }
    }
    if (globalThis.CSS?.highlights) {
      CSS.highlights.set("annotation", new Highlight(...annotationRanges.values()));
    }
  }

  // Hovering a note deepens its own span's highlight — the tie-back.
  function focusAnnotation(id) {
    if (!globalThis.CSS?.highlights) return;
    const range = id != null && annotationRanges.get(id);
    if (range) CSS.highlights.set("annotation-active", new Highlight(range));
    else CSS.highlights.delete("annotation-active");
  }

  // --- Add flow: select text → "Annotate" pop → form → POST ---
  let pick = null; // {top, left, quote, anchor} — the floating Annotate button
  let draft = null; // the open form: {quote, anchor} (both null = whole-finding)
  let noteText = "";
  let chunkRef = "";
  let formError = null;
  let saving = false;

  // Map the selection to a raw-body anchor. The selected rendered text must
  // occur verbatim in the raw markdown; repeats are told apart by occurrence
  // order (which repeat in the rendered text = which in the raw body), then
  // pinned with prefix/suffix. A selection that crosses formatting or a
  // citation marker maps to nothing (anchor null) and is offered as a
  // whole-finding note — never a fabricated anchor.
  function onBodyMouseUp() {
    pick = null;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return;
    const range = sel.getRangeAt(0);
    if (!bodyEl.contains(range.commonAncestorContainer)) return;
    const raw = range.toString();
    const quote = raw.trim();
    if (!quote) return;
    const before = document.createRange();
    before.setStart(bodyEl, 0);
    before.setEnd(range.startContainer, range.startOffset);
    const pos = before.toString().length + (raw.length - raw.trimStart().length);
    const k = occurrenceIndex(bodyText().text, quote, pos);
    const rect = range.getBoundingClientRect();
    const box = container.getBoundingClientRect();
    pick = {
      top: rect.bottom - box.top + 6,
      left: Math.max(0, rect.left - box.left),
      quote,
      anchor: anchorFor(data.finding.body, quote, k),
    };
  }

  function startNote(from = null) {
    draft = { quote: from?.quote ?? null, anchor: from?.anchor ?? null };
    pick = null;
    noteText = "";
    chunkRef = "";
    formError = null;
  }

  function focusOnMount(node) {
    node.focus();
    node.scrollIntoView({ block: "center", behavior: "smooth" });
  }

  async function saveNote() {
    const ref = chunkRef.trim();
    if (ref && !/^(chunk:)?\d+$/.test(ref)) {
      formError = "Chunk must look like chunk:123.";
      return;
    }
    if (saving) return;
    saving = true;
    formError = null;
    const res = await fetch(`/findings/${data.finding.finding_id}/annotations`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ body: noteText, anchor: draft.anchor, chunk_id: ref || null }),
    });
    saving = false;
    if (!res.ok) {
      formError = (await res.json().catch(() => null))?.error ?? `Save failed (${res.status}).`;
      return;
    }
    draft = null;
    await invalidateAll();
  }

  async function removeNote(a) {
    if (!confirm("Delete this note?")) return;
    await fetch(`/findings/${data.finding.finding_id}/annotations/${a.annotation_id}`, { method: "DELETE" });
    await invalidateAll();
  }

  // ===== Dagger-aligned sidenote layout (#631) =========================
  // Each margin note's top is pinned to the vertical position of its inline
  // dagger in the prose (Tufte-style). A downward de-overlap pass ensures
  // notes never overlap. On narrow screens (≤56 rem) the gutter collapses to
  // normal flow — the layout function clears inline styles and returns early.
  //
  // The breakpoint mirrors the `@media (max-width: 56rem)` in app.css, checked
  // via matchMedia so it matches the CSS condition exactly.
  const NOTE_GAP = 8; // px — minimum breathing room between stacked notes

  function layoutNotes() {
    if (!citesAside || !container) return;
    // Guard: on narrow screens, clear any inline absolute styles and let static
    // CSS flow take over (the @media block handles display/position).
    const narrow = typeof window !== "undefined" && window.matchMedia("(max-width: 56rem)").matches;
    if (narrow) {
      const noteEls = citesAside.querySelectorAll(".margin-note");
      for (const el of noteEls) {
        el.style.position = "";
        el.style.top = "";
      }
      citesAside.style.height = "";
      return;
    }

    const noteEls = Array.from(citesAside.querySelectorAll(".margin-note"));
    if (noteEls.length === 0) return;

    const asideTop = citesAside.getBoundingClientRect().top + window.scrollY;

    // Pass 1: set position:absolute and initial top from the dagger offset.
    // left/right are set via CSS (.cite-notes .margin-note { left:0; right:0 })
    // so the final width is established before we measure heights in pass 2.
    // A citation note aligns to its inline dagger; an annotation note to the
    // top of its highlighted span.
    const items = noteEls.map((el) => {
      const ref = el.dataset.annotation
        ? annotationRanges.get(Number(el.dataset.annotation))
        : container.querySelector(`.cite-ref[data-note="${el.dataset.note}"]`);
      let top = 0;
      if (ref) {
        top = ref.getBoundingClientRect().top + window.scrollY - asideTop;
        if (top < 0) top = 0;
      }
      el.style.position = "absolute";
      el.style.top = `${top}px`;
      return { el, top };
    });

    // Pass 2: now that each note is absolutely positioned at its final width,
    // read offsetHeight and run the downward de-overlap sweep — in reading
    // order, since annotation notes interleave with citation notes.
    items.sort((a, b) => a.top - b.top);
    let runningBottom = 0;
    for (const item of items) {
      const height = item.el.offsetHeight;
      if (item.top < runningBottom) item.top = runningBottom;
      item.el.style.top = `${item.top}px`;
      runningBottom = item.top + height + NOTE_GAP;
    }

    citesAside.style.height = `${runningBottom - NOTE_GAP}px`;
  }

  // Re-run layout after every DOM update (content/active change, gutter mount).
  // afterUpdate runs once the DOM reflects the latest state, so we measure real
  // dagger positions. It must NOT be a reactive `$:` block calling tick(): doing
  // so re-enters Svelte's flush every cycle and pins the main thread (#631).
  // layoutNotes only mutates inline styles, so it never schedules a new update.
  afterUpdate(() => {
    highlightAnnotations();
    layoutNotes();
  });

  let resizeObserver;

  onMount(() => {
    container.addEventListener("click", (e) => {
      const ref = e.target.closest(".cite-ref");
      if (!ref) return;
      e.preventDefault();
      const el = container.querySelector(
        `.margin-note[data-note="${ref.dataset.note}"]`,
      );
      if (el) el.scrollIntoView({ behavior: "smooth", block: "nearest" });
      const noteEl = el?.querySelector("[data-chunk-id]");
      if (noteEl) activate(Number(noteEl.dataset.chunkId));
    });
    bodyEl.addEventListener("mouseup", onBodyMouseUp);

    // Re-measure on resize. ResizeObserver on the prose body catches both
    // window resize and flex/grid reflow of the prose column.
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => layoutNotes());
      resizeObserver.observe(bodyEl);
    }
  });

  onDestroy(() => {
    resizeObserver?.disconnect();
    globalThis.CSS?.highlights?.delete("annotation");
    globalThis.CSS?.highlights?.delete("annotation-active");
  });

  let copied = false;
  async function copyMarkdown() {
    await navigator.clipboard.writeText(data.finding.body);
    copied = true;
    setTimeout(() => (copied = false), 1500);
  }

  function downloadMarkdown() {
    const blob = new Blob([data.finding.body], { type: "text/markdown" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${slugify(data.finding.title)}.md`;
    a.click();
  }

  // GH-0690: the export route sets Content-Disposition: attachment, so a plain
  // navigation triggers the browser's normal download flow rather than
  // replacing the page.
  function downloadHtml() {
    window.location.href = `/findings/${data.finding.finding_id}/export.html`;
  }
</script>

<div class="split">
  <article class="report surface surface--finding ledger" bind:this={container}>
    <h1 class="ledger-hed">{data.finding.title}</h1>
    <p class="meta">
      <span class="finding-id">#{data.finding.finding_id}</span> · {data.finding.session_name}{#if modelMeta} · {modelMeta}{/if} · {data.finding.created_at}
    </p>
    <p class="ledger-dek">{data.finding.description}</p>

    <div class="toolbar">
      <Button size="sm" type="button" on:click={copyMarkdown}>
        {copied ? "Copied" : "Copy as Markdown"}
      </Button>
      <Button size="sm" type="button" on:click={downloadMarkdown}>Download .md</Button>
      <Button size="sm" type="button" on:click={downloadHtml}>Save as HTML</Button>
      <Button size="sm" type="button" on:click={() => startNote()}>Add note</Button>
    </div>

    <!-- Ledger column: drop-capped prose + a margin-note gutter. The body
         carries inline dagger anchors; the gutter carries the matching notes. -->
    <div class="ledger-column">
      <div class="body markdown-body drop-cap-body" bind:this={bodyEl}>
        {@html bodyHtml}
      </div>

      {#if notes.length || placed.length}
        <aside class="cite-notes" aria-label="Citations" bind:this={citesAside}>
          {#each notes as note (note.n)}
            <div
              class="margin-note margin-note--{note.gone ? 'gone' : note.finding ? 'finding' : 'source'}"
              data-note={note.n}
            >
              <p class="margin-note__head">
                <span class="margin-note__dagger">{note.dagger}{note.n}</span>
                {note.gone ? "missing source" : "source"}
              </p>
              {#if note.gone}
                <p class="margin-note__body">no longer available</p>
              {:else if note.finding}
                <p class="margin-note__body">
                  <a href={note.href} title={note.title}>{note.label}</a>
                </p>
              {:else if note.external === "url"}
                <p class="margin-note__body">
                  <a href={note.href} title={note.title}>{note.label}</a>
                </p>
              {:else if note.external === "doc"}
                <p class="margin-note__body margin-note__body--doc" title={note.title}>{note.label}</p>
              {:else}
                <!-- Primary: clicking the label opens the source in the right pane.
                     Secondary: the ↗ icon navigates to the chunk page. -->
                <span class="margin-note__body margin-note__source-row">
                  <button
                    type="button"
                    class="margin-note__link"
                    class:active={note.active}
                    data-chunk-id={note.chunkId}
                    title={note.title}
                    on:click={() => activate(note.chunkId)}
                  >{note.label}</button><a
                    href="/chunks/{note.chunkId}"
                    class="margin-note__open-chunk"
                    title="Open chunk {note.chunkId}"
                  >{@html CHUNK_ICON}</a>
                </span>
              {/if}
            </div>
          {/each}
          {#each placed as a (a.annotation_id)}
            <!-- svelte-ignore a11y-no-static-element-interactions -->
            <div
              class="margin-note margin-note--annotation"
              data-annotation={a.annotation_id}
              on:mouseenter={() => focusAnnotation(a.annotation_id)}
              on:mouseleave={() => focusAnnotation(null)}
            >
              <AnnotationNote annotation={a} on:delete={(e) => removeNote(e.detail)} />
            </div>
          {/each}
        </aside>
      {/if}
    </div>

    {#if pick}
      <button
        type="button"
        class="annotate-pop"
        style="top: {pick.top}px; left: {pick.left}px"
        on:mousedown|preventDefault
        on:click={() => startNote(pick)}
      >✎ Annotate</button>
    {/if}

    {#if draft}
      <form class="annotation-form" on:submit|preventDefault={saveNote}>
        {#if draft.anchor}
          <p class="annotation-quote">On “{draft.quote}”</p>
        {:else if draft.quote}
          <p class="annotation-warn">
            That selection crosses formatting or a citation marker, so it can't be
            anchored to the finding's text verbatim. Save it as a whole-finding note instead?
          </p>
        {:else}
          <p class="annotation-quote">A note on the whole finding</p>
        {/if}
        <textarea bind:value={noteText} rows="3" required placeholder="Note (plain text)" use:focusOnMount></textarea>
        <input type="text" bind:value={chunkRef} placeholder="chunk:123 (optional)" />
        {#if formError}<p class="annotation-warn">{formError}</p>{/if}
        <div class="toolbar">
          <Button size="sm" type="submit">{draft.quote && !draft.anchor ? "Save as whole-finding note" : "Save note"}</Button>
          <Button size="sm" type="button" on:click={() => (draft = null)}>Cancel</Button>
        </div>
      </form>
    {/if}

    {#if unplaced.length}
      <section class="annotations" aria-label="Notes on this finding">
        <h2 class="annotations__hed">✎ Notes</h2>
        {#each unplaced as a (a.annotation_id)}
          <div class="margin-note margin-note--annotation">
            <AnnotationNote annotation={a} stale={a.anchor_exact != null} on:delete={(e) => removeNote(e.detail)} />
          </div>
        {/each}
      </section>
    {/if}
  </article>

  <aside class="viewer">
    <SourceViewer fileName={active?.file_name} src={activeUrl} />
  </aside>
</div>

<svelte:window on:mousedown={(e) => { if (!e.target.closest?.(".annotate-pop")) pick = null; }} />

<style>
  .ledger {
    position: relative; /* anchors the floating Annotate button */
  }
  .toolbar {
    display: flex;
    gap: var(--space-sm);
    margin-top: var(--space-md);
  }
</style>
