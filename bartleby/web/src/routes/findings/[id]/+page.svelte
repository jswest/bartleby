<script>
  import { onMount, onDestroy, tick } from "svelte";
  import { goto } from "$app/navigation";
  import { marked } from "marked";
  import Button from "$lib/components/Button.svelte";
  import SourceViewer from "$lib/components/SourceViewer.svelte";
  import { stripExt } from "$lib/format.js";
  import { CHUNK_ICON } from "$lib/icons.js";

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
  // notes. Each marker in the body becomes an inline DAGGER anchor (†/‡ +
  // ordinal); the matching note carries the same dagger+ordinal in the gutter.
  // The ordinal is the tie-back — daggers repeat, the number disambiguates.
  //
  // One type-tagged marker grammar (issue #624): [^<scheme>:<ref>].
  //   [^chunk:N]         corpus-chunk citation. Resolves against byId →
  //                        resolved  → † "source" note (amber, filename, jumps)
  //                        unresolved → ‡ "no longer available" note (danger)
  //   [^url:…]/[^doc:…]   external citation → † "source" note (link / ref)
  // Mirrors the backend grammar in skill_scripts/_common.py: `chunk` is the
  // internal scheme, `url`/`doc` are external; any other scheme (incl. a stray
  // document/finding marker) is dropped, not rendered as a citation.
  //
  // `notes` is rebuilt alongside `bodyHtml` in one ordered pass so the gutter
  // order matches reading order. Reactive on `active` so the .active state
  // bakes into the rendered HTML.
  let notes = [];
  $: bodyHtml = renderBody(data.finding.body, byId, active);

  function renderBody(body, byId, active) {
    const collected = [];
    let n = 0;
    const html = marked.parse(
      body.replace(/\[\^([A-Za-z]+):([^\]]+)\]/g, (match, scheme, ref) => {
        const s = scheme.toLowerCase();
        if (s === "chunk") {
          const id = Number(ref.trim());
          // A non-numeric chunk ref can't resolve; leave the marker verbatim.
          if (!Number.isInteger(id)) return esc(match);
          const c = byId.get(id);
          const note = c
            ? sourceNote(++n, c, c === active)
            : goneNote(++n, id);
          collected.push(note);
          return marker(note);
        }
        const note = externalNote(++n, s, ref.trim());
        if (!note) {
          n--; // unknown scheme: drop, don't burn an ordinal
          return esc(match);
        }
        collected.push(note);
        return marker(note);
      }),
    );
    notes = collected;
    return html;
  }

  // The inline dagger anchor that sits at the cited point in the prose. A
  // superscript <sup> with the dagger glyph + ordinal; the ordinal ties it to
  // its gutter note. Stable class hooks (`cite-ref`, kind modifier) so R3 (#592)
  // can splice a pixel icon in without touching this code.
  function marker(note) {
    const kindCls = note.gone ? "cite-ref--gone" : "cite-ref--source";
    const title = note.gone
      ? `${note.dagger} cited source no longer available`
      : `${note.dagger} source · ${note.title}`;
    return `<sup class="cite-ref ${kindCls}" data-note="${note.n}" title="${esc(title)}">${note.dagger}${note.n}</sup>`;
  }

  function sourceNote(n, c, isActive) {
    const name = stripExt(c.file_name);
    const label =
      name && c.page_number ? `${name} p.${c.page_number}`
      : name ? name
      : c.page_number ? `p.${c.page_number}`
      : `chunk ${c.chunk_id}`;
    const title = [
      c.file_name,
      c.page_number && `page ${c.page_number}`,
      `chunk ${c.chunk_id}`,
    ].filter(Boolean).join(" · ");
    return {
      n,
      dagger: "†",
      gone: false,
      chunkId: c.chunk_id,
      active: isActive,
      label,
      title,
    };
  }

  // An unresolved [^N]: the cited source has been removed (deleted, or rebuilt
  // under new chunk ids by an edit/merge), so only the id survives. Danger-toned
  // gutter note rather than leaking raw [^N] text.
  function goneNote(n, chunkId) {
    return {
      n,
      dagger: "‡",
      gone: true,
      chunkId,
      label: "no longer available",
      title: `chunk ${chunkId} · cited source no longer available`,
    };
  }

  // An external (URL / external-dataset doc) citation parsed straight from the
  // body marker — no DB row backs it. A url scheme becomes a real link, a doc
  // scheme a muted ref. Unknown schemes return null (dropped upstream).
  function externalNote(n, scheme, ref) {
    if (scheme === "url") {
      return { n, dagger: "†", gone: false, external: "url", href: ref, label: ref, title: `external source · ${ref}` };
    }
    if (scheme === "doc") {
      return { n, dagger: "†", gone: false, external: "doc", label: ref, title: `external dataset doc · ${ref}` };
    }
    return null;
  }

  function esc(s) {
    return s.replace(/[&<>"]/g, (c) => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
    ));
  }

  let active = null;
  $: activeUrl = active ? citationUrl(active) : null;

  function citationUrl(c) {
    if (c.document_id == null) return null;
    const page = c.page_number ? `#page=${c.page_number}` : "";
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

  // ===== Dagger-aligned sidenote layout (#631) =========================
  // Each margin note's top is pinned to the vertical position of its inline
  // dagger in the prose (Tufte-style). A downward de-overlap pass ensures
  // notes never overlap. On narrow screens (≤56 rem) the gutter collapses to
  // normal flow — the layout function no-ops there.
  //
  // The breakpoint mirrors the `@media (max-width: 56rem)` in app.css. We
  // check container width rather than window width so it degrades correctly
  // even in unusual viewport configurations.
  const NOTE_GAP = 8; // px — minimum breathing room between stacked notes

  function layoutNotes() {
    if (!citesAside || !container) return;
    // Guard: if the aside is not in its normal (wide) grid track, fall back to
    // static flow. We detect this by checking whether the grid column gives the
    // aside visible width; on narrow screens its column collapses to 0/hidden.
    // Using the aside's offsetWidth: if it is ≤ 0 the layout is stacked (mobile).
    if (citesAside.offsetWidth <= 0) return;

    const noteEls = Array.from(citesAside.querySelectorAll(".margin-note"));
    if (noteEls.length === 0) return;

    const asideTop = citesAside.getBoundingClientRect().top + window.scrollY;

    // Measure desired tops (aligned to dagger) and note heights.
    const desired = noteEls.map((el) => {
      const n = el.dataset.note;
      const ref = container.querySelector(`.cite-ref[data-note="${n}"]`);
      let top = 0;
      if (ref) {
        top = ref.getBoundingClientRect().top + window.scrollY - asideTop;
        // Clamp to 0 — a dagger above the aside's top edge (shouldn't happen
        // in practice) would produce a negative top, which is invisible.
        if (top < 0) top = 0;
      }
      return { el, top, height: el.offsetHeight };
    });

    // Downward de-overlap pass: sweep top→bottom; push a note down if it
    // would overlap the one above.
    let runningBottom = 0;
    for (const item of desired) {
      if (item.top < runningBottom) item.top = runningBottom;
      runningBottom = item.top + item.height + NOTE_GAP;
    }

    // Apply positions and set the aside height so it doesn't collapse.
    for (const { el, top } of desired) {
      el.style.position = "absolute";
      el.style.top = `${top}px`;
      el.style.left = "0";
      el.style.right = "0";
    }
    citesAside.style.height = `${runningBottom - NOTE_GAP}px`;
  }

  function resetNoteLayout() {
    if (!citesAside) return;
    const noteEls = citesAside.querySelectorAll(".margin-note");
    for (const el of noteEls) {
      el.style.position = "";
      el.style.top = "";
      el.style.left = "";
      el.style.right = "";
    }
    citesAside.style.height = "";
  }

  // Re-run layout whenever notes or bodyHtml change (after DOM settles).
  $: if (notes, bodyHtml, citesAside) {
    tick().then(layoutNotes);
  }

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

    // Re-measure on resize. ResizeObserver on the prose body catches both
    // window resize and flex/grid reflow of the prose column.
    const proseEl = container.querySelector(".body");
    if (proseEl && typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        // On narrow screens the aside collapses; reset absolute styles so
        // normal flow takes over; on wide screens re-layout.
        if (citesAside && citesAside.offsetWidth <= 0) {
          resetNoteLayout();
        } else {
          layoutNotes();
        }
      });
      resizeObserver.observe(proseEl);
    }
  });

  onDestroy(() => {
    resizeObserver?.disconnect();
  });

  let copied = false;
  async function copyMarkdown() {
    await navigator.clipboard.writeText(data.finding.body);
    copied = true;
    setTimeout(() => (copied = false), 1500);
  }

  function downloadMarkdown() {
    const blob = new Blob([data.finding.body], { type: "text/markdown" });
    const name = data.finding.title.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${name}.md`;
    a.click();
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
    </div>

    <!-- Ledger column: drop-capped prose + a margin-note gutter. The body
         carries inline dagger anchors; the gutter carries the matching notes. -->
    <div class="ledger-column">
      <div class="body markdown-body drop-cap-body">
        {@html bodyHtml}
      </div>

      {#if notes.length}
        <aside class="cite-notes" aria-label="Citations" bind:this={citesAside}>
          {#each notes as note (note.n)}
            <div
              class="margin-note margin-note--{note.gone ? 'gone' : 'source'}"
              data-note={note.n}
            >
              <p class="margin-note__head">
                <span class="margin-note__dagger">{note.dagger}{note.n}</span>
                {note.gone ? "missing source" : "source"}
              </p>
              {#if note.gone}
                <p class="margin-note__body">no longer available</p>
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
                    on:click|preventDefault={() => goto(`/chunks/${note.chunkId}`)}
                  >{@html CHUNK_ICON}</a>
                </span>
              {/if}
            </div>
          {/each}
        </aside>
      {/if}
    </div>
  </article>

  <aside class="viewer">
    <SourceViewer fileName={active?.file_name} src={activeUrl} />
  </aside>
</div>

<style>
  .toolbar {
    display: flex;
    gap: var(--space-sm);
    margin-top: var(--space-md);
  }
</style>
