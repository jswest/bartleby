<script context="module">
  import { marked } from "marked";
  import DOMPurify from "isomorphic-dompurify";

  // Every `{@html marked.parse(...)}` site (summaries, finding bodies, markdown
  // sources) flows through this one global parser. Summaries are model-generated
  // and finding bodies agent-authored — both untrusted — and `marked` passes raw
  // HTML through, so an embedded <script> would run. Sanitize first (DOMPurify
  // strips scripts/handlers/js: URLs while keeping the cite-chip <button>), then
  // add link attrs to the clean output so target/rel don't depend on the sanitizer.
  //
  // Module scope, not instance: `marked.use` registers globally and *chains*
  // passthrough hooks rather than replacing them, so per-SSR-request registration
  // re-ran the hook once per prior request in a long-lived `serve` — duplicating
  // target/rel and burning CPU. The module body runs once per process.
  marked.use({
    hooks: {
      postprocess: (html) =>
        DOMPurify.sanitize(html).replace(
          /<a /g,
          '<a target="_blank" rel="noopener noreferrer" ',
        ),
    },
  });
</script>

<script>
  import "../app.css";
  import { page } from "$app/stores";
  import {
    SEARCH_ICON,
    FINDINGS_ICON,
    DOCUMENTS_ICON,
    TAGS_ICON,
  } from "$lib/icons.js";
  export let data;

  // A nav link is "active" if the current path matches exactly OR is a child
  // of the link's target (so /findings/15 lights up the Findings nav too).
  // Home is exact-match only — otherwise it'd be active everywhere.
  //
  // isActive must be reactive ($:), not a plain function: Svelte only
  // re-evaluates `class:active={isActive(...)}` when a variable named in the
  // expression changes. A plain function references only itself (constant), so
  // the binding would be computed once and freeze — leaving whichever link was
  // active on first load stuck underlined across client-side navigations.
  $: path = $page.url.pathname;
  $: isActive = (href) =>
    href === "/" ? path === "/" : path === href || path.startsWith(href + "/");
</script>

<header>
  <nav class="bar">
    <a class="brand display" href="/" class:active={isActive("/")}>Bartleby</a>
    <div class="links">
      <a href="/search" class:active={isActive("/search")}><span class="nav-icon">{@html SEARCH_ICON}</span>Search</a>
      <a href="/findings" class:active={isActive("/findings")}><span class="nav-icon">{@html FINDINGS_ICON}</span>Findings</a>
      <a href="/documents" class:active={isActive("/documents")}><span class="nav-icon">{@html DOCUMENTS_ICON}</span>Documents</a>
      <a href="/tags" class:active={isActive("/tags")}><span class="nav-icon">{@html TAGS_ICON}</span>Tags</a>
    </div>
    <span class="project display">{data.project}</span>
  </nav>
</header>

<main>
  <slot />
</main>

<style>
  /* The machine shell's header: a dark scanlined band tinted with the brand
     sage. The scanline is a repeating 1px sage stripe over near-black — the
     "console" texture — kept faint so the chrome stays legible. */
  header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 10;
    background-color: var(--color-shell-raised);
    background-image: repeating-linear-gradient(
      to bottom,
      rgba(105, 136, 121, 0.12) 0,
      rgba(105, 136, 121, 0.12) 1px,
      transparent 1px,
      transparent 3px
    );
    border-bottom: 1px solid var(--color-off);
  }
  .bar {
    display: flex;
    align-items: baseline;
    gap: var(--space-xl);
    padding: var(--space-md) var(--space-lg);
  }
  /* The wordmark: amber dot-matrix with a soft amber glow — the lit nameplate
     on the console. Doto sits at --text-xl, well above the display floor. */
  .brand {
    font-size: var(--text-xl);
    font-weight: 600;
    color: var(--color-token-dark);
    text-shadow: 0 0 8px rgba(var(--color-token-dark-rgb), 0.55);
    text-decoration: none;
  }
  .links {
    display: flex;
    gap: var(--space-lg);
  }
  .links a {
    color: var(--color-shell-text);
    text-decoration: none;
    font-family: var(--font-sans);
    font-size: var(--text-base);
    padding-bottom: 2px;
    border-bottom: 2px solid transparent;
    /* Icon + label ride together on the chrome line. The link's own baseline
       still aligns with the wordmark/project (.bar is baseline-aligned). */
    display: inline-flex;
    align-items: center;
    gap: var(--space-2xs);
  }
  /* R3 (#592): the Pixelarticon sits just before each nav label, inheriting the
     shell-text color (it goes mint on hover/active with the label). The icon is
     a fixed square so it never reflows the wrap math B1 (#595) depends on. */
  .nav-icon {
    display: inline-flex;
    color: var(--color-shell-text-soft);
  }
  .links a:hover .nav-icon,
  .links a.active .nav-icon {
    color: var(--color-token-dark);
  }
  .links a:hover {
    color: var(--color-off-light);
  }
  .links a.active {
    color: var(--color-off-light);
    border-bottom-color: var(--color-token-dark);
  }
  .project {
    margin-left: auto;
    color: var(--color-shell-text-soft);
    font-size: var(--text-base);
  }
  main {
    /* Offset the fixed header (a fixed structural height) so content doesn't
       slide under it; the horizontal value is the shared page gutter. */
    padding: 4rem var(--space-lg) var(--space-2xl);
  }

  /* =========================================================================
     B1 · Mobile nav — responsive wrap at ≤480px (#595)
     Strategy: wrap .bar to two lines so every nav destination stays reachable
     at 375px with no page-level horizontal scroll.
       Row 1: wordmark (.brand) + project badge (.project, right-aligned)
       Row 2: nav links (.links), spanning full width
     No JS, no hamburger — all four links stay visible and tappable.
     NOTE: R3 (#592) will add Pixelarticons; keep link markup clean.
     ========================================================================= */
  @media (max-width: 480px) {
    .bar {
      flex-wrap: wrap;
      gap: var(--space-sm) var(--space-lg);
      /* Reduce vertical padding so the two-line header doesn't consume too
         much of the small viewport. */
      padding: var(--space-sm) var(--space-md);
    }
    /* Row 1: brand fills available space; .project stays at the right end via
       margin-left:auto inherited from the default rule above. */
    .brand {
      /* Let the wordmark shrink but keep it above the dot-matrix floor so
         Doto characters stay legible (--text-display-floor = 1.25rem). */
      font-size: var(--text-display-floor);
    }
    /* Row 2: links row takes the full bar width, baseline-aligns with smaller
       gaps so all four items fit across 375px. */
    .links {
      width: 100%;
      gap: var(--space-md);
    }
    .links a {
      font-size: var(--text-sm);
      /* Tighter icon↔label gap so four icon+label pairs still fit the 375px
         row without forcing a third line (B1 #595 wrap math). */
      gap: var(--space-3xs);
    }
    /* Shrink the glyph a hair on the narrow row to keep the links compact. */
    .nav-icon :global(.pixel) {
      width: 13px;
      height: 13px;
    }
    main {
      /* Taller offset: two-line header is ~3rem taller at mobile. */
      padding-top: 6.5rem;
    }
  }
</style>
