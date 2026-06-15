<script>
  // WaitingIndicator — a shared ouroboros chase for any waiting / loading state.
  //
  // Props:
  //   label   string   Text shown beside the indicator. E.g. "Searching…",
  //                    "Loading…", "awaiting query". Caller owns the copy.
  //   active  boolean  true  → ouroboros animates (system is busy).
  //                    false → static dim frame (idle / waiting for user input).
  //
  // Usage:
  //   <WaitingIndicator label="Searching…" active={true}  />   ← active search
  //   <WaitingIndicator label="awaiting query" active={false} /> ← idle empty state
  //   <WaitingIndicator label="Loading…" active={true}  />   ← #621 navigation
  //
  // The 4 Doto block cells are arranged in a clockwise square track:
  //   [0] top-left  [1] top-right
  //   [3] bot-left  [2] bot-right
  // When active, a "lit" highlight sweeps clockwise (0→1→2→3→0…) using
  // animation-delay offsets. When idle the track dims to shell-text-soft; no
  // motion. prefers-reduced-motion: holds a static frame regardless of active.

  export let label = "awaiting query";
  export let active = false;
</script>

<span
  class="waiting-indicator"
  class:active
  aria-label={label}
  aria-live={active ? "polite" : undefined}
  aria-busy={active}
  role="status"
>
  <!--
    4-cell ouroboros track. Order matches clockwise sweep:
      cell-0 (top-left) → cell-1 (top-right) → cell-2 (bot-right) → cell-3 (bot-left)
    CSS animation-delay staggers each cell's brightness peak by one quarter cycle.
  -->
  <span class="track" aria-hidden="true">
    <span class="cell cell-0">█</span><span class="cell cell-1">█</span><!--
    --><span class="cell cell-3">█</span><span class="cell cell-2">█</span>
  </span>
  <span class="label">{label}</span>
</span>

<style>
  /* Container: inline-flex so it fits inside StatusBanner copy or a <p>. */
  .waiting-indicator {
    display: inline-flex;
    align-items: baseline;
    gap: var(--space-sm);
    /* Anchor font-family and size so the Doto cells share metrics with the label. */
  }

  /* The 2×2 block track, rendered in Doto dot-matrix. Laid out as a 2-column
     inline-grid so the four cells form a square regardless of font metrics.
     Fixed dimensions prevent layout jitter — width is exactly 2 Doto "cells". */
  .track {
    font-family: var(--font-display);
    font-size: max(var(--text-display-floor), 0.9em);
    line-height: 0.9;
    display: inline-grid;
    grid-template-columns: repeat(2, 1ch);
    grid-template-rows: repeat(2, 0.9em);
    /* Align baseline of the bottom row with the label baseline. */
    align-self: baseline;
  }

  .cell {
    display: block;
    /* Idle: dim cells (system is not busy). */
    opacity: 0.18;
    transition: opacity 0.1s;
  }

  /* ── Active / chase animation ──────────────────────────────────────────────
     Clockwise sweep: cell-0 → cell-1 → cell-2 → cell-3 → repeat.
     Each cell peaks at opacity 1, decays to ~0.12, over a shared 1.6 s cycle.
     Stagger = cycle / 4 = 0.4 s per step. No steps(); easing is ease-in-out
     so the peak is soft — no hard on/off, no strobe.
  */
  @keyframes ouroboros-pulse {
    0%   { opacity: 1;    }
    20%  { opacity: 0.12; }
    75%  { opacity: 0.12; }
    100% { opacity: 1;    }
  }

  .active .cell {
    animation: ouroboros-pulse 1.6s ease-in-out infinite;
  }

  /* Clockwise stagger: cell-0 leads, each subsequent cell is +0.4 s behind. */
  .active .cell-0 { animation-delay: 0s; }
  .active .cell-1 { animation-delay: -1.2s; } /* 0.4 s into cycle */
  .active .cell-2 { animation-delay: -0.8s; } /* 0.8 s into cycle */
  .active .cell-3 { animation-delay: -0.4s; } /* 1.2 s into cycle */

  /* Label: Iosevka (UI chrome). Inherits parent color (shell-text or banner). */
  .label {
    font-family: var(--font-sans);
    font-size: inherit;
  }

  /* Reduced-motion: hold a static frame — one cell lit, no animation. */
  @media (prefers-reduced-motion: reduce) {
    .active .cell {
      animation: none;
    }
    /* Keep cell-0 lit as the static "active" indicator. */
    .active .cell-0 {
      opacity: 1;
    }
    .active .cell-1,
    .active .cell-2,
    .active .cell-3 {
      opacity: 0.18;
    }
  }
</style>
