#!/usr/bin/env bash
#
# Build the research-VM image with Apple `container`. Bartleby (latest release
# tag) and decant (main) are pulled from GitHub inside the build, so there's no
# local source to stage — the build context is just this directory. Re-run to
# pick up a newer Bartleby release / newer decant main.
#
# Self-cleaning so it can't silently bloat your disk (see "Disk & cleanup" in the
# runbook): deletes the prior image before building and resets the BuildKit cache
# afterward. During development that cache grew to 75 GB — and it's invisible to
# `container system df`, so nothing warns you.
#
# Env:
#   IMAGE             image tag                   (default: bartleby-pi:latest)
#   WITH_DECANT       set=1 to add decant, a VM-local Ollama, and the ~7.2 GB
#                     distill model for in-VM web fetch/search. Default: 0 — a
#                     lean, corpus-only image with no web reach.
#   DECANT_MODEL      distill model baked into VM (default: gemma4:e2b; ignored
#                     when WITH_DECANT=0).
#   KEEP_BUILD_CACHE  set=1 to keep the BuildKit cache after a successful build
#                     (fast incremental rebuilds while iterating on the
#                     Containerfile). Default: reset it so it can't accumulate.
#
# See docs/pi-vm-runbook.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-bartleby-pi:latest}"
WITH_DECANT="${WITH_DECANT:-0}"
DECANT_MODEL="${DECANT_MODEL:-gemma4:e2b}"

command -v container >/dev/null || {
  echo "Apple 'container' not found. Install with: brew install container" >&2
  exit 1
}

# Drop any prior image first: each rebuild re-tags :latest and orphans the old
# digest's layers, which `container` has no reliable prune for. Deleting up front
# keeps exactly one image on disk. (If this build fails, just re-run to recover.)
container image delete "${IMAGE}" >/dev/null 2>&1 || true

if [ "${WITH_DECANT}" = "1" ]; then
  echo "Building ${IMAGE} (bartleby=latest release, decant=main, DECANT_MODEL=${DECANT_MODEL}) ..." >&2
else
  echo "Building ${IMAGE} (bartleby=latest release, WITH_DECANT=0 — lean, no decant/Ollama/model) ..." >&2
fi
container build \
  --build-arg "WITH_DECANT=${WITH_DECANT}" \
  --build-arg "DECANT_MODEL=${DECANT_MODEL}" \
  -t "${IMAGE}" \
  -f "${HERE}/Containerfile" \
  "${HERE}"

# Reset the BuildKit cache unless we're iterating. Its storage lives inside the
# builder container — invisible to `container system df` — so it grows silently
# with every build. The builder recreates itself automatically on the next build.
if [ -z "${KEEP_BUILD_CACHE:-}" ]; then
  echo "Resetting BuildKit cache (set KEEP_BUILD_CACHE=1 to keep it for fast rebuilds) ..." >&2
  container builder delete >/dev/null 2>&1 || true
fi

echo "Built ${IMAGE}. Next: ./scripts/pi-vm/run.sh" >&2
