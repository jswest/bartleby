#!/usr/bin/env bash
#
# Build the research-VM image with Apple `container`. Bartleby (latest release
# tag) and decant (main) are pulled from GitHub inside the build, so there's no
# local source to stage — the build context is just this directory. Re-run to
# pick up a newer Bartleby release / newer decant main.
#
# Env:
#   IMAGE        image tag                     (default: bartleby-pi:latest)
#   DECANT_MODEL distill model baked into VM   (default: gemma4:e2b)
#
# See docs/pi-vm-runbook.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-bartleby-pi:latest}"
DECANT_MODEL="${DECANT_MODEL:-gemma4:e2b}"

command -v container >/dev/null || {
  echo "Apple 'container' not found. Install with: brew install container" >&2
  exit 1
}

echo "Building ${IMAGE} (bartleby=latest release, decant=main, DECANT_MODEL=${DECANT_MODEL}) ..." >&2
container build \
  --build-arg "DECANT_MODEL=${DECANT_MODEL}" \
  -t "${IMAGE}" \
  -f "${HERE}/Containerfile" \
  "${HERE}"

echo "Built ${IMAGE}. Next: ./scripts/pi-vm/run.sh" >&2
