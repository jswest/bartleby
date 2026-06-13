#!/usr/bin/env bash
#
# Pure build: stage the local Bartleby + decant source into a clean context and
# build the research-VM image with Apple `container`. Re-run whenever Bartleby
# or decant changes (pure build means no live source mount).
#
# Env:
#   DECANT_SRC   path to the decant repo      (default: ~/Code/decant)
#   IMAGE        image tag                     (default: bartleby-pi:latest)
#   DECANT_MODEL distill model baked into VM   (default: gemma4:e2b)
#
# See docs/pi-vm-runbook.md.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"          # bartleby repo root
DECANT_SRC="${DECANT_SRC:-${HOME}/Code/decant}"
IMAGE="${IMAGE:-bartleby-pi:latest}"
DECANT_MODEL="${DECANT_MODEL:-gemma4:e2b}"

command -v container >/dev/null || {
  echo "Apple 'container' not found. Install with: brew install container" >&2
  exit 1
}
[ -d "${DECANT_SRC}" ] || {
  echo "decant source not found at ${DECANT_SRC}; set DECANT_SRC." >&2
  exit 1
}

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
echo "Staging build context in ${STAGE} ..." >&2

EXCLUDES=(--exclude '.git' --exclude '.venv' --exclude '__pycache__'
          --exclude '.pytest_cache' --exclude 'node_modules')

rsync -a "${EXCLUDES[@]}" \
      --exclude 'benchmarks/results' --exclude 'benchmarks/judgements' \
      "${REPO_ROOT}/" "${STAGE}/bartleby/"
rsync -a "${EXCLUDES[@]}" "${DECANT_SRC}/" "${STAGE}/decant/"

# pi-vm/ holds the Containerfile + configs + entrypoint the Containerfile COPYs.
mkdir -p "${STAGE}/pi-vm"
cp "${HERE}/models.json" "${HERE}/decant-config.yaml" "${HERE}/entrypoint.sh" \
   "${STAGE}/pi-vm/"
cp "${HERE}/Containerfile" "${STAGE}/Containerfile"

echo "Building ${IMAGE} (DECANT_MODEL=${DECANT_MODEL}) ..." >&2
container build \
  --build-arg "DECANT_MODEL=${DECANT_MODEL}" \
  -t "${IMAGE}" \
  -f "${STAGE}/Containerfile" \
  "${STAGE}"

echo "Built ${IMAGE}. Next: ./scripts/pi-vm/run.sh" >&2
