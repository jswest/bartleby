#!/usr/bin/env bash
#
# Launch the research VM. Mounts ONLY the corpus dir (read-write — the agent
# writes findings/tags through Bartleby's typed helpers). Everything else (Pi,
# Bartleby, decant, the distill model) is baked into the image.
#
# Prereq: the host agent Ollama must be running first:
#   ./scripts/pi-vm/host-agent-ollama.sh
#
# Env:
#   IMAGE               image tag           (default: bartleby-pi:latest)
#   BARTLEBY_HOME       host corpus root    (default: ~/.bartleby)
#   AGENT_OLLAMA_PORT   host agent port     (default: 11435)
#   HOST_OLLAMA_IP      override gateway autodetect (optional)
#   BRAVE_SEARCH_API_KEY  passed through for `decant search` (optional)
#
# Pass a command to run instead of interactive Pi, e.g.:
#   ./run.sh pi -p "Summarize project X"
#   ./run.sh bash
#
# See docs/pi-vm-runbook.md.
set -euo pipefail

IMAGE="${IMAGE:-bartleby-pi:latest}"
CORPUS="${BARTLEBY_HOME:-${HOME}/.bartleby}"
AGENT_PORT="${AGENT_OLLAMA_PORT:-11435}"

command -v container >/dev/null || {
  echo "Apple 'container' not found. Install with: brew install container" >&2
  exit 1
}
[ -d "${CORPUS}" ] || {
  echo "Corpus dir ${CORPUS} not found; set BARTLEBY_HOME to your corpus root." >&2
  exit 1
}

ARGS=(run -it --rm --name bartleby-pi
      -v "${CORPUS}:/corpus"
      -e "AGENT_OLLAMA_PORT=${AGENT_PORT}")
[ -n "${HOST_OLLAMA_IP:-}" ]      && ARGS+=(-e "HOST_OLLAMA_IP=${HOST_OLLAMA_IP}")
[ -n "${BRAVE_SEARCH_API_KEY:-}" ] && ARGS+=(-e "BRAVE_SEARCH_API_KEY=${BRAVE_SEARCH_API_KEY}")

exec container "${ARGS[@]}" "${IMAGE}" "$@"
