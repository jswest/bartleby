#!/usr/bin/env bash
#
# Host-side: launch a DEDICATED Ollama instance for the Pi agent, separate from
# your daily-driver Ollama on :11434. This one runs on the Apple GPU (Metal) and
# serves the big agent model fast. It is a distinct process/port, so it does not
# touch your interactive Ollama and can be stopped independently.
#
# It binds to 0.0.0.0 so the Apple `container` VM can reach it (Apple container
# has no host.docker.internal). That also exposes it to your LAN — enable the
# macOS firewall or bind to the vmnet bridge IP if that matters to you.
#
# Usage:
#   ./host-agent-ollama.sh [--model <id>]
#
# Options:
#   --model <id>        Model to serve (overrides AGENT_MODEL env var).
#                       Must match what run.sh requests (use --model on both).
#
# Env:
#   AGENT_MODEL         Model to serve      (default: qwen3.6:35b). --model takes
#                       precedence if given.
#   AGENT_OLLAMA_PORT   Port to bind on     (default: 11435)
#   AGENT_OLLAMA_BIND   Full bind address   (default: 0.0.0.0:<port>)
#
# See docs/pi-vm-runbook.md.
set -euo pipefail

# Parse --model; this script takes no other args, so reject anything else.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   [ $# -ge 2 ] || { echo "--model requires a value" >&2; exit 1; }
               AGENT_MODEL="$2"; shift 2 ;;
    --model=*) AGENT_MODEL="${1#--model=}"; shift ;;
    *)         echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PORT="${AGENT_OLLAMA_PORT:-11435}"
MODEL="${AGENT_MODEL:-qwen3.6:35b}"
BIND="${AGENT_OLLAMA_BIND:-0.0.0.0:${PORT}}"

command -v ollama >/dev/null || { echo "ollama not found on PATH" >&2; exit 1; }

echo "Starting dedicated agent Ollama on ${BIND} (model: ${MODEL})..." >&2
OLLAMA_HOST="${BIND}" ollama serve &
SRV=$!
trap 'kill "${SRV}" 2>/dev/null || true' EXIT INT TERM

# Wait for the server, then make sure the model is present (instant if the blob
# is already in your shared ~/.ollama model store).
until OLLAMA_HOST="${BIND}" ollama list >/dev/null 2>&1; do sleep 0.5; done
OLLAMA_HOST="${BIND}" ollama pull "${MODEL}"

echo "Agent Ollama ready on ${BIND}. Leave this running; Ctrl-C to stop." >&2
wait "${SRV}"
