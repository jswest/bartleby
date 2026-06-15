#!/usr/bin/env bash
#
# In-VM entrypoint. Runs inside the Apple `container` Linux guest.
#   1. (if decant is baked in) start the VM-local Ollama (CPU) for its model
#   2. detect the host bridge IP so Pi can reach the host agent Ollama
#   3. render Pi's models.json (-> host :11435) and, if present, decant's config
#   4. hand off to Pi (interactive) or whatever command was passed
#
# Decant is optional at build time (WITH_DECANT=0); we detect it at runtime so
# the same entrypoint works for both the full and the lean image.
#
# See docs/pi-vm-runbook.md.
set -euo pipefail

# Lean image (WITH_DECANT=0) has neither decant nor the VM-local Ollama.
if command -v decant >/dev/null 2>&1 && command -v ollama >/dev/null 2>&1; then
  HAS_DECANT=1
else
  HAS_DECANT=0
fi

# 1. VM-local Ollama (serves decant's small model, CPU-only — expected).
if [ "${HAS_DECANT}" = "1" ]; then
  ollama serve >/tmp/ollama.log 2>&1 &
  until ollama list >/dev/null 2>&1; do sleep 0.5; done
fi

# 2. Host bridge IP. Apple `container` has no host.docker.internal; the default
#    gateway points at the host bridge. Override with HOST_OLLAMA_IP if wrong.
HOST_IP="${HOST_OLLAMA_IP:-$(ip route 2>/dev/null | awk '/default/ {print $3; exit}')}"
AGENT_PORT="${AGENT_OLLAMA_PORT:-11435}"
if [ -z "${HOST_IP}" ]; then
  echo "WARN: could not detect host gateway IP; set HOST_OLLAMA_IP." >&2
  HOST_IP="127.0.0.1"
fi

# 3a. Pi -> host agent Ollama (model id overridable via AGENT_MODEL).
AGENT_MODEL="${AGENT_MODEL:-qwen3.6:35b}"
mkdir -p /root/.pi/agent
sed -e "s|__HOST_OLLAMA__|http://${HOST_IP}:${AGENT_PORT}/v1|g" \
    -e "s|__AGENT_MODEL__|${AGENT_MODEL}|g" \
    /opt/pi-vm/models.json > /root/.pi/agent/models.json

# 3b. decant -> VM-local Ollama. Brave key from env if provided, else null.
if [ "${HAS_DECANT}" = "1" ]; then
  mkdir -p /root/.decant
  sed "s|__BRAVE_KEY__|${BRAVE_SEARCH_API_KEY:-null}|g" \
      /opt/pi-vm/decant-config.yaml > /root/.decant/config.yaml
  echo "Pi -> ${AGENT_MODEL} @ http://${HOST_IP}:${AGENT_PORT}  |  decant -> VM-local Ollama" >&2
else
  echo "Pi -> ${AGENT_MODEL} @ http://${HOST_IP}:${AGENT_PORT}  |  decant: not installed (lean image)" >&2
fi

# Cross-kernel WAL warning: while this container holds the corpus, the host must
# not touch it (a host serve goes malformed-or-stale). See docs/pi-vm-runbook.md.
echo "WARN: while this container runs, do not access this corpus from the host -- stop 'bartleby serve' for this project." >&2

# 4. Hand off.
if [ "$#" -eq 0 ]; then
  exec pi
else
  exec "$@"
fi
