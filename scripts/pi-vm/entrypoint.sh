#!/usr/bin/env bash
#
# In-VM entrypoint. Runs inside the Apple `container` Linux guest.
#   1. start the VM-local Ollama (CPU) that serves decant's distill model
#   2. detect the host bridge IP so Pi can reach the host agent Ollama
#   3. render Pi's models.json (-> host :11435) and decant's config (-> VM-local)
#   4. hand off to Pi (interactive) or whatever command was passed
#
# See docs/pi-vm-runbook.md.
set -euo pipefail

# 1. VM-local Ollama (serves decant's small model, CPU-only — expected).
ollama serve >/tmp/ollama.log 2>&1 &
until ollama list >/dev/null 2>&1; do sleep 0.5; done

# 2. Host bridge IP. Apple `container` has no host.docker.internal; the default
#    gateway points at the host bridge. Override with HOST_OLLAMA_IP if wrong.
HOST_IP="${HOST_OLLAMA_IP:-$(ip route 2>/dev/null | awk '/default/ {print $3; exit}')}"
AGENT_PORT="${AGENT_OLLAMA_PORT:-11435}"
if [ -z "${HOST_IP}" ]; then
  echo "WARN: could not detect host gateway IP; set HOST_OLLAMA_IP." >&2
  HOST_IP="127.0.0.1"
fi

# 3a. Pi -> host agent Ollama.
mkdir -p /root/.pi/agent
sed "s|__HOST_OLLAMA__|http://${HOST_IP}:${AGENT_PORT}/v1|g" \
    /opt/pi-vm/models.json > /root/.pi/agent/models.json

# 3b. decant -> VM-local Ollama. Brave key from env if provided, else null.
mkdir -p /root/.decant
sed "s|__BRAVE_KEY__|${BRAVE_SEARCH_API_KEY:-null}|g" \
    /opt/pi-vm/decant-config.yaml > /root/.decant/config.yaml

echo "Pi agent model -> http://${HOST_IP}:${AGENT_PORT}  |  decant -> VM-local Ollama" >&2

# 4. Hand off.
if [ "$#" -eq 0 ]; then
  exec pi
else
  exec "$@"
fi
