# Running Bartleby with Pi in an isolated VM

A runbook for driving the Bartleby research skill with [Pi](https://pi.dev) and
local Ollama models, while keeping Pi's unsandboxed shell off your Mac.

> **Status: first-draft, needs a shakeout run.** The helper scripts in
> [`scripts/pi-vm/`](../scripts/pi-vm/) encode the design below, but they have
> not yet been run end-to-end on a clean machine. Three things are most likely
> to need adjustment on first use: Apple `container`'s host-networking (reaching
> the host Ollama), where the Pi installer puts the `pi` binary on `PATH`, and
> the in-build model pull. Each is flagged inline below and in the scripts.

---

## Why bother

Pi is a *minimal* agent harness: four core tools (`read`, `write`, `edit`,
`bash`) and **no permission gates or filesystem/network sandboxing by default**
([pi.dev](https://pi.dev)). That's a deliberate design choice — you adapt Pi to
your workflow — but it means a local model gets an unconfined shell on whatever
machine Pi runs on. With Claude Code you'd have both Anthropic-grade alignment
*and* the permission system in front of `bash`; with Pi + a local model you have
neither. The VM is the substitute for both.

So: **run Pi inside a disposable Linux VM**, mount only the one corpus it should
touch, and keep the blast radius to the box.

## The constraint that shapes everything

You **cannot** usefully run Ollama *inside* a Linux VM/container on Apple
Silicon. The Apple GPU can't be passed through to a Linux guest (macOS holds it
for the display), so containerized Ollama falls back to **CPU-only inference —
3–5× slower** ([ollama#5652](https://github.com/ollama/ollama/issues/5652)).
For a 35B agent model that's unusable.

The resolution: **isolate the agent, not the inference.** The model just emits
text; the danger is the agent *acting* on that text (`bash`/`write`/`edit`). So
inference stays where it's fast (the host GPU), and the agent's hands stay in
the box.

## Topology

```
┌─ macOS host (M4 Max, Metal GPU) ──────────────────────────────────┐
│                                                                     │
│  ollama :11434   ← your daily-driver instance (untouched)           │
│  ollama :11435   ← DEDICATED agent instance, qwen3.6:35b  (GPU)     │ ← host-agent-ollama.sh
│       ▲                                                             │
│       │ only VM→host path: Pi → agent model                        │
│  ┌────┼──────────────────── Apple `container` VM ────────────────┐ │
│  │    │                                                           │ │
│  │   Pi ──bash/write/edit──▶ confined to the VM filesystem        │ │
│  │    └─ runs `bartleby skill …` over the mounted corpus DB        │ │
│  │                                                                 │ │
│  │   ollama :11434 (VM-local, CPU) ◀── decant, gemma4:e2b          │ │
│  │   decant ──▶ web (outbound NAT) ──▶ Brave + fetched URLs         │ │
│  │                                                                 │ │
│  │   mount: host ~/.bartleby  →  /corpus   (read-write)            │ │
│  └─────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

Two Ollama roles, deliberately split:

| Role | Where | Model | Why |
| --- | --- | --- | --- |
| **Agent** (drives Pi) | Host, dedicated instance `:11435` | `qwen3.6:35b` | Big model → needs the GPU. A *separate* process/port from your daily `:11434`, independently start/stoppable. |
| **decant distill** | VM-local `:11434` (CPU) | `gemma4:e2b` | Small model, summarization is its whole job. Keeping it in the VM means raw web content never reaches the host. |

### Can I run Ollama on the host *and* in the VM at once?

Yes. They're separate processes in separate network namespaces — no port or
socket clash, and they happily share the on-disk model blobs. The only cost is
that the in-VM instance is CPU-only, which is exactly why we give it the small
`gemma4:e2b` distill model and leave the 35B agent model on the host GPU.

### Model picks (from `bartleby benchmark`)

The benchmark ranks models on a *summarization* task. Top local finishers:
`gemma4:31b` (4.75/5), `qwen3.6:35b` (4.72/5 at ~175 tok/s), `gemma4:e2b`
(4.69/5 at ~585 tok/s). Two caveats drove the picks here:

- **Agent ≠ summarizer.** An agent loop is many multi-turn *tool calls*, and
  tool-calling reliability is not what the summarization benchmark measures. The
  README already warns local models follow tool-use protocols less reliably. The
  qwen3 family is a steadier tool-caller than gemma4, and at ~175 tok/s it's
  ~3.7× faster than `gemma4:31b` — throughput compounds over a long loop. So the
  **agent** model is `qwen3.6:35b` (swap to `gpt-oss:120b` for more reasoning
  headroom; you have the RAM).
- **decant's job *is* summarization**, so the benchmark transfers cleanly:
  `gemma4:e2b` at 4.69/5 and ~585 tok/s is the ideal distill model — and small
  enough to tolerate CPU-only inference in the VM.

## Security posture

What this buys you, and what it doesn't:

- **Pi's `bash` is confined to the VM.** A confused or jailbroken local model
  can't read your SSH keys, touch other repos, or `rm -rf $HOME`. Worst case is
  a trashed container, which you rebuild.
- **Only the corpus is mounted** — `~/.bartleby` → `/corpus`, read-write
  (the agent writes findings/tags, which go through Bartleby's typed `chunks`
  helpers, never raw SQL). Nothing else of your home directory is visible. With
  a **pure-build** image (the choice here) the agent never even sees the
  Bartleby or decant *source* — it's baked in, not mounted.
- **decant's web access stays in the box.** Raw fetched pages and their
  summarization never leave the VM; only what the agent pulls back via the skill
  crosses the mount.

### Will decant reach the web outside the box?

**Yes — outbound internet works.** Apple `container` gives containers NAT
internet by default, so decant reaches Brave and the URLs you fetch. Outbound is
the easy direction; the *hard* direction on Apple `container` is reaching host
services (see the networking note below), which is why the agent-Ollama hop
needs explicit wiring while decant's web access just works.

**The trade-off to know:** Pi shares the container's network namespace, so if
decant can reach the web, **so can Pi's `bash`** — the agent could phone home or
exfiltrate the corpus. That's acceptable here because the box holds no host
secrets and only your corpus, but if you want to close it:

- Run with an **egress allowlist** — a proxy or firewall that permits only
  Brave's API + the domains you're researching, and the host agent-Ollama IP.
- Or accept the open egress and rely on the filesystem isolation (the corpus is
  the only sensitive thing in the box, and you put it there on purpose).

This is a deliberate, documented gap rather than a silent one. Tightening egress
is left as a follow-up; the filesystem boundary is the load-bearing protection.

## Prerequisites

- **macOS 26+** (you're on 26.3 — full Apple Containerization support; macOS 15
  is degraded).
- **Apple `container`**: `brew install container` (formula, not a cask).
- **Ollama on the host** with `qwen3.6:35b` and `gemma4:e2b` pulled
  (`ollama pull qwen3.6:35b` — `gemma4:e2b` is re-pulled inside the VM at build
  time, so the host copy is optional).
- A **Brave Search API key** if you want `decant search` (not needed for
  `decant url`). Pass it as `BRAVE_SEARCH_API_KEY`, or leave it in your host
  `~/.decant/config.yaml` — `run.sh` reads the value from there as a fallback
  (the config file itself is never mounted into the box).
- The **decant** source at `~/Code/decant` (override with `DECANT_SRC`).

## The Apple `container` networking note

This is the one genuinely fiddly part. Apple `container` does **not** provide
`host.docker.internal`, and reaching host services from a container is a known
gap ([apple/container#346](https://github.com/apple/container/issues/346)).
Two moving parts:

1. **Host side** — the dedicated agent Ollama must bind somewhere the VM can
   reach, not `127.0.0.1`. [`host-agent-ollama.sh`](../scripts/pi-vm/host-agent-ollama.sh)
   binds it to `0.0.0.0:11435`. That also exposes it to your LAN; if that
   bothers you, enable the macOS firewall or bind to the vmnet bridge IP only.
2. **VM side** — the container reaches the host via its **default gateway**,
   which points at the host bridge. [`entrypoint.sh`](../scripts/pi-vm/entrypoint.sh)
   auto-detects it with `ip route` and renders Pi's `models.json` to
   `http://<gateway>:11435/v1`. If auto-detection picks the wrong address, pass
   `HOST_OLLAMA_IP=<addr>` to `run.sh` to override.

If the host hop proves painful, **colima** is the fallback — it gives you
`host.docker.internal` for free at the cost of Apple `container`'s cleaner
VM-per-container model.

## Build vs. mount: why pure build

This setup is **pure build** — the image bakes Pi + Bartleby + decant + the
distill model + both configs, and mounts *only* the corpus. The alternative
(bind-mounting the Bartleby/decant source for live editing) was rejected because
it would let the agent see and modify your tool source, which partly defeats the
isolation. Pure build gives a reproducible, disposable image and the tightest
blast radius. The cost is a rebuild when Bartleby or decant changes — run
`build.sh` again.

## Step by step

```bash
# 0. one-time: install Apple container
brew install container
container system start          # starts the container runtime

# 1. host: start the dedicated agent Ollama (GPU), separate from your daily one
./scripts/pi-vm/host-agent-ollama.sh          # serves qwen3.6:35b on :11435

# 2. build the research-VM image (stages bartleby + ~/Code/decant source)
./scripts/pi-vm/build.sh                       # -> bartleby-pi:latest

# 3. run it — mounts ~/.bartleby as /corpus, drops you into Pi
BRAVE_SEARCH_API_KEY=brv-... ./scripts/pi-vm/run.sh

#    or run Pi non-interactively:
./scripts/pi-vm/run.sh pi -p "Summarize the strongest findings in project X"
```

Inside the VM, Pi sees the Bartleby skill (installed to `~/.pi/agent/skills/`
at build time) and drives it against `/corpus` exactly as Claude Code would.

## Verifying the wiring

Once you're in the container shell (`./scripts/pi-vm/run.sh bash`):

```bash
# agent Ollama reachable on the host?
curl -s "http://$(ip route | awk '/default/{print $3}'):11435/v1/models" | head

# decant's local Ollama up?
curl -s http://127.0.0.1:11434/api/tags | head

# decant can reach the web?
decant url https://example.com --question "what is this page"

# Bartleby sees the corpus?
bartleby project list

# Pi sees the skill?
pi --help            # confirm `pi` is on PATH; see shakeout note if not
```

## Shakeout notes (things to confirm on first run)

- **`pi` on `PATH`.** The curl installer (`https://pi.dev/install.sh`) may drop
  the binary somewhere not on the default `PATH`. If `pi` isn't found, either
  symlink it into `/usr/local/bin` in the `Containerfile` or switch to the npm
  install (`npm install -g --ignore-scripts @earendil-works/pi-coding-agent`,
  which needs Node in the image).
- **In-build model pull.** The `Containerfile` runs `ollama serve` in the
  background during build to `ollama pull gemma4:e2b`. If that's flaky, move the
  pull to `entrypoint.sh` (first-run) backed by a persistent volume — at the
  cost of pure-build self-containment.
- **Gateway detection.** If `ip route` inside the VM doesn't yield a
  host-reachable address, set `HOST_OLLAMA_IP` explicitly (see networking note).
- **`container system start`.** Apple `container` needs its runtime started once
  per boot before `build`/`run`.

## Files

| File | Role |
| --- | --- |
| [`scripts/pi-vm/host-agent-ollama.sh`](../scripts/pi-vm/host-agent-ollama.sh) | Host: dedicated agent Ollama on `:11435` (GPU) |
| [`scripts/pi-vm/build.sh`](../scripts/pi-vm/build.sh) | Stage source + build the image (pure build) |
| [`scripts/pi-vm/Containerfile`](../scripts/pi-vm/Containerfile) | The research-VM image |
| [`scripts/pi-vm/entrypoint.sh`](../scripts/pi-vm/entrypoint.sh) | In-VM: start local Ollama, render configs, launch Pi |
| [`scripts/pi-vm/run.sh`](../scripts/pi-vm/run.sh) | Run the VM, mount the corpus |
| [`scripts/pi-vm/models.json`](../scripts/pi-vm/models.json) | Pi provider config (→ host agent Ollama) |
| [`scripts/pi-vm/decant-config.yaml`](../scripts/pi-vm/decant-config.yaml) | decant config (→ VM-local Ollama) |
