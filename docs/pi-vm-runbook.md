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

**Trying a different agent model.** The agent model is a *runtime* knob — pass
`--model` to **both** the host serve and the run (no rebuild; it must match a
model your host Ollama serves):

```bash
./scripts/pi-vm/host-agent-ollama.sh --model gpt-oss:120b   # host GPU serves it
./scripts/pi-vm/run.sh --model gpt-oss:120b                 # Pi requests it
```

You can also use the `AGENT_MODEL` environment variable if you prefer:

```bash
AGENT_MODEL=gpt-oss:120b ./scripts/pi-vm/host-agent-ollama.sh
AGENT_MODEL=gpt-oss:120b ./scripts/pi-vm/run.sh
```

The `--model` flag takes precedence over `AGENT_MODEL` if both are set.
Defaults to `qwen3.6:35b` if neither is given. (decant's distill model is a
*build-time* knob instead — `DECANT_MODEL=<model> ./scripts/pi-vm/build.sh` —
since it's baked into the image.)

### Web fetch is opt-in: add decant with `WITH_DECANT=1`

decant — and the VM-local Ollama plus the baked distill model that exist only to
serve it — is the largest thing in the image, so the **default build is lean**:
Pi + Bartleby against `/corpus`, no decant, no web reach. That fits corpus-only
runs, where the agent searches/reads/cites the already-ingested corpus. When you
want in-VM web fetch/search, build it in:

```bash
WITH_DECANT=1 ./scripts/pi-vm/build.sh
```

That adds the decant install, the VM-local Ollama, and the ~7.2 GB `gemma4:e2b`
pull. The entrypoint detects decant at runtime, so the same entrypoint serves
both the lean and full images. Default is `WITH_DECANT=0` (lean); the trade-off
is exactly that capability — a lean image has **no in-VM web fetch/search**.

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

No local Bartleby/decant checkout is needed — the build pulls both from GitHub
(Bartleby's latest release tag, decant's `main`).

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
blast radius.

Bartleby and decant are pulled **from GitHub at build time** — Bartleby's latest
release tag (resolved via `git ls-remote`) and decant's `main` — not from your
local working copy. That keeps the build context tiny and the image pinned to
*published* code, which is the right default for a VM whose job is to *run*
research over an already-ingested corpus (ingestion happens on the host, so the
image needs no `[docling]`/`[sec2md]` extras). The trade-off: the VM runs
published versions, not your in-flight local edits to Bartleby/decant — to test
those you'd push/release first. Re-run `build.sh` to pick up a newer release or
newer decant `main`.

## Step by step

```bash
# 0. one-time: install Apple container
brew install container
container system start          # starts the container runtime

# 1. host: start the dedicated agent Ollama (GPU), separate from your daily one
./scripts/pi-vm/host-agent-ollama.sh          # serves qwen3.6:35b on :11435

# 2. build the research-VM image (lean by default: Bartleby release, no decant)
./scripts/pi-vm/build.sh                       # -> bartleby-pi:latest
#    need in-VM web fetch? add decant + its ~7.2 GB model: WITH_DECANT=1 ./scripts/pi-vm/build.sh

# 3. run it — mounts ~/.bartleby as /corpus, drops you into Pi
BRAVE_SEARCH_API_KEY=brv-... ./scripts/pi-vm/run.sh

#    or run Pi non-interactively:
./scripts/pi-vm/run.sh pi -p "Summarize the strongest findings in project X"
```

Inside the VM, Pi sees the Bartleby skill (installed to `~/.pi/agent/skills/`
at build time) and drives it against `/corpus` exactly as Claude Code would.

A couple of things to expect while running these:

- **Step 0, first run:** `container system start` reports "No default kernel
  configured" and offers to download the recommended guest Linux kernel (the
  Kata Containers static kernel) — answer `Y` (one-time; the Linux VM needs a
  kernel).
- **Step 1 stays running:** `host-agent-ollama.sh` runs `ollama serve` in the
  **foreground**, so keep it in its own terminal window for the whole session
  (`Ctrl-C` stops it). On macOS it also surfaces the Ollama desktop/menu-bar app
  — that's your *daily-driver* `:11434` instance; leave it running (our agent
  instance is the separate `:11435` process in the terminal). During a session
  you'll have both alive — different ports, no conflict, shared model store.
- **Which project?** Pi acts on Bartleby's active project. Set it on the host
  first with `bartleby project use <name>` — it's stored in
  `~/.bartleby/config.yaml`, which is shared into the VM via the `/corpus` mount,
  so the agent inherits it (or pass `--project <name>` on a skill call to
  override per-call).
- **Give the VM real RAM.** Apple `container` defaults a guest to **1 GB**, which
  is far too small here — embeddings (CPU-only in the guest) plus a multi-GB
  corpus DB will thrash it until it wedges at 100% CPU and the runtime auto-removes
  it (`--rm`). `run.sh` sets `-m 8g` by default; raise it for heavy ingest with
  `VM_MEMORY=16g ./scripts/pi-vm/run.sh`. You have 128 GB — don't starve it.

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

- **`pi` on `PATH`.** The image installs Pi via npm
  (`npm install -g --ignore-scripts @earendil-works/pi-coding-agent`) on a
  `node:22` base — the `pi.dev/install.sh` script needs a TTY and won't run in a
  non-interactive build. The global npm install puts `pi` in `/usr/local/bin`
  (on `PATH`). If `pi` isn't found at runtime, confirm the global npm bin dir is
  on `PATH` (`npm root -g` / `npm bin -g`).
- **In-build model pull.** The `Containerfile` runs `ollama serve` in the
  background during build to `ollama pull gemma4:e2b`. If that's flaky, move the
  pull to `entrypoint.sh` (first-run) backed by a persistent volume — at the
  cost of pure-build self-containment.
- **Gateway detection.** If `ip route` inside the VM doesn't yield a
  host-reachable address, set `HOST_OLLAMA_IP` explicitly (see networking note).
- **`container system start`.** Apple `container` needs its runtime started once
  per boot before `build`/`run`, and may need a moment to settle — if `build` or
  `run` errors with `XPC connection error: Connection invalid`, the runtime
  wasn't up yet; just re-run.

## Don't touch a pi-vm corpus from the host while it runs

**Operational rule: while a pi-vm (containerized) session is running against a
corpus, do not access that corpus from the host.** Stop `bartleby serve` for that
project, and don't start a host research session on it. View between runs, or once
the container is idle.

This is a **cross-kernel** limitation, **not** a concurrency one. Same-kernel
concurrent access *is* supported and tested — multiple host research sessions, or
host `serve` + a host session, are fine (`busy_timeout` + `BEGIN IMMEDIATE`
serialize writers; `tests/test_skill_concurrent_writes.py` lands six concurrent
writers with zero BusyError). The breakage is **strictly** the container (Linux
guest) and the host (macOS) touching the same corpus *at once*.

Why: Bartleby opens corpus DBs in **WAL mode**, whose write-ahead index lives in a
shared-memory sidecar (`bartleby.db-shm`) that is only coherent *within one
kernel*. With the agent holding the DB read-write over the `/corpus` mount **and**
a host process reading or writing it at the same time, two kernels share one WAL
DB — which SQLite explicitly does not support. This bug has two faces, and both
clear once the container is idle and its writes are checkpointed (`-wal` back to
`0` bytes):

- **Malformed (the loud face).** A host `bartleby serve` throws intermittent
  `500`s — `SqliteError: database disk image is malformed`. Your data is **not**
  corrupt; the host reader is seeing the guest's wal-index as garbage.
- **Stale reads (the quiet face).** A host `bartleby serve` that *doesn't* error
  will silently show **stale** data — new findings written by the containerized
  session won't appear until you restart `serve`. Trusting a viewer that looks
  fine but is hiding the run's output is the more dangerous half.

There is no clean reader-side fix — `better-sqlite3`, which the web UI uses,
doesn't support SQLite's `immutable=1` URI escape hatch.

## Disk & cleanup

The **container** is disposable — `run.sh` uses `--rm`, so `Ctrl-C` removes the
running VM and no stopped containers pile up. Two things *do* accumulate, and
`build.sh` now cleans up both for you:

1. **Orphaned images.** Each build re-tags `:latest` and orphans the previous
   digest's layers (~10–15 GB — the baked `gemma4:e2b` plus Node/Ollama/deps),
   which `container` has no reliable prune for. `build.sh` deletes the prior
   image *before* building, so exactly one stays on disk.
2. **The BuildKit cache.** The builder container accretes a layer cache on every
   build — it grew to **75 GB** during development. The nasty part: this cache
   lives *inside the builder container* and is **invisible to `container system
   df`**, so nothing warns you. `build.sh` resets it after a successful build
   (`container builder delete`; the builder recreates itself on the next build).
   Set **`KEEP_BUILD_CACHE=1`** to keep it for fast incremental rebuilds while
   you're iterating on the `Containerfile`.

So in normal use, repeated `build.sh` runs no longer stack up ghosts.

### If a build fails

When `container build` dies under `set -euo pipefail`, the script aborts between
the two cleanup steps: the prior `:latest` image is already deleted (step 1 runs
up front), the new image never finished, and the post-build cache reset never
runs (step 2 is skipped). A failed build therefore leaves you with **no working
image and an un-reset BuildKit cache** — and repeated failures let that cache
accumulate just like it did before `build.sh` was self-cleaning.

**Recovery:** once you've fixed the underlying cause (network/GitHub flake,
decant `main` broken, flaky `ollama pull`), just re-run `build.sh` — a
successful build resets the cache. If you're seeing persistent failures and
want to reclaim space in the meantime, use the manual commands below
(`container builder delete` for the cache, `container image ls` /
`container image delete` for images, and `du` to actually see the invisible
builder cache).

### To inspect or reclaim by hand

(Note: `container image` is **singular** — `container images` is not a valid verb.)

```bash
container system df                          # images/containers/volumes usage…
                                             # …but NOT the builder cache (see above)
container image ls                           # images on disk
container image delete bartleby-pi:latest    # delete by name — the reliable way
container builder delete                     # nuke the BuildKit cache (recreates on next build)
container ls -a                              # stopped containers (should be none; --rm)
```

Don't rely on `container image prune` to reclaim the big images — it's
[known-buggy](https://github.com/apple/container/issues/901) (removes only
orphaned blobs, not whole images). And remember `du` is the *only* way to see the
builder cache: `du -sh "$HOME/Library/Application Support/com.apple.container/containers"`.

## Files

| File | Role |
| --- | --- |
| [`scripts/pi-vm/host-agent-ollama.sh`](../scripts/pi-vm/host-agent-ollama.sh) | Host: dedicated agent Ollama on `:11435` (GPU) |
| [`scripts/pi-vm/build.sh`](../scripts/pi-vm/build.sh) | Build the image (pulls Bartleby release + decant main from GitHub) |
| [`scripts/pi-vm/Containerfile`](../scripts/pi-vm/Containerfile) | The research-VM image |
| [`scripts/pi-vm/entrypoint.sh`](../scripts/pi-vm/entrypoint.sh) | In-VM: start local Ollama, render configs, launch Pi |
| [`scripts/pi-vm/run.sh`](../scripts/pi-vm/run.sh) | Run the VM, mount the corpus |
| [`scripts/pi-vm/models.json`](../scripts/pi-vm/models.json) | Pi provider config (→ host agent Ollama) |
| [`scripts/pi-vm/decant-config.yaml`](../scripts/pi-vm/decant-config.yaml) | decant config (→ VM-local Ollama) |
