# Pi-in-a-VM helper scripts

Run the Bartleby research skill with [Pi](https://pi.dev) and local Ollama
models inside an isolated Apple `container` VM. Full rationale, topology, and
the security trade-offs are in
[`docs/pi-vm-runbook.md`](../../docs/pi-vm-runbook.md) — **read that first.**

> First-draft scripts, not yet shaken out end-to-end. See the shakeout notes in
> the runbook.

## Quick start

```bash
brew install container && container system start   # one-time
./host-agent-ollama.sh                             # host GPU: qwen3.6:35b on :11435
./build.sh                                         # pure-build the image
BRAVE_SEARCH_API_KEY=brv-... ./run.sh              # mount ~/.bartleby, launch Pi
```

To use a different agent model, pass `--model` to both scripts (they must match):

```bash
./host-agent-ollama.sh --model gpt-oss:120b
./run.sh --model gpt-oss:120b
```

## What each file does

| File | Side | Role |
| --- | --- | --- |
| `host-agent-ollama.sh` | host | Dedicated agent Ollama on `:11435` (GPU), separate from your daily `:11434` |
| `build.sh` | host | Build `bartleby-pi:latest` (pulls Bartleby release + decant main from GitHub) |
| `Containerfile` | — | The research-VM image (Pi + Bartleby + decant + VM-local Ollama) |
| `run.sh` | host | Run the VM, mount the corpus read-write |
| `entrypoint.sh` | VM | Start local Ollama, render configs, launch Pi |
| `models.json` | VM | Pi provider config → host agent Ollama (`__HOST_OLLAMA__` templated) |
| `decant-config.yaml` | VM | decant config → VM-local Ollama (`__BRAVE_KEY__` templated) |

The split: the **agent** model (`qwen3.6:35b`) runs on the host GPU; **decant's**
distill model (`gemma4:e2b`) runs CPU-only inside the VM so web content stays in
the box. The only VM→host network path is Pi reaching the agent model.
