# pi-vm --model flag: flag takes precedence over env var (issue #602)

> Source: [#602](https://github.com/jswest/bartleby/issues/602)

Both `run.sh` and `host-agent-ollama.sh` now accept `--model <id>`. When `--model` is given it sets `AGENT_MODEL` unconditionally, overwriting any pre-set env var for the rest of the script. The alternative — giving the env var precedence and treating `--model` as a lower-priority default — would make the flag useless in any environment where `AGENT_MODEL` is exported in a shell profile or wrapper script. The principle is "explicit beats ambient": a flag the user types at invocation time is the most explicit signal and wins. The env var remains fully supported when no flag is given. No new config, no new state — the flag collapses into the existing `AGENT_MODEL` variable before any downstream logic runs, so `entrypoint.sh` is unchanged.
