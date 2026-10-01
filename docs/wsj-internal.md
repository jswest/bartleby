# WSJ-internal: the wsjpt provider

The `wsjpt` provider routes Gemini through WSJ's internal [parsing toolkit](https://github.dowjones.net/data/wsjpt), so model aliases (`fast` / `smart` / `smartest`) resolve centrally — no concrete model names live in Bartleby's config. It's WSJ-internal only: its git source is unreachable outside WSJ, and it authenticates to Gemini via **Vertex AI / Application Default Credentials, not an API key** — run `gcloud auth application-default login` (or equivalent) rather than setting `GEMINI_API_KEY`.

## Install

Add one `--with` flag to whichever install/update command you're using from the main [README](../README.md#install-and-update):

```
uv tool install '.[docling,sec2md]' --with 'git+ssh://git@github.dowjones.net/data/wsjpt.git' --force
```

Repeat the `--with` flag on every reinstall or update — it isn't persisted anywhere else. wsjpt pins its own dependencies (including `pydantic-ai`), so no extra `--with` entries are needed.

## Why `--with`, not the locked deps

wsjpt's git source is unreachable outside WSJ, so adding it to the locked dependency set would break `uv lock`/`uv sync` for everyone else. `--with` injects it into the **tool's** isolated environment instead; a separate `uv pip install` lands somewhere the running tool can't see. `--force` re-applies over an existing install.

## Sidestepping SSH

Swap the source for HTTPS and git authenticates through the normal credential helper (a PAT, usually already cached in the macOS keychain from prior HTTPS clones):

```
--with 'git+https://github.dowjones.net/data/wsjpt.git'
```

## SSH hangs at "resolving dependencies..."

A passphrase-protected SSH key is the likely cause — `uv` runs git non-interactively, so the key can't prompt for its passphrase and the fetch silently blocks rather than erroring. Fix by loading the key into `ssh-agent`:

```
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

On macOS, persist it across reboots with `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` and this `~/.ssh/config` stanza:

```
Host github.dowjones.net
  AddKeysToAgent yes
  UseKeychain yes
```

## Local checkout

`--with '/abs/path/to/wsjpt'` avoids the network fetch entirely.

## Verify

Once wsjpt is configured, `bartleby config` loads the provider with no `ModuleNotFoundError: No module named 'wsjpt'`.
