"""User configuration for Bartleby — ``~/.bartleby/config.yaml`` read/write."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


BARTLEBY_DIR = Path.home() / ".bartleby"
PROJECTS_DIR = BARTLEBY_DIR / "projects"
CONFIG_PATH = BARTLEBY_DIR / "config.yaml"


def scratch_dir() -> Path:
    """The skill's default scratch location for finding bodies (``--body-file``).

    Lives under ``~/.bartleby/`` alongside the rest of bartleby's state rather
    than world-readable ``/tmp`` — in-progress research notes shouldn't leak to
    other local users on shared machines.
    """
    return BARTLEBY_DIR / "tmp"


def ensure_scratch_dir() -> Path:
    """Create the scratch dir (mode 700) if missing; return its path.

    ``chmod`` is the load-bearing line: ``mkdir``'s mode argument is masked by
    the umask and is a no-op when the directory already exists, so we set the
    bits explicitly and idempotently on every call.
    """
    d = scratch_dir()
    d.mkdir(parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


def save_config(config: dict) -> None:
    BARTLEBY_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def save_config_field(key: str, value: Any) -> None:
    config = load_config()
    if value is None:
        config.pop(key, None)
    else:
        config[key] = value
    save_config(config)


def ensure_provider_env(provider: str | None, config: dict) -> None:
    """Populate environment variables expected by the provider SDKs.

    Sets ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` from
    config (without overwriting an existing env var) and ``OLLAMA_API_BASE``
    for Ollama. The wsjpt provider routes Gemini calls through ``GEMINI_API_KEY``
    when present; without it, wsjpt falls back to Vertex AI via ADC.
    """
    if not provider:
        return

    if provider in ("anthropic", "openai"):
        env_var = f"{provider.upper()}_API_KEY"
        config_key = f"{provider}_api_key"
        config_value = config.get(config_key)
        if config_value and not os.environ.get(env_var):
            os.environ[env_var] = config_value
    elif provider == "wsjpt":
        config_value = config.get("wsjpt_api_key")
        if config_value and not os.environ.get("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = config_value
    elif provider == "ollama":
        os.environ["OLLAMA_API_BASE"] = config.get(
            "ollama_base_url", "http://localhost:11434"
        )
