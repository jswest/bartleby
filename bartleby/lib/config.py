"""Centralized configuration management for Bartleby."""

import os
from pathlib import Path

from loguru import logger
import yaml


BARTLEBY_DIR = Path.home() / ".bartleby"
PROJECTS_DIR = BARTLEBY_DIR / "projects"
CONFIG_PATH = BARTLEBY_DIR / "config.yaml"


def load_config() -> dict:
    """Load config from ~/.bartleby/config.yaml."""
    if not CONFIG_PATH.exists():
        return {}

    try:
        with CONFIG_PATH.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return {}


def save_config(config: dict):
    """Save config to ~/.bartleby/config.yaml, preserving project fields."""
    BARTLEBY_DIR.mkdir(parents=True, exist_ok=True)

    existing = load_config()
    if "active_project" in existing and "active_project" not in config:
        config["active_project"] = existing["active_project"]

    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def save_config_field(key: str, value):
    """Read-modify-write a single config field without clobbering others."""
    config = load_config()
    if value is None:
        config.pop(key, None)
    else:
        config[key] = value
    BARTLEBY_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def setup_provider_env(config: dict):
    """Set API keys and provider-specific env vars from config."""
    provider = config.get("provider")
    if not provider:
        return

    api_key_field = f"{provider}_api_key"
    config_api_key = config.get(api_key_field)
    env_var_name = f"{provider.upper()}_API_KEY"

    if config_api_key and not os.environ.get(env_var_name):
        os.environ[env_var_name] = config_api_key

    if provider == "ollama":
        base_url = config.get("ollama_base_url", "http://localhost:11434")
        os.environ["OLLAMA_API_BASE"] = base_url
