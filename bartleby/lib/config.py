"""Centralized configuration management for Bartleby."""

import os
from pathlib import Path
from typing import Any, Optional

from loguru import logger
import yaml


BARTLEBY_DIR = Path.home() / ".bartleby"
PROJECTS_DIR = BARTLEBY_DIR / "projects"
CONFIG_PATH = BARTLEBY_DIR / "config.yaml"


class ConfigManager:
    """Singleton configuration manager with caching and auto-reload."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[dict] = None
    _config_mtime: Optional[float] = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_if_stale(self) -> dict:
        """Reload config if file changed since last load."""
        if not CONFIG_PATH.exists():
            self._config = {}
            self._config_mtime = None
            return self._config

        try:
            mtime = CONFIG_PATH.stat().st_mtime
            if self._config is None or mtime != self._config_mtime:
                with CONFIG_PATH.open("r") as f:
                    self._config = yaml.safe_load(f) or {}
                self._config_mtime = mtime
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            if self._config is None:
                self._config = {}

        return self._config

    @property
    def config(self) -> dict:
        """Get current config (cached, auto-reloads if file changed)."""
        return self._load_if_stale()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value and save to disk."""
        self._load_if_stale()
        if value is None:
            self._config.pop(key, None)
        else:
            self._config[key] = value
        self._save()

    def _save(self) -> None:
        """Write config to disk."""
        BARTLEBY_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, sort_keys=False)
        if CONFIG_PATH.exists():
            self._config_mtime = CONFIG_PATH.stat().st_mtime

    def invalidate_cache(self) -> None:
        """Force reload on next access."""
        self._config = None
        self._config_mtime = None

    def setup_provider_env(self) -> None:
        """Set API keys and provider-specific env vars from config."""
        provider = self.get("provider")
        if not provider:
            return

        api_key_field = f"{provider}_api_key"
        config_api_key = self.get(api_key_field)
        env_var_name = f"{provider.upper()}_API_KEY"

        if config_api_key and not os.environ.get(env_var_name):
            os.environ[env_var_name] = config_api_key

        if provider == "ollama":
            base_url = self.get("ollama_base_url", "http://localhost:11434")
            os.environ["OLLAMA_API_BASE"] = base_url

    @property
    def model_id(self) -> Optional[str]:
        """Build LiteLLM-compatible model_id from config."""
        provider = self.get("provider")
        model = self.get("model")

        if not provider or not model:
            return None

        self.setup_provider_env()

        if provider == "anthropic":
            return f"anthropic/{model}"
        elif provider == "openai":
            return model
        elif provider == "ollama":
            return f"ollama_chat/{model}"
        return None

    def has_vision(self) -> bool:
        """Check if configured model supports vision."""
        provider = self.get("provider", "")
        model = self.get("model", "")

        if provider == "anthropic":
            return "claude-3" in model or "claude-4" in model
        elif provider == "openai":
            return "gpt-4" in model and "vision" in model
        return False

    def load_model(self):
        """Load a smolagents LiteLLMModel from config.

        Returns:
            Initialized LiteLLMModel or None if not configured
        """
        from smolagents import LiteLLMModel

        model_id = self.model_id
        if not model_id:
            return None

        try:
            temperature = self.get("temperature", 0)
            temperature = max(0.0, min(1.0, float(temperature)))
            return LiteLLMModel(model_id=model_id, temperature=temperature)
        except Exception as e:
            logger.error(f"Failed to load model from config: {e}")
            return None


def get_config_manager() -> ConfigManager:
    """Get the singleton ConfigManager instance."""
    return ConfigManager()


# Backward-compatible wrapper functions


def load_config() -> dict:
    """Load config from ~/.bartleby/config.yaml.

    Returns a copy to maintain backward compatibility (original was not cached).
    """
    return get_config_manager().config.copy()


def save_config(config: dict) -> None:
    """Save config to ~/.bartleby/config.yaml, preserving project fields."""
    mgr = get_config_manager()
    existing = mgr.config

    # Preserve active_project if not in new config
    if "active_project" in existing and "active_project" not in config:
        config["active_project"] = existing["active_project"]

    mgr._config = config
    mgr._save()


def save_config_field(key: str, value: Any) -> None:
    """Read-modify-write a single config field without clobbering others."""
    get_config_manager().set(key, value)


def setup_provider_env(config: dict) -> None:
    """Set API keys and provider-specific env vars from config.

    This wrapper accepts a config dict for backward compatibility,
    but delegates to ConfigManager.setup_provider_env() which uses cached config.
    """
    # For backward compatibility, we update the cached config with any passed values
    # then call the manager's setup method
    mgr = get_config_manager()
    # Ensure config is loaded first
    mgr._load_if_stale()
    if config:
        for key in ("provider", "anthropic_api_key", "openai_api_key", "ollama_base_url"):
            if key in config:
                mgr._config[key] = config[key]
    mgr.setup_provider_env()
