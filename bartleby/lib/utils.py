"""Utility functions."""

import json
from typing import Any


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 characters per token for English text)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def truncate_result(data: Any, max_tokens: int = 5000) -> Any:
    """Truncate result if it exceeds max tokens."""
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, default=str)

    token_count = _estimate_tokens(text)

    if token_count <= max_tokens:
        return data

    if isinstance(data, list):
        return {
            "truncated": True,
            "original_count": len(data),
            "token_count": token_count,
            "max_tokens": max_tokens,
            "message": f"Result contained {len(data)} items ({token_count} tokens). "
                      f"This exceeds the {max_tokens} token limit. "
                      f"Try a more specific query with smaller limit or more specific criteria."
        }
    else:
        return {
            "truncated": True,
            "token_count": token_count,
            "max_tokens": max_tokens,
            "message": f"Result was {token_count} tokens, exceeding the {max_tokens} token limit. "
                      f"Try a more specific query."
        }


# Backward-compatible imports for callers that import from utils
# These delegate to ConfigManager
def build_model_id(config: dict):
    """Build a LiteLLM-compatible model_id string from config.

    DEPRECATED: Use get_config_manager().model_id instead.
    """
    from bartleby.lib.config import get_config_manager

    mgr = get_config_manager()
    # Ensure config is loaded first
    mgr._load_if_stale()
    # Update manager with passed config for backward compat
    if config:
        for key in ("provider", "model"):
            if key in config:
                mgr._config[key] = config[key]
    return mgr.model_id


def load_model_from_config(config: dict):
    """Load a smolagents LiteLLMModel from config.

    DEPRECATED: Use get_config_manager().load_model() instead.
    """
    from bartleby.lib.config import get_config_manager

    mgr = get_config_manager()
    # Ensure config is loaded first
    mgr._load_if_stale()
    # Update manager with passed config for backward compat
    if config:
        for key in ("provider", "model", "temperature"):
            if key in config:
                mgr._config[key] = config[key]
    return mgr.load_model()


def has_vision(config: dict) -> bool:
    """Check if the configured model supports vision.

    DEPRECATED: Use get_config_manager().has_vision() instead.
    """
    from bartleby.lib.config import get_config_manager

    mgr = get_config_manager()
    # Ensure config is loaded first
    mgr._load_if_stale()
    # Update manager with passed config for backward compat
    if config:
        for key in ("provider", "model"):
            if key in config:
                mgr._config[key] = config[key]
    return mgr.has_vision()
