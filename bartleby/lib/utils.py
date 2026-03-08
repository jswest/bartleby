"""Utility functions."""

import json
from typing import Any


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 characters per token for English text)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    for end in (".  ", ".\n", ". "):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1]
    # No sentence boundary found — return first 120 chars
    return text[:120] + "..." if len(text) > 120 else text


def truncate_result(data: Any, max_tokens: int = 5000) -> Any:
    """Truncate result to fit within token budget.

    For lists of dicts (search results), progressively truncates body text
    of lower-ranked results rather than discarding everything. Higher-ranked
    results keep their full (already-preview-length) body; lower-ranked ones
    get reduced to first sentence + a body_truncated flag.
    """
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, default=str)

    token_count = _estimate_tokens(text)

    if token_count <= max_tokens:
        return data

    # For lists of dicts (search results), progressively truncate bodies
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Start truncating from the last (lowest-ranked) result upward
        truncated_data = [dict(item) for item in data]  # shallow copy each dict
        for i in range(len(truncated_data) - 1, -1, -1):
            body = truncated_data[i].get("body", "")
            if body:
                truncated_data[i]["body"] = _first_sentence(body)
                truncated_data[i]["body_truncated"] = True
            # Check if we're under budget now
            current = json.dumps(truncated_data, default=str)
            if _estimate_tokens(current) <= max_tokens:
                return truncated_data

        # Still too large — drop bodies entirely but keep metadata
        for item in truncated_data:
            item.pop("body", None)
            item["body_truncated"] = True
        return truncated_data

    # Non-list data — fall back to simple truncation message
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
