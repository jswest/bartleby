"""Utility functions."""

import json
from typing import Any, Optional

from loguru import logger

from bartleby.lib.config import setup_provider_env


def _estimate_tokens(text: str) -> int:
    """Estimate token count (~4 characters per token for English text)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_model_id(config: dict) -> Optional[str]:
    """
    Build a LiteLLM-compatible model_id string from config.

    Returns:
        Model ID string (e.g., "anthropic/claude-3-5-sonnet-20241022") or None
    """
    provider = config.get("provider")
    model = config.get("model")

    if not provider or not model:
        return None

    setup_provider_env(config)

    if provider == "anthropic":
        return f"anthropic/{model}"
    elif provider == "openai":
        return model
    elif provider == "ollama":
        return f"ollama_chat/{model}"
    else:
        return None


def load_model_from_config(config: dict):
    """
    Load a smolagents LiteLLMModel from config.

    Returns:
        Initialized LiteLLMModel or None if not configured
    """
    from smolagents import LiteLLMModel

    model_id = build_model_id(config)
    if not model_id:
        return None

    try:
        temperature = config.get("temperature", 0)
        temperature = max(0.0, min(1.0, float(temperature)))

        return LiteLLMModel(model_id=model_id, temperature=temperature)
    except Exception as e:
        logger.error(f"Failed to load model from config: {e}")
        return None


def has_vision(config: dict) -> bool:
    """Check if the configured model supports vision."""
    provider = config.get("provider", "")
    model = config.get("model", "")

    if provider == "anthropic":
        return "claude-3" in model or "claude-4" in model
    elif provider == "openai":
        return "gpt-4" in model and "vision" in model
    return False


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
