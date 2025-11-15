"""Utility functions."""

import json
import os
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models import BaseLanguageModel
from loguru import logger
import yaml


CONFIG_PATH = Path.home() / ".bartleby" / "config.yaml"


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses a simple heuristic: ~4 characters per token for English text.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def load_config() -> dict:
    """Load config from ~/.bartleby/config.yaml"""
    if not CONFIG_PATH.exists():
        return {}

    try:
        with CONFIG_PATH.open("r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return {}


def load_llm_from_config(config: dict) -> Optional[BaseLanguageModel]:
    """
    Load LLM from ~/.bartleby/config.yaml configuration.

    Returns:
        Initialized LLM or None if not configured
    """
    config = load_config()
    if not config:
        return None

    try:
        provider = config.get("provider")
        model = config.get("model")

        if not provider or not model:
            return None

        # Get temperature (default to 0)
        temperature = config.get("temperature", 0)
        # Clamp to valid range [0, 1]
        temperature = max(0.0, min(1.0, float(temperature)))

        # Set API key from config if available
        api_key_field = f"{provider}_api_key"
        config_api_key = config.get(api_key_field)
        env_var_name = f"{provider.upper()}_API_KEY"

        if config_api_key and not os.environ.get(env_var_name):
            os.environ[env_var_name] = config_api_key

        # Initialize LLM based on provider
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature)
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama
            # Ollama doesn't need API keys, runs locally
            base_url = config.get("ollama_base_url", "http://localhost:11434")
            return ChatOllama(model=model, base_url=base_url, temperature=temperature)
        else:
            return None

    except Exception as e:
        logger.error(f"Failed to load LLM from config: {e}")
        return None


def truncate_result(data: Any, max_tokens: int = 5000) -> Any:
    """
    Truncate result if it exceeds max tokens.

    Args:
        data: Data to potentially truncate
        max_tokens: Maximum tokens allowed

    Returns:
        Original data or truncated version with metadata
    """
    # Convert data to string for counting
    if isinstance(data, str):
        text = data
    else:
        text = json.dumps(data, default=str)

    token_count = estimate_tokens(text)

    if token_count <= max_tokens:
        return data

    # Result is too large - return summary instead
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
