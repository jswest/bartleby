"""Model pricing data for cost calculation.

Pricing is per 1 million tokens (USD).
Last updated: 2025-01-15
"""

# Pricing per 1M tokens (USD)
# Format: {"model_name": {"input": price, "output": price}}
MODEL_PRICING = {
    # Anthropic Claude models
    # Source: https://docs.anthropic.com/en/docs/about-claude/pricing
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-opus-4": {"input": 15.00, "output": 75.00},
    "claude-opus-4-1": {"input": 15.00, "output": 75.00},

    # OpenAI models
    # Source: https://openai.com/api/pricing/
    "gpt-5-1": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},

    # Ollama models (local, free)
    # All Ollama models have zero cost
}


def get_model_pricing(model_name: str) -> dict[str, float] | None:
    """
    Get pricing for a model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with "input" and "output" prices per 1M tokens, or None if not found
    """
    # Check for exact match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Check if it's an Ollama model (local, free)
    if model_name.startswith("ollama/") or ":" in model_name:
        return {"input": 0.0, "output": 0.0}

    # Unknown model
    return None


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate cost for given model and token counts.

    Args:
        model_name: Name of the model
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        Total cost in USD (0.0 if model pricing is unknown)
    """
    pricing = get_model_pricing(model_name)
    if not pricing:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
