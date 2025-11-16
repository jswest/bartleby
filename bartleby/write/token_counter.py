"""Token usage tracking for LLM calls."""

from typing import Any
from langchain_core.callbacks import BaseCallbackHandler

from bartleby.lib.pricing import calculate_cost, get_model_pricing


class TokenCounterCallback(BaseCallbackHandler):
    def __init__(self, model_name: str = ""):
        """
        Initialize token counters.

        Args:
            model_name: Model name from config (used for cost calculation)
        """
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.has_unknown_cost_model = False
        self.model_name = model_name
        self.max_recursions = 0
        self.recursions_used = 0
        self.token_budget: int | None = None

        # For fallback token estimation (Ollama)
        self._accumulated_prompts = []
        self._accumulated_output = ""

        # Check if model has known pricing
        if model_name:
            pricing = get_model_pricing(model_name)
            if pricing is None and not self._is_free_model(model_name):
                self.has_unknown_cost_model = True

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Capture prompts for fallback token estimation (Ollama)."""
        if self._is_ollama_model():
            self._accumulated_prompts = prompts

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Accumulate streaming tokens for fallback estimation (Ollama)."""
        if self._is_ollama_model():
            self._accumulated_output += token

    def _estimate_tokens_from_chars(self, text: str) -> int:
        """
        Estimate token count from character count.
        Rule of thumb: ~1 token per 5 characters for English text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return max(1, len(text) // 5)

    def on_llm_end(self, response, **kwargs: Any) -> None:
        # Try to extract token usage from response (OpenAI format)
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

            if token_usage:
                prompt = token_usage.get("prompt_tokens", 0)
                completion = token_usage.get("completion_tokens", 0)

                self.prompt_tokens += prompt
                self.completion_tokens += completion
                self.total_tokens += token_usage.get("total_tokens", 0)

                # Calculate cost using model from config
                if self.model_name:
                    cost = calculate_cost(self.model_name, prompt, completion)
                    self.total_cost += cost

        # Also check for usage_metadata (used by Anthropic, Ollama)
        found_tokens = False
        if hasattr(response, "generations") and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "message") and hasattr(generation.message, "usage_metadata"):
                        usage = generation.message.usage_metadata
                        prompt = getattr(usage, "input_tokens", 0)
                        completion = getattr(usage, "output_tokens", 0)

                        if prompt > 0 or completion > 0:
                            found_tokens = True
                            self.prompt_tokens += prompt
                            self.completion_tokens += completion
                            self.total_tokens += getattr(usage, "total_tokens", 0)

                            # Calculate cost using model from config
                            if self.model_name:
                                cost = calculate_cost(self.model_name, prompt, completion)
                                self.total_cost += cost

        # Fallback: Estimate tokens from characters for Ollama if no tokens found
        if not found_tokens and self._is_ollama_model():
            # Estimate prompt tokens from accumulated prompts
            if self._accumulated_prompts:
                total_prompt_text = "".join(self._accumulated_prompts)
                estimated_prompt = self._estimate_tokens_from_chars(total_prompt_text)
                self.prompt_tokens += estimated_prompt

            # Estimate completion tokens from accumulated output
            if self._accumulated_output:
                estimated_completion = self._estimate_tokens_from_chars(self._accumulated_output)
                self.completion_tokens += estimated_completion

            # Update total
            self.total_tokens = self.prompt_tokens + self.completion_tokens

            # Reset accumulators
            self._accumulated_prompts = []
            self._accumulated_output = ""

    def _is_free_model(self, model_name: str) -> bool:
        """Check if a model is free (local/Ollama)."""
        return self._is_ollama_model()

    def _is_ollama_model(self) -> bool:
        """Check if using an Ollama model."""
        return self.model_name.startswith("ollama/") or ":" in self.model_name

    def get_stats(self) -> str:
        def format_tokens(count: int) -> str:
            if count < 1000:
                return str(count)
            rounded = round(count / 100) * 100
            return f"{rounded / 1000:.1f}k"

        # Format cost display
        if self.has_unknown_cost_model:
            # Show known cost + indicator for unknown models
            cost_str = f" (~${self.total_cost:.2f}, unknown cost)" if self.total_cost > 0 else " (unknown cost)"
        elif self.total_cost > 0:
            # Show cost rounded to nearest cent with tilde
            cost_str = f" (~${self.total_cost:.2f})"
        else:
            # No cost (free models or no usage)
            cost_str = ""

        return f"↑{format_tokens(self.prompt_tokens)}/↓{format_tokens(self.completion_tokens)}/+{format_tokens(self.total_tokens)}{cost_str}"

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.has_unknown_cost_model = False
        self._accumulated_prompts = []
        self._accumulated_output = ""
        self.recursions_used = 0
        self.max_recursions = 0
        self.token_budget = None
