"""Token usage tracking for LLM calls."""

from bartleby.lib.pricing import calculate_cost, get_model_pricing


class TokenCounter:
    def __init__(self, model_name: str = ""):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.has_unknown_cost_model = False
        self.model_name = model_name

        # Check if model has known pricing
        if model_name:
            pricing = get_model_pricing(model_name)
            if pricing is None and not self._is_free_model():
                self.has_unknown_cost_model = True

    def on_step(self, step) -> None:
        """
        Update token counts from a smolagents step.

        Args:
            step: A smolagents ActionStep or similar step object
        """
        token_usage = getattr(step, "token_usage", None)
        if token_usage is None:
            return
        input_tokens = getattr(token_usage, "input_tokens", 0) or 0
        output_tokens = getattr(token_usage, "output_tokens", 0) or 0

        if input_tokens > 0 or output_tokens > 0:
            self.prompt_tokens += input_tokens
            self.completion_tokens += output_tokens

            if self.model_name:
                cost = calculate_cost(self.model_name, input_tokens, output_tokens)
                self.total_cost += cost

    def _is_free_model(self) -> bool:
        return self.model_name.startswith("ollama/") or ":" in self.model_name

    def get_stats(self) -> str:
        def format_tokens(count: int) -> str:
            if count < 1000:
                return str(count)
            rounded = round(count / 100) * 100
            return f"{rounded / 1000:.1f}k"

        tokens_str = f"Tokens: {format_tokens(self.prompt_tokens)} in / {format_tokens(self.completion_tokens)} out"

        if self.has_unknown_cost_model:
            cost_str = f" | Cost: ~${self.total_cost:.2f} (unknown pricing)" if self.total_cost > 0 else " | Cost: unknown"
        elif self.total_cost > 0:
            cost_str = f" | Cost: ~${self.total_cost:.2f}"
        else:
            cost_str = ""

        return f"{tokens_str}{cost_str}"

    def reset(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
