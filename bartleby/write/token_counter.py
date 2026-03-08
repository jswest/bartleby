"""Token usage tracking for LLM calls."""

from loguru import logger

from bartleby.lib.pricing import calculate_cost, get_model_pricing


class BudgetExceededError(Exception):
    """Raised when the token budget is exceeded."""


class TokenCounter:
    def __init__(self, model_name: str = "", token_budget: int | None = None):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0
        self.has_unknown_cost_model = False
        self.model_name = model_name
        self.token_budget = token_budget
        self._warned_at_80 = False

        # Check if model has known pricing
        if model_name:
            pricing = get_model_pricing(model_name)
            if pricing is None and not self._is_free_model():
                self.has_unknown_cost_model = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def on_step(self, step) -> None:
        """Update token counts from a smolagents step."""
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

        self._check_budget()

    def _check_budget(self) -> None:
        """Check token usage against budget, warn at 80%, raise at 100%."""
        if not self.token_budget:
            return

        usage_ratio = self.total_tokens / self.token_budget

        if usage_ratio >= 1.0:
            raise BudgetExceededError(
                f"Token budget exceeded: {self.total_tokens} / {self.token_budget} tokens"
            )

        if usage_ratio >= 0.8 and not self._warned_at_80:
            self._warned_at_80 = True
            logger.warning(
                f"Token budget 80% used: {self.total_tokens} / {self.token_budget} tokens"
            )

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
        self._warned_at_80 = False
