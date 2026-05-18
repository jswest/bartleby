"""Interactive configuration wizard for Bartleby."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from bartleby.config import CONFIG_PATH, load_config, save_config

ALLOWED_PROVIDERS = ["anthropic", "openai", "ollama"]
ALLOWED_SUMMARY_DEPTHS = ["none", "one-shot"]

PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
    "ollama": "gpt-oss:20b",
}

DEFAULT_SUMMARY_DEPTH = "one-shot"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_SUMMARIZE_TOKENS = 50_000
DEFAULT_MAX_READ_TOKENS = 50_000
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

console = Console()


def _prompt_provider(existing: dict) -> str:
    default_provider = existing.get("provider", "anthropic")
    choices = " / ".join(ALLOWED_PROVIDERS)
    while True:
        provider = Prompt.ask(
            f"LLM provider ({choices})", default=default_provider
        ).lower()
        if provider in ALLOWED_PROVIDERS:
            return provider
        console.print(f"[red]Invalid provider. Choose from: {choices}[/red]")


def _prompt_model(provider: str, existing: dict) -> str:
    default_model = existing.get("model") or PROVIDER_DEFAULT_MODEL[provider]
    return Prompt.ask("Model name", default=default_model)


def _prompt_api_key(provider: str, existing: dict) -> str | None:
    field = f"{provider}_api_key"
    existing_key = existing.get(field, "")
    if existing_key:
        if Confirm.ask("API key already configured. Update it?", default=False):
            entered = Prompt.ask("API key", password=True)
            return entered or existing_key
        return existing_key
    entered = Prompt.ask(
        "API key (optional, can also use env var)", password=True, default=""
    )
    return entered or None


def _prompt_summary_depth(existing: dict) -> str:
    default = existing.get("summary_depth", DEFAULT_SUMMARY_DEPTH)
    choices = " / ".join(ALLOWED_SUMMARY_DEPTHS)
    while True:
        depth = Prompt.ask(
            f"Summary depth ({choices})", default=default
        ).lower()
        if depth in ALLOWED_SUMMARY_DEPTHS:
            return depth
        console.print(f"[red]Invalid summary depth. Choose from: {choices}[/red]")


def _prompt_temperature(existing: dict) -> float:
    default = float(existing.get("temperature", DEFAULT_TEMPERATURE))
    while True:
        t = FloatPrompt.ask(
            "Temperature (0=deterministic, 1=creative)", default=default
        )
        if 0 <= t <= 1:
            return t
        console.print("[red]Temperature must be between 0 and 1[/red]")


def _prompt_positive_int(prompt: str, default: int) -> int:
    while True:
        n = IntPrompt.ask(prompt, default=default)
        if n > 0:
            return n
        console.print("[red]Must be a positive integer[/red]")


def main():
    console.print(Panel.fit(
        "[bold cyan]Bartleby Configuration[/bold cyan]\n"
        "Let's set up your preferences and API keys.",
        border_style="cyan",
    ))

    existing = load_config()
    # Rebuild the config from prompts; only preserve project-pointer state.
    config: dict = {}
    if existing.get("active_project"):
        config["active_project"] = existing["active_project"]

    console.print("\n[bold]LLM Configuration[/bold] (for ingest-time summarization)")
    use_llm = Confirm.ask(
        "Do you want to configure an LLM provider?",
        default=bool(existing.get("provider")),
    )

    if use_llm:
        provider = _prompt_provider(existing)
        config["provider"] = provider
        config["model"] = _prompt_model(provider, existing)

        if provider in ("anthropic", "openai"):
            key = _prompt_api_key(provider, existing)
            field = f"{provider}_api_key"
            if key:
                config[field] = key
            else:
                config.pop(field, None)
        elif provider == "ollama":
            config["ollama_base_url"] = Prompt.ask(
                "Ollama server URL",
                default=existing.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
            )

        console.print("\n[bold]Summarization[/bold]")
        depth = _prompt_summary_depth(existing)
        config["summary_depth"] = depth
        if depth == "one-shot":
            config["temperature"] = _prompt_temperature(existing)
            config["max_summarize_tokens"] = _prompt_positive_int(
                "Max input tokens for summarization (longer documents are truncated)",
                int(existing.get("max_summarize_tokens", DEFAULT_MAX_SUMMARIZE_TOKENS)),
            )
        else:
            config.pop("temperature", None)
            config.pop("max_summarize_tokens", None)
    else:
        # No LLM → no summarization.
        for k in ("provider", "model", "summary_depth", "temperature",
                 "max_summarize_tokens", "ollama_base_url",
                 "anthropic_api_key", "openai_api_key"):
            config.pop(k, None)
        config["summary_depth"] = "none"

    console.print("\n[bold]Document reading[/bold]")
    config["max_read_tokens"] = _prompt_positive_int(
        "Max tokens before `read_document` requires --force",
        int(existing.get("max_read_tokens", DEFAULT_MAX_READ_TOKENS)),
    )

    save_config(config)

    console.print("\n[bold green]Configuration saved.[/bold green]")
    console.print(f"Config location: [cyan]{CONFIG_PATH}[/cyan]")

    if not config.get("active_project"):
        console.print(
            "\n[yellow]Tip:[/yellow] Create a project with "
            "[bold]bartleby project create <name>[/bold]"
        )
