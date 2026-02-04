from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from bartleby.lib.config import CONFIG_PATH, load_config, save_config

ALLOWED_PROVIDERS = ["anthropic", "openai", "ollama"]

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]Bartleby Configuration[/bold cyan]\n"
        "Let's set up your preferences and API keys.",
        border_style="cyan"
    ))

    existing_config = load_config()

    config = {}

    default_workers = existing_config.get("max_workers", 4)
    config["max_workers"] = IntPrompt.ask(
        "Maximum number of worker threads",
        default=default_workers
    )

    # Provider setup
    console.print("\n[bold]LLM Configuration[/bold] (optional, for generating summaries)")
    use_llm = Confirm.ask("Do you want to configure an LLM provider?", default=bool(existing_config.get("provider")))

    if use_llm:
        # Provider
        provider_choices = " / ".join(ALLOWED_PROVIDERS)
        default_provider = existing_config.get("provider", "anthropic")

        while True:
            provider = Prompt.ask(
                f"LLM provider ({provider_choices})",
                default=default_provider
            ).lower()

            if provider in ALLOWED_PROVIDERS:
                config["provider"] = provider
                break
            else:
                console.print(f"[red]Invalid provider. Choose from: {provider_choices}[/red]")

        # Model
        default_model = existing_config.get("model", "")
        if provider == "anthropic":
            default_model = default_model or "claude-3-5-sonnet-20241022"
        elif provider == "openai":
            default_model = default_model or "gpt-4-turbo"
        elif provider == "ollama":
            default_model = default_model or "llama3.2"

        config["model"] = Prompt.ask(
            "Model name",
            default=default_model
        )

        # API Key (not needed for Ollama)
        if provider in ["anthropic", "openai"]:
            console.print(f"\n[bold]API Key for {provider}[/bold]")
            api_key_field = f"{provider}_api_key"
            existing_key = existing_config.get(api_key_field, "")

            if existing_key:
                update_key = Confirm.ask(
                    f"API key already configured. Update it?",
                    default=False
                )
                if update_key:
                    api_key = Prompt.ask(f"API key", password=True)
                    if api_key:
                        config[api_key_field] = api_key
                else:
                    config[api_key_field] = existing_key
            else:
                api_key = Prompt.ask(f"API key (optional, can also use env var)", password=True, default="")
                if api_key:
                    config[api_key_field] = api_key
        elif provider == "ollama":
            # Ollama-specific config
            console.print(f"\n[bold]Ollama Configuration[/bold]")
            default_url = existing_config.get("ollama_base_url", "http://localhost:11434")
            ollama_url = Prompt.ask(
                "Ollama server URL",
                default=default_url
            )
            config["ollama_base_url"] = ollama_url

        # Summarization settings
        console.print(f"\n[bold]Summarization Settings[/bold]")
        default_pages_to_summarize = existing_config.get("pdf_pages_to_summarize", 10)
        config["pdf_pages_to_summarize"] = IntPrompt.ask(
            "Number of pages to summarize per PDF (0 = no summaries)",
            default=default_pages_to_summarize
        )

        console.print(f"\n[bold]Generation Settings[/bold]")

        default_temperature = existing_config.get("temperature", 0)
        while True:
            temperature = FloatPrompt.ask(
                "Temperature (0-1, 0=deterministic, 1=creative)",
                default=default_temperature
            )
            if 0 <= temperature <= 1:
                config["temperature"] = temperature
                break
            else:
                console.print("[red]Temperature must be between 0 and 1[/red]")

    # Save config
    save_config(config)

    console.print(f"\n[bold green]✓ Configuration saved![/bold green]")
    console.print(f"Config location: [cyan]{CONFIG_PATH}[/cyan]")

    if not config.get("active_project"):
        console.print("\n[yellow]Tip:[/yellow] Create a project with [bold]bartleby project create <name>[/bold]")
    else:
        console.print("\nYou can now run: [bold]bartleby read[/bold] or [bold]bartleby write[/bold]")
