"""Interactive configuration wizard for Bartleby."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from bartleby.config import CONFIG_PATH, load_config, save_config

ALLOWED_PROVIDERS = ["anthropic", "openai", "ollama"]
ALLOWED_SUMMARY_DEPTHS = ["none", "one-shot"]
ALLOWED_PDF_CONVERTERS = ["pdfplumber", "docling"]
ALLOWED_HTML_CONVERTERS = ["docling", "sec2md"]

PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
    "ollama": "qwen3-vl:30b",
}

VISION_PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
    "ollama": "qwen3-vl:30b",
}

DEFAULT_SUMMARY_DEPTH = "one-shot"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_SUMMARIZE_TOKENS = 50_000
DEFAULT_MAX_READ_TOKENS = 50_000
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

DEFAULT_PDF_CONVERTER = "pdfplumber"
DEFAULT_HTML_CONVERTER = "docling"
DEFAULT_VISION_MAX_DIMENSION = 1024
DEFAULT_SPARSE_TEXT_THRESHOLD = 100
DEFAULT_OCR_MIN_CONFIDENCE = 30

console = Console()


def _prompt_provider(existing: dict, *, label: str = "LLM provider",
                     key: str = "provider") -> str:
    default_provider = existing.get(key, "anthropic")
    if default_provider not in ALLOWED_PROVIDERS:
        default_provider = "anthropic"
    choices = " / ".join(ALLOWED_PROVIDERS)
    while True:
        provider = Prompt.ask(
            f"{label} ({choices})", default=default_provider
        ).lower()
        if provider in ALLOWED_PROVIDERS:
            return provider
        console.print(f"[red]Invalid provider. Choose from: {choices}[/red]")


def _prompt_model(provider: str, existing: dict,
                  *, key: str = "model",
                  defaults: dict[str, str] = PROVIDER_DEFAULT_MODEL) -> str:
    default_model = existing.get(key) or defaults[provider]
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


def _prompt_pdf_converter(existing: dict) -> str:
    default = existing.get("pdf_converter", DEFAULT_PDF_CONVERTER)
    if default not in ALLOWED_PDF_CONVERTERS:
        default = DEFAULT_PDF_CONVERTER
    choices = " / ".join(ALLOWED_PDF_CONVERTERS)
    while True:
        b = Prompt.ask(
            f"PDF converter ({choices})", default=default
        ).lower()
        if b in ALLOWED_PDF_CONVERTERS:
            return b
        console.print(f"[red]Invalid converter. Choose from: {choices}[/red]")


def _prompt_html_converter(existing: dict) -> str:
    default = existing.get("html_converter", DEFAULT_HTML_CONVERTER)
    if default not in ALLOWED_HTML_CONVERTERS:
        default = DEFAULT_HTML_CONVERTER
    choices = " / ".join(ALLOWED_HTML_CONVERTERS)
    while True:
        b = Prompt.ask(
            f"HTML converter ({choices}; sec2md routes iXBRL filings, "
            f"others stay on docling)",
            default=default,
        ).lower()
        if b in ALLOWED_HTML_CONVERTERS:
            return b
        console.print(f"[red]Invalid converter. Choose from: {choices}[/red]")


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

    console.print("\n[bold]Converters[/bold]")
    config["pdf_converter"] = _prompt_pdf_converter(existing)
    config["html_converter"] = _prompt_html_converter(existing)
    config["sparse_text_threshold"] = _prompt_positive_int(
        "Sparse-text threshold (chars/page below which a page is treated as scanned)",
        int(existing.get("sparse_text_threshold", DEFAULT_SPARSE_TEXT_THRESHOLD)),
    )

    console.print("\n[bold]Image analysis[/bold] (VLM captions/OCR for embedded + standalone images)")
    use_vision = Confirm.ask(
        "Configure a vision provider for image analysis?",
        default=bool(existing.get("vision_provider")),
    )
    if use_vision:
        vprovider = _prompt_provider(
            existing, label="Vision provider", key="vision_provider",
        )
        config["vision_provider"] = vprovider
        config["vision_model"] = _prompt_model(
            vprovider, existing,
            key="vision_model", defaults=VISION_PROVIDER_DEFAULT_MODEL,
        )
        # If the vision provider is the same as the LLM provider, reuse the
        # api key already prompted for above. Otherwise prompt fresh.
        if vprovider in ("anthropic", "openai"):
            if vprovider != config.get("provider"):
                key = _prompt_api_key(vprovider, existing)
                field = f"{vprovider}_api_key"
                if key:
                    config[field] = key
        elif vprovider == "ollama":
            # Reuse ollama_base_url if already set; otherwise prompt.
            if "ollama_base_url" not in config:
                config["ollama_base_url"] = Prompt.ask(
                    "Ollama server URL",
                    default=existing.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
                )
        config["vision_max_dimension"] = _prompt_positive_int(
            "Max image dimension (long edge in pixels before sending to VLM)",
            int(existing.get("vision_max_dimension", DEFAULT_VISION_MAX_DIMENSION)),
        )
        config["ocr_min_confidence"] = _prompt_positive_int(
            "Tesseract minimum avg confidence (0-100; fallback to VLM below this)",
            int(existing.get("ocr_min_confidence", DEFAULT_OCR_MIN_CONFIDENCE)),
        )
    else:
        for k in ("vision_provider", "vision_model",
                 "vision_max_dimension", "ocr_min_confidence"):
            config.pop(k, None)

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
