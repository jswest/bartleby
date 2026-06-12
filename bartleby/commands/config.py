"""Interactive configuration wizard for Bartleby."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt

from bartleby.config import config_path, load_config, save_config
from bartleby.lib.consts import (
    ALLOWED_HTML_CONVERTERS,
    ALLOWED_PDF_CONVERTERS,
    ALLOWED_PROVIDERS,
    ALLOWED_REASONING_EFFORTS,
    DEFAULT_CAPTION_WORKERS,
    DEFAULT_HTML_CONVERTER,
    DEFAULT_MAX_SUMMARIZE_TOKENS,
    DEFAULT_OCR_MIN_CONFIDENCE,
    DEFAULT_PDF_CONVERTER,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SPARSE_TEXT_THRESHOLD,
    DEFAULT_SUMMARIZE_WORKERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_VISION_MAX_DIMENSION,
    DEFAULT_VISION_MIN_DIMENSION,
)

ALLOWED_SUMMARY_DEPTHS = ["none", "one-shot"]

PROVIDER_DEFAULT_MODEL = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-5-mini",
    "ollama": "qwen3-vl:30b",
    "wsjpt": "fast",
}

DEFAULT_SUMMARY_DEPTH = "one-shot"
DEFAULT_MAX_READ_TOKENS = 50_000
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

console = Console()


def _help(text: str) -> None:
    """Print a concise, dim help blurb above the upcoming prompt.

    The wizard's prompt labels stay short (Rich renders them on one line); the
    orienting detail lives here, indented and dimmed so it reads as secondary.
    Multi-line strings are printed line-for-line.
    """
    for line in text.split("\n"):
        console.print(f"  {line}", style="dim")


def _prompt_choice(label: str, choices: list[str], existing_value: str | None,
                   default: str, *, help_text: str, invalid_noun: str) -> str:
    """Prompt for a value from a fixed allowlist, looping until it's valid.

    The effective default is the existing config value when it's still in the
    allowlist, else the caller-supplied canonical default.
    """
    effective_default = existing_value if existing_value in choices else default
    rendered = " / ".join(choices)
    _help(help_text)
    while True:
        value = Prompt.ask(
            f"{label} ({rendered})", default=effective_default
        ).lower()
        if value in choices:
            return value
        console.print(
            f"[red]Invalid {invalid_noun}. Choose from: {rendered}[/red]"
        )


def _prompt_model(provider: str, existing: dict,
                  *, help_text: str, key: str = "model") -> str:
    default_model = existing.get(key) or PROVIDER_DEFAULT_MODEL[provider]
    _help(help_text)
    return Prompt.ask("Model name", default=default_model)


def _prompt_api_key(provider: str, existing: dict, *, help_text: str | None = None) -> str | None:
    field = f"{provider}_api_key"
    existing_key = existing.get(field, "")
    _help(help_text or (
        "Saved to the config file. Leave blank to fall back to the "
        f"{provider.upper()}_API_KEY environment variable."
    ))
    if existing_key:
        if Confirm.ask("API key already configured. Update it?", default=False):
            entered = Prompt.ask("API key", password=True)
            return entered or existing_key
        return existing_key
    entered = Prompt.ask(
        "API key (optional)", password=True, default=""
    )
    return entered or None


def _prompt_temperature(existing: dict, *, help_text: str) -> float:
    default = float(existing.get("temperature", DEFAULT_TEMPERATURE))
    _help(help_text)
    while True:
        t = FloatPrompt.ask("Temperature", default=default)
        if 0 <= t <= 1:
            return t
        console.print("[red]Temperature must be between 0 and 1[/red]")


def _prompt_ollama_url(existing: dict) -> str:
    _help("Where your local Ollama server listens.")
    return Prompt.ask(
        "Ollama server URL",
        default=existing.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
    )


def _prompt_positive_int(prompt: str, default: int, *, help_text: str) -> int:
    _help(help_text)
    while True:
        n = IntPrompt.ask(prompt, default=default)
        if n > 0:
            return n
        console.print("[red]Must be a positive integer[/red]")


def _prompt_bounded_int(
    prompt: str, default: int, *, min_value: int, max_value: int, help_text: str
) -> int:
    """Prompt for an int in [min_value, max_value], looping until it's in range.

    Unlike ``_prompt_positive_int`` (n > 0), this admits a meaningful 0 and caps
    the top end — e.g. ocr_min_confidence is a 0–100 percentage where 0 ("always
    trust OCR") is valid and 250 is not.
    """
    _help(help_text)
    while True:
        n = IntPrompt.ask(prompt, default=default)
        if min_value <= n <= max_value:
            return n
        console.print(f"[red]Must be between {min_value} and {max_value}[/red]")


def _api_key_help(provider: str) -> str:
    """The default api-key help, naming the env-var fallback for this provider.

    wsjpt authenticates against the Gemini API, so its key lands in
    GEMINI_API_KEY (via ensure_provider_env), not the WSJPT_API_KEY the
    generic ``{PROVIDER}_API_KEY`` rule would otherwise advertise.
    """
    env_var = "GEMINI_API_KEY" if provider == "wsjpt" else f"{provider.upper()}_API_KEY"
    return (
        "Saved to the config file. Leave blank to fall back to the "
        f"{env_var} environment variable."
    )


def _prompt_max_workers(existing: dict) -> int | None:
    """Returns the worker count, or None for auto (omit the key from config)."""
    _help(
        "How many documents scribe parses in parallel. 0 = auto: the min of your "
        "CPU cores (less a couple held back for the OS) and what free RAM allows."
        "\nRaise it for a faster bulk ingest on a big machine — note a value you "
        "set here can use every core; lower it if memory is tight."
    )
    current = existing.get("max_workers")
    while True:
        n = IntPrompt.ask("Parse workers (0 = auto)", default=int(current) if current else 0)
        if n >= 0:
            return n or None
        console.print("[red]Must be 0 (auto) or a positive integer[/red]")


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
    _help(
        "An LLM summarizes each document at ingest. Summaries are searchable "
        "and shown when browsing.\nSkip this to ingest and index raw text only."
    )
    use_llm = Confirm.ask(
        "Do you want to configure an LLM provider?",
        default=bool(existing.get("provider")),
    )

    if use_llm:
        provider = _prompt_choice(
            "LLM provider", ALLOWED_PROVIDERS,
            existing.get("provider"), "anthropic",
            invalid_noun="provider",
            help_text="The service that runs your summarization model.",
        )
        config["provider"] = provider
        config["model"] = _prompt_model(
            provider, existing,
            help_text="The summarization model. The default is a fast, "
            "low-cost choice for this provider.",
        )

        if provider in ("anthropic", "openai", "wsjpt"):
            key = _prompt_api_key(provider, existing, help_text=_api_key_help(provider))
            if key:
                config[f"{provider}_api_key"] = key
        elif provider == "ollama":
            config["ollama_base_url"] = _prompt_ollama_url(existing)

        console.print("\n[bold]Summarization[/bold]")
        depth = _prompt_choice(
            "Summary depth", ALLOWED_SUMMARY_DEPTHS,
            existing.get("summary_depth"), DEFAULT_SUMMARY_DEPTH,
            invalid_noun="summary depth",
            help_text="none = skip summaries; one-shot = one summary per "
            "document.\nSummaries are indexed for search/scan and shown in listings.",
        )
        config["summary_depth"] = depth
        if depth == "one-shot":
            config["temperature"] = _prompt_temperature(
                existing,
                help_text="0 = deterministic, repeatable summaries; higher = "
                "more varied wording.\nLeave at 0 unless summaries feel too rigid.",
            )
            config["reasoning_effort"] = _prompt_choice(
                "Reasoning effort", ALLOWED_REASONING_EFFORTS,
                existing.get("reasoning_effort"), DEFAULT_REASONING_EFFORT,
                invalid_noun="effort",
                help_text="How hard the model reasons before summarizing. Lower = "
                "fewer billed reasoning tokens and faster, which is plenty for "
                "summaries.\nApplies to OpenAI (gpt-5) and effort-capable Anthropic "
                "models; ignored by Ollama/wsjpt.",
            )
            config["max_summarize_tokens"] = _prompt_positive_int(
                "Max summarization input tokens",
                int(existing.get("max_summarize_tokens", DEFAULT_MAX_SUMMARIZE_TOKENS)),
                help_text="Caps how much document text is sent to the summarizer; "
                "longer documents are truncated.\nHigher = more context, higher cost.",
            )
            if provider != "ollama":
                # Ollama serializes (OLLAMA_NUM_PARALLEL=1), so summarize workers
                # auto-clamp to 1 (#243) — no count to prompt for.
                config["summarize_workers"] = _prompt_positive_int(
                    "Summarize workers",
                    int(existing.get("summarize_workers", DEFAULT_SUMMARIZE_WORKERS)),
                    help_text="How many documents summarize in parallel after "
                    "parsing — LLM calls are network-bound, so this runs separately "
                    "from parse workers.\nRaise it for a rate-tolerant cloud provider.",
                )
    else:
        # No LLM → no summarization.
        config["summary_depth"] = "none"

    console.print("\n[bold]Converters[/bold]")
    config["pdf_converter"] = _prompt_choice(
        "PDF converter", ALLOWED_PDF_CONVERTERS,
        existing.get("pdf_converter"), DEFAULT_PDF_CONVERTER,
        invalid_noun="converter",
        help_text="pdfplumber = fast text extraction; docling = slower but "
        "layout-aware (tables, columns, reading order).",
    )
    config["html_converter"] = _prompt_choice(
        "HTML converter", ALLOWED_HTML_CONVERTERS,
        existing.get("html_converter"), DEFAULT_HTML_CONVERTER,
        invalid_noun="converter",
        help_text="docling handles general HTML/Markdown; sec2md is specialized "
        "for iXBRL EDGAR filings (preserves SEC tables/headings, others stay on "
        "docling).",
    )
    config["sparse_text_threshold"] = _prompt_positive_int(
        "Sparse-text threshold",
        int(existing.get("sparse_text_threshold", DEFAULT_SPARSE_TEXT_THRESHOLD)),
        help_text="Pages with fewer than this many extracted characters are "
        "treated as scanned and routed to OCR/VLM.\nLower = more pages OCR'd "
        "(slower, catches more).",
    )
    max_workers = _prompt_max_workers(existing)
    if max_workers is not None:
        config["max_workers"] = max_workers

    console.print("\n[bold]Image analysis[/bold] (VLM captions/OCR for embedded + standalone images)")
    _help(
        "A vision model captions and OCRs images (embedded figures and "
        "standalone image files).\nSkip to leave images unanalyzed."
    )
    use_vision = Confirm.ask(
        "Configure a vision provider for image analysis?",
        default=bool(existing.get("vision_provider")),
    )
    if use_vision:
        vprovider = _prompt_choice(
            "Vision provider", ALLOWED_PROVIDERS,
            existing.get("vision_provider"), "anthropic",
            invalid_noun="provider",
            help_text="The service that runs your vision (image) model.",
        )
        config["vision_provider"] = vprovider
        config["vision_model"] = _prompt_model(
            vprovider, existing,
            key="vision_model",
            help_text="The vision model used to caption/OCR images. The default "
            "is a fast, low-cost choice for this provider.",
        )
        # If the vision provider is the same as the LLM provider, reuse the
        # api key already prompted for above. Otherwise prompt fresh.
        if vprovider in ("anthropic", "openai", "wsjpt"):
            if vprovider != config.get("provider"):
                key = _prompt_api_key(
                    vprovider, existing, help_text=_api_key_help(vprovider)
                )
                if key:
                    config[f"{vprovider}_api_key"] = key
        elif vprovider == "ollama":
            # Reuse ollama_base_url if already set; otherwise prompt.
            if "ollama_base_url" not in config:
                config["ollama_base_url"] = _prompt_ollama_url(existing)
        config["vision_max_dimension"] = _prompt_positive_int(
            "Max image dimension (px)",
            int(existing.get("vision_max_dimension", DEFAULT_VISION_MAX_DIMENSION)),
            help_text="Images are downscaled so their long edge is at most this "
            "many pixels before the VLM sees them.\nLower = faster/cheaper, less detail.",
        )
        config["vision_min_dimension"] = _prompt_positive_int(
            "Min image dimension (px)",
            int(existing.get("vision_min_dimension", DEFAULT_VISION_MIN_DIMENSION)),
            help_text="Images with an edge smaller than this are skipped — "
            "avoids wasting VLM calls (and crashes) on thin slivers.",
        )
        config["ocr_min_confidence"] = _prompt_bounded_int(
            "Tesseract min confidence",
            int(existing.get("ocr_min_confidence", DEFAULT_OCR_MIN_CONFIDENCE)),
            min_value=0, max_value=100,
            help_text="Tesseract average confidence (0-100); pages scoring below "
            "this fall back to the VLM.\nHigher = trust OCR less, use the VLM more "
            "(0 = always trust OCR).",
        )
        if vprovider != "ollama":
            # Ollama serializes (OLLAMA_NUM_PARALLEL=1), so caption workers
            # auto-clamp to 1 (#243) — no count to prompt for.
            config["caption_workers"] = _prompt_positive_int(
                "Caption workers",
                int(existing.get("caption_workers", DEFAULT_CAPTION_WORKERS)),
                help_text="How many images caption in parallel after parsing — VLM "
                "calls are network-bound, so this runs separately from parse "
                "workers.\nRaise it for a rate-tolerant cloud provider.",
            )

    console.print("\n[bold]Document reading[/bold]")
    config["max_read_tokens"] = _prompt_positive_int(
        "Max read tokens",
        int(existing.get("max_read_tokens", DEFAULT_MAX_READ_TOKENS)),
        help_text="An agent's `read_document` must pass --force to pull a "
        "document larger than this many tokens.\nGuards against blowing the "
        "context window on a huge file.",
    )

    save_config(config)

    console.print("\n[bold green]Configuration saved.[/bold green]")
    console.print(f"Config location: [cyan]{config_path()}[/cyan]")

    if not config.get("active_project"):
        console.print(
            "\n[yellow]Tip:[/yellow] Create a project with "
            "[bold]bartleby project create <name>[/bold]"
        )
