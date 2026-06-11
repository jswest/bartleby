"""Tests for the `bartleby config` interactive wizard."""

from __future__ import annotations

import yaml
import pytest

import bartleby.config
import bartleby.commands.config as config
from bartleby.config import config_drift, redact_config, save_config


@pytest.fixture
def isolated_config():
    # Namespace isolation is suite-wide via conftest's _isolate_bartleby_home.
    yield bartleby.config.config_path()


def _scripted_inputs(monkeypatch, answers: list[str]):
    """Feed each call to Rich's Prompt/Confirm/IntPrompt/FloatPrompt a value
    by patching ``rich.prompt.Prompt.ask`` to consume from a list."""
    queue = list(answers)

    def _next(*args, **kwargs):
        return queue.pop(0)

    monkeypatch.setattr("rich.prompt.Prompt.ask", _next)
    monkeypatch.setattr("rich.prompt.IntPrompt.ask", lambda *a, **k: int(_next()))
    monkeypatch.setattr("rich.prompt.FloatPrompt.ask", lambda *a, **k: float(_next()))
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **k: _next() in ("y", "Y", True, "true"))
    return queue


def _read_yaml(path):
    with path.open() as f:
        return yaml.safe_load(f) or {}


def test_config_writes_v1_keys_with_anthropic_one_shot(isolated_config, monkeypatch):
    _scripted_inputs(monkeypatch, [
        "y",                   # Confirm: configure LLM?
        "anthropic",           # Provider
        "claude-haiku-4-5",    # Model
        "sk-test-key",         # API key
        "one-shot",            # Summary depth
        "0",                   # Temperature
        "low",                 # Reasoning effort
        "50000",               # Max summarize tokens
        "4",                   # Summarize workers
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",                 # Sparse text threshold
        "0",                   # Parse workers (0 = auto → omitted)
        "n",                   # Configure vision?
        "50000",               # Max read tokens
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert cfg["provider"] == "anthropic"
    assert cfg["model"] == "claude-haiku-4-5"
    assert cfg["anthropic_api_key"] == "sk-test-key"
    assert cfg["summary_depth"] == "one-shot"
    assert cfg["temperature"] == 0
    assert cfg["reasoning_effort"] == "low"
    assert cfg["max_summarize_tokens"] == 50000
    assert cfg["max_read_tokens"] == 50000
    assert cfg["pdf_converter"] == "pdfplumber"
    assert cfg["html_converter"] == "docling"
    assert cfg["sparse_text_threshold"] == 100
    # Vision opted out.
    assert "vision_provider" not in cfg
    assert "vision_model" not in cfg
    # Legacy keys must not appear.
    assert "max_workers" not in cfg
    assert "pdf_pages_to_summarize" not in cfg
    assert "pages_to_summarize" not in cfg


def test_config_with_summary_depth_none_omits_summarize_settings(
    isolated_config, monkeypatch
):
    _scripted_inputs(monkeypatch, [
        "y",                   # Configure LLM?
        "anthropic",
        "claude-haiku-4-5",
        "sk-x",
        "none",                # Summary depth → none
        # No temperature/max_summarize_tokens prompted.
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",                 # Sparse text threshold
        "0",                   # Parse workers (0 = auto → omitted)
        "n",                   # Configure vision?
        "60000",               # Max read tokens
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert cfg["summary_depth"] == "none"
    assert "temperature" not in cfg
    assert "max_summarize_tokens" not in cfg
    assert cfg["max_read_tokens"] == 60000


def test_config_with_ollama_writes_base_url_not_api_key(isolated_config, monkeypatch):
    _scripted_inputs(monkeypatch, [
        "y",
        "ollama",
        "gpt-oss:20b",
        "http://localhost:11434",
        "one-shot",
        "0",
        "low",
        "50000",
        # No "Summarize workers" prompt — Ollama auto-clamps to 1 (#243).
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",                 # Sparse text threshold
        "0",                   # Parse workers (0 = auto → omitted)
        "n",                   # Configure vision?
        "50000",
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert cfg["provider"] == "ollama"
    assert cfg["model"] == "gpt-oss:20b"
    assert cfg["ollama_base_url"] == "http://localhost:11434"
    assert "anthropic_api_key" not in cfg
    assert "openai_api_key" not in cfg
    # Ollama serializes, so the wizard never stores a summarize-worker count.
    assert "summarize_workers" not in cfg


def test_config_without_llm_writes_summary_depth_none(isolated_config, monkeypatch):
    _scripted_inputs(monkeypatch, [
        "n",                   # No LLM
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",                 # Sparse text threshold
        "0",                   # Parse workers (0 = auto → omitted)
        "n",                   # Configure vision?
        "50000",               # Max read tokens
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert cfg.get("provider") is None
    assert cfg["summary_depth"] == "none"
    assert cfg["max_read_tokens"] == 50000
    assert cfg["pdf_converter"] == "pdfplumber"
    assert cfg["html_converter"] == "docling"


def test_config_strips_legacy_keys_from_existing_config(isolated_config, monkeypatch):
    # Pre-populate config with v0 keys.
    with isolated_config.open("w") as f:
        yaml.safe_dump({
            "max_workers": 8,
            "pdf_pages_to_summarize": 10,
            "active_project": "alpha",
        }, f)

    _scripted_inputs(monkeypatch, [
        "n",                   # No LLM
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",                 # Sparse text threshold
        "0",                   # Parse workers (0 = auto → omitted)
        "n",                   # Configure vision?
        "50000",               # Max read tokens
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert "max_workers" not in cfg
    assert "pdf_pages_to_summarize" not in cfg
    # Non-legacy keys (active_project) are preserved.
    assert cfg["active_project"] == "alpha"


def test_config_with_vision_writes_vision_keys(isolated_config, monkeypatch):
    _scripted_inputs(monkeypatch, [
        "y",                   # Configure LLM?
        "openai",
        "gpt-5-mini",
        "sk-openai",
        "one-shot",
        "0",
        "low",
        "50000",
        "4",                   # Summarize workers
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",
        "0",                   # Parse workers (0 = auto → omitted)
        "y",                   # Configure vision?
        "openai",              # Vision provider (same as LLM → no fresh api key)
        "gpt-5-mini",          # Vision model
        "1024",                # vision_max_dimension
        "32",                  # vision_min_dimension
        "30",                  # ocr_min_confidence
        "4",                   # caption_workers
        "50000",               # max_read_tokens
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert cfg["vision_provider"] == "openai"
    assert cfg["vision_model"] == "gpt-5-mini"
    assert cfg["vision_max_dimension"] == 1024
    assert cfg["vision_min_dimension"] == 32
    assert cfg["ocr_min_confidence"] == 30
    assert cfg["caption_workers"] == 4
    # openai_api_key already set from the LLM block; no double prompt.
    assert cfg["openai_api_key"] == "sk-openai"


def test_config_vision_with_different_provider_prompts_for_fresh_key(
    isolated_config, monkeypatch
):
    _scripted_inputs(monkeypatch, [
        "y",
        "openai",
        "gpt-5-mini",
        "sk-openai",
        "one-shot",
        "0",
        "low",
        "50000",
        "4",                   # Summarize workers
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",
        "0",                   # Parse workers (0 = auto → omitted)
        "y",                   # Configure vision?
        "anthropic",           # Different provider → prompt for key
        "claude-haiku-4-5",
        "sk-anthro",
        "1024",
        "32",
        "30",
        "4",                   # caption_workers
        "50000",
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert cfg["vision_provider"] == "anthropic"
    assert cfg["openai_api_key"] == "sk-openai"
    assert cfg["anthropic_api_key"] == "sk-anthro"


def test_config_ollama_vision_skips_caption_workers_prompt(
    isolated_config, monkeypatch
):
    # Cloud LLM (so summarize_workers is still prompted) + Ollama vision: the
    # caption-worker prompt is skipped because Ollama serializes (#243).
    _scripted_inputs(monkeypatch, [
        "y",
        "anthropic",
        "claude-haiku-4-5",
        "sk-anthro",
        "one-shot",
        "0",
        "low",
        "50000",
        "4",                   # Summarize workers (anthropic → still prompted)
        "pdfplumber",          # PDF converter
        "docling",             # HTML converter
        "100",
        "0",                   # Parse workers (0 = auto → omitted)
        "y",                   # Configure vision?
        "ollama",              # Vision provider
        "qwen3-vl:30b",        # Vision model
        "http://localhost:11434",  # Ollama base URL (not yet set by the LLM block)
        "1024",                # vision_max_dimension
        "32",                  # vision_min_dimension
        "30",                  # ocr_min_confidence
        # No "Caption workers" prompt — Ollama auto-clamps to 1.
        "50000",               # max_read_tokens
    ])
    config.main()
    cfg = _read_yaml(isolated_config)
    assert cfg["vision_provider"] == "ollama"
    assert cfg["ollama_base_url"] == "http://localhost:11434"
    # Cloud LLM still carries a summarize count; Ollama vision carries none.
    assert cfg["summarize_workers"] == 4
    assert "caption_workers" not in cfg


# -------------------- ingest provenance: redaction + drift --------------------


def test_redact_config_strips_secret_keys():
    cfg = {
        "provider": "anthropic",
        "model": "claude",
        "anthropic_api_key": "sk-secret",
        "openai_api_key": "sk-other",
        "wsjpt_api_key": "tok",
        "ollama_base_url": "http://localhost:11434",
    }
    redacted = redact_config(cfg)
    assert redacted == {
        "provider": "anthropic",
        "model": "claude",
        "ollama_base_url": "http://localhost:11434",
    }
    # Original is untouched (a copy is returned).
    assert "anthropic_api_key" in cfg


def test_redact_config_matches_token_secret_password_credential():
    cfg = {
        "keep": 1,
        "auth_token": "t",
        "client_secret": "s",
        "db_password": "p",
        "gcp_credential": "c",
    }
    assert redact_config(cfg) == {"keep": 1}


# -------------------- config file permissions --------------------


def test_save_config_writes_owner_only_mode(isolated_config):
    # The config holds provider API keys; it must not be world-readable.
    save_config({"anthropic_api_key": "sk-secret"})
    assert isolated_config.stat().st_mode & 0o777 == 0o600


def test_save_config_tightens_preexisting_loose_mode(isolated_config):
    # write_text leaves an existing file's mode untouched, so an already-0644
    # config must still get clamped back to 0600 on the next save.
    isolated_config.write_text("anthropic_api_key: old\n", encoding="utf-8")
    isolated_config.chmod(0o644)
    save_config({"anthropic_api_key": "sk-new"})
    assert isolated_config.stat().st_mode & 0o777 == 0o600


def test_config_drift_none_prior_is_silent():
    assert config_drift(None, {"provider": "anthropic"}) == []


def test_config_drift_reports_changed_added_and_removed_fields():
    prior = {"provider": "anthropic", "model": "old", "temperature": 0}
    current = {"provider": "anthropic", "model": "new", "vision_model": "v"}
    drift = config_drift(prior, current)
    # model changed; temperature dropped; vision_model added — provider stable.
    assert drift == [
        "model: 'old' → 'new'",
        "temperature: 0 → '<unset>'",
        "vision_model: '<unset>' → 'v'",
    ]


def test_config_drift_identical_is_silent():
    cfg = {"provider": "anthropic", "model": "x"}
    assert config_drift(cfg, dict(cfg)) == []
