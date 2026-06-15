import json
from types import SimpleNamespace

import pytest

from bartleby.benchmark import sources as sources_mod
from bartleby.benchmark import summarize as summarize_mod
from bartleby.benchmark.refs import ModelRef
from bartleby.benchmark.stores import BenchmarkRoot, read_records
from bartleby.providers.base import DocumentSummary

SUMMARY = {"title": "T", "description": "D", "text": "body text",
           "authored_date": None}


def _valid_summary_json() -> str:
    return DocumentSummary(**SUMMARY).model_dump_json()


class FakeOllama:
    """Streams a canned response in two chunks, with Ollama timing fields."""

    def __init__(self, content: str | None = None, fail: bool = False):
        self.content = content if content is not None else _valid_summary_json()
        self.fail = fail
        self.calls: list[dict] = []

    def chat(self, model, messages, format, options, stream):
        self.calls.append({"model": model, "options": options})
        if self.fail:
            raise RuntimeError("model exploded")
        half = len(self.content) // 2
        yield SimpleNamespace(message=SimpleNamespace(content=self.content[:half]),
                              done=False)
        yield SimpleNamespace(
            message=SimpleNamespace(content=self.content[half:]), done=True,
            total_duration=int(5e9), load_duration=int(1e9),
            prompt_eval_count=100, prompt_eval_duration=int(1e9),
            eval_count=50, eval_duration=int(2e9),
        )

    def list(self):
        return SimpleNamespace(models=[])


class FakeOpenAI:
    def __init__(self, parsed=None):
        self.parsed = parsed if parsed is not None else DocumentSummary(**SUMMARY)
        parse_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(parsed=self.parsed, content="raw"))],
            usage=SimpleNamespace(completion_tokens=42, prompt_tokens=2000),
        )
        self.chat = SimpleNamespace(completions=SimpleNamespace(
            parse=lambda **kw: parse_response))


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.pdf").write_bytes(b"%PDF")
    (tmp_path / "corpus" / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "corpus.yaml").write_text("doc-a: a.pdf\ndoc-b: b.pdf\n")
    monkeypatch.setattr(
        sources_mod, "build_summary_input",
        lambda pdf: (f"source of {pdf.name}", {"page_count": 1,
                                               "image_routed_pages": []}),
    )
    return BenchmarkRoot(tmp_path)


def test_run_appends_one_record_per_cell(root):
    refs = [ModelRef("ollama", "tiny:1b")]
    summarize_mod.run(root, refs, None, runs=1, seed=1,
                      ollama_client=FakeOllama())
    for doc_id in ("doc-a", "doc-b"):
        records = read_records(root.result_path(refs[0], doc_id))
        assert len(records) == 1
        r = records[0]
        assert (r["provider"], r["model"], r["doc"]) == ("ollama", "tiny:1b", doc_id)
        assert r["ok"] and r["summary"]["title"] == "T"
        assert r["run_index"] == 0
        assert r["tokens_per_second"] == pytest.approx(25.0)
        # provenance travels on every record
        assert r["source_sha"] and r["prompt_sha"] and r["bartleby_version"]
        assert r["temperature"] == 0.0
        assert r["source_tokens"] > 0


def test_second_invocation_continues_run_index(root):
    refs = [ModelRef("ollama", "tiny:1b")]
    summarize_mod.run(root, refs, ["doc-a"], runs=2, seed=1,
                      ollama_client=FakeOllama())
    summarize_mod.run(root, refs, ["doc-a"], runs=1, seed=1,
                      ollama_client=FakeOllama())
    records = read_records(root.result_path(refs[0], "doc-a"))
    assert [r["run_index"] for r in records] == [0, 1, 2]


def test_schema_failure_recorded_not_raised(root):
    refs = [ModelRef("ollama", "tiny:1b")]
    summarize_mod.run(root, refs, ["doc-a"], runs=1,
                      ollama_client=FakeOllama(content="not json"))
    r = read_records(root.result_path(refs[0], "doc-a"))[0]
    assert not r["ok"]
    assert "schema validation failed" in r["error"]
    assert r["raw_output"] == "not json"


def test_call_error_recorded_not_raised(root):
    refs = [ModelRef("ollama", "tiny:1b")]
    summarize_mod.run(root, refs, ["doc-a"], runs=1,
                      ollama_client=FakeOllama(fail=True))
    r = read_records(root.result_path(refs[0], "doc-a"))[0]
    assert not r["ok"] and "model exploded" in r["error"]


def test_openai_reference_row_has_no_local_timings(root):
    refs = [ModelRef("openai", "gpt-5-nano")]
    summarize_mod.run(root, refs, ["doc-a"], runs=1,
                      openai_client=FakeOpenAI())
    r = read_records(root.result_path(refs[0], "doc-a"))[0]
    assert r["ok"] and r["provider"] == "openai"
    assert r["temperature"] is None  # provider default, deliberately unpinned
    assert r.get("tokens_per_second") is None
    assert r.get("load_duration_ns") is None
    assert r["eval_count"] == 42


def test_ollama_temperature_pinned_to_production(root):
    client = FakeOllama()
    summarize_mod.run(root, [ModelRef("ollama", "tiny:1b")], ["doc-a"], runs=1,
                      ollama_client=client)
    assert client.calls[0]["options"] == {"temperature": 0.0}


def test_source_text_reaches_the_model(root):
    seen = {}

    class Spy(FakeOllama):
        def chat(self, model, messages, format, options, stream):
            seen["prompt"] = messages[0]["content"]
            return super().chat(model, messages, format, options, stream)

    summarize_mod.run(root, [ModelRef("ollama", "tiny:1b")], ["doc-a"], runs=1,
                      ollama_client=Spy())
    assert "source of a.pdf" in seen["prompt"]


def test_prompt_sha_stable_and_short():
    assert summarize_mod.prompt_sha() == summarize_mod.prompt_sha()
    assert len(summarize_mod.prompt_sha()) == 16


def test_records_parse_as_json_lines(root):
    refs = [ModelRef("ollama", "tiny:1b")]
    summarize_mod.run(root, refs, ["doc-a"], runs=1, ollama_client=FakeOllama())
    raw = root.result_path(refs[0], "doc-a").read_text().strip()
    assert json.loads(raw)["doc"] == "doc-a"


def test_extraction_field_recorded_on_run(root):
    """The extraction field is stored on every run record (default: pdfplumber)."""
    refs = [ModelRef("ollama", "tiny:1b")]
    summarize_mod.run(root, refs, ["doc-a"], runs=1, ollama_client=FakeOllama())
    r = read_records(root.result_path(refs[0], "doc-a"))[0]
    assert r["extraction"] == "pdfplumber"


def test_named_extraction_uses_separate_store_and_fixture(root, monkeypatch, tmp_path):
    """A named extraction writes to a separate store file and reads the fixture source."""
    # Pre-commit a fixture source for the named extraction
    root.sources_dir.mkdir(parents=True, exist_ok=True)
    root.source_path("doc-a", "docling").write_text("docling text")

    refs = [ModelRef("ollama", "tiny:1b")]
    # build_summary_input must NOT be called (fixture serves the text)
    monkeypatch.setattr(sources_mod, "build_summary_input",
                        lambda pdf: (_ for _ in ()).throw(AssertionError("should not extract")))

    summarize_mod.run(root, refs, ["doc-a"], runs=1, extraction="docling",
                      ollama_client=FakeOllama())

    # Wrote to the extraction-specific store file
    store_path = root.result_path(refs[0], "doc-a", "docling")
    assert store_path.exists()
    assert "_x-docling" in store_path.name
    r = read_records(store_path)[0]
    assert r["extraction"] == "docling"
    assert r["ok"]

    # Default store is untouched
    assert not root.result_path(refs[0], "doc-a").exists()
