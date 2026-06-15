import pytest

from bartleby.benchmark import sources as sources_mod
from bartleby.benchmark.sources import (
    SourceText,
    ensure_source,
    load_corpus,
    load_source,
    select_documents,
    source_sha,
)
from bartleby.benchmark.stores import BenchmarkRoot


@pytest.fixture
def root(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "corpus" / "b.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "corpus.yaml").write_text("doc-a: a.pdf\ndoc-b: b.pdf\n")
    return BenchmarkRoot(tmp_path)


def test_load_corpus_maps_ids_to_paths(root):
    corpus = load_corpus(root)
    assert sorted(corpus) == ["doc-a", "doc-b"]
    assert corpus["doc-a"].name == "a.pdf"


def test_load_corpus_rejects_bad_doc_id(root):
    root.corpus_yaml.write_text("Bad_ID: a.pdf\n")
    with pytest.raises(SystemExit):
        load_corpus(root)


def test_load_corpus_rejects_missing_pdf(root):
    root.corpus_yaml.write_text("doc-c: missing.pdf\n")
    with pytest.raises(SystemExit):
        load_corpus(root)


def test_select_documents_filters_and_refuses_unknown(root):
    corpus = load_corpus(root)
    assert list(select_documents(corpus, ["doc-b"])) == ["doc-b"]
    assert select_documents(corpus, None) is corpus
    with pytest.raises(SystemExit):
        select_documents(corpus, ["doc-x"])


def test_ensure_source_extracts_once_then_serves_cache(root, monkeypatch):
    calls = []

    def fake_extract(pdf):
        calls.append(pdf)
        return "the extracted text", {"page_count": 1, "image_routed_pages": []}

    monkeypatch.setattr(sources_mod, "build_summary_input", fake_extract)
    corpus = load_corpus(root)

    first = ensure_source(root, "doc-a", corpus["doc-a"])
    again = ensure_source(root, "doc-a", corpus["doc-a"])
    assert len(calls) == 1  # second call served from sources/<doc-id>.txt
    assert first == again
    assert first.sha == source_sha("the extracted text")
    assert first.tokens > 0
    assert (root.sources_dir / "doc-a.txt").read_text() == "the extracted text"


def test_ensure_source_truncates_and_caches_truncated_text(root, monkeypatch):
    monkeypatch.setattr(
        sources_mod, "build_summary_input",
        lambda pdf: ("word " * 1000, {"page_count": 1, "image_routed_pages": []}),
    )
    corpus = load_corpus(root)
    src = ensure_source(root, "doc-a", corpus["doc-a"], max_tokens=10)
    assert src.tokens == 10  # post-truncation count — what the model actually sees
    assert load_source(root, "doc-a").sha == src.sha


def test_ensure_source_refuses_empty_extraction(root, monkeypatch):
    monkeypatch.setattr(
        sources_mod, "build_summary_input",
        lambda pdf: ("  ", {"page_count": 1, "image_routed_pages": []}),
    )
    corpus = load_corpus(root)
    with pytest.raises(SystemExit):
        ensure_source(root, "doc-a", corpus["doc-a"])


def test_load_source_none_when_uncached(root):
    assert load_source(root, "doc-a") is None


def test_source_text_is_frozen_value():
    s = SourceText("d", "t", "sha", 1)
    with pytest.raises(AttributeError):
        s.text = "other"


def test_named_extraction_served_from_precommitted_fixture(root):
    """A named extraction variant is served from a pre-committed fixture file,
    not extracted live — ensure_source serves it without calling build_summary_input."""
    # Write a pre-committed fixture (e.g. docling output):
    root.sources_dir.mkdir(parents=True, exist_ok=True)
    root.source_path("doc-a", "docling").write_text("docling extracted text")

    corpus = load_corpus(root)
    src = ensure_source(root, "doc-a", corpus["doc-a"], extraction="docling")
    assert src.text == "docling extracted text"
    assert src.sha == source_sha("docling extracted text")


def test_named_extraction_raises_when_fixture_missing(root):
    """A named extraction with no pre-committed fixture raises with guidance."""
    corpus = load_corpus(root)
    with pytest.raises(SystemExit, match="pre-committed fixture"):
        ensure_source(root, "doc-a", corpus["doc-a"], extraction="docling")


def test_source_path_encoding(root):
    """Default extraction uses <doc-id>.txt; named uses <doc-id>-<extraction>.txt."""
    assert root.source_path("doc-a") == root.sources_dir / "doc-a.txt"
    assert root.source_path("doc-a", "pdfplumber") == root.sources_dir / "doc-a.txt"
    assert root.source_path("doc-a", "docling") == root.sources_dir / "doc-a-docling.txt"
    assert (root.source_path("doc-a", "image-fixture")
            == root.sources_dir / "doc-a-image-fixture.txt")
