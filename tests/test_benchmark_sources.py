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
