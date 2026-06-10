import json

import pytest

from bartleby.benchmark.refs import ModelRef
from bartleby.benchmark.stores import (
    BenchmarkRoot,
    append_record,
    in_window,
    parse_when,
    read_records,
    read_store,
)


def test_append_creates_store_and_stamps_timestamp(tmp_path):
    path = tmp_path / "results" / "cell.jsonl"
    append_record(path, {"ok": True})
    append_record(path, {"ok": False, "timestamp": 123.0})
    records = read_records(path)
    assert len(records) == 2
    assert records[0]["timestamp"] > 0
    assert records[1]["timestamp"] == 123.0  # explicit timestamp not clobbered


def test_read_records_skips_malformed_line(tmp_path, capsys):
    path = tmp_path / "cell.jsonl"
    path.write_text(json.dumps({"a": 1}) + "\n{not json\n" + json.dumps({"a": 2}) + "\n")
    assert [r["a"] for r in read_records(path)] == [1, 2]
    assert "malformed" in capsys.readouterr().err


def test_read_records_missing_file_is_empty(tmp_path):
    assert read_records(tmp_path / "nope.jsonl") == []


def test_read_store_concatenates_in_name_order(tmp_path):
    append_record(tmp_path / "b.jsonl", {"n": 2})
    append_record(tmp_path / "a.jsonl", {"n": 1})
    assert [r["n"] for r in read_store(tmp_path)] == [1, 2]


def test_cell_paths_use_slugs():
    root = BenchmarkRoot("benchmarks")
    ref = ModelRef("ollama", "gemma4:e2b")
    judge = ModelRef("openai", "gpt-5.5")
    assert root.result_path(ref, "psc-order-0109").name == \
        "ollama_gemma4-e2b_psc-order-0109.jsonl"
    assert root.judgement_path(ref, "psc-order-0109", judge).name == \
        "ollama_gemma4-e2b_psc-order-0109_openai_gpt-5.5.jsonl"


def test_require_demands_corpus_yaml(tmp_path):
    with pytest.raises(SystemExit):
        BenchmarkRoot(tmp_path).require()
    (tmp_path / "corpus.yaml").write_text("doc: doc.pdf\n")
    BenchmarkRoot(tmp_path).require()


def test_load_models_and_judges(tmp_path):
    (tmp_path / "models.yaml").write_text("models:\n  - ollama/gemma4:e2b\n")
    (tmp_path / "judges.yaml").write_text("judges:\n  - openai/gpt-5.5\n")
    root = BenchmarkRoot(tmp_path)
    assert [str(m) for m in root.load_models()] == ["ollama/gemma4:e2b"]
    assert [str(j) for j in root.load_judges()] == ["openai/gpt-5.5"]


def test_load_models_empty_refused(tmp_path):
    (tmp_path / "models.yaml").write_text("models: []\n")
    with pytest.raises(SystemExit):
        BenchmarkRoot(tmp_path).load_models()


def test_window_filtering():
    since = parse_when("2026-06-01")
    until = parse_when("2026-06-09", end=True)  # through the whole day
    inside = {"timestamp": parse_when("2026-06-09") + 3600}
    before = {"timestamp": parse_when("2026-05-31")}
    after = {"timestamp": parse_when("2026-06-10") + 1}
    assert in_window(inside, since, until)
    assert not in_window(before, since, until)
    assert not in_window(after, since, until)
    assert in_window(inside, None, None)
    assert not in_window({}, None, None)


def test_parse_when_rejects_garbage():
    with pytest.raises(SystemExit):
        parse_when("last tuesday")
