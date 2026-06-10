import csv
import json

import pytest

from bartleby.benchmark import report
from bartleby.benchmark.judging import summary_sha
from bartleby.benchmark.refs import ModelRef
from bartleby.benchmark.stores import BenchmarkRoot, append_record

ALPHA = ModelRef("ollama", "alpha:1b")
BETA = ModelRef("ollama", "beta:7b")
GAMMA = ModelRef("ollama", "gamma:3b")
NANO = ModelRef("openai", "gpt-5-nano")
JUDGE = ModelRef("openai", "gpt-5.5")
DOCS = ["d1", "d2", "d3"]


def _summary(tag):
    return {"title": f"T {tag}", "description": "D", "text": f"body {tag}",
            "authored_date": None}


def _run(root, ref, doc, idx, summary=None, ok=True, tps=None, ts=None,
         prompt_sha="p1", source_sha="s1", temperature=0.0, eval_count=None):
    record = {
        "provider": ref.provider, "model": ref.model, "doc": doc,
        "run_index": idx, "ok": ok, "wall_seconds": 5.0,
        "load_duration_ns": int(1e9),
        "source_sha": source_sha, "prompt_sha": prompt_sha,
        "temperature": temperature,
    }
    if eval_count is not None:
        record["eval_count"] = eval_count
    if ok:
        record["summary"] = summary
        if tps is not None:
            record["tokens_per_second"] = tps
    else:
        record["error"] = "schema validation failed: boom"
        record["raw_output"] = "not json"
    if ts is not None:
        record["timestamp"] = ts
    append_record(root.result_path(ref, doc), record)


def _judgment(root, ref, doc, summary, mean, ts=None):
    record = {
        "provider": ref.provider, "model": ref.model, "doc": doc,
        "judge_provider": JUDGE.provider, "judge_model": JUDGE.model,
        "summary_sha": summary_sha(summary), "judge_pass": 0, "ok": True,
        "scores": {"mean": mean},
    }
    if ts is not None:
        record["timestamp"] = ts
    append_record(root.judgement_path(ref, doc, JUDGE), record)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "corpus.yaml").write_text("d1: a.pdf\n")  # require() only
    return BenchmarkRoot(tmp_path)


@pytest.fixture
def populated(root):
    """The synthetic scenario whose numbers were hand-verified:

    - alpha: 3 identical runs/doc; per-doc pass means avg to base + 1/12 for
      bases 4.5/4.0/3.5 → cells 4.58/4.08/3.58 → overall 4.08.
    - beta: per doc, 2 runs of summary X (pass-mean 4.0833) + 1 run of
      summary Y (pass-mean 3.0833) → run-weighted (2·4.0833 + 3.0833)/3 = 3.75.
    - gamma: 2 OK + 1 schema-fail per doc → 67% → disqualified.
    - nano: cloud reference row, 1 run/doc, quality 4.9, no local throughput.
    """
    for di, doc in enumerate(DOCS):
        base = 4.5 - 0.5 * di
        a = _summary(f"alpha-{doc}")
        for i in range(3):
            _run(root, ALPHA, doc, i, a, tps=100.0)
        for mean in (base, base + 0.25, base):
            _judgment(root, ALPHA, doc, a, mean)

        x, y = _summary(f"beta-x-{doc}"), _summary(f"beta-y-{doc}")
        _run(root, BETA, doc, 0, x, tps=50.0)
        _run(root, BETA, doc, 1, x, tps=50.0)
        _run(root, BETA, doc, 2, y, tps=50.0)
        for mean in (4.0, 4.25, 4.0):
            _judgment(root, BETA, doc, x, mean)
        for mean in (3.0, 3.25, 3.0):
            _judgment(root, BETA, doc, y, mean)

        g = _summary(f"gamma-{doc}")
        _run(root, GAMMA, doc, 0, g, tps=80.0)
        _run(root, GAMMA, doc, 1, g, tps=80.0)
        _run(root, GAMMA, doc, 2, None, ok=False)

        n = _summary(f"nano-{doc}")
        _run(root, NANO, doc, 0, n, temperature=None)  # no tps — cloud
        _judgment(root, NANO, doc, n, 4.9)
    return root


def test_quality_aggregation_matches_hand_verified_numbers(populated):
    runs = report.load_runs(populated)
    judgments = report.load_judgments(populated)
    cells = report.quality_cells(runs, judgments)
    assert cells[(ALPHA, "d1")]["score"] == pytest.approx(4.5 + 1 / 12)
    assert cells[(BETA, "d1")]["score"] == pytest.approx(3.75)
    quality = report.quality_by_ref(cells)
    assert quality[ALPHA] == pytest.approx(4.0 + 1 / 12)
    assert quality[BETA] == pytest.approx(3.75)


def test_weights_recomputed_from_runs_not_judgments(populated):
    """Three more runs of beta's summary X shift the weight 5:1 without any
    new judgments — stale judgment-side counts must not pin the old 2:1."""
    x = _summary("beta-x-d1")
    for i in range(3, 6):
        _run(populated, BETA, "d1", i, x, tps=50.0)
    cells = report.quality_cells(report.load_runs(populated),
                                 report.load_judgments(populated))
    expected = (5 * (4.0833333) + 1 * (3.0833333)) / 6
    assert cells[(BETA, "d1")]["score"] == pytest.approx(expected, abs=1e-4)


def test_leaderboard_renders_markdown_with_frontier_and_disqualified(populated, capsys):
    assert report.leaderboard(populated) == 0
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("| ")]
    # Sorted by quality: nano (4.9, reference) above alpha above beta.
    assert lines[1].startswith("| openai/gpt-5-nano †")
    assert "★" in lines[2] and "ollama/alpha:1b" in lines[2]   # frontier: local best
    assert "★" not in lines[1]  # cloud row never starred
    assert "ollama/beta:7b" in lines[3] and "★" not in lines[3]  # dominated
    assert "gamma" in out and "67%" in out  # disqualified table
    assert "Mean quality by document" in out
    assert "(3r/3p)" in out  # per-cell evidence counts


def test_leaderboard_window_filter_excludes_old_records(populated, capsys):
    old = 1.0  # epoch dawn
    _run(populated, ALPHA, "d1", 99, _summary("ancient"), tps=1.0, ts=old)
    report.leaderboard(populated, since=100.0)
    out = capsys.readouterr().out
    # the ancient run is outside the window: alpha still shows 9 OK runs
    row = next(l for l in out.splitlines() if "alpha" in l)
    assert "| 9 |" in row


def test_leaderboard_csv_output(populated, tmp_path, capsys):
    out_csv = tmp_path / "lb.csv"
    report.leaderboard(populated, output=out_csv)
    capsys.readouterr()
    rows = list(csv.DictReader(out_csv.open()))
    assert [r["model"] for r in rows] == ["gpt-5-nano", "alpha:1b", "beta:7b"]
    alpha = rows[1]
    assert alpha["pareto_optimal"] == "True"
    assert float(alpha["mean_quality"]) == pytest.approx(4.0 + 1 / 12)
    assert float(alpha["quality_d3"]) == pytest.approx(3.5 + 1 / 12)
    assert rows[0]["tokens_per_second"] == ""  # cloud: no local throughput


def test_wall_seconds_per_token_measured_for_cloud_too(root, tmp_path, capsys):
    # wall 5.0 − load 1.0 = 4.0 inference s. Local has tps; cloud doesn't, but
    # both carry eval_count, so both get a Wall s/tok number (the apples-to-
    # apples speed comparison): 4.0/200 = 0.02 local, 4.0/100 = 0.04 cloud.
    _run(root, ALPHA, "d1", 0, _summary("a"), tps=100.0, eval_count=200)
    _run(root, NANO, "d1", 0, _summary("n"), temperature=None, eval_count=100)
    out_csv = tmp_path / "lb.csv"
    assert report.leaderboard(root, output=out_csv) == 0
    out = capsys.readouterr().out
    assert "Wall s/tok" in out and "Pareto optimal" in out
    assert "Inference (s)" not in out and "Frontier" not in out
    alpha_row = next(l for l in out.splitlines() if "alpha" in l)
    nano_row = next(l for l in out.splitlines() if "gpt-5-nano" in l)
    assert "0.0200" in alpha_row
    assert nano_row.split("|")[3].strip() == "—"  # no Tok/s for cloud
    assert "0.0400" in nano_row                    # but it does get Wall s/tok
    rows = {r["model"]: r for r in csv.DictReader(out_csv.open())}
    assert float(rows["alpha:1b"]["wall_seconds_per_token"]) == pytest.approx(0.02)
    assert float(rows["gpt-5-nano"]["wall_seconds_per_token"]) == pytest.approx(0.04)


def test_wall_seconds_per_token_blank_without_eval_count(root, capsys):
    _run(root, ALPHA, "d1", 0, _summary("a"), tps=100.0)  # no eval_count
    assert report.leaderboard(root) == 0
    out = capsys.readouterr().out
    alpha_row = next(l for l in out.splitlines() if "alpha" in l)
    assert alpha_row.split("|")[4].strip() == "—"  # Wall s/tok column blank


def test_heterogeneity_warning_on_mixed_prompt_sha(populated, capsys):
    _run(populated, ALPHA, "d1", 99, _summary("alpha-d1"), tps=100.0,
         prompt_sha="p2")
    report.leaderboard(populated)
    out = capsys.readouterr().out
    assert "mixes prompt_sha" in out and "alpha" in out


def test_heterogeneity_warning_on_mixed_temperature(populated, capsys):
    _run(populated, ALPHA, "d1", 99, _summary("alpha-d1"), tps=100.0,
         temperature=0.2)
    report.leaderboard(populated)
    out = capsys.readouterr().out
    assert "mixes temperature" in out and "alpha" in out


def test_no_temperature_warning_for_cloud_provider_defaults(populated, capsys):
    # Reference rows record temperature=None (provider default) on every run;
    # None is "unpinned", not a regime, so an all-None cell must not warn.
    _run(populated, NANO, "d1", 99, _summary("nano-d1"), temperature=None)
    report.leaderboard(populated)
    out = capsys.readouterr().out
    assert "gpt-5-nano · d1 mixes" not in out


def test_judges_filter_selects_judge(populated, capsys):
    other_judge = ModelRef("openai", "other")
    judgments = report.load_judgments(populated, judges=[other_judge])
    assert judgments == []
    judgments = report.load_judgments(populated, judges=[JUDGE])
    assert judgments


def test_models_and_documents_filters(populated):
    runs = report.load_runs(populated, models=[ALPHA], documents=["d2"])
    assert {(r["model"], r["doc"]) for r in runs} == {("alpha:1b", "d2")}


def test_blind_writes_per_doc_sections_and_key(populated, tmp_path, capsys):
    out = tmp_path / "blind"
    assert report.blind(populated, out, seed=7) == 0
    md = (out / "blinded_summaries.md").read_text()
    key = json.loads((out / "key.json").read_text())
    assert md.count("# Document:") == 3
    # 4 models × 3 docs but gamma fails on no doc (it has OK runs) → 12 entries
    assert len(key) == 12
    assert {k["doc"] for k in key.values()} == set(DOCS)
    # Blinded: model identifiers live only in the key file, never the markdown.
    # (Summary *content* may echo anything — only the refs must not leak.)
    for ref in (ALPHA, BETA, GAMMA, NANO):
        assert str(ref) not in md
    assert "/" in key["A"]["model"]  # provider/model form


def test_errors_lists_failures_with_doc(populated, capsys):
    assert report.errors(populated) == 0
    out = capsys.readouterr().out
    assert "gamma:3b · d1" in out
    assert "not json" in out


def test_errors_clean_when_none(root, capsys):
    _run(root, ALPHA, "d1", 0, _summary("a"), tps=1.0)
    assert report.errors(root) == 0
    assert "No errors" in capsys.readouterr().out


def test_leaderboard_empty_window_errors(populated, capsys):
    assert report.leaderboard(populated, since=4102444800.0) == 1  # year 2100
