import json
from types import SimpleNamespace

import pytest

from bartleby.benchmark import judging
from bartleby.benchmark.refs import ModelRef
from bartleby.benchmark.sources import source_sha
from bartleby.benchmark.stores import BenchmarkRoot, append_record, read_records

REF = ModelRef("ollama", "tiny:1b")
JUDGE = ModelRef("openai", "gpt-5.5")
SOURCE = "the source document text"


class FakeJudge:
    """Counts calls; returns a fixed parsed JudgeScore."""

    def __init__(self):
        self.calls = 0
        score = judging.JudgeScore(
            faithfulness=5, coverage=4, conciseness=5,
            constraint_compliance=5, rationale="fine")

        def parse(**kw):
            self.calls += 1
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(parsed=score))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=parse))


def _summary(text="body"):
    return {"title": "T", "description": "D", "text": text, "authored_date": None}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "corpus.yaml").write_text("doc-a: a.pdf\n")  # require() only
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "doc-a.txt").write_text(SOURCE)
    (tmp_path / "judges.yaml").write_text("judges:\n  - openai/gpt-5.5\n")
    return BenchmarkRoot(tmp_path)


def _add_run(root, summary, run_index, ok=True, sha=None):
    append_record(root.result_path(REF, "doc-a"), {
        "provider": REF.provider, "model": REF.model, "doc": "doc-a",
        "run_index": run_index, "ok": ok,
        **({"summary": summary} if ok else {"error": "boom"}),
        "source_sha": sha or source_sha(SOURCE),
    })


def test_topup_judges_each_distinct_summary(root):
    _add_run(root, _summary(), 0)
    _add_run(root, _summary(), 1)          # identical → same sha, judged once-set
    _add_run(root, _summary("other"), 2)   # distinct → its own passes
    client = FakeJudge()
    judging.run(root, passes=3, judge_client=client)
    assert client.calls == 6  # 2 distinct × 3 passes, not 3 runs × 3

    records = read_records(root.judgement_path(REF, "doc-a", JUDGE))
    assert len(records) == 6
    assert {r["summary_sha"] for r in records} == {
        judging.summary_sha(_summary()), judging.summary_sha(_summary("other"))}
    assert all(r["judge_model"] == "gpt-5.5" for r in records)
    assert sorted(r["judge_pass"] for r in records) == [0, 0, 1, 1, 2, 2]
    assert records[0]["scores"]["mean"] == pytest.approx(4.75)


def test_second_invocation_is_a_noop(root, capsys):
    _add_run(root, _summary(), 0)
    judging.run(root, passes=3, judge_client=FakeJudge())
    again = FakeJudge()
    judging.run(root, passes=3, judge_client=again)
    assert again.calls == 0
    assert "Nothing to judge" in capsys.readouterr().err


def test_raising_passes_tops_up_the_difference(root):
    _add_run(root, _summary(), 0)
    judging.run(root, passes=3, judge_client=FakeJudge())
    more = FakeJudge()
    judging.run(root, passes=5, judge_client=more)
    assert more.calls == 2
    passes = [r["judge_pass"] for r in
              read_records(root.judgement_path(REF, "doc-a", JUDGE))]
    assert passes == [0, 1, 2, 3, 4]


def test_failed_judgments_dont_count_toward_passes(root):
    _add_run(root, _summary(), 0)
    append_record(root.judgement_path(REF, "doc-a", JUDGE), {
        "provider": REF.provider, "model": REF.model, "doc": "doc-a",
        "judge_provider": JUDGE.provider, "judge_model": JUDGE.model,
        "summary_sha": judging.summary_sha(_summary()), "judge_pass": 0,
        "ok": False, "error": "rate limited",
    })
    client = FakeJudge()
    judging.run(root, passes=1, judge_client=client)
    assert client.calls == 1  # the failure is retried, not counted


def test_failed_runs_are_not_judged(root):
    _add_run(root, None, 0, ok=False)
    client = FakeJudge()
    judging.run(root, passes=3, judge_client=client)
    assert client.calls == 0


def test_source_drift_aborts_loudly(root):
    _add_run(root, _summary(), 0, sha="0123456789abcdef")
    with pytest.raises(SystemExit, match="source drift"):
        judging.run(root, passes=1, judge_client=FakeJudge())


def test_missing_source_cache_aborts(root):
    _add_run(root, _summary(), 0)
    (root.sources_dir / "doc-a.txt").unlink()
    with pytest.raises(SystemExit, match="sources/doc-a.txt missing"):
        judging.run(root, passes=1, judge_client=FakeJudge())


def test_default_judge_comes_from_judges_yaml(root):
    _add_run(root, _summary(), 0)
    judging.run(root, passes=1, judge_client=FakeJudge())
    assert root.judgement_path(REF, "doc-a", JUDGE).exists()


def test_unsupported_judge_provider_refused(root):
    with pytest.raises(SystemExit, match="Unsupported judge provider"):
        judging.run(root, judge=ModelRef("ollama", "qwen3.6:35b"),
                    judge_client=FakeJudge())


def test_run_dispatches_to_anthropic_cc(root, monkeypatch):
    """An anthropic-cc judge tops up via the injected subprocess runner — same
    idempotent bookkeeping, records tagged with the cc provider."""
    monkeypatch.setattr("bartleby.benchmark.cc_judge.preflight", lambda: None)
    _add_run(root, _summary(), 0)
    seen = []

    def fake_runner(argv, *, env):
        seen.append((argv, env))
        payload = {"faithfulness": 5, "coverage": 4, "conciseness": 5,
                   "constraint_compliance": 5, "rationale": "ok"}
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(
            {"subtype": "success", "is_error": False, "structured_output": payload}))

    judge = ModelRef("anthropic-cc", "claude-opus-4-8")
    judging.run(root, judge=judge, passes=2, judge_client=fake_runner)

    records = read_records(root.judgement_path(REF, "doc-a", judge))
    assert len(records) == 2
    assert all(r["judge_provider"] == "anthropic-cc" for r in records)
    assert records[0]["scores"]["mean"] == pytest.approx(4.75)
    argv = seen[0][0]
    assert "--model" in argv and "claude-opus-4-8" in argv and "--bare" not in argv


def test_judge_is_blind_to_model_name(root):
    _add_run(root, _summary(), 0)
    seen = {}

    class Spy(FakeJudge):
        def __init__(self):
            super().__init__()
            inner = self.chat.completions.parse

            def parse(**kw):
                seen["messages"] = kw["messages"]
                return inner(**kw)

            self.chat = SimpleNamespace(completions=SimpleNamespace(parse=parse))

    judging.run(root, passes=1, judge_client=Spy())
    prompt_text = " ".join(m["content"] for m in seen["messages"])
    assert "tiny" not in prompt_text and "ollama" not in prompt_text
    assert SOURCE in prompt_text
