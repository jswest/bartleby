"""`bartleby benchmark judge` — blind, idempotent multi-pass scoring.

OK runs dedupe into distinct summaries per cell (by ``summary_sha``, the hash
of the canonical summary JSON — at temp 0.0 repeat runs are usually
byte-identical), and each distinct summary is topped up to ``--passes`` OK
judgments from the chosen judge. Running the command twice in a row is a
no-op; raising ``--passes`` later adds only the difference. A 1–5 integer
rubric draws different scores for the same text on different calls, so the
leaderboard averages a summary's passes rather than trusting any single one.

The judge is blind by construction: its prompt contains only the source text
and the candidate summary, never the model name. The source is the cached
``sources/<doc-id>.txt`` — verified by sha against what each run actually saw,
so drifted text aborts loudly instead of being silently scored against.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time

from pydantic import BaseModel, Field

from bartleby.benchmark.refs import ModelRef
from bartleby.benchmark.sources import load_source
from bartleby.benchmark.stores import BenchmarkRoot, append_record, read_records

DEFAULT_PASSES = 3
RUBRIC_AXES = ("faithfulness", "coverage", "conciseness", "constraint_compliance")


class JudgeScore(BaseModel):
    """Per-summary rubric the judge fills in. Each axis is 1 (worst)-5 (best)."""

    faithfulness: int = Field(ge=1, le=5, description="No invented figures, dates, names, or claims.")
    coverage: int = Field(ge=1, le=5, description="Captures the document's key facts and structure.")
    conciseness: int = Field(ge=1, le=5, description="Dense and non-redundant; no filler.")
    constraint_compliance: int = Field(
        ge=1, le=5,
        description="Honors the schema limits: title <= 60 chars, description ~20 words / <= 200 chars.",
    )
    rationale: str = Field(description="One or two sentences justifying the scores; cite a specific flaw if any.")


# Framing only — the per-axis definitions and the exact constraint thresholds
# live in the JudgeScore field descriptions, which reach the model through the
# response_format schema. Don't restate them here or the two copies will drift.
JUDGE_INSTRUCTIONS = (
    "You are grading a machine-generated summary against its source document. "
    "Score each rubric axis from 1 (worst) to 5 (best) per its definition, then "
    "give a one-line rationale. Be strict and specific. You do not know which "
    "model wrote this summary; judge only what is in front of you."
)


def summary_sha(summary: dict) -> str:
    return hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode()).hexdigest()[:16]


def _user_prompt(source_text: str, summary: dict) -> str:
    """The grading prompt — source document + candidate summary. Shared by the
    OpenAI judge (as the user message) and the anthropic-cc judge (as the
    ``claude -p`` prompt), so both score identical text."""
    candidate = (
        f"title: {summary['title']}\n"
        f"description: {summary['description']}\n"
        f"text:\n{summary['text']}"
    )
    return (
        f"SOURCE DOCUMENT:\n{source_text}\n\n"
        f"CANDIDATE SUMMARY:\n{candidate}"
    )


def _build_prompt(source_text: str, summary: dict) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_INSTRUCTIONS},
        {"role": "user", "content": _user_prompt(source_text, summary)},
    ]


def judge_summary(client, model: str, source_text: str, summary: dict) -> dict:
    wall_start = time.perf_counter()
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=_build_prompt(source_text, summary),
            response_format=JudgeScore,
        )
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "wall_seconds": time.perf_counter() - wall_start}
    parsed = response.choices[0].message.parsed
    if parsed is None:
        return {"ok": False, "error": "judge returned no parsed payload",
                "wall_seconds": time.perf_counter() - wall_start}
    scores = parsed.model_dump()
    scores["mean"] = sum(scores[a] for a in RUBRIC_AXES) / len(RUBRIC_AXES)
    return {"ok": True, "scores": scores, "wall_seconds": time.perf_counter() - wall_start}


def _collect_work(root: BenchmarkRoot, judge: ModelRef, passes: int) -> list[dict]:
    """One work item per distinct summary still short of ``passes`` OK
    judgments. Failed judgment records don't count — they're retried."""
    work: list[dict] = []
    for path in sorted(root.results_dir.glob("*.jsonl")):
        ok_runs = [r for r in read_records(path) if r.get("ok")]
        if not ok_runs:
            continue
        first = ok_runs[0]
        ref = ModelRef(first["provider"], first["model"])
        doc_id = first["doc"]

        source = load_source(root, doc_id)
        if source is None:
            raise SystemExit(
                f"sources/{doc_id}.txt missing — the judge scores against the "
                f"exact text the models saw; re-run `bartleby benchmark "
                f"summarize` to rebuild it.")

        groups: dict[str, dict] = {}
        for r in ok_runs:
            if r["source_sha"] != source.sha:
                raise SystemExit(
                    f"source drift for doc {doc_id!r}: run record carries "
                    f"source_sha {r['source_sha']} but sources/{doc_id}.txt "
                    f"hashes to {source.sha}. Refusing to judge against text "
                    f"the model never saw. (Did sources/ get re-extracted? "
                    f"Old runs can't be judged against new text.)")
            groups.setdefault(summary_sha(r["summary"]), r["summary"])

        jpath = root.judgement_path(ref, doc_id, judge)
        have: dict[str, int] = {}
        for j in read_records(jpath):
            if j.get("ok"):
                have[j["summary_sha"]] = have.get(j["summary_sha"], 0) + 1

        for sha, summary in groups.items():
            need = passes - have.get(sha, 0)
            if need > 0:
                work.append({"ref": ref, "doc_id": doc_id, "sha": sha,
                             "summary": summary, "source_text": source.text,
                             "source_sha": source.sha, "jpath": jpath,
                             "done": have.get(sha, 0), "need": need})
    return work


def _make_judge_fn(root: BenchmarkRoot, judge: ModelRef, judge_client):
    """A ``(source_text, summary) -> result`` callable for the judge's provider.
    ``judge_client`` is the test-injection seam: an OpenAI client for ``openai``,
    a subprocess runner for ``anthropic-cc``."""
    if judge.provider == "openai":
        if judge_client is None:
            from bartleby.benchmark.clients import make_openai_client
            judge_client = make_openai_client(root)
        return lambda src, summary: judge_summary(
            judge_client, judge.model, src, summary)

    from bartleby.benchmark import cc_judge
    cc_judge.preflight()
    return lambda src, summary: cc_judge.judge_summary_cc(
        judge.model, src, summary, runner=judge_client)


def run(root: BenchmarkRoot, judge: ModelRef | None = None,
        passes: int = DEFAULT_PASSES, judge_client=None) -> None:
    """Top up every distinct summary to ``passes`` judgments. ``judge_client``
    is injectable for tests: an OpenAI client for ``openai`` judges, or a
    subprocess runner callable for ``anthropic-cc`` judges."""
    root.require()
    judge = judge or root.load_judges()[0]
    if judge.provider not in ("openai", "anthropic-cc"):
        raise SystemExit(
            f"Unsupported judge provider {judge.provider!r} (got {judge}); "
            f"judges must be openai/<model> or anthropic-cc/<model>")

    work = _collect_work(root, judge, passes)
    if not work:
        print(f"Nothing to judge — every distinct summary already has "
              f"{passes} pass(es) from {judge}.", file=sys.stderr)
        return

    judge_fn = _make_judge_fn(root, judge, judge_client)

    total = sum(item["need"] for item in work)
    print(f"Judging {len(work)} distinct summary(ies), {total} call(s) "
          f"with {judge}...", file=sys.stderr)
    call_no = 0
    for item in work:
        for i in range(item["need"]):
            call_no += 1
            print(f"  [{call_no}/{total}] {item['ref']} · {item['doc_id']} "
                  f"(summary {item['sha'][:8]}, pass {item['done'] + i})",
                  file=sys.stderr, flush=True)
            result = judge_fn(item["source_text"], item["summary"])
            append_record(item["jpath"], {
                "provider": item["ref"].provider,
                "model": item["ref"].model,
                "doc": item["doc_id"],
                "judge_provider": judge.provider,
                "judge_model": judge.model,
                "summary_sha": item["sha"],
                "judge_pass": item["done"] + i,
                "source_sha": item["source_sha"],
                **result,
            })
    print(f"\nAppended {total} judgment(s) under {root.judgements_dir}/",
          file=sys.stderr)
