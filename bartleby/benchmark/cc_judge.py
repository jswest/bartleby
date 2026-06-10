"""``anthropic-cc`` judge backend — grades summaries via the local ``claude -p``
CLI under the user's own Claude subscription (no OpenAI key, no API-key billing).

Why the CLI and not the API or the Agent SDK: the headless ``claude`` binary is
already installed with Claude Code, needs no extra dependency, and — run as a
subprocess — lets us hand it an explicit environment. That last point is
load-bearing for billing safety: Claude Code's auth precedence puts
``ANTHROPIC_API_KEY``/``ANTHROPIC_AUTH_TOKEN`` *above* the subscription OAuth
login, so leaving those vars in the child's environment would silently bill a
pay-as-you-go API account at API rates. We strip them (``_subscription_env``) so
the judge can only authenticate via the user's local subscription login, and we
never pass ``--bare`` (which forces API-key auth and never reads OAuth/keychain).

Sanctioned for ordinary, individual use (your own login, your own machine), not
for shared/production automation — that's the API-key path. See
``benchmarks/README.md`` → "Judge backends — auth & legal".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

# Friendly judge-model shorthands → the canonical Claude model the CLI's
# ``--model`` accepts. Canonical ``claude-*`` ids and the bare ``opus``/
# ``sonnet``/``haiku`` aliases already work, so they pass straight through.
_MODEL_ALIASES = {
    "fable5": "claude-fable-5",
    "opus4.8": "claude-opus-4-8",
    "sonnet4.6": "claude-sonnet-4-6",
    "haiku4.5": "claude-haiku-4-5",
}

# Auth vars that would override subscription OAuth and silently bill an API
# account; stripped from the `claude` subprocess environment.
_BILLING_OVERRIDE_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

_CALL_TIMEOUT_SECONDS = 300


def resolve_model(token: str) -> str:
    """Map a judge-model token to the model string ``claude --model`` wants."""
    return _MODEL_ALIASES.get(token, token)


def _subscription_env() -> tuple[dict, list[str]]:
    """A copy of the environment with the API-key auth vars removed, forcing the
    `claude` CLI onto the user's subscription login. Returns (env, stripped)."""
    env = dict(os.environ)
    stripped = [v for v in _BILLING_OVERRIDE_VARS if env.pop(v, None) is not None]
    return env, stripped


def _judge_schema() -> dict:
    """The ``JudgeScore`` JSON schema, with the 1-5 axes expressed as ``enum``.
    Claude's structured outputs don't enforce numeric ``minimum``/``maximum``,
    so an enum is the constraint that actually bites; ``JudgeScore.model_validate``
    (with its ``ge=1, le=5``) stays as the final client-side check."""
    from bartleby.benchmark.judging import JudgeScore, RUBRIC_AXES

    schema = JudgeScore.model_json_schema()
    schema["additionalProperties"] = False
    for axis in RUBRIC_AXES:
        prop = schema["properties"][axis]
        prop.pop("minimum", None)
        prop.pop("maximum", None)
        prop["enum"] = [1, 2, 3, 4, 5]
    return schema


def preflight() -> None:
    """Fail fast (``SystemExit``) unless the local ``claude`` CLI is installed
    and logged in to a Claude subscription. Never bills — uses ``claude auth
    status``. Also warns (once) if API-key env vars are present and being
    stripped, so a near-miss on billing can't pass silently."""
    if shutil.which("claude") is None:
        raise SystemExit(
            "anthropic-cc judge needs the Claude Code CLI on PATH "
            "(https://claude.com/claude-code); `claude` was not found.")

    env, stripped = _subscription_env()
    if stripped:
        print(f"anthropic-cc: stripping {', '.join(stripped)} from the judge "
              f"environment so it bills your Claude subscription, not a "
              f"pay-as-you-go API account.", file=sys.stderr)

    try:
        proc = subprocess.run(["claude", "auth", "status"], env=env,
                              capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001 — surface any spawn/timeout failure
        raise SystemExit(f"could not run `claude auth status`: {e}")
    try:
        status = json.loads(proc.stdout)
    except json.JSONDecodeError:
        status = {}
    if proc.returncode != 0 or not status.get("loggedIn"):
        raise SystemExit(
            "anthropic-cc judge requires a logged-in Claude subscription; run "
            "`claude` (or `claude auth login`) first "
            f"(`claude auth status` reported loggedIn={status.get('loggedIn')}).")


def _default_runner(argv: list[str], *, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(argv, env=env, capture_output=True, text=True,
                          timeout=_CALL_TIMEOUT_SECONDS)


def judge_summary_cc(model: str, source_text: str, summary: dict, *,
                     runner=None) -> dict:
    """Grade one summary via ``claude -p`` with structured output. ``runner`` is
    injectable for tests — a callable ``(argv, *, env) -> CompletedProcess`` — so
    tests never spawn the real CLI. Any failure (spawn, non-zero exit, unparseable
    output, retry-exhausted, schema mismatch) becomes an ``ok: False`` record,
    matching the OpenAI judge's failure shape — never a crash, never silent spend."""
    from pydantic import ValidationError

    from bartleby.benchmark.judging import (
        JUDGE_INSTRUCTIONS, JudgeScore, RUBRIC_AXES, _user_prompt)

    wall_start = time.perf_counter()

    def fail(msg: str) -> dict:
        return {"ok": False, "error": msg,
                "wall_seconds": time.perf_counter() - wall_start}

    argv = [
        "claude", "-p", _user_prompt(source_text, summary),
        "--output-format", "json",
        "--json-schema", json.dumps(_judge_schema()),
        "--model", resolve_model(model),
        "--append-system-prompt", JUDGE_INSTRUCTIONS,
        "--tools", "",  # a judge is a pure completion — no web_search etc.
    ]
    env, _ = _subscription_env()
    run = runner or _default_runner
    try:
        proc = run(argv, env=env)
    except FileNotFoundError:
        return fail("`claude` CLI not found on PATH")
    except Exception as e:  # noqa: BLE001 — timeouts etc. become failure records
        return fail(f"{type(e).__name__}: {e}")

    if proc.returncode != 0:
        return fail(f"claude exited {proc.returncode}: "
                    f"{(proc.stderr or '').strip()[:300]}")
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return fail(f"unparseable claude output: {e}")
    if envelope.get("is_error") or envelope.get("subtype") != "success":
        return fail("claude judge did not succeed "
                    f"(subtype={envelope.get('subtype')!r})")
    payload = envelope.get("structured_output")
    if payload is None:
        return fail("claude returned no structured_output")
    try:
        parsed = JudgeScore.model_validate(payload)
    except ValidationError as e:
        return fail(f"judge output failed schema validation: {e}")

    scores = parsed.model_dump()
    scores["mean"] = sum(scores[a] for a in RUBRIC_AXES) / len(RUBRIC_AXES)
    return {"ok": True, "scores": scores,
            "wall_seconds": time.perf_counter() - wall_start}
