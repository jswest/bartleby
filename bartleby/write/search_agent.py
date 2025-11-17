"""Search Agent - Focused searcher with 5-search limit per invocation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.errors import GraphRecursionError
from loguru import logger

from bartleby.lib.consts import (
    DEFAULT_AGENT_CONTEXT_TOKENS,
    DEFAULT_MAX_SEARCH_OPERATIONS,
)
from bartleby.write.findings_schema import SearchFindings, render_findings_to_markdown
from bartleby.write.token_counter import TokenCounterCallback
from bartleby.write.tools import get_tools


@dataclass
class SearchBudget:
    """Tracks tool usage and produces warnings when the limit approaches."""

    max_calls: int
    calls_made: int = 0
    exhausted: bool = False

    def before_tool_call(self, tool_name: str) -> Any | None:
        if self.calls_made >= self.max_calls:
            self.exhausted = True
            return {
                "error": "SEARCH_LIMIT_REACHED",
                "message": (
                    f"You already used all {self.max_calls} allowed tool calls for this task. "
                    "Stop searching and summarize your findings."
                ),
                "tool_attempt": tool_name,
            }
        self.calls_made += 1
        return None

    def warning_messages(self) -> list[str]:
        remaining = self.max_calls - self.calls_made
        if self.exhausted:
            return [
                "🚨 SEARCH BUDGET EXHAUSTED: The Search Agent attempted another tool call "
                f"after using {self.calls_made}/{self.max_calls} searches. You must summarize now."
            ]
        if self.calls_made >= self.max_calls:
            return [
                f"🚨 SEARCH BUDGET EXHAUSTED: You have used all {self.max_calls} searches. "
                "You MUST summarize your findings NOW. No more tool calls are available."
            ]
        if remaining == 1:
            return [
                f"🚨 FINAL SEARCH: You have used {self.calls_made}/{self.max_calls} searches. "
                "Only 1 search remaining. Make it count, then summarize."
            ]
        if remaining == 2:
            return [
                f"⏰ WARNING: You have used {self.calls_made}/{self.max_calls} searches. "
                "Only 2 searches remaining. Plan carefully."
            ]
        return []

    def tools_available(self) -> list[str] | None:
        if self.exhausted or self.calls_made >= self.max_calls:
            return []
        return None


def _load_search_agent_prompt() -> str:
    """Load the search agent system prompt."""
    prompt_path = Path(__file__).parent / "prompts" / "search_agent.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()


def build_search_limit_middleware(
    max_calls: int = DEFAULT_MAX_SEARCH_OPERATIONS,
    tracker: SearchBudget | None = None,
):
    """Create middleware that enforces tool call limits."""

    def _count_tool_messages(messages) -> int:
        return sum(
            1
            for msg in messages
            if hasattr(msg, "type") and msg.type == "tool"
        )

    @before_model(name="search_limit")
    def _limit_searches(state, runtime):
        tool_usage = tracker.calls_made if tracker else _count_tool_messages(state["messages"])

        warnings = tracker.warning_messages() if tracker else []
        if not tracker:
            temp_tracker = SearchBudget(max_calls, calls_made=tool_usage)
            warnings = temp_tracker.warning_messages()

        trimmed = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=DEFAULT_AGENT_CONTEXT_TOKENS,
            start_on="human",
            end_on=("human", "tool"),
        )

        if warnings:
            constraint_msg = HumanMessage(
                content="[SYSTEM ALERT]\n\n" + "\n\n".join(warnings)
            )
            trimmed = list(trimmed) + [constraint_msg]

        response = {"llm_input_messages": trimmed}
        tools_available = tracker.tools_available() if tracker else None
        if tools_available is not None:
            response["tools"] = tools_available

        return response

    return _limit_searches


def run_search_agent(
    task: str,
    details: str,
    llm: BaseLanguageModel,
    db_path: Path,
    findings_dir: Path,
    todos_path: Path,
    embedding_model,
    run_uuid: str,
    sequence: int,
    token_counter: TokenCounterCallback | None = None,
    activity_logger=None,
    display_callback=None,
    max_search_operations: int = DEFAULT_MAX_SEARCH_OPERATIONS,
) -> Dict[str, Any]:
    """
    Run the search agent for a single task.

    Returns:
        Dictionary with {"summary": str, "findings_file": str}
    """
    allowed_tools = {
        "search_documents_fts",
        "search_documents_semantic",
        "get_chunk_window",
        "get_full_document",
    }
    budget_tracker = SearchBudget(max_search_operations)
    tools = get_tools(
        db_path=db_path,
        todos_path=todos_path,
        embedding_model=embedding_model,
        allowed_tools=allowed_tools,
        before_hook=budget_tracker.before_tool_call,
    )

    system_prompt = _load_search_agent_prompt()
    detail_block = f"Details: {details}\n\n" if details else ""
    task_prompt = (
        f"Task: {task}\n\n"
        f"{detail_block}"
        f"You have {max_search_operations} tool calls to gather evidence for this task. Make them count."
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            build_search_limit_middleware(
                max_search_operations, tracker=budget_tracker
            )
        ],
    )

    config = {"recursion_limit": max_search_operations * 4 + 10}
    if token_counter:
        config["callbacks"] = [token_counter]

    final_response = _stream_agent(
        agent=agent,
        task_prompt=task_prompt,
        config=config,
        budget_tracker=budget_tracker,
        token_counter=token_counter,
        activity_logger=activity_logger,
        display_callback=display_callback,
        max_search_operations=max_search_operations,
    )

    summary = final_response or "Search Agent completed but returned no summary."
    findings_filename = _write_findings(
        findings_dir=findings_dir,
        run_uuid=run_uuid,
        sequence=sequence,
        task=task,
        summary=summary,
    )

    return {"summary": summary, "findings_file": findings_filename}


def _stream_agent(
    agent,
    task_prompt: str,
    config: Dict[str, Any],
    budget_tracker: SearchBudget,
    token_counter: TokenCounterCallback | None,
    activity_logger,
    display_callback,
    max_search_operations: int,
) -> str:
    final_response = ""
    original_agent_name = None
    if activity_logger:
        original_agent_name = activity_logger.agent_name
        activity_logger.agent_name = "Search Agent"
        activity_logger.search_count = 0
        activity_logger.max_searches = max_search_operations

    try:
        for event in agent.stream(
            {"messages": [("user", task_prompt)]},
            stream_mode="values",
            config=config,
        ):
            if "messages" not in event or not event["messages"]:
                continue
            last_message = event["messages"][-1]
            final_response = _handle_agent_message(
                last_message,
                final_response,
                activity_logger,
                token_counter,
                budget_tracker,
                display_callback,
                max_search_operations,
            )
    except GraphRecursionError:
        logger.error(
            "Search Agent hit the recursion limit before finishing its task. "
            "Increase the search budget or recursion limit if this persists."
        )
        final_response = (
            "Search Agent stopped after reaching its internal recursion limit. "
            "Try refining the task or increasing the search budget."
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.error(f"Search Agent error: {exc}")
        final_response = f"Search Agent encountered an error: {exc}"
    finally:
        if activity_logger and original_agent_name is not None:
            activity_logger.agent_name = original_agent_name
            activity_logger.search_count = 0
            activity_logger.max_searches = 0

    return final_response


def _handle_agent_message(
    last_message,
    current_response: str,
    activity_logger,
    token_counter: TokenCounterCallback | None,
    budget_tracker: SearchBudget,
    display_callback,
    max_search_operations: int,
) -> str:
    if hasattr(last_message, "type") and last_message.type == "ai":
        if activity_logger:
            activity_logger.on_ai_message(last_message)
        if getattr(last_message, "content", None):
            current_response = last_message.content

    tool_calls = getattr(last_message, "tool_calls", None)
    if activity_logger and tool_calls:
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_call_id = tool_call.get("id", "")
            tool_args = tool_call.get("args", {})
            activity_logger.on_tool_call(tool_name, tool_call_id, tool_args)

    if hasattr(last_message, "type") and last_message.type == "tool":
        if activity_logger:
            tool_call_id = getattr(last_message, "tool_call_id", None)
            if tool_call_id and token_counter:
                activity_logger.on_tool_result(
                    tool_call_id, last_message.content, token_counter
                )
                activity_logger.search_count = min(
                    budget_tracker.calls_made, max_search_operations
                )
                if display_callback:
                    display_callback()

    return current_response


def _write_findings(
    findings_dir: Path,
    run_uuid: str,
    sequence: int,
    task: str,
    summary: str,
) -> str:
    findings = SearchFindings(
        task=task,
        searches_performed=[],
        key_findings=[summary],
        documents_cited=[],
        summary=summary,
    )
    findings_md = render_findings_to_markdown(findings)
    findings_filename = f"{run_uuid}-{sequence:02d}.md"
    findings_file = findings_dir / findings_filename
    findings_file.write_text(findings_md, encoding="utf-8")
    return findings_filename
