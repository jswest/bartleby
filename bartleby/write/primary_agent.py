"""Primary Agent - Strategic planner and report writer."""

from pathlib import Path
from typing import Iterator, Dict, Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_model_call
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.tools import tool
from loguru import logger

from bartleby.lib.consts import (
    DEFAULT_AGENT_CONTEXT_TOKENS,
    DEFAULT_MAX_SEARCH_OPERATIONS,
    DEFAULT_MAX_TODO_ROUNDS,
    DEFAULT_MAX_TOTAL_ROUNDS,
)
from bartleby.write.token_counter import TokenCounterCallback
from bartleby.write.tools import get_tools


# Primary Agent constraints (defaults, overridable via config)
MAX_TODO_ROUNDS = DEFAULT_MAX_TODO_ROUNDS
MAX_TOTAL_ROUNDS = DEFAULT_MAX_TOTAL_ROUNDS


def _load_primary_agent_prompt() -> str:
    """Load the primary agent system prompt."""
    prompt_path = Path(__file__).parent / "prompts" / "primary_agent.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()


def build_primary_agent_tools(
    llm: BaseLanguageModel,
    db_path: Path,
    findings_dir: Path,
    todos_path: Path,
    embedding_model,
    run_uuid: str,
    token_counter: TokenCounterCallback | None = None,
    streaming_logger=None,
    display_callback=None,
    max_search_operations: int = DEFAULT_MAX_SEARCH_OPERATIONS,
) -> list:
    """
    Build tools for the Primary Agent.

    The Primary Agent has:
    - manage_todo_tool
    - delegate_search (invokes Search Agent)
    - read_findings (synthesis phase only)
    """
    allowed_tools = {"manage_todo_tool", "delegate_search", "read_findings"}

    return get_tools(
        db_path=db_path,
        findings_dir=findings_dir,
        todos_path=todos_path,
        embedding_model=embedding_model,
        run_uuid=run_uuid,
        allowed_tools=allowed_tools,
        llm=llm,
        token_counter=token_counter,
        logger=streaming_logger,
        display_callback=display_callback,
        max_search_operations=max_search_operations,
    )


def build_round_limit_middleware(
    max_todo_rounds: int,
    max_total_rounds: int,
    token_counter: TokenCounterCallback | None = None,
    todo_status_provider: Callable[[], list[Dict[str, str]]] | None = None,
    round_provider: Callable[[], int] | None = None,
    delegate_tool=None,
    read_tool=None,
):
    """
    Create middleware that enforces round-based and token budget constraints.

    - Rounds 1-{max_todo_rounds}: Can use all tools (manage_todos, delegate_search)
    - Rounds {max_todo_rounds+1}-{max_total_rounds-1}: Synthesis mode (read_findings, manage_todo_tool)
    - Round ≥ {max_total_rounds}: All tools disabled; agent must immediately produce the report
    - Token budget warnings at 75% and 90%
    """

    @before_model(name="round_limits")
    def _enforce_round_limits(state, runtime):
        if round_provider:
            current_round = max(1, round_provider())
        else:
            # Estimate current round from message count
            message_count = len(state["messages"])
            current_round = max(1, message_count // 3)

        # Build constraint messages
        warnings = []
        tools_override = None
        final_round = current_round >= max_total_rounds

        incomplete_todos: list[str] = []
        if todo_status_provider:
            try:
                todos = todo_status_provider() or []
                incomplete_todos = [
                    todo.get("task", "")
                    for todo in todos
                    if todo.get("status") != "complete"
                ]
            except Exception:  # pragma: no cover - defensive
                incomplete_todos = []

        # Check token budget first (higher priority)
        if token_counter and token_counter.token_budget:
            token_ratio = token_counter.total_tokens / token_counter.token_budget

            if token_ratio >= 0.9:
                warnings.append(
                    f"🚨 TOKEN BUDGET CRITICAL ({token_counter.total_tokens}/{token_counter.token_budget} tokens, {token_ratio:.0%}): "
                    "You MUST deliver your final report in the next 2-3 rounds. "
                    "Stop ALL delegation and searches. Write your report NOW!"
                )
            elif token_ratio >= 0.75:
                warnings.append(
                    f"⚠️ TOKEN BUDGET WARNING ({token_counter.total_tokens}/{token_counter.token_budget} tokens, {token_ratio:.0%}): "
                    "Approaching token limit. Complete current work and prepare to write final report. "
                    "Avoid delegate_search. Prioritize synthesis over additional research."
                )

        # Check round constraints
        if final_round:
            warnings.append(
                f"🚨 FINAL ROUND ({current_round}/{max_total_rounds}): "
                "You have reached the overall round limit. "
                "Stop using tools and deliver your final research report immediately."
            )
            if incomplete_todos:
                warnings.append(
                    "Outstanding todos: " + "; ".join(incomplete_todos[:3])
                    + ("" if len(incomplete_todos) <= 3 else " …")
                )
            tools_override = []
        elif current_round > max_todo_rounds:
            warnings.append(
                f"⚠️ SYNTHESIS MODE (Round {current_round}/{max_total_rounds}): "
                f"You can no longer add new todos. Work with existing todos or write your final report. "
                f"You have {max_total_rounds - current_round} rounds remaining."
            )
            if incomplete_todos:
                warnings.append(
                    "Finish these todos immediately: " + "; ".join(incomplete_todos[:3])
                    + ("" if len(incomplete_todos) <= 3 else " …")
                )
            tools_override = [read_tool] if read_tool else []
        elif current_round > max_todo_rounds - 2:
            warnings.append(
                f"⏰ WARNING (Round {current_round}/{max_total_rounds}): "
                f"You have {max_todo_rounds - current_round} rounds left to add todos. "
                "After that, you must finalize and write the report."
            )
            tools_override = [delegate_tool] if delegate_tool else tools_override
        else:
            tools_override = [delegate_tool] if delegate_tool else tools_override
        # Trim messages
        trimmed = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=DEFAULT_AGENT_CONTEXT_TOKENS,
            start_on="human",
            end_on=("human", "tool"),
        )

        # Add constraint messages if any
        if warnings:
            constraint_msg = HumanMessage(content=f"[SYSTEM ALERT]\n\n" + "\n\n".join(warnings))
            trimmed = list(trimmed) + [constraint_msg]

        response = {"llm_input_messages": trimmed}
        if tools_override is not None:
            response["tools"] = tools_override

        return response

    return _enforce_round_limits


def run_primary_agent(
    user_direction: str,
    llm: BaseLanguageModel,
    db_path: Path,
    findings_dir: Path,
    todos_path: Path,
    embedding_model,
    run_uuid: str,
    token_counter: TokenCounterCallback | None = None,
    max_todo_rounds: int = MAX_TODO_ROUNDS,
    max_total_rounds: int = MAX_TOTAL_ROUNDS,
    max_search_operations: int = DEFAULT_MAX_SEARCH_OPERATIONS,
    logger=None,
    display_callback=None,
) -> Iterator[Dict[str, Any]]:
    """
    Run the Primary Agent.

    Args:
        user_direction: User's research question/direction
        llm: Language model
        db_path: Path to document database
        findings_dir: Directory for findings files
        todos_path: Path to todos.json
        embedding_model: SentenceTransformer model
        run_uuid: UUID of current run
        token_counter: Optional token counter
        max_todo_rounds: Maximum rounds to allow adding todos (default: 10)
        max_total_rounds: Maximum total rounds before forcing report (default: 15)
        max_search_operations: Maximum number of tool calls per delegated Search Agent task
        logger: Optional StreamingLogger for tracking tool calls
        display_callback: Optional callback to update display

    Yields:
        Agent execution events
    """
    # Build all tools
    all_tools = build_primary_agent_tools(
        llm,
        db_path,
        findings_dir,
        todos_path,
        embedding_model,
        run_uuid,
        token_counter,
        logger,
        display_callback,
        max_search_operations=max_search_operations,
    )

    tool_lookup = {tool.name: tool for tool in all_tools}
    delegate_tool = tool_lookup.get("delegate_search")
    read_tool = tool_lookup.get("read_findings")
    agent_tools = [tool for tool in (delegate_tool, read_tool) if tool]

    # Load system prompt
    system_prompt = _load_primary_agent_prompt()

    # Add user direction and constraints to prompt
    full_prompt = f"""{system_prompt}

## Investigation Direction

{user_direction}

## Workflow Constraints

- Use `delegate_search` to create and execute todos. Provide one todo per call or a newline-separated list (max 3 per call). Do NOT call manage_todo_tool.
- Rounds 1-{max_todo_rounds}: Explore, delegate searches, and gather findings.
- Round {max_todo_rounds + 1}: Stop delegating. Session will end next round.
- Round {max_total_rounds}: Tools are disabled. You MUST deliver the complete report immediately.

## Your Task

1. Break the investigation into 2-4 focused research questions (todos).
2. Delegate each todo via `delegate_search` and review the returned findings.
3. Once all research is complete (or time is up), synthesize a comprehensive Markdown report.

Begin your investigation now.
"""

    # Load TodoList for middleware
    from bartleby.write.memory import TodoList
    todo_list = TodoList(str(todos_path))

    # Create agent with round limit middleware
    agent = create_agent(
        model=llm,
        tools=agent_tools,
        system_prompt=full_prompt,
        middleware=[
            build_round_limit_middleware(
                max_todo_rounds,
                max_total_rounds,
                token_counter,
                todo_status_provider=todo_list.get_all_todos,
                round_provider=(lambda: logger.get_round()) if logger else None,
                delegate_tool=delegate_tool,
                read_tool=read_tool,
            )
        ],
    )

    # Calculate max recursions: each round ~3 recursions
    max_recursions = max_total_rounds * 3

    config = {}
    if token_counter:
        config["callbacks"] = [token_counter]

    # Stream agent execution
    final_response = ""

    for event in agent.stream(
        {"messages": [("user", f"Begin your investigation: {user_direction}")]},
        stream_mode="values",
        config={**config, "recursion_limit": max_recursions},
    ):
        # Yield chunk for main.py to handle display
        yield {"chunk": event}

        if "messages" in event and len(event["messages"]) > 0:
            last_message = event["messages"][-1]

            # Capture final AI response
            if hasattr(last_message, 'content') and last_message.content and hasattr(last_message, 'type'):
                if last_message.type == 'ai':
                    final_response = last_message.content

    # Yield final result
    yield {
        "agent": {
            "output": final_response,
        }
    }
