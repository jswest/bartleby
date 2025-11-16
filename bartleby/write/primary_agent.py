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

from bartleby.lib.consts import DEFAULT_AGENT_CONTEXT_TOKENS
from bartleby.write.tools import DocumentSearchTools
from bartleby.write.search_agent import run_search_agent
from bartleby.write.token_counter import TokenCounterCallback


# Primary Agent constraints
MAX_TODO_ROUNDS = 10  # Can add todos in rounds 1-10
MAX_TOTAL_ROUNDS = 15  # Must complete by round 15


def _load_primary_agent_prompt() -> str:
    """Load the primary agent system prompt."""
    prompt_path = Path(__file__).parent / "prompts" / "primary_agent.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()

    # Fallback prompt
    return """You are a strategic research coordinator and report writer.

You delegate research tasks to a Search Agent and synthesize findings into a final report.

Rounds 1-10: Create todos and delegate searches
Rounds 11-15: Finalize research and write the report

Tools:
- manage_todo_tool: Create, update, and list todos
- delegate_search: Send a task to the Search Agent (gets 5 searches per task)
- read_scratchpad_tool: Read accumulated evidence

Your final output must be a complete Markdown research report."""


def build_primary_agent_tools(
    llm: BaseLanguageModel,
    search_tools: DocumentSearchTools,
    token_counter: TokenCounterCallback | None = None,
    streaming_logger=None,
    display_callback=None,
) -> list:
    """
    Build tools for the Primary Agent.

    The Primary Agent has:
    - manage_todo_tool (from search_tools)
    - read_scratchpad_tool (from search_tools)
    - delegate_search (custom tool that invokes Search Agent)
    """

    todo_list = getattr(search_tools, "todo_list", None)

    def _mark_todo(task_description: str, status: str) -> str:
        """Update todo status if a matching todo exists."""
        if not todo_list:
            return ""
        try:
            result = todo_list.update_todo_status(task_description, status)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to update todo '{task_description}' -> {status}: {exc}")
            return ""
        if result.get("error"):
            return ""
        return result.get("todo", {}).get("task", task_description)

    @tool
    def delegate_search(task: str, details: str = "") -> str:
        """
        Delegate a research task to the Search Agent.

        The Search Agent will execute up to 5 searches to answer your question,
        write findings to the scratchpad, and return a summary.

        Args:
            task: The research question or task description
            details: Additional context or specific things to look for

        Returns:
            Summary of what the Search Agent found
        """
        activated_todo = _mark_todo(task, "active")

        # Run the search agent
        summary = run_search_agent(
            task=task,
            details=details,
            llm=llm,
            search_tools=search_tools,
            token_counter=token_counter,
            activity_logger=streaming_logger,
            display_callback=display_callback,
        )

        completed_todo = _mark_todo(task, "complete")
        follow_up_note = ""
        if activated_todo:
            follow_up_note += f"\n\n(🗂️ Marked todo '{activated_todo}' as active.)"
        if completed_todo:
            follow_up_note += f"\n\n(✅ Marked todo '{completed_todo}' complete.)"

        return summary + follow_up_note if follow_up_note else summary

    # Get todo and scratchpad tools from search_tools
    allowed_tools = {"manage_todo_tool", "read_scratchpad_tool"}
    base_tools = search_tools.get_tools(allowed_tools)

    # Add delegation tool
    return base_tools + [delegate_search]


def build_round_limit_middleware(
    max_todo_rounds: int,
    max_total_rounds: int,
    token_counter: TokenCounterCallback | None = None,
    todo_status_provider: Callable[[], list[Dict[str, str]]] | None = None,
    round_provider: Callable[[], int] | None = None,
):
    """
    Create middleware that enforces round-based and token budget constraints.

    - Rounds 1-{max_todo_rounds}: Can use all tools (manage_todos, delegate_search, read_scratchpad)
    - Rounds {max_todo_rounds+1}-{max_total_rounds}: Cannot add new todos via manage_todo_tool action='add'
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
                    "Stop ALL delegation and searches. Review scratchpad and write report NOW."
                )
            elif token_ratio >= 0.75:
                warnings.append(
                    f"⚠️ TOKEN BUDGET WARNING ({token_counter.total_tokens}/{token_counter.token_budget} tokens, {token_ratio:.0%}): "
                    "Approaching token limit. Complete current work and prepare to write final report. "
                    "Avoid delegate_search. Prioritize synthesis over additional research."
                )

        # Check round constraints
        if current_round > max_total_rounds:
            warnings.append(
                f"🚨 FINAL ROUND ({current_round}/{max_total_rounds}): "
                "You MUST deliver your complete research report NOW. "
                "No more searches or planning. Write the report based on your scratchpad."
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
        elif current_round > max_todo_rounds - 2:
            warnings.append(
                f"⏰ WARNING (Round {current_round}/{max_total_rounds}): "
                f"You have {max_todo_rounds - current_round} rounds left to add todos. "
                "After that, you must finalize and write the report."
            )

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
    search_tools: DocumentSearchTools,
    token_counter: TokenCounterCallback | None = None,
    max_todo_rounds: int = MAX_TODO_ROUNDS,
    max_total_rounds: int = MAX_TOTAL_ROUNDS,
    logger=None,
    display_callback=None,
) -> Iterator[Dict[str, Any]]:
    """
    Run the Primary Agent.

    Args:
        user_direction: User's research question/direction
        llm: Language model
        search_tools: DocumentSearchTools instance
        token_counter: Optional token counter
        max_todo_rounds: Maximum rounds to allow adding todos (default: 10)
        max_total_rounds: Maximum total rounds before forcing report (default: 15)
        logger: Optional StreamingLogger for tracking tool calls
        display_callback: Optional callback to update display

    Yields:
        Agent execution events
    """
    # Build tools
    tools = build_primary_agent_tools(llm, search_tools, token_counter, logger, display_callback)

    # Load system prompt
    system_prompt = _load_primary_agent_prompt()

    # Add user direction and constraints to prompt
    full_prompt = f"""{system_prompt}

## Investigation Direction

{user_direction}

## Budget Constraints

- **Rounds 1-{max_todo_rounds}**: You can create todos and delegate searches to the Search Agent
- **Rounds {max_todo_rounds + 1}-{max_total_rounds}**: You CANNOT add new todos. Finalize research and write the report.
- **After round {max_total_rounds}**: Session ends. You MUST deliver the complete report.

## Your Task

1. Break down the investigation into 2-4 focused research questions (todos)
2. Delegate each todo to the Search Agent
3. Review the Search Agent's findings in the scratchpad
4. Synthesize everything into a comprehensive Markdown report

Begin your investigation now.
"""

    # Create agent with round limit middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=full_prompt,
        middleware=[
            build_round_limit_middleware(
                max_todo_rounds,
                max_total_rounds,
                token_counter,
                todo_status_provider=search_tools.todo_list.get_all_todos,
                round_provider=(lambda: logger.get_round()) if logger else None,
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
