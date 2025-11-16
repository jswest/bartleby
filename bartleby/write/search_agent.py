"""Search Agent - Focused searcher with 5-search limit per invocation."""

from pathlib import Path
from typing import Iterator, Dict, Any

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain.agents.middleware import before_model
from loguru import logger

try:  # pragma: no cover - fallback if langgraph isn't available
    from langgraph.errors import GraphRecursionError
except ImportError:  # pragma: no cover
    class GraphRecursionError(RuntimeError):
        """Placeholder for environments without LangGraph installed."""
        pass

from bartleby.lib.consts import DEFAULT_AGENT_CONTEXT_TOKENS
from bartleby.write.tools import DocumentSearchTools
from bartleby.write.token_counter import TokenCounterCallback


# Maximum tool calls per search agent invocation
MAX_SEARCH_OPERATIONS = 5


class SearchBudgetTracker:
    """Tracks how many tool calls the Search Agent has made in this task."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls_made = 0
        self.exhausted = False

    def before_tool_call(self, tool_name: str) -> Any | None:
        """
        Hook executed before any search tool is invoked.

        Returns a short-circuit payload if the budget is exhausted so that the
        actual expensive tool call never fires.
        """
        if self.calls_made >= self.max_calls:
            self.exhausted = True
            logger.warning(
                f"Search Agent budget exhausted: attempted '{tool_name}' after "
                f"{self.calls_made}/{self.max_calls} tool calls."
            )
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


def _load_search_agent_prompt() -> str:
    """Load the search agent system prompt."""
    prompt_path = Path(__file__).parent / "prompts" / "search_agent.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()

    # Fallback prompt if file doesn't exist yet
    return """You are a focused research assistant. You have been delegated a specific research task.

Your job:
1. Execute searches to answer the task question
2. Read relevant passages using get_chunk_window
3. Write your findings to the scratchpad with citations
4. You have a MAXIMUM of 5 tool calls per task

Guidelines:
- Be efficient: You only get 5 tool calls, so make them count
- Always write findings to scratchpad immediately after reading
- Include document IDs, chunk IDs, and specific quotes
- Prefer get_chunk_window over get_full_document
- Use semantic search for concepts, FTS for exact terms

After your 5 tool calls, summarize what you learned in your final response."""


def build_search_limit_middleware(
    max_calls: int = MAX_SEARCH_OPERATIONS,
    tracker: SearchBudgetTracker | None = None,
):
    """Create middleware that enforces tool call limits."""

    calls_made = {"count": 0}

    @before_model(name="search_limit")
    def _limit_searches(state, runtime):
        # Count tool usage
        if tracker:
            tool_count = tracker.calls_made
        else:
            tool_count = sum(
                1 for msg in state["messages"]
                if hasattr(msg, 'type') and msg.type == 'tool'
            )

        calls_made["count"] = tool_count

        # Build warning messages based on usage
        warnings = []
        remaining = max_calls - tool_count

        tools_available = None

        if tracker and tracker.exhausted:
            warnings.append(
                "🚨 SEARCH BUDGET EXHAUSTED: The Search Agent attempted to call another tool "
                f"after using {tracker.calls_made}/{tracker.max_calls} searches. You must summarize now."
            )
            tools_available = []
        elif tool_count >= max_calls:
            # Budget exhausted - remove tools and force summary
            logger.info(f"Search Agent: Reached {max_calls} tool call limit. Forcing summary.")
            warnings.append(
                f"🚨 SEARCH BUDGET EXHAUSTED: You have used all {max_calls} searches. "
                "You MUST summarize your findings NOW. No more tool calls are available."
            )
            tools_available = []

        elif tool_count == max_calls - 1:
            # One search remaining
            warnings.append(
                f"🚨 FINAL SEARCH: You have used {tool_count}/{max_calls} searches. "
                f"Only {remaining} search remaining. Make it count, then summarize."
            )
        elif tool_count >= max_calls - 2:
            # Two searches remaining
            warnings.append(
                f"⏰ WARNING: You have used {tool_count}/{max_calls} searches. "
                f"Only {remaining} searches remaining. Plan carefully."
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

        # Add warning messages if any
        if warnings:
            constraint_msg = HumanMessage(content=f"[SYSTEM ALERT]\n\n" + "\n\n".join(warnings))
            trimmed = list(trimmed) + [constraint_msg]

        response = {"llm_input_messages": trimmed}
        if tools_available is not None:
            response["tools"] = tools_available

        return response

    return _limit_searches


def run_search_agent(
    task: str,
    details: str,
    llm: BaseLanguageModel,
    search_tools: DocumentSearchTools,
    token_counter: TokenCounterCallback | None = None,
    activity_logger=None,
    display_callback=None,
) -> str:
    """
    Run the search agent for a single task.

    Args:
        task: The task description from the todo item
        details: Additional details about what to search for
        llm: Language model to use
        search_tools: DocumentSearchTools instance
        token_counter: Optional token counter
        activity_logger: Optional StreamingLogger for tracking tool calls
        display_callback: Optional callback to update display after each tool

    Returns:
        Summary of findings
    """
    # Get search tools (all search tools + scratchpad)
    allowed_tools = {
        "search_documents_fts",
        "search_documents_semantic",
        "get_chunk_window",
        "get_full_document",
        "append_to_scratchpad_tool",
        "read_scratchpad_tool",
    }
    budget_tracker = SearchBudgetTracker(MAX_SEARCH_OPERATIONS)
    tools = search_tools.get_tools(
        allowed_tools,
        before_tool_call=budget_tracker.before_tool_call,
    )

    # Build system prompt
    system_prompt = _load_search_agent_prompt()
    task_prompt = f"""
Task: {task}

{f'Details: {details}' if details else ''}

You have {MAX_SEARCH_OPERATIONS} tool calls to gather evidence for this task. Make them count.
"""

    # Create agent with search limit middleware
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[build_search_limit_middleware(MAX_SEARCH_OPERATIONS, tracker=budget_tracker)],
    )

    # Run agent with a strict (but generous) recursion limit.
    # Each tool call tends to span ~4 recursions once scratchpad operations and budget
    # warnings are included, so keep a healthy buffer to avoid premature Graph errors.
    max_recursions = MAX_SEARCH_OPERATIONS * 4 + 10

    config = {}
    if token_counter:
        config["callbacks"] = [token_counter]

    final_response = ""

    # Temporarily switch logger context to Search Agent
    original_agent_name = None
    if activity_logger:
        original_agent_name = activity_logger.agent_name
        activity_logger.agent_name = "Search Agent"
        activity_logger.search_count = 0
        activity_logger.max_searches = MAX_SEARCH_OPERATIONS

    try:
        for event in agent.stream(
            {"messages": [("user", task_prompt)]},
            stream_mode="values",
            config={**config, "recursion_limit": max_recursions},
        ):
            if "messages" in event and len(event["messages"]) > 0:
                last_message = event["messages"][-1]

                # Log AI messages (for recursion tracking)
                if hasattr(last_message, 'type') and last_message.type == 'ai':
                    if activity_logger:
                        activity_logger.on_ai_message(last_message)

                    # Capture final response
                    if hasattr(last_message, 'content') and last_message.content:
                        final_response = last_message.content

                # Log tool calls
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    if activity_logger:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get('name', '')
                            tool_call_id = tool_call.get('id', '')
                            tool_args = tool_call.get('args', {})
                            activity_logger.on_tool_call(tool_name, tool_call_id, tool_args)

                # Log tool results
                if hasattr(last_message, 'type') and last_message.type == 'tool':
                    if activity_logger:
                        tool_call_id = getattr(last_message, 'tool_call_id', None)
                        if tool_call_id and token_counter:
                            activity_logger.on_tool_result(
                                tool_call_id,
                                last_message.content,
                                token_counter
                            )
                            # Increment search count (clamped to budget tracker)
                            activity_logger.search_count = min(
                                budget_tracker.calls_made,
                                MAX_SEARCH_OPERATIONS,
                            )
                            # Update display if callback provided
                            if display_callback:
                                display_callback()

    except GraphRecursionError:
        logger.error(
            "Search Agent hit the recursion limit before finishing its task. "
            "Increase MAX_SEARCH_OPERATIONS or recursion limit if this persists."
        )
        final_response = (
            "Search Agent stopped after reaching its internal recursion limit. "
            "Try refining the task or increasing the search budget."
        )
    except Exception as e:
        logger.error(f"Search Agent error: {e}")
        final_response = f"Search Agent encountered an error: {e}"
    finally:
        # Restore original agent context
        if activity_logger and original_agent_name is not None:
            activity_logger.agent_name = original_agent_name
            activity_logger.search_count = 0
            activity_logger.max_searches = 0

    return final_response or "Search Agent completed but returned no summary."
