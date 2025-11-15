"""ReAct agent for document research and report generation."""

from pathlib import Path
from typing import Iterator, Dict, Any

from langchain.agents import create_agent
from langchain.agents.middleware import before_model, wrap_model_call
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately,
)
from sentence_transformers import SentenceTransformer

from bartleby.lib.consts import DEFAULT_MAX_RECURSIONS, DEFAULT_AGENT_CONTEXT_TOKENS
from bartleby.lib.console import send
from bartleby.write.tools import DocumentSearchTools
from bartleby.write.token_counter import TokenCounterCallback

with open(Path(__file__).parent / "prompts" / "system.md", "r") as in_file:
    SYSTEM_PROMPT = in_file.read()


def build_trim_middleware(max_tokens: int):
    """Create a before_model middleware that trims conversation history."""

    @before_model(name="trim_messages")
    def _trim_history(state, runtime):  # pragma: no cover - LangChain runtime hook
        trimmed_messages = trim_messages(
            state["messages"],
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=max_tokens,
            start_on="human",
            end_on=("human", "tool"),
        )
        return {"llm_input_messages": trimmed_messages}

    return _trim_history


def _is_json_parse_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in (
        "error parsing tool call",
        "jsondecodeerror",
        "unexpected end of json",
        "invalid character",
        "unexpected token",
        "unterminated string",
    ))


def build_model_retry_middleware(max_attempts: int):
    """Retry malformed tool call generations without restarting the agent."""
    if max_attempts <= 0:
        return None

    @wrap_model_call(name="retry_malformed_tool_json")
    def _retry_model_call(request, handler):  # pragma: no cover - runtime hook
        attempts_left = max_attempts
        while True:
            try:
                return handler(request)
            except Exception as exc:
                if attempts_left <= 0 or not _is_json_parse_error(exc):
                    raise
                attempts_left -= 1
                send("Model returned malformed tool JSON. Retrying that call...", "WARN")

    return _retry_model_call


def create_agent_graph(
    llm: BaseLanguageModel,
    tools: list,
    user_direction: str,
    context_token_limit: int = DEFAULT_AGENT_CONTEXT_TOKENS,
    model_retry_attempts: int = 0,
) -> Any:
    """
    Create a ReAct agent for document research.

    Args:
        llm: Language model for the agent
        tools: List of LangChain tools
        user_direction: User's investigation direction/question

    Returns:
        Compiled agent
    """
    # Build system prompt with user direction
    system_prompt = SYSTEM_PROMPT + f"\n\nUser's investigation direction: {user_direction}"

    agent_kwargs = {
        "model": llm,
        "tools": tools,
        "system_prompt": system_prompt,
    }
    middleware = []
    if context_token_limit and context_token_limit > 0:
        middleware.append(build_trim_middleware(context_token_limit))

    retry_middleware = build_model_retry_middleware(model_retry_attempts)
    if retry_middleware:
        middleware.append(retry_middleware)

    if middleware:
        agent_kwargs["middleware"] = middleware

    agent = create_agent(**agent_kwargs)

    return agent


def run_agent(
    user_direction: str,
    llm: BaseLanguageModel,
    token_counter: TokenCounterCallback,
    search_tools: DocumentSearchTools,
    max_recursions: int = DEFAULT_MAX_RECURSIONS,
    context_token_limit: int = DEFAULT_AGENT_CONTEXT_TOKENS,
    model_retry_attempts: int = 0,
) -> Iterator[Dict[str, Any]]:
    """
    Run the research agent with streaming output.

    Args:
        db_path: Path to the database
        user_direction: User's investigation direction
        llm: Language model
        embedding_model: Embedding model for search
        token_counter: Token usage tracker
        search_tools: DocumentSearchTools instance
        max_recursions: Maximum graph recursions (super-steps)

    Yields:
        Agent execution events as dictionaries
    """
    # Get tools from search_tools instance
    tools = search_tools.get_tools()

    # Create agent
    agent = create_agent_graph(
        llm=llm,
        tools=tools,
        user_direction=user_direction,
        context_token_limit=context_token_limit,
        model_retry_attempts=model_retry_attempts,
    )

    # Stream agent execution (yield chunks for main.py to process)
    final_response = ""

    stream_modes: list[str] = ["values", "debug"]

    for event in agent.stream(
        {"messages": [("user", f"Begin your investigation: {user_direction}")]},
        stream_mode=stream_modes,
        config={"callbacks": [token_counter], "recursion_limit": max_recursions},
    ):
        mode = None
        chunk = event
        if isinstance(event, tuple) and len(event) == 2:
            mode, payload = event
            if mode == "debug":
                yield {"debug": payload}
                continue
            if mode == "values":
                chunk = payload
            else:
                chunk = payload

        # Yield chunk for main.py to handle display
        yield {"chunk": chunk}

        if "messages" in chunk and len(chunk["messages"]) > 0:
            last_message = chunk["messages"][-1]

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
