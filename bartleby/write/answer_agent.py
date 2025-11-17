"""Answer Agent for interactive Q&A after report generation."""

from pathlib import Path
from typing import Dict, Any, Callable

from langchain.agents import create_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from bartleby.lib.consts import DEFAULT_MAX_SEARCH_OPERATIONS
from bartleby.write.logging import StreamingLogger
from bartleby.write.search_agent import run_search_agent
from bartleby.write.token_counter import TokenCounterCallback
from bartleby.write.tools.read_findings import create_read_findings_tool


ANSWER_AGENT_SYSTEM_PROMPT = """You are an Answer Agent helping users understand completed research.

REPORT:
{report_text}

RESEARCH FINDINGS:
{findings_text}

Your job: Answer questions about this research.

Strategy:
1. FIRST: Try to answer from the report and findings above.
2. Write the entire response in well-structured Markdown (headings, bullet lists, tables when helpful).
3. You MUST cite specific passages from the report/findings for every answer.
4. If you cannot cite the report/findings for a claim, you MUST call delegate_answer_search before replying.
5. Explain what new information came from any delegated research.

If the user explicitly requests additional research, you MUST call delegate_answer_search before finalizing the answer.

When answering:
- Reference specific sections from the report or findings (with citation IDs)
- Use Markdown formatting for structure and readability
- If you use delegation, explain what new information you found
- Be conversational and helpful"""


def _estimate_tokens(text: str) -> int:
    """Estimate token counts (roughly 1 token per 5 characters)."""
    if not text:
        return 0
    return max(1, len(text) // 5)


def _format_token_count(count: int) -> str:
    """Format counts, rounding to the nearest 100 past 1k and showing k-suffix."""
    if count >= 1000:
        rounded = int(round(count / 100.0) * 100)
        return f"+{rounded / 1000:.1f}k"
    return f"+{count}"


def _extract_message_text(message: Any) -> str:
    """Extract human-readable text from LangChain message content."""
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            text = ""

            if isinstance(part, str):
                text = part
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    text = part["text"]
                elif isinstance(part.get("content"), str):
                    text = part["content"]
                elif isinstance(part.get("data"), str):
                    text = part["data"]
            elif hasattr(part, "text") and isinstance(part.text, str):
                text = part.text
            elif hasattr(part, "content") and isinstance(part.content, str):
                text = part.content
            else:
                text = str(part)

            if text:
                parts.append(text.strip())

        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def initialize_answer_agent(
    report_text: str,
    findings_dir: Path,
    run_uuid: str,
    logger: StreamingLogger,
) -> dict:
    """
    Load report and all findings into context for Q&A session.

    Args:
        report_text: The generated report text
        findings_dir: Directory containing findings files
        run_uuid: UUID of the run
        logger: StreamingLogger instance to switch to Q&A mode

    Returns:
        Dict with report, findings, and context summary
    """
    # Switch logger to Q&A mode
    logger.switch_to_qa_mode()

    # Load all research findings
    findings_tool = create_read_findings_tool(findings_dir, run_uuid)
    all_findings = findings_tool.invoke({})

    report_tokens = _estimate_tokens(report_text)
    findings_tokens = _estimate_tokens(all_findings)
    context_summary = (
        f"Report: {_format_token_count(report_tokens)} tokens, "
        f"Findings: {_format_token_count(findings_tokens)} tokens"
    )

    return {
        "report": report_text,
        "findings": all_findings,
        "context_summary": context_summary
    }


def create_delegate_answer_search_tool(
    llm: BaseLanguageModel,
    db_path: Path,
    findings_dir: Path,
    embedding_model,
    sequence_counter: dict,
    token_counter: TokenCounterCallback,
    logger: StreamingLogger,
    display_callback: Callable,
    max_search_operations: int = DEFAULT_MAX_SEARCH_OPERATIONS,
) -> tuple[StructuredTool, Callable[[str, str], str]]:
    """
    Create tool for delegating Q&A research to Search Agent.

    Args:
        llm: Language model for Search Agent
        db_path: Path to document database
        findings_dir: Directory for findings files
        embedding_model: SentenceTransformer model
        sequence_counter: Dict tracking delegation sequence {"count": int}
        token_counter: Token counter for tracking usage
        logger: StreamingLogger for logging events
        display_callback: Callback to update Live display
        max_search_operations: Max tool calls per delegation

    Returns:
        Tuple of (StructuredTool, delegate_fn)
    """

    class DelegateAnswerSearchInput(BaseModel):
        question: str = Field(
            ...,
            description="Research question to delegate to Search Agent"
        )
        focus: str = Field(
            default="",
            description="Optional focus or context for the search"
        )

    def _delegate_answer_search(question: str, focus: str = "") -> str:
        """
        Delegate research question to Search Agent.

        Use this ONLY if the answer isn't in the report or findings.

        Args:
            question: The research question to investigate
            focus: Optional additional context or focus

        Returns:
            Summary of findings from Search Agent
        """
        sequence_counter["count"] += 1

        # Create answer findings subdirectory
        answer_findings_dir = findings_dir / "answers"
        answer_findings_dir.mkdir(exist_ok=True)

        result = run_search_agent(
            task=question,
            details=focus,
            llm=llm,
            db_path=db_path,
            findings_dir=answer_findings_dir,
            todos_path=None,  # Q&A doesn't use todos
            embedding_model=embedding_model,
            run_uuid="",  # Not used when findings_prefix is set
            findings_prefix="answer",  # Creates answer-01.md, answer-02.md, etc.
            sequence=sequence_counter["count"],
            token_counter=token_counter,  # Continues tracking
            activity_logger=logger,  # Continues logging to same log.json
            display_callback=display_callback,  # Updates same Live display
            max_search_operations=max_search_operations,
        )

        return result["summary"]

    tool = StructuredTool.from_function(
        func=_delegate_answer_search,
        name="delegate_answer_search",
        description=f"Delegate a research question to Search Agent (up to {max_search_operations} searches). Use ONLY if answer not in report/findings.",
        args_schema=DelegateAnswerSearchInput,
    )

    return tool, _delegate_answer_search


def run_answer_agent(
    question: str,
    context: dict,
    llm: BaseLanguageModel,
    db_path: Path,
    findings_dir: Path,
    embedding_model,
    token_counter: TokenCounterCallback,
    logger: StreamingLogger,
    display_callback: Callable,
    sequence_counter: dict,
    max_search_operations: int = DEFAULT_MAX_SEARCH_OPERATIONS,
    force_delegate: bool = False,
) -> Dict[str, Any]:
    """
    Answer a single question with strategic delegation.

    Args:
        question: User's question
        context: Context dict from initialize_answer_agent
        llm: Language model
        db_path: Path to document database
        findings_dir: Directory for findings files
        embedding_model: SentenceTransformer model
        token_counter: Token counter for tracking usage
        logger: StreamingLogger for logging events
        display_callback: Callback to update Live display
        sequence_counter: Dict tracking delegation sequence
        max_search_operations: Max tool calls per delegation

    Returns:
        Dict with {"answer": str, "delegated": bool}
    """
    # Build system prompt with report + findings
    system_prompt = ANSWER_AGENT_SYSTEM_PROMPT.format(
        report_text=context["report"],
        findings_text=context["findings"]
    )

    # Create delegation tool
    delegate_tool, delegate_fn = create_delegate_answer_search_tool(
        llm=llm,
        db_path=db_path,
        findings_dir=findings_dir,
        embedding_model=embedding_model,
        sequence_counter=sequence_counter,
        token_counter=token_counter,
        logger=logger,
        display_callback=display_callback,
        max_search_operations=max_search_operations,
    )
    tools = [delegate_tool]

    # Create agent
    force_instructions = ""
    if force_delegate:
        force_instructions = (
            "\n\nUSER REQUESTED FRESH RESEARCH:\n"
            "- You MUST call delegate_answer_search before you provide your final answer.\n"
            "- After delegating, incorporate the new findings into your response."
        )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt + force_instructions,
    )

    # Track if delegation occurred by checking sequence counter
    initial_count = sequence_counter["count"]

    # Execute agent
    stream_config = {}
    if token_counter:
        stream_config["callbacks"] = [token_counter]

    try:
        final_answer = ""
        for chunk in agent.stream(
            {"messages": [("user", question)]},
            stream_mode="values",
            config=stream_config if stream_config else None,
        ):
            if "messages" not in chunk or not chunk["messages"]:
                continue

            for msg in chunk["messages"]:
                text = _extract_message_text(msg)
                if text:
                    final_answer = text

        answer = final_answer or "I couldn't generate an answer."
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    # Check if delegation occurred
    delegated = sequence_counter["count"] > initial_count

    if force_delegate and not delegated:
        fallback_summary = delegate_fn(question=question, focus="")
        delegated = True
        answer = (
            f"{answer.strip()}\n\n[Additional research]\n{fallback_summary}"
            if answer.strip()
            else f"[Additional research]\n{fallback_summary}"
        )

    return {
        "answer": answer,
        "delegated": delegated
    }


def save_qa_session(qa_history: list, output_path: Path):
    """
    Save Q&A session to Markdown file.

    Args:
        qa_history: List of Q&A dicts with question, answer, delegated, sequence
        output_path: Path to save the Q&A session file
    """
    content = [
        "# Q&A Session\n\n",
        f"Questions answered: {len(qa_history)}\n\n",
        "---\n\n"
    ]

    for item in qa_history:
        content.append(f"## Question {item['sequence']}\n\n")
        content.append(f"**Q:** {item['question']}\n\n")
        content.append(f"**A:** {item['answer']}\n\n")

        if item.get('delegated'):
            content.append("*Note: Required additional research*\n\n")

        content.append("---\n\n")

    output_path.write_text("".join(content), encoding="utf-8")
