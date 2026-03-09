"""Session parsing for bartleby book."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Note:
    """A note or finding from memory."""
    filename: str
    title: str
    content: str
    timestamp: datetime | None
    is_explicit_note: bool = True


@dataclass
class ToolCall:
    """A single tool call from the logs."""
    timestamp: datetime
    tool_name: str
    inputs: dict
    outputs_summary: dict | str
    input_tokens: int
    output_tokens: int


@dataclass
class Session:
    """A research session."""
    session_name: str
    session_dir: Path | None
    start_time: datetime | None
    end_time: datetime | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)

    @property
    def total_input_tokens(self) -> int:
        return sum(tc.input_tokens for tc in self.tool_calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(tc.output_tokens for tc in self.tool_calls)

    @property
    def duration_seconds(self) -> float | None:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def _parse_log_jsonl(log_path: Path) -> list[ToolCall]:
    """Parse a log.jsonl file into ToolCall objects."""
    if not log_path.exists():
        return []

    tool_calls = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                tc = ToolCall(
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                    tool_name=entry.get("tool_name", "unknown"),
                    inputs=entry.get("inputs", {}),
                    outputs_summary=entry.get("outputs_summary", ""),
                    input_tokens=entry.get("tokens", {}).get("input", 0),
                    output_tokens=entry.get("tokens", {}).get("output", 0),
                )
                tool_calls.append(tc)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return tool_calls


def _parse_note_file(filepath: Path) -> Note | None:
    """Parse a note markdown file from memory/."""
    content = filepath.read_text(encoding="utf-8")

    # Extract title from first # heading
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else filepath.stem

    try:
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
    except OSError:
        mtime = None

    return Note(
        filename=filepath.name,
        title=title,
        content=content,
        timestamp=mtime,
    )


def parse_sessions(book_dir: Path, memory_dir: Path | None = None) -> list[Session]:
    """Parse all sessions from a book directory.

    Reads from book/sessions/*/log.jsonl (new structure).
    Falls back to book/log.json + book/findings/ (legacy structure).

    Returns sessions sorted by most recent first.
    """
    if not book_dir.exists():
        return []

    sessions_dir = book_dir / "sessions"

    if sessions_dir.exists() and any(sessions_dir.iterdir()):
        sessions = _parse_new_sessions(sessions_dir)
    else:
        sessions = _parse_legacy_sessions(book_dir)

    # Sort by most recent first
    sessions.sort(key=lambda s: s.start_time or datetime.min, reverse=True)
    return sessions


def _parse_new_sessions(sessions_dir: Path) -> list[Session]:
    """Parse sessions from the new sessions/ directory structure."""
    sessions = []

    for session_path in sessions_dir.iterdir():
        if not session_path.is_dir():
            continue

        session_name = session_path.name
        tool_calls = _parse_log_jsonl(session_path / "log.jsonl")

        # Determine time range from tool calls
        start_time = None
        end_time = None
        if tool_calls:
            timestamps = [tc.timestamp for tc in tool_calls]
            start_time = min(timestamps)
            end_time = max(timestamps)
        else:
            # Fall back to directory mtime
            try:
                mtime = datetime.fromtimestamp(session_path.stat().st_mtime)
                start_time = mtime
                end_time = mtime
            except OSError:
                pass

        session = Session(
            session_name=session_name,
            session_dir=session_path,
            start_time=start_time,
            end_time=end_time,
            tool_calls=tool_calls,
        )
        sessions.append(session)

    return sessions


def _parse_legacy_sessions(book_dir: Path) -> list[Session]:
    """Parse sessions from legacy log.json + findings/ structure."""
    log_path = book_dir / "log.json"
    if not log_path.exists():
        return []

    # Group tool calls by run_uuid
    calls_by_uuid: dict[str, list[ToolCall]] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                run_uuid = entry.get("run_uuid", "unknown")
                tc = ToolCall(
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                    tool_name=entry.get("tool_name", "unknown"),
                    inputs=entry.get("inputs", {}),
                    outputs_summary=entry.get("outputs_summary", ""),
                    input_tokens=entry.get("tokens", {}).get("input", 0),
                    output_tokens=entry.get("tokens", {}).get("output", 0),
                )
                if run_uuid not in calls_by_uuid:
                    calls_by_uuid[run_uuid] = []
                calls_by_uuid[run_uuid].append(tc)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    # Parse legacy findings for notes
    findings_dir = book_dir / "findings"
    notes_by_uuid: dict[str, list[Note]] = {}

    if findings_dir.exists():
        for md_file in findings_dir.glob("*.md"):
            match = re.match(r"([a-f0-9]+)-", md_file.name)
            if not match:
                continue
            run_uuid = match.group(1)
            is_explicit = "-note-" in md_file.name

            content = md_file.read_text(encoding="utf-8")
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else md_file.name

            try:
                mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
            except OSError:
                mtime = None

            note = Note(
                filename=md_file.name,
                title=title,
                content=content,
                timestamp=mtime,
                is_explicit_note=is_explicit,
            )
            if run_uuid not in notes_by_uuid:
                notes_by_uuid[run_uuid] = []
            notes_by_uuid[run_uuid].append(note)

    # Build sessions
    all_uuids = set(calls_by_uuid.keys()) | set(notes_by_uuid.keys())
    sessions = []

    for run_uuid in all_uuids:
        tool_calls = calls_by_uuid.get(run_uuid, [])
        notes = notes_by_uuid.get(run_uuid, [])

        start_time = None
        end_time = None
        if tool_calls:
            timestamps = [tc.timestamp for tc in tool_calls]
            start_time = min(timestamps)
            end_time = max(timestamps)
        elif notes:
            note_times = [n.timestamp for n in notes if n.timestamp]
            if note_times:
                start_time = min(note_times)
                end_time = max(note_times)

        session = Session(
            session_name=f"legacy-{run_uuid}",
            session_dir=None,
            start_time=start_time,
            end_time=end_time,
            tool_calls=tool_calls,
            notes=sorted(notes, key=lambda n: n.timestamp or datetime.min),
        )
        sessions.append(session)

    return sessions


def parse_memory_notes(memory_dir: Path) -> list[Note]:
    """Parse all notes from the memory directory."""
    if not memory_dir.exists():
        return []

    notes = []
    for md_file in sorted(memory_dir.glob("*.md")):
        note = _parse_note_file(md_file)
        if note:
            notes.append(note)

    return notes


def get_reports(book_dir: Path) -> list[Path]:
    """Get all report files from the book directory."""
    if not book_dir.exists():
        return []

    # Check new location first, then legacy
    reports_dir = book_dir / "reports"
    if reports_dir.exists():
        reports = list(reports_dir.glob("report-*.md"))
    else:
        reports = list(book_dir.glob("report-*.md"))

    return sorted(reports, reverse=True)
