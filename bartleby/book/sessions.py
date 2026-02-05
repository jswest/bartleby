"""Session parsing and cute naming for bartleby book."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Word lists for cute session names (~100 each for 10,000 combinations)
ADJECTIVES = [
    "amber", "azure", "bold", "bright", "calm", "clever", "coral", "crisp",
    "daring", "deep", "eager", "early", "fair", "fancy", "fast", "fierce",
    "gentle", "gleaming", "golden", "grand", "happy", "hardy", "hazy", "hidden",
    "idle", "ivory", "jade", "jolly", "keen", "kind", "lively", "lone",
    "lucky", "lunar", "merry", "mighty", "misty", "noble", "nimble", "odd",
    "olive", "opal", "pale", "peak", "plain", "proud", "quiet", "quick",
    "rare", "rapid", "regal", "rich", "rosy", "ruby", "rusty", "sage",
    "sandy", "serene", "sharp", "shy", "silent", "silver", "sleek", "slim",
    "smooth", "snowy", "soft", "solar", "solid", "spry", "stark", "steady",
    "steep", "still", "stone", "stormy", "sunny", "sweet", "swift", "tame",
    "tart", "tawny", "teal", "tender", "thin", "tidy", "tiny", "topaz",
    "twilight", "vast", "velvet", "vivid", "warm", "wavy", "wild", "wise",
    "witty", "young", "zesty", "zinc",
]

NOUNS = [
    "anchor", "arrow", "badge", "beacon", "bell", "bird", "bloom", "brook",
    "candle", "canyon", "cedar", "cliff", "cloud", "clover", "comet", "coral",
    "crane", "creek", "crown", "dawn", "delta", "dove", "drift", "dune",
    "eagle", "echo", "ember", "falcon", "fern", "finch", "flame", "flint",
    "flower", "forest", "fox", "frost", "garden", "gate", "glade", "grove",
    "harbor", "hawk", "haven", "heron", "hill", "hollow", "horizon", "inlet",
    "island", "ivy", "jasper", "jewel", "jungle", "lantern", "lark", "laurel",
    "leaf", "ledge", "lily", "lotus", "maple", "marsh", "meadow", "mesa",
    "moon", "moss", "nest", "oak", "ocean", "orchid", "otter", "owl",
    "palm", "path", "peak", "pearl", "pebble", "pine", "plain", "pond",
    "prairie", "quartz", "rain", "raven", "reef", "ridge", "river", "robin",
    "rose", "sage", "shore", "sky", "sparrow", "spring", "star", "stone",
    "storm", "stream", "summit", "sun", "swan", "thistle", "thorn", "tide",
    "trail", "tree", "valley", "wave", "willow", "wind", "wing", "wren",
]


def cute_name(run_uuid: str) -> str:
    """Generate a deterministic cute name from a run_uuid.

    Example: "85f3377d" -> "calm-meadow-85f3377d"
    """
    try:
        h = int(run_uuid, 16)
    except ValueError:
        # Fallback for non-hex strings
        h = hash(run_uuid) & 0xFFFFFFFF

    adj = ADJECTIVES[h % len(ADJECTIVES)]
    noun = NOUNS[(h // len(ADJECTIVES)) % len(NOUNS)]
    return f"{adj}-{noun}-{run_uuid}"


def short_cute_name(run_uuid: str) -> str:
    """Generate just the cute part without the uuid.

    Example: "85f3377d" -> "calm-meadow"
    """
    try:
        h = int(run_uuid, 16)
    except ValueError:
        h = hash(run_uuid) & 0xFFFFFFFF

    adj = ADJECTIVES[h % len(ADJECTIVES)]
    noun = NOUNS[(h // len(ADJECTIVES)) % len(NOUNS)]
    return f"{adj}-{noun}"


@dataclass
class Note:
    """A note or finding from a session."""
    filename: str
    title: str
    content: str
    timestamp: datetime | None
    run_uuid: str
    is_explicit_note: bool  # True if saved via save_note, False if auto-generated


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
    """A research session identified by run_uuid."""
    run_uuid: str
    cute_name: str
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


def _parse_note_file(filepath: Path) -> Note | None:
    """Parse a note/finding markdown file."""
    content = filepath.read_text(encoding="utf-8")
    filename = filepath.name

    # Extract run_uuid from filename pattern: {run_uuid}-{index}.md or {run_uuid}-note-{seq}.md
    match = re.match(r"([a-f0-9]+)-", filename)
    if not match:
        return None

    run_uuid = match.group(1)
    is_explicit_note = "-note-" in filename

    # Extract title from first # heading
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    title = title_match.group(1) if title_match else filename

    # Get file modification time as approximate timestamp
    try:
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
    except OSError:
        mtime = None

    return Note(
        filename=filename,
        title=title,
        content=content,
        timestamp=mtime,
        run_uuid=run_uuid,
        is_explicit_note=is_explicit_note,
    )


def _parse_log_file(log_path: Path) -> list[ToolCall]:
    """Parse log.json into ToolCall objects."""
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


def parse_sessions(book_dir: Path) -> list[Session]:
    """Parse all sessions from a book directory.

    Returns sessions sorted by most recent first.
    """
    if not book_dir.exists():
        return []

    # Parse log file
    log_path = book_dir / "log.json"
    all_tool_calls = _parse_log_file(log_path)

    # Group tool calls by run_uuid
    calls_by_uuid: dict[str, list[ToolCall]] = {}
    for tc in all_tool_calls:
        # We need to get run_uuid from log entries - check the raw file
        pass

    # Re-parse to get run_uuid
    calls_by_uuid = {}
    if log_path.exists():
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

    # Parse notes/findings
    findings_dir = book_dir / "findings"
    notes_by_uuid: dict[str, list[Note]] = {}

    if findings_dir.exists():
        for md_file in findings_dir.glob("*.md"):
            note = _parse_note_file(md_file)
            if note:
                if note.run_uuid not in notes_by_uuid:
                    notes_by_uuid[note.run_uuid] = []
                notes_by_uuid[note.run_uuid].append(note)

    # Build sessions
    all_uuids = set(calls_by_uuid.keys()) | set(notes_by_uuid.keys())
    sessions = []

    for run_uuid in all_uuids:
        tool_calls = calls_by_uuid.get(run_uuid, [])
        notes = notes_by_uuid.get(run_uuid, [])

        # Determine time range
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
            run_uuid=run_uuid,
            cute_name=cute_name(run_uuid),
            start_time=start_time,
            end_time=end_time,
            tool_calls=tool_calls,
            notes=sorted(notes, key=lambda n: n.timestamp or datetime.min),
        )
        sessions.append(session)

    # Sort by most recent first
    sessions.sort(key=lambda s: s.start_time or datetime.min, reverse=True)
    return sessions


def get_reports(book_dir: Path) -> list[Path]:
    """Get all report files from the book directory."""
    if not book_dir.exists():
        return []
    return sorted(book_dir.glob("report-*.md"), reverse=True)
