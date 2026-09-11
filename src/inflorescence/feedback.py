"""Agent-written feedback about using this server: an append-only log on disk.

The agent on the other end of the MCP connection is the only witness to most of what is
wrong here. It is the one that read a tool description and picked the wrong tool, got an
empty page it could not distinguish from "nothing matches", or spent three calls on a
question one call should have answered. None of that reaches the person who maintains the
server: the agent works around it inside a session that is then closed, and the same
workaround is re-invented tomorrow. ``send_feedback`` is the channel that keeps it, and
this module is where it lands.

**Not in Memgraph, on purpose.** The most valuable report is written exactly when the
database is unreachable, the project is not indexed, or a tool just raised — a sink that
shares a failure domain with the thing being reported is empty at the only moment it
matters. A JSONL file under the user's config directory has no dependencies at all: an
agent can report that the whole database is down and the report survives.

**Append-only, one JSON object per line.** Appends are made with a single ``os.write`` to
a file opened ``O_APPEND``, so two processes writing at once (an MCP server per session,
plus the watch daemon) interleave whole lines rather than corrupting each other's. Per-entry
caps (:data:`MAX_MESSAGE_CHARS`, :data:`MAX_FIELD_CHARS`) keep a line small enough for that
to hold in practice and stop one agent from pasting a transcript into the log; truncation
is recorded in the entry and reported back to the caller rather than done silently (INV-7).
Nothing in this module ever deletes or rewrites an entry — reading is a separate, offline
act (``inflorescence feedback``).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inflorescence import __version__
from inflorescence.config import feedback_file

logger = logging.getLogger(__name__)

# What an entry is about. Deliberately short and non-overlapping — a taxonomy an agent has
# to think about is a taxonomy it gets wrong, and every value here answers a different
# question for whoever reads the log. `experience_report` is the catch-all for the ordinary
# case the server asks for after a session: what the tools were like to use.
KINDS: tuple[str, ...] = (
    "bug",
    "wrong_result",
    "missing_capability",
    "confusing_docs",
    "performance",
    "praise",
    "experience_report",
    "other",
)

# How much it got in the way, from the reporting agent's point of view.
SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "blocking")

DEFAULT_KIND = "other"
DEFAULT_SEVERITY = "medium"

# Per-entry caps. Generous enough for a paragraph of context plus the call that broke, small
# enough that a line stays a single atomic append and one report cannot fill a disk.
MAX_MESSAGE_CHARS = 20_000
MAX_FIELD_CHARS = 4_000


@dataclass(frozen=True)
class FeedbackLog:
    """Everything readable in the log, plus an honest count of what was not.

    ``unreadable`` exists so a reader is never quietly shown a subset: a half-written line
    (a process killed mid-append) or a hand-edited file yields lines that do not parse, and
    "12 entries" over a file holding 13 is the kind of silent lie INV-7 forbids.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)
    unreadable: int = 0


def _now() -> datetime:
    return datetime.now(UTC)


def _entry_id(moment: datetime) -> str:
    """A sortable, collision-free id: the timestamp an eye can read plus six random hex."""
    return f"fb_{moment.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"


def _clip(text: str, limit: int) -> tuple[str, bool]:
    """Bound a field, reporting whether it was cut rather than cutting silently."""
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + "…", True


def normalize_choice(value: str | None, allowed: Sequence[str], default: str) -> tuple[str, str | None]:
    """Map a caller's enum value onto *allowed*, returning ``(value, rejected_raw)``.

    An unknown value is normalized rather than refused. Refusing would be the tidier API,
    but the payload is a report that something is wrong with this server — losing it over a
    misspelled category is a worse outcome than filing it under ``other``. The raw string is
    kept in the entry (``kind_as_given``) and named in the response, so nothing is silently
    reinterpreted and the caller can correct it in a follow-up entry.
    """
    if value is None:
        return default, None
    candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
    if candidate in allowed:
        return candidate, None
    return default, value


class FeedbackStore:
    """Append entries to, and read them back from, one JSONL file."""

    def __init__(self, path: Path | str | None = None) -> None:
        # ``expanduser`` because FEEDBACK_PATH is typed by a human into a dotenv file, where
        # ``~/...`` is the natural way to write a home-relative path and nothing else expands it.
        self._path = Path(path).expanduser() if path else feedback_file()

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        message: str,
        *,
        kind: str | None = None,
        severity: str | None = None,
        tool: str | None = None,
        reproduction: str | None = None,
        context: Mapping[str, Any] | None = None,
        source: str = "mcp",
    ) -> dict[str, Any]:
        """Append one entry and return it exactly as stored.

        Raises ``OSError`` if the entry could not be written — the caller turns that into a
        visible error rather than a thank-you, because an agent told "recorded" about a
        report that went nowhere will not write it again.
        """
        moment = _now()
        kind_value, kind_raw = normalize_choice(kind, KINDS, DEFAULT_KIND)
        severity_value, severity_raw = normalize_choice(severity, SEVERITIES, DEFAULT_SEVERITY)
        body, body_clipped = _clip(message.strip(), MAX_MESSAGE_CHARS)

        entry: dict[str, Any] = {
            "id": _entry_id(moment),
            "recorded_at": moment.isoformat().replace("+00:00", "Z"),
            "kind": kind_value,
            "severity": severity_value,
            "message": body,
            "source": source,
            "server_version": __version__,
        }
        if kind_raw is not None:
            entry["kind_as_given"] = kind_raw
        if severity_raw is not None:
            entry["severity_as_given"] = severity_raw
        if tool:
            entry["tool"], _ = _clip(tool.strip(), MAX_FIELD_CHARS)
        if reproduction:
            entry["reproduction"], repro_clipped = _clip(reproduction.strip(), MAX_FIELD_CHARS)
            if repro_clipped:
                entry["reproduction_truncated"] = True
        if context:
            entry["context"] = dict(context)
        if body_clipped:
            entry["message_truncated"] = True

        self._append(entry)
        logger.info("Feedback recorded: %s (%s/%s)", entry["id"], kind_value, severity_value)
        return entry

    def _append(self, entry: Mapping[str, Any]) -> None:
        """Write one line under ``O_APPEND`` — the whole line, or nothing.

        ``0o600`` because feedback quotes source paths, code and error text from the
        machine it was written on; the log is the user's, not the world's. The mode only
        applies at creation, which is correct: a file the user has re-permissioned on
        purpose is not silently re-locked on every append.
        """
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    def read(self, limit: int | None = None) -> FeedbackLog:
        """Read the log oldest-first, keeping the newest *limit* entries.

        A missing file is an empty log, not an error: nobody has left feedback yet is the
        normal state of a fresh install.
        """
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return FeedbackLog()
        entries: list[dict[str, Any]] = []
        unreadable = 0
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                unreadable += 1
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
            else:
                unreadable += 1
        if limit is not None and limit >= 0:
            entries = entries[-limit:] if limit else []
        return FeedbackLog(entries=entries, unreadable=unreadable)
