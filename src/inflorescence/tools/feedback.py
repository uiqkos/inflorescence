"""The send_feedback MCP tool: the agent's channel back to whoever maintains this server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from inflorescence.feedback import KINDS, SEVERITIES, FeedbackStore
from inflorescence.tools.freshness import describe_origin
from inflorescence.tools.responses import error_response

logger = logging.getLogger(__name__)


def _context(directory: str | None) -> dict[str, Any]:
    """Pin the report to a tree: which path, which branch, which commit.

    A report reading "search_text returned nothing for `parse_frontmatter`" is unusable
    without it — nobody can tell later whether the index was missing the symbol, the file
    was never indexed, or the agent was querying a worktree of a branch that predates it.
    This is the same line every read tool already reports as ``meta.answered_from``, so the
    entry and the answer it is about name the tree identically.

    Best effort by construction: a path that does not exist, or is not a git checkout, is
    recorded as given. Feedback about a directory that turned out to be wrong is exactly
    the kind of report worth keeping, so nothing here can refuse an entry.
    """
    if not directory:
        return {}
    context: dict[str, Any] = {"directory": directory}
    try:
        root = Path(directory).expanduser()
        context["exists"] = root.exists()
        if context["exists"]:
            context["answered_from"] = describe_origin(root.resolve())
    except (OSError, ValueError, RuntimeError):  # unreadable path metadata is not a failure
        logger.debug("Could not describe %s for a feedback entry", directory, exc_info=True)
    return context


def register_feedback_tools(mcp: FastMCP, store: FeedbackStore) -> None:
    """Register the feedback tool with the MCP server."""

    @mcp.tool()
    async def send_feedback(
        message: str,
        kind: str = "other",
        severity: str = "medium",
        tool: str | None = None,
        directory: str | None = None,
        reproduction: str | None = None,
    ) -> dict:
        """Report to this server's maintainer what using it was actually like.

        PLEASE USE THIS. You are the only witness to how these tools behave in a real task:
        which description sent you to the wrong tool, which result was empty when the code
        plainly exists, what took four calls that one should have answered, and what worked
        so well it should not be changed. That evidence dies with your session unless you
        write it down here. **After you have finished working with this server — a search,
        an index, a whole task — send one entry describing the experience** (kind=
        "experience_report"), and send another the moment something is wrong, without
        waiting for the end of the task. Nobody sees these tools from the inside except you.

        Costs nothing, touches no LLM, and does not need an indexed project or even a
        reachable database — feedback about a broken database is exactly the feedback worth
        having, so this tool deliberately depends on neither. Entries are appended to a
        JSONL file on the user's machine (the path is in the response) and go nowhere else:
        nothing is uploaded, and no telemetry leaves the machine. Never put a secret, a
        token or a customer's data in a report; quote code and paths freely, as they are
        already on this machine.

        WHAT MAKES A REPORT USABLE: the call you made, the result you got, the result you
        expected, and what you did instead. "search_semantic is bad" cannot be acted on;
        "search_semantic(query='retry backoff') returned 5 weak hits from tests while the
        real implementation is in `net/retry.go`, which search_text found by the word
        'backoff'" can. One entry per issue — separate reports are easier to act on than
        one long list.

        Args:
            message: The report itself, in English. Free-form prose; be concrete about what
                you called, what came back, and what you expected. Up to 20 000 characters
                (a longer message is stored truncated and the response says so).
            kind: What the entry is about. One of: "bug" (something is broken or raised),
                "wrong_result" (it answered, but the answer was wrong, empty, or
                misranked), "missing_capability" (the question was reasonable and no tool
                could answer it), "confusing_docs" (a tool description or error message
                misled you — including this one), "performance" (correct but too slow or
                too many calls), "praise" (worked notably well — say what, so it survives a
                refactor), "experience_report" (the general after-the-task note this server
                asks for), "other". An unrecognized value is filed as "other" and the
                response says so; nothing is ever refused over its category.
            severity: How much it got in the way: "low", "medium" (default), "high", or
                "blocking" (the task could not be completed). Unrecognized values become
                "medium", reported the same way.
            tool: The tool the entry is about, by name (e.g. "search_text",
                "index_directory"), or a non-tool area like "server instructions". Omit for
                feedback about the server as a whole.
            directory: The project path you were working in. Worth passing whenever there
                was one: the entry then records the tree, its branch and its commit, so the
                report can be reproduced later against the same state of the code.
            reproduction: The exact call that shows the problem — tool name plus arguments,
                verbatim — when there is one. Up to 4 000 characters.

        Returns:
            ``{"stored": true, "id", "recorded_at", "path", "note"}`` — ``path`` is the
            file the entry landed in, and ``id`` identifies it in a follow-up entry (say
            "follow-up to fb_…" in the message if you later find the cause). On a failure to
            write, a ``feedback_not_stored`` error carrying the reason instead: report it to
            the user in that case, since the entry was NOT kept.
        """
        if not message or not message.strip():
            return error_response(
                "feedback_empty",
                "message must not be empty — describe what you called, what came back, and "
                "what you expected instead",
            )
        try:
            entry = store.record(
                message,
                kind=kind,
                severity=severity,
                tool=tool,
                reproduction=reproduction,
                context=_context(directory),
            )
        except OSError as exc:
            # Losing the report silently is the one outcome worse than not having the tool:
            # the agent would move on believing it was filed (INV-7).
            logger.warning("Feedback could not be written to %s: %s", store.path, exc)
            return error_response(
                "feedback_not_stored",
                f"nothing was recorded — {store.path} could not be written: {exc}. Tell the "
                "user your feedback directly instead, and mention this failure.",
            )

        notes = ["Recorded — thank you. Send another entry whenever something else surprises you."]
        if entry.get("kind_as_given"):
            notes.append(f"kind={entry['kind_as_given']!r} is not a known category; filed as 'other'.")
        if entry.get("severity_as_given"):
            notes.append(f"severity={entry['severity_as_given']!r} is not a known level; filed as 'medium'.")
        if entry.get("message_truncated"):
            notes.append("The message was longer than 20 000 characters and was stored truncated.")
        if entry.get("reproduction_truncated"):
            notes.append("The reproduction was longer than 4 000 characters and was stored truncated.")

        response: dict[str, Any] = {
            "stored": True,
            "id": entry["id"],
            "recorded_at": entry["recorded_at"],
            "kind": entry["kind"],
            "severity": entry["severity"],
            "path": str(store.path),
            "note": " ".join(notes),
        }
        # The valid values ride along only when one was rejected — an agent that needs them
        # is exactly the agent that just guessed wrong, and everyone else pays no tokens.
        if entry.get("kind_as_given"):
            response["valid_kinds"] = list(KINDS)
        if entry.get("severity_as_given"):
            response["valid_severities"] = list(SEVERITIES)
        return response
