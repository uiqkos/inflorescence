"""The agent-facing feedback channel: the store, the MCP tool, and the CLI that reads it."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from inflorescence.config import Settings, feedback_file
from inflorescence.feedback import (
    KINDS,
    MAX_MESSAGE_CHARS,
    SEVERITIES,
    FeedbackStore,
    normalize_choice,
)
from inflorescence.tools.feedback import register_feedback_tools


def _payload(result):
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def _store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "nested" / "feedback.jsonl")


def _server(store: FeedbackStore) -> FastMCP:
    mcp = FastMCP("test")
    register_feedback_tools(mcp, store)
    return mcp


class TestFeedbackStore:
    def test_records_one_json_line_per_entry_creating_the_directory(self, tmp_path) -> None:
        store = _store(tmp_path)

        store.record("first", kind="bug")
        store.record("second", kind="praise")

        lines = store.path.read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(line)["message"] for line in lines] == ["first", "second"]

    def test_entry_carries_kind_severity_context_and_a_sortable_id(self, tmp_path) -> None:
        store = _store(tmp_path)

        entry = store.record(
            "search_text found nothing",
            kind="wrong_result",
            severity="high",
            tool="search_text",
            reproduction="search_text(query='x')",
            context={"directory": "/repo"},
        )

        assert entry["kind"] == "wrong_result"
        assert entry["severity"] == "high"
        assert entry["tool"] == "search_text"
        assert entry["reproduction"] == "search_text(query='x')"
        assert entry["context"] == {"directory": "/repo"}
        assert entry["id"].startswith("fb_") and entry["recorded_at"].endswith("Z")

    def test_unknown_kind_is_filed_not_refused_and_the_raw_value_is_kept(self, tmp_path) -> None:
        """Losing a bug report over a misspelled category is worse than filing it as other."""
        store = _store(tmp_path)

        entry = store.record("something", kind="catastrophe", severity="urgent")

        assert entry["kind"] == "other"
        assert entry["kind_as_given"] == "catastrophe"
        assert entry["severity"] == "medium"
        assert entry["severity_as_given"] == "urgent"

    def test_spelling_variants_of_a_known_kind_are_accepted_as_that_kind(self, tmp_path) -> None:
        store = _store(tmp_path)

        entry = store.record("x", kind="Wrong-Result", severity=" HIGH ")

        assert (entry["kind"], entry["severity"]) == ("wrong_result", "high")
        assert "kind_as_given" not in entry and "severity_as_given" not in entry

    def test_an_oversized_message_is_stored_truncated_and_flagged(self, tmp_path) -> None:
        store = _store(tmp_path)

        entry = store.record("x" * (MAX_MESSAGE_CHARS + 500))

        assert entry["message_truncated"] is True
        assert len(entry["message"]) <= MAX_MESSAGE_CHARS + 1  # the ellipsis

    def test_the_log_is_written_owner_only(self, tmp_path) -> None:
        """Reports quote source paths, code and error text from this machine."""
        store = _store(tmp_path)

        store.record("private detail")

        if os.name != "nt":  # POSIX permission bits only
            assert store.path.stat().st_mode & 0o777 == 0o600

    def test_reading_a_missing_log_is_an_empty_log_not_an_error(self, tmp_path) -> None:
        log = _store(tmp_path).read()

        assert log.entries == [] and log.unreadable == 0

    def test_unparsable_lines_are_counted_rather_than_hidden(self, tmp_path) -> None:
        """INV-7: "2 entries" over a file holding three is a silent lie."""
        store = _store(tmp_path)
        store.record("kept")
        with store.path.open("a", encoding="utf-8") as handle:
            handle.write("half a line, no closing brace\n")
            handle.write("\n")

        log = store.read()

        assert [entry["message"] for entry in log.entries] == ["kept"]
        assert log.unreadable == 1

    def test_read_limit_keeps_the_newest_entries(self, tmp_path) -> None:
        store = _store(tmp_path)
        for i in range(5):
            store.record(f"entry {i}")

        assert [e["message"] for e in store.read(limit=2).entries] == ["entry 3", "entry 4"]

    def test_appends_survive_interleaved_writers(self, tmp_path) -> None:
        """One MCP server per session plus the daemon can all hold the file open at once."""
        path = tmp_path / "feedback.jsonl"
        first, second = FeedbackStore(path), FeedbackStore(path)

        first.record("from one")
        second.record("from two")
        first.record("from one again")

        assert [e["message"] for e in FeedbackStore(path).read().entries] == [
            "from one",
            "from two",
            "from one again",
        ]

    def test_a_home_relative_configured_path_is_expanded(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert FeedbackStore("~/notes/feedback.jsonl").path == tmp_path / "notes/feedback.jsonl"


class TestNormalizeChoice:
    def test_none_falls_back_to_the_default_without_reporting_a_rejection(self) -> None:
        assert normalize_choice(None, KINDS, "other") == ("other", None)

    def test_an_unknown_value_is_returned_alongside_the_default(self) -> None:
        assert normalize_choice("nope", SEVERITIES, "medium") == ("medium", "nope")


@pytest.mark.asyncio
class TestSendFeedbackTool:
    async def test_a_report_is_stored_and_acknowledged_with_its_id_and_path(self, tmp_path) -> None:
        store = _store(tmp_path)

        payload = _payload(await _server(store).call_tool("send_feedback", {
            "message": "search_semantic ranked tests above the implementation",
            "kind": "wrong_result",
            "severity": "high",
            "tool": "search_semantic",
        }))

        assert payload["stored"] is True
        assert payload["id"].startswith("fb_")
        assert payload["path"] == str(store.path)
        assert store.read().entries[0]["message"].startswith("search_semantic ranked")

    async def test_an_empty_message_is_refused_with_a_hint_about_what_to_write(self, tmp_path) -> None:
        store = _store(tmp_path)

        payload = _payload(await _server(store).call_tool("send_feedback", {"message": "   "}))

        assert payload["error"]["code"] == "feedback_empty"
        assert store.read().entries == []

    async def test_a_directory_pins_the_report_to_a_tree_and_its_commit(self, tmp_path) -> None:
        store = _store(tmp_path)

        await _server(store).call_tool("send_feedback", {
            "message": "empty page", "directory": str(tmp_path)
        })

        context = store.read().entries[0]["context"]
        assert context["directory"] == str(tmp_path)
        assert context["exists"] is True

    async def test_a_directory_that_does_not_exist_still_files_the_report(self, tmp_path) -> None:
        """Feedback about a path that turned out to be wrong is feedback worth keeping."""
        store = _store(tmp_path)

        payload = _payload(await _server(store).call_tool("send_feedback", {
            "message": "nothing indexed here", "directory": "/no/such/tree"
        }))

        assert payload["stored"] is True
        assert store.read().entries[0]["context"] == {"directory": "/no/such/tree", "exists": False}

    async def test_an_unknown_kind_is_filed_and_the_valid_values_ride_back(self, tmp_path) -> None:
        store = _store(tmp_path)

        payload = _payload(await _server(store).call_tool("send_feedback", {
            "message": "x", "kind": "disaster"
        }))

        assert payload["kind"] == "other"
        assert "'disaster' is not a known category" in payload["note"]
        assert payload["valid_kinds"] == list(KINDS)

    async def test_a_well_formed_report_does_not_pay_for_the_enum_lists(self, tmp_path) -> None:
        payload = _payload(await _server(_store(tmp_path)).call_tool("send_feedback", {
            "message": "x", "kind": "praise", "severity": "low"
        }))

        assert "valid_kinds" not in payload and "valid_severities" not in payload

    async def test_a_write_failure_is_reported_as_an_error_never_as_a_thank_you(self, tmp_path) -> None:
        """An agent told "recorded" about a lost report will not write it again."""
        blocked = tmp_path / "a-file"
        blocked.write_text("not a directory", encoding="utf-8")
        store = FeedbackStore(blocked / "feedback.jsonl")

        payload = _payload(await _server(store).call_tool("send_feedback", {"message": "x"}))

        assert payload["error"]["code"] == "feedback_not_stored"
        assert "Tell the user your feedback directly" in payload["error"]["message"]

    async def test_the_tool_needs_no_database_and_no_indexed_project(self, tmp_path) -> None:
        """Feedback about a database that is down is exactly the feedback worth having."""
        store = _store(tmp_path)
        mcp = _server(store)  # built with nothing but the store — no repo, no connection

        payload = _payload(await mcp.call_tool("send_feedback", {
            "message": "Memgraph refuses connections; every tool returns the same error",
            "kind": "bug",
            "severity": "blocking",
        }))

        assert payload["stored"] is True


class TestFeedbackDocumentation:
    """The tool descriptions are the only place an agent learns any of this (CLAUDE.md)."""

    def test_the_server_instructions_ask_agents_to_report_their_experience(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        from inflorescence.server import create_server

        mcp, _conn, _llm, watcher = create_server()
        try:
            instructions = mcp.instructions or ""
        finally:
            watcher.stop_all()

        assert "send_feedback" in instructions
        assert "experience_report" in instructions

    def test_the_tool_description_enumerates_every_valid_kind_and_severity(self, tmp_path) -> None:
        mcp = _server(_store(tmp_path))

        description = mcp._tool_manager._tools["send_feedback"].description or ""

        assert all(f'"{kind}"' in description for kind in KINDS)
        assert all(f'"{severity}"' in description for severity in SEVERITIES)

    def test_the_configured_default_lands_next_to_the_user_dotenv(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

        assert feedback_file() == tmp_path / "inflorescence" / "feedback.jsonl"
        assert Settings(_env_file=None).feedback_path is None


class TestFeedbackCli:
    def test_it_prints_entries_newest_last_with_their_kind_and_tool(self, tmp_path, capsys) -> None:
        from inflorescence.__main__ import _cmd_feedback, build_parser

        store = FeedbackStore(tmp_path / "feedback.jsonl")
        store.record("older", kind="bug", severity="high", tool="search_text")
        store.record("newer", kind="praise")
        args = build_parser().parse_args(["feedback"])

        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("FEEDBACK_PATH", str(store.path))
            assert _cmd_feedback(args) == 0

        out = capsys.readouterr().out
        assert "2 entries" in out
        assert out.index("older") < out.index("newer")
        assert "bug/high" in out and "[search_text]" in out

    def test_kind_filter_and_json_output(self, tmp_path, capsys) -> None:
        from inflorescence.__main__ import _cmd_feedback, build_parser

        store = FeedbackStore(tmp_path / "feedback.jsonl")
        store.record("a bug", kind="bug")
        store.record("an experience", kind="experience_report")
        args = build_parser().parse_args(["feedback", "--kind", "bug", "--json"])

        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("FEEDBACK_PATH", str(store.path))
            assert _cmd_feedback(args) == 0

        assert [e["message"] for e in json.loads(capsys.readouterr().out)] == ["a bug"]

    def test_an_empty_log_says_how_entries_get_there(self, tmp_path, capsys) -> None:
        from inflorescence.__main__ import _cmd_feedback, build_parser

        args = build_parser().parse_args(["feedback"])

        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
            assert _cmd_feedback(args) == 0

        assert "send_feedback" in capsys.readouterr().out

    def test_path_flag_prints_only_the_file(self, tmp_path, capsys) -> None:
        from inflorescence.__main__ import _cmd_feedback, build_parser

        args = build_parser().parse_args(["feedback", "--path"])

        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("FEEDBACK_PATH", str(tmp_path / "feedback.jsonl"))
            assert _cmd_feedback(args) == 0

        assert capsys.readouterr().out.strip() == str(tmp_path / "feedback.jsonl")
