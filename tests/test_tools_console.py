"""Tests for the dashboard's MCP tools console."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from mcp.types import TextContent

from inflorescence.config import Settings
from inflorescence.dashboard.app import create_app, is_cross_origin
from inflorescence.dashboard.tools_console import (
    ToolsConsole,
    _arg_preview,
    _error_envelope,
    _normalize,
    confirmation_required,
)


def _settings() -> Settings:
    return Settings(_env_file=None)


class TestConfirmationRequired:
    """Indexing for real is the one console action that spends LLM money."""

    def test_index_without_preview_needs_confirmation(self):
        assert confirmation_required("index_directory", {"path": "/x"}) is not None

    def test_preview_is_free_and_needs_nothing(self):
        assert confirmation_required("index_directory", {"path": "/x", "preview": True}) is None

    def test_other_tools_are_not_gated(self):
        assert confirmation_required("search_semantic", {"query": "x"}) is None


class TestNormalize:
    def test_json_text_block_is_parsed_back_into_a_payload(self):
        blocks, structured = _normalize([TextContent(type="text", text='{"items": [1]}')])
        assert blocks == [{"type": "text", "text": '{"items": [1]}'}]
        assert structured == {"items": [1]}

    def test_non_json_text_leaves_no_structured_payload(self):
        blocks, structured = _normalize([TextContent(type="text", text="plain")])
        assert blocks and structured is None

    def test_structured_tuple_form_is_kept_as_is(self):
        result = ([TextContent(type="text", text="ignored")], {"a": 1})
        _, structured = _normalize(result)
        assert structured == {"a": 1}

    def test_empty_result(self):
        assert _normalize([]) == ([], None)


class TestErrorEnvelope:
    """The tools report failure as data; a call that returned one still succeeded."""

    def test_tool_error_payload_is_recognised(self):
        envelope = _error_envelope({"error": {"code": "project_not_indexed", "message": "nope"}})
        assert envelope == {"code": "project_not_indexed", "message": "nope"}

    def test_ordinary_payload_has_no_envelope(self):
        assert _error_envelope({"items": []}) is None
        assert _error_envelope(None) is None


class TestArgPreview:
    def test_prefers_the_argument_that_identifies_the_call(self):
        assert _arg_preview({"directory": "/repo", "query": "parse"}) == "parse"

    def test_falls_back_to_the_directory(self):
        assert _arg_preview({"directory": "/repo"}) == "/repo"

    def test_long_values_are_clipped(self):
        assert len(_arg_preview({"query": "x" * 200})) == 60

    def test_no_interesting_arguments(self):
        assert _arg_preview({"limit": 10}) == ""


class _FakeMCP:
    """Stands in for the FastMCP server so the console can be tested without one."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]):
        self.calls.append((name, arguments))
        return await self._handler(name, arguments)


def _console_with(monkeypatch, handler) -> ToolsConsole:
    console = ToolsConsole(_settings())
    fake = _FakeMCP(handler)

    async def _mcp(self):
        return fake

    monkeypatch.setattr(ToolsConsole, "_mcp", _mcp)
    return console


def _wait_for_finish(console: ToolsConsole, run_id: str, timeout: float = 5.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = console.get(run_id)
        if run is not None and run.state != "running":
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never finished")


class TestRunLifecycle:
    """Runs execute on the console's own loop, off the dashboard's."""

    async def test_successful_call_records_payload_and_duration(self, monkeypatch):
        async def handler(name, arguments):
            return [TextContent(type="text", text=json.dumps({"echo": arguments}))]

        console = _console_with(monkeypatch, handler)
        try:
            run = console.start("search_entities", {"directory": "/repo", "query": "x"})
            finished = _wait_for_finish(console, run.id)
            assert finished.state == "ok"
            assert finished.structured == {"echo": {"directory": "/repo", "query": "x"}}
            assert finished.duration_ms is not None
            assert finished.error is None
        finally:
            await console.close()

    async def test_raising_tool_becomes_an_error_run(self, monkeypatch):
        async def handler(name, arguments):
            raise RuntimeError("boom")

        console = _console_with(monkeypatch, handler)
        try:
            run = console.start("graph_schema", {"directory": "/repo"})
            finished = _wait_for_finish(console, run.id)
            assert finished.state == "error"
            assert "boom" in finished.error
        finally:
            await console.close()

    async def test_error_envelope_is_surfaced_separately_from_a_failure(self, monkeypatch):
        async def handler(name, arguments):
            payload = {"error": {"code": "project_not_indexed", "message": "index it first"}}
            return [TextContent(type="text", text=json.dumps(payload))]

        console = _console_with(monkeypatch, handler)
        try:
            run = console.start("search_code", {"directory": "/repo", "query": "x"})
            finished = _wait_for_finish(console, run.id)
            # The call itself succeeded — only its payload reports a problem.
            assert finished.state == "ok"
            assert finished.error_envelope == {"code": "project_not_indexed", "message": "index it first"}
        finally:
            await console.close()

    async def test_cancel_stops_a_run_that_is_awaiting(self, monkeypatch):
        async def handler(name, arguments):
            await asyncio.sleep(30)
            return []

        console = _console_with(monkeypatch, handler)
        try:
            run = console.start("index_directory", {"path": "/repo", "preview": True})
            deadline = time.monotonic() + 3
            while getattr(console.get(run.id), "task", None) is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert console.cancel(run.id) is True
            finished = _wait_for_finish(console, run.id)
            assert finished.state == "cancelled"
            assert finished.cancel_requested is True
        finally:
            await console.close()

    async def test_cancelling_an_unknown_or_finished_run_is_a_no_op(self, monkeypatch):
        async def handler(name, arguments):
            return [TextContent(type="text", text="{}")]

        console = _console_with(monkeypatch, handler)
        try:
            assert console.cancel("nope") is False
            run = console.start("graph_schema", {"directory": "/repo"})
            _wait_for_finish(console, run.id)
            assert console.cancel(run.id) is False
        finally:
            await console.close()

    async def test_runs_are_listed_newest_first(self, monkeypatch):
        async def handler(name, arguments):
            return [TextContent(type="text", text="{}")]

        console = _console_with(monkeypatch, handler)
        try:
            first = console.start("graph_schema", {"directory": "/a"})
            second = console.start("graph_schema", {"directory": "/b"})
            _wait_for_finish(console, first.id)
            _wait_for_finish(console, second.id)
            assert [r["id"] for r in console.runs()] == [second.id, first.id]
        finally:
            await console.close()

    async def test_catalog_reports_a_server_that_cannot_be_built(self, monkeypatch):
        console = ToolsConsole(_settings())

        async def _boom(self):
            raise RuntimeError("no memgraph")

        monkeypatch.setattr(ToolsConsole, "_mcp", _boom)
        try:
            catalog = await console.catalog()
            assert catalog["available"] is False
            assert "no memgraph" in catalog["error"]
            assert catalog["tools"] == []
        finally:
            await console.close()


class _Req:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class TestCrossOriginGuard:
    """The dashboard has no auth, so a page on another origin must not be able to
    POST to it — a tool call spends money and reads arbitrary directories."""

    def test_request_without_origin_is_allowed(self):
        assert is_cross_origin(_Req({"host": "localhost:8321"})) is False  # type: ignore[arg-type]

    def test_same_origin_is_allowed(self):
        assert is_cross_origin(_Req({"origin": "http://localhost:8321", "host": "localhost:8321"})) is False  # type: ignore[arg-type]

    def test_foreign_origin_is_refused(self):
        assert is_cross_origin(_Req({"origin": "https://evil.example", "host": "localhost:8321"})) is True  # type: ignore[arg-type]


class TestToolsRoutes:
    def test_routes_are_registered(self):
        app = create_app(_settings())
        paths = {getattr(r, "path", None) for r in app.routes}
        assert "/api/tools" in paths
        assert "/api/tools/call" in paths
        assert "/api/tools/runs/{run_id}" in paths
        assert "/api/tools/runs/{run_id}/cancel" in paths


@pytest.fixture
def stub_console(monkeypatch):
    """A console whose catalog is fixed and whose calls never touch a real tool."""

    async def _catalog(self):
        return {
            "available": True,
            "tools": [{"name": "search_entities"}, {"name": "index_directory"}],
            "categories": [],
        }

    started: list[tuple[str, dict[str, Any]]] = []

    def _start(self, tool, arguments):
        started.append((tool, arguments))
        from inflorescence.dashboard.tools_console import ToolRun

        return ToolRun(id="1", tool=tool, arguments=arguments)

    monkeypatch.setattr(ToolsConsole, "catalog", _catalog)
    monkeypatch.setattr(ToolsConsole, "start", _start)
    return started


class TestCallEndpoint:
    def test_unknown_tool_is_refused(self, stub_console):
        from starlette.testclient import TestClient

        with TestClient(create_app(_settings())) as client:
            body = client.post("/api/tools/call", json={"tool": "nope", "arguments": {}}).json()
        assert "unknown tool" in body["error"]
        assert stub_console == []

    def test_spending_call_without_confirmation_is_refused(self, stub_console):
        from starlette.testclient import TestClient

        with TestClient(create_app(_settings())) as client:
            body = client.post(
                "/api/tools/call", json={"tool": "index_directory", "arguments": {"path": "/repo"}}
            ).json()
        assert body["needs_confirm"] is True
        assert stub_console == []

    def test_confirmed_spending_call_runs(self, stub_console):
        from starlette.testclient import TestClient

        with TestClient(create_app(_settings())) as client:
            body = client.post(
                "/api/tools/call",
                json={"tool": "index_directory", "arguments": {"path": "/repo"}, "confirm": True},
            ).json()
        assert body["run"]["tool"] == "index_directory"
        assert stub_console == [("index_directory", {"path": "/repo"})]

    def test_cross_origin_post_is_refused_before_the_tool_is_reached(self, stub_console):
        from starlette.testclient import TestClient

        with TestClient(create_app(_settings())) as client:
            response = client.post(
                "/api/tools/call",
                json={"tool": "search_entities", "arguments": {}},
                headers={"Origin": "https://evil.example"},
            )
        assert response.status_code == 403
        assert stub_console == []

    def test_non_object_arguments_are_refused(self, stub_console):
        from starlette.testclient import TestClient

        with TestClient(create_app(_settings())) as client:
            body = client.post("/api/tools/call", json={"tool": "search_entities", "arguments": [1]}).json()
        assert "must be an object" in body["error"]
        assert stub_console == []
