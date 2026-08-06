from __future__ import annotations

from pathlib import Path

import pytest

from inflorescence import __version__
from inflorescence.__main__ import build_parser
from inflorescence.server import create_server


def test_package_exports_version() -> None:
    assert __version__ == "0.1.0"


def test_create_server_registers_expected_tools(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mcp, conn, llm, watcher = create_server()

    try:
        assert mcp.name == "inflorescence"
        tool_manager = mcp._tool_manager
        tool_names = set(tool_manager._tools)
        assert tool_names == {
            "index_directory",
            "list_entities",
            "get_entity_structure",
            "get_entity_context",
            "get_related_entities",
            "search_entities",
            "search_code",
            "search_semantic",
            "search_hybrid",
            "search_text",
            "cypher_query",
            "graph_schema",
        }
    finally:
        watcher.stop_all()


def test_index_directory_tool_accepts_include_and_exclude_filters(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mcp, conn, llm, watcher = create_server()

    try:
        parameters = mcp._tool_manager._tools["index_directory"].parameters
        assert set(parameters["properties"]) == {
            "path", "include_patterns", "exclude_patterns", "preview", "max_cost_usd",
            "force_rebuild",
        }
        assert parameters["required"] == ["path"]
    finally:
        watcher.stop_all()


def test_cli_dry_run_previews_without_db(tmp_path, capsys) -> None:
    from inflorescence.__main__ import _preview

    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    _preview(str(tmp_path))

    out = capsys.readouterr().out
    assert "Dry-run preview" in out
    assert "'files': 1" in out


def test_create_server_does_not_register_removed_pre_release_tools(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    mcp, conn, llm, watcher = create_server()

    try:
        tool_names = set(mcp._tool_manager._tools)
        assert "get_source_code" not in tool_names
        assert "get_file_structure" not in tool_names
        assert "list_modules" not in tool_names
        assert "explore_structure" not in tool_names
        assert "search_entity" not in tool_names
        assert "get_element_context" not in tool_names
        assert "get_related_elements" not in tool_names
    finally:
        watcher.stop_all()


def test_default_log_file_is_outside_the_installed_package() -> None:
    """The log must never land in site-packages.

    The path used to be derived from ``__file__`` by walking up three parents, which is the
    repo root for an editable checkout but the Python lib directory for an installed wheel.
    On a system-wide install that directory is root-owned, so opening the log raised
    PermissionError and every CLI command died before doing any work.
    """
    import inflorescence
    from inflorescence.__main__ import _default_log_file

    log_path = _default_log_file()
    package_dir = Path(inflorescence.__file__).resolve().parent

    assert not log_path.resolve().is_relative_to(package_dir.parent)
    assert log_path.is_absolute()
    assert log_path.name == "inflorescence.log"


def test_log_handler_rotates_and_tolerates_an_unwritable_path(tmp_path) -> None:
    """Rotation is required (this log reached 80 MB unrotated); an unwritable path is not fatal."""
    from logging.handlers import RotatingFileHandler

    from inflorescence.__main__ import _LOG_BACKUP_COUNT, _LOG_MAX_BYTES, _build_log_handler

    handler = _build_log_handler(str(tmp_path / "nested" / "dir" / "inflorescence.log"))
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes == _LOG_MAX_BYTES
    assert handler.backupCount == _LOG_BACKUP_COUNT
    handler.close()

    # A path that cannot be opened degrades to no file logging rather than crashing.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    assert _build_log_handler(str(blocker / "sub" / "inflorescence.log")) is None


def test_log_level_setting_is_honoured(monkeypatch) -> None:
    """`LOG_LEVEL` was documented and declared but had no consumer — setting it did nothing."""
    import logging

    from inflorescence.__main__ import _resolve_log_level

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    assert _resolve_log_level(debug=False) == logging.WARNING
    # --debug still wins over the setting.
    assert _resolve_log_level(debug=True) == logging.DEBUG
    monkeypatch.setenv("LOG_LEVEL", "not-a-level")
    assert _resolve_log_level(debug=False) == logging.INFO


def test_build_parser_serve_accepts_allow_degraded() -> None:
    args = build_parser().parse_args(["serve", "--allow-degraded"])
    assert args.command == "serve"
    assert args.allow_degraded is True


def test_build_parser_serve_defaults_allow_degraded_false() -> None:
    args = build_parser().parse_args(["serve"])
    assert args.allow_degraded is False


def test_build_parser_default_command_is_none() -> None:
    args = build_parser().parse_args([])
    assert args.command is None
    assert getattr(args, "allow_degraded", False) is False


async def test_startup_returns_true_and_sets_up_schema_when_healthy(monkeypatch) -> None:
    from inflorescence import server as server_mod

    class FakeConn:
        async def health_check(self) -> bool:
            return True

    called = {}

    async def fake_setup(conn) -> None:
        called["schema"] = True

    monkeypatch.setattr(server_mod, "setup_schema", fake_setup)
    healthy = await server_mod.startup(FakeConn(), allow_degraded=False)
    assert healthy is True
    assert called.get("schema") is True


async def test_startup_returns_false_and_skips_schema_when_unhealthy(monkeypatch) -> None:
    from inflorescence import server as server_mod

    class FakeConn:
        async def health_check(self) -> bool:
            return False

    called = {}

    async def fake_setup(conn) -> None:
        called["schema"] = True

    monkeypatch.setattr(server_mod, "setup_schema", fake_setup)
    healthy = await server_mod.startup(FakeConn(), allow_degraded=True)
    assert healthy is False
    assert "schema" not in called


def _fake_server_parts(healthy: bool, ran: dict):
    class FakeConn:
        async def health_check(self) -> bool:
            return healthy

        async def close(self) -> None:
            ran["conn_closed"] = True

    class FakeLLM:
        async def close(self) -> None:
            ran["llm_closed"] = True

    class FakeWatcher:
        def stop_all(self) -> None:
            ran["watcher_stopped"] = True

    class FakeMCP:
        async def run_stdio_async(self) -> None:
            ran["ran_server"] = True

    def fake_create_server():
        return FakeMCP(), FakeConn(), FakeLLM(), FakeWatcher()

    return fake_create_server


def test_serve_exits_1_when_memgraph_unreachable(monkeypatch) -> None:
    from inflorescence import __main__ as m
    from inflorescence import server as server_mod

    ran: dict = {}
    monkeypatch.setattr(server_mod, "create_server", _fake_server_parts(healthy=False, ran=ran))

    with pytest.raises(SystemExit) as exc:
        m._serve(allow_degraded=False)

    assert exc.value.code == 1
    assert ran.get("ran_server") is None  # server loop never entered
    assert ran.get("conn_closed") is True  # cleanup still ran


def test_serve_allow_degraded_starts_despite_unreachable(monkeypatch) -> None:
    from inflorescence import __main__ as m
    from inflorescence import server as server_mod

    ran: dict = {}
    monkeypatch.setattr(server_mod, "create_server", _fake_server_parts(healthy=False, ran=ran))

    m._serve(allow_degraded=True)  # must NOT raise

    assert ran.get("ran_server") is True
    assert ran.get("conn_closed") is True
