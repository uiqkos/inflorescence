from __future__ import annotations

import argparse
import json
from pathlib import Path


async def test_schema_present_true_when_required_indexes_exist(monkeypatch) -> None:
    from inflorescence.db import schema

    async def fake_existing(conn) -> set[str]:
        return {"property::Code::id", "property::Code::project", "named::summary_text"}

    monkeypatch.setattr(schema, "_existing_indexes", fake_existing)
    assert await schema.schema_present(object()) is True


async def test_schema_present_false_when_missing(monkeypatch) -> None:
    from inflorescence.db import schema

    async def fake_existing(conn) -> set[str]:
        return set()

    monkeypatch.setattr(schema, "_existing_indexes", fake_existing)
    assert await schema.schema_present(object()) is False


async def test_existing_indexes_normalises_list_valued_property() -> None:
    """Memgraph 3.x returns `property` as a list (e.g. ["id"]); the key must
    still be `property::Label::col`, not `property::Label::['id']`."""
    from inflorescence.db import schema

    class _Record:
        def __init__(self, data: dict[str, object]) -> None:
            self._data = data

        def data(self) -> dict[str, object]:
            return self._data

    class _Conn:
        async def execute_query(self, query: str) -> list[_Record]:
            assert query == "SHOW INDEX INFO;"
            return [
                _Record({"index type": "label+property", "label": "Code", "property": ["id"]}),
                _Record({"index type": "label+property", "label": "Code", "property": ["project"]}),
            ]

    keys = await schema._existing_indexes(_Conn())
    assert {"property::Code::id", "property::Code::project"}.issubset(keys)
    assert await schema.schema_present(_Conn()) is True


def _stub_db(monkeypatch, *, healthy: bool, schema_ok: bool = True) -> None:
    async def fake_health(self) -> bool:
        return healthy

    async def fake_close(self) -> None:
        return None

    async def fake_schema(conn) -> bool:
        return schema_ok

    monkeypatch.setattr("inflorescence.db.connection.MemgraphConnection.health_check", fake_health)
    monkeypatch.setattr("inflorescence.db.connection.MemgraphConnection.close", fake_close)
    monkeypatch.setattr("inflorescence.db.schema.schema_present", fake_schema)


async def test_doctor_flags_missing_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _stub_db(monkeypatch, healthy=True, schema_ok=True)

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}
    assert results["OPENROUTER_API_KEY"].ok is False
    assert results["OPENROUTER_API_KEY"].critical is True


async def test_doctor_reports_unreachable_memgraph(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=False)

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}
    assert results["Memgraph reachable"].ok is False
    assert results["Memgraph reachable"].critical is True
    # schema is not probed when the DB is down
    assert results["Schema present"].ok is False


async def test_doctor_all_green(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = await _doctor_checks(Settings())
    assert all(r.ok for r in results)


def test_cmd_doctor_returns_1_on_critical_failure(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _stub_db(monkeypatch, healthy=False)

    from inflorescence.__main__ import _cmd_doctor
    from inflorescence.config import Settings

    code = _cmd_doctor(Settings())
    assert code == 1
    out = capsys.readouterr().out
    assert "OPENROUTER_API_KEY" in out


def test_cmd_doctor_returns_0_when_all_pass(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)

    from inflorescence.__main__ import _cmd_doctor
    from inflorescence.config import Settings

    assert _cmd_doctor(Settings()) == 0


def test_render_env_content_sets_api_key() -> None:
    from inflorescence.__main__ import render_env_content

    assert "OPENROUTER_API_KEY=sk-123" in render_env_content("sk-123")


def test_write_env_file_writes_key(tmp_path) -> None:
    from inflorescence.__main__ import write_env_file

    target = tmp_path / ".env"
    write_env_file(target, "sk-abc")
    assert "OPENROUTER_API_KEY=sk-abc" in target.read_text()


def test_resolve_console_script_returns_absolute_path(monkeypatch, tmp_path) -> None:
    from inflorescence import __main__ as m

    fake = tmp_path / "bin" / "inflorescence"
    fake.parent.mkdir(parents=True)
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setattr(m.shutil, "which", lambda name: str(fake))

    resolved = m.resolve_console_script()
    assert resolved == str(fake.resolve())
    assert Path(resolved).is_absolute()


def test_build_mcp_config_uses_absolute_command_and_serve_args() -> None:
    from inflorescence.__main__ import build_mcp_config

    server = build_mcp_config("/abs/path/inflorescence")["mcpServers"]["inflorescence"]
    assert server["command"] == "/abs/path/inflorescence"
    assert server["args"] == ["serve"]


def test_render_mcp_config_json_is_valid_json() -> None:
    from inflorescence.__main__ import render_mcp_config_json

    parsed = json.loads(render_mcp_config_json("/abs/inflorescence"))
    assert parsed["mcpServers"]["inflorescence"]["command"] == "/abs/inflorescence"


def test_build_mcp_config_omits_env_when_no_api_key() -> None:
    from inflorescence.__main__ import build_mcp_config

    server = build_mcp_config("/abs/inflorescence")["mcpServers"]["inflorescence"]
    assert "env" not in server


def test_build_mcp_config_carries_api_key_in_env_when_provided() -> None:
    from inflorescence.__main__ import build_mcp_config

    server = build_mcp_config("/abs/inflorescence", api_key="sk-xyz")["mcpServers"]["inflorescence"]
    assert server["env"] == {"OPENROUTER_API_KEY": "sk-xyz"}


def test_build_claude_mcp_add_command_includes_absolute_path_and_serve() -> None:
    from inflorescence.__main__ import build_claude_mcp_add_command

    cmd = build_claude_mcp_add_command("/abs/inflorescence")
    assert cmd[:3] == ["claude", "mcp", "add"]
    assert "/abs/inflorescence" in cmd
    assert cmd[-1] == "serve"


def _stub_memgraph(monkeypatch, started: bool = True) -> None:
    """`init` must never touch a real docker daemon from a test."""
    import inflorescence.docker_env as de

    monkeypatch.setattr(de, "ensure_memgraph", lambda *a, **k: started)


def test_cmd_init_writes_key_to_user_config(monkeypatch, tmp_path) -> None:
    from inflorescence import __main__ as m
    from inflorescence.config import user_env_file

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(m.getpass, "getpass", lambda *a, **k: "sk-test")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    monkeypatch.setattr(m.shutil, "which", lambda name: None)  # no docker, no claude
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(m, "_cmd_doctor", lambda settings: 0)
    _stub_memgraph(monkeypatch)

    code = m._cmd_init(argparse.Namespace(command="init"))
    assert code == 0
    # The key lands in the per-user file, not beside the working directory: an MCP client
    # launches the server from somewhere else entirely and would never see a local .env.
    assert "sk-test" in user_env_file().read_text()
    assert not (tmp_path / ".env").exists()


def test_cmd_init_does_not_overwrite_key_without_confirm(monkeypatch, tmp_path) -> None:
    from inflorescence import __main__ as m
    from inflorescence.config import user_env_file

    monkeypatch.chdir(tmp_path)
    env = user_env_file()
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text("OPENROUTER_API_KEY=existing\n")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")  # decline overwrite
    monkeypatch.setattr(m.getpass, "getpass", lambda *a, **k: "SHOULD_NOT_BE_USED")
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(m, "_cmd_doctor", lambda settings: 0)
    _stub_memgraph(monkeypatch)

    m._cmd_init(argparse.Namespace(command="init"))
    assert env.read_text() == "OPENROUTER_API_KEY=existing\n"


def test_cmd_init_registers_via_claude_when_available(monkeypatch, tmp_path) -> None:
    from inflorescence import __main__ as m

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(m.getpass, "getpass", lambda *a, **k: "sk-test")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    monkeypatch.setattr(m.shutil, "which", lambda name: f"/usr/bin/{name}")  # docker + claude present
    calls: list[list[str]] = []
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd))
    monkeypatch.setattr(m, "_cmd_doctor", lambda settings: 0)
    _stub_memgraph(monkeypatch)

    m._cmd_init(argparse.Namespace(command="init"))
    # one of the subprocess calls must be the `claude mcp add ... serve` registration
    assert any(c[:3] == ["claude", "mcp", "add"] and c[-1] == "serve" for c in calls)


def test_settings_read_key_from_user_config_regardless_of_cwd(monkeypatch, tmp_path) -> None:
    """The bug this whole path exists to kill: a server launched from an unrelated
    directory came up with an empty key and failed every LLM call."""
    from inflorescence.config import Settings, user_env_file

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = user_env_file()
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text("OPENROUTER_API_KEY=sk-from-user-config\n")

    elsewhere = tmp_path / "unrelated"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert Settings().llm_api_key == "sk-from-user-config"


def test_project_env_outranks_user_config(monkeypatch, tmp_path) -> None:
    from inflorescence.config import Settings, user_env_file

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    user = user_env_file()
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text("OPENROUTER_API_KEY=sk-user\n")

    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("OPENROUTER_API_KEY=sk-project\n")
    monkeypatch.chdir(project)

    assert Settings().llm_api_key == "sk-project"
