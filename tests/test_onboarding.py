from __future__ import annotations

import argparse
import json
import subprocess
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


def _stub_db(
    monkeypatch,
    *,
    healthy: bool,
    schema_ok: bool = True,
    projects: list[dict] | None = None,
    watch_state: dict | None = None,
) -> None:
    async def fake_health(self) -> bool:
        return healthy

    async def fake_close(self) -> None:
        return None

    async def fake_schema(conn) -> bool:
        return schema_ok

    # The watch-daemon check reads the project registry. Unstubbed it would query the
    # developer's own Memgraph, and the verdict would depend on which projects that machine
    # happens to have registered — a test that passes or fails by accident of the host.
    async def fake_list_projects(self) -> list[dict]:
        return [dict(row) for row in (projects or [])]

    async def fake_watch_state(self, project: str) -> dict:
        return dict(watch_state or {"watched": False, "live": False, "holder": None})

    monkeypatch.setattr(
        "inflorescence.db.repository.GraphRepository.list_projects", fake_list_projects
    )
    monkeypatch.setattr(
        "inflorescence.db.repository.GraphRepository.get_watch_state", fake_watch_state
    )

    monkeypatch.setattr("inflorescence.db.connection.MemgraphConnection.health_check", fake_health)
    monkeypatch.setattr("inflorescence.db.connection.MemgraphConnection.close", fake_close)
    monkeypatch.setattr("inflorescence.db.schema.schema_present", fake_schema)
    # `doctor` counts the containers over the database volume, which on a developer's own
    # machine is a real and occasionally messy number. Pin it to the clean case; the test
    # that cares about duplicates supplies its own.
    monkeypatch.setattr("inflorescence.docker_env.containers_on_volume", lambda *a, **k: [])


def _stub_scip(monkeypatch, runner: str, *, images_local: bool = True) -> None:
    """Pin what `doctor` sees for the SCIP rung.

    Without this the check reads the *host*: a machine with Docker reports "docker",
    a bare CI runner reports "none". Asserting on that makes the outcome depend on
    which runner picked the job up — the macOS runner has no Docker and reddened a
    green suite over a documented, non-critical degradation. `image_availability`
    shells out to `docker image inspect`, so it is pinned for the same reason.
    """
    from inflorescence.code_indexer.scip_semantic import ImageStatus

    def fake_availability(config) -> dict[str, str]:
        return {"go": runner, "python": runner, "typescript": runner}

    def fake_images(config) -> list[ImageStatus]:
        if runner != "docker":
            return []
        return [
            ImageStatus("ghcr.io/uiqkos/inflorescence-scip-go:v1", ["go"], images_local),
            ImageStatus("ghcr.io/uiqkos/inflorescence-scip-node:v1", ["python", "typescript"], images_local),
        ]

    monkeypatch.setattr(
        "inflorescence.code_indexer.scip_semantic.indexer_availability", fake_availability
    )
    monkeypatch.setattr(
        "inflorescence.code_indexer.scip_semantic.image_availability", fake_images
    )


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
    _stub_scip(monkeypatch, "docker")

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = await _doctor_checks(Settings())
    assert all(r.ok for r in results)


async def test_doctor_reports_registered_projects_that_nothing_is_following(monkeypatch, tmp_path) -> None:
    """The state after a reboot: the registry says followed, and nothing is following.

    It fails no other check — config, key, database and schema are all fine — and every tool
    keeps answering from an index that stopped moving. Non-critical, because the machine still
    works; reported, because nothing else here would say it.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(
        monkeypatch,
        healthy=True,
        schema_ok=True,
        projects=[{"project": "p_1", "root_path": "/repo", "watched": True}],
        watch_state={"watched": True, "live": False, "holder": None},
    )
    _stub_scip(monkeypatch, "docker")

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}

    assert results["Watch daemon"].ok is False
    assert results["Watch daemon"].critical is False
    assert "0 of 1" in results["Watch daemon"].detail
    assert "watch run" in (results["Watch daemon"].hint or "")


async def test_doctor_is_green_when_a_daemon_follows_every_registered_project(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(
        monkeypatch,
        healthy=True,
        schema_ok=True,
        projects=[{"project": "p_1", "root_path": "/repo", "watched": True}],
        watch_state={"watched": True, "live": True, "holder": "host:1:x"},
    )
    _stub_scip(monkeypatch, "docker")

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}

    assert results["Watch daemon"].ok is True
    assert "1 of 1" in results["Watch daemon"].detail


async def test_doctor_reports_a_second_container_over_the_database_volume(monkeypatch, tmp_path) -> None:
    """The fault that reads as "the MCP server just stopped connecting" and survives every
    restart: two containers over one Memgraph volume. Nothing is wrong until one of them
    claims the published port, after which the other runs on unreachably — so `doctor` must
    name it while the database still answers, not once it no longer does."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)
    _stub_scip(monkeypatch, "docker")
    monkeypatch.setattr(
        "inflorescence.docker_env.containers_on_volume",
        lambda *a, **k: [("inflorescence-memgraph", "running"), ("inflorescence-memgraph-1", "exited")],
    )

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}
    duplicate = results["Single Memgraph container"]
    assert duplicate.ok is False
    assert "inflorescence-memgraph-1" in duplicate.detail
    assert "docker stop inflorescence-memgraph-1" in duplicate.hint
    # Not critical: the machine works right now, and exiting 1 would fail a suite over a
    # database that answers. The printed verdict is what has to stop saying "OK".
    assert duplicate.critical is False


async def test_doctor_is_silent_about_containers_on_a_clean_machine(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)
    _stub_scip(monkeypatch, "docker")

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name for r in await _doctor_checks(Settings())}
    assert "Single Memgraph container" not in results


async def test_doctor_stays_green_without_any_scip_indexer(monkeypatch, tmp_path) -> None:
    """No Docker and no indexer binary is a degradation, not a fault.

    The rung is reported so "why are my CALLS heuristic" has an answer, but it must
    never be critical: `doctor` still exits 0 on a machine that will index perfectly
    well through the syntactic ladder.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)
    _stub_scip(monkeypatch, "none")

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}
    assert results["SCIP indexers"].ok is False
    assert results["SCIP indexers"].critical is False
    assert all(r.ok for r in results.values() if r.critical)


async def test_doctor_names_the_image_and_where_it_comes_from(monkeypatch, tmp_path) -> None:
    """The docker rung must not be a black box: `doctor` says which reference will run
    and whether `docker run` will touch the network for it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)
    _stub_scip(monkeypatch, "docker", images_local=True)

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}
    go_row = results["SCIP image (go)"]
    assert go_row.ok is True and go_row.critical is False
    assert "ghcr.io/uiqkos/inflorescence-scip-go:v1" in go_row.detail
    assert "present locally" in go_row.detail
    # python and typescript share the node image — one row, not two
    node_row = results["SCIP image (python, typescript)"]
    assert "inflorescence-scip-node" in node_row.detail


async def test_doctor_says_when_the_image_would_be_pulled(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)
    _stub_scip(monkeypatch, "docker", images_local=False)

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = {r.name: r for r in await _doctor_checks(Settings())}
    assert "will be pulled" in results["SCIP image (go)"].detail


async def test_doctor_shows_no_image_rows_when_the_rung_is_not_docker(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    _stub_db(monkeypatch, healthy=True, schema_ok=True)
    _stub_scip(monkeypatch, "native")

    from inflorescence.__main__ import _doctor_checks
    from inflorescence.config import Settings

    results = await _doctor_checks(Settings())
    assert not any(r.name.startswith("SCIP image (") for r in results)


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


def _script(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    return path


def test_resolve_console_script_prefers_the_running_install_over_path(monkeypatch, tmp_path) -> None:
    """Field case: a fresh `uv tool install` ran `init`, an older pip copy sat earlier on PATH,
    and the PATH one was registered — every session would have served the old code."""
    from inflorescence import __main__ as m

    running = _script(tmp_path / "tool" / "bin" / "inflorescence")
    stale = _script(tmp_path / "old" / "bin" / "inflorescence")
    monkeypatch.setattr(m.sys, "argv", [str(running), "init"])
    monkeypatch.setattr(m.shutil, "which", lambda name: str(stale))

    resolved = m.resolve_console_script()
    assert resolved == str(running.resolve())
    assert Path(resolved).is_absolute()


def test_resolve_console_script_uses_the_venv_bin_unresolved(monkeypatch, tmp_path) -> None:
    """`python -m inflorescence`: argv[0] is not the script, so the interpreter's own bin is
    next — taken as the venv path, since its `python` is a symlink into a shared install."""
    from inflorescence import __main__ as m

    shared = _script(tmp_path / "uv-python" / "bin" / "python3")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(shared)
    script = _script(venv_bin / "inflorescence")
    monkeypatch.setattr(m.sys, "argv", ["-m"])
    monkeypatch.setattr(m.sys, "executable", str(venv_bin / "python"))
    monkeypatch.setattr(m.shutil, "which", lambda name: None)

    assert m.resolve_console_script() == str(script.resolve())


def test_resolve_console_script_falls_back_to_path(monkeypatch, tmp_path) -> None:
    from inflorescence import __main__ as m

    on_path = _script(tmp_path / "bin" / "inflorescence")
    monkeypatch.setattr(m.sys, "argv", ["-m"])
    monkeypatch.setattr(m.sys, "executable", str(tmp_path / "nowhere" / "python"))
    monkeypatch.setattr(m.shutil, "which", lambda name: str(on_path))

    assert m.resolve_console_script() == str(on_path.resolve())


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
    # Claude Code's default "local" scope binds the server to the directory init ran in.
    assert cmd[cmd.index("--scope") + 1] == "user"
    assert cmd.index("--scope") < cmd.index("--")


def _stub_memgraph(monkeypatch, started: bool = True) -> None:
    """`init` must never touch a real docker daemon from a test."""
    from inflorescence import __main__ as m

    monkeypatch.setattr(m, "_bootstrap_memgraph", lambda cwd: None)


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

    def run(cmd, *a, **k):  # type: ignore[no-untyped-def]
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(m.subprocess, "run", run)
    monkeypatch.setattr(m, "_cmd_doctor", lambda settings: 0)
    _stub_memgraph(monkeypatch)

    m._cmd_init(argparse.Namespace(command="init"))
    # one of the subprocess calls must be the `claude mcp add ... serve` registration
    assert any(c[:3] == ["claude", "mcp", "add"] and c[-1] == "serve" for c in calls)


def test_cmd_init_says_when_claude_kept_an_existing_registration(monkeypatch, tmp_path, capsys) -> None:
    """A re-run: `claude mcp add` refuses a name it already has, and that registration may
    point at another install — init must say so, not imply the new one is in use."""
    from inflorescence import __main__ as m

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(m.getpass, "getpass", lambda *a, **k: "sk-test")
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    monkeypatch.setattr(m.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1))
    monkeypatch.setattr(m, "_cmd_doctor", lambda settings: 0)
    _stub_memgraph(monkeypatch)

    m._cmd_init(argparse.Namespace(command="init"))
    out = capsys.readouterr().out
    assert "kept its existing registration" in out
    assert "claude mcp remove --scope user inflorescence" in out


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


def _stub_docker_env(
    monkeypatch,
    *,
    port_already_open: bool = False,
    ensured: bool = True,
    containers: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Pin every docker_env probe `_bootstrap_memgraph` makes; returns the event log."""
    import inflorescence.docker_env as de

    events: list[str] = []
    monkeypatch.setattr(de, "port_open", lambda *a, **k: port_already_open)
    monkeypatch.setattr(de, "ensure_memgraph", lambda url, **k: events.append("ensure_memgraph") or ensured)
    # Unpinned this shells out to the *host's* docker, so a developer laptop with a
    # container left over from real use would steer the test down a different branch.
    monkeypatch.setattr(de, "containers_on_volume", lambda *a, **k: list(containers or []))
    return events


def test_bootstrap_memgraph_skips_docker_when_the_port_already_answers(monkeypatch, tmp_path, capsys) -> None:
    from inflorescence import __main__ as m

    events = _stub_docker_env(monkeypatch, port_already_open=True)
    monkeypatch.setattr(m, "_ensure_docker_interactive", lambda: events.append("docker") or True)

    m._bootstrap_memgraph(tmp_path)
    assert events == []  # neither docker nor ensure_memgraph consulted
    assert "already up" in capsys.readouterr().out


def test_bootstrap_memgraph_stops_when_docker_cannot_be_had(monkeypatch, tmp_path) -> None:
    """`_ensure_docker_interactive` prints its own guidance; the step must not pile a
    second failure (a doomed `docker compose`/`docker run`) on top of it."""
    from inflorescence import __main__ as m

    events = _stub_docker_env(monkeypatch)
    monkeypatch.setattr(m, "_ensure_docker_interactive", lambda: False)

    m._bootstrap_memgraph(tmp_path)
    assert events == []


def test_bootstrap_memgraph_starts_the_container_outside_a_checkout(monkeypatch, tmp_path, capsys) -> None:
    from inflorescence import __main__ as m

    events = _stub_docker_env(monkeypatch, ensured=True)
    monkeypatch.setattr(m, "_ensure_docker_interactive", lambda: True)
    monkeypatch.setattr(m.shutil, "which", lambda name: f"/usr/bin/{name}")

    m._bootstrap_memgraph(tmp_path)  # tmp_path has no compose file
    assert events == ["ensure_memgraph"]
    assert "Memgraph is up" in capsys.readouterr().out


def test_bootstrap_memgraph_prefers_compose_inside_a_checkout(monkeypatch, tmp_path) -> None:
    from inflorescence import __main__ as m

    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    events = _stub_docker_env(monkeypatch)
    monkeypatch.setattr(m, "_ensure_docker_interactive", lambda: True)
    monkeypatch.setattr(m.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd))

    m._bootstrap_memgraph(tmp_path)
    assert ["docker", "compose", "up", "-d", "memgraph"] in calls
    assert events == []


def test_bootstrap_memgraph_leaves_an_existing_container_to_the_direct_path(monkeypatch, tmp_path) -> None:
    """Compose refuses to adopt a container it did not create, so on a machine that already
    has one it would only print a name conflict. Worse, before the container name was pinned
    it did not conflict at all — it built a second Memgraph over the same volume, and the two
    then fought for the bolt port until one was left running yet attached to no network."""
    from inflorescence import __main__ as m

    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    events = _stub_docker_env(monkeypatch, containers=[("inflorescence-memgraph", "exited")])
    monkeypatch.setattr(m, "_ensure_docker_interactive", lambda: True)
    monkeypatch.setattr(m.shutil, "which", lambda name: f"/usr/bin/{name}")
    calls: list[list[str]] = []
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd))

    m._bootstrap_memgraph(tmp_path)
    assert calls == []
    assert events == ["ensure_memgraph"]


def test_ensure_docker_interactive_is_silent_when_docker_already_works(monkeypatch, capsys) -> None:
    import inflorescence.docker_env as de
    from inflorescence import __main__ as m

    monkeypatch.setattr(de, "docker_available", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no prompt")))

    assert m._ensure_docker_interactive() is True
    assert capsys.readouterr().out == ""


def test_ensure_docker_interactive_launches_a_stopped_daemon(monkeypatch) -> None:
    import inflorescence.docker_env as de
    from inflorescence import __main__ as m

    monkeypatch.setattr(de, "docker_available", lambda: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: True)
    monkeypatch.setattr(de, "ensure_docker_daemon", lambda **k: True)

    assert m._ensure_docker_interactive() is True


def test_ensure_docker_interactive_points_at_installers_when_docker_is_absent(monkeypatch, capsys) -> None:
    import inflorescence.docker_env as de
    from inflorescence import __main__ as m

    monkeypatch.setattr(de, "docker_available", lambda: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: False)
    monkeypatch.setattr(m.shutil, "which", lambda name: None)  # no brew either

    assert m._ensure_docker_interactive() is False
    out = capsys.readouterr().out
    assert "docker.com" in out and "orbstack.dev" in out


def test_ensure_docker_interactive_respects_a_declined_brew_install(monkeypatch) -> None:
    import inflorescence.docker_env as de
    from inflorescence import __main__ as m

    monkeypatch.setattr(de, "docker_available", lambda: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: False)
    monkeypatch.setattr(m.sys, "platform", "darwin")
    monkeypatch.setattr(m.shutil, "which", lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")
    calls: list[list[str]] = []
    monkeypatch.setattr(m.subprocess, "run", lambda cmd, *a, **k: calls.append(cmd))

    assert m._ensure_docker_interactive() is False
    assert calls == []  # declining must not install anything


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
