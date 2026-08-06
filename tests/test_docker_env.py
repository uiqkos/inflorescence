"""Bootstrap of the Memgraph container.

Nothing here talks to a real daemon: the argv and the decision of *whether* to start
anything are the parts worth pinning, and both are pure.
"""

from __future__ import annotations

import inflorescence.docker_env as de


def test_local_bolt_port_reads_the_port_from_the_url() -> None:
    assert de.local_bolt_port("bolt://localhost:7687") == 7687
    assert de.local_bolt_port("bolt://127.0.0.1:7999") == 7999
    assert de.local_bolt_port("bolt://localhost") == 7687


def test_a_remote_database_is_never_bootstrapped() -> None:
    """Starting a local container for a remote URL would shadow a real database with an
    empty one — silently, and with the user's graph nowhere in sight."""
    assert de.local_bolt_port("bolt://memgraph.internal:7687") is None
    assert de.local_bolt_port("bolt://10.0.0.5:7687") is None


def test_run_argv_binds_loopback_and_keeps_a_named_volume() -> None:
    argv = de.memgraph_run_argv(7687)
    assert "-p" in argv and "127.0.0.1:7687:7687" in argv, "the graph holds source code; do not publish it"
    assert f"{de.MEMGRAPH_VOLUME}:/var/lib/memgraph" in argv, "without a volume, docker rm discards the index"
    assert de.MEMGRAPH_IMAGE in argv
    assert "memgraph-mage" in de.MEMGRAPH_IMAGE, "vector indexes do not exist on plain memgraph"


def test_ensure_memgraph_is_a_noop_when_the_port_already_answers(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: True)
    monkeypatch.setattr(de, "docker_cli_present", lambda: False)  # must not even be consulted
    assert de.ensure_memgraph("bolt://localhost:7687") is True


def test_ensure_memgraph_gives_up_quietly_without_docker(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: False)
    assert de.ensure_memgraph("bolt://localhost:7687") is False


def test_ensure_memgraph_starts_an_existing_stopped_container(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: True)
    monkeypatch.setattr(de, "container_state", lambda name: "exited")
    monkeypatch.setattr(de, "_wait_for_port", lambda port, seconds: True)
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is True
    # `docker start`, not `docker run` — a second container on the same volume would fail.
    assert ran == [["docker", "start", de.MEMGRAPH_CONTAINER]]


def test_ensure_memgraph_creates_the_container_when_absent(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: True)
    monkeypatch.setattr(de, "container_state", lambda name: None)
    monkeypatch.setattr(de, "_wait_for_port", lambda port, seconds: True)
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is True
    assert ran and ran[0][:3] == ["docker", "run", "-d"]
