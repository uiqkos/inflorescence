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


def test_stop_grace_period_allows_a_full_exit_snapshot() -> None:
    from pathlib import Path

    argv = de.memgraph_run_argv(7687)
    timeout = argv[argv.index("--stop-timeout") + 1]
    assert int(timeout) >= 600, "the observed multi-GB snapshot cannot finish in Docker's default 10s"
    compose = (Path(__file__).parents[1] / "docker-compose.yml").read_text()
    assert f"stop_grace_period: {timeout}s" in compose


def test_manual_stop_mode_never_calls_docker_or_launches_its_app(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("autostart is disabled; even consulting Docker is unnecessary")

    monkeypatch.setattr(de, "docker_cli_present", forbidden)
    monkeypatch.setattr(de, "ensure_docker_daemon", forbidden)
    monkeypatch.setattr(de, "_run", forbidden)
    assert de.ensure_memgraph("bolt://localhost:7687", auto_start=False) is False


def test_manual_stop_mode_can_use_an_already_running_database(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: True)
    assert de.ensure_memgraph("bolt://localhost:7687", auto_start=False) is True


def test_docker_available_requires_a_server_version_not_just_exit_zero(monkeypatch) -> None:
    """`docker info` exits 0 even with the daemon down (the error goes to stderr), so the
    old exit-code check reported a dead daemon as available — and the daemon-launch path
    this exists to trigger never ran. Only a non-empty ServerVersion proves an answer."""
    monkeypatch.setattr(de, "docker_cli_present", lambda: True)
    monkeypatch.setattr(de, "_run", lambda argv, **k: "\n")
    assert de.docker_available() is False
    monkeypatch.setattr(de, "_run", lambda argv, **k: "28.0.4\n")
    assert de.docker_available() is True


def test_ensure_memgraph_is_a_noop_when_the_port_already_answers(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: True)
    monkeypatch.setattr(de, "docker_cli_present", lambda: False)  # must not even be consulted
    assert de.ensure_memgraph("bolt://localhost:7687") is True


def test_ensure_memgraph_gives_up_quietly_without_docker(monkeypatch) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: False)
    assert de.ensure_memgraph("bolt://localhost:7687") is False


def _stub_started_daemon(monkeypatch, *, image_local: bool = True) -> None:
    monkeypatch.setattr(de, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: True)
    monkeypatch.setattr(de, "ensure_docker_daemon", lambda *a, **k: True)
    monkeypatch.setattr(de, "image_present", lambda image: image_local)
    monkeypatch.setattr(de, "_wait_for_port", lambda port, seconds: True)
    # No rival container over the volume: the ordinary machine. The tests that care about
    # one say so themselves, and the rest get to assert on the docker calls they mean.
    monkeypatch.setattr(de, "containers_on_volume", lambda *a, **k: [])


def test_ensure_memgraph_starts_an_existing_stopped_container(monkeypatch) -> None:
    _stub_started_daemon(monkeypatch)
    monkeypatch.setattr(de, "container_state", lambda name: "exited")
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv, **k: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is True
    # `docker start`, not `docker run` — a second container on the same volume would fail.
    assert ran == [["docker", "start", de.MEMGRAPH_CONTAINER]]


def test_ensure_memgraph_creates_the_container_when_absent(monkeypatch) -> None:
    _stub_started_daemon(monkeypatch)
    monkeypatch.setattr(de, "container_state", lambda name: None)
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv, **k: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is True
    assert ran and ran[0][:3] == ["docker", "run", "-d"]


def test_ensure_memgraph_pulls_the_image_before_the_first_run(monkeypatch) -> None:
    """The memgraph-mage image is >1 GB. `docker run` under the 30s call timeout was
    killed mid-pull on every fresh machine — the pull must be its own, patient step."""
    _stub_started_daemon(monkeypatch, image_local=False)
    monkeypatch.setattr(de, "container_state", lambda name: None)
    pulled: list[str] = []
    monkeypatch.setattr(de, "pull_image", lambda image, **k: pulled.append(image) or True)
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv, **k: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is True
    assert pulled == [de.MEMGRAPH_IMAGE]
    assert ran and ran[0][:3] == ["docker", "run", "-d"]


def test_ensure_memgraph_gives_up_when_the_pull_fails(monkeypatch) -> None:
    _stub_started_daemon(monkeypatch, image_local=False)
    monkeypatch.setattr(de, "container_state", lambda name: None)
    monkeypatch.setattr(de, "pull_image", lambda image, **k: False)
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv, **k: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is False
    assert ran == []  # no half-started container on a missing image


def test_ensure_memgraph_gives_up_when_the_daemon_cannot_start(monkeypatch) -> None:
    """Docker Desktop installed but not running is the default state of a fresh laptop;
    when even launching it fails, the answer is False — not a docker error cascade."""
    monkeypatch.setattr(de, "port_open", lambda *a, **k: False)
    monkeypatch.setattr(de, "docker_cli_present", lambda: True)
    monkeypatch.setattr(de, "ensure_docker_daemon", lambda *a, **k: False)
    consulted: list[str] = []
    monkeypatch.setattr(de, "container_state", lambda name: consulted.append(name) or None)

    assert de.ensure_memgraph("bolt://localhost:7687") is False
    assert consulted == []


def test_containers_on_volume_reads_name_and_state(monkeypatch) -> None:
    monkeypatch.setattr(
        de, "_run", lambda argv, **k: "inflorescence-memgraph\trunning\ninflorescence-memgraph-1\texited\n"
    )
    assert de.containers_on_volume() == [
        ("inflorescence-memgraph", "running"),
        ("inflorescence-memgraph-1", "exited"),
    ]


def test_containers_on_volume_asks_docker_for_stopped_ones_too(monkeypatch) -> None:
    """A stopped rival still counts: `docker start` on it restarts the fight. That is also
    why the filter is `volume=` — docker matches `publish=` only against running containers,
    so the port filter would call a machine clean while the duplicate sat there waiting."""
    seen: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv, **k: seen.append(argv) or "")

    de.containers_on_volume()
    assert seen and "-a" in seen[0]
    assert f"volume={de.MEMGRAPH_VOLUME}" in seen[0]


def test_ensure_memgraph_adopts_a_rival_container_instead_of_making_a_second(monkeypatch) -> None:
    """`docker compose up` in a checkout of this repository creates the database under its
    own name. Creating ours beside it does not add a database, it splits one: two Memgraphs
    over one volume fight for the bolt port and the loser is left running yet unreachable."""
    _stub_started_daemon(monkeypatch)
    monkeypatch.setattr(de, "container_state", lambda name: None)
    monkeypatch.setattr(de, "containers_on_volume", lambda *a, **k: [("inflorescence-memgraph-1", "exited")])
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv, **k: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is True
    assert ran == [["docker", "start", "inflorescence-memgraph-1"]]


def test_ensure_memgraph_only_waits_on_a_rival_that_already_runs(monkeypatch) -> None:
    _stub_started_daemon(monkeypatch)
    monkeypatch.setattr(de, "container_state", lambda name: None)
    monkeypatch.setattr(de, "containers_on_volume", lambda *a, **k: [("inflorescence-memgraph-1", "running")])
    ran: list[list[str]] = []
    monkeypatch.setattr(de, "_run", lambda argv, **k: ran.append(argv) or "")

    assert de.ensure_memgraph("bolt://localhost:7687") is True
    assert ran == []


def test_recovery_hint_names_the_rival_container(monkeypatch) -> None:
    monkeypatch.setattr(de, "containers_on_volume", lambda *a, **k: [("inflorescence-memgraph-1", "running")])
    hint = de.recovery_hint()
    assert "inflorescence-memgraph-1" in hint
    assert "docker stop" in hint


def test_recovery_hint_tells_a_networkless_container_apart_from_a_slow_one(monkeypatch) -> None:
    """Both refuse connections; only one is fixed by waiting. A container attached to no
    network publishes nothing and never will — restarting does not reattach it."""
    monkeypatch.setattr(de, "containers_on_volume", lambda *a, **k: [])
    monkeypatch.setattr(de, "container_is_networkless", lambda name: True)
    assert "docker rm" in de.recovery_hint()

    monkeypatch.setattr(de, "container_is_networkless", lambda name: False)
    slow = de.recovery_hint()
    assert "docker rm" not in slow, "a database still replaying its snapshot must not be recreated"
    assert "docker logs" in slow


def test_container_is_networkless_only_on_a_zero_network_count(monkeypatch) -> None:
    monkeypatch.setattr(de, "_run", lambda argv, **k: "0\n")
    assert de.container_is_networkless("x") is True
    monkeypatch.setattr(de, "_run", lambda argv, **k: "1\n")
    assert de.container_is_networkless("x") is False
    monkeypatch.setattr(de, "_run", lambda argv, **k: None)  # container gone
    assert de.container_is_networkless("x") is False


def test_bolt_budget_outlasts_snapshot_replay() -> None:
    """memgraph-mage was measured opening bolt ~100s after start on a 187k-node graph.
    The old 60s budget reported a database that was merely busy starting as unreachable."""
    assert de._BOLT_OPEN_TIMEOUT >= 150.0
