"""Docker-backed dependencies, started on demand.

Inflorescence needs a graph database, and optionally three language indexers. Every one of
them is a container, and asking a person to start them by hand is the whole difference
between "one command" and "an afternoon" — worse, the MCP path has no shell in front of it
where such a command could even be typed: a client launches ``inflorescence serve``
directly, so an unreachable database can only be reported, never fixed, by the user at that
moment.

Nothing here is required. Every function degrades to False rather than raising, and the
caller falls back to the behavior it had before: an error message, or the syntactic ladder.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MEMGRAPH_IMAGE = "memgraph/memgraph-mage:latest"
MEMGRAPH_CONTAINER = "inflorescence-memgraph"
MEMGRAPH_VOLUME = "inflorescence_mg_data"

_DOCKER_CALL_TIMEOUT = 30.0
# `docker run`/`docker start` on a daemon that just booted can take well over the
# 30s that suffices for an inspect.
_DOCKER_START_TIMEOUT = 120.0
# The memgraph-mage image is >1 GB. Pulling it inside `docker run` under the 30-second
# call timeout is what made every first run on a fresh machine "fail to start Memgraph":
# the CLI was killed mid-pull, silently, every time. The pull is therefore its own step
# with a budget measured in minutes, not seconds.
_PULL_TIMEOUT = 1800.0
# memgraph-mage opens bolt only after loading its query modules and replaying the last
# snapshot. Measured at ~100s on a 187k-node graph, so the old 60s budget gave up on a
# database that was merely busy starting — and reported it as unreachable.
_BOLT_OPEN_TIMEOUT = 180.0
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}

# macOS has no native dockerd — the daemon lives inside one of these apps, and "installed
# but not launched" is the default state right after installation. Checked in order;
# OrbStack first because it boots in seconds and needs no license click-through.
_MACOS_DOCKER_APPS = ("OrbStack", "Docker")


def docker_cli_present() -> bool:
    """True when a ``docker`` executable exists. Says nothing about the daemon."""
    return shutil.which("docker") is not None


def docker_available() -> bool:
    """True when the docker CLI exists *and* its daemon answers.

    Distinct from :func:`docker_cli_present` because "Docker Desktop is installed but not
    running" is the common state on a laptop, and it fails very differently from "no docker".
    """
    if not docker_cli_present():
        return False
    # `docker info` exits 0 even when the daemon is unreachable (the error goes to
    # stderr); only a non-empty ServerVersion on stdout proves the daemon answered.
    out = _run(["docker", "info", "--format", "{{.ServerVersion}}"])
    return bool(out and out.strip())


def _run(argv: list[str], *, timeout: float = _DOCKER_CALL_TIMEOUT) -> str | None:
    """Run *argv*, returning stdout on success and None on any failure."""
    try:
        proc = subprocess.run(  # noqa: S603 — argv is built here, never from user input
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("docker call %r failed: %s", argv[:2], exc)
        return None
    if proc.returncode != 0:
        logger.debug("docker call %r exited %d: %s", argv[:2], proc.returncode, proc.stderr.strip()[-300:])
        return None
    return proc.stdout


def docker_desktop_app() -> str | None:
    """The name of an installed macOS Docker app ("OrbStack" or "Docker"), or None."""
    if sys.platform != "darwin":
        return None
    for name in _MACOS_DOCKER_APPS:
        for base in (Path("/Applications"), Path.home() / "Applications"):
            if (base / f"{name}.app").exists():
                return name
    return None


def ensure_docker_daemon(wait_seconds: float = 90.0) -> bool:
    """Make the docker daemon answer, launching its macOS app if that is what it takes.

    "Docker Desktop is installed but not running" is the normal state of a laptop that
    just installed it — and the state every fresh `init` used to fail in. `open -g` keeps
    the app from stealing focus; the daemon is polled until it answers or the budget runs out.
    """
    if not docker_cli_present():
        return False
    if docker_available():
        return True
    app = docker_desktop_app()
    if app is None:
        return False
    logger.info("Docker daemon is not running; launching %s", app)
    if _run(["open", "-g", "-a", app]) is None:
        return False
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if docker_available():
            return True
        time.sleep(2.0)
    logger.warning("%s was launched but the docker daemon did not answer within %.0fs", app, wait_seconds)
    return False


def image_present(image: str) -> bool:
    """True when *image* exists locally, so `docker run` will not touch the network."""
    return _run(["docker", "image", "inspect", image]) is not None


def pull_image(image: str, *, timeout: float = _PULL_TIMEOUT) -> bool:
    """Pull *image*, streaming progress to stderr.

    stderr, never stdout: under `serve` stdout carries the MCP protocol, and a progress
    bar there corrupts the session. Under pytest capture stderr may have no real fd; the
    progress is then simply dropped.
    """
    try:
        out = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        out = subprocess.DEVNULL
    try:
        proc = subprocess.run(  # noqa: S603
            ["docker", "pull", image], stdout=out, stderr=out, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("docker pull %s failed: %s", image, exc)
        return False
    return proc.returncode == 0


def local_bolt_port(memgraph_url: str) -> int | None:
    """The port to bootstrap for *memgraph_url*, or None when it is not a local database.

    A URL pointing at someone else's host is never something to start a container for; that
    would silently shadow a remote database with an empty local one.
    """
    try:
        parsed = urlparse(memgraph_url)
    except ValueError:
        return None
    if parsed.hostname is None or parsed.hostname not in _LOOPBACK_HOSTS:
        return None
    return parsed.port or 7687


# Storage flags, kept byte-identical with the `command:` block in docker-compose.yml at the
# repository root. The two start the same container over the same named volume — `init` goes
# through compose, `serve` runs docker directly from anywhere — so a flag changed in one place
# and not the other means the machine behaves differently depending on which path last
# started it, which is indistinguishable from a bug in the database. Compose carries the long
# rationale; the short version is that a snapshot dumps the whole graph, this graph is mostly
# 1536-float embeddings, and at 190k vectors that dump is 5.6 GB. The old five-minute interval
# therefore rewrote 5.6 GB roughly every five minutes, forever, because the watch daemon's
# 30-second lease heartbeat guarantees the "anything committed since the last snapshot?" test
# is always true. Durability comes from the WAL and from snapshot-on-exit, not from this
# interval; the interval only bounds how much WAL a cold start replays.
MEMGRAPH_STORAGE_ARGS = [
    "--storage-snapshot-on-exit=true",
    "--storage-snapshot-interval-sec=21600",
    "--storage-snapshot-retention-count=2",
]


def memgraph_run_argv(port: int) -> list[str]:
    """``docker run`` for the Memgraph container.

    Published on loopback only: the graph holds the source of every indexed repository, and
    docker's default of binding 0.0.0.0 would hand it to anyone on the same network. The
    named volume is what makes the graph outlive the container — without it, a `docker rm`
    silently discards every index the user paid an LLM to build.

    Storage flags come from :data:`MEMGRAPH_STORAGE_ARGS`, which compose duplicates verbatim.
    """
    return [
        "docker", "run", "-d",
        "--name", MEMGRAPH_CONTAINER,
        "--restart", "unless-stopped",
        "-p", f"127.0.0.1:{port}:7687",
        "-v", f"{MEMGRAPH_VOLUME}:/var/lib/memgraph",
        MEMGRAPH_IMAGE,
        *MEMGRAPH_STORAGE_ARGS,
    ]


def container_state(name: str) -> str | None:
    """Docker's status word for *name* ("running", "exited", ...), or None if absent."""
    out = _run(["docker", "inspect", "-f", "{{.State.Status}}", name])
    return out.strip() if out and out.strip() else None


def containers_on_volume(volume: str = MEMGRAPH_VOLUME) -> list[tuple[str, str]]:
    """Every container mounting *volume*, running or not, as ``(name, state)``.

    Two containers over one Memgraph data directory is the failure this exists to prevent.
    They fight for the same snapshot and the same published port, and the loser is not
    merely stopped: it can be left *running* yet attached to no network, which no amount of
    `docker start` recovers from. Stopped containers count — starting the wrong one restarts
    the fight — and that is why the filter is `volume=` and not `publish=`: docker matches
    a published port only on containers that are currently up.
    """
    out = _run(["docker", "ps", "-a", "--filter", f"volume={volume}", "--format", "{{.Names}}\t{{.State}}"])
    if not out:
        return []
    found: list[tuple[str, str]] = []
    for line in out.splitlines():
        name, _, state = line.partition("\t")
        if name.strip():
            found.append((name.strip(), state.strip()))
    return found


def container_is_networkless(name: str) -> bool:
    """True when *name* is running but attached to no network — so it publishes nothing.

    Observed under OrbStack after a second container claimed the same host port: `docker
    inspect` keeps saying "running", `docker port` prints nothing, and every bolt connection
    dies with an incomplete handshake. `docker restart` does not reattach it; only recreating
    the container does, which is why this is reported rather than repaired here.
    """
    out = _run(["docker", "inspect", "-f", "{{len .NetworkSettings.Networks}}", name])
    return out is not None and out.strip() == "0"


def port_open(port: int, *, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(port: int, deadline_seconds: float) -> bool:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if port_open(port):
            return True
        time.sleep(0.5)
    return False


def ensure_memgraph(memgraph_url: str, *, wait_seconds: float = _BOLT_OPEN_TIMEOUT) -> bool:
    """Make a local Memgraph reachable, starting the container if needed.

    Returns True when the bolt port accepts connections. Fast path — an already-listening
    port — costs one loopback connect, so this is cheap enough to call on every startup.

    Never creates a *second* container for a database that already has one. The container
    this manages by name is only half the story: `docker compose up` in a checkout of this
    repository makes its own, under its own name, over the same named volume — and two
    Memgraphs over one data directory is a broken database, not a spare.
    """
    port = local_bolt_port(memgraph_url)
    if port is None:
        return False
    if port_open(port):
        return True
    if not docker_cli_present():
        return False
    if not ensure_docker_daemon():
        logger.warning(
            "Docker is installed but its daemon is not running and could not be started. "
            "Start Docker Desktop (or OrbStack) and retry."
        )
        return False

    container = MEMGRAPH_CONTAINER
    state = container_state(container)
    if state is None:
        # No container under our name — but another may already hold this database.
        rivals = [(name, rival) for name, rival in containers_on_volume() if name != container]
        if rivals:
            container, state = rivals[0]
            logger.info("Reusing the existing Memgraph container %s (%s)", container, state)

    if state is None:
        # Pull explicitly before `docker run`: the run call has a timeout sized for
        # starting a container, not for downloading a multi-gigabyte image.
        if not image_present(MEMGRAPH_IMAGE):
            logger.info("Pulling %s — first run only, this is >1 GB and can take a few minutes", MEMGRAPH_IMAGE)
            if not pull_image(MEMGRAPH_IMAGE):
                logger.warning("Could not pull %s. Check the network and retry.", MEMGRAPH_IMAGE)
                return False
        argv = memgraph_run_argv(port)
    elif state == "running":
        argv = None  # up but not yet listening — only wait
    else:
        argv = ["docker", "start", container]

    if argv is not None:
        logger.info("Starting Memgraph (%s)", container)
        if _run(argv, timeout=_DOCKER_START_TIMEOUT) is None:
            logger.warning(
                "Could not start the Memgraph container. Is Docker running? "
                "Start it yourself with: %s", " ".join(memgraph_run_argv(port))
            )
            return False

    started = _wait_for_port(port, wait_seconds)
    if not started:
        logger.warning(
            "Memgraph container %s is up but port %d never opened within %.0fs. %s",
            container, port, wait_seconds, recovery_hint(container, port),
        )
    return started


def recovery_hint(container: str = MEMGRAPH_CONTAINER, port: int = 7687) -> str:
    """Why a running Memgraph container is unreachable, phrased as what to do about it.

    Three states look identical from a refused connection, and only one of them is fixed by
    waiting: still recovering its snapshot, still fighting a rival container, or already
    orphaned onto no network. Naming the right one is the difference between a user who
    waits another minute and one who restarts docker for an hour.
    """
    rivals = [name for name, _ in containers_on_volume() if name != container]
    if rivals:
        return (
            f"Another container holds the same database volume ({', '.join(rivals)}). "
            f"Two Memgraphs over one data directory fight for port {port} and neither wins reliably — "
            f"stop the spare: docker stop {rivals[0]}"
        )
    if container_is_networkless(container):
        return (
            f"{container} is running but attached to no network, so it publishes nothing "
            f"(`docker port {container}` prints nothing). Restarting does not reattach it; "
            f"recreate it — the graph lives in the {MEMGRAPH_VOLUME} volume and survives:\n"
            f"  docker rm -f {container} && " + " ".join(memgraph_run_argv(port))
        )
    return (
        f"The image loads its query modules and replays the last snapshot before opening bolt, "
        f"which takes ~2 minutes on a populated graph. Watch it finish: docker logs -f {container}"
    )
