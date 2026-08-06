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
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

MEMGRAPH_IMAGE = "memgraph/memgraph-mage:latest"
MEMGRAPH_CONTAINER = "inflorescence-memgraph"
MEMGRAPH_VOLUME = "inflorescence_mg_data"

_DOCKER_CALL_TIMEOUT = 30.0
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


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
    return _run(["docker", "info", "--format", "{{.ServerVersion}}"]) is not None


def _run(argv: list[str]) -> str | None:
    """Run *argv*, returning stdout on success and None on any failure."""
    try:
        proc = subprocess.run(  # noqa: S603 — argv is built here, never from user input
            argv, capture_output=True, text=True, timeout=_DOCKER_CALL_TIMEOUT, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("docker call %r failed: %s", argv[:2], exc)
        return None
    if proc.returncode != 0:
        logger.debug("docker call %r exited %d: %s", argv[:2], proc.returncode, proc.stderr.strip()[-300:])
        return None
    return proc.stdout


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


def memgraph_run_argv(port: int) -> list[str]:
    """``docker run`` for the Memgraph container.

    Published on loopback only: the graph holds the source of every indexed repository, and
    docker's default of binding 0.0.0.0 would hand it to anyone on the same network. The
    named volume is what makes the graph outlive the container — without it, a `docker rm`
    silently discards every index the user paid an LLM to build.
    """
    return [
        "docker", "run", "-d",
        "--name", MEMGRAPH_CONTAINER,
        "--restart", "unless-stopped",
        "-p", f"127.0.0.1:{port}:7687",
        "-v", f"{MEMGRAPH_VOLUME}:/var/lib/memgraph",
        MEMGRAPH_IMAGE,
        "--storage-snapshot-on-exit=true",
        "--storage-snapshot-interval-sec=300",
    ]


def container_state(name: str) -> str | None:
    """Docker's status word for *name* ("running", "exited", ...), or None if absent."""
    out = _run(["docker", "inspect", "-f", "{{.State.Status}}", name])
    return out.strip() if out and out.strip() else None


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


def ensure_memgraph(memgraph_url: str, *, wait_seconds: float = 60.0) -> bool:
    """Make a local Memgraph reachable, starting the container if needed.

    Returns True when the bolt port accepts connections. Fast path — an already-listening
    port — costs one loopback connect, so this is cheap enough to call on every startup.
    """
    port = local_bolt_port(memgraph_url)
    if port is None:
        return False
    if port_open(port):
        return True
    if not docker_cli_present():
        return False

    state = container_state(MEMGRAPH_CONTAINER)
    if state is None:
        argv = memgraph_run_argv(port)
    elif state == "running":
        argv = None  # up but not yet listening — only wait
    else:
        argv = ["docker", "start", MEMGRAPH_CONTAINER]

    if argv is not None:
        logger.info("Starting Memgraph (%s)", MEMGRAPH_CONTAINER)
        if _run(argv) is None:
            logger.warning(
                "Could not start the Memgraph container. Is Docker running? "
                "Start it yourself with: %s", " ".join(memgraph_run_argv(port))
            )
            return False

    started = _wait_for_port(port, wait_seconds)
    if not started:
        logger.warning("Memgraph container started but port %d never opened within %.0fs", port, wait_seconds)
    return started
