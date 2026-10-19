"""Entry point: python -m inflorescence."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inflorescence.config import Settings

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
# Cap the log at 40 MB total. Unrotated, this file reached 80 MB in ordinary development use;
# an MCP server the user never explicitly stops has no natural moment to truncate it.
_LOG_MAX_BYTES = 10 * 1024 * 1024
_LOG_BACKUP_COUNT = 3
_LOG_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _default_log_file() -> Path:
    """Return the per-user log path for this platform.

    Deriving the path from ``__file__`` (as this once did) only works for an editable
    ``src/`` checkout: for an installed wheel the same arithmetic lands inside
    ``site-packages``, where a system-wide install is root-owned — so opening the log raised
    PermissionError and *every* CLI command died at startup.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "inflorescence" / "inflorescence.log"


_LOG_FILE = str(_default_log_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inflorescence — code intelligence")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", default=_LOG_FILE, help=f"Log file path (default: {_LOG_FILE})")
    sub = parser.add_subparsers(dest="command")

    # `inflorescence serve` — MCP server (default)
    srv = sub.add_parser("serve", help="Start MCP server over stdio")
    srv.add_argument(
        "--allow-degraded",
        action="store_true",
        help="Start even if Memgraph is unreachable (DB-backed tools will fail at call time)",
    )

    # `inflorescence index <path>` — index a repo into Memgraph
    idx = sub.add_parser("index", help="Index a directory into Memgraph")
    idx.add_argument("path", help="Path to the repository to index")
    idx.add_argument(
        "--include",
        action="append",
        dest="include_patterns",
        default=None,
        help="Relative path or glob to include. Repeatable. If omitted, include all supported files.",
    )
    idx.add_argument(
        "--exclude",
        action="append",
        dest="exclude_patterns",
        default=None,
        help="Relative path, directory name, or glob to exclude. Repeatable.",
    )
    idx.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=False,
        help="Estimate volume/cost without indexing (no DB writes, no LLM calls).",
    )
    idx.add_argument(
        "--max-cost",
        type=float,
        dest="max_cost",
        default=None,
        metavar="USD",
        help="Abort the index if the estimated cost exceeds this many dollars. A first index is\n"
             "priced over the whole tree, a re-index over only the files that changed since the\n"
             "last run.",
    )
    idx.add_argument(
        "--force",
        action="store_true",
        dest="force_rebuild",
        default=False,
        help=(
            "Re-parse and re-resolve even when no file changed. Needed after an upgrade that "
            "improves parsing or edge resolution: the incremental path compares content "
            "checksums, so a better indexer reaches an unchanged file only when a run re-parses "
            "the tree anyway. Summaries and embeddings are reused by content hash; what it pays "
            "for is the summaries of entities the upgrade newly recognizes (or now reads a "
            "different signature or docstring for) and of the containers above them."
        ),
    )

    # `inflorescence prune` — narrow an indexed project's scope, explicitly
    prune = sub.add_parser(
        "prune",
        help="Delete what an indexed project holds for paths a narrowed scope no longer covers",
        description=(
            "Indexing never deletes an excluded file's nodes: the file is still on disk, so a "
            "re-index cannot tell a narrowed scope from a failed scan. This command is that "
            "narrowing, made explicit. It costs nothing — no scan, no parse, no LLM — and it "
            "merges the new excludes into the project's stored scan scope so the watcher stops "
            "re-admitting them."
        ),
    )
    prune.add_argument("path", help="Path to the indexed repository")
    prune.add_argument(
        "--exclude",
        action="append",
        dest="exclude_patterns",
        default=None,
        help="Relative path, directory name, or glob to drop from the index. Repeatable.",
    )
    prune.add_argument(
        "--include",
        action="append",
        dest="include_patterns",
        default=None,
        help=(
            "Replace the project's include scope; anything it does not cover is pruned. "
            "Repeatable. Omit to keep the stored includes."
        ),
    )
    prune.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Report what would be deleted and delete nothing.",
    )

    # `inflorescence watch` — the daemon that keeps indexes current between sessions
    watch = sub.add_parser(
        "watch",
        help="Keep indexed projects current from a process that outlives your editor session",
        description=(
            "An index is a snapshot. The watcher that `index_directory` starts lives inside "
            "the MCP server process of one session and dies with it, so by the next morning "
            "the graph describes yesterday's code while every tool answers in the present "
            "tense. This daemon is the process that is supposed to be running: it follows the "
            "projects you register, reconciles each one against disk when it starts, and "
            "re-indexes what you save. Registration is per project and explicit — following "
            "every tree that was ever indexed would spend money summarizing branches nobody "
            "reads."
        ),
    )
    watch_sub = watch.add_subparsers(dest="watch_command")

    w_add = watch_sub.add_parser(
        "add", help="Register an already-indexed project for the daemon to follow"
    )
    w_add.add_argument("path", help="Path to the indexed repository")

    w_rm = watch_sub.add_parser("rm", help="Stop following a project (its index is left alone)")
    w_rm.add_argument("path", help="Path to the repository")

    w_status = watch_sub.add_parser(
        "status",
        help="Which projects are registered, which are actually being followed, and by whom",
    )
    w_status.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Limit the report to one repository (default: every indexed project)",
    )
    w_status.add_argument(
        "--stale",
        action="store_true",
        help=(
            "Also re-hash every indexed file to report how far each project has drifted from "
            "disk. Authoritative, and the slow part: seconds per thousand files."
        ),
    )

    watch_sub.add_parser(
        "run", help="Run the daemon in the foreground (this is what the LaunchAgent runs)"
    )
    watch_sub.add_parser(
        "install", help="Install and start a launchd agent that runs the daemon at login (macOS)"
    )
    watch_sub.add_parser("uninstall", help="Stop and remove the launchd agent (macOS)")

    # `inflorescence feedback` — read what consuming agents reported about the MCP tools
    fb = sub.add_parser(
        "feedback",
        help="Show what agents reported through the send_feedback MCP tool",
    )
    fb.add_argument(
        "--limit", type=int, default=20, help="How many of the newest entries to show (0 = all)"
    )
    fb.add_argument("--kind", help="Show only entries of this kind (bug, wrong_result, …)")
    fb.add_argument("--json", action="store_true", help="Print the raw entries as JSON")
    fb.add_argument("--path", action="store_true", help="Print the log file path and exit")

    # `inflorescence dashboard` — local web UI over the indexed graphs
    dash = sub.add_parser(
        "dashboard",
        help="Start the web dashboard (overview, explorer, specimen graph, Cypher, MCP tools)",
    )
    dash.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    dash.add_argument("--port", type=int, default=8321, help="Port (default: 8321)")
    dash.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")

    # `inflorescence init` — interactive onboarding

    sub.add_parser("init", help="Interactive setup: .env, Memgraph, MCP registration, doctor")

    # `inflorescence doctor` — health checklist
    sub.add_parser("doctor", help="Check config, API key, Memgraph, and schema")

    # `inflorescence build-images` — build the SCIP indexer images from the packaged Dockerfiles
    bld = sub.add_parser(
        "build-images",
        help="Build the SCIP indexer images locally from the Dockerfiles shipped in this package",
    )
    bld.add_argument("--tag", default="v1", help="Image tag (default: v1)")
    bld.add_argument("--registry", default="ghcr.io/uiqkos", help="Registry prefix (default: ghcr.io/uiqkos)")
    bld.add_argument(
        "--language",
        choices=["go", "node", "all"],
        default="all",
        help="Which image to build: go (scip-go), node (scip-typescript + scip-python), all (default)",
    )

    return parser


def _resolve_log_level(*, debug: bool) -> int:
    """Resolve the root log level from ``--debug``, else the ``LOG_LEVEL`` setting.

    ``LOG_LEVEL`` was documented and declared on ``Settings`` but had no consumer, so setting
    it did nothing. Logging has to come up before configuration errors can be reported, so a
    Settings failure here degrades to INFO rather than propagating.
    """
    if debug:
        return logging.DEBUG
    try:
        from inflorescence.config import Settings

        name = Settings().log_level
    except Exception:
        return logging.INFO
    return _LOG_LEVELS.get(str(name).strip().upper(), logging.INFO)


def _build_log_handler(path: str) -> logging.Handler | None:
    """Return a size-rotating file handler for *path*, or None if it cannot be opened.

    An unwritable log directory must not be fatal: the log is a diagnostic, and refusing to
    start over it would take down the MCP server for a read-only or misconfigured HOME.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            path, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding="utf-8"
        )
    except OSError as exc:
        print(f"Warning: cannot write log file {path!r} ({exc}); continuing without file logging.", file=sys.stderr)
        return None
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    return handler


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    level = _resolve_log_level(debug=args.debug)
    is_serve = args.command is None or args.command == "serve"

    file_handler = _build_log_handler(args.log_file)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if file_handler is not None:
        root_logger.addHandler(file_handler)
    if not is_serve:
        # CLI commands log to stderr as well; the MCP server cannot, because it speaks
        # protocol over stdio.
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root_logger.addHandler(stderr_handler)
        if file_handler is not None:
            print(f"Logging to {args.log_file}", file=sys.stderr)

    if args.command == "index":
        if args.dry_run:
            _preview(args.path, args.include_patterns, args.exclude_patterns)
        else:
            asyncio.run(
                _index(
                    args.path,
                    args.include_patterns,
                    args.exclude_patterns,
                    max_cost=args.max_cost,
                    force_rebuild=args.force_rebuild,
                )
            )
    elif args.command == "prune":
        asyncio.run(
            _prune(
                args.path,
                args.include_patterns,
                args.exclude_patterns,
                dry_run=args.dry_run,
            )
        )
    elif args.command == "watch":
        if args.watch_command == "add":
            sys.exit(asyncio.run(_watch_register(args.path, watched=True)))
        elif args.watch_command == "rm":
            sys.exit(asyncio.run(_watch_register(args.path, watched=False)))
        elif args.watch_command == "status":
            sys.exit(asyncio.run(_watch_status(args.path, args.stale)))
        elif args.watch_command == "run":
            sys.exit(asyncio.run(_watch_run()))
        elif args.watch_command == "install":
            sys.exit(_cmd_watch_install())
        elif args.watch_command == "uninstall":
            sys.exit(_cmd_watch_uninstall())
        else:
            # `inflorescence watch` alone: the status report is the only harmless default,
            # and it is what someone typing the bare command is asking for.
            sys.exit(asyncio.run(_watch_status(None, False)))
    elif args.command == "feedback":
        sys.exit(_cmd_feedback(args))
    elif args.command == "dashboard":
        from inflorescence.dashboard.app import run_dashboard

        run_dashboard(host=args.host, port=args.port, open_browser=not args.no_browser)
    elif args.command == "init":
        sys.exit(_cmd_init(args))
    elif args.command == "build-images":
        sys.exit(_cmd_build_images(args))
    elif args.command == "doctor":
        from inflorescence.config import Settings

        try:
            settings = Settings()
        except Exception as exc:  # noqa: BLE001 — surface any config error as a failed check
            print(f"[FAIL] Config loads: {exc}", file=sys.stderr)
            sys.exit(1)
        sys.exit(_cmd_doctor(settings))
    else:
        _serve(allow_degraded=getattr(args, "allow_degraded", False))


def _ensure_service_database(settings: Settings, *, allow_degraded: bool = False) -> None:
    """Honor a manual database stop before importing the MCP/indexing stack."""
    from inflorescence.docker_env import ensure_memgraph, local_bolt_port

    # An MCP client launches this process directly — there is no shell in front of it where a
    # `docker compose up` could have been typed, and no way for the user to act on the error
    # below at the moment it appears. Bring the database up ourselves instead.
    reachable = ensure_memgraph(settings.memgraph_url, auto_start=settings.memgraph_autostart)
    if (
        not settings.memgraph_autostart
        and not reachable
        and local_bolt_port(settings.memgraph_url) is not None
        and not allow_degraded
    ):
        print(
            "Memgraph is stopped or unreachable; MEMGRAPH_AUTOSTART=false leaves it stopped.\n"
            "Start it explicitly with `docker start inflorescence-memgraph`, or set "
            "MEMGRAPH_AUTOSTART=true to restore automatic startup.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _serve(allow_degraded: bool = False) -> None:
    from inflorescence.config import Settings

    settings = Settings()
    _ensure_service_database(settings, allow_degraded=allow_degraded)

    from inflorescence.server import create_server, shutdown, startup

    mcp, conn, llm, watcher = create_server()

    async def run() -> None:
        healthy = await startup(conn, allow_degraded=allow_degraded)
        if not healthy and not allow_degraded:
            # This text is the *entire* diagnosis a user gets: exiting here is what the MCP
            # client reports as "connection closed", and stderr from a failed server is
            # somewhere they have to go digging for. Spend it on what is actually wrong with
            # this machine rather than on advice that fits any machine.
            print(
                "Error: Memgraph is not reachable and could not be started.\n"
                f"  {_memgraph_recovery_hint(settings)}\n"
                "Run `inflorescence doctor` for the full picture, or `--allow-degraded` to "
                "start the server anyway.",
                file=sys.stderr,
            )
            await shutdown(conn, llm, watcher)
            raise SystemExit(1)
        try:
            await mcp.run_stdio_async()
        finally:
            await shutdown(conn, llm, watcher)

    asyncio.run(run())


# ----------------------------------------------------------------------------------
# `inflorescence watch` — the daemon that keeps indexes current between sessions
# ----------------------------------------------------------------------------------


async def _watch_components():  # noqa: ANN201 — internal helper
    """Build the stack and verify Memgraph, or exit with the same diagnosis `serve` gives."""
    from inflorescence.config import Settings

    settings = Settings()
    _ensure_service_database(settings)

    from inflorescence.server import create_components

    components = create_components()
    if not await components.conn.health_check():
        print(
            "Error: Memgraph is not reachable.\n"
            f"  {_memgraph_recovery_hint(settings)}\n"
            "Run `inflorescence doctor` for the full picture.",
            file=sys.stderr,
        )
        await components.conn.close()
        sys.exit(1)
    return components


def _resolve_repo_path(path: str) -> Path:
    root = Path(path).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)
    return root


async def _watch_register(path: str, *, watched: bool) -> int:
    """Add or remove one project from the watch registry."""
    root = _resolve_repo_path(path)
    components = await _watch_components()
    try:
        project = components.manager.get_project(str(root))
        if watched and not await components.repo.project_is_indexed(project):
            # Registering an unindexed tree would promise something the daemon must not do:
            # a first index costs real money and stays an explicit, previewed act.
            print(
                f"Error: {root} has no index yet — nothing to keep current.\n"
                "  Index it first (the MCP tool index_directory, or `inflorescence index "
                f"{root}`), then run this again.",
                file=sys.stderr,
            )
            return 1
        if not await components.repo.set_project_watched(project, watched):
            print(f"Error: no indexed project found for {root}", file=sys.stderr)
            return 1
        state = await components.repo.get_watch_state(project)
        if watched:
            print(f"Watching {root} (project={project}).")
            # The machine-level fact, not this project's claim: a freshly registered project is
            # never claimed yet, so reading its own claim here announced "no daemon is running"
            # on a machine whose daemon was running fine and simply had not reached it.
            if state.get("daemon_live"):
                interval = int(components.settings.watch_refresh_seconds)
                print(f"  A daemon is running; it picks this up within {interval}s.")
            else:
                print("  No daemon is running yet — start one: `inflorescence watch run` "
                      "(or `inflorescence watch install` to run it at login).")
        else:
            print(f"No longer watching {root} (project={project}). Its index is untouched.")
            if state.get("live"):
                print("  The daemon following it releases it on its next registry pass.")
        return 0
    finally:
        await components.conn.close()


def _watch_status_line(
    row: dict, state: dict, pending: int | None, cost_usd: float | None = None
) -> str:
    """One project's line: who follows it, where it lives, how far it has drifted, at what price.

    The dollar figure is the one number that decides whether starting a daemon right now is a
    good idea. A project last indexed before the extension set widened can have hundreds of
    files that were never indexed at all, and its catch-up is the first thing a daemon does —
    so the estimate belongs *here*, before anyone starts one, not in a log afterwards.
    """
    if state.get("live"):
        mark, note = "WATCHED", f"daemon {state.get('holder') or '?'}"
    elif state.get("watched"):
        # Two very different reasons to be registered and unfollowed, and only one of them is
        # fixed by starting a daemon. The other — a daemon that is running and has not taken
        # this project — is either a registration made seconds ago or one it refused, and
        # printing "no daemon running" under a header that says one is running is nonsense.
        note = (
            "registered, not followed yet (the running daemon picks it up on its next pass)"
            if state.get("daemon_live")
            else "registered, no daemon running"
        )
        mark = "IDLE   "
    else:
        mark, note = "OFF    ", "not registered"
    line = f"{mark} {row['root_path']}"
    if pending is not None:
        if pending:
            price = f", ~${cost_usd:.2f} to catch up" if cost_usd is not None else ""
            line += f"  [{pending} file(s) to reindex{price}]"
        else:
            line += "  [current]"
    return f"{line}\n        {row['project']} — {note}"


async def _watch_status(path: str | None, show_stale: bool) -> int:
    """Report the registry, the daemon's actual grip on each project, and optionally drift."""
    components = await _watch_components()

    from inflorescence.cost import estimate_index_cost
    from inflorescence.db.lease import daemon_claim_key, read_lease

    try:
        # The machine-level fact first: per-project lines answer "is this tree followed", and
        # every one of them says "no" for the same single reason when the daemon is simply not
        # running. Saying that once, at the top, is the difference between a report and a list
        # of symptoms.
        held = await read_lease(components.conn, daemon_claim_key())
        if held and held.get("live"):
            print(f"Daemon: running on this machine ({held['holder']})\n")
        else:
            print(
                "Daemon: not running on this machine — start it with `inflorescence watch run`, "
                "or `inflorescence watch install` to run it at login\n"
            )
        rows = await components.repo.list_projects()
        if path is not None:
            wanted = str(_resolve_repo_path(path))
            rows = [r for r in rows if str(r.get("root_path")) == wanted]
            if not rows:
                print(f"No indexed project rooted at {wanted}", file=sys.stderr)
                return 1
        if not rows:
            print("No indexed projects yet.")
            return 0
        rows.sort(key=lambda r: str(r.get("root_path") or ""))
        for row in rows:
            project = str(row["project"])
            state = await components.repo.get_watch_state(project)
            pending: int | None = None
            cost_usd: float | None = None
            root = Path(str(row.get("root_path") or ""))
            if show_stale and root.is_dir():
                include, exclude = await components.repo.get_project_scope(project)
                changed = await components.builder.pending_change_paths(
                    project, root, include or None, exclude or None
                )
                pending = len(changed)
                if changed:
                    estimate = estimate_index_cost(
                        components.builder.estimate_paths(root, changed), components.settings
                    )
                    cost_usd = float(estimate["usd"])
            elif show_stale:
                print(f"MISSING {root}\n        {project} — root path is gone")
                continue
            print(_watch_status_line(row, state, pending, cost_usd))
        if not show_stale:
            print("\nPass --stale to re-hash every indexed file and report actual drift.")
        return 0
    finally:
        await components.conn.close()


async def _watch_run() -> int:
    """Run the daemon in the foreground until SIGINT/SIGTERM."""
    import signal

    components = await _watch_components()

    from inflorescence.project_manager import IndexPreflightError
    from inflorescence.server import shutdown, startup
    from inflorescence.watch_daemon import WatchDaemon, claim_singleton

    singleton = None
    try:
        # One daemon per machine. Refusing exits non-zero on purpose: the launchd job is
        # configured to retry a failed start (and not to restart a clean one), so a daemon
        # killed with SIGKILL is replaced as soon as its claim expires, instead of leaving the
        # machine unwatched until the next login.
        singleton, refusal = await claim_singleton(components.conn)
        if refusal is not None:
            print(f"Error: {refusal}", file=sys.stderr)
            return 1
        if not components.settings.llm_api_key.strip():
            # Otherwise this surfaces as a 401 from the pre-flight, which sends the reader
            # looking for a bad key rather than for a missing one. Under launchd the cause is
            # almost always the working directory: a project-local .env is invisible from `/`.
            from inflorescence.config import user_env_file

            print(
                f"Error: no OPENROUTER_API_KEY visible from {Path.cwd()}.\n"
                f"  Settings are read from {user_env_file()} and from a .env in the working "
                "directory. Run `inflorescence init` to write the per-user file.",
                file=sys.stderr,
            )
            return 1
        await startup(components.conn, repo=components.repo, settings=components.settings)
        try:
            # A daemon nobody is watching must fail at boot with one clear message rather than
            # come up "live", claim every project, and fail identically on every file saved.
            await components.manager.preflight_dependencies()
        except IndexPreflightError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        daemon = WatchDaemon(
            manager=components.manager,
            repo=components.repo,
            conn=components.conn,
            settings=components.settings,
            watcher=components.watcher,
        )
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, daemon.request_stop)
        print(
            "Watch daemon started. Registered projects are followed; Ctrl-C to stop.",
            flush=True,
        )
        await daemon.run()
        return 0
    finally:
        if singleton is not None:
            with contextlib.suppress(Exception):
                await singleton.__aexit__(None, None, None)
        await shutdown(components.conn, components.llm, components.watcher)


LAUNCH_AGENT_LABEL = "com.inflorescence.watch"


def launch_agent_path(label: str = LAUNCH_AGENT_LABEL) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def render_launch_agent(
    command: str,
    *,
    label: str = LAUNCH_AGENT_LABEL,
    path_env: str = "",
    working_dir: str = "",
) -> str:
    """The launchd job that runs the daemon at login and restarts it if it dies.

    ``PATH`` is written out rather than inherited on purpose: launchd hands a job a minimal
    ``/usr/bin:/bin:/usr/sbin:/sbin``, and the daemon shells out to ``docker`` to bring
    Memgraph up. Without the installing shell's PATH the agent starts, finds no docker, and
    fails in a log nobody is reading — the exact failure mode this whole feature exists to
    stop being invisible.

    ``WorkingDirectory`` is written for the same class of reason. Settings are read from a
    project-local ``.env`` as well as the per-user one, and on a machine where the key lives
    only in a checkout's ``.env`` — the normal shape when the MCP server is launched with
    ``--directory`` — a job started in launchd's ``/`` finds no key at all. It then fails its
    dependency pre-flight with a bare 401 and, being configured to retry failures, does so
    every ten seconds. Pinning the directory the install ran from makes the job load exactly
    what the installing shell loads.

    ``KeepAlive`` restarts a *failed* run only. Restarting unconditionally would spin: the
    daemon is a per-machine singleton, so a start that finds another daemon already running
    exits, and launchd would relaunch it every ten seconds forever. Tied to that, the refusal
    exits non-zero, so the one case that must self-heal does: a daemon killed with ``SIGKILL``
    leaves its claim to expire, and each throttled retry keeps failing until it can take over —
    rather than the machine sitting unwatched until the next login. A clean stop
    (``launchctl bootout``, Ctrl-C) exits zero and stays stopped.
    """
    log_dir = _default_log_file().parent
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{command}</string>
        <string>watch</string>
        <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>ThrottleInterval</key>
    <integer>10</integer>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{path_env}</string>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_dir / "watch.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{log_dir / "watch.err.log"}</string>
</dict>
</plist>
"""


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, check=False)  # noqa: S603, S607


async def _daemon_already_running() -> bool:
    """Is a live daemon holding this machine's claim? Never raises; unknown reads as no."""
    from inflorescence.config import Settings
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.lease import daemon_claim_key, read_lease

    conn = MemgraphConnection(Settings())
    try:
        held = await read_lease(conn, daemon_claim_key())
        return bool(held and held.get("live"))
    except Exception:  # noqa: BLE001 — a courtesy warning must never fail an install
        return False
    finally:
        with contextlib.suppress(Exception):
            await conn.close()


def _cmd_watch_install() -> int:
    """Write the LaunchAgent and start it, replacing any previous copy."""
    if sys.platform != "darwin":
        print(
            "Error: `watch install` sets up a launchd agent, which is macOS-only.\n"
            "  On Linux, run `inflorescence watch run` under systemd (a user unit with "
            "Restart=always) or any supervisor.",
            file=sys.stderr,
        )
        return 1
    # A daemon that cannot read a key is not an install worth making: it would fail its
    # pre-flight with a 401 and, being configured to retry failures, do it every ten seconds
    # in a log nobody opens. Check here, where the person is still watching.
    from inflorescence.config import Settings, user_env_file

    working_dir = str(Path.cwd())
    if not Settings().llm_api_key.strip():
        print(
            "Error: no OPENROUTER_API_KEY is visible from this directory, so the daemon would "
            "start and fail on every run.\n"
            f"  Settings are read from {user_env_file()} and from a .env in the working "
            "directory — neither has a key right now.\n"
            "  Run `inflorescence init` to write the per-user file, or re-run this from the "
            "directory whose .env holds the key.",
            file=sys.stderr,
        )
        return 1

    plist = launch_agent_path()
    plist.parent.mkdir(parents=True, exist_ok=True)
    command = resolve_console_script()
    plist.write_text(
        render_launch_agent(
            command, path_env=os.environ.get("PATH", ""), working_dir=working_dir
        )
    )

    target = f"gui/{os.getuid()}"
    # A previous copy must go first: bootstrap refuses a label that is already loaded, and
    # the old job would otherwise keep running the old command.
    _launchctl("bootout", f"{target}/{LAUNCH_AGENT_LABEL}")
    result = _launchctl("bootstrap", target, str(plist))
    if result.returncode != 0:
        legacy = _launchctl("load", "-w", str(plist))
        if legacy.returncode != 0:
            print(
                f"Error: could not start the launchd agent.\n  {result.stderr.strip() or result.stdout.strip()}\n"
                f"  The job is written to {plist} — load it manually, or run "
                "`inflorescence watch run` in a terminal.",
                file=sys.stderr,
            )
            return 1
    if asyncio.run(_daemon_already_running()):
        # The launchd job will now fail every retry until that one stops, which looks like a
        # broken install unless it is said here, once, at the moment it is caused.
        print(
            "Note: a watch daemon is already running on this machine (probably `watch run` in a "
            "terminal). Only one runs at a time, so stop it — the launchd job takes over on its "
            "next retry."
        )
    print(f"Watch daemon installed as {LAUNCH_AGENT_LABEL} and started ({plist}).")
    print(f"  It runs `{command} watch run` at every login and restarts if it dies.")
    print(f"  Working directory: {working_dir} (this is where its .env is read from).")
    print("  Logs: " + str(_default_log_file().parent / "watch.err.log"))
    print("  Register projects with `inflorescence watch add <path>`; check with `watch status`.")
    return 0


def _cmd_watch_uninstall() -> int:
    """Stop the LaunchAgent and remove its job file. Indexes are untouched."""
    if sys.platform != "darwin":
        print("Error: launchd agents are macOS-only; nothing to uninstall.", file=sys.stderr)
        return 1
    plist = launch_agent_path()
    _launchctl("bootout", f"gui/{os.getuid()}/{LAUNCH_AGENT_LABEL}")
    if plist.exists():
        plist.unlink()
        print(f"Watch daemon uninstalled ({plist} removed). Registrations and indexes are untouched.")
    else:
        print("No launchd agent was installed.")
    return 0


@dataclass
class CheckResult:
    name: str
    ok: bool
    critical: bool
    detail: str
    hint: str = ""


def _feedback_line(entry: dict) -> str:
    """One entry as a header line plus its indented body."""
    # Second precision for the eye; the stored value keeps microseconds so entries written
    # in the same second still sort.
    when = str(entry.get("recorded_at", "?"))[:19].replace("T", " ")
    kind = entry.get("kind", "other")
    severity = entry.get("severity", "medium")
    header = f"{entry.get('id', '?')}  {when}  {kind}/{severity}"
    if entry.get("tool"):
        header += f"  [{entry['tool']}]"
    lines = [header]
    body = str(entry.get("message", "")).strip()
    if body:
        lines += [f"    {line}" for line in body.splitlines()]
    if entry.get("reproduction"):
        lines.append("    repro: " + str(entry["reproduction"]).strip().replace("\n", " "))
    context = entry.get("context") or {}
    origin = context.get("answered_from") or context.get("directory")
    if origin:
        lines.append(f"    tree:  {origin}")
    for given, label in (("kind_as_given", "kind"), ("severity_as_given", "severity")):
        if entry.get(given):
            lines.append(f"    note:  reported {label}={entry[given]!r}, filed as {entry.get(label)!r}")
    return "\n".join(lines)


def _cmd_feedback(args: argparse.Namespace) -> int:
    """Print what agents wrote through ``send_feedback``.

    The write side is an MCP tool an agent calls; without a read side the log is a
    write-only hole, so this is the other half of the feature rather than a convenience.
    Reads the file directly — no database, no server — because feedback is most likely to
    exist precisely on a machine where those are broken.
    """
    from inflorescence.config import Settings
    from inflorescence.feedback import FeedbackStore

    try:
        settings = Settings()
        store = FeedbackStore(settings.feedback_path)
    except Exception as exc:  # noqa: BLE001 — a broken config must not hide the log path
        print(f"Could not load configuration: {exc}", file=sys.stderr)
        return 1

    if args.path:
        print(store.path)
        return 0

    try:
        log = store.read()
    except OSError as exc:
        print(f"Could not read {store.path}: {exc}", file=sys.stderr)
        return 1

    entries = log.entries
    if args.kind:
        wanted = args.kind.strip().lower().replace("-", "_")
        entries = [e for e in entries if str(e.get("kind", "")).lower() == wanted]
    total = len(entries)
    if args.limit:
        entries = entries[-args.limit:]

    if args.json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return 0

    print(f"Feedback log: {store.path}")
    if not log.entries:
        print(
            "No entries yet. Agents write here with the send_feedback MCP tool — the server "
            "asks them to after every task."
        )
        return 0
    shown = len(entries)
    suffix = "" if shown == total else f" ({shown} of {total} shown, newest last)"
    filtered = f", kind={args.kind}" if args.kind else ""
    print(f"{total} entr{'y' if total == 1 else 'ies'}{filtered}{suffix}")
    if log.unreadable:
        print(f"WARNING: {log.unreadable} line(s) in the file could not be parsed and are not shown")
    for entry in entries:
        print()
        print(_feedback_line(entry))
    return 0


def _memgraph_duplicate_containers() -> list[str]:
    """Containers sharing the database volume beyond the one this tool manages."""
    from inflorescence import docker_env

    return [name for name, _ in docker_env.containers_on_volume() if name != docker_env.MEMGRAPH_CONTAINER]


def _memgraph_recovery_hint(settings: Settings) -> str:
    """What to actually do about an unreachable Memgraph, read off the machine's own state."""
    from inflorescence import docker_env

    port = docker_env.local_bolt_port(settings.memgraph_url)
    if port is None:
        return f"{settings.memgraph_url} is not a local database; nothing here can start it"
    if not settings.memgraph_autostart:
        return (
            "Automatic startup is disabled (MEMGRAPH_AUTOSTART=false). "
            "Start Memgraph explicitly: docker start inflorescence-memgraph"
        )
    if docker_env.container_state(docker_env.MEMGRAPH_CONTAINER) is None:
        return "Run `inflorescence init` to start it"
    return docker_env.recovery_hint(docker_env.MEMGRAPH_CONTAINER, port)


async def _watch_doctor_check(conn) -> CheckResult | None:  # noqa: ANN001 — MemgraphConnection
    """How many registered projects a live daemon is actually following.

    Nothing registered is not a failure — plenty of people re-index by hand — so that reports
    as a pass with the command that changes it. Registered-but-unfollowed *is* a failure worth
    surfacing: it is the state after a reboot, it looks fine in the registry, and it is exactly
    how an index goes stale behind a green light.
    """
    from inflorescence.db.repository import GraphRepository

    repo = GraphRepository(conn)
    try:
        rows = await repo.list_projects()
    except Exception:  # noqa: BLE001 — a doctor check must not fail the doctor
        return None
    registered = [str(r["project"]) for r in rows if r.get("watched")]
    if not registered:
        return CheckResult(
            "Watch daemon",
            True,
            False,
            f"no projects registered ({len(rows)} indexed)",
            hint="Keep an index current: `inflorescence watch add <path>`, then `watch install`",
        )
    live = 0
    for project in registered:
        state = await repo.get_watch_state(project)
        live += bool(state.get("live"))
    return CheckResult(
        "Watch daemon",
        live == len(registered),
        False,
        f"{live} of {len(registered)} registered project(s) followed by a live daemon",
        hint="Start it: `inflorescence watch run` (or `inflorescence watch install` for login)",
    )


async def _doctor_checks(settings: Settings) -> list[CheckResult]:
    from inflorescence import docker_env
    from inflorescence.config import user_env_file
    from inflorescence.db import schema
    from inflorescence.db.connection import MemgraphConnection

    results: list[CheckResult] = []

    # 1. Config loaded (if we got here, Settings() constructed successfully)
    results.append(CheckResult("Config loads", True, True, "pydantic Settings constructed"))

    # 2. API key present / non-empty
    key_ok = bool(settings.llm_api_key.strip())
    results.append(
        CheckResult(
            "OPENROUTER_API_KEY",
            key_ok,
            True,
            "present" if key_ok else "missing or empty",
            hint=f"Run `inflorescence init`, or set OPENROUTER_API_KEY in {user_env_file()}",
        )
    )

    # 3. Memgraph reachable
    conn = MemgraphConnection(settings)
    try:
        reachable = await conn.health_check()
    except Exception:
        reachable = False
    results.append(
        CheckResult(
            "Memgraph reachable",
            reachable,
            True,
            settings.memgraph_url,
            # An unreachable database has more than one cause and only one of them is
            # "not started yet"; ask docker which one this is instead of guessing.
            hint=(
                "Run `inflorescence init` to start it (or `docker start inflorescence-memgraph`)"
                if reachable
                else _memgraph_recovery_hint(settings)
            ),
        )
    )

    # 3b. Exactly one container over the database volume. A second one is not a spare: the
    # two fight for the bolt port, and the loser can be left running yet unreachable — the
    # state that reads as "the server just stopped working" and survives every restart.
    duplicates = _memgraph_duplicate_containers()
    if duplicates:
        results.append(
            CheckResult(
                "Single Memgraph container",
                False,
                False,
                f"{len(duplicates) + 1} containers share the database volume: "
                + ", ".join([docker_env.MEMGRAPH_CONTAINER, *duplicates]),
                hint="Keep one and stop the rest: " + "; ".join(f"docker stop {name}" for name in duplicates),
            )
        )

    # 4. Schema present (only meaningful when reachable)
    schema_ok = False
    if reachable:
        try:
            schema_ok = await schema.schema_present(conn)
        except Exception:
            schema_ok = False
    # 4b. Is anything keeping the indexes current? An index is a snapshot, and the watcher
    # that `index_directory` starts dies with the session that started it — so "everything is
    # green but every answer is a week old" is a real state of this system, and one nothing
    # else in this checklist would report.
    if reachable:
        watch_check = await _watch_doctor_check(conn)
        if watch_check is not None:
            results.append(watch_check)
    with contextlib.suppress(Exception):
        await conn.close()
    results.append(
        CheckResult(
            "Schema present",
            schema_ok,
            False,
            "core indexes found" if schema_ok else "not found",
            hint="Run `inflorescence index <path>` to create the schema",
        )
    )

    # 5. SCIP rungs. Never critical: an absent indexer is a documented degradation to the
    # syntactic ladder, not a fault. Reported anyway because "why are my CALLS heuristic"
    # is otherwise only answerable by reading the log.
    if settings.use_scip_semantic:
        from inflorescence.code_indexer.models import IndexerConfig
        from inflorescence.code_indexer.scip_semantic import image_availability, indexer_availability

        config = IndexerConfig.from_settings(settings)
        availability = indexer_availability(config)
        results.append(
            CheckResult(
                "SCIP indexers",
                any(how != "none" for how in availability.values()),
                False,
                ", ".join(f"{lang}={how}" for lang, how in availability.items()),
                hint="Start Docker for zero-install SCIP, or install the binaries yourself (see README)",
            )
        )
        # Which image each docker rung would run, and whether `docker run` will touch the
        # network for it. Informational, never failing: the answer to "what exactly will
        # execute on my machine", not a health problem.
        for status in image_availability(config):
            results.append(
                CheckResult(
                    "SCIP image (" + ", ".join(status.languages) + ")",
                    True,
                    False,
                    status.image
                    + (" — present locally, no pull" if status.local else " — will be pulled on first use"),
                )
            )

    return results


def _cmd_doctor(settings: Settings) -> int:
    results = asyncio.run(_doctor_checks(settings))
    failed_critical = False
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"[{mark}] {r.name}: {r.detail}")
        if not r.ok:
            if r.hint:
                print(f"       hint: {r.hint}")
            if r.critical:
                failed_critical = True
    exit_code = 1 if failed_critical else 0
    # The exit code answers "can this machine work right now" — only a critical failure
    # says no. The printed verdict answers "is anything wrong", which is not the same
    # question: a second container over the database volume breaks nothing until the
    # moment it breaks everything, and "Doctor: OK" above that line is a lie.
    verdict = "OK" if all(r.ok for r in results) else "problems found"
    print(f"\nDoctor: {verdict}")
    return exit_code


# --language values -> image base names. "node" carries both scip-typescript and
# scip-python (they are both Node programs); there is no per-target-language image.
_IMAGE_NAMES: dict[str, str] = {"go": "scip-go", "node": "scip-node"}


def packaged_dockerfile(name: str) -> Path:
    """Path to a Dockerfile shipped inside the package (works from a wheel install)."""
    return Path(__file__).resolve().parent / "docker" / f"{name}.Dockerfile"


def build_image_argv(name: str, *, registry: str, tag: str) -> list[str]:
    """The `docker build` invocation for one image.

    The resulting reference must be exactly what the docker rung looks for
    (`_DEFAULT_IMAGE_TAGS` in scip_semantic.py): `docker run` uses a locally present tag
    without contacting any registry, so building under the expected name is what makes
    this a fully local path rather than a convenience wrapper.
    """
    dockerfile = packaged_dockerfile(name)
    image = f"{registry}/inflorescence-{name}:{tag}"
    return ["docker", "build", "-t", image, "-f", str(dockerfile), str(dockerfile.parent)]


def _cmd_build_images(args: argparse.Namespace) -> int:
    if not shutil.which("docker"):
        print(
            "Error: docker not found on PATH. Install Docker, or skip containers entirely: "
            "a native indexer on PATH is used as-is, and without either the CALLS ladder "
            "degrades to the syntactic rungs.",
            file=sys.stderr,
        )
        return 1
    names = list(_IMAGE_NAMES.values()) if args.language == "all" else [_IMAGE_NAMES[args.language]]
    for name in names:
        argv = build_image_argv(name, registry=args.registry, tag=args.tag)
        print(f"==> {argv[3]}")
        proc = subprocess.run(argv, check=False)
        if proc.returncode != 0:
            print(f"Error: docker build failed for {argv[3]} (exit {proc.returncode})", file=sys.stderr)
            return proc.returncode or 1
    return 0


def render_env_content(api_key: str) -> str:
    return f"OPENROUTER_API_KEY={api_key}\n"


def write_env_file(path: Path, api_key: str) -> None:
    """Write the key, creating the directory and keeping the file owner-readable only.

    A dotenv holding a billable API key is written 0600 where the platform supports it;
    on Windows chmod is a no-op and the file inherits directory ACLs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_env_content(api_key), encoding="utf-8")
    with contextlib.suppress(OSError, NotImplementedError):
        path.chmod(0o600)


def resolve_console_script(name: str = "inflorescence") -> str:
    """Absolute path to the installed console script (for a PATH-independent MCP config)."""
    found = shutil.which(name)
    if found:
        return str(Path(found).resolve())
    candidate = Path(sys.executable).resolve().parent / name
    if candidate.exists():
        return str(candidate)
    return name  # last resort: bare name, resolved via the client's PATH


def build_mcp_config(command: str, *, name: str = "inflorescence", api_key: str | None = None) -> dict:
    server: dict = {"command": command, "args": ["serve"]}
    # An MCP client may launch the server with a cwd where the project `.env` is
    # not found; carrying the key in `env` keeps the server functional regardless.
    if api_key is not None:
        server["env"] = {"OPENROUTER_API_KEY": api_key}
    return {"mcpServers": {name: server}}


def render_mcp_config_json(command: str, *, api_key: str | None = None) -> str:
    return json.dumps(build_mcp_config(command, api_key=api_key), indent=2)


def build_claude_mcp_add_command(command: str, *, name: str = "inflorescence") -> list[str]:
    # `claude mcp add <name> -- <command> serve`; `--` guards against flag ambiguity.
    return ["claude", "mcp", "add", name, "--", command, "serve"]


def _compose_file_present(cwd: Path) -> bool:
    return any((cwd / name).exists() for name in ("docker-compose.yml", "docker-compose.yaml", "compose.yml"))


_DOCKER_INSTALL_HINT = (
    "Install Docker Desktop (https://www.docker.com/products/docker-desktop/) or "
    "OrbStack (https://orbstack.dev/), then re-run `inflorescence init`."
)


def _ensure_docker_interactive() -> bool:
    """Get a working docker daemon for `init`, installing/launching it with consent.

    `init` is the one entry point with a human at a terminal, so this is the one place
    that may ask a question or run Homebrew. Everything it does is otherwise the user's
    own next step typed for them: install the app, launch it, wait for the daemon.
    """
    from inflorescence import docker_env as de

    if de.docker_available():
        return True

    if de.docker_cli_present():
        print("Docker is installed but not running — starting it (this can take a minute)...")
        if de.ensure_docker_daemon(wait_seconds=180.0):
            return True
        print("Docker did not come up. Start Docker Desktop (or OrbStack) manually, then re-run `inflorescence init`.")
        return False

    if sys.platform == "darwin" and shutil.which("brew"):
        answer = (
            input(
                "Docker is not installed (needed to run the Memgraph database).\n"
                "Install Docker Desktop now with `brew install --cask docker-desktop`? [y/N] "
            )
            .strip()
            .lower()
        )
        if answer == "y":
            subprocess.run(["brew", "install", "--cask", "docker-desktop"], check=False)
            print(
                "Launching Docker Desktop — if this is its first start, accept the service "
                "agreement in the window that opens."
            )
            subprocess.run(["open", "-a", "Docker"], check=False)
            # The app links the `docker` CLI into /usr/local/bin on first launch, so both
            # the CLI and the daemon can appear only after a while.
            deadline = time.monotonic() + 300.0
            while time.monotonic() < deadline:
                if de.docker_available():
                    return True
                time.sleep(3.0)
            print("Docker was installed but its daemon never answered. " + _DOCKER_INSTALL_HINT)
            return False

    print("Docker is not installed. " + _DOCKER_INSTALL_HINT)
    return False


def _bootstrap_memgraph(cwd: Path) -> None:
    """`init` step 2: make Memgraph reachable, whatever state the machine is in."""
    from inflorescence.config import Settings
    from inflorescence.docker_env import (
        containers_on_volume,
        ensure_memgraph,
        local_bolt_port,
        memgraph_run_argv,
        port_open,
    )

    memgraph_url = Settings().memgraph_url
    port = local_bolt_port(memgraph_url)
    if port is not None and port_open(port):
        print(f"Memgraph is already up at {memgraph_url}")
        return
    if port is not None and not _ensure_docker_interactive():
        return  # _ensure_docker_interactive printed what to do

    # Inside a checkout of this repository compose is the friendlier tool — it brings up
    # Memgraph Lab alongside the database. Anywhere else (an installed tool, the normal
    # case) a standalone container is started directly. Both make the *same* container
    # (compose pins container_name), so the two paths cannot split one database in two —
    # but compose refuses to adopt a container it did not create, so once one exists the
    # direct path is the only one that still works on it.
    existing = [name for name, _ in containers_on_volume()]
    if _compose_file_present(cwd) and shutil.which("docker") and not existing:
        print("Starting Memgraph via docker compose...")
        subprocess.run(["docker", "compose", "up", "-d", "memgraph"], check=False)
    elif ensure_memgraph(memgraph_url):
        print(f"Memgraph is up at {memgraph_url}")
    else:
        print(
            "Could not start Memgraph automatically. Install/start Docker, or run it yourself:\n"
            "  " + " ".join(memgraph_run_argv(port or 7687))
        )


def _cmd_init(args: argparse.Namespace) -> int:
    from inflorescence.config import Settings, user_env_file

    cwd = Path.cwd()
    # The key goes to the per-user file, not to a .env beside whatever directory `init` ran
    # in. An MCP client launches the server from its own working directory, so a
    # project-local key is invisible to the very process that needs it.
    env_path = user_env_file()

    # 1. API key -> user config (never overwrite without confirmation)
    write_env = True
    if env_path.exists():
        answer = input(f"{env_path} already exists. Overwrite? [y/N] ").strip().lower()
        write_env = answer == "y"
    resolved_key = os.environ.get("OPENROUTER_API_KEY", "")
    if write_env:
        prompt = "Enter OPENROUTER_API_KEY"
        if resolved_key:
            prompt += " (detected in environment; press Enter to reuse)"
        entered = getpass.getpass(prompt + ": ").strip()
        resolved_key = entered or resolved_key
        write_env_file(env_path, resolved_key)
        print(f"Wrote {env_path}")
    else:
        print(f"Keeping existing {env_path}")

    # 2. Memgraph — install/start Docker if needed, pull the image, start the container.
    _bootstrap_memgraph(cwd)

    # 3. MCP registration — always with the ABSOLUTE path to the console script
    command = resolve_console_script()
    if shutil.which("claude"):
        add_cmd = build_claude_mcp_add_command(command)
        print("Registering MCP server with Claude Code:\n  " + " ".join(add_cmd))
        subprocess.run(add_cmd, check=False)
    else:
        print(
            "Claude Code CLI not found. Add this to your Claude Desktop MCP config:\n"
            + render_mcp_config_json(command, api_key=resolved_key or "your_key_here")
        )

    # 4. doctor
    print("\nRunning doctor...\n")
    return _cmd_doctor(Settings())


def _preview(
    path: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> None:
    from inflorescence.code_indexer.graph_builder import GraphBuilder
    from inflorescence.code_indexer.models import IndexerConfig
    from inflorescence.config import Settings

    root = Path(path).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    settings = Settings()
    builder = GraphBuilder(
        repo=None,
        llm=None,
        settings=settings,
        config=IndexerConfig.from_settings(settings),
    )
    from inflorescence.cost import estimate_index_cost, price_largest_files

    preview = price_largest_files(
        builder.preview(root, include_patterns=include_patterns, exclude_patterns=exclude_patterns), settings
    )
    estimate = estimate_index_cost(preview, settings)
    print(f"Dry-run preview: {{'estimated_cost_usd': {estimate['usd']}, {str(preview)[1:-1]}}}")


async def _prune(
    path: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    """Drop an indexed project's nodes for paths a narrowed scope no longer covers.

    Structural only: no LLM client and no embedder is built, because nothing here is priced.
    """
    from inflorescence.code_indexer.graph_builder import GraphBuilder
    from inflorescence.code_indexer.models import IndexerConfig
    from inflorescence.config import Settings
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.repository import GraphRepository
    from inflorescence.project_manager import ProjectManager
    from inflorescence.rag.indexer import RAGIndexer

    root = Path(path).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not (include_patterns or exclude_patterns):
        print("Error: pass at least one --exclude or --include to narrow the scope", file=sys.stderr)
        sys.exit(1)

    settings = Settings()
    conn = MemgraphConnection(settings)
    if not await conn.health_check():
        print("Error: Memgraph is not reachable at " + settings.memgraph_url, file=sys.stderr)
        await conn.close()
        sys.exit(1)

    repo = GraphRepository(conn, batch_size=settings.db_write_batch_size)
    builder = GraphBuilder(
        repo=repo, llm=None, settings=settings, config=IndexerConfig.from_settings(settings)
    )
    manager = ProjectManager(
        repo=repo,
        graph_builder=builder,
        rag_indexer=RAGIndexer(repo=repo, settings=settings),
        settings=settings,
        conn=conn,
    )
    try:
        stats = await manager.prune_out_of_scope(
            str(root),
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            dry_run=dry_run,
        )
        verb = "Would prune" if dry_run else "Pruned"
        print(f"{verb}: {stats}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        await conn.close()


async def _index(
    path: str,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    max_cost: float | None = None,
    force_rebuild: bool = False,
) -> None:
    from inflorescence.code_indexer.graph_builder import GraphBuilder
    from inflorescence.code_indexer.models import IndexerConfig
    from inflorescence.config import Settings
    from inflorescence.cost import IndexCostExceededError
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.repository import GraphRepository
    from inflorescence.db.schema import setup_schema
    from inflorescence.llm_client import LLMClient
    from inflorescence.project_manager import ProjectManager
    from inflorescence.rag.indexer import RAGIndexer

    root = Path(path).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    settings = Settings()
    conn = MemgraphConnection(settings)

    healthy = await conn.health_check()
    if not healthy:
        print("Error: Memgraph is not reachable at " + settings.memgraph_url, file=sys.stderr)
        await conn.close()
        sys.exit(1)

    await setup_schema(conn, dimension=settings.embedding_dimension)

    repo = GraphRepository(conn, batch_size=settings.db_write_batch_size)
    llm = LLMClient(settings)
    builder = GraphBuilder(
        repo=repo,
        llm=llm,
        settings=settings,
        config=IndexerConfig.from_settings(settings),
    )
    rag = RAGIndexer(repo=repo, settings=settings)
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=settings, conn=conn)

    from inflorescence.progress import StderrProgress

    try:
        stats = await manager.index_directory(
            str(root),
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            max_cost_usd=max_cost,
            progress=StderrProgress(),
            force_rebuild=force_rebuild,
        )
        print(f"Indexing complete: {stats}")
    except IndexCostExceededError as exc:
        print(f"Aborted: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        await llm.close()
        await conn.close()


if __name__ == "__main__":
    main()
