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
        help="Abort a first-time index if the estimated cost exceeds this many dollars.",
    )
    idx.add_argument(
        "--force",
        action="store_true",
        dest="force_rebuild",
        default=False,
        help=(
            "Re-parse and re-resolve even when no file changed. Needed after an upgrade that "
            "improves edge resolution: the incremental path compares content checksums, so a "
            "better indexer never reaches an unchanged project. Summaries are reused by "
            "content hash, so this costs nothing."
        ),
    )

    # `inflorescence dashboard` — local web UI over the indexed graphs
    dash = sub.add_parser("dashboard", help="Start the web dashboard (stats, tree, graph, queries)")
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


def _serve(allow_degraded: bool = False) -> None:
    from inflorescence.config import Settings
    from inflorescence.docker_env import ensure_memgraph
    from inflorescence.server import create_server, shutdown, startup

    # An MCP client launches this process directly — there is no shell in front of it where a
    # `docker compose up` could have been typed, and no way for the user to act on the error
    # below at the moment it appears. Bring the database up ourselves instead.
    ensure_memgraph(Settings().memgraph_url)

    mcp, conn, llm, watcher = create_server()

    async def run() -> None:
        healthy = await startup(conn, allow_degraded=allow_degraded)
        if not healthy and not allow_degraded:
            print(
                "Error: Memgraph is not reachable and could not be started. Check that Docker "
                "is running, or start the database yourself with `docker compose up -d`. "
                "Pass `--allow-degraded` to start the server anyway.",
                file=sys.stderr,
            )
            await shutdown(conn, llm, watcher)
            raise SystemExit(1)
        try:
            await mcp.run_stdio_async()
        finally:
            await shutdown(conn, llm, watcher)

    asyncio.run(run())


@dataclass
class CheckResult:
    name: str
    ok: bool
    critical: bool
    detail: str
    hint: str = ""


async def _doctor_checks(settings: Settings) -> list[CheckResult]:
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
            hint="Start Memgraph with `docker compose up -d`",
        )
    )

    # 4. Schema present (only meaningful when reachable)
    schema_ok = False
    if reachable:
        try:
            schema_ok = await schema.schema_present(conn)
        except Exception:
            schema_ok = False
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
    print("\nDoctor: " + ("OK" if exit_code == 0 else "problems found"))
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


def _cmd_init(args: argparse.Namespace) -> int:
    from inflorescence.config import Settings, user_env_file
    from inflorescence.docker_env import ensure_memgraph, memgraph_run_argv

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

    # 2. Memgraph. Inside a checkout of this repository compose is the friendlier tool — it
    # brings up Memgraph Lab alongside the database. Anywhere else (an installed tool, which
    # is the normal case) a standalone container is started directly.
    memgraph_url = Settings().memgraph_url
    if _compose_file_present(cwd) and shutil.which("docker"):
        print("Starting Memgraph via docker compose...")
        subprocess.run(["docker", "compose", "up", "-d", "memgraph"], check=False)
    elif ensure_memgraph(memgraph_url):
        print(f"Memgraph is up at {memgraph_url}")
    else:
        print(
            "Could not start Memgraph automatically. Install/start Docker, or run it yourself:\n"
            "  " + " ".join(memgraph_run_argv(7687))
        )

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
    from inflorescence.cost import estimate_index_cost

    preview = builder.preview(root, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
    estimate = estimate_index_cost(preview, settings)
    print(f"Dry-run preview: {{'estimated_cost_usd': {estimate['usd']}, {str(preview)[1:-1]}}}")


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

    await setup_schema(conn)

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
