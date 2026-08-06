"""Starlette application serving the dashboard UI and its JSON API."""

from __future__ import annotations

import contextlib
import logging
import threading
import webbrowser
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from inflorescence.config import Settings
from inflorescence.dashboard.service import DashboardService
from inflorescence.dashboard.tools_console import ToolsConsole, confirmation_required
from inflorescence.db.connection import MemgraphConnection

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

DEFAULT_PORT = 8321


def is_cross_origin(request: Request) -> bool:
    """True when the browser says this request came from another site.

    The dashboard binds localhost and has no auth, so any page the user happens to
    have open could otherwise POST to it — and a tool call spends money and reads
    arbitrary directories. Browsers attach ``Origin`` to every cross-origin POST; a
    request without one (curl, a same-origin GET) is left alone.
    """
    origin = request.headers.get("origin")
    if not origin:
        return False
    return origin.split("//", 1)[-1] != request.headers.get("host", "")


def create_app(settings: Settings | None = None) -> Starlette:
    settings = settings or Settings()
    conn = MemgraphConnection(settings)
    service = DashboardService(conn, settings)
    console = ToolsConsole(settings)

    def api(handler):  # noqa: ANN001, ANN202 — starlette endpoint wrapper
        # Live graph data must never be served stale from the browser cache — a
        # re-index would otherwise keep showing the old shape until a hard reload.
        no_cache = {"Cache-Control": "no-store"}

        async def endpoint(request: Request) -> Response:
            if request.method == "POST" and is_cross_origin(request):
                return JSONResponse(
                    {"error": "cross-origin request refused"}, status_code=403, headers=no_cache
                )
            try:
                payload = await handler(request)
            except Exception as exc:  # noqa: BLE001 — surface any failure as JSON
                logger.exception("Dashboard API error on %s", request.url.path)
                return JSONResponse({"error": str(exc)}, status_code=500, headers=no_cache)
            if payload is None:
                return JSONResponse({"error": "not found"}, status_code=404, headers=no_cache)
            return JSONResponse(payload, headers=no_cache)

        return endpoint

    async def health(request: Request):
        return {"ok": await conn.health_check()}

    async def projects(request: Request):
        return {"projects": await service.list_projects()}

    async def stats(request: Request):
        return await service.project_stats(request.path_params["project"])

    async def status(request: Request):
        return await service.project_status(request.path_params["project"])

    async def tree(request: Request):
        return await service.tree_children(
            request.path_params["project"], request.query_params.get("parent")
        )

    async def entity(request: Request):
        node_id = request.query_params.get("id")
        if not node_id:
            return {"error": "missing id"}
        return await service.entity(request.path_params["project"], node_id)

    async def code(request: Request):
        node_id = request.query_params.get("id")
        if not node_id:
            return {"error": "missing id"}
        return await service.entity_code(
            request.path_params["project"],
            node_id,
            whole_file=request.query_params.get("whole_file") == "1",
        )

    async def graph(request: Request):
        project = request.path_params["project"]
        params = request.query_params
        if params.get("mode") == "full":
            return await service.graph_full(project, cap=int(params.get("cap", "4000")))
        if params.get("mode") == "overview" or not params.get("root"):
            return await service.graph_overview(project)
        relations = [r for r in (params.get("rels") or "").split(",") if r]
        return await service.graph_neighborhood(
            project,
            params["root"],
            depth=int(params.get("depth", "2")),
            relations=relations or None,
            cap=int(params.get("cap", "300")),
        )

    async def search(request: Request):
        return await service.search(
            request.path_params["project"],
            request.query_params.get("q", ""),
            mode=request.query_params.get("mode", "quick"),
        )

    async def query(request: Request):
        body = await request.json()
        return await service.run_query(
            request.path_params["project"],
            str(body.get("query", "")),
            limit=int(body.get("limit", 100)),
            offset=int(body.get("offset", 0)),
        )

    async def set_root(request: Request):
        body = await request.json()
        return await service.set_root_path(
            request.path_params["project"], str(body.get("path", ""))
        )

    # -- MCP tools console -------------------------------------------------
    # Not project-scoped: the tools take a `directory` argument of their own, which
    # is the whole point of showing a human exactly what an agent has to pass.

    async def tools_catalog(request: Request):
        return await console.catalog()

    async def tools_call(request: Request):
        body = await request.json()
        tool = str(body.get("tool") or "")
        arguments = body.get("arguments") or {}
        if not isinstance(arguments, dict):
            return {"error": "arguments must be an object"}
        catalog = await console.catalog()
        if not catalog["available"]:
            return {"error": f"MCP tools unavailable: {catalog['error']}"}
        if tool not in {entry["name"] for entry in catalog["tools"]}:
            return {"error": f"unknown tool: {tool}"}
        reason = confirmation_required(tool, arguments)
        if reason and not body.get("confirm"):
            return {"error": reason, "needs_confirm": True}
        return {"run": console.start(tool, arguments).payload()}

    async def tools_runs(request: Request):
        return {"runs": console.runs()}

    async def tools_run(request: Request):
        run = console.get(request.path_params["run_id"])
        return run.payload() if run else None

    async def tools_cancel(request: Request):
        return {"cancelled": console.cancel(request.path_params["run_id"])}

    async def index(request: Request) -> Response:
        return FileResponse(_STATIC_DIR / "index.html")

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await console.close()
            with contextlib.suppress(Exception):
                await conn.close()

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/health", api(health)),
            Route("/api/projects", api(projects)),
            Route("/api/projects/{project}/stats", api(stats)),
            Route("/api/projects/{project}/status", api(status)),
            Route("/api/projects/{project}/tree", api(tree)),
            Route("/api/projects/{project}/entity", api(entity)),
            Route("/api/projects/{project}/code", api(code)),
            Route("/api/projects/{project}/graph", api(graph)),
            Route("/api/projects/{project}/search", api(search)),
            Route("/api/projects/{project}/query", api(query), methods=["POST"]),
            Route("/api/projects/{project}/root", api(set_root), methods=["POST"]),
            Route("/api/tools", api(tools_catalog)),
            Route("/api/tools/call", api(tools_call), methods=["POST"]),
            Route("/api/tools/runs", api(tools_runs)),
            Route("/api/tools/runs/{run_id}", api(tools_run)),
            Route("/api/tools/runs/{run_id}/cancel", api(tools_cancel), methods=["POST"]),
            Mount("/static", StaticFiles(directory=_STATIC_DIR), name="static"),
        ],
        lifespan=lifespan,
    )


def run_dashboard(host: str = "127.0.0.1", port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    """Run the dashboard with uvicorn (blocking)."""
    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"Inflorescence dashboard: {url}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")
