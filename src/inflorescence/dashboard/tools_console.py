"""Call the server's own MCP tools from the dashboard, exactly as an agent would.

The console builds a FastMCP instance in this process and invokes the very functions
an agent reaches over stdio, so what it prints is what an agent receives — not a
second implementation that can drift from the server.

Two properties are worth the machinery here:

*Runs are backgrounded.* ``index_directory`` works for minutes; a run that outlives
its HTTP request survives a page reload and can be cancelled.

*Runs happen on their own event loop, on their own thread.* The embedding client is
the synchronous OpenAI SDK, so an embedding call blocks whatever loop it runs on. On
the dashboard's loop that would freeze the live graph this dashboard exists to show.
One worker thread serializes the calls (Memgraph's index lease serializes the
expensive one anyway) and leaves the UI responsive throughout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Coroutine, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from inflorescence.config import Settings

logger = logging.getLogger(__name__)

# Runs kept for inspection. A console exists to compare calls, so history matters,
# but every entry holds a whole tool payload — 25 is plenty and bounds the memory.
MAX_RUNS = 25

# Display order and grouping of the tool list, cheapest question first. Also the
# single source of truth for which category a tool belongs to.
TOOL_CATEGORY: dict[str, str] = {
    "search_entities": "search",
    "search_text": "search",
    "search_semantic": "search",
    "search_code": "search",
    "search_hybrid": "search",
    "list_entities": "graph",
    "get_entity_structure": "graph",
    "get_entity_context": "graph",
    "get_related_entities": "graph",
    "graph_schema": "cypher",
    "cypher_query": "cypher",
    "index_directory": "indexing",
}
CATEGORY_ORDER = ("search", "graph", "cypher", "indexing", "other")

# What a call costs beyond a Memgraph read. Money is spent consciously: the UI
# labels these, and a spending call is refused unless it was explicitly confirmed.
TOOL_SPEND: dict[str, str] = {
    "index_directory": "llm",
    "search_semantic": "embedding",
    "search_code": "embedding",
    "search_hybrid": "embedding",
}


def confirmation_required(tool: str, arguments: Mapping[str, Any]) -> str | None:
    """Reason this call must carry an explicit confirmation, or None.

    Indexing for real is the one console action that spends LLM money and can run
    for minutes. ``preview=true`` is free and is exactly what the tool's own
    description tells an agent to run first.
    """
    if tool == "index_directory" and not arguments.get("preview"):
        return (
            "index_directory with preview=false runs the real index: LLM summaries and "
            "embeddings cost money, and the run can take minutes. Call it with "
            "preview=true first to see the estimate."
        )
    return None


def _block_payload(block: Any) -> dict[str, Any]:
    """Serialize one MCP content block the way a client would see it."""
    dump = getattr(block, "model_dump", None)
    if dump is None:  # pragma: no cover — every ContentBlock is a pydantic model
        return {"type": "text", "text": str(block)}
    return dump(mode="json", exclude_none=True)


def _normalize(result: Any) -> tuple[list[dict[str, Any]], Any]:
    """Split a FastMCP tool result into (content blocks, structured payload).

    Tools here are annotated ``-> dict`` with no output schema, so FastMCP ships a
    single JSON text block and no structuredContent. Parsing that block back is what
    lets the console render the result as data instead of as a wall of text; if a
    tool ever gains a real output schema, the ``(blocks, structured)`` tuple form is
    handled too.
    """
    structured: Any = None
    if isinstance(result, tuple) and len(result) == 2:
        result, structured = result
    blocks = [_block_payload(block) for block in (result or [])]
    if structured is None:
        texts = [b.get("text") for b in blocks if b.get("type") == "text" and b.get("text")]
        if len(texts) == 1:
            try:
                structured = json.loads(str(texts[0]))
            except (TypeError, ValueError):
                structured = None
    return blocks, structured


def _error_envelope(structured: Any) -> dict[str, str] | None:
    """The tools' own ``{"error": {code, message}}`` payload, if that is what came back.

    A tool that reports "not indexed" or "cost exceeded" succeeded as a call — the
    console has to show that differently from a call that raised.
    """
    if isinstance(structured, dict) and isinstance(structured.get("error"), dict):
        error = structured["error"]
        return {
            "code": str(error.get("code", "error")),
            "message": str(error.get("message", "")),
        }
    return None


@dataclass
class ToolRun:
    """One console invocation: its arguments, its state, and what came back."""

    id: str
    tool: str
    arguments: dict[str, Any]
    started_at: float = field(default_factory=time.time)
    state: str = "running"  # running | ok | error | cancelled
    duration_ms: int | None = None
    content: list[dict[str, Any]] = field(default_factory=list)
    structured: Any = None
    error: str | None = None
    error_envelope: dict[str, str] | None = None
    # Cancellation lands at the coroutine's next await. A tool in the middle of a
    # blocking step (the preview walks the filesystem synchronously) runs to
    # completion regardless, so the request is recorded and reported rather than
    # presented as a stop that always works.
    cancel_requested: bool = False
    task: asyncio.Task[None] | None = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        """Compact form for the run list — no payloads."""
        return {
            "id": self.id,
            "tool": self.tool,
            "state": self.state,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "error_code": (self.error_envelope or {}).get("code"),
            "arg_preview": _arg_preview(self.arguments),
            "cancel_requested": self.cancel_requested,
        }

    def payload(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "arguments": self.arguments,
            "content": self.content,
            "structured": self.structured,
            "error": self.error,
            "error_envelope": self.error_envelope,
        }


def _arg_preview(arguments: Mapping[str, Any]) -> str:
    """One-line rendering of the arguments that identify a call in a list."""
    interesting = [k for k in ("query", "node_id", "file_path", "path", "directory") if arguments.get(k)]
    if not interesting:
        return ""
    value = str(arguments[interesting[0]]).replace("\n", " ")
    return value if len(value) <= 60 else value[:59] + "…"


class _Worker:
    """A private event loop running on its own daemon thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="mcp-tools-console", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call_soon(self, fn: Any) -> None:
        self.loop.call_soon_threadsafe(fn)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


class ToolsConsole:
    """Catalog of the MCP tools plus the runs the dashboard has started."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, ToolRun] = OrderedDict()
        self._seq = 0
        self._worker: _Worker | None = None
        self._components: tuple[Any, Any, Any, Any] | None = None

    # ------------------------------------------------------------------
    # Bootstrap — everything is lazy: a dashboard that never opens the
    # console never builds an MCP server, an LLM client or a second driver.
    # ------------------------------------------------------------------

    def _submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        with self._lock:
            if self._worker is None:
                self._worker = _Worker()
            worker = self._worker
        return worker.submit(coro)

    async def _mcp(self) -> Any:
        """The FastMCP instance. Only ever touched from the worker loop."""
        if self._components is None:
            from inflorescence.server import create_server

            self._components = create_server(enable_watcher=False)
        return self._components[0]

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    async def catalog(self) -> dict[str, Any]:
        try:
            tools = await asyncio.wrap_future(self._submit(self._list_tools()))
        except Exception as exc:  # noqa: BLE001 — a broken console must not break the dashboard
            logger.exception("MCP tool catalog unavailable")
            return {"available": False, "error": str(exc), "tools": [], "categories": list(CATEGORY_ORDER)}
        return {"available": True, "tools": tools, "categories": list(CATEGORY_ORDER)}

    async def _list_tools(self) -> list[dict[str, Any]]:
        mcp = await self._mcp()
        order = list(TOOL_CATEGORY)
        tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
                "category": TOOL_CATEGORY.get(tool.name, "other"),
                "spend": TOOL_SPEND.get(tool.name),
            }
            for tool in await mcp.list_tools()
        ]
        tools.sort(key=lambda t: order.index(t["name"]) if t["name"] in order else len(order))
        return tools

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def start(self, tool: str, arguments: dict[str, Any]) -> ToolRun:
        with self._lock:
            self._seq += 1
            run = ToolRun(id=str(self._seq), tool=tool, arguments=arguments)
            self._runs[run.id] = run
            self._evict_locked()
        self._submit(self._execute(run))
        return run

    def _evict_locked(self) -> None:
        """Drop the oldest *finished* runs past the cap; a running one keeps its slot."""
        while len(self._runs) > MAX_RUNS:
            for run_id, run in self._runs.items():
                if run.state != "running":
                    del self._runs[run_id]
                    break
            else:
                return

    async def _execute(self, run: ToolRun) -> None:
        run.task = asyncio.current_task()
        started = time.monotonic()
        try:
            mcp = await self._mcp()
            result = await mcp.call_tool(run.tool, run.arguments)
        except asyncio.CancelledError:
            self._finish(run, "cancelled", started, error="Cancelled from the dashboard.")
            raise
        except Exception as exc:  # noqa: BLE001 — the failure is the thing being shown
            logger.warning("MCP tool %s failed: %s", run.tool, exc)
            self._finish(run, "error", started, error=str(exc))
            return
        blocks, structured = _normalize(result)
        self._finish(run, "ok", started, content=blocks, structured=structured)

    def _finish(
        self,
        run: ToolRun,
        state: str,
        started: float,
        *,
        content: list[dict[str, Any]] | None = None,
        structured: Any = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            run.state = state
            run.duration_ms = int((time.monotonic() - started) * 1000)
            run.content = content or []
            run.structured = structured
            run.error = error
            run.error_envelope = _error_envelope(structured)
            run.task = None

    def get(self, run_id: str) -> ToolRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def runs(self) -> list[dict[str, Any]]:
        """Newest first — the order a console is read in."""
        with self._lock:
            return [run.summary() for run in reversed(self._runs.values())]

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            worker = self._worker
            task = run.task if run else None
            if run is not None and run.state == "running":
                run.cancel_requested = True
        if run is None or task is None or worker is None or run.state != "running":
            return False
        worker.call_soon(task.cancel)
        return True

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    async def close(self) -> None:
        with self._lock:
            worker, self._worker = self._worker, None
        if worker is None:
            return
        try:
            await asyncio.wrap_future(worker.submit(self._close_components()))
        except Exception:  # noqa: BLE001 — shutdown is best-effort
            logger.debug("Tools console shutdown failed", exc_info=True)
        finally:
            worker.close()

    async def _close_components(self) -> None:
        for run in list(self._runs.values()):
            if run.task is not None:
                run.task.cancel()
        if self._components is None:
            return
        from inflorescence.server import shutdown

        _, conn, llm, watcher = self._components
        self._components = None
        await shutdown(conn, llm, watcher)
