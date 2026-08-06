"""LLM fallback parser for files that fail AST parsing."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable
from inspect import isawaitable
from pathlib import Path
from typing import Protocol

from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, NodeType

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a code analysis assistant. Given a Python source file, extract its structure as JSON.
Return ONLY valid JSON with this schema:
{
  "nodes": [
    {"name": "...", "type": "module|class|function|method", "line_start": N, "line_end": N,
     "signature": "...", "docstring": "...", "parent": null|"ClassName"}
  ],
  "edges": [
    {"source": "...", "target": "...", "type": "imports|calls|inherits|contains"}
  ]
}
"""

_NODE_TYPE_MAP: dict[str, NodeType] = {
    "module": NodeType.MODULE,
    "class": NodeType.CLASS,
    "function": NodeType.FUNCTION,
    "method": NodeType.METHOD,
}

_EDGE_TYPE_MAP: dict[str, EdgeType] = {
    "imports": EdgeType.IMPORTS,
    "calls": EdgeType.CALLS,
    "inherits": EdgeType.INHERITS,
    "contains": EdgeType.CONTAINS,
}


class LLMServiceProtocol(Protocol):
    """Minimal protocol for the LLM service dependency."""

    def generate(self, prompt: str, system_prompt: str = "") -> str | Awaitable[str]: ...


async def parse_file_with_llm(
    file_path: Path,
    root_path: Path,
    llm_service: LLMServiceProtocol,
) -> tuple[list[CodeNode], list[Edge]]:
    """Use LLM to extract structure from a file that failed AST parsing."""
    rel = str(file_path.relative_to(root_path))
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.debug("Failed to read %s, skipping", file_path)
        return [], []

    # Truncate to avoid excessive token usage
    prompt = f"Analyze this Python file and extract its structure:\n\n```python\n{source[:8000]}\n```"

    try:
        raw = llm_service.generate(prompt, system_prompt=SYSTEM_PROMPT)
        if isawaitable(raw):
            raw = await raw
        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
        data: dict[str, list[dict[str, str | int | None]]] = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned invalid JSON for %s", file_path)
        return [], []
    except Exception:
        logger.exception("LLM parsing failed for %s", file_path)
        return [], []

    nodes: list[CodeNode] = []
    edges: list[Edge] = []

    for n in data.get("nodes", []):
        ntype = _NODE_TYPE_MAP.get(str(n.get("type", "")), NodeType.FUNCTION)
        parent = n.get("parent")
        parent_str = str(parent) if parent else None
        name = str(n.get("name", ""))
        if ntype == NodeType.MODULE:
            nid = rel
        elif ntype == NodeType.METHOD and parent_str:
            nid = f"{rel}::{parent_str}.{name}"
        else:
            nid = f"{rel}::{name}"
        parent_id = f"{rel}::{parent_str}" if parent_str else (rel if ntype != NodeType.MODULE else None)
        nodes.append(
            CodeNode(
                id=nid,
                name=name,
                node_type=ntype,
                file_path=rel,
                line_start=int(n.get("line_start", 1) or 1),
                line_end=int(n.get("line_end", 1) or 1),
                signature=str(n.get("signature", "")),
                docstring=str(n.get("docstring", "")),
                parent_id=parent_id,
            )
        )

    for e in data.get("edges", []):
        etype = _EDGE_TYPE_MAP.get(str(e.get("type", "")))
        if etype:
            edges.append(Edge(source=str(e["source"]), target=str(e["target"]), edge_type=etype))

    return nodes, edges
