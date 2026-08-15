"""Hierarchical LLM summary generation for code graph nodes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from inflorescence.code_indexer import prompts
from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, NodeType
from inflorescence.llm_client import LLMClient
from inflorescence.progress import ProgressReporter, resolve

logger = logging.getLogger(__name__)


def _child_descriptor(child: CodeNode) -> str:
    """The exact per-child line a container feeds the LLM — mirrors `_summarize_container`."""
    kind = child.node_type.value
    sig_part = f" `{child.signature}`" if child.signature else ""
    summary_part = f" — {child.summary}" if child.summary else ""
    return f"  [{kind}]{sig_part} {child.name}{summary_part}"


def summary_input_hash(node: CodeNode, children: list[CodeNode], source: str = "") -> str:
    """Content hash of a node's summarization input.

    Leaf (no children): H(source + signature + docstring).
    Container: H(ordered child descriptors, each including the child's current summary).
    """
    if children:
        payload = "\n".join(_child_descriptor(c) for c in children)
    else:
        payload = "\x00".join([source, node.signature, node.docstring])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def summarize_hierarchy(
    nodes: list[CodeNode],
    edges: list[Edge],
    root_path: Path,
    llm: LLMClient,
    batch_size: int = 50,
    max_source_chars: int = 12000,
    stored: dict[str, dict[str, object]] | None = None,
    progress: ProgressReporter | None = None,
    on_batch: Callable[[Sequence[CodeNode]], Awaitable[None]] | None = None,
) -> set[str]:
    """Generate summaries bottom-up, reusing stored summaries whose input hash is unchanged.

    Returns the set of node ids whose summary was (re)generated. Mutates nodes in place,
    setting both ``summary`` and ``summary_input_hash``.

    ``on_batch`` is awaited after each *generated* batch (reused nodes never trigger it —
    they are already persisted). Exceptions from the callback propagate and abort the run:
    a caller that persists batches wants a dead DB to stop the spend, not to be swallowed.
    """
    stored = stored or {}
    bar = resolve(progress)
    node_map = {n.id: n for n in nodes}

    children_map: dict[str, list[str]] = defaultdict(list)
    parent_map: dict[str, str] = {}
    parent_set: set[str] = set()
    for e in edges:
        if e.edge_type == EdgeType.CONTAINS:
            children_map[e.source].append(e.target)
            parent_map[e.target] = e.source
            parent_set.add(e.source)

    depth_cache: dict[str, int] = {}

    def _depth(node_id: str) -> int:
        if node_id in depth_cache:
            return depth_cache[node_id]
        p = parent_map.get(node_id)
        d = 1 + _depth(p) if p else 0
        depth_cache[node_id] = d
        return d

    leaves = [n for n in nodes if n.id not in parent_set]
    containers = [n for n in nodes if n.id in parent_set]

    total = len(nodes)
    if total == 0:
        logger.info("No nodes require summarization")
        return set()

    dirty: set[str] = set()
    failed = 0
    done = 0

    def _advance(n: int) -> None:
        nonlocal done
        done += n
        bar.update("summarize", done, total)

    def _reuse(node: CodeNode, input_hash: str) -> bool:
        prev = stored.get(node.id)
        if prev and prev.get("summary_input_hash") == input_hash and prev.get("summary"):
            node.summary = str(prev["summary"])
            node.summary_input_hash = input_hash
            return True
        return False

    def _keep_stale_on_failure(node: CodeNode, fallback: str) -> None:
        """Generation failed: keep the previous stored summary (stale beats empty).

        The input hash is cleared so the node is dirty again on the next run and the
        summary is retried. A kept stale summary stays consistent with its stored
        embedding, so the node is NOT marked dirty for re-embedding.
        """
        nonlocal failed
        failed += 1
        prev = stored.get(node.id)
        prev_summary = str(prev["summary"]) if prev and prev.get("summary") else ""
        node.summary = prev_summary or fallback
        node.summary_input_hash = ""
        if not prev_summary:
            dirty.add(node.id)

    # A node is "attempted" once this run has settled its summary (generated, reused, or
    # failed-with-fallback). A container becomes summarizable the moment ALL its children
    # are attempted — its input hash is built from their now-final summaries.
    attempted: set[str] = set()

    async def _generate_containers(gen: list[CodeNode], label: str) -> None:
        for i in range(0, len(gen), batch_size):
            batch = gen[i : i + batch_size]
            results = await asyncio.gather(
                *[_summarize_container(n, children_map, node_map, llm, max_source_chars) for n in batch],
                return_exceptions=True,
            )
            for node, result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("Failed to summarize container: %s: %s", node.id, result)
                    _keep_stale_on_failure(node, fallback="")
                else:
                    node.summary = str(result)
                    dirty.add(node.id)
            if on_batch is not None:
                await on_batch(batch)
            _advance(len(batch))
            logger.info("Summarized %d/%d nodes (%s)", done, total, label)

    async def _drain_ready_containers() -> None:
        """Summarize every container whose children are all attempted, cascading upward.

        Called after each leaf batch so a file lands right after its functions and a
        directory right after its files — summaries stream through the whole hierarchy
        during the run (the live dashboard shows files popping in continuously) instead
        of every container arriving in a burst at the end. Cost is unchanged: the same
        nodes are generated, only earlier.
        """
        while True:
            ready = [
                n for n in containers
                if n.id not in attempted
                and all(cid in attempted or cid not in node_map for cid in children_map.get(n.id, []))
            ]
            if not ready:
                return
            gen: list[CodeNode] = []
            for node in ready:
                child_nodes = [node_map[cid] for cid in children_map.get(node.id, []) if cid in node_map]
                input_hash = summary_input_hash(node, child_nodes)
                attempted.add(node.id)
                if _reuse(node, input_hash):
                    _advance(1)
                else:
                    node.summary_input_hash = input_hash
                    gen.append(node)
            await _generate_containers(gen, "drained containers")

    # Phase 1: leaves (source-based hash), concurrent generation in batches.
    to_generate: list[CodeNode] = []
    for node in leaves:
        source = node_source(node, root_path)
        input_hash = summary_input_hash(node, [], source=source)
        if _reuse(node, input_hash):
            _advance(1)
            attempted.add(node.id)
        else:
            node.summary_input_hash = input_hash
            to_generate.append(node)
    # A fully-reused subtree (re-index) drains its containers before any LLM call.
    await _drain_ready_containers()

    for i in range(0, len(to_generate), batch_size):
        batch = to_generate[i : i + batch_size]
        results = await asyncio.gather(
            *[_summarize_leaf(n, root_path, llm, max_source_chars) for n in batch],
            return_exceptions=True,
        )
        for node, result in zip(batch, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Failed to summarize leaf: %s: %s", node.id, result)
                _keep_stale_on_failure(node, fallback=node.docstring or "")
            else:
                node.summary = str(result)
                dirty.add(node.id)
            attempted.add(node.id)
        if on_batch is not None:
            await on_batch(batch)
        _advance(len(batch))
        logger.info("Summarized %d/%d nodes (leaves)", done, total)
        await _drain_ready_containers()

    # Phase 2: safety net for containers the drain never reached (it shouldn't leave
    # any — every container's children are attempted once the leaf loop ends — but a
    # CONTAINS anomaly must degrade to the old level-by-level pass, not to a skipped
    # summary). Deepest-first so children summaries are final.
    depth_groups: dict[int, list[CodeNode]] = defaultdict(list)
    for n in containers:
        if n.id not in attempted:
            depth_groups[_depth(n.id)].append(n)

    for d in sorted(depth_groups.keys(), reverse=True):
        gen: list[CodeNode] = []
        for node in depth_groups[d]:
            child_nodes = [node_map[cid] for cid in children_map.get(node.id, []) if cid in node_map]
            input_hash = summary_input_hash(node, child_nodes)
            attempted.add(node.id)
            if _reuse(node, input_hash):
                _advance(1)
            else:
                node.summary_input_hash = input_hash
                gen.append(node)
        await _generate_containers(gen, f"depth {d}")

    # `done` reaches `total` exactly (leaves + containers partition all nodes), so the
    # last _advance already closed the bar — no explicit final update needed.
    if failed:
        logger.error(
            "Summarization finished with %d failure(s) out of %d nodes for this run — "
            "affected nodes keep their previous summary and will retry on the next index",
            failed, total,
        )
    logger.info("Summarization complete: %d nodes, %d regenerated, %d reused", total, len(dirty), total - len(dirty))
    return dirty


async def summarize_single(
    node: CodeNode,
    root_path: Path,
    llm: LLMClient,
    max_source_chars: int,
    children: list[CodeNode] | None = None,
) -> str:
    """Summarize one node outside a full hierarchy pass.

    Leaves (no *children*) are summarized from source; containers from the given
    children's current summaries — same prompts as `summarize_hierarchy`.
    """
    if children:
        children_map = {node.id: [c.id for c in children]}
        node_map = {c.id: c for c in children}
        return await _summarize_container(node, children_map, node_map, llm, max_source_chars)
    return await _summarize_leaf(node, root_path, llm, max_source_chars)


async def _summarize_leaf(
    node: CodeNode,
    root_path: Path,
    llm: LLMClient,
    max_source_chars: int,
) -> str:
    """Summarize a leaf node from its source code. Returns summary text."""
    source = node_source(node, root_path)
    if not source:
        return node.docstring or ""

    # A document is prose, not code: the code prompts ask for call graphs, side effects and
    # return values it has none of, and answering that question about a README produces a
    # summary that indexes badly. Same shape, different prompts.
    if node.node_type == NodeType.DOCUMENT:
        system_prompt = prompts.DOCUMENT_SYSTEM_PROMPT
        map_prompt, reduce_prompt = prompts.DOCUMENT_MAP_SYSTEM_PROMPT, prompts.DOCUMENT_REDUCE_SYSTEM_PROMPT
        prompt = f"Summarize this document, the file '{node.file_path}':\n\n{source}"
    else:
        system_prompt = prompts.LEAF_SYSTEM_PROMPT
        map_prompt, reduce_prompt = prompts.MAP_SYSTEM_PROMPT, prompts.REDUCE_SYSTEM_PROMPT
        prompt = (
            f"Summarize this {node.node_type.value} named '{node.name}':\n\n"
            f"Signature: {node.signature}\n"
            f"Docstring: {node.docstring}\n\n"
            f"```\n{source}\n```"
        )

    if len(source) <= max_source_chars:
        summary = await llm.generate(prompt, system_prompt=system_prompt)
    else:
        summary = await _map_reduce_summarize(
            node, source, llm, max_source_chars, map_prompt=map_prompt, reduce_prompt=reduce_prompt
        )

    return summary.strip()


async def _summarize_container(
    node: CodeNode,
    children_map: dict[str, list[str]],
    node_map: dict[str, CodeNode],
    llm: LLMClient,
    max_chars: int = 12000,
) -> str:
    """Summarize a container node from its children's summaries. Returns summary text.

    When the children don't fit one prompt (many children, or rich child summaries),
    fall back to map-reduce: digest children in batches, then merge the digests.
    """
    child_ids = children_map.get(node.id, [])
    child_descriptions: list[str] = []
    for cid in child_ids:
        child = node_map.get(cid)
        if child:
            kind = child.node_type.value
            summary_part = f" — {child.summary}" if child.summary else ""
            sig_part = f" `{child.signature}`" if child.signature else ""
            child_descriptions.append(f"  [{kind}]{sig_part} {child.name}{summary_part}")

    if not child_descriptions:
        return ""

    contents = "\n".join(child_descriptions)
    if len(contents) > max_chars:
        return await _map_reduce_container(node, child_descriptions, llm, max_chars)

    prompt = f"Summarize this {node.node_type.value} '{node.name}' based on its contents:\n\n{contents}"
    summary = await llm.generate(prompt, system_prompt=prompts.CONTAINER_SYSTEM_PROMPT)
    return summary.strip()


async def _map_reduce_container(
    node: CodeNode,
    child_descriptions: list[str],
    llm: LLMClient,
    max_chars: int,
) -> str:
    """Summarize a wide container via map-reduce over batches of child descriptors.

    Children are packed into char-bounded batches, each batch digested in parallel, and
    the digests merged through the container card prompt — so a directory with hundreds of
    files or a module with hundreds of members never produces one oversized prompt.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for line in child_descriptions:
        if current and size + len(line) > max_chars:
            batches.append(current)
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        batches.append(current)

    logger.info("  Map-reduce container %s: %d children -> %d batches", node.id, len(child_descriptions), len(batches))

    async def _map_batch(i: int, batch: list[str]) -> str:
        contents = "\n".join(batch)
        prompt = (
            f"Summarize group {i + 1}/{len(batches)} of the contents of "
            f"{node.node_type.value} '{node.name}':\n\n{contents}"
        )
        result = await llm.generate(prompt, system_prompt=prompts.CONTAINER_MAP_SYSTEM_PROMPT)
        return result.strip()

    partials = await asyncio.gather(*[_map_batch(i, b) for i, b in enumerate(batches)])

    combined = "\n".join(f"  [group] {p}" for p in partials)
    prompt = f"Summarize this {node.node_type.value} '{node.name}' based on its contents:\n\n{combined}"
    summary = await llm.generate(prompt, system_prompt=prompts.CONTAINER_SYSTEM_PROMPT)
    return summary.strip()


async def _map_reduce_summarize(
    node: CodeNode,
    source: str,
    llm: LLMClient,
    max_chars: int,
    map_prompt: str = prompts.MAP_SYSTEM_PROMPT,
    reduce_prompt: str = prompts.REDUCE_SYSTEM_PROMPT,
) -> str:
    """Summarize large source code via map-reduce: chunk, summarize each in parallel, then combine."""
    overlap = int(max_chars * 0.2)
    chunks: list[str] = []
    start = 0
    while start < len(source):
        end = start + max_chars
        chunks.append(source[start:end])
        start = end - overlap

    logger.info("  Map-reduce for %s: %d chunks (%d chars)", node.id, len(chunks), len(source))

    # Map phase — parallel
    async def _map_chunk(i: int, chunk: str) -> str:
        prompt = (
            f"Summarize chunk {i + 1}/{len(chunks)} of {node.node_type.value} '{node.name}':\n\n"
            f"```\n{chunk}\n```"
        )
        result = await llm.generate(prompt, system_prompt=map_prompt)
        return result.strip()

    chunk_summaries = await asyncio.gather(*[_map_chunk(i, c) for i, c in enumerate(chunks)])

    # Reduce phase
    combined = "\n".join(f"- Part {i + 1}: {s}" for i, s in enumerate(chunk_summaries))
    prompt = (
        f"Combine these partial summaries of {node.node_type.value} '{node.name}' "
        f"into a final summary:\n\n{combined}"
    )
    return await llm.generate(prompt, system_prompt=reduce_prompt)


def node_source(node: CodeNode, root_path: Path) -> str:
    """Extract full source code for a node."""
    try:
        path = root_path / node.file_path
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        snippet = lines[node.line_start - 1 : node.line_end]
        return "\n".join(snippet)
    except (OSError, IndexError):
        return ""
