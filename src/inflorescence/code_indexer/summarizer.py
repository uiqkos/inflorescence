"""Hierarchical LLM summary generation for code graph nodes."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from inflorescence.code_indexer import prompts
from inflorescence.code_indexer.models import DECLARATION_NODE_TYPES, CodeNode, Edge, EdgeType, NodeType
from inflorescence.llm_client import LLMClient
from inflorescence.progress import ProgressReporter, resolve

logger = logging.getLogger(__name__)


def _child_descriptor(child: CodeNode) -> str:
    """The exact per-child line a container feeds the LLM — mirrors `_summarize_container`."""
    kind = child.node_type.value
    sig_part = f" `{child.signature}`" if child.signature else ""
    summary_part = f" — {child.summary}" if child.summary else ""
    return f"  [{kind}]{sig_part} {child.name}{summary_part}"


# How much of a module's header (its docstring, or the comment block a JS/TS/Go/Rust file opens
# with) a container prompt quotes. The header is where a file says what it is FOR — "the
# reconciliation job's retry policy" — which its members' summaries, each written about one
# function, never state; a module summary written without it restates the members.
MODULE_HEADER_MAX_CHARS = 800
# The same header, shorter, in front of every function/class summarized from that file. A field
# report (15.09.2026) found a math module summarized as "a cryptographic sum structure using a
# signature path": its header comment said "tap on an operator sign", the function bodies did
# not, and the agent stopped trusting semantic search on that code. The first lines of a
# header name the domain; the cap keeps the added cost to a few dozen tokens per summary.
LEAF_HEADER_MAX_CHARS = 300

_COMMENT_PREFIXES = ("//", "/*", "*", "#!")


def _where(node: CodeNode) -> str:
    """Where a node lives, for a prompt: the path is context a name alone does not carry.

    ``run`` in ``billing/retry.ts`` and ``run`` in ``cli/main.ts`` are summarized from the same
    few lines of code; the path is what lets the summary say which subsystem it belongs to —
    and that vocabulary is what a semantic search for the subsystem later matches.
    """
    if not node.file_path:
        return ""
    noun = "directory" if node.node_type == NodeType.DIRECTORY else "file"
    return f" in the {noun} '{node.file_path}'"


def _leading_comment(root_path: Path, file_path: str) -> str:
    """The comment block a file opens with (after a shebang and blank lines), or ""."""
    try:
        with (root_path / file_path).open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(MODULE_HEADER_MAX_CHARS * 2)
    except OSError:
        return ""
    lines: list[str] = []
    for raw in head.splitlines():
        line = raw.strip()
        if not line:
            if lines:
                break
            continue
        if not line.startswith(_COMMENT_PREFIXES):
            break
        if not line.startswith("#!"):
            lines.append(line)
    return "\n".join(lines)


def module_header(node: CodeNode, root_path: Path | None) -> str:
    """What a module says about itself: its docstring, else its leading comment, capped.

    Read for the prompt only — never part of :func:`summary_input_hash`, so quoting it costs
    no re-summarization of an index built before it was quoted.
    """
    if node.node_type != NodeType.MODULE:
        return ""
    header = node.docstring.strip()
    if not header and root_path is not None:
        header = _leading_comment(root_path, node.file_path)
    return header[:MODULE_HEADER_MAX_CHARS]


# Bump when the *user message* a summary is generated from changes shape — what is quoted
# around the code (the path, the module header), not the code itself. The system prompts are
# hashed in below, so editing a prompt string needs no bump.
PROMPT_LAYOUT = 2


def _prompt_version() -> str:
    texts = sorted((name, value) for name, value in vars(prompts).items() if name.endswith("_SYSTEM_PROMPT"))
    digest = hashlib.sha256("\x00".join(f"{name}={value}" for name, value in texts).encode("utf-8"))
    return f"{PROMPT_LAYOUT}.{digest.hexdigest()[:8]}"


# Which prompts wrote a summary, stored on the node as ``summary_prompt_version``. Deliberately
# NOT part of :func:`summary_input_hash`: a prompt edit must never re-buy every summary in every
# project on its own. It is a record — which summaries predate a prompt change, so a deliberate,
# priced rebuild of one project can be decided from data (an empty value = written before
# versions were recorded).
SUMMARY_PROMPT_VERSION = _prompt_version()


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
    log_level: int = logging.INFO,
) -> set[str]:
    """Generate summaries bottom-up, reusing stored summaries whose input hash is unchanged.

    Returns the set of node ids whose summary was (re)generated. Mutates nodes in place,
    setting both ``summary`` and ``summary_input_hash``. *log_level* is where the progress
    lines go — :func:`plan_summaries` runs this same pass silently.

    ``on_batch`` is awaited after each *generated* batch (reused nodes never trigger it —
    they are already persisted). Exceptions from the callback propagate and abort the run:
    a caller that persists batches wants a dead DB to stop the spend, not to be swallowed.
    """
    stored = stored or {}
    bar = resolve(progress)
    # Declarations (DECLARATION_NODE_TYPES) are neither summarized nor part of their parent's
    # summary input — so a module that declares constants hashes exactly as it did before they
    # became nodes, and one that declares nothing else stays a leaf summarized from its source.
    declarations = {n.id for n in nodes if n.node_type in DECLARATION_NODE_TYPES}
    nodes = [n for n in nodes if n.id not in declarations]
    node_map = {n.id: n for n in nodes}

    children_map: dict[str, list[str]] = defaultdict(list)
    parent_map: dict[str, str] = {}
    parent_set: set[str] = set()
    for e in edges:
        if e.edge_type == EdgeType.CONTAINS and e.target not in declarations:
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
        logger.log(log_level, "No nodes require summarization")
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
            node.summary_prompt_version = str(prev.get("summary_prompt_version") or "")
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
        node.summary_prompt_version = str(prev.get("summary_prompt_version") or "") if prev_summary and prev else ""
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
                *[
                    _summarize_container(n, children_map, node_map, llm, max_source_chars, root_path=root_path)
                    for n in batch
                ],
                return_exceptions=True,
            )
            for node, result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    logger.warning("Failed to summarize container: %s: %s", node.id, result)
                    _keep_stale_on_failure(node, fallback="")
                else:
                    node.summary = str(result)
                    node.summary_prompt_version = SUMMARY_PROMPT_VERSION
                    dirty.add(node.id)
            if on_batch is not None:
                await on_batch(batch)
            _advance(len(batch))
            logger.log(log_level, "Summarized %d/%d nodes (%s)", done, total, label)

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

    modules = {n.file_path: n for n in nodes if n.node_type == NodeType.MODULE}
    headers: dict[str, str] = {}

    def _leaf_header(node: CodeNode) -> str:
        """The enclosing module's header for a member's prompt (a module leaf quotes its own source)."""
        module = modules.get(node.file_path)
        if module is None or module.id == node.id:
            return ""
        if node.file_path not in headers:
            headers[node.file_path] = module_header(module, root_path)[:LEAF_HEADER_MAX_CHARS]
        return headers[node.file_path]

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
            *[_summarize_leaf(n, root_path, llm, max_source_chars, header=_leaf_header(n)) for n in batch],
            return_exceptions=True,
        )
        for node, result in zip(batch, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("Failed to summarize leaf: %s: %s", node.id, result)
                _keep_stale_on_failure(node, fallback=node.docstring or "")
            else:
                node.summary = str(result)
                node.summary_prompt_version = SUMMARY_PROMPT_VERSION
                dirty.add(node.id)
            attempted.add(node.id)
        if on_batch is not None:
            await on_batch(batch)
        _advance(len(batch))
        logger.log(log_level, "Summarized %d/%d nodes (leaves)", done, total)
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
    logger.log(
        log_level, "Summarization complete: %d nodes, %d regenerated, %d reused",
        total, len(dirty), total - len(dirty),
    )
    return dirty


@dataclass(frozen=True)
class SummaryPlan:
    """What a :func:`summarize_hierarchy` run over the same inputs would send to the LLM.

    ``node_ids`` are the nodes it would (re)generate, ``calls`` the LLM requests that takes
    (a map-reduce container makes several) and ``prompt_chars`` their prompts and system
    prompts, measured on the prompts themselves rather than guessed from file sizes.
    """

    node_ids: frozenset[str]
    calls: int
    prompt_chars: int


class _CountingLLM:
    """Stands in for the LLM in a plan: measures each request, answers a placeholder.

    The placeholder is as long as a typical summary, because a container's prompt quotes its
    children's summaries — a short stub would under-measure every container above a change.
    """

    def __init__(self, placeholder_chars: int) -> None:
        self.calls = 0
        self.prompt_chars = 0
        self._placeholder = "x" * placeholder_chars

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        self.prompt_chars += len(prompt) + len(system_prompt)
        return self._placeholder


async def plan_summaries(
    nodes: list[CodeNode],
    edges: list[Edge],
    root_path: Path,
    *,
    stored: dict[str, dict[str, object]] | None,
    max_source_chars: int,
    placeholder_chars: int,
) -> SummaryPlan:
    """Which summaries a run would regenerate and what it would send, before it sends anything.

    Runs :func:`summarize_hierarchy` itself — same reuse-by-input-hash, same cascade to the
    containers above a change, same prompts — against a counting stand-in for the LLM, then
    puts the nodes back as they were. Because it is the real pass and not a model of it, it
    sees every reason a summary is redone: a changed file, an entity the parser newly
    recognizes, a cleared input hash, a prompt layout change. File-level estimates see only
    the first.

    One assumption makes it an upper bound rather than a replay: a regenerated summary is
    taken to differ from the stored one, so every container above it is counted. A real
    model virtually never returns the old text word for word, and a newly recognized entity
    changes its parent's input whatever its summary says. The bound holds for the tree as it
    stands now: leaves read their source from disk when summarized, so a file saved between
    this plan and the real pass can add the entities it touches.
    """
    saved = [(n, n.summary, n.summary_input_hash, n.summary_prompt_version) for n in nodes]
    counter = _CountingLLM(placeholder_chars)
    try:
        regenerated = await summarize_hierarchy(
            nodes, edges, root_path, counter,  # type: ignore[arg-type]
            batch_size=len(nodes) or 1,
            max_source_chars=max_source_chars,
            stored=stored,
            log_level=logging.DEBUG,
        )
    finally:
        for node, summary, input_hash, version in saved:
            node.summary, node.summary_input_hash, node.summary_prompt_version = summary, input_hash, version
    return SummaryPlan(frozenset(regenerated), counter.calls, counter.prompt_chars)


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
        return await _summarize_container(node, children_map, node_map, llm, max_source_chars, root_path=root_path)
    return await _summarize_leaf(node, root_path, llm, max_source_chars)


async def _summarize_leaf(
    node: CodeNode,
    root_path: Path,
    llm: LLMClient,
    max_source_chars: int,
    *,
    header: str = "",
) -> str:
    """Summarize a leaf node from its source code. Returns summary text.

    *header* is the enclosing module's header (see :data:`LEAF_HEADER_MAX_CHARS`) — prompt
    context only, never part of the input hash.
    """
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
        context = f"The file opens with this header:\n{header}\n\n" if header else ""
        prompt = (
            f"Summarize this {node.node_type.value} named '{node.name}'{_where(node)}:\n\n"
            f"{context}"
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
    *,
    root_path: Path | None = None,
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

    header = module_header(node, root_path)
    intro = f"Summarize this {node.node_type.value} '{node.name}'{_where(node)} based on its contents"
    if header:
        intro += f".\n\nIt opens with this header:\n{header}\n\nIts contents"
    contents = "\n".join(child_descriptions)
    if len(contents) > max_chars:
        return await _map_reduce_container(node, child_descriptions, llm, max_chars, intro=intro)

    prompt = f"{intro}:\n\n{contents}"
    summary = await llm.generate(prompt, system_prompt=prompts.CONTAINER_SYSTEM_PROMPT)
    return summary.strip()


async def _map_reduce_container(
    node: CodeNode,
    child_descriptions: list[str],
    llm: LLMClient,
    max_chars: int,
    *,
    intro: str = "",
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
            f"{node.node_type.value} '{node.name}'{_where(node)}:\n\n{contents}"
        )
        result = await llm.generate(prompt, system_prompt=prompts.CONTAINER_MAP_SYSTEM_PROMPT)
        return result.strip()

    partials = await asyncio.gather(*[_map_batch(i, b) for i, b in enumerate(batches)])

    combined = "\n".join(f"  [group] {p}" for p in partials)
    intro = intro or f"Summarize this {node.node_type.value} '{node.name}'{_where(node)} based on its contents"
    prompt = f"{intro}:\n\n{combined}"
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
            f"Summarize chunk {i + 1}/{len(chunks)} of {node.node_type.value} '{node.name}'{_where(node)}:\n\n"
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
