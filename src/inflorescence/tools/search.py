"""Search MCP tools."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from inflorescence.config import Settings
from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import ProjectManager
from inflorescence.rag.embeddings import EmbeddingClient
from inflorescence.tools.coverage import note_coverage
from inflorescence.tools.freshness import build_checker, describe_origin, unchecked_empty_page
from inflorescence.tools.gates import ensure_indexed, normalize_node_types
from inflorescence.tools.path_filters import PathFilterConfig, filter_records_by_path, is_test_path
from inflorescence.tools.responses import (
    MAX_SEARCH_LIMIT,
    clamp_limit,
    clamp_offset,
    entity_ref,
    error_response,
    paginated_response,
)

logger = logging.getLogger(__name__)


def _overfetch(limit: int, offset: int) -> int:
    return max(100, min((offset + limit + 1) * 5, 500))


def _filters_narrow(filters: PathFilterConfig, node_types: list[str] | None, include_tests: bool = True) -> bool:
    """True when a mandatory filter (path, node_type, or tests) can shrink results.

    Memgraph's vector/text search returns a bounded, similarity-ranked candidate pool;
    path/node_type/test filters run *after* that cut. When such a filter is active the pool
    may have dropped matching entities below the ceiling, so an empty/short result is
    ambiguous — and must be reported as truncated rather than as "nothing matches".
    """
    return bool(node_types) or not include_tests or not filters.include_all or bool(filters.exclude)


# Tantivy query-syntax metacharacters. `search_text` accepts plain
# keyword / identifier / error-text queries, so we neutralize these to spaces
# rather than let Tantivy parse them as field separators / boolean operators
# (which would otherwise raise a parse error and be misreported as "unavailable").
_TANTIVY_SPECIAL_CHARS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/<>=]')


def _sanitize_text_query(query: str) -> str:
    """Neutralize Tantivy metacharacters so literal terms search as plain keywords."""
    return " ".join(_TANTIVY_SPECIAL_CHARS.sub(" ", query).split())


# A document or a long function can match on a dozen chunks; past the first few the list
# is repetition, not evidence. Keep the best few and say how many there were.
MAX_HIT_MATCHES = 3


def _new_hit(row: dict[str, object], score: float, source_type: str) -> dict[str, object]:
    """Build one hit, flagging it when its *path* says test code (heuristic, see is_test_path)."""
    hit: dict[str, object] = {"entity": entity_ref(row), "score": score, "source_type": source_type, "matches": []}
    if is_test_path(str(row.get("file_path") or "")):
        hit["is_test"] = True
    return hit


def _drop_test_rows(rows: list[dict[str, object]], include_tests: bool) -> list[dict[str, object]]:
    """Remove test-like rows *before* hits are built, so pagination pages the kept set."""
    if include_tests:
        return rows
    return [row for row in rows if not is_test_path(str(row.get("file_path") or ""))]


def _cap_matches(hits: list[dict[str, object]]) -> list[dict[str, object]]:
    """Trim every hit's match list to the best MAX_HIT_MATCHES, recording the true count.

    ``matches_total`` appears only where the trim actually happened — its presence *is* the
    statement "there were more", so an absent field means the list is complete.
    """
    for hit in hits:
        matches = hit["matches"]
        if len(matches) <= MAX_HIT_MATCHES:
            continue
        ordered = sorted(matches, key=lambda match: float(match.get("score") or 0), reverse=True)
        hit["matches"] = ordered[:MAX_HIT_MATCHES]
        hit["matches_total"] = len(matches)
    return hits


def _code_hits(rows: list[dict[str, object]], min_score: float) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        score = float(row.get("score", 0) or 0)
        if score < min_score:
            continue
        node_id = str(row["id"])
        hit = grouped.setdefault(node_id, _new_hit(row, score, "code"))
        hit["score"] = max(float(hit["score"]), score)
        hit["matches"].append({
            "chunk_id": row.get("chunk_id"),
            "chunk_kind": row.get("chunk_kind"),
            "line_start": row.get("match_start_line"),
            "line_end": row.get("match_end_line"),
            "score": score,
        })
    return sorted(grouped.values(), key=lambda hit: float(hit["score"]), reverse=True)


def _summary_hits(rows: list[dict[str, object]], min_score: float) -> list[dict[str, object]]:
    hits = []
    for row in rows:
        score = float(row.get("score", 0) or 0)
        if score >= min_score:
            hits.append(_new_hit(row, score, "summary"))
    return sorted(hits, key=lambda hit: float(hit["score"]), reverse=True)


def _text_hits(rows: list[dict[str, object]], min_score: float, source_type: str) -> list[dict[str, object]]:
    hits = []
    for row in rows:
        score = float(row.get("score", 0) or 0)
        if score >= min_score:
            hits.append(_new_hit(row, score, source_type))
    return hits


def _merge_text_hits(
    summary_rows: list[dict[str, object]],
    symbol_rows: list[dict[str, object]],
    code_rows: list[dict[str, object]],
    min_score: float,
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    field_hits = [
        *_code_hits(code_rows, min_score),
        *_text_hits(summary_rows, min_score, "summary"),
        *_text_hits(symbol_rows, min_score, "name"),
    ]
    for hit in field_hits:
        entity = hit["entity"] or {}
        node_id = str(entity.get("id", ""))
        existing = merged.get(node_id)
        if existing is None:
            merged[node_id] = hit
            continue
        existing["score"] = max(float(existing["score"]), float(hit["score"]))
        existing["source_type"] = "text"
        existing["matches"].extend(hit["matches"])
    return sorted(merged.values(), key=lambda hit: float(hit["score"]), reverse=True)


def _filter_node_types(rows: list[dict[str, object]], node_types: list[str] | None) -> list[dict[str, object]]:
    if not node_types:
        return rows
    allowed = set(node_types)
    return [row for row in rows if row.get("node_type") in allowed]


def _page_hits(
    hits: list[dict[str, object]],
    limit: int,
    offset: int,
    meta: dict[str, object],
    truncated: bool = False,
) -> dict[str, object]:
    return paginated_response(hits[offset : offset + limit + 1], limit, offset, meta, truncated=truncated)


def _flag_weak(payload: dict, threshold: float) -> dict:
    """Mark every hit on the page that scored below *threshold*, and say when all of them did.

    A vector search returns its top-k whatever the corpus holds, so a query about code that
    does not exist comes back looking exactly like a query about code that does — same shape,
    same ranking, merely lower numbers. The numbers are already there; what was missing was
    the reading of them, which every caller had to hard-code a cut-off to get.

    Only weak hits carry ``"weak": true`` (the flag convention: absence is the common case),
    and ``meta.weak_results`` appears only when the whole page is weak — the one arrangement
    that is worth acting on, because a page with even one strong hit still has an answer on
    it. Nothing is removed: this is a label, not the ``min_score`` cut. An empty page gets
    neither flag — no hits is not the same claim as "the hits were far away".

    BM25 scores are unbounded relevance, not similarities, so ``search_text`` never calls this.
    """
    items = [item for item in payload["items"] if isinstance(item, dict)]
    weak = [item for item in items if float(item.get("score") or 0) < threshold]
    for item in weak:
        item["weak"] = True
    if items and len(weak) == len(items):
        payload["meta"]["weak_results"] = True
    return payload


def _embed_one(embedder: EmbeddingClient, query: str, label: str) -> list[float] | dict[str, dict[str, str]]:
    try:
        embeddings = embedder.embed([query])
    except Exception as exc:
        logger.exception("%s embedding failed for query: %s", label, query)
        return error_response("embedding_failed", f"Failed to embed query: {exc}")
    if not embeddings or embeddings[0] is None:
        return error_response(
            "embedding_failed",
            f"Embedding returned None (model={embedder.model}). Check API key and model availability.",
        )
    return embeddings[0]


def register_search_tools(
    mcp: FastMCP,
    manager: ProjectManager,
    repo: GraphRepository,
    settings: Settings,
) -> None:
    """Register search tools with the MCP server."""

    code_embedder = EmbeddingClient(settings=settings, model=settings.code_embedding_model)
    summary_embedder = EmbeddingClient(settings=settings, model=settings.summary_embedding_model)

    @mcp.tool()
    async def search_code(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        min_score: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Search code chunk embeddings and return entity hits.

        ``query`` is embedded into the same vector space as the indexed code, which is
        English-dominated: phrase it in English — a query in another language lands far from
        that space and returns an irrelevant top result, not an error. For exact strings,
        identifiers, or non-English text, use ``search_text`` (BM25 keyword search) instead.

        Chunks are embedded in the final phase of an index run: while a first-time index
        is still running this finds nothing yet — that is "not indexed yet", not "no
        matches". ``search_entities`` (name-based) works as soon as the run starts.

        A first query against a git worktree of an indexed checkout that is not indexed
        yet indexes it inline before answering: that one call is noticeably slower
        (parsing and graph writes only — summaries and embeddings are reused from the
        main checkout, so no LLM spend), and every later call is normal speed.

        ``include_paths``/``exclude_paths`` are globs; a pattern with no glob character
        (``"docs/"``, ``"docs"``) covers both that exact path and everything under it, and
        ``meta.path_filters`` echoes the expanded form that was actually applied.

        A hit whose *path* looks like test code (``tests/``, ``spec/``, ``test_*.py``,
        ``*_test.go``, ``conftest.py``, ``*.test.ts``, …) carries ``"is_test": true``. That
        is a heuristic over the path only — it never reads the file — so treat it as a hint,
        not a verdict.

        RECOMMENDED: pass ``include_tests=false`` whenever you are looking for an
        implementation. A test's summary restates the vocabulary of the code it exercises
        ("Verifies that retrying a failed upload reuses the original storage key"), so
        tests out-rank that code on exactly the "what does X do" queries used to find it —
        a field report measured 11 of 15 hits from
        ``tests/`` while the runner named in the query never reached page 1. Example:
        ``search_semantic(directory="/repo", query="job runner start stop",
        include_tests=false)``. For a layout the heuristic does not catch, name the
        directories instead: ``exclude_paths=["tests/**", "fixtures/**"]``. Keep the default
        when the test IS what you want — a usage example, an expected behaviour, a repro.

        A vector search always returns its nearest neighbours, so low scores still look like
        results: a hit below the weak-similarity threshold carries ``"weak": true``, and
        ``meta.weak_results`` is set when *every* hit on the page is weak — nothing matched
        closely, so the code probably is not there; confirm absence with ``search_text`` on
        exact terms. (This only labels; ``min_score`` is the separate hard cut-off.)

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page, and affected hits carry ``"stale": true`` / ``"file_missing": true`` on
        their entity — so an unflagged hit was checked and still matches the index. This is
        file-level: "stale" means the file changed after indexing (lines may have shifted,
        the summary may describe older code), not that this entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        An EMPTY first page carries one more block, ``meta.index_coverage``: the index's file
        count plus a bounded re-check of the directories it knows against the filesystem. It
        either names files that are on disk and NOT in this index — an empty page over one of
        those is a coverage gap, not absence: read them directly and ``index_directory`` to
        make them queryable — or says nothing under that scope is missing, which is what
        licenses the conclusion "this really does not exist here". Coverage is not freshness:
        a file created since the last index run has no node to flag, so ``index_freshness``
        stays clean and the ``watcher`` line stays reassuring while the file sits unqueryable.
        ``meta.nested_checkouts`` appears when the files on the page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) —
        that checkout's HEAD, not ``answered_from``, is the code you are reading.
        """
        # Argument validation first: rejecting a typo'd node type must not trigger the
        # manager's inline (donor-seeded) index of a fresh worktree.
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        embedding = _embed_one(code_embedder, query, "Code")
        if isinstance(embedding, dict):
            return embedding

        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        candidate_k = _overfetch(bounded_limit, bounded_offset)
        raw_rows = await repo.search_code_vectors(project, embedding, candidate_k)
        truncated = len(raw_rows) >= candidate_k and _filters_narrow(filters, node_types, include_tests)
        rows = _drop_test_rows(_filter_node_types(filter_records_by_path(raw_rows, filters), node_types), include_tests)
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(_flag_weak(_page_hits(
            _code_hits(rows, min_score),
            bounded_limit,
            bounded_offset,
            {
                "project": project,
                "query": query,
                "min_score": min_score,
                "node_types": node_types or [],
                "path_filters": filters.meta(),
            },
            truncated=truncated,
        ), settings.semantic_weak_threshold)), manager, directory, checker)

    @mcp.tool()
    async def search_semantic(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        min_score: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Search summary embeddings by meaning (semantic similarity) and return entity hits.

        Use this to find code by *what it does* even when the wording differs. For exact
        identifier / string-literal / error-text matches, use ``search_text`` instead.

        Summaries and their embeddings are English-only, so phrase ``query`` in English —
        a query in another language (e.g. Russian) embeds far from the English summary
        space and returns an irrelevant top result, not an error. For exact strings,
        identifiers, or non-English text, use ``search_text`` (BM25 keyword search) instead.

        While an index run is in progress, only nodes whose summaries already landed are
        searchable — fewer hits then means "not summarized yet", not "no matches".

        A first query against a git worktree of an indexed checkout that is not indexed
        yet indexes it inline before answering: that one call is noticeably slower
        (parsing and graph writes only — summaries and embeddings are reused from the
        main checkout, so no LLM spend), and every later call is normal speed.

        ``include_paths``/``exclude_paths`` are globs; a pattern with no glob character
        (``"docs/"``, ``"docs"``) covers both that exact path and everything under it, and
        ``meta.path_filters`` echoes the expanded form that was actually applied.

        A hit whose *path* looks like test code (``tests/``, ``spec/``, ``test_*.py``,
        ``*_test.go``, ``conftest.py``, ``*.test.ts``, …) carries ``"is_test": true``. That
        is a heuristic over the path only — it never reads the file — so treat it as a hint,
        not a verdict.

        RECOMMENDED: pass ``include_tests=false`` whenever you are looking for an
        implementation. A test's summary restates the vocabulary of the code it exercises
        ("Verifies that retrying a failed upload reuses the original storage key"), so
        tests out-rank that code on exactly the "what does X do" queries used to find it —
        a field report measured 11 of 15 hits from
        ``tests/`` while the runner named in the query never reached page 1. Example:
        ``search_semantic(directory="/repo", query="job runner start stop",
        include_tests=false)``. For a layout the heuristic does not catch, name the
        directories instead: ``exclude_paths=["tests/**", "fixtures/**"]``. Keep the default
        when the test IS what you want — a usage example, an expected behaviour, a repro.

        A vector search always returns its nearest neighbours, so low scores still look like
        results: a hit below the weak-similarity threshold carries ``"weak": true``, and
        ``meta.weak_results`` is set when *every* hit on the page is weak — nothing matched
        closely, so the code probably is not there; confirm absence with ``search_text`` on
        exact terms. (This only labels; ``min_score`` is the separate hard cut-off.)

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page, and affected hits carry ``"stale": true`` / ``"file_missing": true`` on
        their entity — so an unflagged hit was checked and still matches the index. This is
        file-level: "stale" means the file changed after indexing (lines may have shifted,
        the summary may describe older code), not that this entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        An EMPTY first page carries one more block, ``meta.index_coverage``: the index's file
        count plus a bounded re-check of the directories it knows against the filesystem. It
        either names files that are on disk and NOT in this index — an empty page over one of
        those is a coverage gap, not absence: read them directly and ``index_directory`` to
        make them queryable — or says nothing under that scope is missing, which is what
        licenses the conclusion "this really does not exist here". Coverage is not freshness:
        a file created since the last index run has no node to flag, so ``index_freshness``
        stays clean and the ``watcher`` line stays reassuring while the file sits unqueryable.
        ``meta.nested_checkouts`` appears when the files on the page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) —
        that checkout's HEAD, not ``answered_from``, is the code you are reading.
        """
        # Argument validation first: rejecting a typo'd node type must not trigger the
        # manager's inline (donor-seeded) index of a fresh worktree.
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        embedding = _embed_one(summary_embedder, query, "Summary")
        if isinstance(embedding, dict):
            return embedding

        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        candidate_k = _overfetch(bounded_limit, bounded_offset)
        raw_rows = await repo.search_summary_vectors(project, embedding, candidate_k)
        truncated = len(raw_rows) >= candidate_k and _filters_narrow(filters, node_types, include_tests)
        rows = _drop_test_rows(_filter_node_types(filter_records_by_path(raw_rows, filters), node_types), include_tests)
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(_flag_weak(_page_hits(
            _summary_hits(rows, min_score),
            bounded_limit,
            bounded_offset,
            {
                "project": project,
                "query": query,
                "min_score": min_score,
                "node_types": node_types or [],
                "path_filters": filters.meta(),
            },
            truncated=truncated,
        ), settings.semantic_weak_threshold)), manager, directory, checker)

    @mcp.tool()
    async def search_hybrid(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        min_score: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Search code and summary embeddings, deduplicated by entity.

        Summary embeddings (and code identifiers they're matched against) are
        English-only, so phrase ``query`` in English — a non-English query (e.g.
        Russian) embeds far from that space and returns an irrelevant top result,
        not an error. For exact strings, identifiers, or non-English text, use
        ``search_text`` (BM25 keyword search) instead.

        While an index run is in progress, coverage is partial: summary vectors exist only
        for already-summarized nodes, and code-chunk vectors land in the run's final phase.

        A hit lists at most 3 chunk ``matches`` — the highest-scoring ones. ``matches_total``
        gives the real count and is present only when the list was trimmed.

        A first query against a git worktree of an indexed checkout that is not indexed
        yet indexes it inline before answering: that one call is noticeably slower
        (parsing and graph writes only — summaries and embeddings are reused from the
        main checkout, so no LLM spend), and every later call is normal speed.

        ``include_paths``/``exclude_paths`` are globs; a pattern with no glob character
        (``"docs/"``, ``"docs"``) covers both that exact path and everything under it, and
        ``meta.path_filters`` echoes the expanded form that was actually applied.

        A hit whose *path* looks like test code (``tests/``, ``spec/``, ``test_*.py``,
        ``*_test.go``, ``conftest.py``, ``*.test.ts``, …) carries ``"is_test": true``. That
        is a heuristic over the path only — it never reads the file — so treat it as a hint,
        not a verdict.

        RECOMMENDED: pass ``include_tests=false`` whenever you are looking for an
        implementation. A test's summary restates the vocabulary of the code it exercises
        ("Verifies that retrying a failed upload reuses the original storage key"), so
        tests out-rank that code on exactly the "what does X do" queries used to find it —
        a field report measured 11 of 15 hits from
        ``tests/`` while the runner named in the query never reached page 1. Example:
        ``search_semantic(directory="/repo", query="job runner start stop",
        include_tests=false)``. For a layout the heuristic does not catch, name the
        directories instead: ``exclude_paths=["tests/**", "fixtures/**"]``. Keep the default
        when the test IS what you want — a usage example, an expected behaviour, a repro.

        A vector search always returns its nearest neighbours, so low scores still look like
        results: a hit below the weak-similarity threshold carries ``"weak": true``, and
        ``meta.weak_results`` is set when *every* hit on the page is weak — nothing matched
        closely, so the code probably is not there; confirm absence with ``search_text`` on
        exact terms. (This only labels; ``min_score`` is the separate hard cut-off.)

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page, and affected hits carry ``"stale": true`` / ``"file_missing": true`` on
        their entity — so an unflagged hit was checked and still matches the index. This is
        file-level: "stale" means the file changed after indexing (lines may have shifted,
        the summary may describe older code), not that this entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        An EMPTY first page carries one more block, ``meta.index_coverage``: the index's file
        count plus a bounded re-check of the directories it knows against the filesystem. It
        either names files that are on disk and NOT in this index — an empty page over one of
        those is a coverage gap, not absence: read them directly and ``index_directory`` to
        make them queryable — or says nothing under that scope is missing, which is what
        licenses the conclusion "this really does not exist here". Coverage is not freshness:
        a file created since the last index run has no node to flag, so ``index_freshness``
        stays clean and the ``watcher`` line stays reassuring while the file sits unqueryable.
        ``meta.nested_checkouts`` appears when the files on the page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) —
        that checkout's HEAD, not ``answered_from``, is the code you are reading.
        """
        # Argument validation first: rejecting a typo'd node type must not trigger the
        # manager's inline (donor-seeded) index of a fresh worktree.
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        code_embedding = _embed_one(code_embedder, query, "Code")
        if isinstance(code_embedding, dict):
            return code_embedding
        summary_embedding = _embed_one(summary_embedder, query, "Summary")
        if isinstance(summary_embedding, dict):
            return summary_embedding

        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        top_k = _overfetch(bounded_limit, bounded_offset)
        raw_code = await repo.search_code_vectors(project, code_embedding, top_k)
        raw_summary = await repo.search_summary_vectors(project, summary_embedding, top_k)
        truncated = (
            (len(raw_code) >= top_k or len(raw_summary) >= top_k)
            and _filters_narrow(filters, node_types, include_tests)
        )
        code_rows = _drop_test_rows(_filter_node_types(filter_records_by_path(raw_code, filters), node_types), include_tests)
        summary_rows = _drop_test_rows(
            _filter_node_types(filter_records_by_path(raw_summary, filters), node_types), include_tests
        )

        merged: dict[str, dict[str, object]] = {}
        for hit in [*_code_hits(code_rows, min_score), *_summary_hits(summary_rows, min_score)]:
            entity = hit["entity"] or {}
            node_id = str(entity.get("id", ""))
            existing = merged.get(node_id)
            if existing is None:
                merged[node_id] = hit
                continue
            if float(hit["score"]) > float(existing["score"]):
                existing["score"] = hit["score"]
            existing["source_type"] = "hybrid"
            existing["matches"].extend(hit["matches"])

        hits = sorted(merged.values(), key=lambda hit: float(hit["score"]), reverse=True)
        for hit in hits:
            if hit["source_type"] != "hybrid" and hit["matches"]:
                hit["source_type"] = "hybrid"
        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(_flag_weak(_page_hits(
            _cap_matches(hits),
            bounded_limit,
            bounded_offset,
            {
                "project": project,
                "query": query,
                "min_score": min_score,
                "node_types": node_types or [],
                "path_filters": filters.meta(),
            },
            truncated=truncated,
        ), settings.semantic_weak_threshold)), manager, directory, checker)

    @mcp.tool()
    async def search_text(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        include_tests: bool = True,
        min_score: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Exact keyword (BM25) full-text search over symbol names, signatures, summaries, and raw code.

        Use this to find *literal* terms — a specific identifier, a string literal, or the text of
        an error message — matched by keyword. For meaning-based search (find code that *does*
        something, regardless of wording) use ``search_semantic`` instead. Scores are BM25
        relevance scores (unbounded, larger = better), not 0..1 similarities. Query metacharacters
        (``:``, ``[]``, ``+``, ``-``, ``*``, …) are treated as separators, so terms like
        ``ValueError: invalid literal`` or ``arr[0]`` match as plain keywords. Hits on raw code
        chunks are aggregated up to their owning entity, then deduplicated across fields. Returns
        compact entity refs with pagination — never source code or embeddings. A hit lists at most
        3 ``matches`` (the highest-scoring ones); ``matches_total`` gives the real count and is
        present only when the list was trimmed — a long document matches on many chunks and the
        tail of that list is repetition, not evidence.

        A multi-word phrase in any language is tokenized into independent, stemmed words with
        no phrase matching (Russian ``"Загружаем доску"`` also matches anything about
        ``загруж…`` or ``доск…``), so the top hit is usually right while the tail can be token
        noise — narrow with the most distinctive single word, or add ``include_paths``.

        While an index run is in progress, summary and raw-code fields cover only what has
        already landed; name/signature hits are complete as soon as the skeleton is stored.

        A first query against a git worktree of an indexed checkout that is not indexed
        yet indexes it inline before answering: that one call is noticeably slower
        (parsing and graph writes only — summaries and embeddings are reused from the
        main checkout, so no LLM spend), and every later call is normal speed.

        ``include_paths``/``exclude_paths`` are globs; a pattern with no glob character
        (``"docs/"``, ``"docs"``) covers both that exact path and everything under it, and
        ``meta.path_filters`` echoes the expanded form that was actually applied.

        A hit whose *path* looks like test code (``tests/``, ``spec/``, ``test_*.py``,
        ``*_test.go``, ``conftest.py``, ``*.test.ts``, …) carries ``"is_test": true``. That
        is a heuristic over the path only — it never reads the file — so treat it as a hint,
        not a verdict.

        RECOMMENDED: pass ``include_tests=false`` whenever you are looking for an
        implementation. A test's summary restates the vocabulary of the code it exercises
        ("Verifies that retrying a failed upload reuses the original storage key"), so
        tests out-rank that code on exactly the "what does X do" queries used to find it —
        a field report measured 11 of 15 hits from
        ``tests/`` while the runner named in the query never reached page 1. Example:
        ``search_semantic(directory="/repo", query="job runner start stop",
        include_tests=false)``. For a layout the heuristic does not catch, name the
        directories instead: ``exclude_paths=["tests/**", "fixtures/**"]``. Keep the default
        when the test IS what you want — a usage example, an expected behaviour, a repro.

        ``meta.answered_from`` names the tree this answer came from: path, branch @ short
        commit, and — for a git worktree — the checkout it belongs to.
        ``meta.index_freshness`` ({checked, stale, missing, unknown}) covers the files on
        this page, and affected hits carry ``"stale": true`` / ``"file_missing": true`` on
        their entity — so an unflagged hit was checked and still matches the index. This is
        file-level: "stale" means the file changed after indexing (lines may have shifted,
        the summary may describe older code), not that this entity changed.
        ``index_freshness.live_worktrees``, when present, is a sentence naming how many git
        worktrees of this checkout are NOT in this index — their branch state is elsewhere;
        ask with ``directory`` set to the worktree itself, which indexes it almost for free
        (no LLM spend).

        An EMPTY first page carries one more block, ``meta.index_coverage``: the index's file
        count plus a bounded re-check of the directories it knows against the filesystem. It
        either names files that are on disk and NOT in this index — an empty page over one of
        those is a coverage gap, not absence: read them directly and ``index_directory`` to
        make them queryable — or says nothing under that scope is missing, which is what
        licenses the conclusion "this really does not exist here". Coverage is not freshness:
        a file created since the last index run has no node to flag, so ``index_freshness``
        stays clean and the ``watcher`` line stays reassuring while the file sits unqueryable.
        ``meta.nested_checkouts`` appears when the files on the page live in an independent
        git checkout inside this tree (a submodule or vendored repo on its own branch) —
        that checkout's HEAD, not ``answered_from``, is the code you are reading.
        """
        # Argument validation first: rejecting a typo'd node type must not trigger the
        # manager's inline (donor-seeded) index of a fresh worktree.
        node_types = normalize_node_types(node_types)
        if isinstance(node_types, dict):
            return node_types
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        top_k = _overfetch(bounded_limit, bounded_offset)
        meta = {
            "project": project,
            "query": query,
            "min_score": min_score,
            "node_types": node_types or [],
            "path_filters": filters.meta(),
        }
        text_query = _sanitize_text_query(query)
        if not text_query:
            # Nothing to search, so nothing to hash — but the blocks still ship, since a
            # caller may only read the item flags as "fresh" if they are always present.
            meta["answered_from"] = describe_origin(Path(directory).resolve())
            meta["index_freshness"] = unchecked_empty_page(
                directory, await repo.get_watch_state(project)
            )
            return _page_hits([], bounded_limit, bounded_offset, meta)
        try:
            summary_rows = await repo.search_summary_text(project, text_query, top_k)
            symbol_rows = await repo.search_symbol_text(project, text_query, top_k)
            code_rows = await repo.search_code_text(project, text_query, top_k)
        except Exception as exc:
            logger.exception("Text search failed for query: %s", query)
            detail = str(exc).lower()
            if any(marker in detail for marker in ("text_search", "text index", "does not exist", "no procedure")):
                return error_response(
                    "text_search_unavailable",
                    f"Full-text search is unavailable — it requires the Memgraph MAGE image with text "
                    f"indexes (summary_text, code_symbols_text, chunk_content_text): {exc}",
                )
            return error_response("text_search_failed", f"Text search failed: {exc}")

        truncated = (
            max(len(summary_rows), len(symbol_rows), len(code_rows)) >= top_k
            and _filters_narrow(filters, node_types, include_tests)
        )
        summary_rows = _drop_test_rows(_filter_node_types(filter_records_by_path(summary_rows, filters), node_types), include_tests)
        symbol_rows = _drop_test_rows(_filter_node_types(filter_records_by_path(symbol_rows, filters), node_types), include_tests)
        code_rows = _drop_test_rows(_filter_node_types(filter_records_by_path(code_rows, filters), node_types), include_tests)

        checker = await build_checker(repo, project, directory)
        return await note_coverage(checker.stamp(_page_hits(
            _cap_matches(_merge_text_hits(summary_rows, symbol_rows, code_rows, min_score)),
            bounded_limit,
            bounded_offset,
            meta,
            truncated=truncated,
        )), manager, directory, checker)
