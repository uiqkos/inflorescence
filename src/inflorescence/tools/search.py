"""Search MCP tools."""

from __future__ import annotations

import logging
import re

from mcp.server.fastmcp import FastMCP

from inflorescence.config import Settings
from inflorescence.db.repository import GraphRepository
from inflorescence.project_manager import ProjectManager
from inflorescence.rag.embeddings import EmbeddingClient
from inflorescence.tools.path_filters import PathFilterConfig, filter_records_by_path
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


def _filters_narrow(filters: PathFilterConfig, node_types: list[str] | None) -> bool:
    """True when a mandatory filter (path include/exclude or node_type) can shrink results.

    Memgraph's vector/text search returns a bounded, similarity-ranked candidate pool;
    path/node_type filters run *after* that cut. When such a filter is active the pool may
    have dropped matching entities below the ceiling, so an empty/short result is ambiguous.
    """
    return bool(node_types) or not filters.include_all or bool(filters.exclude)


# Tantivy query-syntax metacharacters. `search_text` accepts plain
# keyword / identifier / error-text queries, so we neutralize these to spaces
# rather than let Tantivy parse them as field separators / boolean operators
# (which would otherwise raise a parse error and be misreported as "unavailable").
_TANTIVY_SPECIAL_CHARS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/<>=]')


def _sanitize_text_query(query: str) -> str:
    """Neutralize Tantivy metacharacters so literal terms search as plain keywords."""
    return " ".join(_TANTIVY_SPECIAL_CHARS.sub(" ", query).split())


def _code_hits(rows: list[dict[str, object]], min_score: float) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        score = float(row.get("score", 0) or 0)
        if score < min_score:
            continue
        node_id = str(row["id"])
        hit = grouped.setdefault(
            node_id,
            {"entity": entity_ref(row), "score": score, "source_type": "code", "matches": []},
        )
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
            hits.append({"entity": entity_ref(row), "score": score, "source_type": "summary", "matches": []})
    return sorted(hits, key=lambda hit: float(hit["score"]), reverse=True)


def _text_hits(rows: list[dict[str, object]], min_score: float, source_type: str) -> list[dict[str, object]]:
    hits = []
    for row in rows:
        score = float(row.get("score", 0) or 0)
        if score >= min_score:
            hits.append({"entity": entity_ref(row), "score": score, "source_type": source_type, "matches": []})
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


async def _not_indexed_error(manager: ProjectManager, project: str, directory: str) -> dict | None:
    """Return a distinct error when *directory* was never indexed, else None.

    ``get_project`` synthesises a valid id from any path, so a typo or an adjacent directory
    otherwise yields an empty result indistinguishable from a genuine "no matches" (audit
    finding T5 / INV-7). Surfacing it as an error tells the caller to index first instead of
    trusting a silent empty page.
    """
    if not await manager.project_is_indexed(project):
        return error_response(
            "project_not_indexed",
            f"No index found for {directory!r} (project '{project}'). Index it with index_directory "
            "first — this is different from an indexed project that simply has no matches.",
        )
    return None


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
        min_score: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Search code chunk embeddings and return entity hits.

        Chunks are embedded in the final phase of an index run: while a first-time index
        is still running this finds nothing yet — that is "not indexed yet", not "no
        matches". ``search_entities`` (name-based) works as soon as the run starts.
        """
        project = manager.get_project(directory)
        not_indexed = await _not_indexed_error(manager, project, directory)
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
        truncated = len(raw_rows) >= candidate_k and _filters_narrow(filters, node_types)
        rows = _filter_node_types(filter_records_by_path(raw_rows, filters), node_types)
        return _page_hits(
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
        )

    @mcp.tool()
    async def search_semantic(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        min_score: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Search summary embeddings by meaning (semantic similarity) and return entity hits.

        Use this to find code by *what it does* even when the wording differs. For exact
        identifier / string-literal / error-text matches, use ``search_text`` instead.

        While an index run is in progress, only nodes whose summaries already landed are
        searchable — fewer hits then means "not summarized yet", not "no matches".
        """
        project = manager.get_project(directory)
        not_indexed = await _not_indexed_error(manager, project, directory)
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
        truncated = len(raw_rows) >= candidate_k and _filters_narrow(filters, node_types)
        rows = _filter_node_types(filter_records_by_path(raw_rows, filters), node_types)
        return _page_hits(
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
        )

    @mcp.tool()
    async def search_hybrid(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        min_score: float = 0.0,
        limit: int = 10,
        offset: int = 0,
    ) -> dict:
        """Search code and summary embeddings, deduplicated by entity.

        While an index run is in progress, coverage is partial: summary vectors exist only
        for already-summarized nodes, and code-chunk vectors land in the run's final phase.
        """
        project = manager.get_project(directory)
        not_indexed = await _not_indexed_error(manager, project, directory)
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
            and _filters_narrow(filters, node_types)
        )
        code_rows = _filter_node_types(filter_records_by_path(raw_code, filters), node_types)
        summary_rows = _filter_node_types(filter_records_by_path(raw_summary, filters), node_types)

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
        return _page_hits(
            hits,
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
        )

    @mcp.tool()
    async def search_text(
        directory: str,
        query: str,
        node_types: list[str] | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
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
        compact entity refs with pagination — never source code or embeddings.

        While an index run is in progress, summary and raw-code fields cover only what has
        already landed; name/signature hits are complete as soon as the skeleton is stored.
        """
        project = manager.get_project(directory)
        not_indexed = await _not_indexed_error(manager, project, directory)
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
            and _filters_narrow(filters, node_types)
        )
        summary_rows = _filter_node_types(filter_records_by_path(summary_rows, filters), node_types)
        symbol_rows = _filter_node_types(filter_records_by_path(symbol_rows, filters), node_types)
        code_rows = _filter_node_types(filter_records_by_path(code_rows, filters), node_types)

        return _page_hits(
            _merge_text_hits(summary_rows, symbol_rows, code_rows, min_score),
            bounded_limit,
            bounded_offset,
            meta,
            truncated=truncated,
        )
