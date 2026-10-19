"""Search MCP tools."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from inflorescence.config import Settings
from inflorescence.db.repository import GraphRepository, ProjectHits, RowFilter
from inflorescence.project_manager import ProjectManager
from inflorescence.rag.chunker import VERBATIM_CHUNK_KINDS
from inflorescence.rag.embeddings import EmbeddingClient
from inflorescence.tools import descriptions as doc
from inflorescence.tools.archive import demote_archived, load_archive
from inflorescence.tools.coverage import note_coverage
from inflorescence.tools.freshness import build_checker, describe_origin, unchecked_empty_page
from inflorescence.tools.gates import ensure_indexed, normalize_node_types
from inflorescence.tools.hints import (
    add_hints,
    archived_hint,
    page_shape_hints,
    unmatched_terms_hint,
    weak_page_hint,
)
from inflorescence.tools.path_filters import PathFilterConfig, filter_records_by_path, is_test_path
from inflorescence.tools.responses import (
    MAX_SEARCH_LIMIT,
    clamp_limit,
    clamp_offset,
    entity_ref,
    error_response,
    paginated_response,
)
from inflorescence.tools.text_terms import (
    fuzzy_distance_for,
    fuzzy_matching_lines,
    matching_lines,
    query_words,
    tokens,
)

logger = logging.getLogger(__name__)

Hit = dict[str, Any]


def _overfetch(limit: int, offset: int) -> int:
    """How many entities a page asks each index for: every one up to the page's end plus one
    (so ``has_more`` is known), and never fewer than 100, so a fusion of several indexes ranks
    over more than the page itself. The repository counts entities, not chunks."""
    return max(100, offset + limit + 1)


def _row_filter(filters: PathFilterConfig, node_types: list[str] | None, include_tests: bool) -> RowFilter:
    """Everything the caller filters on besides the project, as one function over raw rows.

    Handed to the repository so the filters run *inside* its widening loop: a filter that
    removes rows is exactly what can leave a page unfilled while matches remain outside the
    candidate window, and only the loop can do something about that.
    """

    def keep(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return _drop_test_rows(_filter_node_types(filter_records_by_path(rows, filters), node_types), include_tests)

    return keep


# Tantivy query-syntax metacharacters. `search_text` accepts plain
# keyword / identifier / error-text queries, so we neutralize these to spaces
# rather than let Tantivy parse them as field separators / boolean operators
# (which would otherwise raise a parse error and be misreported as "unavailable").
_TANTIVY_SPECIAL_CHARS = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/<>=]')


def _sanitize_text_query(query: str) -> str:
    """Neutralize Tantivy metacharacters so literal terms search as plain keywords.

    Lower-cased as well: the index lower-cases every token anyway, so this changes no match,
    but it stops Tantivy from reading an upper-case ``AND`` / ``OR`` / ``NOT`` in pasted text
    (``NOT_FOUND AND retry``) as a boolean operator.
    """
    return " ".join(_TANTIVY_SPECIAL_CHARS.sub(" ", query).split()).lower()


# A document or a long function can match on a dozen chunks; past the first few the list
# is repetition, not evidence. Keep the best few and say how many there were.
MAX_HIT_MATCHES = 3

# How many distinct query words are checked for "matched nothing". A query longer than this
# is prose, and the first words are the ones a reader chose deliberately.
_MAX_WORDS_PROBED = 8


def _new_hit(row: Mapping[str, object], score: float, source_type: str) -> Hit:
    """Build one hit, flagging it when its *path* says test code (heuristic, see is_test_path)."""
    hit: Hit = {"entity": entity_ref(row), "score": score, "source_type": source_type, "matches": []}
    if is_test_path(str(row.get("file_path") or "")):
        hit["is_test"] = True
    return hit


def _drop_test_rows(rows: list[dict[str, object]], include_tests: bool) -> list[dict[str, object]]:
    """Remove test-like rows *before* hits are built, so pagination pages the kept set."""
    if include_tests:
        return rows
    return [row for row in rows if not is_test_path(str(row.get("file_path") or ""))]


def _cap_matches(hits: list[Hit]) -> list[Hit]:
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


def _chunk_match(row: Mapping[str, object], score: float, lines: list[dict[str, object]] | None) -> dict[str, object]:
    match: dict[str, object] = {
        "chunk_id": row.get("chunk_id"),
        "chunk_kind": row.get("chunk_kind"),
        "line_start": row.get("match_start_line"),
        "line_end": row.get("match_end_line"),
        "score": score,
    }
    if lines:
        match["lines"] = lines
    return match


def _lines_in(row: Mapping[str, object], words: list[tuple[str, list[str]]] | None) -> list[dict[str, object]] | None:
    """The file lines of a matched chunk that hold a query word, when the chunk is file text.

    Only verbatim chunks have file lines to point at: a module or class *context* chunk is a
    generated listing of its members, so a word matched there has no line of its own.
    """
    if not words or row.get("chunk_kind") not in VERBATIM_CHUNK_KINDS:
        return None
    content = row.get("chunk_content")
    start = row.get("match_start_line")
    if not isinstance(content, str) or not isinstance(start, int):
        return None
    return matching_lines(content, start, words)


def _code_hits(
    rows: list[dict[str, object]],
    min_score: float,
    words: list[tuple[str, list[str]]] | None = None,
) -> list[Hit]:
    grouped: dict[str, Hit] = {}
    for row in rows:
        score = float(row.get("score", 0) or 0)
        if score < min_score:
            continue
        node_id = str(row["id"])
        hit = grouped.setdefault(node_id, _new_hit(row, score, "code"))
        hit["score"] = max(float(hit["score"]), score)
        hit["matches"].append(_chunk_match(row, score, _lines_in(row, words)))
    return sorted(grouped.values(), key=lambda hit: float(hit["score"]), reverse=True)


def _summary_hits(rows: list[dict[str, object]], min_score: float) -> list[Hit]:
    hits = []
    for row in rows:
        score = float(row.get("score", 0) or 0)
        if score >= min_score:
            hits.append(_new_hit(row, score, "summary"))
    return sorted(hits, key=lambda hit: float(hit["score"]), reverse=True)


def _text_hits(rows: list[dict[str, object]], min_score: float, source_type: str) -> list[Hit]:
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
    words: list[tuple[str, list[str]]] | None = None,
) -> list[Hit]:
    merged: dict[str, Hit] = {}
    field_hits = [
        *_code_hits(code_rows, min_score, words),
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


def _fuzzy_hits(rows: list[dict[str, object]], word: str, distance: int, known: set[str]) -> list[Hit]:
    """Hits found only by tolerating edits in *word*, for entities no exact hit already names."""
    grouped: dict[str, Hit] = {}
    for row in rows:
        node_id = str(row["id"])
        if node_id in known:
            continue
        hit = grouped.get(node_id)
        if hit is None:
            hit = _new_hit(row, float(row.get("score", 0) or 0), "text")
            hit["fuzzy"] = True
            grouped[node_id] = hit
        lines = None
        content, start = row.get("chunk_content"), row.get("match_start_line")
        if row.get("chunk_kind") in VERBATIM_CHUNK_KINDS and isinstance(content, str) and isinstance(start, int):
            lines = fuzzy_matching_lines(content, start, word, distance)
        hit["matches"].append(_chunk_match(row, float(row.get("score", 0) or 0), lines))
    return list(grouped.values())


def _filter_node_types(rows: list[dict[str, object]], node_types: list[str] | None) -> list[dict[str, object]]:
    if not node_types:
        return rows
    allowed = set(node_types)
    return [row for row in rows if row.get("node_type") in allowed]


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


def _words_seen_in_hits(hits: Sequence[Hit], words: list[tuple[str, list[str]]]) -> set[str]:
    """Query words already proven present: a candidate's name, summary or quoted line holds them.

    A one-word query with any hit matched that word by definition. Everything this proves
    saves a probe of the index; what it cannot prove is probed.
    """
    if len(words) == 1 and hits:
        return {words[0][0]}
    seen: set[str] = set()
    for hit in hits:
        entity = hit.get("entity") or {}
        texts = [str(entity.get("name") or ""), str(entity.get("summary") or "")]
        for match in hit.get("matches") or []:
            texts.extend(str(line.get("text") or "") for line in match.get("lines") or [])
        for text in texts:
            text_tokens = tokens(text)
            for word, phrase in words:
                if word not in seen and all(part in text_tokens for part in phrase):
                    seen.add(word)
        if len(seen) == len(words):
            break
    return seen


# Appended to an empty search_text page's coverage note: the one way this tool misses text that
# IS in an indexed file, said where the "not a coverage gap" conclusion is read.
_TEXT_COVERAGE_CAVEAT = (
    "search_text matches whole words exactly (no stemming, no partial words), so text in another "
    "word form or spelling is still missed — find_text reads the files for a literal or a regex."
)


_SEARCH_COMMON = (
    doc.PATH_FILTERS,
    doc.IS_TEST,
    doc.ARCHIVED,
    doc.COMPLETE_POOL,
    doc.WORKTREE_FIRST_QUERY,
)
_SEARCH_TAIL = (doc.HINTS, doc.PROVENANCE, doc.EMPTY_PAGE)

SEARCH_CODE_DESCRIPTION = doc.join(
    """Search code chunk embeddings and return entity hits.

``query`` is embedded into the same vector space as the indexed code, which is English-dominated:
phrase it in English — a query in another language lands far from that space and returns an
irrelevant top result, not an error. For exact strings, identifiers, or non-English text, use
``search_text`` (BM25 keyword search) or ``find_text`` (every literal occurrence) instead.

Chunks are embedded in the final phase of an index run: while a first-time index is still
running this finds nothing yet — that is "not indexed yet", not "no matches".
``search_entities`` (name-based) works as soon as the run starts.""",
    doc.SCOPE,
    doc.NODE_TYPES,
    *_SEARCH_COMMON,
    doc.WEAK,
    *_SEARCH_TAIL,
)

SEARCH_SEMANTIC_DESCRIPTION = doc.join(
    """Search summary embeddings by meaning (semantic similarity) and return entity hits.

Use this to find code by *what it does* even when the wording differs — the one question grep
cannot answer, because you do not know the name yet. For exact identifier / string-literal /
error-text matches, use ``search_text``; for every occurrence of a literal, ``find_text``.

Summaries and their embeddings are English-only, so phrase ``query`` in English — a query in
another language (e.g. Russian) embeds far from the English summary space and returns an
irrelevant top result, not an error. For non-English text use ``search_text``.

While an index run is in progress, only nodes whose summaries already landed are searchable —
fewer hits then means "not summarized yet", not "no matches".""",
    doc.SCOPE,
    doc.NODE_TYPES,
    *_SEARCH_COMMON,
    doc.WEAK,
    *_SEARCH_TAIL,
)

SEARCH_HYBRID_DESCRIPTION = doc.join(
    """Search code and summary embeddings, deduplicated by entity.

Summary embeddings (and code identifiers they're matched against) are English-only, so phrase
``query`` in English — a non-English query (e.g. Russian) embeds far from that space and returns
an irrelevant top result, not an error. For exact strings, identifiers, or non-English text, use
``search_text`` (BM25 keyword search) instead.

While an index run is in progress, coverage is partial: summary vectors exist only for
already-summarized nodes, and code-chunk vectors land in the run's final phase.

A hit lists at most 3 chunk ``matches`` — the highest-scoring ones. ``matches_total`` gives the
real count and is present only when the list was trimmed.""",
    doc.SCOPE,
    doc.NODE_TYPES,
    *_SEARCH_COMMON,
    doc.WEAK,
    *_SEARCH_TAIL,
)

SEARCH_TEXT_DESCRIPTION = doc.join(
    """Exact keyword (BM25) full-text search over symbol names, signatures, summaries, and raw code — ranked.

Use this to find *literal* terms — a specific identifier, a string literal, the text of an error
or log message, non-English text — matched by keyword and ranked by relevance. For meaning-based
search use ``search_semantic``; when you need EVERY occurrence of a literal (a rename, a
"is this used anywhere under X" proof, the definition line of a constant) use ``find_text``,
which is exhaustive, unranked and reads the files on disk. Scores are BM25 relevance (unbounded,
larger = better), not 0..1 similarities.

Every line of an indexed file is reachable: function bodies, module-level code (constants,
rule tables, exported objects, imports), class and type bodies, and document text are all
chunked. A hit is an entity; its ``matches`` (at most 3, ``matches_total`` when trimmed) name
the chunk spans that matched, and for chunks that are file text each carries ``lines`` — up to
3 ``{"line", "text"}`` pairs quoting the file lines that hold your words, so a citation needs no
second read. Returns compact entity refs — never whole source or embeddings.

HOW WORDS MATCH. Query metacharacters (``:``, ``[]``, ``+``, ``-``, ``*``, …) are separators,
so ``ValueError: invalid literal`` or ``arr[0]`` match as plain keywords. Each word is matched
whole and exactly, case-insensitively — there is NO stemming in any language: ``running`` does
not match ``run``, and Russian ``выбрасывания`` does not match ``выбрасывание``. A word with
punctuation inside (``parser_strict``, ``task.start``) is split at it and searched as a
phrase, so it matches only where those pieces are adjacent. A multi-word query matches ANY of
its words, so the top hit is usually right while the tail can be noise from a common word —
narrow with the most distinctive single word, or add ``include_paths``. When a word matches
nothing in this project at all, ``meta.unmatched_terms`` lists it (a guessed identifier, a
different inflection, a typo) and a sentence in ``meta.hints`` says so; a word of five or more
letters is then retried within one or two edits, and hits found only that way carry
``"fuzzy": true`` and are listed after exact ones.

While an index run is in progress, summary and raw-code fields cover only what has already
landed; name/signature hits are complete as soon as the skeleton is stored.""",
    doc.SCOPE,
    doc.NODE_TYPES,
    *_SEARCH_COMMON,
    *_SEARCH_TAIL,
)


def register_search_tools(
    mcp: FastMCP,
    manager: ProjectManager,
    repo: GraphRepository,
    settings: Settings,
) -> None:
    """Register search tools with the MCP server."""

    code_embedder = EmbeddingClient(settings=settings, model=settings.code_embedding_model)
    summary_embedder = EmbeddingClient(settings=settings, model=settings.summary_embedding_model)

    async def _respond(
        hits: list[Hit],
        *,
        directory: str,
        project: str,
        limit: int,
        offset: int,
        meta: dict[str, object],
        truncated: bool,
        include_tests: bool,
        node_types: list[str] | None,
        include_paths: Sequence[str] | None = None,
        weak_threshold: float | None = None,
        extra_hints: Sequence[str] = (),
        coverage_caveat: str = "",
    ) -> MutableMapping[str, Any]:
        """Order, page, annotate and explain a ranked hit list — the same way for every search.

        Archived hits are moved below live ones over the *whole* candidate list before paging,
        so page 1 holds the best live answers and later pages stay consistent with it.
        """
        ordered, archived = demote_archived(hits, load_archive(Path(directory).resolve()))
        payload = paginated_response(ordered[offset : offset + limit + 1], limit, offset, meta, truncated=truncated)
        checker = await build_checker(repo, project, directory)
        checker.stamp(payload)
        hints = list(extra_hints)
        if weak_threshold is not None:
            _flag_weak(payload, weak_threshold)
            if payload["meta"].get("weak_results"):
                hints.append(weak_page_hint(weak_threshold))
        hints.extend(
            page_shape_hints(
                payload["items"], include_tests=include_tests, node_types=node_types, include_paths=include_paths
            )
        )
        if archived and offset == 0:
            hints.append(archived_hint(archived))
        add_hints(payload, hints)
        return await note_coverage(payload, manager, directory, checker, caveat=coverage_caveat)

    async def _prepare(
        directory: str, node_types: list[str] | None
    ) -> tuple[str, list[str] | None] | dict:
        # Argument validation first: rejecting a typo'd node type must not trigger the
        # manager's inline (donor-seeded) index of a fresh worktree.
        normalized = normalize_node_types(node_types)
        if isinstance(normalized, dict):
            return normalized
        project, not_indexed = await ensure_indexed(manager, directory)
        if not_indexed is not None:
            return not_indexed
        return project, normalized

    def _meta(project: str, query: str, min_score: float, node_types: list[str] | None, filters: PathFilterConfig) -> dict[str, object]:
        return {
            "project": project,
            "query": query,
            "min_score": min_score,
            "node_types": node_types or [],
            "path_filters": filters.meta(),
        }

    @mcp.tool(description=SEARCH_CODE_DESCRIPTION)
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
        """Vector search over raw code chunks (published description: SEARCH_CODE_DESCRIPTION)."""
        prepared = await _prepare(directory, node_types)
        if isinstance(prepared, dict):
            return prepared
        project, types = prepared
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        embedding = _embed_one(code_embedder, query, "Code")
        if isinstance(embedding, dict):
            return embedding
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        found = await repo.vector_hits(
            "code", project, embedding, _overfetch(bounded_limit, bounded_offset),
            _row_filter(filters, types, include_tests),
        )
        return await _respond(
            _code_hits(found.rows, min_score),
            directory=directory, project=project, limit=bounded_limit, offset=bounded_offset,
            meta=_meta(project, query, min_score, types, filters), truncated=not found.complete,
            include_tests=include_tests, node_types=types, include_paths=include_paths,
            weak_threshold=settings.semantic_weak_threshold,
        )

    @mcp.tool(description=SEARCH_SEMANTIC_DESCRIPTION)
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
        """Vector search over summaries (published description: SEARCH_SEMANTIC_DESCRIPTION)."""
        prepared = await _prepare(directory, node_types)
        if isinstance(prepared, dict):
            return prepared
        project, types = prepared
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        embedding = _embed_one(summary_embedder, query, "Summary")
        if isinstance(embedding, dict):
            return embedding
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        found = await repo.vector_hits(
            "summary", project, embedding, _overfetch(bounded_limit, bounded_offset),
            _row_filter(filters, types, include_tests),
        )
        return await _respond(
            _summary_hits(found.rows, min_score),
            directory=directory, project=project, limit=bounded_limit, offset=bounded_offset,
            meta=_meta(project, query, min_score, types, filters), truncated=not found.complete,
            include_tests=include_tests, node_types=types, include_paths=include_paths,
            weak_threshold=settings.semantic_weak_threshold,
        )

    @mcp.tool(description=SEARCH_HYBRID_DESCRIPTION)
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
        """Both vector spaces, merged per entity (published description: SEARCH_HYBRID_DESCRIPTION)."""
        prepared = await _prepare(directory, node_types)
        if isinstance(prepared, dict):
            return prepared
        project, types = prepared
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        code_embedding = _embed_one(code_embedder, query, "Code")
        if isinstance(code_embedding, dict):
            return code_embedding
        summary_embedding = _embed_one(summary_embedder, query, "Summary")
        if isinstance(summary_embedding, dict):
            return summary_embedding

        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        keep = _row_filter(filters, types, include_tests)
        needed = _overfetch(bounded_limit, bounded_offset)
        code_found = await repo.vector_hits("code", project, code_embedding, needed, keep)
        summary_found = await repo.vector_hits("summary", project, summary_embedding, needed, keep)

        merged: dict[str, Hit] = {}
        for hit in [*_code_hits(code_found.rows, min_score), *_summary_hits(summary_found.rows, min_score)]:
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
        return await _respond(
            _cap_matches(hits),
            directory=directory, project=project, limit=bounded_limit, offset=bounded_offset,
            meta=_meta(project, query, min_score, types, filters),
            truncated=not (code_found.complete and summary_found.complete),
            include_tests=include_tests, node_types=types, include_paths=include_paths,
            weak_threshold=settings.semantic_weak_threshold,
        )

    async def _word_is_indexed(project: str, word: str) -> bool | None:
        """Whether *word* occurs anywhere in this project's text indexes (None: could not tell).

        Unfiltered on purpose: the question is "does the index know this word at all", which
        is what separates a misspelt or guessed term from one the path filters excluded.
        """
        try:
            for kind in ("code", "symbol", "summary"):
                probe: ProjectHits = await repo.text_hits(kind, project, word, 1)
                if probe.rows:
                    return True
                if not probe.complete:
                    return None
        except Exception:  # noqa: BLE001 — an explanation must never break the answer it explains
            logger.debug("Word probe failed for %r in project=%s", word, project, exc_info=True)
            return None
        return False

    @mcp.tool(description=SEARCH_TEXT_DESCRIPTION)
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
        """BM25 over names, signatures, summaries and chunks (published description: SEARCH_TEXT_DESCRIPTION)."""
        prepared = await _prepare(directory, node_types)
        if isinstance(prepared, dict):
            return prepared
        project, types = prepared
        bounded_limit = clamp_limit(limit, MAX_SEARCH_LIMIT)
        bounded_offset = clamp_offset(offset)
        filters = PathFilterConfig.from_paths(include_paths, exclude_paths)
        needed = _overfetch(bounded_limit, bounded_offset)
        meta = _meta(project, query, min_score, types, filters)
        text_query = _sanitize_text_query(query)
        words = query_words(text_query)
        if not words:
            # Nothing to search, so nothing to hash — but the blocks still ship, since a
            # caller may only read the item flags as "fresh" if they are always present.
            meta["answered_from"] = describe_origin(Path(directory).resolve())
            meta["index_freshness"] = unchecked_empty_page(
                directory, await repo.get_watch_state(project)
            )
            return paginated_response([], bounded_limit, bounded_offset, meta)
        keep = _row_filter(filters, types, include_tests)
        try:
            summary_found = await repo.text_hits("summary", project, text_query, needed, keep)
            symbol_found = await repo.text_hits("symbol", project, text_query, needed, keep)
            code_found = await repo.text_hits("code", project, text_query, needed, keep)
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

        truncated = not (summary_found.complete and symbol_found.complete and code_found.complete)
        hits = _merge_text_hits(summary_found.rows, symbol_found.rows, code_found.rows, min_score, words)

        # Which words the index does not know at all — the difference between "your guess at the
        # name is wrong" and "it is there, but not where you filtered". A word already quoted
        # in a matched line is proven present and costs nothing; the rest are probed.
        present = _words_seen_in_hits(hits, words)
        unmatched: list[str] = []
        for word, _phrase in words[:_MAX_WORDS_PROBED]:
            if word in present:
                continue
            if await _word_is_indexed(project, word) is False:
                unmatched.append(word)

        extra_hints: list[str] = []
        fuzzy_tried: dict[str, int] = {}
        if unmatched:
            meta["unmatched_terms"] = unmatched
            known = {str((hit["entity"] or {}).get("id", "")) for hit in hits}
            for word in unmatched:
                distance = fuzzy_distance_for(word)
                if not distance:
                    continue
                fuzzy_tried[word] = distance
                try:
                    fuzzy = await repo.text_hits("code", project, word, needed, keep, fuzzy_distance=distance)
                except Exception:  # noqa: BLE001 — the fallback must never break the exact answer
                    logger.debug("Fuzzy retry failed for %r", word, exc_info=True)
                    continue
                truncated = truncated or not fuzzy.complete
                extra = _fuzzy_hits([r for r in fuzzy.rows if float(r.get("score", 0) or 0) >= min_score], word, distance, known)
                known.update(str((hit["entity"] or {}).get("id", "")) for hit in extra)
                hits.extend(extra)
            extra_hints.append(unmatched_terms_hint(unmatched, fuzzy_tried))
        elif not hits and (include_paths or exclude_paths or types or not include_tests):
            extra_hints.append(
                "Every word of this query occurs somewhere in this project's index, but nothing "
                "the filters keep matched it (include_paths / exclude_paths / node_types / "
                "include_tests). The answer may live outside the scope you chose — an operational "
                "procedure is often in reports or runbooks rather than in the config directory."
            )

        return await _respond(
            _cap_matches(hits),
            directory=directory, project=project, limit=bounded_limit, offset=bounded_offset,
            meta=meta, truncated=truncated, include_tests=include_tests, node_types=types, include_paths=include_paths,
            extra_hints=extra_hints, coverage_caveat=_TEXT_COVERAGE_CAVEAT,
        )
