"""Idempotent schema setup for Memgraph — property, vector, and full-text indexes."""

from __future__ import annotations

import logging
import re

from inflorescence.db.connection import MemgraphConnection

logger = logging.getLogger(__name__)

# Current Memgraph exposes a text index's name only inside its "index type" string
# (``label_text (name: summary_text)``), not as a column.
_TEXT_INDEX_NAME_RE = re.compile(r"name:\s*([A-Za-z0-9_]+)")

# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------

PROPERTY_INDEXES: list[tuple[str, str]] = [
    ("Code", "id"),
    ("Code", "file_path"),
    ("Code", "project"),
    ("Code", "parent_id"),
    ("Code", "node_type"),
    ("Directory", "project"),
    ("Module", "project"),
    ("Document", "project"),
    ("Class", "project"),
    ("Function", "project"),
    ("Method", "project"),
    ("Interface", "project"),
    ("Enum", "project"),
    ("Struct", "project"),
    ("Trait", "project"),
]

VECTOR_INDEXES: list[dict[str, str | int]] = [
    {"name": "code_embeddings", "label": "CodeChunk", "property": "embedding", "dimension": 1536, "metric": "cos", "capacity": 10000},
    {"name": "summary_embeddings", "label": "Code", "property": "summary_embedding", "dimension": 1536, "metric": "cos", "capacity": 10000},
]

TEXT_INDEXES: list[dict[str, str]] = [
    {"name": "summary_text", "label": "Code", "property": "summary"},
    {"name": "code_symbols_text", "label": "Code", "property": "name, signature"},
    {"name": "chunk_content_text", "label": "CodeChunk", "property": "content"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _existing_indexes(conn: MemgraphConnection) -> set[str]:
    """Return a set of human-readable keys that describe every existing index.

    For property indexes the key is ``"property::Label::prop"``.
    For named indexes (vector / text) the key is ``"named::index_name"``.
    """
    records = await conn.execute_query("SHOW INDEX INFO;")
    keys: set[str] = set()
    for record in records:
        # Memgraph returns index type, label, property, and (for named) name.
        # Column names may vary across versions; we access by position to stay safe.
        data = record.data()
        index_type: str = str(data.get("index type", data.get("type", "")))
        label: str = str(data.get("label", ""))
        name: str = str(data.get("name", ""))

        # `property` may be a scalar (older versions) or a list of columns
        # (e.g. ``["id"]`` on Memgraph 3.x). Normalise to individual names so a
        # single-column property index registers as ``property::Label::col``.
        prop_raw = data.get("property", "")
        if isinstance(prop_raw, (list, tuple)):
            props = [str(p) for p in prop_raw if str(p)]
        elif str(prop_raw):
            props = [str(prop_raw)]
        else:
            props = []
        prop = ", ".join(props)

        if name:
            keys.add(f"named::{name}")
        if label:
            for p in props:
                keys.add(f"property::{label}::{p}")
        # Also store the raw type so callers could inspect if needed.
        if index_type and label:
            keys.add(f"type::{index_type}::{label}::{prop}")
    return keys


REQUIRED_INDEX_KEYS: frozenset[str] = frozenset({"property::Code::id", "property::Code::project"})

# Vector indexes whose absence means embeddings were (or would be) paid for but can never be
# searched — the exact DB-5 waste. Present only on the Memgraph MAGE image. The cost gate
# (ProjectManager preflight) refuses to index when any of these is missing, so a search-less
# server does not pretend to work (INV-2). Text/BM25 indexes are surfaced but not gated: they
# index already-stored text, so their absence wastes no money. Identified by (label, property)
# because the vector index exposes no name in SHOW INDEX INFO on current Memgraph.
REQUIRED_SEARCH_VECTORS: tuple[tuple[str, str, str], ...] = (
    ("CodeChunk", "embedding", "code_embeddings"),
    ("Code", "summary_embedding", "summary_embeddings"),
)


async def _existing_search_indexes(conn: MemgraphConnection) -> tuple[set[tuple[str, str]], set[str]]:
    """Return ``(vector_targets, text_names)`` present in Memgraph.

    Read structurally, not by a ``name`` column: on current Memgraph a vector index exposes no
    name (type ``label+property_vector``, label ``:Code``, scalar property) and a text index
    carries its name inside the type string (``label_text (name: summary_text)``). Detecting by
    name silently reported every search index as missing (audit finding DB-5 regression trap).
    """
    records = await conn.execute_query("SHOW INDEX INFO;")
    vectors: set[tuple[str, str]] = set()
    texts: set[str] = set()
    for record in records:
        data = record.data()
        index_type = str(data.get("index type", data.get("type", ""))).lower()
        label = str(data.get("label", "")).lstrip(":")
        prop_raw = data.get("property", "")
        if isinstance(prop_raw, (list, tuple)):
            props = [str(p) for p in prop_raw if str(p)]
        elif str(prop_raw):
            props = [str(prop_raw)]
        else:
            props = []
        if "vector" in index_type:
            for p in props:
                vectors.add((label, p))
        elif "text" in index_type:
            match = _TEXT_INDEX_NAME_RE.search(index_type) or _TEXT_INDEX_NAME_RE.search(
                str(data.get("name", ""))
            )
            if match:
                texts.add(match.group(1))
    return vectors, texts


async def schema_present(conn: MemgraphConnection) -> bool:
    """Return True if the core property indexes created by setup_schema exist."""
    existing = await _existing_indexes(conn)
    return REQUIRED_INDEX_KEYS.issubset(existing)


async def missing_search_indexes(conn: MemgraphConnection) -> list[str]:
    """Return the required vector search indexes that do not exist.

    Empty means the search circuit is healthy. Non-empty means the DB (typically a plain
    Memgraph without MAGE) cannot serve vector search, so paying to embed would be waste —
    the indexing preflight uses this to fail closed (audit finding DB-5 / INV-2).
    """
    vectors, _texts = await _existing_search_indexes(conn)
    return sorted(name for label, prop, name in REQUIRED_SEARCH_VECTORS if (label, prop) not in vectors)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def backfill_node_type(conn: MemgraphConnection) -> int:
    """Copy each Code node's type from its label into a ``node_type`` property.

    The type is written both as a label (``:Method``) and, since this change, as the
    ``node_type`` property (``"method"``) so raw Cypher — ``RETURN n.node_type`` or
    ``WHERE n.node_type IN [...]`` — works without deriving it from ``labels(n)``. Graphs
    indexed by an older version carry the label but not the property; this fills it in so
    they become queryable without a full reindex.

    Idempotent and cheap in steady state: a single probe short-circuits the whole pass once
    every node already carries the property. The label is the source of truth — the property
    is a mirror set only here, alongside the label, so the two cannot drift. Returns the
    number of label groups updated (0 when nothing needed backfilling).
    """
    from inflorescence.db.repository import _NODE_TYPE_LABEL_BY_VALUE

    probe = await conn.execute_query("MATCH (n:Code) WHERE n.node_type IS NULL RETURN n.id AS id LIMIT 1")
    if not probe:
        return 0
    updated = 0
    for value, label in _NODE_TYPE_LABEL_BY_VALUE.items():
        # value/label are trusted enum-derived constants — never user input, safe to format in.
        await conn.execute_write(f"MATCH (n:{label}) WHERE n.node_type IS NULL SET n.node_type = '{value}'")
        updated += 1
    logger.info("Backfilled node_type property across %d label groups", updated)
    return updated


async def _ensure_index_lock_constraint(conn: MemgraphConnection) -> None:
    """Make the per-project lease node unique, so the cross-process lock is atomic.

    ``db/lease.py`` acquires a project lock by ``MERGE``-ing a single ``:IndexLock {project}``
    node. Without a uniqueness constraint, two processes racing the first-ever acquire could
    each create their own lock node and both believe they hold it. The constraint makes
    concurrent creates conflict, so exactly one node — and one holder — survives. Idempotent:
    a re-create on an existing constraint is a warning, not a failure.
    """
    try:
        await conn.execute_write("CREATE CONSTRAINT ON (l:IndexLock) ASSERT l.project IS UNIQUE;")
        logger.info("Created uniqueness constraint on :IndexLock(project)")
    except Exception:
        logger.debug("Uniqueness constraint on :IndexLock(project) already present or unsupported")


async def setup_schema(conn: MemgraphConnection) -> list[str]:
    """Create all required indexes in Memgraph, skipping any that already exist.

    Returns the names of required vector search indexes that could NOT be created (empty on a
    healthy MAGE image). A non-empty result is logged at ERROR here and makes the indexing
    preflight fail closed, so a search-less server neither wastes money on unsearchable
    embeddings nor pretends search works (audit finding DB-5 / INV-2).
    """
    existing = await _existing_indexes(conn)
    existing_vectors, existing_texts = await _existing_search_indexes(conn)

    # -- Cross-process lock constraint ---------------------------------------
    await _ensure_index_lock_constraint(conn)

    # -- Property indexes ----------------------------------------------------
    for label, prop in PROPERTY_INDEXES:
        key = f"property::{label}::{prop}"
        if key in existing:
            logger.info("Property index :%s(%s) already exists — skipped", label, prop)
            continue
        query = f"CREATE INDEX ON :{label}({prop});"
        await conn.execute_write(query)
        logger.info("Created property index :%s(%s)", label, prop)

    # -- Vector indexes (USearch HNSW) ---------------------------------------
    for vdef in VECTOR_INDEXES:
        name = vdef["name"]
        label = vdef["label"]
        prop = vdef["property"]
        if (str(label), str(prop)) in existing_vectors:
            logger.info("Vector index %s already exists — skipped", name)
            continue
        dim = vdef["dimension"]
        metric = vdef["metric"]
        capacity = vdef.get("capacity", 10000)
        query = (
            f'CREATE VECTOR INDEX {name} ON :{label}({prop}) '
            f'WITH CONFIG {{"dimension": {dim}, "capacity": {capacity}, "metric": "{metric}"}};'
        )
        try:
            await conn.execute_write(query)
            logger.info("Created vector index %s on :%s(%s) [dim=%s, metric=%s]", name, label, prop, dim, metric)
        except Exception:
            # A required vector index that won't create means embeddings can't be searched —
            # ERROR, not WARNING, because the server is about to pay for and store vectors it
            # can never query (audit finding DB-5). Detected authoritatively below via a state
            # re-read, so a swallowed exception here can't hide it.
            logger.error("Failed to create vector index %s (requires the Memgraph MAGE image)", name)

    # -- Full-text indexes (Tantivy BM25) ------------------------------------
    for tdef in TEXT_INDEXES:
        name = tdef["name"]
        if name in existing_texts:
            logger.info("Text index %s already exists — skipped", name)
            continue
        label = tdef["label"]
        prop = tdef["property"]
        query = f"CREATE TEXT INDEX {name} ON :{label}({prop});"
        try:
            await conn.execute_write(query)
            logger.info("Created text index %s on :%s(%s)", name, label, prop)
        except Exception:
            # BM25/full-text failure degrades text search but wastes no money (it indexes
            # already-stored text), so it is surfaced loudly but not gated like the vectors.
            logger.error("Failed to create text index %s (requires the Memgraph MAGE image)", name)

    # Backfill the node_type property on graphs indexed before it existed (idempotent).
    await backfill_node_type(conn)

    # Authoritative re-read: which required vector search indexes actually exist now? A
    # swallowed creation error above can't hide a missing index from this check.
    missing = await missing_search_indexes(conn)
    if missing:
        logger.error(
            "Search circuit unavailable: vector indexes %s were not created — indexing will "
            "fail closed (set REQUIRE_SEARCH_INDEXES=false for graph-only use on this DB)",
            ", ".join(missing),
        )
    logger.info("Schema setup complete")
    return missing
