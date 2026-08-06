"""Named Cypher query constants for the inflorescence project.

All queries use parameterized placeholders ($project, $node_id, etc.) — no string interpolation.
"""

# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

UPSERT_NODE_TEMPLATE = """
MERGE (n:Code {{id: $id, project: $project}})
WITH n, ($summary = '' AND coalesce(n.summary, '') <> '') AS keep_stored
REMOVE n:Directory
REMOVE n:Module
REMOVE n:Class
REMOVE n:Function
REMOVE n:Method
REMOVE n:Interface
REMOVE n:Enum
REMOVE n:Struct
REMOVE n:Trait
SET n:{node_label}
SET n.node_type = $node_type,
    n.name = $name,
    n.file_path = $file_path,
    n.line_start = $line_start,
    n.line_end = $line_end,
    n.signature = $signature,
    n.docstring = $docstring,
    n.summary = CASE WHEN keep_stored THEN n.summary ELSE $summary END,
    n.summary_input_hash = CASE WHEN keep_stored THEN n.summary_input_hash ELSE $summary_input_hash END,
    n.parent_id = $parent_id,
    n.external_calls = $external_calls,
    n.unresolved_calls = $unresolved_calls,
    n.calls_provenance = $calls_provenance
"""

# Batched upsert: one query writes a whole batch of same-label nodes via UNWIND.
# Node labels can't be parameterized, so callers group rows by label and format
# the label into the template — every row in $rows then shares that label.
#
# An empty incoming summary never overwrites a non-empty stored one (INV-1/INV-3): the
# builder upserts freshly-parsed nodes (summary='') BEFORE summarization, and a
# node removed from the codebase goes through delete_nodes_by_ids — so there is
# no legitimate "clear a summary to empty" write. A non-empty incoming summary
# always wins (re-generated summaries must land).
UPSERT_NODES_BATCH_TEMPLATE = """
UNWIND $rows AS row
MERGE (n:Code {{id: row.id, project: $project}})
WITH n, row, (row.summary = '' AND coalesce(n.summary, '') <> '') AS keep_stored
REMOVE n:Directory
REMOVE n:Module
REMOVE n:Class
REMOVE n:Function
REMOVE n:Method
REMOVE n:Interface
REMOVE n:Enum
REMOVE n:Struct
REMOVE n:Trait
SET n:{node_label}
SET n.node_type = row.node_type,
    n.name = row.name,
    n.file_path = row.file_path,
    n.line_start = row.line_start,
    n.line_end = row.line_end,
    n.signature = row.signature,
    n.docstring = row.docstring,
    n.summary = CASE WHEN keep_stored THEN n.summary ELSE row.summary END,
    n.summary_input_hash = CASE WHEN keep_stored THEN n.summary_input_hash ELSE row.summary_input_hash END,
    n.parent_id = row.parent_id,
    n.external_calls = row.external_calls,
    n.unresolved_calls = row.unresolved_calls,
    n.calls_provenance = row.calls_provenance
"""

# ---------------------------------------------------------------------------
# Checksum operations
# ---------------------------------------------------------------------------

GET_CHECKSUMS = """
MATCH (n:Code:Module {project: $project})
RETURN n.file_path AS file_path, n.checksum AS checksum
"""

# Cheap existence probe: does this project have any indexed node at all? Bounded by LIMIT 1
# and resolved through the project index, so it does not scan the graph. Search/query tools
# use it to tell "this project was never indexed" (a typo'd or adjacent directory synthesises
# a valid-but-empty project id) apart from "indexed, but no matches" (audit finding T5 / INV-7).
PROJECT_IS_INDEXED = """
MATCH (n:Code {project: $project})
RETURN 1 AS ok
LIMIT 1
"""

GET_PROJECT_STATS = """
OPTIONAL MATCH (n {project: $project})
WHERE n:Code OR n:CodeChunk
WITH
  count(CASE WHEN n:Code THEN 1 ELSE null END) AS nodes,
  count(CASE WHEN n:CodeChunk THEN 1 ELSE null END) AS chunks,
  count(CASE WHEN n:Code AND n.summary_embedding IS NOT NULL THEN 1 ELSE null END) AS summaries
OPTIONAL MATCH (s {project: $project})-[r]->(t {project: $project})
WHERE (s:Code OR s:CodeChunk) AND (t:Code OR t:CodeChunk)
RETURN nodes, count(r) AS edges, chunks, summaries
"""

UPDATE_CHECKSUM = """
MATCH (n:Code {id: $node_id, project: $project})
SET n.checksum = $checksum
"""

# Batched checksum update: each row is {node_id, checksum}. RETURN count(*) reports the
# rows that actually matched a node, so telemetry counts what the DB wrote, not what was
# submitted (INV-7 / audit finding 14).
UPDATE_CHECKSUMS_BATCH = """
UNWIND $rows AS row
MATCH (n:Code {id: row.node_id, project: $project})
SET n.checksum = row.checksum
RETURN count(*) AS written
"""

SET_PROJECT_ROOT_PATH = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
SET n.root_path = $root_path
"""

# The include/exclude scan universe of the explicit index, recorded on the project root
# node so a background watcher update reproduces exactly the same scan (audit finding 5c).
# Without this the watcher rescans with only the global config excludes, so a project
# indexed with e.g. exclude=["vendor"] would start parsing, summarizing and embedding
# vendor/** on the first file save — silently widening scope and spending money (INV-6/INV-10).
SET_PROJECT_SCOPE = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
SET n.index_include = $include, n.index_exclude = $exclude
"""

GET_PROJECT_SCOPE = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
RETURN n.index_include AS include, n.index_exclude AS exclude
"""

GET_STORED_SUMMARIES = """
MATCH (n:Code {project: $project})
WHERE n.summary IS NOT NULL AND n.summary <> ''
RETURN n.id AS id, n.summary AS summary,
       n.summary_input_hash AS summary_input_hash,
       n.summary_embedding AS summary_embedding,
       n.summary_embedding_hash AS summary_embedding_hash
"""

GET_STORED_CHUNK_EMBEDDINGS = """
MATCH (c:CodeChunk {project: $project})
RETURN c.id AS chunk_id, c.embedding AS embedding
"""

# Repair queries: detect state a killed index run left incomplete. The unchanged
# fast path in ProjectManager.index_directory uses these to converge to a full
# index without a rebuild.

_NODE_FIELDS = """n.id AS id, n.name AS name, n.node_type AS node_type, labels(n) AS labels,
       n.file_path AS file_path, n.line_start AS line_start, n.line_end AS line_end,
       n.signature AS signature, n.docstring AS docstring, n.summary AS summary,
       n.summary_input_hash AS summary_input_hash"""

# A summary embedding is stale — not just missing — when it was computed against a
# different summary than the one now stored. `summary_embedding_hash` records the
# `summary_input_hash` the embedding was written for; a mismatch means the summary was
# regenerated but its vector never caught up (audit finding 8a). A NULL embedding_hash
# is a pre-marker (legacy) embedding, treated as fresh so an upgrade doesn't re-embed
# every summary — new desyncs (hash present, mismatched) are always caught.
GET_STALE_SUMMARY_EMBEDDINGS = f"""
MATCH (n:Code {{project: $project}})
WHERE n.summary IS NOT NULL AND n.summary <> ''
  AND (n.summary_embedding IS NULL
       OR (n.summary_embedding_hash IS NOT NULL
           AND n.summary_embedding_hash <> n.summary_input_hash))
RETURN {_NODE_FIELDS}
"""

# Missing (NULL/empty summary) OR retry-marked: an empty `summary_input_hash` is the
# marker both summarize_hierarchy and repair_missing_summaries write when generation
# failed, so a docstring surrogate never permanently masquerades as a real summary
# (audit finding 8c). The next healthy run re-summarizes it.
GET_NODES_MISSING_SUMMARIES = f"""
MATCH (n:Code {{project: $project}})
WHERE n.summary IS NULL OR n.summary = '' OR n.summary_input_hash = ''
OPTIONAL MATCH (n)-[:CONTAINS]->(c:Code {{project: $project}})
RETURN {_NODE_FIELDS}, count(c) AS child_count
"""

GET_CHILD_SUMMARY_ROWS = """
MATCH (p:Code {id: $node_id, project: $project})-[:CONTAINS]->(n:Code {project: $project})
RETURN n.id AS id, n.name AS name, n.node_type AS node_type, labels(n) AS labels,
       n.file_path AS file_path, n.line_start AS line_start, n.line_end AS line_end,
       n.signature AS signature, n.summary AS summary
ORDER BY n.file_path, n.line_start, n.id
"""

# Files whose stored chunks don't match current content. `chunks_checksum` is stamped
# on the module node only once every chunk of that file's current content is stored
# (see RAGIndexer.index_code); so NULL means never-fully-chunked (missing chunks, or a
# run killed mid-window left a partial set), and a mismatch means the file changed but
# its chunks are still the old content (audit finding 8b). Either way the next healthy
# run re-chunks the file. A legacy graph indexed before the marker existed has NULL
# here and is re-chunked once — cheap: unchanged content reuses stored embeddings (0 API).
GET_FILES_WITH_STALE_CHUNKS = """
MATCH (m:Code:Module {project: $project})
WHERE m.file_path IS NOT NULL AND m.file_path <> ''
  AND (m.chunks_checksum IS NULL OR m.chunks_checksum <> m.checksum)
RETURN m.file_path AS file_path
"""

# Stamp the coverage marker: chunks for these files are fully stored for the given
# content checksum. Set together with (i.e. equal to) the file's stored checksum.
MARK_CHUNKS_COVERED = """
UNWIND $rows AS row
MATCH (m:Code:Module {project: $project, file_path: row.file_path})
SET m.chunks_checksum = row.checksum
RETURN count(*) AS written
"""

GET_NODES_FOR_FILES = f"""
MATCH (n:Code {{project: $project}})
WHERE n.file_path IN $file_paths
RETURN {_NODE_FIELDS}
"""

# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------

LIST_ENTITIES = """
MATCH (n:Code {project: $project})
WHERE ($node_types IS NULL OR ANY(label IN labels(n) WHERE toLower(label) IN $node_types))
RETURN n.id, n.name, labels(n) AS `n.labels`, n.file_path, n.line_start, n.line_end, n.summary
ORDER BY n.file_path, n.line_start, n.name
SKIP $offset
LIMIT $limit
"""

GET_STRUCTURE_ROOT_BY_ID = """
MATCH (root:Code {project: $project, id: $node_id})
RETURN root.id, root.name, labels(root) AS `root.labels`, root.file_path, root.line_start, root.line_end, root.summary
"""

GET_STRUCTURE_ROOT_BY_FILE = """
MATCH (root:Code:Module {project: $project, file_path: $file_path})
RETURN root.id, root.name, labels(root) AS `root.labels`, root.file_path, root.line_start, root.line_end, root.summary
"""

GET_PROJECT_ROOT = """
MATCH (root:Code {project: $project, id: 'dir:.'})
RETURN root.id, root.name, labels(root) AS `root.labels`, root.file_path, root.line_start, root.line_end, root.summary
"""

GET_ENTITY_STRUCTURE = """
MATCH (root:Code {{project: $project, id: $root_id}})-[:CONTAINS*1..{depth}]->(n:Code)
WHERE ($node_types IS NULL OR ANY(label IN labels(n) WHERE toLower(label) IN $node_types))
RETURN n.id, n.name, labels(n) AS `n.labels`, n.file_path, n.line_start, n.line_end, n.summary
ORDER BY n.file_path, n.line_start, n.name
SKIP $offset
LIMIT $limit
"""

# $relation_types filters the incoming/outgoing relationship sections (null = all types);
# parent and CONTAINS children are the structural skeleton and are always returned.
# Each section is computed in its own CALL{} subquery — sorted, DISTINCT-ed, then sliced
# [$offset .. $offset + $limit] — so pagination is (a) stable: collect follows the sort,
# so separate page requests never skip or repeat an item, and (b) terminating: a deep
# offset yields an empty slice (has_more=False) instead of forever re-serving the first
# page (audit finding 10 — otherwise an agent walking next_offset loops on a hub node).
# The subqueries keep each aggregate off the others' fan-out (a list carried through a
# later MATCH+aggregate is corrupted on this Memgraph), so the sections stay independent.
GET_ENTITY_CONTEXT = """
MATCH (n:Code {project: $project, id: $node_id})
OPTIONAL MATCH (parent:Code {project: $project, id: n.parent_id})
CALL {
  WITH n
  OPTIONAL MATCH (n)-[:CONTAINS]->(child:Code {project: $project})
  WITH child ORDER BY child.id
  RETURN collect(DISTINCT CASE WHEN child IS NULL THEN null ELSE {id: child.id, name: child.name, labels: labels(child), file_path: child.file_path, line_start: child.line_start, line_end: child.line_end, summary: child.summary} END)[$offset..$offset + $limit] AS children
}
CALL {
  WITH n
  OPTIONAL MATCH (source:Code {project: $project})-[incoming]->(n)
    WHERE $relation_types IS NULL OR type(incoming) IN $relation_types
  WITH source, incoming ORDER BY type(incoming), source.id
  RETURN collect(DISTINCT CASE WHEN source IS NULL THEN null ELSE {relation: type(incoming), entity: {id: source.id, name: source.name, labels: labels(source), file_path: source.file_path, line_start: source.line_start, line_end: source.line_end, summary: source.summary}} END)[$offset..$offset + $limit] AS incoming_rows
}
CALL {
  WITH n
  OPTIONAL MATCH (n)-[outgoing]->(target {project: $project})
    WHERE (target:Code OR target:External)
      AND ($relation_types IS NULL OR type(outgoing) IN $relation_types)
  WITH target, outgoing ORDER BY type(outgoing), target.id
  RETURN collect(DISTINCT CASE WHEN target IS NULL THEN null ELSE {relation: type(outgoing), entity: {id: target.id, name: target.name, labels: labels(target), file_path: target.file_path, line_start: target.line_start, line_end: target.line_end, summary: target.summary}} END)[$offset..$offset + $limit] AS outgoing_rows
}
RETURN
  {id: n.id, name: n.name, labels: labels(n), file_path: n.file_path, line_start: n.line_start, line_end: n.line_end, summary: n.summary} AS entity,
  CASE WHEN parent IS NULL THEN null ELSE {id: parent.id, name: parent.name, labels: labels(parent), file_path: parent.file_path, line_start: parent.line_start, line_end: parent.line_end, summary: parent.summary} END AS parent,
  children AS children,
  incoming_rows AS incoming,
  outgoing_rows AS outgoing
"""

GET_RELATED_ENTITIES = """
MATCH (n:Code {project: $project, id: $node_id})-[r]-(related {project: $project})
WHERE
  (related:Code OR related:External)
  AND
  ($relation_types IS NULL OR type(r) IN $relation_types)
  AND (
    $direction = 'both'
    OR ($direction = 'outgoing' AND startNode(r).id = $node_id)
    OR ($direction = 'incoming' AND endNode(r).id = $node_id)
  )
RETURN
  type(r) AS relation,
  CASE WHEN startNode(r).id = $node_id THEN 'outgoing' ELSE 'incoming' END AS direction,
  related.id, related.name, labels(related) AS `related.labels`, related.file_path, related.line_start, related.line_end, related.summary
ORDER BY related.file_path, related.line_start, related.name
SKIP $offset
LIMIT $limit
"""

SEARCH_ENTITIES = """
MATCH (n:Code {project: $project})
WHERE
  toLower(n.name) CONTAINS toLower($query)
  AND ($node_types IS NULL OR ANY(label IN labels(n) WHERE toLower(label) IN $node_types))
RETURN n.id, n.name, labels(n) AS `n.labels`, n.file_path, n.line_start, n.line_end, n.summary
ORDER BY n.file_path, n.line_start, n.name
SKIP $offset
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Vector / FTS search
# ---------------------------------------------------------------------------

VECTOR_SEARCH_CODE = """
CALL vector_search.search('code_embeddings', $top_k, $embedding)
YIELD node, similarity
WITH node, similarity
WHERE node.project = $project
MATCH (code:Code {id: node.node_id, project: $project})
RETURN
  code.id AS id,
  code.name AS name,
  labels(code) AS labels,
  code.file_path AS file_path,
  code.line_start AS line_start,
  code.line_end AS line_end,
  code.summary AS summary,
  node.id AS chunk_id,
  node.chunk_kind AS chunk_kind,
  node.start_line AS match_start_line,
  node.end_line AS match_end_line,
  similarity AS score
"""

VECTOR_SEARCH_SUMMARY = """
CALL vector_search.search('summary_embeddings', $top_k, $embedding)
YIELD node, similarity
WITH node, similarity
WHERE node.project = $project
RETURN node.id AS id, node.name AS name, labels(node) AS labels,
       node.file_path AS file_path, node.line_start AS line_start,
       node.line_end AS line_end, similarity AS score, node.summary AS summary
"""

# BM25 full-text search (Tantivy indexes via MAGE text_search). Scores are
# unbounded positive relevance values, so we ORDER BY score DESC and cap in Cypher.
TEXT_SEARCH_SUMMARY = """
CALL text_search.search_all('summary_text', $query)
YIELD node, score
WITH node, score
WHERE node.project = $project
RETURN node.id AS id, node.name AS name, labels(node) AS labels,
       node.file_path AS file_path, node.line_start AS line_start,
       node.line_end AS line_end, node.summary AS summary, score AS score
ORDER BY score DESC
LIMIT $top_k
"""

TEXT_SEARCH_SYMBOL = """
CALL text_search.search_all('code_symbols_text', $query)
YIELD node, score
WITH node, score
WHERE node.project = $project
RETURN node.id AS id, node.name AS name, labels(node) AS labels,
       node.file_path AS file_path, node.line_start AS line_start,
       node.line_end AS line_end, node.summary AS summary, score AS score
ORDER BY score DESC
LIMIT $top_k
"""

TEXT_SEARCH_CODE = """
CALL text_search.search_all('chunk_content_text', $query)
YIELD node, score
WITH node, score
WHERE node.project = $project
MATCH (code:Code {id: node.node_id, project: $project})
RETURN
  code.id AS id,
  code.name AS name,
  labels(code) AS labels,
  code.file_path AS file_path,
  code.line_start AS line_start,
  code.line_end AS line_end,
  code.summary AS summary,
  node.id AS chunk_id,
  node.chunk_kind AS chunk_kind,
  node.start_line AS match_start_line,
  node.end_line AS match_end_line,
  score AS score
ORDER BY score DESC
LIMIT $top_k
"""

STORE_CODE_CHUNK = """
MATCH (n:Code {id: $node_id, project: $project})
MERGE (c:CodeChunk {id: $chunk_id, project: $project})
SET c.node_id = $node_id,
    c.file_path = $file_path,
    c.content = $content,
    c.start_line = $start_line,
    c.end_line = $end_line,
    c.language = $language,
    c.chunk_kind = $chunk_kind,
    c.embedding = $embedding
MERGE (n)-[:HAS_CHUNK]->(c)
"""

# Batched chunk store: each row carries the full CodeChunk payload.
STORE_CODE_CHUNKS_BATCH = """
UNWIND $rows AS row
MATCH (n:Code {id: row.node_id, project: $project})
MERGE (c:CodeChunk {id: row.chunk_id, project: $project})
SET c.node_id = row.node_id,
    c.file_path = row.file_path,
    c.content = row.content,
    c.start_line = row.start_line,
    c.end_line = row.end_line,
    c.language = row.language,
    c.chunk_kind = row.chunk_kind,
    c.embedding = row.embedding
MERGE (n)-[:HAS_CHUNK]->(c)
RETURN count(*) AS written
"""

STORE_SUMMARY_EMBEDDING = """
MATCH (n:Code {id: $node_id, project: $project})
SET n.summary_embedding = $embedding
"""

# Batched summary-embedding store: each row is {node_id, embedding, embedding_hash}.
# `summary_embedding_hash` records which `summary_input_hash` this vector was computed
# for, so a later run can tell a fresh vector from a stale one (audit finding 8a).
# RETURN count(*) reports rows the DB actually wrote — a store racing a delete, or an
# unresolved id, reports the shortfall instead of claiming success (INV-7 / finding 14).
STORE_SUMMARY_EMBEDDINGS_BATCH = """
UNWIND $rows AS row
MATCH (n:Code {id: row.node_id, project: $project})
SET n.summary_embedding = row.embedding,
    n.summary_embedding_hash = row.embedding_hash
RETURN count(*) AS written
"""

STORE_SUMMARIES_BATCH = """
UNWIND $rows AS row
MATCH (n:Code {id: row.node_id, project: $project})
SET n.summary = row.summary,
    n.summary_input_hash = row.summary_input_hash
RETURN count(*) AS written
"""

DELETE_PROJECT = """
MATCH (n {project: $project})
WHERE n:Code OR n:CodeChunk OR n:External
DETACH DELETE n
"""

# Dependencies the project names but does not contain. Deliberately NOT labelled :Code —
# every query in this module is :Code-scoped, so stdlib and npm packages stay out of the
# entity listing, the dashboard counts, search and summarization, while remaining reachable
# by traversal and by a project-scoped cypher_query.
UPSERT_EXTERNAL_NODES = """
UNWIND $rows AS row
MERGE (n:External {id: row.id, project: $project})
SET n.name = row.name, n.root = row.root, n.kind = row.kind, n.language = row.language
RETURN count(n) AS written
"""

# Reconcile cannot reach an :External node — it has no file to have been re-parsed — so a
# dependency that was dropped from the code would keep its node after its edge is gone.
DELETE_ORPHAN_EXTERNAL_NODES = """
MATCH (n:External {project: $project})
WHERE NOT (n)<-[]-()
DELETE n
RETURN count(n) AS deleted
"""

# Reconcile queries: an update replaces stale state piecewise instead of wiping
# the project. Node/chunk ids are read first, diffed in Python against the fresh
# build, and only the stale ids are deleted — so summary embeddings and chunk
# vectors on surviving nodes are never destroyed.

GET_NODE_IDS = """
MATCH (n:Code {project: $project})
RETURN n.id AS id
"""

# Richer than GET_NODE_IDS: carries file_path and type so a reconcile can delete
# only nodes of files it authoritatively re-processed or confirmed gone from disk,
# instead of every id a missing parser or read failure dropped from the scan.
GET_NODE_INDEX = """
MATCH (n:Code {project: $project})
RETURN n.id AS id, n.file_path AS file_path, n.node_type AS node_type, labels(n) AS labels
"""

GET_CHUNK_IDS = """
MATCH (c:CodeChunk {project: $project})
RETURN c.id AS id
"""

# {chunk_id: file_path} — lets chunk pruning be scoped to the files a run
# authoritatively reconciled, so a read/parse failure can't strip a good file's chunks.
GET_CHUNK_INDEX = """
MATCH (c:CodeChunk {project: $project})
RETURN c.id AS id, c.file_path AS file_path
"""

# All stored Code->Code relationships as (source, TYPE, target). Diffed in Python
# against the fresh build so only genuinely-removed edges are deleted — no wipe.
# Both ends of a stored edge, including edges into :External dependencies — the reconcile
# diffs against this set, so an end it cannot see is an edge it can never prune.
GET_CODE_EDGES = """
MATCH (s:Code {project: $project})-[r]->(t {project: $project})
WHERE t:Code OR t:External
RETURN s.id AS source, type(r) AS type, t.id AS target
"""

DELETE_NODES_BY_IDS = """
UNWIND $ids AS id
MATCH (n:Code {id: id, project: $project})
DETACH DELETE n
"""

DELETE_CHUNKS_BY_IDS = """
UNWIND $ids AS id
MATCH (c:CodeChunk {id: id, project: $project})
DETACH DELETE c
"""

# Deletes only Code->Code relationships (CONTAINS, CALLS, ...). HAS_CHUNK edges
# point at :CodeChunk targets, so they are not matched and chunk links survive.
# Retained for full-project teardown; the reconcile path no longer wipes edges
# (it MERGEs fresh edges, then deletes only stale ones via DELETE_EDGES_BY_KEYS).
DELETE_CODE_EDGES = """
MATCH (:Code {project: $project})-[r]->(:Code {project: $project})
DELETE r
"""

# Targeted stale-edge removal: each row is {source, type, target}. Matches the
# exact endpoint pair by id (indexed) and filters the relationship by type, so a
# reconcile deletes only edges the fresh build dropped — never the whole set.
DELETE_EDGES_BY_KEYS = """
UNWIND $rows AS row
MATCH (s:Code {id: row.source, project: $project})-[r]->(t {id: row.target, project: $project})
WHERE type(r) = row.type AND (t:Code OR t:External)
DELETE r
"""

# Repair queries for the structural (CONTAINS) backbone. Every non-root node stores its
# parent's id in `parent_id` and should have an incoming CONTAINS edge from that parent;
# a run killed after deleting nodes/edges but before re-MERGEing them can leave the tree
# edgeless, which the summary/chunk repairs don't touch (audit finding 9). CONTAINS is
# fully reconstructable from `parent_id` (unlike CALLS, which needs re-resolution), so
# repair rebuilds exactly the missing links. The two OPTIONAL MATCHes are separate on
# purpose: a single `(p)-[c:CONTAINS]->(n)` would bind p and c together, so a missing
# edge would also null out p and the parent-exists check couldn't tell the two apart.
GET_NODES_MISSING_CONTAINS = """
MATCH (n:Code {project: $project})
WHERE n.parent_id IS NOT NULL
OPTIONAL MATCH (p:Code {project: $project, id: n.parent_id})
OPTIONAL MATCH (p)-[c:CONTAINS]->(n)
WITH n, p, c
WHERE p IS NOT NULL AND c IS NULL
RETURN n.id AS id, n.parent_id AS parent_id
"""

# Rebuild the missing CONTAINS edges (idempotent MERGE); RETURN count(*) reports how many
# were actually re-linked (both endpoints matched).
REPAIR_CONTAINS_EDGES = """
UNWIND $rows AS row
MATCH (p:Code {id: row.parent_id, project: $project})
MATCH (c:Code {id: row.id, project: $project})
MERGE (p)-[:CONTAINS]->(c)
RETURN count(*) AS written
"""

