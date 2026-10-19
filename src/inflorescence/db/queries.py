"""Named Cypher query constants for the inflorescence project.

All queries use parameterized placeholders ($project, $node_id, etc.) — no string interpolation
of values. Node *labels* cannot be parameters in Cypher; the few templates that name them take
them from ``NODE_TYPE_LABELS``, which is derived from the ``NodeType`` enum.
"""

from inflorescence.code_indexer.models import NODE_TYPE_LABELS

# A node's type can change between runs (a const that became a function); its old label must
# go before the new one is set.
_REMOVE_TYPE_LABELS = "\n".join(f"REMOVE n:{label}" for label in NODE_TYPE_LABELS.values())

# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

UPSERT_NODE_TEMPLATE = """
MERGE (n:Code {{id: $id, project: $project}})
WITH n, ($summary = '' AND coalesce(n.summary, '') <> '') AS keep_stored
""" + _REMOVE_TYPE_LABELS + """
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
    n.summary_prompt_version = CASE WHEN keep_stored THEN n.summary_prompt_version ELSE $summary_prompt_version END,
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
""" + _REMOVE_TYPE_LABELS + """
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
    n.summary_prompt_version = CASE WHEN keep_stored THEN n.summary_prompt_version ELSE row.summary_prompt_version END,
    n.parent_id = row.parent_id,
    n.external_calls = row.external_calls,
    n.unresolved_calls = row.unresolved_calls,
    n.calls_provenance = row.calls_provenance
"""

# ---------------------------------------------------------------------------
# Checksum operations
# ---------------------------------------------------------------------------

# One node per file carries that file's checksum: its module node, or — for a prose file,
# which has no module — its document node. Matching only :Module would leave every indexed
# document without a readable checksum, so every scan would see it as changed and re-parse,
# re-summarize and re-embed it on every update (INV-5: unchanged content costs nothing).
GET_CHECKSUMS = """
MATCH (n:Code {project: $project})
WHERE n:Module OR n:Document
RETURN n.file_path AS file_path, n.checksum AS checksum
"""

# Cheap existence probe: does this project have any indexed node at all? Bounded by LIMIT 1
# and resolved through the project index, so it does not scan the graph. Search/query tools
# use it to tell "this project was never indexed" (a typo'd or adjacent directory synthesises
# a valid-but-empty project id) apart from "indexed, but no matches" (audit finding T5 / INV-7).
#
# A project whose :CloneMarker is still present is reported as NOT indexed even though it
# already holds nodes: those nodes are a server-side clone in flight (see CLONE_* below), so
# they are a partial copy of another project, with none of this project's own identity
# stamped yet. Answering "indexed" there would let a read tool serve a half-copied graph —
# the empty/partial page INV-7 exists to prevent. The marker is written before the first
# cloned node and removed only after the clone is verified complete, so this probe flips to
# True exactly when the graph is whole. The OPTIONAL MATCH runs first (a handful of marker
# nodes exist at most), so the common case costs one extra label scan of a nearly empty label.
PROJECT_IS_INDEXED = """
OPTIONAL MATCH (m:CloneMarker {project: $project})
WITH m LIMIT 1
MATCH (n:Code {project: $project})
WHERE m IS NULL
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

# "This project is a linked git worktree of that one." Written on the same project root node
# as root_path/scope, and only for a directory detection proved to be a worktree — an ordinary
# project never carries it. Two facts are stored, not one: the donor's *project id* (whose
# cache seeded this index, and whose deletion should be considered together with this one) and
# the main checkout's *path*, which is what a janitor re-reads git's worktree registry from.
# The path alone would not survive the main checkout being re-indexed under a different id, and
# the id alone cannot be turned back into a filesystem location.
SET_PROJECT_WORKTREE_META = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
SET n.worktree_of = $worktree_of, n.worktree_main_root = $main_root_path
"""

GET_PROJECT_WORKTREE_META = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
RETURN n.worktree_of AS worktree_of, n.worktree_main_root AS main_root_path
"""

# Every project that has ever been indexed, with the two facts a janitor reasons about:
# where its tree lives and whether it was indexed as a worktree. Scoped to the project root
# node (one per project) so this is a handful of rows, not a scan of the graph. A project
# without root_path predates path recording — it is returned by no filter here, and so can
# never be reasoned about, let alone deleted.
LIST_PROJECTS = """
MATCH (n:Code:Directory {id: 'dir:.'})
WHERE n.root_path IS NOT NULL AND n.root_path <> ''
RETURN n.project AS project, n.root_path AS root_path,
       n.worktree_of AS worktree_of, n.worktree_main_root AS main_root_path,
       n.watched AS watched
"""

# The watch registry lives on the same project root node as root_path and scope, because a
# watched project is by definition an indexed one: the daemon updates an existing graph, it
# never creates one. Storing it here (rather than in a file beside the daemon) keeps one
# source of truth that `watch add`, the daemon and any MCP server all read from the database
# they already share, and makes `watch status` answerable without the daemon running.
# Registry flag and daemon liveness in one round trip, because they are only meaningful
# together: `watched` without a live claim is the state after a reboot (registered, but
# nothing is running it) and reads very differently from never having been registered.
# The lease node is read, never written, here — see `db.lease` for the claim itself.
GET_WATCH_STATE = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
OPTIONAL MATCH (l:IndexLock {project: $claim_key})
OPTIONAL MATCH (d:IndexLock {project: $daemon_key})
RETURN coalesce(n.watched, false) AS watched,
       l.holder AS holder, l.expires_at AS expires_at,
       d.holder AS daemon_holder, d.expires_at AS daemon_expires_at
"""

# The commit this project's index last matched the disk at — stamped by every run that
# leaves the index equal to the working tree (a full index, an update, a no-op re-check, a
# worktree clone after its reconcile). A later answer compares it with the checkout's HEAD:
# when they differ, a merge or a checkout moved the tree without the index following, and
# the answer says so instead of describing the old commit in the present tense.
SET_INDEXED_HEAD = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
SET n.indexed_head = $head
"""

GET_INDEXED_HEAD = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
RETURN n.indexed_head AS head
"""

SET_PROJECT_WATCHED = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
SET n.watched = $watched
RETURN count(n) AS matched
"""

# The ONLY deletion of a whole project the janitor is allowed to issue. The marker is
# re-asserted inside the same transaction as the delete (the WHERE on the root node), so a
# marker that disappears between a caller's check and this write cannot be raced into
# destroying an ordinary project's index: with no marked root node the MATCH yields nothing
# and nothing is deleted. `deleted` reports how many nodes actually went, which is how the
# caller tells "refused" from "done".
DELETE_WORKTREE_PROJECT = """
MATCH (root:Code:Directory {id: 'dir:.', project: $project})
WHERE root.worktree_of IS NOT NULL AND root.worktree_of <> ''
  AND root.worktree_main_root IS NOT NULL AND root.worktree_main_root <> ''
WITH root
MATCH (n {project: $project})
WHERE n:Code OR n:CodeChunk OR n:External OR n:CloneMarker
DETACH DELETE n
RETURN count(n) AS deleted
"""

GET_STORED_SUMMARIES = """
MATCH (n:Code {project: $project})
WHERE n.summary IS NOT NULL AND n.summary <> ''
RETURN n.id AS id, n.summary AS summary,
       n.summary_input_hash AS summary_input_hash,
       n.summary_prompt_version AS summary_prompt_version,
       n.summary_embedding AS summary_embedding,
       n.summary_embedding_hash AS summary_embedding_hash
"""

# The same rows as GET_STORED_SUMMARIES **without the vectors**. A summary embedding is
# 1536 floats; a project of a few thousand nodes moves tens of megabytes over Bolt just to
# answer "which summaries can I reuse by hash?" — which needs only the text and the two
# hashes. Measured on 5.5k nodes: 5.3s with the vectors, ~0.1s without.
#
# `has_embedding` replaces the vector for the one remaining decision it was used for: an
# incremental run must know whether a *fresh* vector already exists for this node (skip it
# entirely) or not (embed it). The vector itself is only needed when it has to be COPIED
# somewhere else — the client-side worktree seeding, which reads another project's summaries
# and writes them into a new one — and that path still uses GET_STORED_SUMMARIES.
GET_STORED_SUMMARIES_LIGHT = """
MATCH (n:Code {project: $project})
WHERE n.summary IS NOT NULL AND n.summary <> ''
RETURN n.id AS id, n.summary AS summary,
       n.summary_input_hash AS summary_input_hash,
       n.summary_prompt_version AS summary_prompt_version,
       n.summary_embedding_hash AS summary_embedding_hash,
       (n.summary_embedding IS NOT NULL) AS has_embedding
"""

GET_STORED_CHUNK_EMBEDDINGS = """
MATCH (c:CodeChunk {project: $project})
RETURN c.id AS chunk_id, c.embedding AS embedding
"""

# "Which of these chunks does the database already hold exactly as this run would write
# them?" — the question that lets an incremental index stop re-writing thousands of
# byte-identical chunk rows (INV-5). A chunk id hashes the owning node, the file path, the
# line span, the chunk kind and the content, so an id match already proves all of those
# equal; this returns the only two facts a matching id does *not* pin down, so the writer can
# prove the row is a no-op before skipping it:
#
#   * `language` is not part of the id hash, so a chunker whose language mapping changed
#     would otherwise leave the old value behind forever;
#   * the owning node's id comes from the HAS_CHUNK edge, so a chunk whose edge is missing
#     (its Code node was deleted and recreated) is re-written rather than left dangling —
#     STORE_CODE_CHUNKS_BATCH creates that edge, and skipping the write would skip the edge.
#
# `c.embedding IS NOT NULL` is the third condition: a chunk row with no vector is unusable
# for search, so it is not "already stored" in any sense that matters.
#
# Scoped by id, never by project: the whole point is that an update pays in proportion to
# what changed, so the caller batches its own chunk ids in instead of scanning the project.
GET_STORED_CHUNK_FACTS = """
MATCH (c:CodeChunk {project: $project})
WHERE c.id IN $ids AND c.embedding IS NOT NULL
OPTIONAL MATCH (n:Code)-[:HAS_CHUNK]->(c)
RETURN c.id AS id, c.language AS language, n.id AS node_id
"""

# The stored vectors of named chunks. Used for the one case GET_STORED_CHUNK_FACTS above
# reports as "present but not what this run would write": the row has to be re-written, and
# its embedding is already in the database, so re-embedding it would be paying twice for the
# same vector (INV-5). Addressed by id and expected to match a handful of chunks — the
# whole-project and per-file variants exist for the paths that really do need bulk vectors.
GET_STORED_CHUNK_EMBEDDINGS_FOR_IDS = """
MATCH (c:CodeChunk {project: $project})
WHERE c.id IN $ids
RETURN c.id AS chunk_id, c.embedding AS embedding
"""

# The node properties an *unchanged file's* row can still legitimately differ in, and
# nothing else. Everything else a node row carries — name, lines, signature, docstring,
# labels, parent — is a pure function of the file's own content, so an unchanged checksum
# already proves it unchanged. These three are not: they are produced by the whole-project
# call resolution, so editing file B can change what file A's untouched function is
# recorded as calling (an unresolved call becoming an external one, a heuristic file being
# upgraded to semantic provenance).
#
# An incremental update re-parses and re-resolves the whole project (it must — CALLS edges
# are cross-file), then re-upserts every node in it, which is thousands of idempotent MERGEs
# for data that did not move. Reading these three fields for the unchanged files is what lets
# the writer prove a row is already correct and skip it. A node id absent from the result
# has no stored row at all and must always be written.
GET_STORED_CALL_FACTS = """
MATCH (n:Code {project: $project})
WHERE n.file_path IN $file_paths
RETURN n.id AS id, n.file_path AS file_path,
       n.external_calls AS external_calls,
       n.unresolved_calls AS unresolved_calls,
       n.calls_provenance AS calls_provenance
"""

# Repair queries: detect state a killed index run left incomplete. The unchanged
# fast path in ProjectManager.index_directory uses these to converge to a full
# index without a rebuild.

_NODE_FIELDS = """n.id AS id, n.name AS name, n.node_type AS node_type, labels(n) AS labels,
       n.file_path AS file_path, n.line_start AS line_start, n.line_end AS line_end,
       n.signature AS signature, n.docstring AS docstring, n.summary AS summary,
       n.summary_input_hash AS summary_input_hash,
       n.summary_prompt_version AS summary_prompt_version"""

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
WHERE (n.summary IS NULL OR n.summary = '' OR n.summary_input_hash = '')
  AND NOT n.node_type IN $declaration_types
OPTIONAL MATCH (n)-[:CONTAINS]->(c:Code {{project: $project}})
WHERE NOT c.node_type IN $declaration_types
RETURN {_NODE_FIELDS}, count(c) AS child_count
"""

GET_CHILD_SUMMARY_ROWS = """
MATCH (p:Code {id: $node_id, project: $project})-[:CONTAINS]->(n:Code {project: $project})
WHERE NOT n.node_type IN $declaration_types
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
#
# The marker is the checksum *plus* the chunk-format suffix (rag.chunker.CHUNK_MARKER_SUFFIX),
# so a chunker change that alters the chunks of unchanged content marks every file stale once
# and the next run reaches it — the only way a new chunk kind ever reaches a file whose bytes
# did not move. The size is returned so a cost gate can price the re-chunk before it runs.
GET_FILES_WITH_STALE_CHUNKS = """
MATCH (m:Code {project: $project})
WHERE (m:Module OR m:Document)
  AND m.file_path IS NOT NULL AND m.file_path <> ''
  AND (m.chunks_checksum IS NULL OR m.chunks_checksum <> m.checksum + $marker_suffix)
RETURN m.file_path AS file_path
"""

# Stamp the coverage marker: chunks for these files are fully stored for the given
# content checksum. Set together with (i.e. equal to) the file's stored checksum.
#
# The file-node test is a `labels(m)` membership check, NOT the `m:Module OR m:Document`
# label disjunction it stands for — that form is MISCOMPILED by Memgraph 3.12.0 here. With
# the label-property indexes this schema creates (Module.project, Document.project), a
# batched (UNWIND) write whose WHERE is a disjunction of *pure label tests* is planned as an
# indexed union scan and the SET reaches exactly ONE node per query, however many rows
# matched — and `RETURN count(*)` reports that same 1, so the caller cannot even see it was
# cheated. Probed live on 3.12.0 with 5 matching rows: `m:Module OR m:Document` → 1 written;
# `m:Module` alone → 5; `m:Module OR m.node_type = 'document'` → 5; the form below → 5. The
# bug is specific to this shape: the same disjunction in a read, in a write without UNWIND,
# or AND-ed with a non-label predicate executes correctly (probed live for GET_CHECKSUMS,
# GET_FILES_WITH_STALE_CHUNKS, DELETE_PROJECT and DELETE_EDGES_BY_KEYS).
#
# Under-writing here is not cosmetic: an unstamped file stays "never fully chunked" for
# GET_FILES_WITH_STALE_CHUNKS forever, so every re-index re-chunks and re-embeds unchanged
# content — repeated paid work, which is exactly what INV-5 forbids.
#
# `labels(m)` is used rather than the equally-valid `m.node_type IN ['module','document']`
# so that this stamper and the GET_FILES_WITH_STALE_CHUNKS detector above keep testing the
# *same* fact about a node. Were one to key off labels and the other off node_type, any node
# where the two disagreed would be detected as stale forever yet never be stampable — the
# exact endless-re-chunk loop this marker exists to prevent.
MARK_CHUNKS_COVERED = """
UNWIND $rows AS row
MATCH (m:Code {project: $project, file_path: row.file_path})
WHERE ANY(label IN labels(m) WHERE label IN ['Module', 'Document'])
SET m.chunks_checksum = row.checksum + $marker_suffix
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
WHERE ($node_types IS NULL OR n.node_type IN $node_types)
RETURN n.id, n.name, labels(n) AS `n.labels`, n.file_path, n.line_start, n.line_end, n.summary, n.signature
ORDER BY n.file_path, n.line_start, n.name
SKIP $offset
LIMIT $limit
"""

GET_STRUCTURE_ROOT_BY_ID = """
MATCH (root:Code {project: $project, id: $node_id})
RETURN root.id, root.name, labels(root) AS `root.labels`, root.file_path, root.line_start, root.line_end, root.summary, root.signature
"""

# The node that stands for a whole path: a file's module (or document node for a prose
# file), or a directory's node — so an agent walking the tree down by path never has to know
# the `dir:` id convention. A file and a directory never share a path.
GET_STRUCTURE_ROOT_BY_FILE = """
MATCH (root:Code {project: $project, file_path: $file_path})
WHERE root:Module OR root:Document OR root:Directory
RETURN root.id, root.name, labels(root) AS `root.labels`, root.file_path, root.line_start, root.line_end, root.summary, root.signature
"""

GET_PROJECT_ROOT = """
MATCH (root:Code {project: $project, id: 'dir:.'})
RETURN root.id, root.name, labels(root) AS `root.labels`, root.file_path, root.line_start, root.line_end, root.summary, root.signature
"""

GET_ENTITY_STRUCTURE = """
MATCH (root:Code {{project: $project, id: $root_id}})-[:CONTAINS*1..{depth}]->(n:Code)
WHERE ($node_types IS NULL OR n.node_type IN $node_types)
RETURN n.id, n.name, labels(n) AS `n.labels`, n.file_path, n.line_start, n.line_end, n.summary, n.signature
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
  RETURN collect(DISTINCT CASE WHEN child IS NULL THEN null ELSE {id: child.id, name: child.name, labels: labels(child), file_path: child.file_path, line_start: child.line_start, line_end: child.line_end, summary: child.summary, signature: child.signature} END)[$offset..$offset + $limit] AS children
}
CALL {
  WITH n
  OPTIONAL MATCH (source:Code {project: $project})-[incoming]->(n)
    WHERE $relation_types IS NULL OR type(incoming) IN $relation_types
  WITH source, incoming ORDER BY type(incoming), source.id
  RETURN collect(DISTINCT CASE WHEN source IS NULL THEN null ELSE {relation: type(incoming), edge: properties(incoming), entity: {id: source.id, name: source.name, labels: labels(source), file_path: source.file_path, line_start: source.line_start, line_end: source.line_end, summary: source.summary, signature: source.signature}} END)[$offset..$offset + $limit] AS incoming_rows
}
CALL {
  WITH n
  OPTIONAL MATCH (n)-[outgoing]->(target {project: $project})
    WHERE (target:Code OR target:External)
      AND ($relation_types IS NULL OR type(outgoing) IN $relation_types)
  WITH target, outgoing ORDER BY type(outgoing), target.id
  RETURN collect(DISTINCT CASE WHEN target IS NULL THEN null ELSE {relation: type(outgoing), edge: properties(outgoing), entity: {id: target.id, name: target.name, labels: labels(target), file_path: target.file_path, line_start: target.line_start, line_end: target.line_end, summary: target.summary, signature: target.signature}} END)[$offset..$offset + $limit] AS outgoing_rows
}
RETURN
  {id: n.id, name: n.name, labels: labels(n), file_path: n.file_path, line_start: n.line_start, line_end: n.line_end, summary: n.summary, signature: n.signature} AS entity,
  CASE WHEN parent IS NULL THEN null ELSE {id: parent.id, name: parent.name, labels: labels(parent), file_path: parent.file_path, line_start: parent.line_start, line_end: parent.line_end, summary: parent.summary, signature: parent.signature} END AS parent,
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
  properties(r) AS edge,
  related.id, related.name, labels(related) AS `related.labels`, related.file_path, related.line_start, related.line_end, related.summary, related.signature
ORDER BY related.file_path, related.line_start, related.name
SKIP $offset
LIMIT $limit
"""

# Every entity with a line span in the given files — how find_text names the entity that owns a
# matching line (the innermost span holding it) without a round trip per hit.
GET_FILE_ENTITIES = """
MATCH (n:Code {project: $project})
WHERE n.file_path IN $file_paths AND n.line_start IS NOT NULL AND NOT n:Directory
RETURN n.id, n.name, labels(n) AS `n.labels`, n.file_path, n.line_start, n.line_end
"""

# Names spelled exactly as asked come first (exact case, then ignoring case), every other
# substring match after them in path order. The tool's name resolution relies on this: its
# candidate scan is bounded, and an exact `TASKS` must not sort behind fifty `readTasks`.
SEARCH_ENTITIES = """
MATCH (n:Code {project: $project})
WHERE
  toLower(n.name) CONTAINS toLower($query)
  AND ($node_types IS NULL OR n.node_type IN $node_types)
RETURN n.id, n.name, labels(n) AS `n.labels`, n.file_path, n.line_start, n.line_end, n.summary, n.signature
ORDER BY
  CASE WHEN n.name = $query THEN 0 WHEN toLower(n.name) = toLower($query) THEN 1 ELSE 2 END,
  n.file_path, n.line_start, n.name
SKIP $offset
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Vector / FTS search
# ---------------------------------------------------------------------------

# Memgraph's search procedures read GLOBAL indexes — one per property across every project in
# the database — and have no project-filtered variant. Each statement below therefore asks for
# a global window of `$fetch` candidates and keeps this project's rows; the repository widens
# that window until the project's own top rows are provably inside it (see
# GraphRepository._widening_search). A fixed window used to decide the answer silently: the
# text procedures stop at 1000 rows by default, so a common word matched 1000 chunks of
# *other* projects and this one got nothing, and a vector window of 256 held 32 of the
# project's rows once its worktree clones had filled the rest (measured, 25.09.2026).

VECTOR_SEARCH_CODE = """
CALL vector_search.search('code_embeddings', $fetch, $embedding)
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
  code.signature AS signature,
  node.id AS chunk_id,
  node.chunk_kind AS chunk_kind,
  node.start_line AS match_start_line,
  node.end_line AS match_end_line,
  similarity AS score
ORDER BY score DESC
"""

# `{index}` is summary_embeddings, or one of the label-scoped summary indexes
# (schema.LEVEL_SUMMARY_INDEXES) when every requested node type has one.
VECTOR_SEARCH_SUMMARY_TEMPLATE = """
CALL vector_search.search('{index}', $fetch, $embedding)
YIELD node, similarity
WITH node, similarity
WHERE node.project = $project
RETURN node.id AS id, node.name AS name, labels(node) AS labels,
       node.file_path AS file_path, node.line_start AS line_start,
       node.line_end AS line_end, similarity AS score, node.summary AS summary, node.signature AS signature
ORDER BY score DESC
"""

# How many candidates the global window actually held. Asked only when the project's rows
# did not fill the page: a window that came back short of `$fetch` has nothing further to
# give, so widening it cannot find more — which is what makes an unfilled page honest.
VECTOR_SEARCH_WINDOW_TEMPLATE = """
CALL vector_search.search('{index}', $fetch, $embedding)
YIELD node
RETURN count(node) AS scanned
"""

# BM25 full-text search (Tantivy indexes via MAGE text_search). Scores are unbounded positive
# relevance values. `{limit: $fetch}` is not an optimisation: without it the procedure stops
# at 1000 rows across ALL projects, before the project filter below ever runs.
TEXT_SEARCH_SUMMARY = """
CALL text_search.search_all('summary_text', $query, {limit: $fetch})
YIELD node, score
WITH node, score
WHERE node.project = $project
RETURN node.id AS id, node.name AS name, labels(node) AS labels,
       node.file_path AS file_path, node.line_start AS line_start,
       node.line_end AS line_end, node.summary AS summary, node.signature AS signature, score AS score
ORDER BY score DESC
"""

TEXT_SEARCH_SYMBOL = """
CALL text_search.search_all('code_symbols_text', $query, {limit: $fetch})
YIELD node, score
WITH node, score
WHERE node.project = $project
RETURN node.id AS id, node.name AS name, labels(node) AS labels,
       node.file_path AS file_path, node.line_start AS line_start,
       node.line_end AS line_end, node.summary AS summary, node.signature AS signature, score AS score
ORDER BY score DESC
"""

# The chunk's own text rides along so the tool can point at the matching *lines* inside a
# window instead of only its span — an 80-line window around a one-line rule is otherwise a
# second round trip to grep.
TEXT_SEARCH_CODE = """
CALL text_search.search_all('chunk_content_text', $query, {limit: $fetch})
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
  code.signature AS signature,
  node.id AS chunk_id,
  node.chunk_kind AS chunk_kind,
  node.start_line AS match_start_line,
  node.end_line AS match_end_line,
  node.content AS chunk_content,
  score AS score
ORDER BY score DESC
"""

# Typo-tolerant variant for a word that matched nothing as typed: Tantivy's Levenshtein
# automaton over the index's own terms, so "выбрасывания" reaches "выбрасывание" (one edit)
# without this code knowing which language either word is in. Same window discipline as
# above. Scores come from the fuzzy query and are not comparable to exact BM25 scores.
TEXT_SEARCH_CODE_FUZZY = """
CALL text_search.search_all('chunk_content_text', $query, {limit: $fetch, fuzzy_distance: $distance})
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
  code.signature AS signature,
  node.id AS chunk_id,
  node.chunk_kind AS chunk_kind,
  node.start_line AS match_start_line,
  node.end_line AS match_end_line,
  node.content AS chunk_content,
  score AS score
ORDER BY score DESC
"""

TEXT_SEARCH_WINDOW_TEMPLATE = """
CALL text_search.search_all('{index}', $query, {{limit: $fetch{extra}}})
YIELD node
RETURN count(node) AS scanned
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
    n.summary_input_hash = row.summary_input_hash,
    n.summary_prompt_version = row.summary_prompt_version
RETURN count(*) AS written
"""

DELETE_PROJECT = """
MATCH (n {project: $project})
WHERE n:Code OR n:CodeChunk OR n:External OR n:CloneMarker
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

# ---------------------------------------------------------------------------
# Server-side project clone (the worktree fast path)
# ---------------------------------------------------------------------------
#
# Copying one project's graph into another used to be done through the client: parse the
# whole tree again, upsert every node and edge, then re-write every chunk with its 1536-float
# vector over Bolt. Measured on a 5.5k-node / 8.1k-chunk repository that is minutes, and
# nearly all of it is the vectors making a pointless round trip out of the database and back.
# These queries do the copy inside Memgraph: no property value ever crosses the wire.
#
# Two shapes are forced by the engine and are the reason this is a family of templates
# rather than one statement:
#   * labels cannot be parameterized, so nodes are cloned one label at a time and edges one
#     (relationship type, target label) pair at a time — exactly as upsert_nodes/upsert_edges
#     already do for writes;
#   * `MATCH (x {project: ...})` without a label cannot use the label-property indexes this
#     schema creates, so every MATCH here names a label. That is a performance requirement,
#     not a style one: an unlabelled endpoint match inside the edge clone degrades to a full
#     node scan per row.
#
# Every clone statement returns `count(...)` for logging, but the count is NOT what decides
# success: Memgraph 3.12 has been observed writing fewer rows than its own count reports
# (see MARK_CHUNKS_COVERED above). The repository verifies the clone with the independent
# COUNT queries below and refuses the result on any mismatch.

# The in-flight flag. Written before the first cloned node and removed only after the copy
# has been counted and verified, so its presence means "these nodes are a partial copy of
# another project". PROJECT_IS_INDEXED consults it (a half-clone must never answer a query),
# DELETE_PROJECT removes it with the nodes, and ProjectManager.index_directory treats it as
# "delete this project and clone it again".
SET_CLONE_MARKER = """
MERGE (m:CloneMarker {project: $project})
SET m.source = $source, m.started_at = timestamp()
"""

GET_CLONE_MARKER = """
MATCH (m:CloneMarker {project: $project})
RETURN m.source AS source
LIMIT 1
"""

CLEAR_CLONE_MARKER = """
MATCH (m:CloneMarker {project: $project})
DELETE m
"""

# The id ranges one label group is split into so its copy can run on several connections at
# once. Returned as *interior* boundaries: with `$shards` = 4 this yields 3 ids, which the
# repository turns into 4 contiguous `(after, upper]` windows.
#
# Computed server-side and returning only `$shards - 1` ids, so splitting a group costs one
# sorted label scan and a handful of strings on the wire — never the id list itself. The
# `collect` does hold the group's ids in server memory for the length of the query; that is
# the same order of magnitude the copy it plans is about to move, and it happens once per
# group instead of once per batch (the keyset loop re-sorts the group for *every* window).
#
# The CASE guard is not decorative: `ids[k]` on a list shorter than `k` is an error, so a
# group smaller than the shard count declines to be split and is copied by a single worker.
CLONE_SHARD_BOUNDARIES_TEMPLATE = """
MATCH (n:{label} {{project: $source}})
WITH n.id AS id ORDER BY id
WITH collect(id) AS ids
RETURN size(ids) AS total,
       CASE WHEN size(ids) < $shards THEN []
            ELSE [i IN range(1, $shards - 1) | ids[toInteger(size(ids) * i / $shards)]]
       END AS boundaries
"""

# One label group of Code nodes, keyset-paginated by id so a huge project is copied in
# bounded transactions instead of one that has to hold every vector in the project at once.
# `SET m = properties(n)` copies *everything* — summary, summary_embedding, checksum,
# chunks_checksum, the call-provenance lists — and then the project stamp is overwritten;
# the order matters, since properties(n) carries the source's own project value.
# `max(n.id)` is the cursor: ORDER BY + LIMIT yields a contiguous prefix, so the largest id
# in the batch is where the next one resumes.
#
# `$after`/`$upper` bound the window this statement is allowed to touch, so several copies of
# it can run concurrently on disjoint id ranges (see CLONE_SHARD_BOUNDARIES_TEMPLATE). Empty
# string means "unbounded" on both ends — every id sorts above `''`, and the *last* shard is
# always left open so a row appended to the source after the boundaries were sampled is still
# copied rather than silently dropped between the last boundary and the end of the group.
CLONE_NODES_TEMPLATE = """
MATCH (n:Code:{node_label} {{project: $source}})
WHERE n.id > $after AND ($upper = '' OR n.id <= $upper)
WITH n ORDER BY n.id LIMIT $limit
CREATE (m:Code:{node_label})
SET m = properties(n)
SET m.project = $target
RETURN count(m) AS written, max(n.id) AS last
"""

# Chunks carry the code embeddings — the expensive half of the copy, and the whole reason
# this path exists. Same keyset pagination, same shard bounds, and the vector index picks the
# created nodes up exactly as it does for a client-side write (verified live: vector search
# over the clone returns the cloned chunks).
CLONE_CHUNKS = """
MATCH (n:CodeChunk {project: $source})
WHERE n.id > $after AND ($upper = '' OR n.id <= $upper)
WITH n ORDER BY n.id LIMIT $limit
CREATE (m:CodeChunk)
SET m = properties(n)
SET m.project = $target
RETURN count(m) AS written, max(n.id) AS last
"""

# Dependencies the project names but does not contain. Project-scoped like everything else
# (:External nodes carry `project`), so they are cloned rather than shared.
CLONE_EXTERNALS = """
MATCH (n:External {project: $source})
WHERE n.id > $after AND ($upper = '' OR n.id <= $upper)
WITH n ORDER BY n.id LIMIT $limit
CREATE (m:External)
SET m = properties(n)
SET m.project = $target
RETURN count(m) AS written, max(n.id) AS last
"""

# Which (relationship type, target label) pairs actually exist in the source. Discovered
# rather than hard-coded from EdgeType: a clone that silently dropped a relationship type
# someone added later would be a graph that looks complete and answers "calls nothing".
# Whatever this misses still fails the count verification and falls back to the slow path.
CLONE_EDGE_SHAPES = """
MATCH (s:Code {project: $source})-[r]->(t {project: $source})
RETURN DISTINCT type(r) AS type, labels(t) AS target_labels
"""

# The cursor windows for edge cloning: edges are copied for a bounded slice of source nodes
# at a time. Read separately from the clone itself because a slice may legitimately contain
# no edge of the type being copied, and an aggregate over zero rows returns no row at all —
# so the cursor could not advance if it came back from the write.
CLONE_SOURCE_ID_WINDOW = """
MATCH (a:Code {project: $source})
WHERE a.id > $after
RETURN a.id AS id
ORDER BY a.id
LIMIT $limit
"""

# One relationship type between one pair of label classes, for one cursor window. Both
# endpoints are re-found in the target by id — the ids were copied verbatim, so this is an
# indexed lookup, and it is what makes the copy structurally identical rather than merely
# similarly shaped. `SET r2 = properties(r)` carries CALLS provenance (resolution,
# callee_text) across.
CLONE_EDGES_TEMPLATE = """
MATCH (a:Code {{project: $source}})-[r:{edge_type}]->(b:{target_label} {{project: $source}})
WHERE a.id > $after AND a.id <= $upper
MATCH (a2:Code {{project: $target, id: a.id}})
MATCH (b2:{target_label} {{project: $target, id: b.id}})
CREATE (a2)-[r2:{edge_type}]->(b2)
SET r2 = properties(r)
RETURN count(r2) AS written
"""

# A cloned root node arrives carrying the *donor's* identity: its absolute root_path, its
# persisted include/exclude scope, and (if the donor were itself marked) its worktree fields.
# Those are stripped immediately, before the caller stamps this project's own, so that a
# clone interrupted halfway is invisible to everything that reasons about projects by path:
# LIST_PROJECTS filters on a non-empty root_path, so the janitor never sees a half-clone, and
# nothing can mistake it for a second project rooted at the donor's directory.
CLEAR_CLONED_ROOT_IDENTITY = """
MATCH (n:Code:Directory {id: 'dir:.', project: $project})
SET n.root_path = null,
    n.index_include = null,
    n.index_exclude = null,
    n.worktree_of = null,
    n.worktree_main_root = null,
    n.watched = null,
    n.indexed_head = null
"""

# Verification. Counted independently of the writes, per label and per relationship type, on
# both projects — "the clone reported N rows" is not evidence that N rows exist (INV-7, and
# the 3.12 under-write quirk documented at MARK_CHUNKS_COVERED).
#
# The edge count deliberately matches on unlabelled endpoints: it is the one query here that
# must see relationships the clone never thought about, including any that start somewhere
# other than a :Code node. It costs a scan, which is affordable once per clone and is the
# difference between "verified" and "verified against my own assumptions".
COUNT_CODE_NODES = """
MATCH (n:Code {project: $project})
RETURN count(n) AS count
"""

COUNT_CHUNKS = """
MATCH (n:CodeChunk {project: $project})
RETURN count(n) AS count
"""

COUNT_EXTERNALS = """
MATCH (n:External {project: $project})
RETURN count(n) AS count
"""

COUNT_EDGES_BY_TYPE = """
MATCH (s {project: $project})-[r]->(t {project: $project})
RETURN type(r) AS type, count(r) AS count
"""

# Stored chunk vectors for a named set of files. The whole-project variant above unloads
# every vector in the project over Bolt (7.5s for 8.1k chunks, measured) — affordable once,
# ruinous on a path whose entire point is not moving vectors through the client. After a
# server-side clone only the files that actually differ from the donor are re-chunked, and
# those are the only files whose stored vectors can still be reused (a chunk id hashes the
# file path, so a chunk can never be reused across files).
GET_STORED_CHUNK_EMBEDDINGS_FOR_FILES = """
MATCH (c:CodeChunk {project: $project})
WHERE c.file_path IN $file_paths
RETURN c.id AS chunk_id, c.embedding AS embedding
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

# What each stored IMPORTS edge carries (see Edge.import_fingerprint). The incremental store
# skips re-MERGEing an edge whose key already exists; for IMPORTS that is only a no-op when the
# properties match too — otherwise an index built before the properties existed would never
# get them for files that do not change.
GET_IMPORT_FINGERPRINTS = """
MATCH (s:Code {project: $project})-[r:IMPORTS]->(t {project: $project})
RETURN s.id AS source, t.id AS target, r.fingerprint AS fingerprint
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
#
# The target-label test is written over `labels(t)` for the same reason as
# MARK_CHUNKS_COVERED above: a batched write whose WHERE is a disjunction of pure label
# tests is miscompiled on Memgraph 3.12.0. This one was live-probed as correct in its
# original `(t:Code OR t:External)` form — the disjunction was AND-ed with a non-label
# predicate (`type(r) = row.type`), which kept the planner off the broken path — but the
# safety depended on that incidental pairing, so the shape is gone. The `labels()` form is
# live-validated to delete every row of a batch.
DELETE_EDGES_BY_KEYS = """
UNWIND $rows AS row
MATCH (s:Code {id: row.source, project: $project})-[r]->(t {id: row.target, project: $project})
WHERE type(r) = row.type
  AND ANY(label IN labels(t) WHERE label IN ['Code', 'External'])
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

