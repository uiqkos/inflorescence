"""Read-only Cypher query MCP tool."""

from __future__ import annotations

import re

from mcp.server.fastmcp import FastMCP

from inflorescence.code_indexer.models import EdgeType, NodeType
from inflorescence.db.connection import MemgraphConnection
from inflorescence.project_manager import ProjectManager
from inflorescence.tools.responses import clamp_limit, clamp_offset, error_response, paginated_response

# Regex to detect write operations — these are blocked for safety
_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|DROP|REMOVE)\b",
    re.IGNORECASE,
)
# CALL invokes a stored procedure. The search procedures (vector_search.search,
# text_search.search_all) read GLOBAL, cross-project indexes, so a CALL inside an
# otherwise project-scoped query silently returns *other projects'* rows — the
# $project MATCH scope does not constrain what the procedure reads (audit finding 12,
# a cross-project leak in both cypher_query and the dashboard's /query). No procedure
# can be guaranteed project-scoped here, so CALL is blocked entirely; project-scoped
# MATCH/WHERE/RETURN cover every legitimate read. The (?<!\.) lookbehind spares the
# property access `n.call` — the CALL keyword can never directly follow a dot.
_CALL_PATTERN = re.compile(r"(?<!\.)\bCALL\b", re.IGNORECASE)
# LOAD / LOAD CSV reads files from the Memgraph *server's* filesystem, not the graph, so
# `LOAD CSV FROM "/etc/hostname" ...` turns a "read-only" query into host-file exfiltration
# and escapes project scope entirely (audit finding T6). It is neither a write nor a CALL,
# so it slipped both patterns; block it outright. (?<!\.) spares the property `n.load`.
_LOAD_PATTERN = re.compile(r"(?<!\.)\bLOAD\b", re.IGNORECASE)
_LIMIT_OR_SKIP_PATTERN = re.compile(r"\b(LIMIT|SKIP)\b", re.IGNORECASE)
# Memgraph admin/introspection statements are neither writes, CALLs, nor LOADs, so they
# slipped every pattern above: `DUMP DATABASE` returns the whole database (every project's
# source, summaries and embeddings) as executable Cypher, `SHOW CONFIG` returns server
# configuration, and `STORAGE MODE IN_MEMORY_ANALYTICAL` silently disables ACID and
# durability. The leading-clause allowlist below already refuses them, but they are denied
# explicitly too — an allowlist and a denylist fail in different directions. (?<!\.) spares
# property access like `n.dump`.
_ADMIN_PATTERN = re.compile(
    r"(?<!\.)\b(DUMP|SHOW|STORAGE|TERMINATE|FREE|LOCK|REPLICA|TRIGGER|STREAM|PROFILE)\b",
    re.IGNORECASE,
)
# A read query can only begin by matching or streaming rows. Anchoring the first clause is
# what turns the guard from "reject the bad statements I thought of" into "accept only the
# shapes a caller legitimately needs" — every admin statement, and anything Memgraph adds in
# a future version, is refused by default rather than needing its own pattern.
_ALLOWED_LEADING_CLAUSE = re.compile(r"^\s*(?:OPTIONAL\s+MATCH|MATCH|WITH|UNWIND)\b", re.IGNORECASE)
_MATCH_CLAUSE = re.compile(
    r"\b(?:OPTIONAL\s+)?MATCH\b(?P<body>.*?)(?=\b(?:OPTIONAL\s+MATCH|MATCH|WHERE|WITH|RETURN|CALL|UNWIND|ORDER\s+BY|LIMIT|SKIP|UNION)\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_PROJECT_SCOPED_NODE = re.compile(
    r"\([^)]*\{\s*[^}]*\bproject\s*:\s*\$project\b[^}]*\}[^)]*\)",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_readonly_query(query: str) -> str:
    """Return the exact text that must be executed for *query*.

    Comments are blanked and trailing semicolons removed. Callers **must** execute this
    rather than the original: pagination is appended as text, so a query ending in a `//`
    comment would otherwise comment the appended ``SKIP``/``LIMIT`` out and return the whole
    result set. Validation runs over the same sanitized text, so what was checked and what
    runs can never diverge.
    """
    return _blank_comments(query).rstrip().rstrip(";").rstrip()


def validate_readonly_query(query: str) -> str | None:
    """Validate a user-supplied Cypher query for read-only, project-scoped execution.

    Returns an error message, or None if the query is safe to run — after
    :func:`sanitize_readonly_query` — with ``SKIP $offset LIMIT $limit`` appended and a
    ``$project`` parameter bound.
    """
    # Comments are blanked first, before any other check. Cypher supports `//` and `/* */`,
    # and leaving them in defeated the guard three ways at once: a decoy `{project: $project}`
    # inside a comment satisfied the scope check while the real pattern stayed unscoped, a
    # trailing `//` ate the appended pagination, and `DUMP DATABASE /* MATCH (x {project:
    # $project}) */` passed as a "project-scoped read".
    comment_free = _blank_comments(query)
    # Keyword and scope checks then run against a copy with string literals blanked out, so a
    # term that appears *inside a quoted value* — `WHERE n.summary CONTAINS 'DROP'`, common in
    # code about databases — is not mistaken for the write/CALL/LOAD keyword (audit finding
    # T7). The scope check runs on this copy too, not on the original: a literal such as
    # `WHERE n.x = '(y {project: $project})'` otherwise supplied a decoy scope pattern the
    # same way a comment did.
    keyword_scope = _strip_string_literals(comment_free)
    if _WRITE_PATTERN.search(keyword_scope):
        return "Write operations are not allowed."
    if _CALL_PATTERN.search(keyword_scope):
        return "CALL is not allowed — stored procedures (e.g. *.search) read global indexes and bypass project scoping."
    if _LOAD_PATTERN.search(keyword_scope):
        return "LOAD / LOAD CSV is not allowed — it reads files from the Memgraph server host and bypasses project scoping."
    if _ADMIN_PATTERN.search(keyword_scope):
        return (
            "Administrative and introspection statements (DUMP, SHOW, STORAGE MODE, "
            "TERMINATE, ...) are not allowed — use graph_schema to describe the graph."
        )
    if _LIMIT_OR_SKIP_PATTERN.search(keyword_scope):
        return "Use tool-level limit/offset instead of Cypher LIMIT or SKIP."
    if not _ALLOWED_LEADING_CLAUSE.match(keyword_scope):
        return "Query must start with MATCH, OPTIONAL MATCH, WITH or UNWIND."
    # Statement chaining would let a second, unchecked statement ride along behind a
    # well-formed first one; only a single trailing semicolon is tolerated.
    if ";" in keyword_scope.strip().rstrip(";"):
        return "Only a single statement is allowed."
    if not _has_only_project_scoped_matches(keyword_scope):
        return "Query must include a project-scoped MATCH pattern such as (n:Code {project: $project})."
    return None


def _blank_comments(query: str) -> str:
    """Return *query* with Cypher `//` and `/* */` comments replaced by spaces.

    String literals and backtick identifiers are passed through untouched, so a `//` inside
    a quoted URL is data rather than a comment; conversely a quote character inside a comment
    cannot flip the scanner into literal mode. Newlines are preserved so that a `//` comment
    following a block comment still ends at the right line.
    """
    out: list[str] = []
    index = 0
    length = len(query)
    quote: str | None = None
    in_backtick = False
    escaped = False
    while index < length:
        char = query[index]
        if in_backtick:
            out.append(char)
            if char == "`":
                in_backtick = False
            index += 1
            continue
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "`":
            in_backtick = True
            out.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length and query[index + 1] == "/":
            while index < length and query[index] != "\n":
                out.append(" ")
                index += 1
            continue
        if char == "/" and index + 1 < length and query[index + 1] == "*":
            # An unterminated block comment blanks to end of input rather than being left
            # intact — refusing to guess where the author meant it to close.
            out.append("  ")
            index += 2
            while index < length and not (query[index] == "*" and index + 1 < length and query[index + 1] == "/"):
                out.append("\n" if query[index] == "\n" else " ")
                index += 1
            if index < length:
                out.append("  ")
                index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _strip_string_literals(query: str) -> str:
    """Return *query* with string literals and backtick identifiers blanked to spaces.

    Single- and double-quoted literals (with backslash escapes) are blanked so keyword
    scanning never trips on a term that is really data. Backtick-quoted identifiers are
    blanked too, and must be tracked here: backticks have their own quoting rules (no
    backslash escapes; a quote character inside backticks is literal), so treating a
    quote inside backticks as the start of a string would flip the stripper into literal
    mode and blank *real* keywords after it — a write-bypass. Structure outside literals
    (parentheses, braces) is preserved, so the project-scope check runs on this same copy;
    it must not run on the original, or a scope pattern written inside a string literal
    would satisfy it.
    """
    out: list[str] = []
    quote: str | None = None
    in_backtick = False
    escaped = False
    for char in query:
        if in_backtick:
            if char == "`":
                in_backtick = False
            out.append(" ")
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            out.append(" ")
            continue
        if char == "`":
            in_backtick = True
            out.append(" ")
        elif char in {"'", '"'}:
            quote = char
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


def register_cypher_tools(mcp: FastMCP, manager: ProjectManager, conn: MemgraphConnection) -> None:
    """Register the read-only Cypher tool with the MCP server."""

    @mcp.tool()
    async def cypher_query(directory: str, query: str, limit: int = 100, offset: int = 0) -> dict:
        """Execute a paginated read-only Cypher query against the project's graph.

        Call `graph_schema` first if unsure what is queryable (labels, node types,
        relationship types, property keys).

        Rules enforced by this tool:
          - Read-only: CREATE, MERGE, SET, DELETE, DETACH, DROP, REMOVE are blocked.
          - Must start with MATCH, OPTIONAL MATCH, WITH or UNWIND; one statement per call.
          - CALL, LOAD and administrative statements (DUMP, SHOW, STORAGE MODE, ...) are blocked.
          - Every MATCH node pattern must be project-scoped, e.g. (n:Code {project: $project}).
          - Do NOT write LIMIT/SKIP — pass the tool-level `limit`/`offset` args instead.

        Nodes are labelled `Code` plus one type label (`:Method`, `:Function`, `:Struct`, ...).
        The type is also stored as the `node_type` property, so both work:
          MATCH (n:Method {project: $project}) RETURN n.name
          MATCH (n:Code {project: $project}) WHERE n.node_type = 'method' RETURN n.name
        Relationships: CONTAINS, CALLS, IMPORTS, INHERITS, IMPLEMENTS, HAS_CHUNK.

        Args:
            directory: Absolute path of the indexed project.
            query: Cypher query string (read-only).
        """
        validation_error = validate_readonly_query(query)
        if validation_error is not None:
            code = "invalid_argument" if "limit/offset" in validation_error else "unsafe_query"
            return error_response(code, validation_error)

        project = manager.get_project(directory)
        # A path that was never indexed synthesises a valid-but-empty project id; a scoped
        # query then returns an empty page indistinguishable from "no rows matched". Surface
        # it distinctly so the caller indexes first rather than trusting silence (finding T5).
        if not await manager.project_is_indexed(project):
            return error_response(
                "project_not_indexed",
                f"No index found for {directory!r} (project '{project}'). Index it with "
                "index_directory first — this is different from a query that matched no rows.",
            )
        bounded_limit = clamp_limit(limit)
        bounded_offset = clamp_offset(offset)
        # Execute the sanitized text, never the original — see sanitize_readonly_query.
        paginated_query = f"{sanitize_readonly_query(query)} SKIP $offset LIMIT $limit"

        try:
            records = await conn.execute_query(
                paginated_query,
                {"project": project, "limit": bounded_limit + 1, "offset": bounded_offset},
                # Belt to the SKIP/LIMIT braces: the cursor stops after this many rows even
                # if the appended clause never took effect, so a caller can never make the
                # server materialise an entire project into memory.
                max_records=bounded_limit + 1,
            )
            return paginated_response(
                [dict(record) for record in records],
                bounded_limit,
                bounded_offset,
                {"project": project, "query_type": "cypher"},
            )
        except Exception as exc:
            return error_response("query_failed", str(exc))

    @mcp.tool()
    async def graph_schema(directory: str) -> dict:
        """Describe the code graph so you can write a cypher_query without guessing.

        Returns the node labels, node-type values, relationship types, the property keys on
        Code/CodeChunk nodes, the named vector/text indexes, the read-only query rules, and a
        few worked examples. No graph scan — this is a static description of the schema.

        Args:
            directory: Absolute path of the indexed project (used to echo the project id).
        """
        project = manager.get_project(directory)
        type_labels = [value.capitalize() for value in (t.value for t in NodeType)]
        return {
            "project": project,
            "node_labels": {
                "shared": ["Code", "CodeChunk", "External"],
                "by_type": type_labels,
                "note": "Every code entity has label `Code` plus exactly one type label; chunks are "
                        "`CodeChunk`. Dependencies the project names but does not contain — an "
                        "imported package, an inherited base class — are `External` and deliberately "
                        "NOT `Code`, so they stay out of entity listings, counts and search. They "
                        "carry name/root/kind/language and are reached through IMPORTS and INHERITS.",
            },
            "external_properties": {
                "name": "the specifier as written: 'net/http', '@electron-forge/maker-zip', 'pydantic.BaseModel'",
                "root": "the distributable it belongs to: 'net/http', '@electron-forge/maker-zip', 'pydantic'",
                "kind": "module (an import) | class (a base class)",
                "language": "language of the file that named it",
            },
            "node_types": [t.value for t in NodeType],
            "relationship_types": [e.value.upper() for e in EdgeType] + ["HAS_CHUNK"],
            "relationship_properties": {
                "CALLS": {
                    "resolution": "how the target was bound: semantic (SCIP compiler) | exact | "
                                  "import | field-type | self | same-module | same-package; "
                                  "absent on graphs indexed before the resolution ladder",
                    "callee_text": "the call as written at the call site, e.g. 's.content.ForCheck'",
                },
            },
            "entity_properties": [
                "id", "node_type", "name", "file_path", "line_start", "line_end",
                "signature", "docstring", "summary", "summary_input_hash", "parent_id",
                "project", "summary_embedding", "checksum", "root_path",
                "external_calls", "unresolved_calls", "calls_provenance",
            ],
            "calls_semantics": (
                "A function with 0 outgoing CALLS edges only truly calls nothing when its "
                "external_calls AND unresolved_calls lists are also empty. external_calls = "
                "calls into stdlib/third-party code; unresolved_calls = calls the resolver "
                "could not bind (dynamic dispatch etc.). Module nodes carry calls_provenance "
                "('scip-go'/'scip-python'/'scip-typescript' = compiler-grade, 'heuristic' = "
                "syntactic ladder)."
            ),
            "chunk_properties": [
                "id", "node_id", "file_path", "content", "start_line", "end_line",
                "language", "chunk_kind", "embedding", "project",
            ],
            "indexes": {
                "vector": ["code_embeddings", "summary_embeddings"],
                "text": ["summary_text", "code_symbols_text", "chunk_content_text"],
            },
            "query_rules": [
                "Read-only: CREATE, MERGE, SET, DELETE, DETACH, DROP, REMOVE are blocked.",
                "Must start with MATCH, OPTIONAL MATCH, WITH or UNWIND. CALL, LOAD and "
                "administrative statements (DUMP, SHOW, STORAGE MODE, ...) are blocked.",
                "One statement per call — no `;`-chaining.",
                "Every MATCH node pattern must be scoped: (n:Code {project: $project}).",
                "Do not use Cypher LIMIT/SKIP — pass tool-level limit/offset instead.",
                "Comments are allowed but stripped before execution; they cannot carry "
                "the required project scope.",
                "$project is bound automatically from `directory`.",
            ],
            "examples": [
                {
                    "question": "What does this project depend on, most used first?",
                    "cypher": "MATCH (:Code {project: $project})-[:IMPORTS]->(e:External {project: $project}) "
                              "RETURN e.root AS dependency, count(*) AS uses ORDER BY uses DESC",
                },
                {
                    "question": "Which files import net/http?",
                    "cypher": "MATCH (n:Code {project: $project})-[:IMPORTS]->(e:External {project: $project}) "
                              "WHERE e.name = 'net/http' RETURN n.file_path",
                },
                "MATCH (n:Code {project: $project}) RETURN n.node_type AS type, count(*) AS c ORDER BY c DESC",
                "MATCH (n:Method {project: $project}) WHERE n.file_path STARTS WITH 'backend/' RETURN n.name",
                "MATCH (a:Code {project: $project})-[r:CALLS]->(b:Code {project: $project}) "
                "RETURN a.name, b.name, r.resolution",
                "MATCH (n:Code {project: $project}) WHERE n.node_type IN ['function', 'method'] "
                "AND NOT (n)-[:CALLS]->() AND size(coalesce(n.external_calls, [])) = 0 "
                "AND size(coalesce(n.unresolved_calls, [])) = 0 RETURN n.name  // true call-graph leaves",
            ],
        }


def _has_only_project_scoped_matches(query: str) -> bool:
    clauses = [match.group("body") for match in _MATCH_CLAUSE.finditer(query)]
    if not clauses:
        return False
    return all(
        _PROJECT_SCOPED_NODE.search(part) is not None
        for clause in clauses
        for part in _split_top_level_commas(clause)
    )


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False

    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1

    parts.append(text[start:].strip())
    return [part for part in parts if part]
