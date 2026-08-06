"""G6 · Resilience & annoyances — oracles for INV-1/INV-3/INV-5/INV-6/INV-7 and validator correctness.

The contract (definition of done, invariants-fix-plan.md · G6):
  * One broken/unreadable file must not crash the whole run or make a project permanently
    unindexable — a NUL-byte ``.py`` (``ValueError``) or a file that vanishes mid-flush
    (``OSError``) degrades to *stale* data, never a crash and never a delete (INV-1/INV-3).
  * A no-op flush must cost nothing: no paid embedding preflight and no O(project) vector
    dump before it is known there is nothing to do (INV-5).
  * The read-only Cypher tool is actually read-only and cannot read host files: ``LOAD`` /
    ``LOAD CSV`` is blocked; a keyword that appears inside a string *literal* (``CONTAINS
    'DROP'``) is not mistaken for a write (validator correctness).
  * A query against a never-indexed project is distinguishable from one that found no
    matches (INV-7 / finding T5).

The pure-function and filesystem oracles run everywhere. The end-to-end oracles are marked
``@live`` and run only when ``INFLORESCENCE_TEST_MEMGRAPH_URL`` points at a **throwaway**
Memgraph-MAGE instance, so CI and a real ``bolt://localhost:7687`` are never touched. Live
tests fake only the paid external APIs (LLM + embedders, deterministic); every scan,
checksum, reconcile, and query runs for real against the live graph.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inflorescence.code_indexer.models import FileChecksum
from inflorescence.code_indexer.parser.ast_parser import parse_file
from inflorescence.rag.chunker import Chunker
from inflorescence.tools.cypher import sanitize_readonly_query, validate_readonly_query

LIVE_URL = os.environ.get("INFLORESCENCE_TEST_MEMGRAPH_URL")
live = pytest.mark.skipif(
    not LIVE_URL,
    reason="set INFLORESCENCE_TEST_MEMGRAPH_URL to a throwaway Memgraph-MAGE instance to run",
)

_DIM = 1536


# ===========================================================================
# Parser resilience — NUL byte and OSError must not crash the run (INV-1/INV-3)
# ===========================================================================


def test_parser_survives_nul_byte_python_file(tmp_path: Path) -> None:
    """A .py with an embedded NUL byte does not crash the run — no nodes, no raise.

    ast.parse rejects a source string with a NUL byte: older CPython raised ``ValueError``
    (which the old ``except (SyntaxError, UnicodeDecodeError)`` missed → the whole run
    aborted and the project stayed unindexable), while 3.12 raises ``SyntaxError``. The
    widened except now covers ``ValueError`` (and ``OSError``) too, so parse_file returns
    empty on any version and reconcile keeps prior data stale rather than deleting it.
    """
    bad = tmp_path / "bad.py"
    bad.write_bytes(b"def ok():\n    return 1\n\x00\n")
    # Must not raise; produces nothing so reconcile treats it as "failed", never a delete.
    nodes, edges = parse_file(bad, tmp_path)
    assert nodes == []
    assert edges == []


def test_parser_survives_unreadable_file(tmp_path: Path) -> None:
    """A file that cannot be read (vanished mid-flush) yields no nodes rather than raising."""
    missing = tmp_path / "gone.py"  # never created
    nodes, edges = parse_file(missing, tmp_path)
    assert (nodes, edges) == ([], [])


def test_parser_still_parses_valid_python(tmp_path: Path) -> None:
    """Guard: the widened except must not swallow a genuinely valid file (no false negative)."""
    good = tmp_path / "good.py"
    good.write_text("def f():\n    return 2\n")
    nodes, _ = parse_file(good, tmp_path)
    assert any(n.name == "f" for n in nodes)


# ===========================================================================
# FileChecksum.try_compute — unreadable file returns None, not a raise (INV-3)
# ===========================================================================


def test_try_compute_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert FileChecksum.try_compute(tmp_path / "nope.py", tmp_path) is None


def test_try_compute_matches_compute_for_readable_file(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    assert FileChecksum.try_compute(f, tmp_path).md5 == FileChecksum.compute(f, tmp_path).md5


# ===========================================================================
# chunk_repository — an unreadable file is reported, not fatal, and its
# chunks are kept (excluded from prune scope by the caller) (INV-1/INV-3)
# ===========================================================================


def test_chunk_repository_reports_unreadable_file_without_crashing(tmp_path: Path) -> None:
    """A file present at parse time but gone at chunk time is collected, not raised.

    The caller (RAGIndexer.index_code) subtracts these from the prune scope and coverage,
    so a transient read failure never empties good chunks (finding: OSError in chunk_repository).
    """
    from inflorescence.code_indexer.models import CodeNode, NodeType

    present = tmp_path / "present.py"
    present.write_text("def a():\n    return 1\n")
    nodes = [
        CodeNode(id="present.py", name="present", node_type=NodeType.MODULE,
                 file_path="present.py", line_start=1, line_end=2),
        # A node whose file was never written to disk: stands in for a file that vanished
        # between the scan/parse and this chunk pass.
        CodeNode(id="gone.py", name="gone", node_type=NodeType.MODULE,
                 file_path="gone.py", line_start=1, line_end=2),
    ]
    unreadable: set[str] = set()
    chunks = Chunker().chunk_repository(tmp_path, nodes, unreadable_out=unreadable)
    assert "gone.py" in unreadable
    assert any(c.file_path == "present.py" for c in chunks)  # the healthy file still chunked


def test_chunk_repository_reports_permission_error_as_unreadable_not_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file that exists but cannot be opened (EACCES/EIO) is unreadable, not "binary".

    The binary/oversize probes used to swallow OSError as "is binary" → the file skipped
    the unreadable_out report, got stamped covered, and its stored chunks were pruned —
    a data-loss the unreadable path exists to prevent.
    """
    from inflorescence.code_indexer.models import CodeNode, NodeType

    locked = tmp_path / "locked.py"
    locked.write_text("def a():\n    return 1\n")
    real_open = Chunker._is_binary_strict

    def _denied(self: Chunker, path: Path) -> bool:
        if path.name == "locked.py":
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(self, path)

    monkeypatch.setattr(Chunker, "_is_binary_strict", _denied)
    nodes = [
        CodeNode(id="locked.py", name="locked", node_type=NodeType.MODULE,
                 file_path="locked.py", line_start=1, line_end=2),
    ]
    unreadable: set[str] = set()
    chunks = Chunker().chunk_repository(tmp_path, nodes, unreadable_out=unreadable)
    assert "locked.py" in unreadable
    assert chunks == []


# ===========================================================================
# Cypher validator — T6 (LOAD blocked) and T7 (string literals not writes)
# ===========================================================================


def test_validator_blocks_load_csv_host_file_read() -> None:
    """LOAD CSV reads the Memgraph host's filesystem — blocked even though it is not a write."""
    err = validate_readonly_query(
        "LOAD CSV FROM \"/etc/hostname\" WITH HEADER AS row "
        "MATCH (n:Code {project: $project}) RETURN n"
    )
    assert err is not None
    assert "LOAD" in err


def test_validator_blocks_bare_load() -> None:
    assert validate_readonly_query("LOAD 'x' MATCH (n:Code {project: $project}) RETURN n") is not None


def test_validator_allows_write_keyword_inside_string_literal() -> None:
    """`CONTAINS 'DROP'` is data, not a write — a false block on code-about-databases (T7)."""
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) WHERE n.summary CONTAINS 'DROP' RETURN n.name"
    ) is None
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) WHERE n.docstring CONTAINS 'CREATE TABLE' RETURN n"
    ) is None
    # A CALL / LOAD mentioned only inside a literal is likewise not the real keyword.
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) WHERE n.name = 'LOAD CSV helper' RETURN n"
    ) is None


def test_validator_still_blocks_real_write_and_call() -> None:
    """Guard: stripping literals must not let a real write/CALL through."""
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) SET n.x = 1 RETURN n"
    ) is not None
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) WHERE n.name CONTAINS 'safe' DETACH DELETE n"
    ) is not None
    assert validate_readonly_query(
        "CALL vector_search.search('code_embeddings', 5, $v) YIELD node "
        "MATCH (node) WHERE node.project = $project RETURN node"
    ) is not None


def test_validator_backtick_identifier_cannot_smuggle_a_write() -> None:
    """A quote inside a backtick identifier must not flip the stripper into literal mode.

    Backticks follow their own quoting rules (a quote char inside them is literal), so a
    stripper that only tracks '/" would blank the *real* SET after ``n.`a'b``` and let the
    write through — on both the MCP tool and the dashboard /query (write-bypass regression).
    """
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) WHERE n.`a'b` IS NULL "
        "SET n.hacked = 'pwned' RETURN n"
    ) is not None
    assert validate_readonly_query(
        'MATCH (n:Code {project: $project}) WHERE n.`a"b` IS NULL '
        "DETACH DELETE n"
    ) is not None
    # A keyword inside a backtick identifier is a name, not the keyword: allowed.
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) RETURN n.`load avg`, n.`call site`"
    ) is None


def test_validator_allows_property_access_named_like_keywords() -> None:
    """`n.load` / `n.call` are property reads, not the LOAD/CALL keywords."""
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) RETURN n.load"
    ) is None
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) WHERE n.call IS NOT NULL RETURN n.name"
    ) is None


# ===========================================================================
# Cypher validator — comments are not a bypass
# ===========================================================================
#
# Cypher supports `//` and `/* */`, and the validator used to ignore both. That single
# omission defeated the guard three separate ways, so each one gets its own test.


def test_validator_rejects_scope_pattern_hidden_in_a_comment() -> None:
    """A commented-out `{project: $project}` must not satisfy the project-scope check.

    The scope check ran against the raw query while keyword checks ran against a stripped
    copy, so a decoy scope pattern inside a comment made an unscoped MATCH look scoped —
    a cross-project read of every repo indexed on the machine.
    """
    error = validate_readonly_query("MATCH (n:Code /* {project: $project} */) RETURN DISTINCT n.project")
    assert error is not None
    assert "project-scoped" in error
    # Same decoy, line-comment form, spanning the pattern across lines.
    assert validate_readonly_query("MATCH (n:Code\n// {project: $project}\n) RETURN n") is not None


def test_validator_rejects_scope_pattern_hidden_in_a_string_literal() -> None:
    """The string-literal form of the same decoy — literals are blanked before the scope check."""
    assert validate_readonly_query("MATCH (n:Code) WHERE n.x = '(y {project: $project})' RETURN n") is not None


def test_validator_rejects_admin_statements_wrapped_in_a_comment() -> None:
    """`DUMP DATABASE` and friends are neither writes, CALLs nor LOADs — deny them by shape.

    `DUMP DATABASE` returns the entire database (every project's source, summaries and
    embeddings) as executable Cypher; `STORAGE MODE IN_MEMORY_ANALYTICAL` disables ACID and
    durability on the user's database.
    """
    for statement in (
        "DUMP DATABASE /* MATCH (x {project: $project}) */",
        "SHOW CONFIG /* MATCH (x {project: $project}) */",
        "SHOW STORAGE INFO /* MATCH (x {project: $project}) */",
        "STORAGE MODE IN_MEMORY_ANALYTICAL /* MATCH (x {project: $project}) */",
        "TERMINATE TRANSACTIONS 'x' /* MATCH (y {project: $project}) */",
    ):
        assert validate_readonly_query(statement) is not None, statement


def test_validator_rejects_statement_chaining() -> None:
    """A second statement behind a well-formed first one would ride along unchecked."""
    assert validate_readonly_query(
        "MATCH (n:Code {project: $project}) RETURN n; MATCH (m:Code) RETURN m"
    ) is not None
    # A single trailing semicolon is ordinary Cypher and stays allowed.
    assert validate_readonly_query("MATCH (n:Code {project: $project}) RETURN n.name;") is None


def test_sanitize_strips_comments_so_appended_pagination_survives() -> None:
    """Pagination is appended as text, so a trailing comment must not be able to eat it.

    `... RETURN n //` used to yield `... RETURN n // SKIP $offset LIMIT $limit` — the caller
    asked for a page and received every row in the project.
    """
    for query in (
        "MATCH (n:Code {project: $project}) RETURN n.id //",
        "MATCH (n:Code {project: $project}) RETURN n /* unterminated",
        "MATCH (n:Code {project: $project}) RETURN n //\n// trailing",
    ):
        assert validate_readonly_query(query) is None, query
        paginated = f"{sanitize_readonly_query(query)} SKIP $offset LIMIT $limit"
        assert paginated.endswith("SKIP $offset LIMIT $limit"), paginated
        assert "//" not in paginated and "/*" not in paginated, paginated


def test_sanitize_preserves_comment_characters_inside_string_literals() -> None:
    """`//` inside a quoted value is data — blanking it would corrupt the query."""
    query = "MATCH (n:Code {project: $project}) WHERE n.name = 'http://x//y' RETURN n"
    assert validate_readonly_query(query) is None
    assert "'http://x//y'" in sanitize_readonly_query(query)


def test_validator_requires_a_read_clause_first() -> None:
    """Anchoring the first clause is what makes unknown future statements refused by default."""
    assert validate_readonly_query("RETURN 1") is not None
    assert validate_readonly_query("MATCH (n:Code {project: $project}) RETURN n") is None
    assert validate_readonly_query("OPTIONAL MATCH (n:Code {project: $project}) RETURN n") is None
    assert validate_readonly_query(
        "UNWIND [1,2] AS x MATCH (n:Code {project: $project}) WHERE n.line = x RETURN n"
    ) is None


# ===========================================================================
# watcher._is_excluded — glob patterns filtered as early as possible (INV-6)
# ===========================================================================


def test_watcher_excludes_glob_and_component_patterns() -> None:
    from inflorescence.watcher import _DebouncedHandler

    handler = _DebouncedHandler(
        project="p",
        project_manager=None,  # _is_excluded never touches it
        debounce_seconds=0.0,
        exclude_patterns=["node_modules", "*.log", "*.min.js"],
        loop=None,  # _is_excluded never touches it
        extensions=None,
    )
    assert handler._is_excluded("/repo/app/build.log") is True       # *.log glob
    assert handler._is_excluded("/repo/static/app.min.js") is True   # *.min.js glob
    assert handler._is_excluded("/repo/node_modules/x/index.js") is True  # component
    assert handler._is_excluded("/repo/src/main.py") is False        # legitimate source


# ===========================================================================
# LIVE oracles — real Memgraph, fake paid APIs only
# ===========================================================================


class _FakeEmbedder:
    """Deterministic 1536-dim embedder that counts calls (stands in for a paid endpoint)."""

    def __init__(self) -> None:
        self.model = "fake-embed"
        self.embed_calls = 0
        self.preflight_calls = 0

    def preflight(self) -> None:
        self.preflight_calls += 1
        self.embed_calls += 1  # the real preflight embeds ["ok"] — one paid call

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return []
        self.embed_calls += 1
        return [[0.01 * ((i % 97) + 1)] * _DIM for i, _ in enumerate(texts)]


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        return "a concise summary"

    async def close(self) -> None:
        pass


async def _live_manager(project_root: Path):
    """Build a real ProjectManager against the live graph; only LLM + embedders are fake."""
    from inflorescence.code_indexer.graph_builder import GraphBuilder
    from inflorescence.config import Settings
    from inflorescence.db.connection import MemgraphConnection
    from inflorescence.db.repository import GraphRepository
    from inflorescence.db.schema import setup_schema
    from inflorescence.project_manager import ProjectManager
    from inflorescence.rag.indexer import RAGIndexer

    settings = Settings(_env_file=None, memgraph_url=LIVE_URL, embedding_dimension=_DIM)
    conn = MemgraphConnection(settings)
    await setup_schema(conn)
    repo = GraphRepository(conn)
    llm = _FakeLLM()
    builder = GraphBuilder(repo=repo, llm=llm, settings=settings)
    rag = RAGIndexer(repo=repo, settings=settings)
    code_embedder = _FakeEmbedder()
    summary_embedder = _FakeEmbedder()
    rag._code_embedder = code_embedder
    rag._summary_embedder = summary_embedder
    manager = ProjectManager(repo=repo, graph_builder=builder, rag_indexer=rag, settings=settings, conn=conn)
    return manager, repo, conn, code_embedder, summary_embedder, llm


@live
async def test_noop_flush_costs_no_api_calls_or_project_dump_live(tmp_path: Path) -> None:
    """INV-5: a watcher flush with no real change must not embed or dump all project vectors.

    Reproduces the audit scenario (project_manager no-op path): index a small project, then
    fire an update for an unchanged file. Before the fix the flush called `_preflight()`
    (a paid embedding) and `get_stored_chunk_embeddings` (every vector) *before* discovering
    the no-op. After the fix a cheap scan+checksum diff short-circuits both.
    """
    proj_dir = tmp_path / "mini"
    proj_dir.mkdir()
    (proj_dir / "m.py").write_text("def hello():\n    return 'hi'\n")

    manager, repo, conn, code_emb, summ_emb, llm = await _live_manager(proj_dir)
    project = manager.get_project(str(proj_dir))
    try:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        # Full index once (this legitimately spends: embeds + summaries).
        await manager.index_directory(str(proj_dir))
        assert code_emb.embed_calls > 0  # sanity: the first index did real work

        # Spy on the O(project) vector dump so we can prove it is not reached on a no-op.
        dump_calls = {"n": 0}
        original_dump = repo.get_stored_chunk_embeddings

        async def _counting_dump(p: str):
            dump_calls["n"] += 1
            return await original_dump(p)

        repo.get_stored_chunk_embeddings = _counting_dump  # type: ignore[method-assign]

        # Reset counters, then flush an UNCHANGED file — a genuine no-op.
        code_emb.embed_calls = 0
        summ_emb.embed_calls = 0
        summ_emb.preflight_calls = 0
        llm.calls = 0

        result = await manager.update_project(project, [str(proj_dir / "m.py")])

        assert result.get("noop") is True
        assert summ_emb.preflight_calls == 0, "no-op must not run the paid preflight embedding"
        assert summ_emb.embed_calls == 0 and code_emb.embed_calls == 0, "no-op must not embed"
        assert dump_calls["n"] == 0, "no-op must not dump every stored vector (O(project))"
    finally:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await conn.close()


@live
async def test_nul_byte_file_does_not_break_index_of_the_rest_live(tmp_path: Path) -> None:
    """INV-1/INV-3: a NUL-byte .py survives a real index and the rest of the project is indexed."""
    proj_dir = tmp_path / "withbad"
    proj_dir.mkdir()
    (proj_dir / "good.py").write_text("def good():\n    return 1\n")
    (proj_dir / "bad.py").write_bytes(b"def broken():\n\x00\n")

    manager, repo, conn, *_ = await _live_manager(proj_dir)
    project = manager.get_project(str(proj_dir))
    try:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        # Must not raise despite the NUL-byte file.
        stats = await manager.index_directory(str(proj_dir))
        assert stats["nodes"] > 0
        # The healthy file is indexed.
        rows = await conn.execute_query(
            "MATCH (n:Code {project: $p}) WHERE n.name = 'good' RETURN n.id AS id", {"p": project}
        )
        assert rows, "the valid file next to the NUL-byte file was still indexed"
    finally:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await conn.close()


@live
async def test_search_distinguishes_unindexed_project_from_no_matches_live(tmp_path: Path) -> None:
    """INV-7 / T5: a never-indexed directory returns a distinct error, not a silent empty page."""
    from mcp.server.fastmcp import FastMCP

    from inflorescence.config import Settings
    from inflorescence.tools.cypher import register_cypher_tools
    from inflorescence.tools.search import register_search_tools

    proj_dir = tmp_path / "real"
    proj_dir.mkdir()
    (proj_dir / "z.py").write_text("def findme():\n    return 42\n")

    manager, repo, conn, *_ = await _live_manager(proj_dir)
    project = manager.get_project(str(proj_dir))
    try:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await manager.index_directory(str(proj_dir))

        settings = Settings(_env_file=None, memgraph_url=LIVE_URL, embedding_dimension=_DIM)
        mcp = FastMCP("t")
        register_search_tools(mcp, manager, repo, settings)
        register_cypher_tools(mcp, manager, conn)

        def _payload(res):
            content = res[0] if isinstance(res, tuple) else res
            return json.loads(content[0].text)

        # A path that was never indexed → distinct "project_not_indexed" error.
        unindexed_dir = tmp_path / "never_indexed"
        unindexed_dir.mkdir()
        res = await mcp.call_tool("search_text", {"directory": str(unindexed_dir), "query": "findme"})
        payload = _payload(res)
        assert payload.get("error", {}).get("code") == "project_not_indexed", payload

        res_cy = await mcp.call_tool(
            "cypher_query",
            {"directory": str(unindexed_dir), "query": "MATCH (n:Code {project: $project}) RETURN n.name"},
        )
        assert _payload(res_cy).get("error", {}).get("code") == "project_not_indexed"

        # The real, indexed project with a term that matches nothing → a normal empty page,
        # NOT the not-indexed error (the two are distinguishable).
        res_ok = await mcp.call_tool(
            "search_text", {"directory": str(proj_dir), "query": "zzz_no_such_symbol_qqq"}
        )
        ok_payload = _payload(res_ok)
        assert "error" not in ok_payload, ok_payload
        assert ok_payload.get("items") == []
    finally:
        await conn.execute_write("MATCH (n {project: $p}) DETACH DELETE n", {"p": project})
        await conn.close()
