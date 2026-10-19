"""IMPORTS edges carry the statement as written — specifier, line, names, and its kind.

Three field reports in a row (17.09.2026) asked the graph "which import statements lead into
this directory, and are they static or dynamic?" and got "module A imports module B" — the
fact that collapses exactly what a refactor turns on: which spelling (``'../tasks'`` vs
``'../tasks/list-view'``), whether it is a lazy ``import()`` chunk or a static dependency of the
bundle, whether it is type-only. And "who reads the constant TASKS" had no answer at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.server.fastmcp import FastMCP

from inflorescence.code_indexer.models import IMPORT_FACT_FIELDS, IMPORT_FACT_FLAGS, Edge, EdgeType
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.javascript_parser import JavaScriptParser
from inflorescence.code_indexer.parser.resolver import merge_import_edges, resolve_edges
from inflorescence.db.repository import _edge_row, _import_facts, _relation_row
from inflorescence.tools.cypher import register_cypher_tools
from tests.test_calls_resolution import _parse_project


def _imports(edges: list[Edge]) -> list[Edge]:
    return [e for e in edges if e.edge_type is EdgeType.IMPORTS]


def test_every_typescript_import_form_says_how_it_was_written(tmp_path: Path) -> None:
    (tmp_path / "board.ts").write_text(
        "import type { ShowTask } from '../tasks/show'\n"
        "import { type Row, readRow } from './rows'\n"
        "export { TASKS } from '../tasks'\n"
        "export * from './all'\n"
        "import './board.css'\n"
        "const u = require('./util')\n"
        "export function mount() {\n"
        "  return import('./BoardHost')\n"
        "}\n",
        encoding="utf-8",
    )

    edges = _imports(JavaScriptParser().parse_file(tmp_path / "board.ts", tmp_path)[1])
    shape = [(e.specifier, e.line, e.names, e.dynamic, e.type_only, e.reexport) for e in edges]

    assert shape == [
        ("../tasks/show", 1, ["ShowTask"], False, True, False),
        ("./rows", 2, ["readRow"], False, False, False),  # a mixed statement splits by kind
        ("./rows", 2, ["Row"], False, True, False),
        ("../tasks", 3, ["TASKS"], False, False, True),
        ("./all", 4, ["*"], False, False, True),
        ("./board.css", 5, [], False, False, False),
        ("./util", 6, ["*"], False, False, False),
        ("./BoardHost", 8, [], True, False, False),  # inside a function, still the module's edge
    ]
    assert all(e.source == "board.ts" for e in edges)


def test_python_imports_carry_module_line_and_name(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("import os\n\nfrom .rules import RULES, apply\n", encoding="utf-8")

    edges = _imports(PythonAstParser().parse_file(tmp_path / "a.py", tmp_path)[1])

    assert [(e.target, e.specifier, e.line, e.names) for e in edges] == [
        ("os", "os", 1, ["*"]),
        (".rules.RULES", ".rules", 3, ["RULES"]),
        (".rules.apply", ".rules", 3, ["apply"]),
    ]


def test_a_named_import_also_reaches_the_symbol_it_reads(tmp_path: Path) -> None:
    """"Who reads TASKS" becomes incoming IMPORTS on the constant itself."""
    (tmp_path / "tasks").mkdir()
    tasks = tmp_path / "tasks/index.ts"
    tasks.write_text("export const TASKS = [1, 2]\nexport default TASKS\n", encoding="utf-8")
    board = tmp_path / "board.ts"
    board.write_text("import BOARD_TASKS, { TASKS } from './tasks/index'\nimport { Missing } from './tasks/index'\n")

    nodes, edges, facts = _parse_project(JavaScriptParser(), tmp_path, [tasks, board])
    imports = {e.target: e for e in _imports(resolve_edges(nodes, edges, facts).edges) if e.source == "board.ts"}

    assert set(imports) == {"tasks/index.ts", "tasks/index.ts::TASKS"}
    # Two statements into one module fold into one edge; `default` and a name the module does
    # not define stay module-level.
    assert imports["tasks/index.ts"].names == ["default", "TASKS", "Missing"]
    assert imports["tasks/index.ts"].line == 1
    assert imports["tasks/index.ts::TASKS"].names == ["TASKS"]
    assert imports["tasks/index.ts::TASKS"].specifier == "./tasks/index"


def test_folding_keeps_a_dependency_static_when_any_statement_is() -> None:
    def edge(line: int, names: list[str], **flags: bool) -> Edge:
        return Edge(source="a.ts", target="b.ts", edge_type=EdgeType.IMPORTS,
                    specifier="./b", line=line, names=names, **flags)

    lazy_and_static = merge_import_edges([edge(9, [], dynamic=True), edge(2, ["x"])])
    assert len(lazy_and_static) == 1
    assert (lazy_and_static[0].line, lazy_and_static[0].names, lazy_and_static[0].dynamic) == (2, ["x"], False)

    only_lazy = merge_import_edges([edge(3, [], dynamic=True), edge(7, [], dynamic=True)])
    assert only_lazy[0].dynamic is True and only_lazy[0].line == 3

    types = merge_import_edges([edge(1, ["A"], type_only=True), edge(2, ["B"], type_only=True, reexport=True)])
    assert types[0].type_only is True and types[0].reexport is True and types[0].names == ["A", "B"]

    other = Edge(source="a.ts", target="b.ts", edge_type=EdgeType.CALLS)
    assert merge_import_edges([other, other]) == [other, other], "only IMPORTS is folded"


def test_the_store_row_and_the_published_row() -> None:
    edge = Edge(source="a.ts", target="b.ts", edge_type=EdgeType.IMPORTS,
                specifier="./b", line=4, names=["x"], dynamic=True)

    row = _edge_row(edge)
    assert row["specifier"] == "./b" and row["fingerprint"] == edge.import_fingerprint()
    assert "specifier" not in _edge_row(Edge(source="a", target="b", edge_type=EdgeType.CONTAINS))

    stored = {"specifier": "./b", "line": 4, "names": ["x"], "dynamic": True, "type_only": False,
              "reexport": False, "fingerprint": "…"}
    assert _import_facts(stored) == {"specifier": "./b", "line": 4, "names": ["x"], "dynamic": True}
    assert _import_facts({}) is None, "an edge written before the properties existed says nothing"

    related = {"relation": "IMPORTS", "edge": stored}
    entity = {"id": "a.ts", "name": "a", "labels": ["Code", "Module"], "file_path": "a.ts",
              "line_start": 1, "line_end": 9, "summary": "A."}
    assert _relation_row(related, entity)["import"]["dynamic"] is True
    assert "import" not in _relation_row({"relation": "CALLS", "edge": {"resolution": "import"}}, entity)


@pytest.mark.asyncio
async def test_the_schema_tool_describes_every_import_property() -> None:
    """graph_schema is what a cypher_query author reads; it must name the properties that exist."""

    class Manager:
        def get_project(self, directory: str) -> str:
            return "p"

    mcp = FastMCP("t")
    register_cypher_tools(mcp, Manager(), conn=None)
    content = await mcp.call_tool("graph_schema", {"directory": "/x"})
    content = content[0] if isinstance(content, tuple) else content
    described = json.loads(content[0].text)["relationship_properties"]["IMPORTS"]
    assert set(described) - {"note"} == {*IMPORT_FACT_FIELDS, *IMPORT_FACT_FLAGS}
