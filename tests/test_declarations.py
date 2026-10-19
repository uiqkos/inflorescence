"""Top-level constants, variables and type aliases as graph entities.

Field reports kept asking for a rule table, a tuning object or a registry by its name ("who
reads TASKS", "where is RETRY_POLICY defined", `type OrderTotals`) and getting nothing: only
functions, classes and their kin had nodes. These tests pin down the three promises that make
declarations nodes without making them expensive:

* every supported language emits them, with the declaration's first line as the signature;
* they are never summarized and never enter a parent's summary input — a module hashes the
  same with or without them, so no summary is re-bought when they appear;
* their lines stay in the surrounding module's chunks, so search finds them exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import _definitions_win
from inflorescence.code_indexer.models import (
    DECLARATION_NODE_TYPES,
    NODE_TYPE_LABELS,
    CodeNode,
    EdgeType,
    NodeType,
)
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.go_parser import GoParser
from inflorescence.code_indexer.parser.javascript_parser import JavaScriptParser
from inflorescence.code_indexer.parser.rust_parser import RustParser
from inflorescence.code_indexer.summarizer import summarize_hierarchy, summary_input_hash
from inflorescence.db import queries
from inflorescence.rag.chunker import Chunker
from tests.test_code_indexer import CountingLLM


def _by_name(nodes: list[CodeNode]) -> dict[str, CodeNode]:
    return {node.name: node for node in nodes}


def test_typescript_declarations_carry_their_first_line_and_jsdoc(tmp_path: Path) -> None:
    (tmp_path / "engine.ts").write_text(
        "import { list } from './bank'\n"
        "/** Search tuning. */\n"
        "export const RETRY_POLICY = {\n"
        "  depth: 3,\n"
        "}\n"
        "const helper = (a: number) => a + 1\n"
        "export type OrderTotals = { ok: boolean }\n"
        "let a = 1, b = 2\n"
        "const { p, q } = list\n",
        encoding="utf-8",
    )

    nodes, edges = JavaScriptParser().parse_file(tmp_path / "engine.ts", tmp_path)
    named = _by_name(nodes)

    retry_policy = named["RETRY_POLICY"]
    assert retry_policy.node_type == NodeType.VARIABLE
    assert (retry_policy.line_start, retry_policy.line_end) == (3, 5)
    assert retry_policy.signature == "export const RETRY_POLICY = {"
    assert retry_policy.docstring == "Search tuning."
    assert named["OrderTotals"].node_type == NodeType.TYPE_ALIAS
    assert named["helper"].node_type == NodeType.FUNCTION  # an arrow function stays a function
    assert named["a"].signature == "let a = 1" and named["b"].signature == "let b = 2"
    assert "p" not in named and "list" not in named  # a destructuring pattern names no one binding
    contains = {e.target for e in edges if e.edge_type == EdgeType.CONTAINS}
    assert {"engine.ts::RETRY_POLICY", "engine.ts::OrderTotals"} <= contains


def test_go_values_and_named_types(tmp_path: Path) -> None:
    (tmp_path / "checker.go").write_text(
        "package content\n\n"
        "// Status of a job.\n"
        "type Status string\n\n"
        "// checkers maps a task type to its checker.\n"
        "var checkers = map[string]int{\n"
        '\t"parser_strict": 1,\n'
        "}\n\n"
        "const (\n"
        "\t// MaxRetries bounds retries.\n"
        "\tMaxRetries = 3\n"
        "\tminWait, maxWait = 1, 2\n"
        ")\n\n"
        "var _ = fmt.Sprintf\n",
        encoding="utf-8",
    )

    nodes, _ = GoParser().parse_file(tmp_path / "checker.go", tmp_path)
    named = _by_name(nodes)

    assert named["Status"].node_type == NodeType.TYPE_ALIAS
    assert named["Status"].docstring == "Status of a job."
    assert named["checkers"].signature == "var checkers = map[string]int{"
    assert (named["checkers"].line_start, named["checkers"].line_end) == (7, 9)
    assert named["MaxRetries"].docstring == "MaxRetries bounds retries."
    assert {"minWait", "maxWait"} <= named.keys()
    assert "_" not in named and "," not in named


def test_python_assignments_type_aliases_and_attribute_docstrings(tmp_path: Path) -> None:
    (tmp_path / "page_health.py").write_text(
        '"""Page health."""\n'
        "MIN_BATCH_SIZE = 400\n"
        '"""Below this many visible characters a page is treated as empty."""\n'
        "A, B = 1, 2\n"
        "Json: TypeAlias = dict\n"
        "obj.attr = 3\n"
        "foo = None\n"
        "def foo():\n"
        "    return 1\n",
        encoding="utf-8",
    )

    nodes, edges, _ = PythonAstParser().parse_file_with_facts(tmp_path / "page_health.py", tmp_path)
    nodes, edges = _definitions_win(nodes, edges)
    named = _by_name(nodes)

    assert named["MIN_BATCH_SIZE"].node_type == NodeType.VARIABLE
    assert named["MIN_BATCH_SIZE"].docstring.startswith("Below this many")
    assert {"A", "B"} <= named.keys()
    assert named["Json"].node_type == NodeType.TYPE_ALIAS
    assert "attr" not in named
    # `foo = None` above `def foo()`: one id, and the definition keeps it.
    assert [n.node_type for n in nodes if n.id == "page_health.py::foo"] == [NodeType.FUNCTION]
    contains = [e for e in edges if e.edge_type == EdgeType.CONTAINS and e.target == "page_health.py::foo"]
    assert len(contains) == 1


@pytest.mark.skipif(sys.version_info < (3, 12), reason="`type X = ...` is Python 3.12 syntax")
def test_python_type_statement(tmp_path: Path) -> None:
    (tmp_path / "pairs.py").write_text("type Pair = tuple[int, int]\n", encoding="utf-8")

    nodes, _ = PythonAstParser().parse_file(tmp_path / "pairs.py", tmp_path)

    assert _by_name(nodes)["Pair"].node_type == NodeType.TYPE_ALIAS


def test_rust_const_static_and_type_items(tmp_path: Path) -> None:
    (tmp_path / "lib.rs").write_text(
        "/// Max retries.\npub const MAX: u32 = 3;\nstatic NAME: &str = \"x\";\npub type Res<T> = Result<T, E>;\n",
        encoding="utf-8",
    )

    named = _by_name(RustParser().parse_file(tmp_path / "lib.rs", tmp_path)[0])

    assert named["MAX"].node_type == NodeType.VARIABLE and named["MAX"].docstring == "Max retries."
    assert named["NAME"].node_type == NodeType.VARIABLE
    assert named["Res"].node_type == NodeType.TYPE_ALIAS


@pytest.mark.asyncio
async def test_declarations_are_never_summarized_and_leave_their_module_hash_alone(tmp_path: Path) -> None:
    """The cost promise: adding declaration nodes re-buys no summary anywhere."""
    (tmp_path / "rules.py").write_text("RULES = (1, 2)\nLIMIT = 3\n\ndef apply():\n    return RULES\n")
    (tmp_path / "tasks.py").write_text("TASKS = ['a', 'b']\n")
    parser = PythonAstParser()
    rules_nodes, rules_edges = parser.parse_file(tmp_path / "rules.py", tmp_path)
    tasks_nodes, tasks_edges = parser.parse_file(tmp_path / "tasks.py", tmp_path)
    nodes, edges = rules_nodes + tasks_nodes, rules_edges + tasks_edges
    llm = CountingLLM()

    await summarize_hierarchy(nodes, edges, tmp_path, llm)

    declarations = [n for n in nodes if n.node_type in DECLARATION_NODE_TYPES]
    assert {n.name for n in declarations} == {"RULES", "LIMIT", "TASKS"}
    assert all(n.summary == "" and n.summary_input_hash == "" for n in declarations)
    # rules.py, apply, tasks.py — nothing else reached the LLM.
    assert llm.calls == 3
    named = {n.id: n for n in nodes}
    functions_only = [named["rules.py::apply"]]
    assert named["rules.py"].summary_input_hash == summary_input_hash(named["rules.py"], functions_only)
    # A module that declares nothing but constants is still summarized from its source.
    assert "TASKS" in "\n".join(llm.prompts)


def test_declaration_lines_stay_in_the_module_chunk(tmp_path: Path) -> None:
    (tmp_path / "retry_policy.ts").write_text(
        "export const RETRY_POLICY = {\n  depth: 3,\n  width: 5,\n}\n\nexport function run() {\n  return RETRY_POLICY\n}\n",
        encoding="utf-8",
    )
    nodes, _ = JavaScriptParser().parse_file(tmp_path / "retry_policy.ts", tmp_path)

    chunks = Chunker().chunk_repository(tmp_path, nodes)

    module_body = [c for c in chunks if c.chunk_kind == "module_body"]
    assert len(module_body) == 1
    assert module_body[0].node_id == "retry_policy.ts"
    assert "width: 5" in module_body[0].content
    module_context = next(c for c in chunks if c.chunk_kind == "module_context")
    assert "variable RETRY_POLICY lines 1-4" in module_context.content


def test_labels_are_derived_for_every_type_and_every_label_is_cleared_on_upsert() -> None:
    assert NODE_TYPE_LABELS[NodeType.TYPE_ALIAS.value] == "TypeAlias"
    assert NODE_TYPE_LABELS[NodeType.VARIABLE.value] == "Variable"
    assert set(NODE_TYPE_LABELS) == {t.value for t in NodeType}
    for template in (queries.UPSERT_NODE_TEMPLATE, queries.UPSERT_NODES_BATCH_TEMPLATE):
        for label in NODE_TYPE_LABELS.values():
            assert f"REMOVE n:{label}\n" in template
