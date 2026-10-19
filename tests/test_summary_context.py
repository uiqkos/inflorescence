"""Summary prompts say where an entity lives and what its module says about itself.

A field report (15.09.2026) found a math module summarized as "a cryptographic sum structure
using a signature path": its header comment said "tap on an operator sign", the function bodies
did not, and no prompt carried the header — or even the file's path. Both are prompt context only:
neither enters the summary input hash, so an index built before this change is not re-summarized
(and not re-billed) for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.javascript_parser import JavaScriptParser
from inflorescence.code_indexer.summarizer import (
    MODULE_HEADER_MAX_CHARS,
    module_header,
    summarize_hierarchy,
)
from tests.test_code_indexer import CountingLLM

BOARD_TS = """#!/usr/bin/env node
// Board engine: checks one formula line against the goal row.
// Everything here is pure; rendering lives in board-view.ts.

export function checkGoal(row: string) {
  return row.length > 0
}

export function readRow(line: string) {
  return line.trim()
}
"""


@pytest.mark.asyncio
async def test_prompts_carry_the_path_and_the_module_header(tmp_path: Path) -> None:
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine/board.ts").write_text(BOARD_TS, encoding="utf-8")
    nodes, edges = JavaScriptParser().parse_file(tmp_path / "engine/board.ts", tmp_path)
    llm = CountingLLM()

    await summarize_hierarchy(nodes, edges, tmp_path, llm)

    leaf = next(p for p in llm.prompts if "named 'checkGoal'" in p)
    assert "in the file 'engine/board.ts'" in leaf
    assert "// Board engine: checks one formula line against the goal row." in leaf
    module = next(p for p in llm.prompts if "based on its contents" in p)
    assert "in the file 'engine/board.ts'" in module
    assert "// Board engine: checks one formula line against the goal row." in module
    assert "rendering lives in board-view.ts" in module
    assert "#!/usr/bin/env node" not in module


@pytest.mark.asyncio
async def test_the_header_is_context_not_input(tmp_path: Path) -> None:
    """Changing only the header comment must not change any hash — no re-summarization."""
    path = tmp_path / "board.ts"
    path.write_text(BOARD_TS, encoding="utf-8")
    nodes, edges = JavaScriptParser().parse_file(path, tmp_path)
    await summarize_hierarchy(nodes, edges, tmp_path, CountingLLM())
    before = {n.id: n.summary_input_hash for n in nodes}

    path.write_text(BOARD_TS.replace("Board engine", "Board core"), encoding="utf-8")
    renamed, renamed_edges = JavaScriptParser().parse_file(path, tmp_path)
    stored = {n.id: {"summary": n.summary, "summary_input_hash": n.summary_input_hash} for n in nodes}
    llm = CountingLLM()
    await summarize_hierarchy(renamed, renamed_edges, tmp_path, llm, stored=stored)

    assert llm.calls == 0
    assert {n.id: n.summary_input_hash for n in renamed} == before


def test_python_modules_use_their_docstring_and_the_header_is_capped(tmp_path: Path) -> None:
    (tmp_path / "jobs.py").write_text('"""Nightly reconciliation retries."""\n\ndef run():\n    pass\n')
    nodes, _ = PythonAstParser().parse_file(tmp_path / "jobs.py", tmp_path)
    module = next(n for n in nodes if n.id == "jobs.py")
    assert module_header(module, tmp_path) == "Nightly reconciliation retries."

    (tmp_path / "long.ts").write_text("// " + "x" * (MODULE_HEADER_MAX_CHARS * 3) + "\nexport const A = 1\n")
    long_nodes, _ = JavaScriptParser().parse_file(tmp_path / "long.ts", tmp_path)
    long_module = next(n for n in long_nodes if n.id == "long.ts")
    assert len(module_header(long_module, tmp_path)) == MODULE_HEADER_MAX_CHARS
