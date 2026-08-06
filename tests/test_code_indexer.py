from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from inflorescence.code_indexer.graph_builder import GraphBuilder, _classify_changed_files
from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, FileChecksum, IndexerConfig, NodeType
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.llm_parser import parse_file_with_llm
from inflorescence.code_indexer.parser.resolver import resolve_edges
from inflorescence.config import DEFAULT_EXCLUDE_PATTERNS, Settings


class CountingLLM:
    """Fake LLMClient that records call count and returns prompt-derived text."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        self.prompts.append(prompt)
        digest = hashlib.md5(prompt.encode()).hexdigest()[:8]  # noqa: S324
        return f"summary::{digest}"


def test_python_parser_extracts_module_symbols_and_edges(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        '''
"""module docs"""

import os
from pathlib import Path


class Worker(BaseWorker):
    """worker docs"""

    def run(self, target: Path) -> str:
        return helper(str(target))


def helper(value: str) -> str:
    return os.fspath(value)
'''.lstrip(),
        encoding="utf-8",
    )

    nodes, edges = PythonAstParser().parse_file(source, tmp_path)

    by_id = {node.id: node for node in nodes}
    assert by_id["sample.py"].node_type == NodeType.MODULE
    assert by_id["sample.py::Worker"].node_type == NodeType.CLASS
    assert by_id["sample.py::Worker.run"].signature == "def run(self, target: Path) -> str"
    assert by_id["sample.py::helper"].node_type == NodeType.FUNCTION

    typed_edges = {(edge.source, edge.target, edge.edge_type) for edge in edges}
    assert ("sample.py", "os", EdgeType.IMPORTS) in typed_edges
    assert ("sample.py", "pathlib.Path", EdgeType.IMPORTS) in typed_edges
    assert ("sample.py::Worker", "BaseWorker", EdgeType.INHERITS) in typed_edges
    assert ("sample.py::Worker.run", "helper", EdgeType.CALLS) in typed_edges


def test_resolve_edges_prefers_same_file_for_ambiguous_call_targets() -> None:
    nodes = [
        CodeNode(id="pkg/a.py", name="a", node_type=NodeType.MODULE, file_path="pkg/a.py", line_start=1, line_end=1),
        CodeNode(id="pkg/a.py::handler", name="handler", node_type=NodeType.FUNCTION, file_path="pkg/a.py", line_start=1, line_end=2),
        CodeNode(id="pkg/b.py", name="b", node_type=NodeType.MODULE, file_path="pkg/b.py", line_start=1, line_end=1),
        CodeNode(id="pkg/b.py::handler", name="handler", node_type=NodeType.FUNCTION, file_path="pkg/b.py", line_start=1, line_end=2),
    ]
    edges = [Edge(source="pkg/b.py::handler", target="handler", edge_type=EdgeType.CALLS)]

    result = resolve_edges(nodes, edges)

    assert [(e.source, e.target, e.resolution) for e in result.edges] == [
        ("pkg/b.py::handler", "pkg/b.py::handler", "same-module")
    ]


def test_resolve_edges_maps_python_imports_to_module_nodes() -> None:
    nodes = [
        CodeNode(id="src/app/main.py", name="main", node_type=NodeType.MODULE, file_path="src/app/main.py", line_start=1, line_end=1),
        CodeNode(id="src/app/settings.py", name="settings", node_type=NodeType.MODULE, file_path="src/app/settings.py", line_start=1, line_end=1),
    ]
    edges = [Edge(source="src/app/main.py", target="app.settings", edge_type=EdgeType.IMPORTS)]

    resolved = resolve_edges(nodes, edges).edges

    assert resolved == [Edge(source="src/app/main.py", target="src/app/settings.py", edge_type=EdgeType.IMPORTS)]


def test_resolve_edges_maps_python_from_import_to_exported_symbol() -> None:
    nodes = [
        CodeNode(id="cerberus/__init__.py", name="__init__", node_type=NodeType.MODULE, file_path="cerberus/__init__.py", line_start=1, line_end=1),
        CodeNode(id="cerberus/validator.py", name="validator", node_type=NodeType.MODULE, file_path="cerberus/validator.py", line_start=1, line_end=1),
        CodeNode(id="cerberus/validator.py::BareValidator", name="BareValidator", node_type=NodeType.CLASS, file_path="cerberus/validator.py", line_start=1, line_end=10),
        CodeNode(id="consumer.py", name="consumer", node_type=NodeType.MODULE, file_path="consumer.py", line_start=1, line_end=1),
    ]
    edges = [Edge(source="consumer.py", target="cerberus.validator.BareValidator", edge_type=EdgeType.IMPORTS)]

    resolved = resolve_edges(nodes, edges).edges

    assert resolved == [Edge(source="consumer.py", target="cerberus/validator.py::BareValidator", edge_type=EdgeType.IMPORTS)]


def test_resolve_edges_maps_python_relative_import_to_module_symbol() -> None:
    nodes = [
        CodeNode(id="pkg/a.py", name="a", node_type=NodeType.MODULE, file_path="pkg/a.py", line_start=1, line_end=1),
        CodeNode(id="pkg/b.py", name="b", node_type=NodeType.MODULE, file_path="pkg/b.py", line_start=1, line_end=1),
        CodeNode(id="pkg/b.py::helper", name="helper", node_type=NodeType.FUNCTION, file_path="pkg/b.py", line_start=1, line_end=2),
    ]
    edges = [Edge(source="pkg/a.py", target=".b.helper", edge_type=EdgeType.IMPORTS)]

    resolved = resolve_edges(nodes, edges).edges

    assert resolved == [Edge(source="pkg/a.py", target="pkg/b.py::helper", edge_type=EdgeType.IMPORTS)]


def test_resolve_edges_resolves_js_relative_import() -> None:
    nodes = [
        CodeNode(id="src/app.js", name="app", node_type=NodeType.MODULE, file_path="src/app.js", line_start=1, line_end=1),
        CodeNode(id="src/utils.js", name="utils", node_type=NodeType.MODULE, file_path="src/utils.js", line_start=1, line_end=1),
    ]
    edges = [Edge(source="src/app.js", target="./utils", edge_type=EdgeType.IMPORTS)]

    resolved = resolve_edges(nodes, edges).edges

    assert resolved == [Edge(source="src/app.js", target="src/utils.js", edge_type=EdgeType.IMPORTS)]


def test_resolve_edges_resolves_go_package_import() -> None:
    nodes = [
        CodeNode(id="cmd/main.go", name="main", node_type=NodeType.MODULE, file_path="cmd/main.go", line_start=1, line_end=1),
        CodeNode(id="pkg/store/store.go", name="store", node_type=NodeType.MODULE, file_path="pkg/store/store.go", line_start=1, line_end=1),
    ]
    edges = [Edge(source="cmd/main.go", target="pkg/store", edge_type=EdgeType.IMPORTS)]

    resolved = resolve_edges(nodes, edges).edges

    assert resolved == [Edge(source="cmd/main.go", target="pkg/store/store.go", edge_type=EdgeType.IMPORTS)]


def test_resolve_edges_resolves_rust_crate_path_import() -> None:
    nodes = [
        CodeNode(id="src/main.rs", name="main", node_type=NodeType.MODULE, file_path="src/main.rs", line_start=1, line_end=1),
        CodeNode(id="src/config.rs", name="config", node_type=NodeType.MODULE, file_path="src/config.rs", line_start=1, line_end=1),
    ]
    edges = [Edge(source="src/main.rs", target="crate::config", edge_type=EdgeType.IMPORTS)]

    resolved = resolve_edges(nodes, edges).edges

    assert resolved == [Edge(source="src/main.rs", target="src/config.rs", edge_type=EdgeType.IMPORTS)]


def test_resolve_edges_resolves_python_relative_parent_import() -> None:
    nodes = [
        CodeNode(id="pkg/sub/a.py", name="a", node_type=NodeType.MODULE, file_path="pkg/sub/a.py", line_start=1, line_end=1),
        CodeNode(id="pkg/config.py", name="config", node_type=NodeType.MODULE, file_path="pkg/config.py", line_start=1, line_end=1),
    ]
    edges = [Edge(source="pkg/sub/a.py", target="..config", edge_type=EdgeType.IMPORTS)]

    resolved = resolve_edges(nodes, edges).edges

    assert resolved == [Edge(source="pkg/sub/a.py", target="pkg/config.py", edge_type=EdgeType.IMPORTS)]


def test_resolve_edges_records_ambiguous_and_unknown_calls_instead_of_guessing() -> None:
    """A cross-file bare name is ambiguous by construction — the old resolver picked
    candidates[0] (31% of the benchmark repository's CALLS edges were wrong); the ladder records the
    call on the caller node instead of guessing an edge."""
    nodes = [
        CodeNode(id="pkg/a.py", name="a", node_type=NodeType.MODULE, file_path="pkg/a.py", line_start=1, line_end=1),
        CodeNode(id="pkg/a.py::helper", name="helper", node_type=NodeType.FUNCTION, file_path="pkg/a.py", line_start=1, line_end=2),
        CodeNode(id="pkg/b.py::helper", name="helper", node_type=NodeType.FUNCTION, file_path="pkg/b.py", line_start=1, line_end=2),
        CodeNode(id="pkg/c.py::caller", name="caller", node_type=NodeType.FUNCTION, file_path="pkg/c.py", line_start=1, line_end=2),
    ]
    edges = [
        Edge(source="pkg/c.py::caller", target="helper", edge_type=EdgeType.CALLS),
        Edge(source="pkg/c.py::caller", target="nonexistent", edge_type=EdgeType.CALLS),
    ]

    result = resolve_edges(nodes, edges)

    assert [e for e in result.edges if e.edge_type == EdgeType.CALLS] == []
    assert result.unresolved_calls == {"pkg/c.py::caller": ["helper", "nonexistent"]}


def test_chunker_creates_function_and_context_chunks(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        '''
"""module docs"""

class Worker(Base):
    """worker docs"""

    mode = "fast"

    def run(self):
        value = 1
        return value


def helper():
    return "ok"
'''.lstrip(),
        encoding="utf-8",
    )
    nodes, _ = PythonAstParser().parse_file(source, tmp_path)

    from inflorescence.rag.chunker import Chunker

    chunks = Chunker().chunk_repository(tmp_path, nodes)
    by_kind = {(chunk.node_id, chunk.chunk_kind) for chunk in chunks}

    assert ("sample.py", "module_context") in by_kind
    assert ("sample.py::Worker", "class_context") in by_kind
    assert ("sample.py::Worker.run", "function_body") in by_kind
    assert ("sample.py::helper", "function_body") in by_kind
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.start_line <= chunk.end_line for chunk in chunks)


def test_python_parser_disambiguates_property_setters(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        """
class Settings:
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value
""".lstrip(),
        encoding="utf-8",
    )

    nodes, _ = PythonAstParser().parse_file(source, tmp_path)

    node_ids = [node.id for node in nodes]
    assert len(node_ids) == len(set(node_ids))
    assert "sample.py::Settings.value" in node_ids
    assert "sample.py::Settings.value.setter" in node_ids


@pytest.mark.asyncio
async def test_graph_builder_scans_excludes_and_stores_without_llm(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "module.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "ignored.py").write_text("def ignored():\n    pass\n", encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False),
    )

    nodes, edges = await builder.build("fixture", tmp_path)

    node_ids = {node.id for node in nodes}
    assert "pkg/module.py" in node_ids
    assert "pkg/module.py::alpha" in node_ids
    assert "dir:." in node_ids
    assert "dir:pkg" in node_ids
    assert all("__pycache__" not in node.file_path for node in nodes)
    assert any(edge.edge_type == EdgeType.CONTAINS and edge.source == "dir:pkg" for edge in edges)

    assert repo.upserted_nodes == nodes
    assert repo.upserted_edges == edges
    assert repo.checksums == {"pkg/__init__.py", "pkg/module.py"}


@pytest.mark.asyncio
async def test_graph_builder_scans_with_include_and_exclude_filters(tmp_path: Path) -> None:
    (tmp_path / "backend" / "pkg").mkdir(parents=True)
    (tmp_path / "frontend").mkdir()
    (tmp_path / "backend" / ".worktrees" / "copy").mkdir(parents=True)
    (tmp_path / "backend" / "pkg" / "service.py").write_text("def service():\n    return 1\n", encoding="utf-8")
    (tmp_path / "frontend" / "app.py").write_text("def app():\n    return 2\n", encoding="utf-8")
    (tmp_path / "backend" / ".worktrees" / "copy" / "service.py").write_text(
        "def copied():\n    return 3\n",
        encoding="utf-8",
    )

    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(
            include_patterns=["backend/**"],
            exclude_patterns=[".worktrees"],
            use_llm_summaries=False,
            use_llm_fallback_parser=False,
        ),
    )

    nodes, _ = await builder.build("fixture", tmp_path)

    module_paths = {node.file_path for node in nodes if node.node_type == NodeType.MODULE}
    assert module_paths == {"backend/pkg/service.py"}
    assert repo.checksums == {"backend/pkg/service.py"}


@pytest.mark.asyncio
async def test_graph_builder_update_rebuilds_project_to_keep_cross_file_edges(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("import b\n\ndef call():\n    return b.target()\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def target():\n    return 1\n", encoding="utf-8")

    repo = RebuildRepository({"a.py": "old-checksum", "b.py": "current"})
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False),
    )

    nodes, edges = await builder.update("fixture", tmp_path)

    # Reconcile, not wipe: the project is never deleted and edges are MERGEd in place —
    # the destructive wipe-all (delete_code_edges) is never used, so the graph never
    # passes through an edgeless window (audit finding 9). Cross-file relations stay
    # consistent because the fresh edges are upserted, not because old ones were dropped.
    assert repo.deleted_projects == []
    assert repo.deleted_code_edges == 0
    assert {node.id for node in nodes} >= {"a.py", "b.py", "a.py::call", "b.py::target"}
    assert Edge(source="a.py", target="b.py", edge_type=EdgeType.IMPORTS) in edges


@pytest.mark.asyncio
async def test_graph_builder_update_removes_only_stale_nodes_when_indexed_file_was_deleted(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    repo = RebuildRepository(
        {"a.py": FileChecksum.compute(tmp_path / "a.py", tmp_path).md5, "deleted.py": "old"},
        node_ids={"a.py", "a.py::keep", "deleted.py", "deleted.py::gone"},
    )
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False),
    )

    nodes, _ = await builder.update("fixture", tmp_path)

    # The dead file's nodes are deleted individually; the project survives intact.
    assert repo.deleted_projects == []
    assert set(repo.deleted_node_ids) == {"deleted.py", "deleted.py::gone"}
    assert "deleted.py" not in {node.id for node in nodes}


def test_classify_changed_files_normalizes_absolute_and_relative(tmp_path: Path) -> None:
    # The file watcher emits absolute paths; direct callers/tests may pass relative
    # ones. Both must map to the same root-relative key the scan uses, so a modified
    # file lands in `changed`, not silently in `deleted`.
    root = tmp_path.resolve()
    (root / "keep.py").write_text("x = 1\n", encoding="utf-8")
    current = {"keep.py"}

    indexed = {"keep.py", "gone.py"}

    changed, deleted = _classify_changed_files([str(root / "keep.py")], current, root, indexed)
    assert [p.name for p in changed] == ["keep.py"] and deleted == []

    changed, deleted = _classify_changed_files(["keep.py"], current, root, indexed)
    assert [p.name for p in changed] == ["keep.py"] and deleted == []

    changed, deleted = _classify_changed_files([str(root / "gone.py")], current, root, indexed)
    assert changed == [] and deleted == ["gone.py"]

    # A path the index has never seen (ignored/generated/temp file) is dropped —
    # it must not masquerade as a deletion and trigger a project rebuild.
    changed, deleted = _classify_changed_files([str(root / "scratch.tmp.py")], current, root, indexed)
    assert changed == [] and deleted == []

    outside = root.parent / "outside.py"
    changed, deleted = _classify_changed_files([str(outside)], current, root, indexed)
    assert changed == [] and deleted == []


@pytest.mark.asyncio
async def test_graph_builder_update_treats_watcher_absolute_path_as_change(tmp_path: Path) -> None:
    # End-to-end: report a modification the way the watcher does (absolute path) and
    # confirm the newly added function is present in the rebuilt graph.
    src = tmp_path / "a.py"
    src.write_text("def keep():\n    return 1\n", encoding="utf-8")
    repo = RebuildRepository({"a.py": "stale-checksum"})
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False),
    )

    src.write_text("def keep():\n    return 1\n\ndef added():\n    return 2\n", encoding="utf-8")
    nodes, _ = await builder.update("fixture", tmp_path, changed_files=[str(src.resolve())])

    ids = {node.id for node in nodes}
    assert {"a.py::keep", "a.py::added"} <= ids
    assert repo.deleted_projects == []


class RecordingRepository:
    def __init__(self) -> None:
        self.upserted_nodes = []
        self.upserted_edges = []
        self.checksums: set[str] = set()
        self.deleted_node_ids: list[str] = []
        self.deleted_code_edges = 0
        self.deleted_edge_keys: list[dict] = []
        self.summaries_stored: list[dict] = []
        self.upserted_external: list = []
        self.calls: list[str] = []

    async def upsert_external_nodes(self, project: str, refs: list) -> int:
        self.calls.append("upsert_external_nodes")
        self.upserted_external.extend(refs)
        return len(refs)

    async def delete_orphan_external_nodes(self, project: str) -> int:
        self.calls.append("delete_orphan_external_nodes")
        return 0

    async def upsert_nodes(self, project: str, nodes: list) -> int:
        self.calls.append("upsert_nodes")
        self.upserted_nodes = nodes
        return len(nodes)

    async def upsert_edges(self, project: str, edges: list) -> int:
        self.calls.append("upsert_edges")
        self.upserted_edges = edges
        return len(edges)

    async def store_summaries(self, project: str, items: list) -> int:
        self.calls.append("store_summaries")
        self.summaries_stored.extend(items)
        return len(items)

    async def set_project_root_path(self, project: str, root_path: str) -> None:
        self.calls.append("set_project_root_path")

    async def update_checksums(self, project: str, items: list) -> int:
        self.calls.append("update_checksums")
        for item in items:
            self.checksums.add(item["node_id"])
        return len(items)

    async def get_stored_summaries(self, project: str) -> dict[str, dict[str, object]]:
        return {}

    async def get_node_ids(self, project: str) -> set[str]:
        return set()

    async def get_stored_node_index(self, project: str) -> list[tuple[str, str, str]]:
        return []

    async def get_code_edges(self, project: str) -> list[tuple[str, str, str]]:
        return []

    async def delete_nodes_by_ids(self, project: str, ids: list) -> int:
        self.deleted_node_ids.extend(ids)
        return len(ids)

    async def delete_code_edges(self, project: str) -> None:
        self.deleted_code_edges += 1

    async def delete_edges_by_keys(self, project: str, edges: list) -> int:
        self.deleted_edge_keys.extend(edges)
        return len(edges)


class RebuildRepository(RecordingRepository):
    def __init__(
        self,
        checksums: dict[str, str],
        stored: dict[str, dict[str, object]] | None = None,
        node_ids: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._checksums = checksums
        self._stored = stored or {}
        self._node_ids = node_ids or set()
        self.deleted_projects: list[str] = []

    async def get_checksums(self, project: str) -> dict[str, str]:
        return self._checksums

    async def get_stored_summaries(self, project: str) -> dict[str, dict[str, object]]:
        return self._stored

    async def get_node_ids(self, project: str) -> set[str]:
        return set(self._node_ids)

    async def get_stored_node_index(self, project: str) -> list[tuple[str, str, str]]:
        # Synthesize (id, file_path, node_type) from the bare stored ids: "dir:foo" is
        # a directory, "path::name" a sub-node of "path", and a bare "path" a module.
        index: list[tuple[str, str, str]] = []
        for nid in self._node_ids:
            if nid.startswith("dir:"):
                index.append((nid, nid[len("dir:"):], "directory"))
            elif "::" in nid:
                index.append((nid, nid.split("::", 1)[0], "function"))
            else:
                index.append((nid, nid, "module"))
        return index

    async def delete_project(self, project: str) -> None:
        self.deleted_projects.append(project)


@pytest.mark.asyncio
async def test_llm_fallback_parser_awaits_async_llm_service(tmp_path: Path) -> None:
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n", encoding="utf-8")

    nodes, edges = await parse_file_with_llm(source, tmp_path, AsyncLLM())

    assert [node.id for node in nodes] == ["broken.py", "broken.py::Recovered"]
    assert nodes[1].node_type == NodeType.FUNCTION
    assert edges[0].edge_type == EdgeType.CONTAINS


class AsyncLLM:
    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        assert "def broken" in prompt
        assert "Return ONLY valid JSON" in system_prompt
        return """
        {
          "nodes": [
            {"name": "broken", "type": "module", "line_start": 1, "line_end": 1,
             "signature": "", "docstring": "", "parent": null},
            {"name": "Recovered", "type": "function", "line_start": 1, "line_end": 1,
             "signature": "def Recovered()", "docstring": "", "parent": null}
          ],
          "edges": [
            {"source": "broken.py", "target": "broken.py::Recovered", "type": "contains"}
          ]
        }
        """


class FallbackCountingLLM:
    """Records how many times the LLM fallback parser is invoked."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        return (
            '{"nodes":[{"name":"broken","type":"module","line_start":1,"line_end":1,'
            '"signature":"","docstring":"","parent":null}],"edges":[]}'
        )


@pytest.mark.asyncio
async def test_llm_fallback_parse_is_cached_by_checksum(tmp_path: Path) -> None:
    # INV-5/finding 7: a broken .py takes the paid LLM fallback path on every full build. Since a
    # watcher update re-parses ALL files, an unchanged broken file was re-paid on every flush.
    # It must now cost one call per content version, not one per flush.
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    llm = FallbackCountingLLM()
    builder = GraphBuilder(repo=None, llm=llm, config=IndexerConfig(use_llm_summaries=False))

    n1, _, _ = await builder._parse_single_file(broken, tmp_path)
    n2, _, _ = await builder._parse_single_file(broken, tmp_path)   # unchanged -> cache hit
    assert llm.calls == 1
    assert [n.id for n in n1] == [n.id for n in n2] == ["broken.py"]

    # The cache hands out fresh copies: mutating a returned node must not pollute it.
    n1[0].summary = "mutated"
    n3, _, _ = await builder._parse_single_file(broken, tmp_path)
    assert n3[0].summary != "mutated"
    assert llm.calls == 1

    # Changed content -> new checksum -> re-parsed exactly once more (not reused stale).
    broken.write_text("def broken(:\n  pass  # changed\n", encoding="utf-8")
    await builder._parse_single_file(broken, tmp_path)
    assert llm.calls == 2


def test_exclude_list_is_single_broadened_source() -> None:
    for pattern in [
        ".beads", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "target", "vendor", ".idea", ".vscode", "out", ".cache",
        "node_modules", ".git", ".venv", ".worktrees",
    ]:
        assert pattern in DEFAULT_EXCLUDE_PATTERNS
    assert Settings(_env_file=None).exclude_patterns == DEFAULT_EXCLUDE_PATTERNS
    assert IndexerConfig().exclude_patterns == DEFAULT_EXCLUDE_PATTERNS
    # Distinct list instances so mutating one config never leaks into another.
    assert Settings(_env_file=None).exclude_patterns is not IndexerConfig().exclude_patterns


def test_new_cost_guard_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.respect_gitignore is True
    assert settings.max_file_size_bytes == 262144
    assert settings.use_llm_summaries is True
    config = IndexerConfig()
    assert config.respect_gitignore is True
    assert config.max_file_size_bytes == 262144


@pytest.mark.asyncio
async def test_scan_respects_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\nbuilt/\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def gone():\n    return 1\n", encoding="utf-8")
    (tmp_path / "built").mkdir()
    (tmp_path / "built" / "gen.py").write_text("def gen():\n    return 1\n", encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(repo=repo, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False))
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"keep.py"}


@pytest.mark.asyncio
async def test_scan_gitignore_can_be_disabled(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("def still():\n    return 1\n", encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(respect_gitignore=False, use_llm_summaries=False, use_llm_fallback_parser=False),
    )
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"keep.py", "ignored.py"}


@pytest.mark.asyncio
async def test_scan_skips_files_over_size_limit(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("def small():\n    return 1\n", encoding="utf-8")
    (tmp_path / "big.py").write_text("x = 1\n" + ("# pad\n" * 5000), encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(max_file_size_bytes=200, use_llm_summaries=False, use_llm_fallback_parser=False),
    )
    result = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in result.nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"small.py"}
    # Skipped-large is per-run state on the returned result, not a shared instance field.
    assert result.skipped_large_files == ["big.py"]  # skip is surfaced, not just logged


@pytest.mark.asyncio
async def test_scan_prunes_nested_git_worktree(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
    nested = tmp_path / "checkout"
    nested.mkdir()
    (nested / ".git").write_text("gitdir: /somewhere/.git/worktrees/checkout\n", encoding="utf-8")
    (nested / "copy.py").write_text("def copy():\n    return 1\n", encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(repo=repo, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False))
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"app.py"}


@pytest.mark.asyncio
async def test_scan_indexes_nested_independent_repo_and_submodule(tmp_path: Path) -> None:
    """A nested repo (`.git` dir) or submodule (`.git` file → modules/) is NOT a worktree — index it."""
    (tmp_path / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")

    subrepo = tmp_path / "backend"
    subrepo.mkdir()
    (subrepo / ".git").mkdir()  # independent nested repository
    (subrepo / "svc.py").write_text("def svc():\n    return 1\n", encoding="utf-8")

    submodule = tmp_path / "vendored"
    submodule.mkdir()
    (submodule / ".git").write_text("gitdir: ../.git/modules/vendored\n", encoding="utf-8")
    (submodule / "mod.py").write_text("def mod():\n    return 1\n", encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(repo=repo, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False))
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"app.py", "backend/svc.py", "vendored/mod.py"}


@pytest.mark.asyncio
async def test_scan_skips_generated_go_by_marker(tmp_path: Path) -> None:
    (tmp_path / "keep.go").write_text("package x\nfunc F(){}\n", encoding="utf-8")
    (tmp_path / "gen.go").write_text(
        "// Code generated by protoc-gen-go. DO NOT EDIT.\npackage x\nfunc G(){}\n", encoding="utf-8"
    )

    repo = RecordingRepository()
    builder = GraphBuilder(repo=repo, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False))
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"keep.go"}


@pytest.mark.asyncio
async def test_scan_skips_grpc_stub_by_filename(tmp_path: Path) -> None:
    (tmp_path / "app.js").write_text("function foo(){ return 1 }\n", encoding="utf-8")
    (tmp_path / "service_grpc_pb.js").write_text("function foo(){ return 1 }\n", encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(repo=repo, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False))
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"app.js"}


@pytest.mark.asyncio
async def test_scan_generated_can_be_disabled(tmp_path: Path) -> None:
    (tmp_path / "keep.go").write_text("package x\nfunc F(){}\n", encoding="utf-8")
    (tmp_path / "gen.go").write_text(
        "// Code generated by protoc-gen-go. DO NOT EDIT.\npackage x\nfunc G(){}\n", encoding="utf-8"
    )

    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(skip_generated=False, use_llm_summaries=False, use_llm_fallback_parser=False),
    )
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"keep.go", "gen.go"}


@pytest.mark.asyncio
async def test_scan_respects_inflorescenceignore_even_without_gitignore(tmp_path: Path) -> None:
    (tmp_path / ".inflorescenceignore").write_text("secret.py\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (tmp_path / "secret.py").write_text("def secret():\n    return 1\n", encoding="utf-8")

    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo,
        llm=None,
        config=IndexerConfig(respect_gitignore=False, use_llm_summaries=False, use_llm_fallback_parser=False),
    )
    nodes, _ = await builder.build("fixture", tmp_path)
    module_paths = {n.file_path for n in nodes if n.node_type == NodeType.MODULE}

    assert module_paths == {"keep.py"}


def test_chunker_skips_file_over_size_limit(tmp_path: Path) -> None:
    from inflorescence.rag.chunker import Chunker
    from inflorescence.rag.config import RAGConfig

    big = tmp_path / "big.py"
    big.write_text("x = 1\n" + ("# pad\n" * 5000), encoding="utf-8")

    chunker = Chunker(config=RAGConfig(max_file_size_bytes=200))

    assert chunker.chunk_file(big, tmp_path) == []


def test_chunk_repository_skips_oversize_files(tmp_path: Path) -> None:
    from inflorescence.rag.chunker import Chunker
    from inflorescence.rag.config import RAGConfig

    big = tmp_path / "big.py"
    big.write_text("def big():\n    return 1\n" + ("# pad\n" * 5000), encoding="utf-8")
    node = CodeNode(id="big.py", name="big", node_type=NodeType.MODULE, file_path="big.py", line_start=1, line_end=2)

    chunker = Chunker(config=RAGConfig(max_file_size_bytes=200))

    assert chunker.chunk_repository(tmp_path, [node]) == []


def test_indexer_config_from_settings_threads_every_wired_setting(monkeypatch) -> None:
    """One factory, so a setting cannot reach two construction sites and miss the third.

    `use_scip_semantic` and `scip_timeout_seconds` lived on IndexerConfig but were never
    populated from Settings at any of the three sites, so the SCIP pass was permanently on with
    no supported way to turn it off — while the README advertised it as a feature.
    """
    monkeypatch.setenv("USE_SCIP_SEMANTIC", "false")
    monkeypatch.setenv("SCIP_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("MAX_FILE_SIZE_BYTES", "4096")
    monkeypatch.setenv("USE_LLM_SUMMARIES", "false")

    config = IndexerConfig.from_settings(Settings())

    assert config.use_scip_semantic is False
    assert config.scip_timeout_seconds == 120
    assert config.max_file_size_bytes == 4096
    assert config.use_llm_summaries is False


def test_ast_chunking_is_actually_reachable_for_every_supported_language() -> None:
    """The tree-sitter grammars CodeSplitter needs must be installed, not merely hoped for.

    `_chunk_code` wraps CodeSplitter in a try/except. The grammar package it requires was
    never declared, so the ImportError was swallowed and *every* code file in the index was
    chunked as prose — the semantic-search value proposition, silently absent, at DEBUG level.
    Assert the dependency contract directly: a missing grammar must fail this test rather than
    degrade the product in production.
    """
    from llama_index.core.node_parser import CodeSplitter
    from llama_index.core.schema import Document

    from inflorescence.rag.chunker import _EXT_TO_LANGUAGE

    sources = {
        "python": "def a():\n    return 1\n",
        "go": "package m\n\nfunc A() int { return 1 }\n",
        "typescript": "export function a(): number { return 1 }\n",
        "javascript": "export function a() { return 1 }\n",
        "rust": "pub fn a() -> i32 { 1 }\n",
    }
    for language, source in sources.items():
        assert language in _EXT_TO_LANGUAGE.values(), language
        splitter = CodeSplitter(language=language, chunk_lines=25, chunk_lines_overlap=5, max_chars=3000)
        assert splitter.get_nodes_from_documents([Document(text=source)]), language


def test_chunker_uses_ast_split_not_prose_split_for_code(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A real .py file must go through the AST path, with no fallback warning logged."""
    import logging

    from inflorescence.rag.chunker import Chunker

    source = tmp_path / "mod.py"
    source.write_text(
        "\n\n".join(f"def f{i}():\n    return {i}" for i in range(40)) + "\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="inflorescence.rag.chunker"):
        chunks = Chunker().chunk_file(source, tmp_path)

    assert chunks
    assert all("falling back to prose chunking" not in record.message for record in caplog.records)
    # AST splitting respects definition boundaries; prose splitting does not.
    assert any(chunk.content.lstrip().startswith("def f") for chunk in chunks)


def _leaf(node_id: str = "a.py::foo", signature: str = "def foo()", docstring: str = "") -> CodeNode:
    return CodeNode(
        id=node_id, name="foo", node_type=NodeType.FUNCTION,
        file_path="a.py", line_start=1, line_end=2, signature=signature, docstring=docstring,
    )


def test_leaf_hash_changes_with_source_signature_or_docstring() -> None:
    from inflorescence.code_indexer.summarizer import summary_input_hash

    base = summary_input_hash(_leaf(), [], source="return 1")
    assert summary_input_hash(_leaf(), [], source="return 1") == base  # stable
    assert summary_input_hash(_leaf(), [], source="return 2") != base  # body edit
    assert summary_input_hash(_leaf(signature="def foo(x)"), [], source="return 1") != base
    assert summary_input_hash(_leaf(docstring="docs"), [], source="return 1") != base


def test_container_hash_changes_with_child_summary_order_or_membership() -> None:
    from inflorescence.code_indexer.summarizer import summary_input_hash

    parent = CodeNode(id="a.py", name="a", node_type=NodeType.MODULE, file_path="a.py", line_start=1, line_end=9)
    c1 = _leaf("a.py::foo")
    c1.summary = "does foo"
    c2 = _leaf("a.py::bar")
    c2.summary = "does bar"

    base = summary_input_hash(parent, [c1, c2])
    assert summary_input_hash(parent, [c1, c2]) == base           # stable
    assert summary_input_hash(parent, [c2, c1]) != base           # order matters
    assert summary_input_hash(parent, [c1]) != base               # removed sibling
    c1_changed = _leaf("a.py::foo")
    c1_changed.summary = "does foo differently"
    assert summary_input_hash(parent, [c1_changed, c2]) != base   # child summary changed


def _two_module_tree(tmp_path: Path) -> tuple[list[CodeNode], list[Edge]]:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")
    nodes = [
        CodeNode(id="dir:.", name=".", node_type=NodeType.DIRECTORY, file_path=".", line_start=0, line_end=0),
        CodeNode(id="a.py", name="a", node_type=NodeType.MODULE, file_path="a.py", line_start=1, line_end=2, parent_id="dir:."),
        CodeNode(id="a.py::foo", name="foo", node_type=NodeType.FUNCTION, file_path="a.py", line_start=1, line_end=2, signature="def foo()", parent_id="a.py"),
        CodeNode(id="b.py", name="b", node_type=NodeType.MODULE, file_path="b.py", line_start=1, line_end=2, parent_id="dir:."),
        CodeNode(id="b.py::bar", name="bar", node_type=NodeType.FUNCTION, file_path="b.py", line_start=1, line_end=2, signature="def bar()", parent_id="b.py"),
    ]
    edges = [
        Edge(source="dir:.", target="a.py", edge_type=EdgeType.CONTAINS),
        Edge(source="dir:.", target="b.py", edge_type=EdgeType.CONTAINS),
        Edge(source="a.py", target="a.py::foo", edge_type=EdgeType.CONTAINS),
        Edge(source="b.py", target="b.py::bar", edge_type=EdgeType.CONTAINS),
    ]
    return nodes, edges


@pytest.mark.asyncio
async def test_summarize_hierarchy_without_stored_summarizes_every_node(tmp_path: Path) -> None:
    from inflorescence.code_indexer.summarizer import summarize_hierarchy

    nodes, edges = _two_module_tree(tmp_path)
    llm = CountingLLM()

    dirty = await summarize_hierarchy(nodes, edges, tmp_path, llm)

    assert dirty == {n.id for n in nodes}
    assert llm.calls == len(nodes)
    assert all(n.summary and n.summary_input_hash for n in nodes)


class RecordingProgress:
    def __init__(self) -> None:
        self.updates: list[tuple[str, int, int]] = []

    def update(self, phase: str, current: int, total: int) -> None:
        self.updates.append((phase, current, total))


@pytest.mark.asyncio
async def test_summarize_hierarchy_reports_progress(tmp_path: Path) -> None:
    from inflorescence.code_indexer.summarizer import summarize_hierarchy

    nodes, edges = _two_module_tree(tmp_path)
    prog = RecordingProgress()

    await summarize_hierarchy(nodes, edges, tmp_path, CountingLLM(), progress=prog)

    summarize = [u for u in prog.updates if u[0] == "summarize"]
    assert summarize, "expected summarize-phase progress updates"
    assert summarize[-1] == ("summarize", len(nodes), len(nodes))  # reaches total
    currents = [c for _phase, c, _total in summarize]
    assert currents == sorted(currents)  # monotonic non-decreasing


@pytest.mark.asyncio
async def test_build_reports_parse_and_store_progress(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")
    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo, llm=None, config=IndexerConfig(use_llm_summaries=False, use_llm_fallback_parser=False)
    )
    prog = RecordingProgress()

    nodes, _edges = await builder.build("fixture", tmp_path, progress=prog)

    phases = {u[0] for u in prog.updates}
    assert "parse" in phases and "store" in phases
    parse = [u for u in prog.updates if u[0] == "parse"]
    assert parse[-1][1] == parse[-1][2]  # parse reached total files
    store = [u for u in prog.updates if u[0] == "store"]
    assert store[-1][1] == store[-1][2] == len(nodes) + len(_edges)  # store reached node+edge total


@pytest.mark.asyncio
async def test_summarize_hierarchy_reuses_all_when_nothing_changed(tmp_path: Path) -> None:
    from inflorescence.code_indexer.summarizer import summarize_hierarchy

    nodes, edges = _two_module_tree(tmp_path)
    llm = CountingLLM()
    await summarize_hierarchy(nodes, edges, tmp_path, llm)
    stored = {n.id: {"summary": n.summary, "summary_input_hash": n.summary_input_hash, "summary_embedding": None} for n in nodes}

    fresh, edges2 = _two_module_tree(tmp_path)
    llm2 = CountingLLM()
    dirty = await summarize_hierarchy(fresh, edges2, tmp_path, llm2, stored=stored)

    assert dirty == set()
    assert llm2.calls == 0
    assert {n.id: n.summary for n in fresh} == {k: v["summary"] for k, v in stored.items()}


class FailingLLM:
    """Fake LLMClient whose every call fails — models a keyless/misconfigured server."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.calls += 1
        raise RuntimeError("401 no key")


@pytest.mark.asyncio
async def test_summarize_failure_keeps_stale_summary_and_marks_for_retry(tmp_path: Path) -> None:
    """LLM failure must degrade to stale data, not destroy it: the previously stored
    summary is kept, the input hash is cleared so the next run retries, and the node is
    not re-embedded (its stored embedding still matches the kept summary)."""
    from inflorescence.code_indexer.summarizer import summarize_hierarchy

    nodes, edges = _two_module_tree(tmp_path)
    await summarize_hierarchy(nodes, edges, tmp_path, CountingLLM())
    stored = {
        n.id: {"summary": n.summary, "summary_input_hash": "different-hash", "summary_embedding": None}
        for n in nodes
    }
    old_summaries = {n.id: n.summary for n in nodes}

    fresh, edges2 = _two_module_tree(tmp_path)
    dirty = await summarize_hierarchy(fresh, edges2, tmp_path, FailingLLM(), stored=stored)

    assert {n.id: n.summary for n in fresh} == old_summaries  # stale kept, nothing emptied
    assert all(n.summary_input_hash == "" for n in fresh)  # forced retry next run
    assert dirty == set()  # kept summaries match their stored embeddings -> no re-embed


@pytest.mark.asyncio
async def test_summarize_failure_without_stored_summary_falls_back_and_is_dirty(tmp_path: Path) -> None:
    from inflorescence.code_indexer.summarizer import summarize_hierarchy

    nodes, edges = _two_module_tree(tmp_path)
    dirty = await summarize_hierarchy(nodes, edges, tmp_path, FailingLLM())

    # No previous summary to fall back on: leaves get their docstring (may be empty),
    # every failed node is marked dirty and hashless so the next run regenerates it.
    assert all(n.summary_input_hash == "" for n in nodes)
    assert dirty == {n.id for n in nodes}


@pytest.mark.asyncio
async def test_summarize_container_map_reduces_when_children_exceed_budget() -> None:
    """A container that doesn't fit one prompt is digested in batches, then merged."""
    from inflorescence.code_indexer.summarizer import _summarize_container

    parent = CodeNode(id="pkg", name="pkg", node_type=NodeType.DIRECTORY, file_path="pkg", line_start=0, line_end=0)
    node_map = {"pkg": parent}
    children_map: dict[str, list[str]] = {"pkg": []}
    for i in range(6):
        child = CodeNode(
            id=f"pkg/m{i}.py", name=f"m{i}", node_type=NodeType.MODULE,
            file_path=f"pkg/m{i}.py", line_start=1, line_end=1,
        )
        child.summary = "x" * 100
        node_map[child.id] = child
        children_map["pkg"].append(child.id)

    # Comfortably under budget -> a single LLM call.
    single = CountingLLM()
    s1 = await _summarize_container(parent, children_map, node_map, single, max_chars=100_000)
    assert s1 and single.calls == 1

    # Over budget -> one digest call per batch plus one reduce call (> 2 total).
    batched = CountingLLM()
    s2 = await _summarize_container(parent, children_map, node_map, batched, max_chars=150)
    assert s2 and batched.calls > 2


@pytest.mark.asyncio
async def test_editing_one_leaf_resummarizes_only_its_path_to_root(tmp_path: Path) -> None:
    from inflorescence.code_indexer.summarizer import summarize_hierarchy

    nodes, edges = _two_module_tree(tmp_path)
    await summarize_hierarchy(nodes, edges, tmp_path, CountingLLM())
    stored = {n.id: {"summary": n.summary, "summary_input_hash": n.summary_input_hash, "summary_embedding": None} for n in nodes}

    fresh, edges2 = _two_module_tree(tmp_path)            # rewrites files to original content
    (tmp_path / "a.py").write_text("def foo():\n    return 999\n", encoding="utf-8")  # edit ONE leaf
    llm = CountingLLM()

    dirty = await summarize_hierarchy(fresh, edges2, tmp_path, llm, stored=stored)

    assert dirty == {"a.py::foo", "a.py", "dir:."}         # only the edited path to root
    assert llm.calls == 3                                  # sibling subtree cost zero LLM
    by_id = {n.id: n for n in fresh}
    assert by_id["b.py::bar"].summary == stored["b.py::bar"]["summary"]
    assert by_id["b.py"].summary == stored["b.py"]["summary"]


@pytest.mark.asyncio
async def test_build_exposes_dirty_ids_and_threads_stored(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    builder = GraphBuilder(repo=RecordingRepository(), llm=CountingLLM(), config=IndexerConfig(use_llm_fallback_parser=False))
    result = await builder.build("fixture", tmp_path)
    nodes = result.nodes
    # Dirty-set is per-run state on the returned result, not a shared instance field.
    assert result.dirty_summary_ids == {n.id for n in nodes}

    stored = {n.id: {"summary": n.summary, "summary_input_hash": n.summary_input_hash, "summary_embedding": None} for n in nodes}
    llm2 = CountingLLM()
    builder2 = GraphBuilder(repo=RecordingRepository(), llm=llm2, config=IndexerConfig(use_llm_fallback_parser=False))
    result2 = await builder2.build("fixture", tmp_path, stored_summaries=stored)
    assert result2.dirty_summary_ids == set()
    assert llm2.calls == 0


@pytest.mark.asyncio
async def test_update_reads_stored_before_delete_and_reuses(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")

    seed = GraphBuilder(repo=RecordingRepository(), llm=CountingLLM(), config=IndexerConfig(use_llm_fallback_parser=False))
    nodes, _ = await seed.build("fixture", tmp_path)
    stored = {n.id: {"summary": n.summary, "summary_input_hash": n.summary_input_hash, "summary_embedding": None} for n in nodes}

    repo = RebuildRepository({"a.py": "stale-checksum"}, stored=stored)  # force a rebuild
    llm = CountingLLM()
    builder = GraphBuilder(repo=repo, llm=llm, config=IndexerConfig(use_llm_fallback_parser=False))
    result = await builder.update("fixture", tmp_path)

    assert repo.deleted_projects == []  # reconcile: the project is never wiped
    assert repo.deleted_code_edges == 0  # edges are MERGEd in place, never wiped (finding 9)
    assert result.dirty_summary_ids == set()  # no content changed => no LLM
    assert llm.calls == 0
    assert repo.summaries_stored == []  # INV-5: full reuse writes no summary batches either


class FailingStoreRepository(RecordingRepository):
    """store_summaries dies from the Nth call on — a DB lost mid-summarization."""

    def __init__(self, fail_from_call: int) -> None:
        super().__init__()
        self._fail_from = fail_from_call
        self._store_calls = 0

    async def store_summaries(self, project: str, items: list) -> int:
        self._store_calls += 1
        if self._store_calls >= self._fail_from:
            raise RuntimeError("db died")
        return await super().store_summaries(project, items)


class ExplodingLLM(CountingLLM):
    """Fails only on prompts containing a marker string."""

    def __init__(self, marker: str) -> None:
        super().__init__()
        self._marker = marker

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        if self._marker in prompt:
            self.calls += 1
            raise RuntimeError("transient outage")
        return await super().generate(prompt, system_prompt)


@pytest.mark.asyncio
async def test_build_early_upserts_then_persists_summary_batches(tmp_path: Path) -> None:
    """The skeleton lands before any summary batch; checksums still close the run."""
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo, llm=CountingLLM(),
        config=IndexerConfig(batch_size=1, use_llm_fallback_parser=False),
    )
    result = await builder.build("fixture", tmp_path)

    order = repo.calls
    assert order.index("upsert_nodes") < order.index("upsert_edges") < order.index("store_summaries") < order.index("update_checksums")
    stored_ids = {item["node_id"] for item in repo.summaries_stored}
    assert stored_ids == {n.id for n in result.nodes if n.summary} and stored_ids


@pytest.mark.asyncio
async def test_interrupted_summarize_reuses_persisted_batches(tmp_path: Path) -> None:
    """A run killed mid-summarize keeps its spend: the retry re-pays only the lost nodes (INV-8/INV-10)."""
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 2\n", encoding="utf-8")
    cfg = IndexerConfig(batch_size=1, use_llm_fallback_parser=False)

    control = CountingLLM()  # cost of an uninterrupted full build
    await GraphBuilder(repo=RecordingRepository(), llm=control, config=cfg).build("fixture", tmp_path)

    repo = FailingStoreRepository(fail_from_call=2)
    with pytest.raises(RuntimeError):
        await GraphBuilder(repo=repo, llm=CountingLLM(), config=cfg).build("fixture", tmp_path)
    persisted = repo.summaries_stored
    assert persisted  # the first batch survived the crash

    stored = {
        item["node_id"]: {"summary": item["summary"], "summary_input_hash": item["summary_input_hash"], "summary_embedding": None}
        for item in persisted
    }
    llm2 = CountingLLM()
    await GraphBuilder(repo=RecordingRepository(), llm=llm2, config=cfg).build(
        "fixture", tmp_path, stored_summaries=stored
    )
    assert llm2.calls == control.calls - len(persisted)


@pytest.mark.asyncio
async def test_containers_drain_as_soon_as_children_complete(tmp_path: Path) -> None:
    """A file's summary lands right after its functions, not after ALL leaves (drain)."""
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n", encoding="utf-8")
    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo, llm=CountingLLM(),
        config=IndexerConfig(batch_size=1, use_llm_fallback_parser=False),
    )
    await builder.build("fixture", tmp_path)

    order = [i["node_id"] for i in repo.summaries_stored]
    assert order.index("a.py") < order.index("b.py::bar")  # module a.py drained early
    assert order.index("dir:.") == len(order) - 1  # the root drains only when all files are done


@pytest.mark.asyncio
async def test_failed_batch_persists_retry_marker(tmp_path: Path) -> None:
    """A node that fails generation persists its stale summary + the '' retry marker (INV-3)."""
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n\ndef bar():\n    return 424242\n", encoding="utf-8")
    stored = {"a.py::bar": {"summary": "old bar summary", "summary_input_hash": "stale-hash", "summary_embedding": None}}
    repo = RecordingRepository()
    builder = GraphBuilder(
        repo=repo, llm=ExplodingLLM("424242"),
        config=IndexerConfig(batch_size=1, use_llm_fallback_parser=False),
    )
    await builder.build("fixture", tmp_path, stored_summaries=stored)

    items = {i["node_id"]: i for i in repo.summaries_stored}
    assert items["a.py::bar"]["summary"] == "old bar summary"  # stale beats empty
    assert items["a.py::bar"]["summary_input_hash"] == ""  # dirty again next run


def test_preview_counts_files_languages_and_estimates(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("console.log(1)\n", encoding="utf-8")

    builder = GraphBuilder(repo=RecordingRepository(), llm=None, config=IndexerConfig(use_llm_summaries=False))
    preview = builder.preview(tmp_path)

    expected_bytes = (tmp_path / "a.py").stat().st_size + (tmp_path / "b.py").stat().st_size
    assert preview["files"] == 2                              # node_modules excluded
    assert preview["total_bytes"] == expected_bytes
    assert set(preview["languages"]) == {"python"}
    assert preview["languages"]["python"]["files"] == 2
    assert preview["largest_files"][0]["path"] in {"a.py", "b.py"}
    assert preview["estimated_tokens"] == expected_bytes // 4
    assert isinstance(preview["estimated_nodes"], int)


def test_preview_honors_size_limit(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "big.py").write_text("x = 1\n" + ("# pad\n" * 5000), encoding="utf-8")

    builder = GraphBuilder(repo=None, llm=None, config=IndexerConfig(max_file_size_bytes=200, use_llm_summaries=False))
    preview = builder.preview(tmp_path)

    assert preview["files"] == 1
    assert [f["path"] for f in preview["largest_files"]] == ["small.py"]
    assert preview["skipped_large_files"] == ["big.py"]


class RepairRepository:
    async def upsert_external_nodes(self, project: str, refs: list) -> int:
        return len(refs)

    async def delete_orphan_external_nodes(self, project: str) -> int:
        return 0

    """Fake repo for repair_missing_summaries: canned missing nodes + child rows."""

    def __init__(
        self,
        missing: list[tuple[CodeNode, int]],
        children: dict[str, list[CodeNode]] | None = None,
    ) -> None:
        self.missing = missing
        self.children = children or {}
        self.stored: list[dict[str, object]] = []
        self.calls: list[str] = []

    async def get_nodes_missing_summaries(self, project: str) -> list[tuple[CodeNode, int]]:
        self.calls.append("missing")
        return self.missing

    async def get_child_summary_rows(self, project: str, node_id: str) -> list[CodeNode]:
        self.calls.append(f"children:{node_id}")
        return self.children.get(node_id, [])

    async def store_summaries(self, project: str, items: list[dict[str, object]]) -> int:
        self.calls.append("store:" + ",".join(str(i["node_id"]) for i in items))
        self.stored.extend(items)
        return len(items)


@pytest.mark.asyncio
async def test_repair_missing_summaries_heals_leaves_then_containers(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    leaf = CodeNode(
        id="mod.py::helper", name="helper", node_type=NodeType.FUNCTION,
        file_path="mod.py", line_start=1, line_end=2, signature="def helper()",
    )
    module = CodeNode(
        id="mod.py", name="mod.py", node_type=NodeType.MODULE,
        file_path="mod.py", line_start=1, line_end=2,
    )
    directory = CodeNode(
        id="dir:.", name=".", node_type=NodeType.DIRECTORY,
        file_path="", line_start=1, line_end=1,
    )
    repo = RepairRepository(
        missing=[(directory, 1), (leaf, 0), (module, 1)],
        children={"mod.py": [leaf], "dir:.": [module]},
    )
    llm = CountingLLM()
    builder = GraphBuilder(
        repo=repo, llm=llm, settings=Settings(_env_file=None),
        config=IndexerConfig(use_llm_summaries=True),
    )

    repaired = await builder.repair_missing_summaries("proj", tmp_path)

    assert [n.id for n in repaired] == ["mod.py::helper", "mod.py", "dir:."]
    assert all(n.summary.startswith("summary::") for n in repaired)
    assert all(i["summary_input_hash"] for i in repo.stored)
    assert llm.calls == 3
    # Children-first: the leaf is stored before its module, the module before
    # its directory.
    assert repo.calls.index("store:mod.py::helper") < repo.calls.index("store:mod.py")
    assert repo.calls.index("store:mod.py") < repo.calls.index("store:dir:.")


@pytest.mark.asyncio
async def test_repair_missing_summaries_noop_without_llm_summaries(tmp_path: Path) -> None:
    repo = RepairRepository(missing=[])
    builder = GraphBuilder(
        repo=repo, llm=None, settings=Settings(_env_file=None),
        config=IndexerConfig(use_llm_summaries=False),
    )

    assert await builder.repair_missing_summaries("proj", tmp_path) == []
    assert repo.calls == []  # guard short-circuits before any repo query


@pytest.mark.asyncio
async def test_repair_missing_summaries_skips_unsummarizable_nodes(tmp_path: Path) -> None:
    (tmp_path / "__init__.py").write_text("", encoding="utf-8")
    empty = CodeNode(
        id="__init__.py", name="__init__.py", node_type=NodeType.MODULE,
        file_path="__init__.py", line_start=1, line_end=1,
    )
    repo = RepairRepository(missing=[(empty, 0)])
    llm = CountingLLM()
    builder = GraphBuilder(
        repo=repo, llm=llm, settings=Settings(_env_file=None),
        config=IndexerConfig(use_llm_summaries=True),
    )

    repaired = await builder.repair_missing_summaries("proj", tmp_path)

    assert repaired == []                          # nothing to summarize -> not "repaired"
    assert llm.calls == 0                          # empty source never reaches the LLM
    assert not any(c.startswith("store:") for c in repo.calls)


class _BoomLLM:
    """An LLM that always fails — stands in for a transient LLM outage during repair."""

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        raise RuntimeError("LLM unavailable")


@pytest.mark.asyncio
async def test_repair_missing_summaries_marks_surrogate_for_retry_on_llm_failure(tmp_path: Path) -> None:
    """INV-3/INV-8 oracle (audit finding 8c): a repair that falls back to the docstring surrogate
    stores it with an EMPTY input hash (retry marker), so the next healthy run re-summarizes
    it — the old code wrote a *valid* hash and the surrogate was locked in forever.
    """
    (tmp_path / "mod.py").write_text("def helper():\n    'doc'\n    return 1\n", encoding="utf-8")
    leaf = CodeNode(
        id="mod.py::helper", name="helper", node_type=NodeType.FUNCTION,
        file_path="mod.py", line_start=1, line_end=3, signature="def helper()", docstring="doc",
    )
    repo = RepairRepository(missing=[(leaf, 0)])
    builder = GraphBuilder(
        repo=repo, llm=_BoomLLM(), settings=Settings(_env_file=None),
        config=IndexerConfig(use_llm_summaries=True),
    )

    repaired = await builder.repair_missing_summaries("proj", tmp_path)

    assert [n.id for n in repaired] == ["mod.py::helper"]        # surrogate stored, not dropped
    assert repo.stored[0]["summary"] == "doc"                    # docstring fallback
    assert repo.stored[0]["summary_input_hash"] == ""            # retry marker, NOT a valid hash


def test_confirmed_absent_distinguishes_missing_from_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INV-1/INV-3: only a positive not-found may confirm a deletion for the reconcile.

    ``Path.exists()`` returns False on PermissionError too, so a directory that lost read
    permission would have its indexed files counted as confirmed-deleted and destroyed —
    the reconcile must treat "can't stat" as "keep stale", never as "gone"."""
    from inflorescence.code_indexer.graph_builder import _confirmed_absent

    present = tmp_path / "present.py"
    present.write_text("x = 1\n", encoding="utf-8")
    assert _confirmed_absent(present) is False
    assert _confirmed_absent(tmp_path / "missing.py") is True

    def _deny(self: Path, **kwargs: object) -> object:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "stat", _deny)
    assert _confirmed_absent(present) is False  # unreadable ≠ deleted


# ---------------------------------------------------------------------------
# Edges that point outside the indexed project
# ---------------------------------------------------------------------------


def test_external_imports_become_dependency_nodes() -> None:
    """`import "net/http"` has no node to attach to, so the write layer used to discard the
    edge silently — 29% of all edges on a Go service. It is a dependency, not a mistake."""
    from inflorescence.code_indexer.graph_builder import _resolve_external_edges
    from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, NodeType

    nodes = [
        CodeNode(id="main.go", name="main", node_type=NodeType.MODULE, file_path="main.go", line_start=1, line_end=9),
        CodeNode(id="util/util.go", name="util", node_type=NodeType.MODULE, file_path="util/util.go", line_start=1, line_end=9),
    ]
    edges = [
        Edge(source="main.go", target="util/util.go", edge_type=EdgeType.IMPORTS),
        Edge(source="main.go", target="net/http", edge_type=EdgeType.IMPORTS),
        Edge(source="main.go", target="github.com/stretchr/testify/require", edge_type=EdgeType.IMPORTS),
    ]
    resolved, refs = _resolve_external_edges(edges, nodes, {n.id for n in nodes})

    assert len(resolved) == 3, "no edge is dropped any more"
    targets = {e.target for e in resolved}
    assert "util/util.go" in targets
    assert {"ext:module:net/http", "ext:module:github.com/stretchr/testify/require"} <= targets

    by_name = {r.name: r for r in refs}
    assert by_name["net/http"].kind == "module"
    # stdlib is its own root; a module path collapses to the distributable
    assert by_name["net/http"].root == "net/http"
    assert by_name["github.com/stretchr/testify/require"].root == "github.com/stretchr/testify"


def test_external_base_classes_are_recorded_as_class_kind() -> None:
    from inflorescence.code_indexer.graph_builder import _resolve_external_edges
    from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, NodeType

    nodes = [
        CodeNode(id="models.py::Config", name="Config", node_type=NodeType.CLASS, file_path="models.py", line_start=1, line_end=9),
    ]
    edges = [Edge(source="models.py::Config", target="pydantic.BaseModel", edge_type=EdgeType.INHERITS)]
    resolved, refs = _resolve_external_edges(edges, nodes, {n.id for n in nodes})

    assert resolved[0].target == "ext:class:pydantic.BaseModel"
    assert refs[0].kind == "class" and refs[0].root == "pydantic"


def test_one_dependency_used_by_many_files_is_one_node() -> None:
    """The point of a node over a per-node property: net/http imported fifty times is one
    node with fifty edges, not the same string stored fifty times."""
    from inflorescence.code_indexer.graph_builder import _resolve_external_edges
    from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, NodeType

    nodes = [
        CodeNode(id=f"f{i}.go", name=f"f{i}", node_type=NodeType.MODULE, file_path=f"f{i}.go", line_start=1, line_end=9)
        for i in range(3)
    ]
    edges = [Edge(source=n.id, target="net/http", edge_type=EdgeType.IMPORTS) for n in nodes]
    resolved, refs = _resolve_external_edges(edges, nodes, {n.id for n in nodes})

    assert len(resolved) == 3 and len(refs) == 1


def test_npm_scoped_package_root_keeps_the_scope() -> None:
    from inflorescence.code_indexer.graph_builder import _external_root

    assert _external_root("@electron-forge/maker-zip", "typescript") == "@electron-forge/maker-zip"
    assert _external_root("lodash/get", "typescript") == "lodash"
    assert _external_root("pydantic.v1.main", "python") == "pydantic"


def test_calls_never_reach_the_external_path() -> None:
    """An unresolved callee is kept on the caller as external_calls; turning it into a
    dependency node would invent a module that does not exist."""
    from inflorescence.code_indexer.graph_builder import _resolve_external_edges
    from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, NodeType

    nodes = [CodeNode(id="a.py::f", name="f", node_type=NodeType.FUNCTION, file_path="a.py", line_start=1, line_end=2)]
    edges = [Edge(source="a.py::f", target="json.dumps", edge_type=EdgeType.CALLS)]
    resolved, refs = _resolve_external_edges(edges, nodes, {"a.py::f"})

    assert resolved == [] and refs == []
