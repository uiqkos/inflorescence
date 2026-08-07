"""CALLS resolution ladder + SCIP semantic enrichment tests.

Covers the redesign described in docs/calls-resolution.md:
per-language ladders (import / field-type / self / same-module|package), the shadow
guard, honest external/unresolved accounting, the minimal SCIP protobuf reader, the
adapter, and degradation when a semantic indexer is unavailable or fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inflorescence.code_indexer import scip_semantic as _scip_semantic
from inflorescence.code_indexer.graph_builder import _merge_semantic_calls
from inflorescence.code_indexer.models import (
    CallResolution,
    CodeNode,
    Edge,
    EdgeType,
    IndexerConfig,
    NodeType,
)
from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.go_parser import GoParser
from inflorescence.code_indexer.parser.javascript_parser import JavaScriptParser
from inflorescence.code_indexer.parser.resolver import resolve_edges
from inflorescence.code_indexer.scip_semantic import (
    ScipDocument,
    ScipOccurrence,
    SemanticCalls,
    derive_semantic_calls,
    parse_scip_index,
    run_scip_indexer,
    semantic_calls_pass,
)


@pytest.fixture(autouse=True)
def _no_docker_probe(monkeypatch):
    """Keep unit tests off the host's docker daemon.

    With the default images pinned by digest, choosing the docker rung consults the local
    daemon for a locally built tag (`_prefer_local_tag`). Real daemon state would make
    the resolved reference depend on which machine ran the suite — same reason
    `_stub_scip` exists in test_onboarding.py.
    """
    monkeypatch.setattr(_scip_semantic, "_local_image_present", lambda image: False)


def _parse_project(parser, root: Path, files: list[Path]):
    nodes, edges, facts = [], [], {}
    for f in files:
        n, e, ff = parser.parse_file_with_facts(f, root)
        nodes += n
        edges += e
        if ff is not None:
            facts[ff.file_path] = ff
    return nodes, edges, facts


def _calls(result) -> dict[tuple[str, str], str]:
    return {
        (e.source, e.target): e.resolution
        for e in result.edges
        if e.edge_type == EdgeType.CALLS
    }


# ---------------------------------------------------------------------------
# Go ladder
# ---------------------------------------------------------------------------


def _write_go_project(root: Path) -> list[Path]:
    (root / "go.mod").write_text("module example.com/app\n\ngo 1.22\n")
    main = root / "main.go"
    main.write_text(
        """package main

import (
\t"fmt"

\t"example.com/app/store"
\t"example.com/app/util"
)

func main() {
\tutil.Greet()
\thelper()
\tfmt.Println("hi")
\t_ = store.New()
}

func helper() {
\tshadowed := func() {}
\tshadowed()
}
"""
    )
    (root / "util").mkdir()
    util = root / "util" / "util.go"
    util.write_text(
        """package util

import (
\t"log/slog"

\t"example.com/app/store"
)

type Service struct {
\tstore *store.Store
\tlog   *slog.Logger
}

func Greet() {}

func (s *Service) Run() {
\ts.Helper()
\ts.store.Save()
\ts.log.Info("x")
}

func (s *Service) Helper() {}
"""
    )
    (root / "store").mkdir()
    st = root / "store" / "store.go"
    st.write_text(
        """package store

type Store struct{}

func New() *Store { return &Store{} }

func (st *Store) Save() {}
"""
    )
    return [main, util, st]


def test_go_ladder_resolves_import_samepackage_self_and_fieldtype(tmp_path: Path) -> None:
    files = _write_go_project(tmp_path)
    nodes, edges, facts = _parse_project(GoParser(), tmp_path, files)
    result = resolve_edges(nodes, edges, facts)
    calls = _calls(result)

    assert calls[("main.go::main", "util/util.go::Greet")] == "import"
    assert calls[("main.go::main", "store/store.go::New")] == "import"
    assert calls[("main.go::main", "main.go::helper")] == "same-package"
    assert calls[("util/util.go::Service.Run", "util/util.go::Service.Helper")] == "self"
    # The documented repro shape: s.<field>.<Method> through the field's declared type
    assert calls[("util/util.go::Service.Run", "store/store.go::Store.Save")] == "field-type"

    # stdlib calls are external facts, not missing edges
    assert "fmt.Println" in result.external_calls["main.go::main"]
    assert "s.log.Info" in result.external_calls["util/util.go::Service.Run"]


def test_go_shadow_guard_blocks_local_bindings(tmp_path: Path) -> None:
    files = _write_go_project(tmp_path)
    nodes, edges, facts = _parse_project(GoParser(), tmp_path, files)
    result = resolve_edges(nodes, edges, facts)

    # `shadowed := func(){}; shadowed()` must not bind to any package function
    assert ("main.go::helper", "main.go::shadowed") not in _calls(result)
    assert "shadowed" in result.unresolved_calls["main.go::helper"]


def test_go_facts_extraction(tmp_path: Path) -> None:
    files = _write_go_project(tmp_path)
    _, _, facts = _parse_project(GoParser(), tmp_path, files)

    util = facts["util/util.go"]
    assert util.go_module == "example.com/app"
    assert util.imports["store"] == "example.com/app/store"
    assert util.struct_fields["Service"]["store"] == "*store.Store"
    assert util.receiver_names["Service.Run"] == "s"


# ---------------------------------------------------------------------------
# Python ladder
# ---------------------------------------------------------------------------


def _write_py_project(root: Path) -> list[Path]:
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    util = pkg / "util.py"
    util.write_text(
        """def helper():
    pass


class Base:
    def ping(self):
        pass
"""
    )
    main = pkg / "main.py"
    main.write_text(
        """import json

from pkg.util import Base, helper


def caller():
    helper()
    own()
    json.dumps({})
    len([])


def own():
    pass


def local_shadow(helper):
    helper()


class Child(Base):
    def run(self):
        self.ping()
        self.missing()
"""
    )
    return [pkg / "__init__.py", util, main]


def test_python_ladder_import_samemodule_self_and_external(tmp_path: Path) -> None:
    files = _write_py_project(tmp_path)
    nodes, edges, facts = _parse_project(PythonAstParser(), tmp_path, files)
    result = resolve_edges(nodes, edges, facts)
    calls = _calls(result)

    assert calls[("pkg/main.py::caller", "pkg/util.py::helper")] == "import"
    assert calls[("pkg/main.py::caller", "pkg/main.py::own")] == "same-module"
    # self.ping() found through the INHERITS chain into the imported base class
    assert calls[("pkg/main.py::Child.run", "pkg/util.py::Base.ping")] == "self"

    assert "json.dumps" in result.external_calls["pkg/main.py::caller"]
    # builtins are noise, recorded nowhere
    assert "len" not in result.unresolved_calls.get("pkg/main.py::caller", [])
    assert "self.missing" in result.unresolved_calls["pkg/main.py::Child.run"]


def test_python_shadow_guard_blocks_parameter_shadowing(tmp_path: Path) -> None:
    """Measured failure mode on cerberus: def f(init_validator): init_validator()
    bound to the module-level function of the same name."""
    files = _write_py_project(tmp_path)
    nodes, edges, facts = _parse_project(PythonAstParser(), tmp_path, files)
    result = resolve_edges(nodes, edges, facts)

    assert ("pkg/main.py::local_shadow", "pkg/util.py::helper") not in _calls(result)
    assert "helper" in result.unresolved_calls["pkg/main.py::local_shadow"]


def test_python_module_level_calls_attribute_to_module_node(tmp_path: Path) -> None:
    script = tmp_path / "script.py"
    script.write_text(
        """def setup():
    pass


setup()
unknown_thing()
"""
    )
    nodes, edges, facts = _parse_project(PythonAstParser(), tmp_path, [script])
    result = resolve_edges(nodes, edges, facts)

    assert _calls(result)[("script.py", "script.py::setup")] == "same-module"
    assert "unknown_thing" in result.unresolved_calls["script.py"]


# ---------------------------------------------------------------------------
# JavaScript / TypeScript ladder
# ---------------------------------------------------------------------------


def _write_js_project(root: Path) -> list[Path]:
    src = root / "src"
    src.mkdir()
    util = src / "util.ts"
    util.write_text(
        """export function helper() {}
export function other() {}
"""
    )
    app = src / "app.ts"
    app.write_text(
        """import axios from 'axios'
import * as ns from './util'
import { helper as h } from './util'

export function run() {
  h()
  ns.other()
  inner()
  axios.get('/x')
  fetch('/y').then(() => {})
  const local = () => {}
  local()
}

function inner() {}

class C {
  load() {
    this.save()
  }
  save() {}
}

setupTests()
"""
    )
    return [util, app]


def test_js_ladder_import_namespace_samemodule_self_external(tmp_path: Path) -> None:
    files = _write_js_project(tmp_path)
    nodes, edges, facts = _parse_project(JavaScriptParser(), tmp_path, files)
    result = resolve_edges(nodes, edges, facts)
    calls = _calls(result)

    assert calls[("src/app.ts::run", "src/util.ts::helper")] == "import"
    assert calls[("src/app.ts::run", "src/util.ts::other")] == "import"
    assert calls[("src/app.ts::run", "src/app.ts::inner")] == "same-module"
    assert calls[("src/app.ts::C.load", "src/app.ts::C.save")] == "self"
    # default import from a bare (npm) specifier is an external call
    assert "axios.get" in result.external_calls["src/app.ts::run"]
    # JS runtime globals (fetch) are noise — recorded nowhere
    run_unresolved = result.unresolved_calls.get("src/app.ts::run", [])
    assert not any(u.startswith("fetch") for u in run_unresolved)
    # locally-declared arrow shadows nothing else; it stays unresolved, not a bad edge
    assert "local" in run_unresolved


def test_js_top_level_calls_attribute_to_module_node(tmp_path: Path) -> None:
    files = _write_js_project(tmp_path)
    nodes, edges, facts = _parse_project(JavaScriptParser(), tmp_path, files)
    result = resolve_edges(nodes, edges, facts)

    # `setupTests()` at the top level lands on the module node, honestly unresolved
    assert "setupTests" in result.unresolved_calls["src/app.ts"]


def test_js_cjs_require_bindings(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "utils.js").write_text("function shave(s) { return s }\nmodule.exports = { shave }\n")
    (lib / "view.js").write_text(
        """var utils = require('./utils')
var { shave } = require('./utils')

function render() {
  utils.shave('x')
  shave('y')
}
"""
    )
    nodes, edges, facts = _parse_project(
        JavaScriptParser(), tmp_path, [lib / "utils.js", lib / "view.js"]
    )
    result = resolve_edges(nodes, edges, facts)
    calls = _calls(result)

    assert calls[("lib/view.js::render", "lib/utils.js::shave")] == "import"


def test_calls_deduplicated_per_source_target_pair(tmp_path: Path) -> None:
    f = tmp_path / "m.py"
    f.write_text("def a():\n    b()\n    b()\n    b()\n\n\ndef b():\n    pass\n")
    nodes, edges, facts = _parse_project(PythonAstParser(), tmp_path, [f])
    result = resolve_edges(nodes, edges, facts)

    calls = [e for e in result.edges if e.edge_type == EdgeType.CALLS]
    assert len(calls) == 1
    assert calls[0].callee_text == "b"


# ---------------------------------------------------------------------------
# SCIP: minimal protobuf reader
# ---------------------------------------------------------------------------


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def _ld(field_no: int, payload: bytes) -> bytes:  # length-delimited field
    return _varint((field_no << 3) | 2) + _varint(len(payload)) + payload


def _vi(field_no: int, value: int) -> bytes:  # varint field
    return _varint(field_no << 3) + _varint(value)


def _occurrence(line: int, symbol: str, roles: int = 0, typed_single_line: bool = False) -> bytes:
    if typed_single_line:
        rng = _ld(8, _vi(1, line) + _vi(2, 0) + _vi(3, 4))
    else:
        rng = _ld(1, _varint(line) + _varint(0) + _varint(4))  # packed [line, sc, ec]
    body = rng + _ld(2, symbol.encode()) + (_vi(3, roles) if roles else b"")
    return body


def test_parse_scip_index_reads_synthetic_wire_format() -> None:
    doc = _ld(1, b"a.go") + _ld(2, _occurrence(3, "pkg/Foo().", roles=1)) + _ld(
        2, _occurrence(7, "pkg/Bar().", typed_single_line=True)
    )
    index = _ld(1, b"\x08\x01") + _ld(2, doc)  # metadata (ignored) + document

    docs = parse_scip_index(index)

    assert len(docs) == 1
    assert docs[0].relative_path == "a.go"
    assert [(o.start_line, o.symbol, o.roles) for o in docs[0].occurrences] == [
        (3, "pkg/Foo().", 1),
        (7, "pkg/Bar().", 0),
    ]


def test_parse_scip_index_matches_nothing_on_garbage() -> None:
    assert parse_scip_index(b"") == []


# ---------------------------------------------------------------------------
# SCIP: adapter + merge
# ---------------------------------------------------------------------------


def _nodes_for_adapter() -> list[CodeNode]:
    return [
        CodeNode(id="a.go", name="a", node_type=NodeType.MODULE, file_path="a.go", line_start=1, line_end=40),
        CodeNode(id="a.go::Foo", name="Foo", node_type=NodeType.FUNCTION, file_path="a.go", line_start=3, line_end=10),
        CodeNode(id="a.go::Bar", name="Bar", node_type=NodeType.FUNCTION, file_path="a.go", line_start=12, line_end=20),
    ]


def test_derive_semantic_calls_builds_edges_and_externals() -> None:
    docs = [
        ScipDocument(
            relative_path="a.go",
            occurrences=[
                ScipOccurrence(start_line=2, symbol="mod app/Foo().", roles=1),  # def Foo (line 3)
                ScipOccurrence(start_line=11, symbol="mod app/Bar().", roles=1),  # def Bar (line 12)
                ScipOccurrence(start_line=14, symbol="mod app/Foo().", roles=8),  # Bar calls Foo
                ScipOccurrence(start_line=15, symbol="gomod go1 fmt/Println().", roles=8),
            ],
        ),
        ScipDocument(relative_path="../outside/cache.go", occurrences=[]),
    ]

    result = derive_semantic_calls([("", docs)], _nodes_for_adapter())

    assert [(e.source, e.target, e.resolution) for e in result.edges] == [
        ("a.go::Bar", "a.go::Foo", CallResolution.SEMANTIC.value)
    ]
    assert result.external_calls["a.go::Bar"] == ["fmt.Println"]
    assert result.covered_files == {"a.go"}  # the go-build cache artifact is filtered out


def test_merge_semantic_calls_replaces_heuristic_and_keeps_module_level() -> None:
    nodes = _nodes_for_adapter()
    nodes[1].unresolved_calls = ["x.Do"]
    heuristic = [
        Edge(source="a.go::Foo", target="a.go::Bar", edge_type=EdgeType.CALLS, resolution="same-package"),
        Edge(source="a.go", target="a.go::Foo", edge_type=EdgeType.CALLS, resolution="same-package"),
        Edge(source="a.go", target="a.go::Foo", edge_type=EdgeType.CONTAINS),
    ]
    semantic = SemanticCalls(
        edges=[Edge(source="a.go::Bar", target="a.go::Foo", edge_type=EdgeType.CALLS, resolution="semantic")],
        external_calls={"a.go::Foo": ["fmt.Println"]},
        covered_files={"a.go"},
        provenance={"a.go": "scip-go"},
    )

    merged = _merge_semantic_calls(nodes, heuristic, semantic)

    call_keys = {(e.source, e.target, e.resolution) for e in merged if e.edge_type == EdgeType.CALLS}
    # function-level heuristic edge replaced; module-level heuristic edge kept
    assert ("a.go::Foo", "a.go::Bar", "same-package") not in call_keys
    assert ("a.go", "a.go::Foo", "same-package") in call_keys
    assert ("a.go::Bar", "a.go::Foo", "semantic") in call_keys
    # covered nodes: semantic is authoritative
    assert nodes[1].external_calls == ["fmt.Println"]
    assert nodes[1].unresolved_calls == []
    assert nodes[0].calls_provenance == "scip-go"


# ---------------------------------------------------------------------------
# SCIP: preflight + degradation on broken builds
# ---------------------------------------------------------------------------


def test_run_scip_indexer_degrades_when_binary_missing(tmp_path: Path) -> None:
    # Pinned to the native rung: under "auto" a missing binary is not a failure, it is a
    # fall-through to the container rung, which is what the docker tests below cover.
    config = IndexerConfig(scip_runner="native", scip_commands={"go": "definitely-not-a-real-binary-xyz"})
    assert run_scip_indexer("go", tmp_path, config) is None


def test_run_scip_indexer_degrades_when_indexer_fails(tmp_path: Path) -> None:
    # `false` exits 1 — models a project whose build/typecheck fails
    config = IndexerConfig(scip_runner="native", scip_commands={"go": "false"})
    assert run_scip_indexer("go", tmp_path, config) is None


# ---------------------------------------------------------------------------
# SCIP: choosing a rung (installed binary vs container)
# ---------------------------------------------------------------------------


def test_resolve_prefers_native_binary_for_go(tmp_path: Path, monkeypatch) -> None:
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/local/bin/scip-go")
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    invocation = ss.resolve_scip_invocation("go", tmp_path, tmp_path, IndexerConfig())
    assert invocation is not None
    assert invocation.runner == "native"
    assert invocation.argv[0] == "scip-go"
    assert invocation.cwd == tmp_path


def test_python_prefers_the_container_even_with_a_binary_installed(tmp_path: Path, monkeypatch) -> None:
    """A native scip-python binds intra-project calls to the site-packages copy when the
    project is installed into its own venv, losing roughly a third of the CALLS edges
    (measured: 1202 against 1786 on the same repository). The container has no venv to
    resolve against, so it resolves against the sources."""
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/local/bin/scip-python")
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    invocation = ss.resolve_scip_invocation("python", tmp_path, tmp_path, IndexerConfig())
    assert invocation is not None and invocation.runner == "docker"


def test_python_still_falls_back_to_the_binary_without_docker(tmp_path: Path, monkeypatch) -> None:
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/local/bin/scip-python")
    monkeypatch.setattr(ss, "docker_cli_present", lambda: False)
    invocation = ss.resolve_scip_invocation("python", tmp_path, tmp_path, IndexerConfig())
    assert invocation is not None and invocation.runner == "native"


def test_container_run_keeps_a_persistent_toolchain_cache(tmp_path: Path) -> None:
    """Without it every run recompiles the module graph: 25.1s against 0.7s warm, on a
    pass the watcher repeats after every batch of edits."""
    from inflorescence.code_indexer.scip_semantic import docker_scip_argv

    argv = docker_scip_argv("img:v1", tmp_path, tmp_path, ["scip-go"])
    assert "inflorescence_scip_cache:/cache" in argv
    assert "GOCACHE=/cache/go/build" in argv


def test_resolve_falls_back_to_container_when_binary_absent(tmp_path: Path, monkeypatch) -> None:
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    invocation = ss.resolve_scip_invocation("go", tmp_path, tmp_path, IndexerConfig())
    assert invocation is not None
    assert invocation.runner == "docker"
    assert invocation.argv[:3] == ["docker", "run", "--rm"]
    assert invocation.cwd is None


def test_resolve_returns_none_without_binary_or_docker(tmp_path: Path, monkeypatch) -> None:
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: False)
    assert ss.resolve_scip_invocation("go", tmp_path, tmp_path, IndexerConfig()) is None


def test_docker_runner_never_uses_a_native_binary(tmp_path: Path, monkeypatch) -> None:
    """`docker` pinned means the container, even when the binary happens to be installed."""
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/local/bin/scip-go")
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    invocation = ss.resolve_scip_invocation("go", tmp_path, tmp_path, IndexerConfig(scip_runner="docker"))
    assert invocation is not None and invocation.runner == "docker"


def test_docker_argv_mounts_repo_read_only_and_writes_index_elsewhere(tmp_path: Path) -> None:
    from inflorescence.code_indexer.scip_semantic import docker_scip_argv

    repo, out = tmp_path / "repo", tmp_path / "out"
    argv = docker_scip_argv("img:v1", repo, out, ["scip-go", "--output", "{output}"])

    assert f"{repo}:/repo:ro" in argv, "the indexed repository must never be writable"
    assert f"{out}:/out" in argv
    assert argv[-1] == "/out/index.scip"
    assert argv[argv.index("-w") + 1] == "/repo"
    # Caches are redirected into the container's writable /tmp; with the repo read-only a
    # toolchain that caches beside the sources would fail on its first write.
    assert "GOCACHE=/cache/go/build" in argv
    assert "HOME=/tmp" in argv


def test_docker_argv_appends_output_flag_when_command_has_no_placeholder(tmp_path: Path) -> None:
    from inflorescence.code_indexer.scip_semantic import docker_scip_argv

    argv = docker_scip_argv("img:v1", tmp_path, tmp_path, ["scip-typescript", "index"])
    assert argv[-2:] == ["--output", "/out/index.scip"]


def test_python_and_typescript_share_one_image_go_gets_its_own(tmp_path: Path, monkeypatch) -> None:
    """Splitting by toolchain is what keeps a Python index from pulling a Go toolchain."""
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)

    def image_for(language: str) -> str:
        invocation = ss.resolve_scip_invocation(language, tmp_path, tmp_path, IndexerConfig())
        assert invocation is not None
        return next(a for a in invocation.argv if a.startswith("ghcr.io/"))

    assert image_for("python") == image_for("typescript")
    assert image_for("go") != image_for("python")


def test_python_command_pins_a_project_version(tmp_path: Path, monkeypatch) -> None:
    """Without it scip-python reaches symbol emission and dies on an undefined version —
    which is what happens in any directory git cannot describe."""
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: "/usr/local/bin/scip-python")
    invocation = ss.resolve_scip_invocation("python", tmp_path, tmp_path, IndexerConfig())
    assert invocation is not None
    assert "--project-version" in invocation.argv


def test_image_override_is_honored(tmp_path: Path, monkeypatch) -> None:
    from inflorescence.code_indexer import scip_semantic as ss

    monkeypatch.setattr(ss.shutil, "which", lambda name: None)
    monkeypatch.setattr(ss, "docker_cli_present", lambda: True)
    config = IndexerConfig(scip_images={"go": "my-registry/scip-go:pinned"})
    invocation = ss.resolve_scip_invocation("go", tmp_path, tmp_path, config)
    assert invocation is not None and "my-registry/scip-go:pinned" in invocation.argv


def test_semantic_pass_returns_none_when_nothing_runs(tmp_path: Path) -> None:
    nodes = [
        CodeNode(id="x.go", name="x", node_type=NodeType.MODULE, file_path="x.go", line_start=1, line_end=5)
    ]
    # No go.mod in root -> no scip roots -> nothing to run
    assert semantic_calls_pass(tmp_path, nodes, IndexerConfig()) is None


def test_semantic_pass_degrades_to_none_on_failing_indexer(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/broken\n")
    (tmp_path / "main.go").write_text("package main\nfunc main() { undefined_symbol() }\n")
    nodes = [
        CodeNode(id="main.go", name="main", node_type=NodeType.MODULE, file_path="main.go", line_start=1, line_end=2)
    ]
    config = IndexerConfig(scip_commands={"go": "false"})
    assert semantic_calls_pass(tmp_path, nodes, config) is None


def test_go_parser_survives_syntax_errors(tmp_path: Path) -> None:
    """tree-sitter chews broken code: the file still yields its module node and the
    parseable prefix, so the heuristic plane never dies on a non-compiling project."""
    (tmp_path / "go.mod").write_text("module example.com/broken\n")
    broken = tmp_path / "broken.go"
    broken.write_text("package main\n\nfunc ok() { fine() }\n\nfunc broken( {{{ nonsense\n")
    nodes, edges, facts = _parse_project(GoParser(), tmp_path, [broken])

    ids = {n.id for n in nodes}
    assert "broken.go" in ids  # module node always present
    assert "broken.go::ok" in ids  # parseable prefix survives
    assert facts["broken.go"].go_module == "example.com/broken"
