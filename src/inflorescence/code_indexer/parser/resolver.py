"""Cross-file resolution of import/call edges to actual node IDs.

CALLS edges go through a confidence ladder (see docs/calls-resolution.md — each level
scored 100% precision against SCIP ground truth on real code):

    exact        target is already a node id
    import       first segment bound via the file's imports (pkg.Func / imported name)
    field-type   recv.field.Method via the field's declared type (Go)
    self         self./this./receiver method on the enclosing class
    same-module  bare name defined in the same file (python/js)
    same-package bare name defined in the same directory (Go)

What doesn't resolve is *recorded, not guessed*: calls bound to imports outside the
project land in ``external_calls`` of the caller node, everything else in
``unresolved_calls``. There is deliberately no "globally unique short name" fallback
(39% precision on the benchmark repository) and no ``candidates[0]`` pick for CALLS.

IMPORTS/INHERITS/IMPLEMENTS resolution is unchanged from the legacy behavior.
"""

from __future__ import annotations

import builtins
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from inflorescence.code_indexer.models import CallResolution, CodeNode, Edge, EdgeType, FileFacts

# Extension to language mapping for import resolution
_EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

_PY_BUILTINS = frozenset(dir(builtins))
_GO_BUILTINS = frozenset({
    "len", "cap", "make", "new", "append", "copy", "delete", "panic", "recover",
    "print", "println", "close", "complex", "real", "imag", "min", "max", "clear",
    "string", "bool", "byte", "rune", "error", "any",
    "int", "int8", "int16", "int32", "int64",
    "uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
    "float32", "float64", "complex64", "complex128",
})
# JS runtime globals: calls through these are neither project-internal nor an import —
# recording them would flood the lists with console.log/JSON.parse noise.
_JS_GLOBALS = frozenset({
    "console", "Math", "JSON", "Object", "Array", "String", "Number", "Boolean",
    "Promise", "Date", "RegExp", "Error", "TypeError", "Map", "Set", "WeakMap",
    "WeakSet", "Symbol", "Proxy", "Reflect", "Intl", "parseInt", "parseFloat",
    "isNaN", "isFinite", "encodeURIComponent", "decodeURIComponent", "structuredClone",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval", "queueMicrotask",
    "fetch", "window", "document", "navigator", "location", "history", "localStorage",
    "sessionStorage", "globalThis", "requestAnimationFrame", "cancelAnimationFrame",
    "alert", "confirm", "prompt", "atob", "btoa", "crypto", "performance",
    "AbortController", "URL", "URLSearchParams", "FormData", "Headers", "Request",
    "Response", "WebSocket", "EventSource", "CustomEvent", "Event", "Blob", "File",
    "FileReader", "Audio", "Image", "Option", "Worker", "SharedWorker", "Notification",
    "IntersectionObserver", "MutationObserver", "ResizeObserver", "BigInt", "Function",
    "eval", "require",
})
# Cap the honest-accounting lists so a pathological file can't balloon node properties.
_MAX_RECORDED_CALLS = 100


@dataclass
class ResolutionResult:
    """Resolved edges plus the honest per-node accounting of everything else."""

    edges: list[Edge]
    external_calls: dict[str, list[str]] = field(default_factory=dict)
    unresolved_calls: dict[str, list[str]] = field(default_factory=dict)


def _detect_language(file_path: str) -> str:
    """Determine language from file path extension."""
    suffix = PurePosixPath(file_path).suffix
    return _EXT_LANGUAGE.get(suffix, "python")


def _build_module_index(nodes: Sequence[CodeNode]) -> dict[str, str]:
    """Build a mapping from module-style names to node IDs for import resolution."""
    module_to_id: dict[str, str] = {}
    for n in nodes:
        if n.node_type.value != "module":
            continue
        lang = _detect_language(n.file_path)
        if lang == "python":
            # src/coding_agent/config.py -> coding_agent.config
            mod_name = n.file_path.replace("/", ".").removesuffix(".py")
            if mod_name.startswith("src."):
                mod_name = mod_name[4:]
            module_to_id[mod_name] = n.id
        elif lang in ("javascript", "typescript"):
            # JS/TS: relative imports like ./utils or ../helpers
            # Store by relative path without extension
            p = PurePosixPath(n.file_path)
            for ext in (".js", ".jsx", ".ts", ".tsx"):
                if n.file_path.endswith(ext):
                    module_to_id[str(p.with_suffix(""))] = n.id
                    break
            # Also store with extension
            module_to_id[n.file_path] = n.id
        elif lang == "go":
            # Go: imports are full package paths; store by directory (package)
            p = PurePosixPath(n.file_path)
            module_to_id[str(p.parent)] = n.id
            module_to_id[n.file_path] = n.id
        elif lang == "rust":
            # Rust: use crate::module::item; store by crate-relative path
            p = PurePosixPath(n.file_path)
            # src/lib.rs, src/main.rs, src/foo.rs -> crate::foo
            no_ext = str(p.with_suffix(""))
            parts_list = no_ext.split("/")
            if parts_list and parts_list[0] == "src":
                parts_list = parts_list[1:]
            # Remove lib/main as they map to crate root
            if parts_list and parts_list[-1] in ("lib", "main", "mod"):
                parts_list = parts_list[:-1]
            crate_path = "::".join(["crate", *parts_list]) if parts_list else "crate"
            module_to_id[crate_path] = n.id
            module_to_id[n.file_path] = n.id
    return module_to_id


def _build_python_symbol_index(nodes: Sequence[CodeNode]) -> dict[str, str]:
    index: dict[str, str] = {}
    module_to_id = _build_module_index(nodes)
    for node in nodes:
        if _detect_language(node.file_path) != "python":
            continue
        module_name = node.file_path.replace("/", ".").removesuffix(".py")
        if module_name.startswith("src."):
            module_name = module_name[4:]
        if node.node_type.value == "module":
            index[module_name] = node.id
            if module_name.endswith(".__init__"):
                index[module_name.removesuffix(".__init__")] = node.id
            continue
        index[f"{module_name}.{node.name.rsplit('.', 1)[-1]}"] = node.id
    index.update(module_to_id)
    return index


def _python_module_name(file_path: str) -> str:
    mod = file_path.replace("/", ".").removesuffix(".py")
    if mod.startswith("src."):
        mod = mod[4:]
    return mod.removesuffix(".__init__")


def _resolve_python_relative_import(source_file: str, target: str) -> str:
    if not target.startswith("."):
        return target
    level = len(target) - len(target.lstrip("."))
    remainder = target.lstrip(".")
    source_module_parts = PurePosixPath(source_file).with_suffix("").parts
    package_parts = list(source_module_parts[:-1])
    if PurePosixPath(source_file).name == "__init__.py":
        package_parts = list(source_module_parts)[:-1]
    if level > 1:
        package_parts = package_parts[: -(level - 1)]
    if remainder:
        package_parts.extend(remainder.split("."))
    return ".".join(part for part in package_parts if part)


def _try_resolve_js_relative_import(
    source_file: str,
    target: str,
    module_to_id: dict[str, str],
) -> str | None:
    """Attempt to resolve a JS/TS relative import to a module ID."""
    source_dir = str(PurePosixPath(source_file).parent)
    resolved_path = str(PurePosixPath(source_dir) / target)
    # Normalize (remove ./ and resolve ..)
    resolved_path = str(PurePosixPath(resolved_path))
    if resolved_path in module_to_id:
        return module_to_id[resolved_path]
    # Try with extensions
    for ext in (".js", ".jsx", ".ts", ".tsx"):
        candidate = resolved_path + ext
        if candidate in module_to_id:
            return module_to_id[candidate]
        # Try index file
        idx = resolved_path + "/index" + ext
        if idx in module_to_id:
            return module_to_id[idx]
    return None


class _CallLadder:
    """Per-project indexes + the per-language CALLS resolution ladder."""

    def __init__(self, nodes: Sequence[CodeNode], facts: dict[str, FileFacts]) -> None:
        self.facts = facts
        self.node_by_id = {n.id: n for n in nodes}
        self.module_to_id = _build_module_index(nodes)

        # Per-file and per-directory callable indexes.
        self.funcs_in_file: dict[str, dict[str, str]] = defaultdict(dict)
        self.methods_in_file: dict[str, dict[str, str]] = defaultdict(dict)
        self.classes_in_file: dict[str, dict[str, str]] = defaultdict(dict)
        self.funcs_in_dir: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.methods_in_dir: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        # Python dotted-symbol index: 'pkg.mod.func' / 'pkg.mod.Class' / 'pkg.mod.Class.meth'
        self.py_symbols: dict[str, str] = {}
        self.py_top_packages: set[str] = set()
        # INHERITS raw targets per class node id (for self.method() through bases)
        self.class_bases: dict[str, list[str]] = defaultdict(list)
        # Go package-wide unions (a struct may be declared in a sibling file)
        self.go_pkg_struct_fields: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        self.go_pkg_imports: dict[str, dict[str, str]] = defaultdict(dict)

        for n in nodes:
            d = str(PurePosixPath(n.file_path).parent)
            kind = n.node_type.value
            if kind == "function":
                self.funcs_in_file[n.file_path][n.name] = n.id
                self.funcs_in_dir[d][n.name].append(n.id)
            elif kind == "method":
                self.methods_in_file[n.file_path][n.name] = n.id
                self.methods_in_dir[d][n.name].append(n.id)
            elif kind in ("class", "struct"):
                self.classes_in_file[n.file_path][n.name] = n.id
            lang = _detect_language(n.file_path)
            if lang == "python":
                mod = _python_module_name(n.file_path)
                if kind in ("function", "class", "method"):
                    self.py_symbols[f"{mod}.{n.name}"] = n.id
                if mod:
                    self.py_top_packages.add(mod.split(".")[0])

        for f in facts.values():
            if f.language != "go":
                continue
            d = str(PurePosixPath(f.file_path).parent)
            for struct, flds in f.struct_fields.items():
                self.go_pkg_struct_fields[d].setdefault(struct, {}).update(flds)
            for alias, path in f.imports.items():
                self.go_pkg_imports[d].setdefault(alias, path)

    # -- shared helpers ---------------------------------------------------------

    def _is_shadowed(self, caller: CodeNode, name: str) -> bool:
        f = self.facts.get(caller.file_path)
        if f is None:
            return False
        return name in f.local_names.get(caller.id, ())

    # -- Go ---------------------------------------------------------------------

    def _go_internal_dir(self, f: FileFacts, import_path: str) -> str | None:
        """Map an import path inside the file's module to a root-relative directory."""
        if not f.go_module:
            return None
        if import_path == f.go_module:
            rel = ""
        elif import_path.startswith(f.go_module + "/"):
            rel = import_path[len(f.go_module) + 1 :]
        else:
            return None
        base = f.go_module_root
        combined = f"{base}/{rel}" if base and rel else (rel or base)
        return combined if combined else "."

    def resolve_go(self, caller: CodeNode, target: str) -> tuple[str, str] | str:
        """Return (node_id, resolution) or one of 'external'/'unresolved'/'skip'."""
        f = self.facts.get(caller.file_path)
        segs = target.split(".")
        pkg_dir = str(PurePosixPath(caller.file_path).parent)

        if len(segs) == 1:
            if target in _GO_BUILTINS:
                return "skip"
            if self._is_shadowed(caller, target):
                return "unresolved"
            hits = self.funcs_in_dir[pkg_dir].get(target, [])
            if len(hits) == 1:
                return hits[0], CallResolution.SAME_PACKAGE.value
            return "unresolved"

        first = segs[0]
        imp = f.imports.get(first) if f else None
        if imp is not None and not self._is_shadowed(caller, first):
            internal = self._go_internal_dir(f, imp) if f else None
            if internal is None:
                return "external"
            if len(segs) == 2:
                hits = self.funcs_in_dir[internal].get(segs[1], [])
                if len(hits) == 1:
                    return hits[0], CallResolution.IMPORT.value
            elif len(segs) == 3:
                hits = self.methods_in_dir[internal].get(f"{segs[1]}.{segs[2]}", [])
                if len(hits) == 1:
                    return hits[0], CallResolution.IMPORT.value
            return "unresolved"

        # Receiver-based: s.Method() / s.field.Method()
        if f and caller.node_type.value == "method":
            recv_type = caller.name.split(".")[0]
            recv_var = f.receiver_names.get(caller.name, "")
            if recv_var and first == recv_var:
                if len(segs) == 2:
                    hits = self.methods_in_dir[pkg_dir].get(f"{recv_type}.{segs[1]}", [])
                    if len(hits) == 1:
                        return hits[0], CallResolution.SELF.value
                    return "unresolved"
                if len(segs) == 3:
                    field_type = self.go_pkg_struct_fields[pkg_dir].get(recv_type, {}).get(segs[1])
                    if field_type:
                        resolved = self._go_resolve_type(f, pkg_dir, field_type)
                        if resolved == "external":
                            return "external"
                        if resolved is not None:
                            type_dir, type_name = resolved
                            hits = self.methods_in_dir[type_dir].get(f"{type_name}.{segs[2]}", [])
                            if len(hits) == 1:
                                return hits[0], CallResolution.FIELD_TYPE.value
                    return "unresolved"
        return "unresolved"

    def _go_resolve_type(
        self, f: FileFacts, pkg_dir: str, type_str: str
    ) -> tuple[str, str] | str | None:
        """'*content.Service' -> (dir, 'Service'); same-package type -> (pkg_dir, T)."""
        t = type_str.lstrip("*").strip()
        while t.startswith("[]"):
            t = t[2:]
        if t.startswith(("map[", "chan ", "func(", "interface{", "struct{")) or " " in t:
            return None
        t = t.split("[")[0]  # strip generic parameters
        if "." in t:
            alias, type_name = t.split(".", 1)
            imp = f.imports.get(alias) or self.go_pkg_imports[pkg_dir].get(alias)
            if not imp:
                return None
            internal = self._go_internal_dir(f, imp)
            if internal is None:
                return "external"
            return internal, type_name
        return pkg_dir, t

    # -- Python -----------------------------------------------------------------

    def _py_find_method(self, file_path: str, cls: str, meth: str, depth: int = 0) -> str | None:
        if depth > 8:
            return None
        hit = self.methods_in_file[file_path].get(f"{cls}.{meth}")
        if hit:
            return hit
        cls_id = self.classes_in_file[file_path].get(cls)
        if cls_id is None:
            return None
        f = self.facts.get(file_path)
        for base_raw in self.class_bases.get(cls_id, []):
            imp = (f.imports.get(base_raw) or f.imports.get(base_raw.split(".")[0])) if f else None
            if imp:
                full = _resolve_python_relative_import(file_path, imp)
                if base_raw != base_raw.split(".")[0] and "." in base_raw:
                    # base written as mod.Class with mod imported
                    full = full + "." + base_raw.split(".", 1)[1]
                base_id = self.py_symbols.get(full)
                if base_id is not None:
                    base_node = self.node_by_id[base_id]
                    hit = self._py_find_method(base_node.file_path, base_node.name, meth, depth + 1)
                    if hit:
                        return hit
            else:
                hit = self._py_find_method(file_path, base_raw.split(".")[-1], meth, depth + 1)
                if hit:
                    return hit
        return None

    def resolve_python(self, caller: CodeNode, target: str) -> tuple[str, str] | str:
        f = self.facts.get(caller.file_path)
        segs = target.split(".")
        first = segs[0]

        if first in ("self", "cls") and caller.node_type.value == "method":
            if len(segs) == 2:
                cls = caller.name.split(".", 1)[0]
                hit = self._py_find_method(caller.file_path, cls, segs[1])
                if hit:
                    return hit, CallResolution.SELF.value
            return "unresolved"

        if f and first in f.imports and not self._is_shadowed(caller, first):
            dotted = f.imports[first]
            full = _resolve_python_relative_import(caller.file_path, dotted)
            if len(segs) > 1:
                full = full + "." + ".".join(segs[1:])
            hit = self.py_symbols.get(full)
            if hit is not None:
                return hit, CallResolution.IMPORT.value
            if full.split(".")[0] not in self.py_top_packages:
                return "external"
            return "unresolved"

        if len(segs) == 1:
            if self._is_shadowed(caller, target):
                return "unresolved"
            hit = self.funcs_in_file[caller.file_path].get(target) or self.classes_in_file[
                caller.file_path
            ].get(target)
            if hit:
                return hit, CallResolution.SAME_MODULE.value
            if target in _PY_BUILTINS:
                return "skip"
            return "unresolved"

        # ClassName.method() in the same module
        if len(segs) == 2 and not self._is_shadowed(caller, first):
            hit = self.methods_in_file[caller.file_path].get(target)
            if hit:
                return hit, CallResolution.SAME_MODULE.value
        return "unresolved"

    # -- JavaScript / TypeScript -------------------------------------------------

    def _js_module_file(self, source_file: str, module: str) -> str | None:
        if not module.startswith("."):
            return None
        mod_id = _try_resolve_js_relative_import(source_file, module, self.module_to_id)
        if mod_id is None:
            return None
        return self.node_by_id[mod_id].file_path if mod_id in self.node_by_id else None

    def resolve_js(self, caller: CodeNode, target: str) -> tuple[str, str] | str:
        f = self.facts.get(caller.file_path)
        segs = target.replace("?.", ".").split(".")
        first = segs[0]

        if first == "this" and caller.node_type.value == "method" and len(segs) == 2:
            cls = caller.name.split(".", 1)[0]
            hit = self.methods_in_file[caller.file_path].get(f"{cls}.{segs[1]}")
            if hit:
                return hit, CallResolution.SELF.value
            return "unresolved"

        binding = f.imports.get(first) if f else None
        if binding is not None and not self._is_shadowed(caller, first):
            module, _, orig = binding.rpartition(":")
            target_file = self._js_module_file(caller.file_path, module)
            if target_file is None:
                return "external"
            if len(segs) == 1 and orig != "*":
                hit = self.funcs_in_file[target_file].get(orig) or self.classes_in_file[
                    target_file
                ].get(orig)
                if hit:
                    return hit, CallResolution.IMPORT.value
                return "unresolved"
            if len(segs) == 2 and orig == "*":
                hit = self.funcs_in_file[target_file].get(segs[1]) or self.classes_in_file[
                    target_file
                ].get(segs[1])
                if hit:
                    return hit, CallResolution.IMPORT.value
                return "unresolved"
            return "unresolved"

        if len(segs) == 1:
            if self._is_shadowed(caller, target):
                return "unresolved"
            hit = self.funcs_in_file[caller.file_path].get(target) or self.classes_in_file[
                caller.file_path
            ].get(target)
            if hit:
                return hit, CallResolution.SAME_MODULE.value
            if target in _JS_GLOBALS:
                return "skip"
            return "unresolved"

        if first in _JS_GLOBALS:
            return "skip"
        return "unresolved"

    # -- default (rust / llm-parsed / unknown) -----------------------------------

    def resolve_default(self, caller: CodeNode, target: str) -> tuple[str, str] | str:
        if "." not in target and ":" not in target:
            hit = self.funcs_in_file[caller.file_path].get(target)
            if hit:
                return hit, CallResolution.SAME_MODULE.value
        return "unresolved"

    def resolve(self, caller: CodeNode, target: str) -> tuple[str, str] | str:
        # A callee expression that isn't a plain dotted name (chained call results,
        # subscripts, literals) can't be bound syntactically — record it as-is.
        if "(" in target or "[" in target or " " in target:
            return "unresolved"
        lang = _detect_language(caller.file_path)
        if lang == "go":
            return self.resolve_go(caller, target)
        if lang == "python":
            return self.resolve_python(caller, target)
        if lang in ("javascript", "typescript"):
            return self.resolve_js(caller, target)
        return self.resolve_default(caller, target)


def resolve_edges(
    nodes: Sequence[CodeNode],
    edges: Sequence[Edge],
    facts: dict[str, FileFacts] | None = None,
) -> ResolutionResult:
    """Resolve name-based edge targets to actual node IDs where possible.

    CALLS edges run the per-language confidence ladder; a CALLS edge that cannot be
    bound is *not* emitted — the call is recorded in the caller node's
    ``external_calls``/``unresolved_calls`` instead (never a guessed edge, INV-8).
    Non-CALLS edges keep the legacy resolution behavior unchanged.
    """
    facts = facts or {}
    node_ids = {n.id for n in nodes}
    # Legacy lookup for non-CALLS edges: short name -> list of node IDs
    name_to_ids: dict[str, list[str]] = {}
    for n in nodes:
        name_to_ids.setdefault(n.name, []).append(n.id)
        # Also index by last segment (e.g. "Class.method" -> method id)
        parts = n.name.rsplit(".", 1)
        if len(parts) == 2:
            name_to_ids.setdefault(parts[-1], []).append(n.id)

    module_to_id = _build_module_index(nodes)
    python_symbol_index = _build_python_symbol_index(nodes)
    ladder = _CallLadder(nodes, facts)
    for e in edges:
        if e.edge_type == EdgeType.INHERITS and e.source in ladder.node_by_id:
            ladder.class_bases[e.source].append(e.target)

    resolved: list[Edge] = []
    seen_calls: set[tuple[str, str]] = set()
    external_calls: dict[str, set[str]] = defaultdict(set)
    unresolved_calls: dict[str, set[str]] = defaultdict(set)

    def emit_call(source: str, target: str, resolution: str, callee_text: str) -> None:
        if (source, target) in seen_calls:
            return
        seen_calls.add((source, target))
        resolved.append(
            Edge(
                source=source,
                target=target,
                edge_type=EdgeType.CALLS,
                resolution=resolution,
                callee_text=callee_text,
            )
        )

    for e in edges:
        target = e.target

        if e.edge_type == EdgeType.CALLS:
            if target in node_ids:
                emit_call(e.source, target, e.resolution or CallResolution.EXACT.value, e.callee_text or target)
                continue
            caller = ladder.node_by_id.get(e.source)
            if caller is None:
                continue
            outcome = ladder.resolve(caller, target)
            if isinstance(outcome, tuple):
                emit_call(e.source, outcome[0], outcome[1], target)
            elif outcome == "external":
                external_calls[e.source].add(target[:200])
            elif outcome == "unresolved":
                unresolved_calls[e.source].add(target[:200])
            # "skip": builtins/globals — noise, recorded nowhere
            continue

        if target in node_ids:
            resolved.append(e)
            continue

        # Try module resolution for imports
        if e.edge_type == EdgeType.IMPORTS:
            if target in module_to_id:
                resolved.append(Edge(source=e.source, target=module_to_id[target], edge_type=e.edge_type))
                continue

            source_file = e.source.split("::")[0]
            lang = _detect_language(source_file)
            if lang == "python":
                python_target = _resolve_python_relative_import(source_file, target)
                resolved_target: str | None = python_symbol_index.get(python_target)
                module_parts = python_target.split(".")
                while len(module_parts) > 1 and resolved_target is None:
                    module_parts.pop()
                    module_candidate = ".".join(module_parts)
                    resolved_target = python_symbol_index.get(module_candidate)
                if resolved_target is None:
                    resolved_target = module_to_id.get(python_target)
                if resolved_target is not None:
                    resolved.append(Edge(source=e.source, target=resolved_target, edge_type=e.edge_type))
                    continue

            # JS relative import resolution
            if lang in ("javascript", "typescript") and target.startswith("."):
                resolved_target = _try_resolve_js_relative_import(source_file, target, module_to_id)
                if resolved_target is not None:
                    resolved.append(Edge(source=e.source, target=resolved_target, edge_type=e.edge_type))
                    continue

        # Legacy name-based resolution for non-CALLS edges (INHERITS/IMPLEMENTS/…)
        candidates = name_to_ids.get(target, [])
        if len(candidates) == 1:
            resolved.append(Edge(source=e.source, target=candidates[0], edge_type=e.edge_type))
        elif len(candidates) > 1:
            # Prefer node in same file
            source_file = e.source.split("::")[0]
            same_file = [c for c in candidates if c.startswith(source_file)]
            if same_file:
                resolved.append(Edge(source=e.source, target=same_file[0], edge_type=e.edge_type))
            else:
                resolved.append(Edge(source=e.source, target=candidates[0], edge_type=e.edge_type))
        else:
            # Keep unresolved edge as-is (target stays as name string)
            resolved.append(e)

    return ResolutionResult(
        edges=resolved,
        external_calls={
            k: sorted(v)[:_MAX_RECORDED_CALLS] for k, v in external_calls.items()
        },
        unresolved_calls={
            k: sorted(v)[:_MAX_RECORDED_CALLS] for k, v in unresolved_calls.items()
        },
    )
