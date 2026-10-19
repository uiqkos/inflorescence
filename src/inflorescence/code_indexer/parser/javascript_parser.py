"""Tree-sitter based parser for JavaScript and TypeScript files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, FileFacts, NodeType
from inflorescence.code_indexer.parser.base_parser import BaseParser

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode
    from tree_sitter import Parser as TSParser

logger = logging.getLogger(__name__)

# Node type names in tree-sitter grammars
_CLASS_TYPES = {"class_declaration", "class"}
_FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
}
_METHOD_TYPES = {"method_definition"}
_INTERFACE_TYPES = {"interface_declaration"}
_ENUM_TYPES = {"enum_declaration"}
_IMPORT_TYPES = {"import_statement"}
_TYPE_ALIAS_TYPES = {"type_alias_declaration"}
# A declaration's signature is its first source line, cut here — enough to show `export const
# RULES: Rule[] = [` or `type PassFacts = {`, never a whole object literal.
_DECLARATION_SIGNATURE_CHARS = 200
# Top-level node types fully consumed by the structural handlers; anything else is a
# plain statement whose calls belong to the module node.
_HANDLED_TOP_LEVEL = (
    _CLASS_TYPES | _FUNCTION_TYPES | _INTERFACE_TYPES | _ENUM_TYPES | _IMPORT_TYPES | _TYPE_ALIAS_TYPES
    | {"lexical_declaration", "variable_declaration", "export_statement", "export_default_declaration", "comment"}
)


def _node_text(node: TSNode, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_child_by_type(node: TSNode, *types: str) -> TSNode | None:
    for child in node.children:
        if child.type in types:
            return child
    return None


def _find_children_by_type(node: TSNode, *types: str) -> list[TSNode]:
    return [c for c in node.children if c.type in types]


def _import_edge(
    mod_id: str,
    module: str,
    statement: TSNode,
    names: list[str],
    *,
    type_only: bool = False,
    reexport: bool = False,
    dynamic: bool = False,
) -> Edge:
    """An IMPORTS edge carrying the statement as written (see ``Edge.specifier``)."""
    return Edge(
        source=mod_id,
        target=module,
        edge_type=EdgeType.IMPORTS,
        specifier=module,
        line=statement.start_point[0] + 1,
        names=names,
        type_only=type_only,
        reexport=reexport,
        dynamic=dynamic,
    )


def _first_line(text: str) -> str:
    stripped = text.strip()
    line = stripped.splitlines()[0].rstrip() if stripped else ""
    if len(line) > _DECLARATION_SIGNATURE_CHARS:
        return line[: _DECLARATION_SIGNATURE_CHARS - 1] + "…"
    return line


def _get_jsdoc(node: TSNode, source_bytes: bytes) -> str:
    """Extract JSDoc comment preceding a node."""
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = _node_text(prev, source_bytes)
        if text.startswith("/**"):
            # Strip /** ... */ and leading * on each line
            lines = text.split("\n")
            cleaned: list[str] = []
            for line in lines:
                # removeprefix/removesuffix also cover the one-line form `/** Tuning. */`,
                # which used to keep its delimiters.
                line = line.strip().removeprefix("/**").removesuffix("*/").strip()
                if not line:
                    continue
                line = line.lstrip("* ").rstrip()
                if line:
                    cleaned.append(line)
            return "\n".join(cleaned)
    return ""


def _callee_name(node: TSNode, source_bytes: bytes) -> str:
    """Normalize a callee expression to a dotted name, or "" if it isn't one.

    ``api`` -> "api"; ``ns.f`` -> "ns.f"; ``this.load`` -> "this.load";
    ``x?.y`` -> "x.y". Chained-call callees (``fetch().then``), subscripts and other
    computed expressions return "" — they can't be bound syntactically and would
    otherwise pollute the graph with argument-bearing raw text.
    """
    parts: list[str] = []
    current = node
    while True:
        if current.type in ("identifier", "this", "super", "property_identifier"):
            parts.append(_node_text(current, source_bytes))
            return ".".join(reversed(parts))
        if current.type == "member_expression":
            prop = current.child_by_field_name("property")
            obj = current.child_by_field_name("object")
            if prop is None or obj is None or prop.type != "property_identifier":
                return ""
            parts.append(_node_text(prop, source_bytes))
            current = obj
            continue
        if current.type in ("non_null_expression", "parenthesized_expression"):
            inner = next((c for c in current.children if c.type not in ("(", ")", "!")), None)
            if inner is None:
                return ""
            current = inner
            continue
        return ""


def _collect_local_names(node: TSNode, source_bytes: bytes) -> list[str]:
    """Parameter and locally-declared names inside a function body (shadow guard)."""
    names: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in ("required_parameter", "optional_parameter", "formal_parameters", "variable_declarator"):
            for child in current.children:
                if child.type == "identifier":
                    names.add(_node_text(child, source_bytes))
                elif child.type in ("object_pattern", "array_pattern"):
                    sub = [child]
                    while sub:
                        p = sub.pop()
                        if p.type in ("identifier", "shorthand_property_identifier_pattern"):
                            names.add(_node_text(p, source_bytes))
                        sub.extend(p.children)
        stack.extend(current.children)
    return sorted(names)


class JavaScriptParser(BaseParser):
    """Parser for JS/TS files using tree-sitter."""

    def __init__(self) -> None:
        self._js_parser: TSParser | None = None
        self._ts_parser: TSParser | None = None
        self._tsx_parser: TSParser | None = None

    def _get_parser_and_language(self, suffix: str) -> TSParser:
        """Lazily initialize tree-sitter parsers."""
        from tree_sitter import Language, Parser  # type: ignore[import-untyped] — tree-sitter lacks stubs

        if suffix in (".ts", ".tsx"):
            if suffix == ".tsx":
                if self._tsx_parser is None:
                    import tree_sitter_typescript as ts_ts  # type: ignore[import-untyped] — no stubs

                    lang = Language(ts_ts.language_tsx())
                    p = Parser(lang)
                    self._tsx_parser = p
                assert self._tsx_parser is not None
                return self._tsx_parser
            else:
                if self._ts_parser is None:
                    import tree_sitter_typescript as ts_ts  # type: ignore[import-untyped] — no stubs

                    lang = Language(ts_ts.language_typescript())
                    p = Parser(lang)
                    self._ts_parser = p
                assert self._ts_parser is not None
                return self._ts_parser
        else:
            if self._js_parser is None:
                import tree_sitter_javascript as ts_js  # type: ignore[import-untyped] — no stubs

                lang = Language(ts_js.language())
                p = Parser(lang)
                self._js_parser = p
            assert self._js_parser is not None
            return self._js_parser

    def get_supported_extensions(self) -> list[str]:
        return [".js", ".jsx", ".ts", ".tsx"]

    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        nodes, edges, _ = self.parse_file_with_facts(file_path, root_path)
        return nodes, edges

    def parse_file_with_facts(
        self, file_path: Path, root_path: Path
    ) -> tuple[list[CodeNode], list[Edge], FileFacts | None]:
        rel = str(file_path.relative_to(root_path))
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.debug("Failed to read %s, skipping", file_path)
            return [], [], None

        source_bytes = source.encode("utf-8")
        parser = self._get_parser_and_language(file_path.suffix)
        tree = parser.parse(source_bytes)

        lines = source.splitlines()
        nodes: list[CodeNode] = []
        edges: list[Edge] = []
        facts = FileFacts(
            file_path=rel,
            language="typescript" if file_path.suffix in (".ts", ".tsx") else "javascript",
        )

        # Module node
        mod_id = rel
        nodes.append(
            CodeNode(
                id=mod_id,
                name=file_path.stem,
                node_type=NodeType.MODULE,
                file_path=rel,
                line_start=1,
                line_end=len(lines),
            )
        )

        self._walk_top_level(tree.root_node, mod_id, rel, source_bytes, nodes, edges, facts)
        self._dynamic_imports(tree.root_node, mod_id, source_bytes, edges)
        return nodes, edges, facts

    def _walk_top_level(
        self,
        root: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
        facts: FileFacts,
    ) -> None:
        for child in root.children:
            # Handle export wrappers
            actual = child
            if child.type in ("export_statement", "export_default_declaration"):
                # The actual declaration is a child of the export
                decl = _find_child_by_type(
                    child,
                    "class_declaration",
                    "class",
                    "function_declaration",
                    "generator_function_declaration",
                    "interface_declaration",
                    "enum_declaration",
                    "lexical_declaration",
                    "variable_declaration",
                    "type_alias_declaration",
                )
                actual = decl or child

            if actual.type in _CLASS_TYPES:
                self._handle_class(actual, mod_id, rel, source_bytes, nodes, edges, facts)
            elif actual.type in _FUNCTION_TYPES:
                self._handle_function(actual, mod_id, rel, source_bytes, nodes, edges, facts)
            elif actual.type in _INTERFACE_TYPES:
                self._handle_interface(actual, mod_id, rel, source_bytes, nodes, edges)
            elif actual.type in _ENUM_TYPES:
                self._handle_enum(actual, mod_id, rel, source_bytes, nodes, edges)
            elif actual.type in ("lexical_declaration", "variable_declaration"):
                # const Foo = () => {}, const utils = require('./utils'), var x = f()
                self._handle_var_decl(actual, mod_id, rel, source_bytes, nodes, edges, facts, outer=child)
            elif actual.type in _TYPE_ALIAS_TYPES:
                self._handle_type_alias(actual, child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type in _IMPORT_TYPES:
                self._handle_import(child, mod_id, source_bytes, edges, facts)
            elif child.type == "export_statement" and actual is child:
                self._handle_reexport(child, mod_id, source_bytes, edges)

            if child.type not in _HANDLED_TOP_LEVEL and actual is child:
                # Plain top-level statement (expression, if, for, describe(...) blocks in
                # tests): its calls belong to the module node — a file whose whole body is
                # top-level code must not read as "calls nothing" (honest call accounting, docs/calls-resolution.md).
                self._extract_calls(child, mod_id, source_bytes, edges)

    def _handle_class(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
        facts: FileFacts,
    ) -> None:
        name_node = _find_child_by_type(node, "type_identifier", "identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        cls_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        nodes.append(
            CodeNode(
                id=cls_id,
                name=name,
                node_type=NodeType.CLASS,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                docstring=doc,
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=cls_id, edge_type=EdgeType.CONTAINS))

        # Heritage: extends / implements
        heritage = _find_child_by_type(node, "class_heritage")
        if heritage:
            self._handle_heritage(heritage, cls_id, source_bytes, edges)

        # Methods
        body = _find_child_by_type(node, "class_body")
        if body:
            for item in body.children:
                if item.type in _METHOD_TYPES:
                    self._handle_method(item, cls_id, name, rel, source_bytes, nodes, edges, facts)

    def _handle_heritage(
        self,
        heritage: TSNode,
        cls_id: str,
        source_bytes: bytes,
        edges: list[Edge],
    ) -> None:
        """Parse class heritage (extends/implements) clauses."""
        # JS grammar: heritage children are keyword tokens and identifiers directly
        # e.g. [extends, identifier] or [extends, identifier, implements, type_identifier, ...]
        mode: str | None = None
        for clause in heritage.children:
            if clause.type == "extends":
                mode = "extends"
            elif clause.type == "implements":
                mode = "implements"
            elif clause.type == "extends_clause":
                base = _find_child_by_type(clause, "identifier", "member_expression", "type_identifier")
                if base:
                    edges.append(
                        Edge(source=cls_id, target=_node_text(base, source_bytes), edge_type=EdgeType.INHERITS)
                    )
            elif clause.type == "implements_clause":
                for iface in _find_children_by_type(clause, "type_identifier", "generic_type"):
                    name_n = (
                        iface if iface.type == "type_identifier" else _find_child_by_type(iface, "type_identifier")
                    )
                    if name_n:
                        edges.append(
                            Edge(
                                source=cls_id,
                                target=_node_text(name_n, source_bytes),
                                edge_type=EdgeType.IMPLEMENTS,
                            )
                        )
            elif clause.type in ("identifier", "member_expression", "type_identifier"):
                if mode == "extends":
                    edges.append(
                        Edge(source=cls_id, target=_node_text(clause, source_bytes), edge_type=EdgeType.INHERITS)
                    )
                elif mode == "implements":
                    edges.append(
                        Edge(source=cls_id, target=_node_text(clause, source_bytes), edge_type=EdgeType.IMPLEMENTS)
                    )

    def _handle_method(
        self,
        node: TSNode,
        cls_id: str,
        cls_name: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
        facts: FileFacts,
    ) -> None:
        name_node = _find_child_by_type(node, "property_identifier", "computed_property_name")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        meth_id = f"{rel}::{cls_name}.{name}"
        doc = _get_jsdoc(node, source_bytes)

        # Build signature
        params_node = _find_child_by_type(node, "formal_parameters")
        sig = f"{name}({_node_text(params_node, source_bytes) if params_node else ''})"

        nodes.append(
            CodeNode(
                id=meth_id,
                name=f"{cls_name}.{name}",
                node_type=NodeType.METHOD,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=doc,
                parent_id=cls_id,
            )
        )
        edges.append(Edge(source=cls_id, target=meth_id, edge_type=EdgeType.CONTAINS))
        facts.local_names[meth_id] = _collect_local_names(node, source_bytes)
        self._extract_calls(node, meth_id, source_bytes, edges)

    def _handle_function(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
        facts: FileFacts,
    ) -> None:
        name_node = _find_child_by_type(node, "identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        func_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        params_node = _find_child_by_type(node, "formal_parameters")
        sig = f"function {name}({_node_text(params_node, source_bytes) if params_node else ''})"

        nodes.append(
            CodeNode(
                id=func_id,
                name=name,
                node_type=NodeType.FUNCTION,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=doc,
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=func_id, edge_type=EdgeType.CONTAINS))
        facts.local_names[func_id] = _collect_local_names(node, source_bytes)
        self._extract_calls(node, func_id, source_bytes, edges)

    def _handle_type_alias(
        self,
        node: TSNode,
        outer: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        """`type PassFacts = {...}` / `export type Mode = 'a' | 'b'` -> a TYPE_ALIAS node."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return
        name = _node_text(name_node, source_bytes)
        alias_id = f"{rel}::{name}"
        if any(existing.id == alias_id for existing in nodes):
            return
        nodes.append(
            CodeNode(
                id=alias_id,
                name=name,
                node_type=NodeType.TYPE_ALIAS,
                file_path=rel,
                line_start=outer.start_point[0] + 1,
                line_end=outer.end_point[0] + 1,
                signature=_first_line(_node_text(outer, source_bytes)),
                docstring=_get_jsdoc(outer, source_bytes),
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=alias_id, edge_type=EdgeType.CONTAINS))

    def _handle_interface(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        name_node = _find_child_by_type(node, "type_identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        iface_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        nodes.append(
            CodeNode(
                id=iface_id,
                name=name,
                node_type=NodeType.INTERFACE,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                docstring=doc,
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=iface_id, edge_type=EdgeType.CONTAINS))

        # extends clause
        extends = _find_child_by_type(node, "extends_type_clause")
        if extends:
            for tid in _find_children_by_type(extends, "type_identifier"):
                edges.append(
                    Edge(source=iface_id, target=_node_text(tid, source_bytes), edge_type=EdgeType.INHERITS)
                )

    def _handle_enum(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        name_node = _find_child_by_type(node, "identifier")
        name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
        enum_id = f"{rel}::{name}"
        doc = _get_jsdoc(node, source_bytes)

        nodes.append(
            CodeNode(
                id=enum_id,
                name=name,
                node_type=NodeType.ENUM,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                docstring=doc,
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=enum_id, edge_type=EdgeType.CONTAINS))

    def _handle_var_decl(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
        facts: FileFacts,
        outer: TSNode | None = None,
    ) -> None:
        """Handle `const foo = () => {}`, `const utils = require('./utils')` and friends.

        Any other top-level binding (`export const RULES = [...]`, `let cache = new Map()`)
        becomes a VARIABLE node: its first line as signature, a preceding JSDoc as docstring.
        *outer* is the statement as written — the `export` wrapper when there is one — so the
        span and the signature include the `export` keyword.
        """
        outer = outer or node
        declarators = _find_children_by_type(node, "variable_declarator")
        keyword = node.children[0].type if node.children else "const"
        exported = "export " if outer is not node else ""
        handled_declarators = 0
        for decl in declarators:
            name_node = _find_child_by_type(decl, "identifier")
            value_node = _find_child_by_type(decl, "arrow_function", "function_expression", "function")
            if name_node and value_node:
                handled_declarators += 1
                name = _node_text(name_node, source_bytes)
                func_id = f"{rel}::{name}"
                doc = _get_jsdoc(node, source_bytes)

                params_node = _find_child_by_type(value_node, "formal_parameters")
                sig = f"const {name} = ({_node_text(params_node, source_bytes) if params_node else ''}) =>"

                nodes.append(
                    CodeNode(
                        id=func_id,
                        name=name,
                        node_type=NodeType.FUNCTION,
                        file_path=rel,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=sig,
                        docstring=doc,
                        parent_id=mod_id,
                    )
                )
                edges.append(Edge(source=mod_id, target=func_id, edge_type=EdgeType.CONTAINS))
                facts.local_names[func_id] = _collect_local_names(value_node, source_bytes)
                self._extract_calls(value_node, func_id, source_bytes, edges)
                continue

            if self._record_require(decl, mod_id, source_bytes, edges, facts):
                handled_declarators += 1
                continue

            binding = decl.child_by_field_name("name")
            if binding is None or binding.type != "identifier":
                continue  # a destructuring pattern binds several names; none is "the" name
            name = _node_text(binding, source_bytes)
            var_id = f"{rel}::{name}"
            if any(existing.id == var_id for existing in nodes):
                continue  # `const X` next to `type X` (TypeScript declaration merging): keep the first
            span = outer if len(declarators) == 1 else decl
            nodes.append(
                CodeNode(
                    id=var_id,
                    name=name,
                    node_type=NodeType.VARIABLE,
                    file_path=rel,
                    line_start=span.start_point[0] + 1,
                    line_end=span.end_point[0] + 1,
                    signature=f"{exported}{keyword} {_first_line(_node_text(decl, source_bytes))}",
                    docstring=_get_jsdoc(outer, source_bytes),
                    parent_id=mod_id,
                )
            )
            edges.append(Edge(source=mod_id, target=var_id, edge_type=EdgeType.CONTAINS))

        if handled_declarators == 0:
            # Initializers with plain calls (`const store = createStore()`) belong to the
            # module node.
            self._extract_calls(node, mod_id, source_bytes, edges)

    def _record_require(
        self, decl: TSNode, mod_id: str, source_bytes: bytes, edges: list[Edge], facts: FileFacts
    ) -> bool:
        """Record `const x = require('m')` / `const {a, b: c} = require('m')` bindings."""
        call = _find_child_by_type(decl, "call_expression")
        if call is None:
            return False
        fn = call.children[0] if call.children else None
        if fn is None or fn.type != "identifier" or _node_text(fn, source_bytes) != "require":
            return False
        args = _find_child_by_type(call, "arguments")
        s = _find_child_by_type(args, "string") if args else None
        if s is None:
            return False
        module = _node_text(s, source_bytes).strip("'\"")
        name_node = _find_child_by_type(decl, "identifier", "object_pattern")
        names: list[str] = []
        if name_node is not None and name_node.type == "object_pattern":
            for pair in name_node.children:
                if pair.type == "shorthand_property_identifier_pattern":
                    names.append(_node_text(pair, source_bytes))
                elif pair.type == "pair_pattern" and pair.children:
                    names.append(_node_text(pair.children[0], source_bytes))
        elif name_node is not None:
            names.append("*")
        edges.append(_import_edge(mod_id, module, call, names))
        if name_node is None:
            return True
        if name_node.type == "identifier":
            facts.imports[_node_text(name_node, source_bytes)] = f"{module}:*"
        else:
            for pair in name_node.children:
                if pair.type == "shorthand_property_identifier_pattern":
                    local = _node_text(pair, source_bytes)
                    facts.imports[local] = f"{module}:{local}"
                elif pair.type == "pair_pattern" and len(pair.children) >= 2:
                    key = _node_text(pair.children[0], source_bytes)
                    value = _node_text(pair.children[-1], source_bytes)
                    facts.imports[value] = f"{module}:{key}"
        return True

    def _handle_import(
        self, node: TSNode, mod_id: str, source_bytes: bytes, edges: list[Edge], facts: FileFacts
    ) -> None:
        source_node = _find_child_by_type(node, "string")
        if not source_node:
            return
        module = _node_text(source_node, source_bytes).strip("'\"")
        clause = _find_child_by_type(node, "import_clause")
        # `import type {…}` marks the whole statement; `import {type A, B}` marks specifiers.
        # A mixed statement becomes two edges — one type-only, one not — so a symbol-level
        # edge built from either (resolver) carries the right flag for its own name.
        whole_type_only = _find_child_by_type(node, "type") is not None
        value_names: list[str] = []
        type_names: list[str] = []
        for c in clause.children if clause is not None else ():
            if c.type == "identifier":
                value_names.append("default")
            elif c.type == "namespace_import":
                value_names.append("*")
            elif c.type == "named_imports":
                for spec in _find_children_by_type(c, "import_specifier"):
                    idents = [x for x in spec.children if x.type == "identifier"]
                    if not idents:
                        continue
                    exported = _node_text(idents[0], source_bytes)
                    spec_type_only = _find_child_by_type(spec, "type") is not None
                    (type_names if spec_type_only else value_names).append(exported)
        if whole_type_only:
            type_names, value_names = type_names + value_names, []
        if value_names or not type_names:  # a bare `import './styles.css'` is a value import too
            edges.append(_import_edge(mod_id, module, node, value_names))
        if type_names:
            edges.append(_import_edge(mod_id, module, node, type_names, type_only=True))
        if clause is None:
            return
        for c in clause.children:
            if c.type == "identifier":
                facts.imports[_node_text(c, source_bytes)] = f"{module}:default"
            elif c.type == "namespace_import":
                ident = _find_child_by_type(c, "identifier")
                if ident is not None:
                    facts.imports[_node_text(ident, source_bytes)] = f"{module}:*"
            elif c.type == "named_imports":
                for spec in _find_children_by_type(c, "import_specifier"):
                    idents = [x for x in spec.children if x.type == "identifier"]
                    if len(idents) == 1:
                        local = _node_text(idents[0], source_bytes)
                        facts.imports[local] = f"{module}:{local}"
                    elif len(idents) >= 2:
                        orig = _node_text(idents[0], source_bytes)
                        local = _node_text(idents[-1], source_bytes)
                        facts.imports[local] = f"{module}:{orig}"

    def _handle_reexport(self, node: TSNode, mod_id: str, source_bytes: bytes, edges: list[Edge]) -> None:
        """`export {a, b as c} from 'm'`, `export * from 'm'`, `export * as ns from 'm'`.

        A barrel file is a dependency like any import — a bundle that loads it loads what it
        re-exports — and used to produce no edge at all.
        """
        source_node = _find_child_by_type(node, "string")
        if source_node is None:
            return  # `export { a, b }` of local names: not a module dependency
        module = _node_text(source_node, source_bytes).strip("'\"")
        names: list[str] = []
        clause = _find_child_by_type(node, "export_clause")
        if clause is not None:
            for spec in _find_children_by_type(clause, "export_specifier"):
                idents = [x for x in spec.children if x.type == "identifier"]
                if idents:
                    names.append(_node_text(idents[0], source_bytes))
        else:
            names.append("*")
        type_only = _find_child_by_type(node, "type") is not None
        edges.append(_import_edge(mod_id, module, node, names, type_only=type_only, reexport=True))

    def _dynamic_imports(self, root: TSNode, mod_id: str, source_bytes: bytes, edges: list[Edge]) -> None:
        """`import('./BoardHost')` anywhere in the file, as a dynamic IMPORTS edge of the module.

        A lazy import is a different dependency from a static one — the code lands in its own
        chunk — and "does the production bundle statically reach X" turns on exactly that
        difference. Only string-literal specifiers: a computed one names no module.
        """
        stack = [root]
        while stack:
            current = stack.pop()
            if current.type == "call_expression" and current.children and current.children[0].type == "import":
                args = _find_child_by_type(current, "arguments")
                literal = _find_child_by_type(args, "string") if args is not None else None
                if literal is not None:
                    module = _node_text(literal, source_bytes).strip("'\"")
                    edges.append(_import_edge(mod_id, module, current, [], dynamic=True))
            stack.extend(reversed(current.children))

    def _extract_calls(
        self, node: TSNode, caller_id: str, source_bytes: bytes, edges: list[Edge]
    ) -> None:
        """Walk subtree to find call expressions."""
        cursor = node.walk()
        reached_root = False
        while not reached_root:
            current = cursor.node
            if current is None:
                break
            if current.type == "call_expression":
                func_node = current.children[0] if current.children else None
                if func_node:
                    name = _callee_name(func_node, source_bytes)
                    if name:
                        edges.append(Edge(source=caller_id, target=name, edge_type=EdgeType.CALLS))

            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue
            while True:
                if not cursor.goto_parent():
                    reached_root = True
                    break
                if cursor.node == node:
                    reached_root = True
                    break
                if cursor.goto_next_sibling():
                    break
