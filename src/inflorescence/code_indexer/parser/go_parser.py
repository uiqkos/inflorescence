"""Tree-sitter based parser for Go files."""

from __future__ import annotations

import contextlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, FileFacts, NodeType
from inflorescence.code_indexer.parser.base_parser import BaseParser

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode
    from tree_sitter import Parser as TSParser

logger = logging.getLogger(__name__)

_GO_MODULE_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)


def _node_text(node: TSNode, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _find_child_by_type(node: TSNode, *types: str) -> TSNode | None:
    for child in node.children:
        if child.type in types:
            return child
    return None


def _find_children_by_type(node: TSNode, *types: str) -> list[TSNode]:
    return [c for c in node.children if c.type in types]


def _get_doc_comment(node: TSNode, source_bytes: bytes) -> str:
    """Extract Go doc comment (consecutive // lines before a declaration)."""
    comments: list[str] = []
    prev = node.prev_named_sibling
    while prev and prev.type == "comment":
        text = _node_text(prev, source_bytes).lstrip("/ ").rstrip()
        comments.insert(0, text)
        prev = prev.prev_named_sibling
    return "\n".join(comments)


def _collect_local_names(node: TSNode, source_bytes: bytes) -> list[str]:
    """Names declared inside a function body (params, :=, var/const, range vars).

    The resolver uses these as a shadow guard: a bare call to a name declared locally
    must not bind to a same-package function of the same name.
    """
    names: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "parameter_declaration":
            for ident in _find_children_by_type(current, "identifier"):
                names.add(_node_text(ident, source_bytes))
        elif current.type == "short_var_declaration":
            left = current.children[0] if current.children else None
            if left is not None and left.type == "expression_list":
                for ident in _find_children_by_type(left, "identifier"):
                    names.add(_node_text(ident, source_bytes))
        elif current.type in ("var_spec", "const_spec"):
            for ident in _find_children_by_type(current, "identifier"):
                names.add(_node_text(ident, source_bytes))
        elif current.type == "range_clause":
            left = current.children[0] if current.children else None
            if left is not None and left.type == "expression_list":
                for ident in _find_children_by_type(left, "identifier"):
                    names.add(_node_text(ident, source_bytes))
        stack.extend(current.children)
    return sorted(names)


class GoParser(BaseParser):
    """Parser for Go files using tree-sitter."""

    def __init__(self) -> None:
        self._parser: TSParser | None = None
        # go.mod lookups walk parent dirs; cache per directory to keep parsing O(files).
        self._gomod_cache: dict[Path, tuple[str, str]] = {}

    def _get_parser(self) -> TSParser:
        if self._parser is None:
            import tree_sitter_go as ts_go  # type: ignore[import-untyped] — no stubs
            from tree_sitter import Language, Parser  # type: ignore[import-untyped] — tree-sitter lacks stubs

            lang = Language(ts_go.language())
            self._parser = Parser(lang)
        assert self._parser is not None
        return self._parser

    def get_supported_extensions(self) -> list[str]:
        return [".go"]

    def _find_go_module(self, file_dir: Path, root: Path) -> tuple[str, str]:
        """Return (module path, root-relative module dir) for the go.mod owning *file_dir*."""
        if file_dir in self._gomod_cache:
            return self._gomod_cache[file_dir]
        current = file_dir
        result = ("", "")
        while True:
            gomod = current / "go.mod"
            if gomod.is_file():
                try:
                    match = _GO_MODULE_RE.search(gomod.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    match = None
                if match:
                    rel = "" if current == root else str(current.relative_to(root))
                    result = (match.group(1), rel)
                break
            if current == root or current.parent == current:
                break
            current = current.parent
        self._gomod_cache[file_dir] = result
        return result

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
        parser = self._get_parser()
        tree = parser.parse(source_bytes)

        lines = source.splitlines()
        nodes: list[CodeNode] = []
        edges: list[Edge] = []
        facts = FileFacts(file_path=rel, language="go")
        # A ValueError (file outside root — symlink edge case) leaves facts import-only.
        with contextlib.suppress(ValueError):
            facts.go_module, facts.go_module_root = self._find_go_module(file_path.parent, root_path.resolve())

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

        for child in tree.root_node.children:
            if child.type == "function_declaration":
                self._handle_function(child, mod_id, rel, source_bytes, nodes, edges, facts)
            elif child.type == "method_declaration":
                self._handle_method(child, mod_id, rel, source_bytes, nodes, edges, facts)
            elif child.type == "type_declaration":
                self._handle_type_decl(child, mod_id, rel, source_bytes, nodes, edges, facts)
            elif child.type == "import_declaration":
                self._handle_import(child, mod_id, source_bytes, edges, facts)
            elif child.type in ("var_declaration", "const_declaration"):
                # Top-level initializers can call functions; attribute them to the module
                # node so a scripty file never reads as "calls nothing".
                self._extract_calls(child, mod_id, source_bytes, edges)

        return nodes, edges, facts

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
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)
        func_id = f"{rel}::{name}"
        doc = _get_doc_comment(node, source_bytes)

        params = _find_child_by_type(node, "parameter_list")
        result = _find_child_by_type(node, "result")
        sig = f"func {name}({_node_text(params, source_bytes) if params else ''})"
        if result:
            sig += f" {_node_text(result, source_bytes)}"

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

    def _handle_method(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
        facts: FileFacts,
    ) -> None:
        name_node = _find_child_by_type(node, "field_identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)

        # Get receiver type and variable name
        receiver = _find_child_by_type(node, "parameter_list")
        receiver_type = ""
        receiver_var = ""
        if receiver:
            # The receiver param list contains (name Type) or (*Type)
            for param in receiver.children:
                if param.type == "parameter_declaration":
                    type_node = _find_child_by_type(param, "type_identifier", "pointer_type", "generic_type")
                    if type_node:
                        receiver_type = _node_text(type_node, source_bytes).lstrip("*").split("[")[0]
                    var_node = _find_child_by_type(param, "identifier")
                    if var_node:
                        receiver_var = _node_text(var_node, source_bytes)
                    break

        struct_name = receiver_type or "<unknown>"
        meth_id = f"{rel}::{struct_name}.{name}"
        doc = _get_doc_comment(node, source_bytes)

        # params is the second parameter_list (after receiver)
        param_lists = _find_children_by_type(node, "parameter_list")
        params = param_lists[1] if len(param_lists) > 1 else None
        result = _find_child_by_type(node, "result")
        sig = f"func ({struct_name}) {name}({_node_text(params, source_bytes) if params else ''})"
        if result:
            sig += f" {_node_text(result, source_bytes)}"

        nodes.append(
            CodeNode(
                id=meth_id,
                name=f"{struct_name}.{name}",
                node_type=NodeType.METHOD,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                signature=sig,
                docstring=doc,
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=meth_id, edge_type=EdgeType.CONTAINS))
        if receiver_var:
            facts.receiver_names[f"{struct_name}.{name}"] = receiver_var
        facts.local_names[meth_id] = _collect_local_names(node, source_bytes)
        self._extract_calls(node, meth_id, source_bytes, edges)

    def _handle_type_decl(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
        facts: FileFacts,
    ) -> None:
        for spec in _find_children_by_type(node, "type_spec"):
            name_node = _find_child_by_type(spec, "type_identifier")
            if not name_node:
                continue
            name = _node_text(name_node, source_bytes)
            doc = _get_doc_comment(node, source_bytes)

            # Determine if struct or interface
            struct_type = _find_child_by_type(spec, "struct_type")
            iface_type = _find_child_by_type(spec, "interface_type")

            if struct_type:
                type_id = f"{rel}::{name}"
                nodes.append(
                    CodeNode(
                        id=type_id,
                        name=name,
                        node_type=NodeType.STRUCT,
                        file_path=rel,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        docstring=doc,
                        parent_id=mod_id,
                    )
                )
                edges.append(Edge(source=mod_id, target=type_id, edge_type=EdgeType.CONTAINS))

                field_list = _find_child_by_type(struct_type, "field_declaration_list")
                if field_list:
                    field_types: dict[str, str] = {}
                    for field in _find_children_by_type(field_list, "field_declaration"):
                        # Embedded field has no name, just a type -> INHERITS
                        children = [c for c in field.children if c.type != "comment"]
                        field_names = _find_children_by_type(field, "field_identifier")
                        if not field_names and len(children) >= 1 and children[0].type in (
                            "type_identifier",
                            "qualified_type",
                        ):
                            embedded = _node_text(children[0], source_bytes)
                            edges.append(Edge(source=type_id, target=embedded, edge_type=EdgeType.INHERITS))
                            continue
                        # Named field(s): record the declared type for field-type resolution
                        type_node = next(
                            (
                                c
                                for c in children
                                if c.type
                                not in ("field_identifier", ",", "raw_string_literal", "interpreted_string_literal")
                            ),
                            None,
                        )
                        if type_node is not None:
                            type_text = _node_text(type_node, source_bytes)
                            for fn in field_names:
                                field_types[_node_text(fn, source_bytes)] = type_text
                    if field_types:
                        facts.struct_fields[name] = field_types

            elif iface_type:
                type_id = f"{rel}::{name}"
                nodes.append(
                    CodeNode(
                        id=type_id,
                        name=name,
                        node_type=NodeType.INTERFACE,
                        file_path=rel,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        docstring=doc,
                        parent_id=mod_id,
                    )
                )
                edges.append(Edge(source=mod_id, target=type_id, edge_type=EdgeType.CONTAINS))

                # Embedded interfaces: look for type_elem children containing type_identifier
                for iface_child in iface_type.children:
                    if iface_child.type == "type_elem":
                        tid = _find_child_by_type(iface_child, "type_identifier")
                        if tid:
                            edges.append(
                                Edge(source=type_id, target=_node_text(tid, source_bytes), edge_type=EdgeType.INHERITS)
                            )
                    elif iface_child.type == "type_identifier":
                        edges.append(
                            Edge(
                                source=type_id,
                                target=_node_text(iface_child, source_bytes),
                                edge_type=EdgeType.INHERITS,
                            )
                        )

    def _handle_import(
        self, node: TSNode, mod_id: str, source_bytes: bytes, edges: list[Edge], facts: FileFacts
    ) -> None:
        def record(spec: TSNode) -> None:
            path_node = _find_child_by_type(spec, "interpreted_string_literal")
            if not path_node:
                return
            target = _node_text(path_node, source_bytes).strip('"')
            edges.append(Edge(source=mod_id, target=target, edge_type=EdgeType.IMPORTS))
            alias_node = _find_child_by_type(spec, "package_identifier")
            if alias_node is not None:
                alias = _node_text(alias_node, source_bytes)
            elif _find_child_by_type(spec, "dot", "blank_identifier") is not None:
                return  # dot/blank imports create no usable binding
            else:
                alias = target.rsplit("/", 1)[-1]
            facts.imports[alias] = target

        # Single import or import block
        for spec in node.children:
            if spec.type == "import_spec":
                record(spec)
            elif spec.type == "import_spec_list":
                for imp in _find_children_by_type(spec, "import_spec"):
                    record(imp)

    def _extract_calls(
        self, node: TSNode, caller_id: str, source_bytes: bytes, edges: list[Edge]
    ) -> None:
        cursor = node.walk()
        reached_root = False
        while not reached_root:
            current = cursor.node
            if current is None:
                break
            if current.type == "call_expression":
                func_node = _find_child_by_type(current, "identifier", "selector_expression", "field_identifier")
                if func_node:
                    name = _node_text(func_node, source_bytes)
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
