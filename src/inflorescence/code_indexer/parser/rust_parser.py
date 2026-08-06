"""Tree-sitter based parser for Rust files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, NodeType
from inflorescence.code_indexer.parser.base_parser import BaseParser

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode
    from tree_sitter import Parser as TSParser

logger = logging.getLogger(__name__)


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
    """Extract Rust doc comments (/// and //!) preceding a node."""
    comments: list[str] = []
    prev = node.prev_named_sibling
    while prev and prev.type == "line_comment":
        text = _node_text(prev, source_bytes)
        if text.startswith("///") or text.startswith("//!"):
            line = text.lstrip("/!").strip()
            comments.insert(0, line)
            prev = prev.prev_named_sibling
        else:
            break
    return "\n".join(comments)


class RustParser(BaseParser):
    """Parser for Rust files using tree-sitter."""

    def __init__(self) -> None:
        self._parser: TSParser | None = None

    def _get_parser(self) -> TSParser:
        if self._parser is None:
            import tree_sitter_rust as ts_rust  # type: ignore[import-untyped] — no stubs
            from tree_sitter import Language, Parser  # type: ignore[import-untyped] — tree-sitter lacks stubs

            lang = Language(ts_rust.language())
            self._parser = Parser(lang)
        assert self._parser is not None
        return self._parser

    def get_supported_extensions(self) -> list[str]:
        return [".rs"]

    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        rel = str(file_path.relative_to(root_path))
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            logger.debug("Failed to read %s, skipping", file_path)
            return [], []

        source_bytes = source.encode("utf-8")
        parser = self._get_parser()
        tree = parser.parse(source_bytes)

        lines = source.splitlines()
        nodes: list[CodeNode] = []
        edges: list[Edge] = []

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
            if child.type == "function_item":
                self._handle_function(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "struct_item":
                self._handle_struct(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "enum_item":
                self._handle_enum(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "trait_item":
                self._handle_trait(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "impl_item":
                self._handle_impl(child, mod_id, rel, source_bytes, nodes, edges)
            elif child.type == "use_declaration":
                self._handle_use(child, mod_id, source_bytes, edges)

        return nodes, edges

    def _handle_function(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        name_node = _find_child_by_type(node, "identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)
        func_id = f"{rel}::{name}"
        doc = _get_doc_comment(node, source_bytes)

        params = _find_child_by_type(node, "parameters")
        sig = f"fn {name}({_node_text(params, source_bytes) if params else ''})"
        # Find return type after ->
        for i, child in enumerate(node.children):
            if _node_text(child, source_bytes) == "->":
                if i + 1 < len(node.children):
                    sig += f" -> {_node_text(node.children[i + 1], source_bytes)}"
                break

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
        self._extract_calls(node, func_id, source_bytes, edges)

    def _handle_struct(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        name_node = _find_child_by_type(node, "type_identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)
        struct_id = f"{rel}::{name}"
        doc = _get_doc_comment(node, source_bytes)

        nodes.append(
            CodeNode(
                id=struct_id,
                name=name,
                node_type=NodeType.STRUCT,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                docstring=doc,
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=struct_id, edge_type=EdgeType.CONTAINS))

    def _handle_enum(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        name_node = _find_child_by_type(node, "type_identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)
        enum_id = f"{rel}::{name}"
        doc = _get_doc_comment(node, source_bytes)

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

    def _handle_trait(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        name_node = _find_child_by_type(node, "type_identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)
        trait_id = f"{rel}::{name}"
        doc = _get_doc_comment(node, source_bytes)

        nodes.append(
            CodeNode(
                id=trait_id,
                name=name,
                node_type=NodeType.TRAIT,
                file_path=rel,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                docstring=doc,
                parent_id=mod_id,
            )
        )
        edges.append(Edge(source=mod_id, target=trait_id, edge_type=EdgeType.CONTAINS))

    def _handle_impl(
        self,
        node: TSNode,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        """Handle impl blocks: `impl Type { ... }` or `impl Trait for Type { ... }`."""
        # Determine struct name and optional trait
        type_ids = _find_children_by_type(node, "type_identifier", "scoped_type_identifier", "generic_type")

        # Check for `impl Trait for Type`
        has_for = any(_node_text(c, source_bytes) == "for" for c in node.children)

        trait_name: str | None = None
        struct_name: str | None = None
        if has_for and len(type_ids) >= 2:
            trait_name = _node_text(type_ids[0], source_bytes)
            struct_name = _node_text(type_ids[1], source_bytes)
        elif type_ids:
            struct_name = _node_text(type_ids[0], source_bytes)

        if not struct_name:
            return

        # Strip generic params for ID
        base_struct = struct_name.split("<")[0]

        if trait_name:
            base_trait = trait_name.split("<")[0]
            edges.append(
                Edge(source=f"{rel}::{base_struct}", target=base_trait, edge_type=EdgeType.IMPLEMENTS)
            )

        # Methods inside the impl block
        body = _find_child_by_type(node, "declaration_list")
        if body:
            for item in body.children:
                if item.type == "function_item":
                    self._handle_impl_method(item, base_struct, mod_id, rel, source_bytes, nodes, edges)

    def _handle_impl_method(
        self,
        node: TSNode,
        struct_name: str,
        mod_id: str,
        rel: str,
        source_bytes: bytes,
        nodes: list[CodeNode],
        edges: list[Edge],
    ) -> None:
        name_node = _find_child_by_type(node, "identifier")
        if not name_node:
            return
        name = _node_text(name_node, source_bytes)
        meth_id = f"{rel}::{struct_name}.{name}"
        doc = _get_doc_comment(node, source_bytes)

        params = _find_child_by_type(node, "parameters")
        sig = f"fn {struct_name}::{name}({_node_text(params, source_bytes) if params else ''})"
        for i, child in enumerate(node.children):
            if _node_text(child, source_bytes) == "->":
                if i + 1 < len(node.children):
                    sig += f" -> {_node_text(node.children[i + 1], source_bytes)}"
                break

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
        self._extract_calls(node, meth_id, source_bytes, edges)

    def _handle_use(
        self, node: TSNode, mod_id: str, source_bytes: bytes, edges: list[Edge]
    ) -> None:
        # Extract the full use path
        arg = _find_child_by_type(
            node, "use_as_clause", "scoped_identifier", "use_wildcard", "scoped_use_list", "identifier"
        )
        if arg:
            target = _node_text(arg, source_bytes).rstrip(";").strip()
            if target:
                edges.append(Edge(source=mod_id, target=target, edge_type=EdgeType.IMPORTS))

    def _extract_calls(
        self, node: TSNode, caller_id: str, source_bytes: bytes, edges: list[Edge]
    ) -> None:
        cursor = node.walk()
        reached_root = False
        while not reached_root:
            current = cursor.node
            if current.type == "call_expression":
                func_node = _find_child_by_type(current, "identifier", "scoped_identifier", "field_expression")
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
