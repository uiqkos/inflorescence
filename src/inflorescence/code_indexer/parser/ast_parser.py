"""AST-based extraction of code nodes and edges from a Python file."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from inflorescence.code_indexer.models import CodeNode, Edge, EdgeType, FileFacts, NodeType
from inflorescence.code_indexer.parser.base_parser import BaseParser

logger = logging.getLogger(__name__)


class PythonAstParser(BaseParser):
    """Parser for Python files using the built-in ast module."""

    def get_supported_extensions(self) -> list[str]:
        return [".py"]

    def parse_file(self, file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
        return parse_file(file_path, root_path)

    def parse_file_with_facts(
        self, file_path: Path, root_path: Path
    ) -> tuple[list[CodeNode], list[Edge], FileFacts | None]:
        return parse_file_with_facts(file_path, root_path)


def parse_file(file_path: Path, root_path: Path) -> tuple[list[CodeNode], list[Edge]]:
    """Parse a Python file and return nodes and edges."""
    nodes, edges, _ = parse_file_with_facts(file_path, root_path)
    return nodes, edges


def parse_file_with_facts(
    file_path: Path, root_path: Path
) -> tuple[list[CodeNode], list[Edge], FileFacts | None]:
    """Parse a Python file and return nodes, edges and name-binding facts."""
    rel = str(file_path.relative_to(root_path))
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
        # ValueError covers a source string with an embedded NUL byte: ast.parse raises
        # `ValueError: source code string cannot contain null bytes`, *not* SyntaxError,
        # so an un-caught one used to abort the whole run and leave the project permanently
        # unindexable until the file was excluded by hand (audit finding: NUL-byte parser
        # crash). Returning no nodes lets reconcile keep the file's prior data as stale
        # (it is "scanned but produced nothing" = failure, never a delete) and index the
        # rest of the project.
        logger.debug("Failed to parse %s (%s), skipping", file_path, type(exc).__name__)
        return [], [], None
    except OSError as exc:
        # The file vanished or became unreadable between the scan and this read (e.g. a
        # `git checkout` racing a watcher flush). Treat it as a transient failure, not a
        # deletion — keep prior data stale and retry next run.
        logger.debug("Cannot read %s (%s), skipping", file_path, exc)
        return [], [], None

    lines = source.splitlines()
    nodes: list[CodeNode] = []
    edges: list[Edge] = []
    facts = FileFacts(file_path=rel, language="python")

    # Module node
    mod_id = rel
    mod_doc = ast.get_docstring(tree) or ""
    nodes.append(
        CodeNode(
            id=mod_id,
            name=file_path.stem,
            node_type=NodeType.MODULE,
            file_path=rel,
            line_start=1,
            line_end=len(lines),
            docstring=mod_doc,
        )
    )

    # Walk top-level statements
    for stmt in ast.iter_child_nodes(tree):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            _handle_import(stmt, mod_id, edges, facts)
        elif isinstance(stmt, ast.ClassDef):
            _handle_class(stmt, mod_id, rel, nodes, edges, facts)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _handle_function(stmt, mod_id, rel, nodes, edges, facts)
        else:
            # Module-level statements (script bodies, pytest fixtures via decorators on
            # assignments, __main__ blocks) still make calls; attribute them to the module
            # node so "0 CALLS" keeps meaning "calls nothing" (honest call accounting, docs/calls-resolution.md).
            _extract_calls(stmt, mod_id, edges)

    return nodes, edges, facts


def _collect_local_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Parameter and locally-assigned names that shadow module-level bindings."""
    names: set[str] = set()
    args = func.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        names.add(arg.arg)
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    for child in ast.walk(func):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not func:
            names.add(child.name)
    return sorted(names)


def _handle_import(
    stmt: ast.Import | ast.ImportFrom,
    mod_id: str,
    edges: list[Edge],
    facts: FileFacts,
) -> None:
    if isinstance(stmt, ast.ImportFrom):
        module = "." * stmt.level + (stmt.module or "")
        for alias in stmt.names:
            if alias.name == "*":
                target = module
            elif module:
                target = f"{module}.{alias.name}"
            else:
                target = "." * stmt.level + alias.name
            edges.append(Edge(source=mod_id, target=target, edge_type=EdgeType.IMPORTS))
            if alias.name != "*":
                facts.imports[alias.asname or alias.name] = target
    elif isinstance(stmt, ast.Import):
        for alias in stmt.names:
            edges.append(Edge(source=mod_id, target=alias.name, edge_type=EdgeType.IMPORTS))
            if alias.asname:
                facts.imports[alias.asname] = alias.name
            else:
                # `import a.b` binds the top-level name `a`
                facts.imports[alias.name.split(".")[0]] = alias.name.split(".")[0]


def _handle_class(
    stmt: ast.ClassDef,
    mod_id: str,
    rel: str,
    nodes: list[CodeNode],
    edges: list[Edge],
    facts: FileFacts,
) -> None:
    cls_id = f"{rel}::{stmt.name}"
    doc = ast.get_docstring(stmt) or ""
    nodes.append(
        CodeNode(
            id=cls_id,
            name=stmt.name,
            node_type=NodeType.CLASS,
            file_path=rel,
            line_start=stmt.lineno,
            line_end=stmt.end_lineno or stmt.lineno,
            docstring=doc,
            parent_id=mod_id,
        )
    )
    edges.append(Edge(source=mod_id, target=cls_id, edge_type=EdgeType.CONTAINS))

    # Inheritance
    for base in stmt.bases:
        base_name = _get_name(base)
        if base_name:
            edges.append(Edge(source=cls_id, target=base_name, edge_type=EdgeType.INHERITS))

    # Methods
    for item in ast.iter_child_nodes(stmt):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            meth_id, meth_name = _get_method_identity(rel, stmt.name, item)
            sig = _get_signature(item)
            doc = ast.get_docstring(item) or ""
            nodes.append(
                CodeNode(
                    id=meth_id,
                    name=meth_name,
                    node_type=NodeType.METHOD,
                    file_path=rel,
                    line_start=item.lineno,
                    line_end=item.end_lineno or item.lineno,
                    signature=sig,
                    docstring=doc,
                    parent_id=cls_id,
                )
            )
            edges.append(Edge(source=cls_id, target=meth_id, edge_type=EdgeType.CONTAINS))
            facts.local_names[meth_id] = _collect_local_names(item)
            _extract_calls(item, meth_id, edges)


def _handle_function(
    stmt: ast.FunctionDef | ast.AsyncFunctionDef,
    mod_id: str,
    rel: str,
    nodes: list[CodeNode],
    edges: list[Edge],
    facts: FileFacts,
) -> None:
    func_id = f"{rel}::{stmt.name}"
    sig = _get_signature(stmt)
    doc = ast.get_docstring(stmt) or ""
    nodes.append(
        CodeNode(
            id=func_id,
            name=stmt.name,
            node_type=NodeType.FUNCTION,
            file_path=rel,
            line_start=stmt.lineno,
            line_end=stmt.end_lineno or stmt.lineno,
            signature=sig,
            docstring=doc,
            parent_id=mod_id,
        )
    )
    edges.append(Edge(source=mod_id, target=func_id, edge_type=EdgeType.CONTAINS))
    facts.local_names[func_id] = _collect_local_names(stmt)
    _extract_calls(stmt, func_id, edges)


def _extract_calls(
    node: ast.AST,
    caller_id: str,
    edges: list[Edge],
) -> None:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _get_name(child.func)
            if name:
                edges.append(Edge(source=caller_id, target=name, edge_type=EdgeType.CALLS))


def _get_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        val = _get_name(node.value)
        if val:
            return f"{val}.{node.attr}"
        return node.attr
    return ""


def _get_method_identity(
    rel: str,
    class_name: str,
    item: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[str, str]:
    suffix = _property_accessor_suffix(item)
    qualified_name = f"{class_name}.{item.name}{suffix}"
    return f"{rel}::{qualified_name}", qualified_name


def _property_accessor_suffix(item: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    for decorator in item.decorator_list:
        if isinstance(decorator, ast.Attribute) and decorator.attr in {"setter", "deleter"}:
            return f".{decorator.attr}"
    return ""


def _get_signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args: list[str] = []
    for arg in func.args.args:
        annotation = ""
        if arg.annotation:
            annotation = f": {ast.unparse(arg.annotation)}"
        args.append(f"{arg.arg}{annotation}")
    ret = ""
    if func.returns:
        ret = f" -> {ast.unparse(func.returns)}"
    prefix = "async " if isinstance(func, ast.AsyncFunctionDef) else ""
    return f"{prefix}def {func.name}({', '.join(args)}){ret}"
