"""Parser subpackage -- language-specific code parsers and registry."""

from inflorescence.code_indexer.parser.ast_parser import PythonAstParser
from inflorescence.code_indexer.parser.base_parser import BaseParser
from inflorescence.code_indexer.parser.go_parser import GoParser
from inflorescence.code_indexer.parser.javascript_parser import JavaScriptParser
from inflorescence.code_indexer.parser.registry import ParserRegistry
from inflorescence.code_indexer.parser.resolver import ResolutionResult, resolve_edges
from inflorescence.code_indexer.parser.rust_parser import RustParser

__all__ = [
    "BaseParser",
    "GoParser",
    "JavaScriptParser",
    "ParserRegistry",
    "PythonAstParser",
    "ResolutionResult",
    "RustParser",
    "resolve_edges",
]
