# CLAUDE.md

## Documentation is part of every change

After every block of work — a feature, a fix batch, a refactor — bring the agent-facing
and human-facing texts up to date **in the same block, not later**:

- **MCP surface** (what a consuming agent reads): the server `instructions` string in
  `src/inflorescence/server.py` and every affected tool docstring under
  `src/inflorescence/tools/` (FastMCP publishes docstrings verbatim as tool
  descriptions). If behavior, parameters, valid enum values, cost, or error semantics
  changed — the text must say so. A stale tool description is a bug of the same class
  as a silent empty result (INV-7).
- **README.md** and anything else a human reads first, when the change touches
  install, CLI flow, or user-visible behavior.
- **Error message texts**: hints inside errors must point at commands/tools that still
  exist and are still the recommended path.

Definition of done for a work block includes: descriptions enumerate real valid values
(no guessing), examples still run, and nothing promises behavior the code no longer has.
