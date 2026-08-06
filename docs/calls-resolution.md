# How `CALLS` edges are resolved

A call graph is only useful if you can trust it. This is the design of the resolution ladder
behind every `CALLS` edge, the measurements that shaped it, and what it still cannot do.

Consolidated from the investigation and implementation notes of 2026-07-28.

## The problem: syntax is not semantics

A parser sees `content.Check(...)` and reports *a call whose target text is `content.Check`*.
It does not know whether `content` is an imported package, a local variable, or a struct
field; which declaration `Check` binds to; or, for `s.content.ForCheck`, what the type of the
field `content` is. Answering that is name binding plus type inference — a semantic layer, one
per language, comparable in size to half a compiler.

The original resolver matched call targets against an index of **short** node names, so only
bare identifiers (`foo()`, `NewService()`) ever resolved. Everything qualified — `pkg.Func`,
`recv.Method`, stdlib — failed to match, and an unresolved edge was silently dropped at write
time because `upsert_edges` matches both endpoints by node id.

Measured on a 193-file Go backend (1022 function/method nodes):

- **51%** of nodes had zero outgoing `CALLS` — they looked like leaves and were not,
- and of the edges that *were* written, only **69%** were correct when checked against a
  compiler-grade index. The wrong third came from `candidates[0]`: given several nodes with
  the same short name, the resolver picked the first.

Two failure modes, opposite in direction: silence about real calls, and confident invention of
fake ones. A consumer cannot tell either apart from the truth.

## The ladder

```
call target from the parser (raw text: 'content.Check', 's.content.ForCheck', 'h')
  │
  ├─ L3  semantic     a SCIP indexer for the language (scip-go / scip-python /
  │                   scip-typescript). Replaces heuristic edges per file, wherever
  │                   the project actually builds.
  ├─ L2  import       first segment → the file's imports → a node in that package/module
  │      field-type   recv.field.Method via the declared type of the field (Go)
  │      self         self./this./receiver methods of the own class (+ INHERITS chain)
  │      same-module  a bare name in the same file (py/js) / same package (Go)
  └─ L0  external_calls / unresolved_calls — honest lists stored on the caller node.
```

An edge is **never guessed**. Neither `candidates[0]` nor last-segment matching survives —
see the measurements below for why.

Supporting rules:

- **Shadow guard** — parameter and local-variable names collected into `FileFacts.local_names`
  block a bare call from binding to a same-named function. (Real failure case:
  `def f(init_validator): init_validator()`.)
- **Module-level calls** are attributed to the module node, so a file made of top-level code
  (scripts, `describe`/`it` bodies, initializers) does not look inert.
- **Builtins and globals** (`len`, `make`, `console`, `fetch`, …) are recorded nowhere: noise.

### Why L1 "globally unique short name" was cut

The original design had a rung between L2 and L0: if the last segment of a qualified target
matches exactly one node in the project, bind to it. Measured against SCIP ground truth on the
Go backend, that rung produced 495 edges of which **only 38.9% were correct** (72 of 185
judged). Uniqueness is computed among *project* nodes, but a qualified call usually points at
stdlib or a third-party method — `x.Close()` finds the project's single `Server.Close` and is
wrong. The rung was deleted; between L2 and "write it down raw" nothing belongs.

## Measured precision

L2 heuristic rungs, scored only on files covered by a SCIP index (Go backend, ground truth =
`scip-go`):

| Rung | Edges | Confirmed | Precision |
|---|---|---|---|
| same-package (`foo()`) | 967 | 697/697 judged | **100%** |
| import (`pkg.Func`) | 157 | 102/102 | **100%** |
| self (`s.Method`) | 173 | 169/169 | **100%** |
| field-type (`s.dep.Method`) | 44 | 35/35 | **100%** |
| **L2 total** | 1341 | 1003/1003 | **100%**, recall **84.5%** |
| ~~L1 unique-name~~ | 495 | 72/185 | **38.9%** — cut |

End-to-end, across languages:

| Project | Precision vs SCIP | Heuristic recall | Covered by the honest scheme | True leaves |
|---|---|---|---|---|
| Go backend (1022 nodes) | **100%** (1003/1003) | 84.5% | 84% | 10% |
| Python, cerberus (469) | **100%** (431/431) | 65.8% | 68% | 15% |
| TypeScript frontend (94) | **100%** (25/25) | 17.6% ¹ | 44% | 31% |
| JavaScript, express/lib (11) | — ² | — | 45% | 9% |

¹ TypeScript calls mostly go through methods and hooks on typed values — L3 territory; the
semantic pass covers that frontend completely (53 files via `scip-typescript`).
² Prototype-style CJS (`app.use = function`) is barely modelled by a structural parser: there
are no nodes to bind to for either layer. The honest lists still fill up (73% of nodes carry
`unresolved_calls`).

## The SCIP layer (`scip_semantic.py`)

- **Reader** — a minimal protobuf wire parser (~60 lines, no dependencies) over the fields of
  `scip.proto` that matter: `Index.documents` → `Document.relative_path`/`occurrences` →
  `Occurrence.range`/`symbol`/`roles`/`enclosing_range`. Verified bit-for-bit against the
  official `scip print --json` on four real indexes (113,586 occurrences, 399 documents).
- **Preflight and degradation** — no binary on `PATH` and no Docker means a silent skip; a
  non-zero exit, a timeout, or unreadable output means a warning and a fallback to the
  heuristic ladder. The semantic layer never fails a run.
- **Routes** — Go: every directory with a `go.mod`; TypeScript: every directory with a
  `tsconfig.json` or `package.json` (nested ones collapse, capped at 5); Python: the root.
  Document paths are prefixed with the route's relative path, and documents outside the
  project root (Go build-cache artifacts) are filtered out.
- **Adapter** — a definition occurrence maps to the innermost node by file + line; a reference
  inside a function/method node becomes a `semantic` edge. `local N` symbols are
  document-scoped and are never bound (binding them produced false cross-file edges — caught
  in verification).
- **Merge** — for covered files, semantic edges replace the heuristic `CALLS` of function and
  method nodes (`external_calls` replaced, `unresolved_calls` cleared). Module-level edges stay
  heuristic: in SCIP an import line is indistinguishable from a call. Uncovered files keep
  untouched heuristics.

Speed is not the constraint people expect: indexing a Go backend plus a TypeScript frontend
end-to-end took **4.7 s** — 214 files covered (161 `scip-go` + 53 `scip-typescript`), 1305
semantic edges.

## What is stored

- `[:CALLS {resolution, callee_text}]` — which rung resolved the edge, and the call as written.
- On the node: `external_calls: [str]`, `unresolved_calls: [str]` (deduped, sorted, capped at
  100).
- On the module node: `calls_provenance: 'scip-go' | 'scip-python' | 'scip-typescript' |
  'heuristic'`.
- **"Really calls nothing"** = no `CALLS` edges *and* both lists empty. That is the only
  reading a consumer may make of an empty call list.

Unresolved targets are no longer handed to `upsert_edges`, so an edge-count shortfall means a
bug again rather than background noise (INV-7).

## Known limits

- **Rust** — exact/same-file resolution plus honest lists only. `rust-analyzer scip` would
  slot into the same adapter; not wired up.
- **TypeScript** — `new C()` is not emitted by the heuristic (`new_expression` is not a
  `call_expression`), and classes are excluded from the TS semantic pass because type
  positions generate noise. Class-heavy TS deserves more work.
- **Build tags** — files behind `//go:build integration` are invisible to `scip-go` in its
  default configuration (177 documents vs 213 with `GOFLAGS=-tags=integration`). Configure it
  through `scip_env` when those files matter; the rest still falls through to L2.
- **Incrementality** — the semantic pass runs on every `build()`, which is seconds for Go and
  TypeScript but ~30 s for `scip-python`. Enable that one deliberately via `scip_commands`.
  Debouncing and package-level scoping are the next step.
