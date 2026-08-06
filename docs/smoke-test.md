# End-to-end smoke test

The unit suite runs against nothing — no database, no API key, no network. This is the other
half: a manual gate that exercises the **live** stack, from a fresh Memgraph to real LLM calls,
and proves the two properties tests cannot check cheaply — that the tools return real payloads,
and that editing one file does not re-summarize the project.

Run every command from the repo root. Each step lists its pass condition.

## 0. Prerequisites

- Docker running, `docker compose` available.
- A real `OPENROUTER_API_KEY` **with credit on it** in `.env` (gitignored). This spends a few
  cents of real LLM and embedding calls on a tiny fixture.
- `doctor` checks that a key is *present*, not that it has quota. An exhausted key stays green
  and then 403s every call, which surfaces in steps 5–7 as `summaries=0`, `chunks=0`, and
  `embedding_failed` on the semantic tools. Confirm remaining credit before starting.

## 1. Bring up the stack

```bash
docker compose up -d
```

Wait for Memgraph to report healthy — do not proceed on "starting":

```bash
until [ "$(docker inspect --format '{{.State.Health.Status}}' "$(docker compose ps -q memgraph)")" = healthy ]; do
  echo "waiting for memgraph..."; sleep 2;
done; echo "memgraph healthy"
```

**Pass:** prints `memgraph healthy` within ~30 s, and Bolt answers:

```bash
echo 'RETURN 1;' | docker compose exec -T memgraph mgconsole
```

**Pass:** a result row containing `1`.

## 2. Doctor

```bash
uv run inflorescence doctor; echo "doctor exit: $?"
```

**Pass:** config load, `OPENROUTER_API_KEY`, Memgraph reachability are green.

> `Schema present` only turns green **after** a first successful index — it checks that the
> core `Code(id)` / `Code(project)` property indexes exist. On a freshly-wiped `mg_data` volume
> it reads `not found` until step 5. That is expected.

## 3. A tiny throwaway fixture

Small, deterministic, cheap to summarize — which makes the watch-mode cost delta crisp.

```bash
export SMOKE_DIR="$(mktemp -d -t inflo-smoke-XXXX)"
mkdir -p "$SMOKE_DIR/smokelib"

cat > "$SMOKE_DIR/smokelib/__init__.py" <<'PY'
"""Smoke fixture package."""
PY

cat > "$SMOKE_DIR/smokelib/math_ops.py" <<'PY'
"""Arithmetic helpers for the smoke fixture."""


def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def multiply(a, b):
    """Return the product of two numbers."""
    return a * b


class Calculator:
    """A tiny calculator used to smoke-test entity and graph tools."""

    def compute(self, a, b):
        """Add then multiply, raising on a zero multiplier."""
        total = add(a, b)
        if b == 0:
            raise ValueError("division by zero not allowed")
        return multiply(total, b)
PY

cat > "$SMOKE_DIR/smokelib/greet.py" <<'PY'
"""Greeting helper for the smoke fixture."""


def greet(name):
    """Return a friendly greeting for name."""
    return f"Hello, {name}!"
PY

echo "fixture at: $SMOKE_DIR"; find "$SMOKE_DIR" -name '*.py' | sort
```

**Pass:** three `.py` files under `$SMOKE_DIR/smokelib/`.

## 4. Dry run — no writes, no spend

```bash
uv run inflorescence index "$SMOKE_DIR" --dry-run
```

**Pass:** a preview (3 files, a Python-only language breakdown, total size, estimated nodes,
a token/cost estimate) and **zero** LLM calls. Confirm nothing was written:

```bash
echo 'MATCH (n:Code) RETURN count(n) AS n;' | docker compose exec -T memgraph mgconsole
```

**Pass:** unchanged from before the dry run (0 on a fresh database).

## 5. Real index

```bash
uv run inflorescence index "$SMOKE_DIR"
```

**Pass:** `Indexing complete: {...}` with `nodes` > 0 (about 9 here: 1 directory + 2 modules +
`add`/`multiply`/`greet` + `Calculator` + `compute`), `edges` > 0, `chunks` > 0,
`summaries` > 0, `reused: False`. Verify persistence:

```bash
echo 'MATCH (n:Code) RETURN count(n) AS nodes;' | docker compose exec -T memgraph mgconsole
```

**Pass:** matches the printed count. Re-run `doctor` — the fourth check is now green.

## 6. Exercise the MCP tools

Drives the registered tool coroutines in-process — the same code path an MCP client hits — and
asserts each returns a sensible, non-error payload.

```bash
SMOKE_DIR="$SMOKE_DIR" uv run python - <<'PY'
import asyncio, os
from inflorescence.server import create_server, startup, shutdown

DIR = os.environ["SMOKE_DIR"]

async def main():
    mcp, conn, llm, watcher = create_server()
    tools = {t: mcp._tool_manager._tools[t].fn for t in mcp._tool_manager._tools}
    await startup(conn)
    ok = True
    try:
        def check(name, res):
            nonlocal ok
            err = isinstance(res, dict) and "error" in res
            items = (res or {}).get("items") if isinstance(res, dict) else None
            passed = (not err) and (items is None or len(items) > 0 or name == "cypher_query")
            print(f"[{'PASS' if passed else 'FAIL'}] {name}: "
                  f"{'error=' + str(res['error']) if err else 'items=' + str(len(items) if items is not None else 'n/a')}")
            ok = ok and passed
            return res

        check("search_code",     await tools["search_code"](directory=DIR, query="add and multiply numbers"))
        check("search_semantic", await tools["search_semantic"](directory=DIR, query="a tiny calculator"))
        check("search_hybrid",   await tools["search_hybrid"](directory=DIR, query="calculator compute"))
        check("search_text",     await tools["search_text"](directory=DIR, query="Calculator"))

        cy = check("cypher_query", await tools["cypher_query"](
            directory=DIR,
            query="MATCH (n:Code {project: $project}) RETURN n.id AS id, n.name AS name"))
        node_id = cy["items"][0]["id"] if cy.get("items") else None
        print(f"       cypher rows={len(cy.get('items', []))} first_id={node_id}")

        if node_id:
            ctx = await tools["get_entity_context"](directory=DIR, node_id=node_id)
            passed = "error" not in ctx and ctx.get("entity") is not None
            ok = ok and passed
            print(f"[{'PASS' if passed else 'FAIL'}] get_entity_context: entity={bool(ctx.get('entity'))}")
        else:
            ok = False
            print("[FAIL] get_entity_context: no node_id from cypher_query")
    finally:
        await shutdown(conn, llm, watcher)
    raise SystemExit(0 if ok else 1)

asyncio.run(main())
PY
echo "tools smoke exit: $?"
```

**Pass:** every line reads `[PASS]` and the exit code is 0. Each search returns `items` with at
least one entity ref; `cypher_query` returns rows; `get_entity_context` returns a non-null
`entity` with `sections`.

## 7. The cost gate (INV-5)

Confirms that editing **one** file does not re-summarize the whole project — the property that
makes live indexing affordable. Counts real LLM summary calls around a single-file edit, on a
fresh fixture so the first index is a full build.

```bash
uv run python - <<'PY'
import asyncio, tempfile, textwrap, pathlib
from inflorescence.server import create_server, startup, shutdown

async def main():
    root = pathlib.Path(tempfile.mkdtemp(prefix="inflo-watch-"))
    (root / "smokelib").mkdir()
    (root / "smokelib" / "__init__.py").write_text('"""pkg."""\n')
    mo = root / "smokelib" / "math_ops.py"
    mo.write_text(textwrap.dedent('''\
        """Arithmetic helpers."""


        def add(a, b):
            """Return the sum."""
            return a + b


        def multiply(a, b):
            """Return the product."""
            return a * b
        '''))
    (root / "smokelib" / "greet.py").write_text(textwrap.dedent('''\
        """Greeting helper."""


        def greet(name):
            """Greet name."""
            return f"Hello, {name}!"
        '''))

    mcp, conn, llm, watcher = create_server()
    calls = {"n": 0}
    orig = llm.generate
    async def counting(*a, **k):
        calls["n"] += 1
        return await orig(*a, **k)
    llm.generate = counting

    await startup(conn)
    index_directory = mcp._tool_manager._tools["index_directory"].fn
    try:
        await index_directory(path=str(root))     # full build -> full summarization
        n0 = calls["n"]
        print(f"full index LLM calls: n0={n0}")
        calls["n"] = 0

        # Edit ONE function body in ONE file.
        mo.write_text(mo.read_text().replace("    return a * b\n", "    product = a * b\n    return product\n"))

        # Wait past the debounce window so the watcher flushes and the update runs.
        await asyncio.sleep(9)
        n1 = calls["n"]
        print(f"single-file edit LLM calls: n1={n1}")

        passed = 0 < n1 < n0 and n1 <= 5
        print(f"[{'PASS' if passed else 'FAIL'}] incremental summarization: "
              f"edited-one-file re-summarized {n1} node(s), full project was {n0}")
        raise SystemExit(0 if passed else 1)
    finally:
        await shutdown(conn, llm, watcher)

asyncio.run(main())
PY
echo "watch smoke exit: $?"
```

**Pass:** `n1` is small (~3 — the edited function plus its module and the directory root along
the `CONTAINS` path), strictly less than `n0`, and the exit code is 0. `n1 == n0` means the
whole project was re-summarized: **fail the gate**.

Cross-check the log for the same signal:

```bash
grep "Starting hierarchical summarization" inflorescence.log | tail -2
```

**Pass:** the post-edit line shows a small `= N total` against the larger total of the initial
index.

## 8. Tear down

```bash
docker compose down            # add -v to drop the mg_data volume too
rm -rf "$SMOKE_DIR"
```
