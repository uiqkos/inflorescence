"""The paragraphs several tool descriptions share, written once.

FastMCP publishes each tool's description verbatim, and an agent reads nothing else about a tool
before calling it — so a stale sentence is a wrong instruction, not a typo (CLAUDE.md). The same
contract (how path filters expand, what ``answered_from`` means, what an empty page carries)
used to be restated by hand in up to ten docstrings, and a change to the behaviour had to find
all ten. The tools now compose their descriptions from these fragments, so the contract is
stated in one place and every tool that honours it says so in the same words.

Anything that enumerates values is *derived* from the code that defines them, never typed out:
a node type added to :class:`NodeType` appears here without anyone remembering to add it.
"""

from __future__ import annotations

from inflorescence.code_indexer.models import (
    DECLARATION_NODE_TYPES,
    IMPORT_FACT_FIELDS,
    IMPORT_FACT_FLAGS,
    NodeType,
)
from inflorescence.config import DEFAULT_DOCUMENT_FILENAMES, DOCUMENT_EXTENSION_GROUPS
from inflorescence.tools.gates import NODE_TYPE_GROUPS
from inflorescence.tools.responses import FULL_SUMMARY_MAX_LIMIT, NAV_SUMMARY_MAX_CHARS


def _node_type_values() -> str:
    types = ", ".join(t.value for t in NodeType)
    groups = "; ".join(
        f'``"{name}"`` stands for {", ".join(members)}' for name, members in NODE_TYPE_GROUPS.items()
    )
    declarations = " and ".join(f"``{t.value}``" for t in sorted(DECLARATION_NODE_TYPES, key=lambda t: t.value))
    return (
        f"``node_types`` values (case-insensitive): {types} — plus group names: {groups} (the way "
        "to ask about code rather than about notes, configuration or scripts). ``meta.node_types`` "
        f"echoes the expanded list. {declarations} are top-level declarations — a constant, a "
        "rule table, a registry object, a type alias — found by their name and first line; they "
        "are never summarized, so they carry ``signature`` (that first line) where other "
        "entities carry ``summary``, and their text stays in the enclosing module's code chunks."
    )


NODE_TYPES = _node_type_values()

PATH_FILTERS = (
    "``include_paths``/``exclude_paths`` are globs; a pattern with no glob character "
    "(``\"docs/\"``, ``\"docs\"``) covers both that exact path and everything under it, and "
    "``meta.path_filters`` echoes the expanded form that was actually applied."
)

IS_TEST = (
    "A hit whose *path* looks like test code (``tests/``, ``spec/``, ``test_*.py``, "
    "``*_test.go``, ``conftest.py``, ``*.test.ts``, …) carries ``\"is_test\": true``. That is a "
    "heuristic over the path only — it never reads the file — so treat it as a hint, not a verdict."
)

SCOPE = (
    "STRONGLY RECOMMENDED — scope every query, and exclude what is not the answer, BEFORE you read "
    "the results. A repository holds many copies of the vocabulary you search for that are not "
    "the code you want: tests, mocks and fixtures, prototypes, sample data, reports, old copies of "
    "a feature. They out-rank that code on exactly the queries used to find it — field reports "
    "measured 11 of 15 hits from ``tests/`` while the runner named in the query never reached "
    "page 1, and 9 of 12 hits from a mock tree and archived prototypes with the two real answers "
    "7th and 8th. So, on every call: ``include_paths`` names the area you are asking about; "
    "``exclude_paths`` names every tree that is not the answer; ``include_tests=false`` whenever "
    "you are looking for an implementation (a test restates the vocabulary of the code it "
    "exercises); ``node_types=[\"code\"]`` when the question is about code rather than notes. "
    "Example: ``search_semantic(directory=\"/repo\", query=\"job runner start stop\", "
    "include_paths=[\"backend/\"], exclude_paths=[\"backend/fixtures/\"], include_tests=false)``. "
    "Keep tests in when the test IS what you want — a usage example, an expected behaviour, a "
    "repro. A tree that is never the answer for anyone belongs in ``.inflorescenceignore`` (not "
    "indexed at all), and a tree that is history in ``.inflorescencearchive`` (ranked last) — "
    "said once in the repository instead of on every query."
)

ARCHIVED = (
    "A repository can list paths that are history — kept for reference, never the answer by "
    "default: archived prototypes, a legacy tree — in ``.inflorescencearchive`` at its root "
    "(gitignore syntax, committed with the code). A hit from such a path carries "
    "``\"archived\": true`` and is listed after every live hit instead of interleaved with them; "
    "nothing is dropped (``exclude_paths`` does that for one query)."
)

WORKTREE_FIRST_QUERY = (
    "A first query against a git worktree of an indexed checkout that is not indexed yet indexes it "
    "inline before answering: the main checkout's graph is copied inside the database and only the "
    "files the branch changed are parsed, summarized and embedded (paid for those files alone, "
    "under the MAX_INDEX_COST_USD cap). Files that exist only on the branch are indexed in that "
    "same step, summaries included. That one call is slower — seconds for a small diff, minutes "
    "when the branch changed hundreds of files — and every later call is normal speed."
)

PROVENANCE = (
    "``meta.answered_from`` names the tree this answer came from: path, branch @ short commit, and "
    "— for a git worktree — the checkout it belongs to. ``meta.index_freshness`` ({checked, stale, "
    "missing, unknown}) covers the files on this page, and affected items carry ``\"stale\": true`` "
    "/ ``\"file_missing\": true`` — so an unflagged item was checked and still matches the index. "
    "This is file-level: \"stale\" means the file changed after indexing (lines may have shifted, "
    "the summary may describe older code), not that this entity changed. "
    "``index_freshness.drift``, when present, says the checkout moved on since the index was last "
    "brought up to date — HEAD is a different commit (a merge, a rebase, a checkout) — and how many "
    "indexed files differ between the two: everything the index says about those files describes "
    "the older commit, however clean this page's own flags look, until ``index_directory`` catches "
    "it up. ``index_freshness.live_worktrees``, when present, is a sentence naming how many git "
    "worktrees of this checkout are NOT in this index — their branch state is elsewhere; ask with "
    "``directory`` set to the worktree itself. ``meta.nested_checkouts`` appears when the files on "
    "the page live in an independent git checkout inside this tree (a submodule or vendored repo on "
    "its own branch) — that checkout's HEAD, not ``answered_from``, is the code you are reading."
)

EMPTY_PAGE = (
    "An EMPTY first page carries ``meta.index_coverage``: the index's file count plus a bounded "
    "re-check of the directories it knows (the ones your include_paths name first) against the "
    "filesystem. It answers one question — is the empty page a coverage gap? It either names files "
    "that are on disk and NOT in this index (read them directly and ``index_directory`` to make "
    "them queryable), names file types present there that this index does not admit at all, or "
    "says every file there is indexed — which rules out a coverage gap, and nothing more: it is a "
    "statement about files, not about what the files say."
)

HINTS = (
    "``meta.hints``, when present, is a list of sentences about the shape of this page and the "
    "parameter that changes it — every hit weak, most hits tests, most hits documents, most hits "
    "from one directory while the rest of the page is elsewhere (the ``exclude_paths`` that "
    "frees the page is named), archived hits moved down, query words that matched nothing. Read "
    "it before trusting the top hit."
)

WEAK = (
    "A vector search always returns its nearest neighbours, so low scores still look like results: "
    "a hit below the weak-similarity threshold carries ``\"weak\": true``, and ``meta.weak_results`` "
    "(with a sentence in ``meta.hints``) is set when *every* hit on the page is weak — nothing "
    "matched closely, so the code probably is not there, or is named differently; confirm with "
    "``search_text`` on exact terms or ``find_text`` for a literal. (This only labels; "
    "``min_score`` is the separate hard cut-off.)"
)

COMPLETE_POOL = (
    "The search indexes are shared by every project in the database, so candidates are drawn from "
    "a window that widens until this project's filtered page is full or the index has nothing "
    "further; ``page.truncated: true`` appears only if the widest window was reached first — then "
    "a short or empty page is not evidence that nothing else matches."
)


LISTING_FLAGS = (
    "Items are flagged the way search hits are — ``\"is_test\": true`` when the path looks like "
    "test code (a path heuristic), ``\"archived\": true`` when ``.inflorescencearchive`` "
    "declares the path history — but a listing keeps its order: nothing is moved. "
    "``include_tests=false`` drops test paths before the page is cut, so a directory of "
    "``*.test.ts`` files no longer pushes the implementation off page 1; ``exclude_paths`` drops "
    "any other tree the same way."
)

FULL_SUMMARIES = (
    f"Summaries are cut to a {NAV_SUMMARY_MAX_CHARS}-character preview so a page of hundreds of "
    "entities stays readable. ``full_summaries=true`` returns them whole — the module summaries "
    "(capabilities, uses, side effects, entry points) are the part worth reading on a small "
    f"inventory — and caps the page at {FULL_SUMMARY_MAX_LIMIT} items (``meta.full_summaries`` says so; walk "
    "``next_offset``). ``get_entity_context`` gives one entity's full summary with its "
    "neighbourhood."
)

NAME_RESOLUTION = (
    "``name`` is resolved by these rules, in order, and ``meta.name_match`` says which one "
    "decided: exactly one entity has the name spelled exactly as given (``\"exact\"`` — "
    "``TASKS`` resolves to the constant even when ``readTasks`` and ``TaskSource`` also contain "
    "it); else exactly one has it ignoring case (``\"exact_ignoring_case\"``); else exactly one "
    "contains it as a substring (``\"substring\"``). When that is still more than one, "
    "``include_paths``/``exclude_paths`` narrow the candidates too (``meta.name_match`` then "
    "ends in ``\"_within_path_filters\"``). Anything still ambiguous is never resolved by picking "
    "one — \"who calls X\" answered about the wrong X looks right and is wrong: the call fails "
    "with ``ambiguous_name`` listing the candidates (id, node_type, file_path), exact matches "
    "first, and the next call passes one as ``node_id``. Nothing matching fails with "
    "``not_found``."
)

IMPORT_FACTS = (
    "An IMPORTS row carries ``import`` — the statement as written: "
    + ", ".join(f"``{field}``" for field in IMPORT_FACT_FIELDS)
    + " (the module string, the line of the first statement, the names brought in; ``\"*\"`` = "
    "namespace/whole module, ``\"default\"`` = default import), plus "
    + ", ".join(f"``{flag}``" for flag in IMPORT_FACT_FLAGS)
    + " present only when true (``dynamic`` = only ever a lazy ``import()`` chunk; ``type_only`` = "
    "erased at build; ``reexport`` = ``export … from``). Statements from one file into one target "
    "fold into one edge. A named JS/TS import also points at the symbol itself, so incoming "
    "IMPORTS on a constant or function answer \"who reads it\". An edge indexed before these "
    "properties existed has no ``import`` until the next index run adds it (no LLM spend)."
)



def _document_types() -> str:
    groups = "; ".join(f"{name} ({', '.join(exts)})" for name, exts in DOCUMENT_EXTENSION_GROUPS.items())
    names = ", ".join(DEFAULT_DOCUMENT_FILENAMES)
    return (
        f"{groups}; and the build files identified by exact name rather than suffix ({names} — "
        "never a real .env)"
    )


# What is admitted as a ``document`` by default (DOCUMENT_EXTENSIONS / DOCUMENT_FILENAMES change
# it) — the list an agent checks before concluding a file type is simply not searchable.
DOCUMENT_TYPES = _document_types()


def join(*paragraphs: str) -> str:
    """Join description paragraphs with blank lines, the way FastMCP renders a docstring."""
    return "\n\n".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())
