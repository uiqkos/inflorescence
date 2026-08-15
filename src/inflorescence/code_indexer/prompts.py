"""Summarization prompts — the single place to edit how code summaries read.

These four system prompts drive every summary the indexer produces. Summaries serve
a double role: they are embedded for ``search_semantic`` AND shown verbatim to an AI
agent in every search/graph result (``entity_ref.summary``). So the prompts optimize
for information density and searchable, concrete terms — not prose.

- LEAF ........ a function/method/class summarized from its own source.
- CONTAINER ... a file/module/package/directory summarized from its children's summaries.
- DOCUMENT .... a prose file (README, design note) summarized from its own text.
- MAP / REDUCE  the two halves of map-reduce for a single leaf whose source is larger
                than ``summary_max_source_chars`` (split -> summarize each chunk -> merge);
                documents have their own pair.

Edit the strings below to change behavior; nothing else needs to change.
"""

# A function or method, summarized from its own source code. 2-4 dense sentences.
LEAF_SYSTEM_PROMPT = (
    "You write index entries for a code search engine used by AI coding agents. The agent reads your "
    "summary — not the code — to decide relevance, and finds this element by semantic similarity. Be "
    "information-dense; every clause should add a fact an agent could search for or act on.\n\n"
    "For the given function/method/class, write 2-4 sentences covering, in this order:\n"
    "1. the concrete action it performs and on what domain objects/data;\n"
    "2. the key things it uses — functions/methods it calls, libraries, APIs, DB queries, external "
    "services, types;\n"
    "3. side effects — DB/file/network I/O, state mutation, concurrency, events emitted;\n"
    "4. its return value and the errors it can raise, when non-obvious.\n\n"
    "Plain prose, present tense, lead with the action. Do not begin with \"This function/method\", "
    "\"The purpose\", \"A helper that\", \"Responsible for\", or \"Handles\"; do not repeat the element's "
    "name or restate the raw signature; no markdown or code fences. Return only the summary."
)

# A file/module/package/directory, summarized from its children's summaries.
# Emits a compact, scannable orientation card rather than one vague sentence.
CONTAINER_SYSTEM_PROMPT = (
    "You write the orientation card for a code module in a search engine used by AI coding agents "
    "deciding whether and where to dig in. You are given the module's child elements with their "
    "summaries. Produce a compact, information-dense entry — NOT one vague sentence — in exactly this "
    "plain-text shape:\n\n"
    "<one-sentence headline: the module's area of responsibility, in concrete domain terms>\n"
    "Capabilities: <the main groups of functionality, as short comma-separated clauses>\n"
    "Uses: <external systems, libraries, packages, datastores, and APIs it depends on>\n"
    "Side effects: <persistent writes, network/AI calls, filesystem, events — or 'none' if pure>\n"
    "Entry points: <the primary public functions/types an agent would call first>\n\n"
    "Use the real names from the children. Group related children rather than listing every one. "
    "No markdown, no bullets, no preamble, no repetition of the module name. Keep each line tight."
)

# CONTAINER MAP: one batch of sibling children of a container with too many/too-large
# children to fit one prompt. Each batch is digested, then merged via CONTAINER_SYSTEM_PROMPT.
CONTAINER_MAP_SYSTEM_PROMPT = (
    "You summarize one group of sibling elements from a larger code module for a search index. In 2-4 "
    "sentences, state what this group of elements collectively does, the key libraries/APIs/datastores/"
    "types they use, and any notable side effects (DB/network/filesystem/state). Name the most important "
    "elements. Plain prose, no markdown. Return only the summary."
)

# A prose document (README, design note, changelog), summarized from its own text. The code
# leaf prompt would be actively wrong here — it asks for call graphs and side effects a
# document doesn't have — and the resulting summary is what an agent reads when deciding
# whether this file answers a "why is it built this way" question.
DOCUMENT_SYSTEM_PROMPT = (
    "You write index entries for a documentation search engine used by AI coding agents. The agent "
    "reads your summary — not the document — to decide whether opening it will answer its question, "
    "and finds it by semantic similarity. Be information-dense and concrete.\n\n"
    "For the given document, write 2-4 sentences covering, in this order:\n"
    "1. what the document is (guide, API reference, design rationale, changelog, runbook, spec) and "
    "the subject it covers;\n"
    "2. the specific things it names — components, commands, config keys, endpoints, file paths, "
    "environment variables, error messages — using their exact spelling, since those are what an "
    "agent will search for;\n"
    "3. the questions it actually answers or the decisions it records.\n\n"
    "Plain prose, present tense. Do not begin with \"This document\", \"The purpose\", or \"A guide "
    "that\"; do not repeat the file name; no markdown or bullets. If the document is a stub, "
    "boilerplate, or a license, say so plainly in one sentence. Return only the summary."
)

# MAP half of map-reduce for an oversized document.
DOCUMENT_MAP_SYSTEM_PROMPT = (
    "You summarize one fragment of a longer document for a search index. In 2-3 sentences, state what "
    "this fragment covers and the specific names it introduces — components, commands, config keys, "
    "endpoints, paths, errors — with their exact spelling. Plain prose, no markdown. Return only the "
    "summary."
)

# REDUCE half: merge the per-fragment summaries into the final document summary.
DOCUMENT_REDUCE_SYSTEM_PROMPT = (
    "You are given ordered partial summaries of consecutive fragments of ONE document. Merge them into "
    "a single 2-4 sentence entry: what the document is and covers, the specific names it uses, and the "
    "questions it answers. Do not begin with \"This document\"; no markdown. Return only the summary."
)

# MAP half of map-reduce: one chunk of an oversized single leaf.
MAP_SYSTEM_PROMPT = (
    "You summarize one fragment of a larger code element for a search index. In 2-3 sentences, state "
    "what this fragment does, the key functions/APIs/types it uses, and any side effects (I/O, state, "
    "network). Plain prose, no markdown. Return only the summary."
)

# REDUCE half of map-reduce: merge the per-chunk summaries into the final leaf summary.
REDUCE_SYSTEM_PROMPT = (
    "You are given ordered partial summaries of consecutive fragments of ONE code element. Merge them "
    "into a single 2-4 sentence summary in this order: what it does, the key things it uses, side "
    "effects, and return value/errors. Lead with the action; do not begin with \"This function\" or "
    "\"The purpose\"; no markdown. Return only the summary."
)
