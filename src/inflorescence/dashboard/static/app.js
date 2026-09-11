/* Inflorescence dashboard.
 *
 * No build step: plain DOM, one <canvas> engine (static/pack.js) and highlight.js.
 * The graph view is the site's specimen — the whole index as nested discs, CONTAINS drawn as
 * nesting, CALLS / IMPORTS / INHERITS bundled through the discs they cross — driven here by
 * hand: wheel, drag, pin, route. Everything on screen is a number the index actually holds.
 */
"use strict";

// The whole palette, and it is not decorative: one colour per edge type, plus the blue an
// outgoing call is drawn in. Entity types are told apart by shape and ink, never by a new hue.
const REL_COLORS = {
  CALLS: "#ff3aa8", IMPORTS: "#8b7bff", INHERITS: "#e6ff3d",
  IMPLEMENTS: "#e6ff3d", CONTAINS: "rgba(235,232,225,.55)",
};
const REL_KIND = { C: "CALLS", I: "IMPORTS", N: "INHERITS", M: "IMPLEMENTS", P: "CONTAINS" };
const KIND_OF = { CALLS: "C", IMPORTS: "I", INHERITS: "N", IMPLEMENTS: "M" };
const TYPES = ["directory", "module", "document", "class", "interface", "enum", "struct", "trait", "function", "method", "code"];
const CHIP_TEXT = {
  directory: "dir", module: "mod", class: "cls", function: "fn", method: "mth",
  interface: "if", enum: "enum", struct: "st", trait: "tr", document: "doc", code: "?",
};

const GFX_KEY = "inflorescence.gfx.specimen";
const GFX_DEFAULTS = window.Pack ? window.Pack.GFX_DEFAULTS : {};

// Only keys we know, with the type we expect: a stale or hand-edited localStorage entry must
// never be able to break the view.
function loadGfx() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(GFX_KEY) || "{}") || {}; } catch { saved = {}; }
  const gfx = { ...GFX_DEFAULTS };
  for (const k of Object.keys(GFX_DEFAULTS)) {
    if (typeof saved[k] === typeof GFX_DEFAULTS[k]) gfx[k] = saved[k];
  }
  return gfx;
}
function saveGfx() {
  try { localStorage.setItem(GFX_KEY, JSON.stringify(spec.gfx)); } catch { /* storage off: not fatal */ }
}

const state = {
  projects: [],
  project: null,
  view: "overview",
  entityId: null,
  treeCache: new Map(),
  // Live-index polling: `key` is the last seen nodes:edges:summarized fingerprint, `indexing`
  // the last seen lease state (for edge detection), `merging` a reentry guard.
  live: { indexing: false, key: null, merging: false, lastMerge: 0 },
  freshness: null, // last on-demand drift check, or "checking" while one is in flight
};

// Everything the specimen view owns. `pack` is created once the view is first shown (a canvas
// in a display:none section has no size) and then kept for the life of the page.
const spec = {
  pack: null, G: null, meta: null, loading: false, error: null,
  level: "all", summarized: false, prefix: null,
  off: new Set(),                 // edge kinds switched off in the legend
  route: { a: null, b: null, arm: null },
  gfx: loadGfx(),
  wanted: null,                   // an entity to frame as soon as the data is here
  hits: null, hitLabel: "",
  replaying: false, recording: false,
};

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n) => Number(n || 0).toLocaleString("en-US").replace(/,/g, " ");

async function api(path, opts) {
  const res = await fetch(`/api/projects/${encodeURIComponent(state.project)}${path}`, opts);
  const data = await res.json();
  if (data && data.error && !opts?.allowError) throw new Error(data.error);
  return data;
}

function chip(type) {
  const t = TYPES.includes(type) ? type : "code";
  return `<span class="chip ${t}" title="${esc(type)}">${CHIP_TEXT[t] || t}</span>`;
}
function fileRef(e) {
  if (!e.file_path) return "";
  return e.line_start ? `${e.file_path}:${e.line_start}` : e.file_path;
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

function nav(view, params = {}) {
  const q = new URLSearchParams(params).toString();
  location.hash = `#/p/${encodeURIComponent(state.project)}/${view}${q ? "?" + q : ""}`;
}

function parseHash() {
  const m = location.hash.match(/^#\/p\/([^/]+)\/([a-z]+)(?:\?(.*))?$/);
  if (!m) return null;
  return { project: decodeURIComponent(m[1]), view: m[2], params: new URLSearchParams(m[3] || "") };
}

async function route() {
  $("#search-results").classList.add("hidden");
  const parsed = parseHash();
  if (parsed && state.projects.some((p) => p.project === parsed.project)) {
    if (state.project !== parsed.project) {
      state.project = parsed.project;
      resetProjectState();
    }
    state.view = parsed.view;
  } else if (state.projects.length) {
    state.project = state.project || state.projects[0].project;
    state.view = "overview";
  } else {
    showView("empty");
    return;
  }
  $("#project-select").value = state.project;
  document.querySelectorAll("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === state.view));

  const params = parsed?.params || new URLSearchParams();
  showView(state.view);
  if (state.view === "overview") await renderOverview();
  else if (state.view === "explorer") await renderExplorer(params.get("id"));
  else if (state.view === "graph") await renderGraph(params);
  else if (state.view === "query") renderQueryView();
  else if (state.view === "tools") await renderToolsView(params);
}

function resetProjectState() {
  state.treeCache = new Map();
  state.live.key = null;
  $("#tree").innerHTML = "";
  $("#entity-content").innerHTML = `<div class="dim pad">Select an entity from the tree.</div>`;
  clearFreshness();
  // A specimen belongs to one project; keep nothing of the old one.
  if (spec.pack) { spec.pack.destroy(); spec.pack = null; }
  spec.G = null; spec.meta = null; spec.prefix = null; spec.wanted = null;
  spec.hits = null; spec.route = { a: null, b: null, arm: null };
  // A load still in flight belongs to the project we just left; let the next one start.
  spec.loading = false;
  $("#route-a").value = ""; $("#route-b").value = "";
  $("#route-out").innerHTML = "";
  closeRoutePanel();
  $("#graph-side").classList.add("hidden");
}

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  $(`#view-${name}`)?.classList.remove("hidden");
  // The canvas has no size while its section is display:none, so it measures itself
  // the moment the section is shown.
  if (name === "graph" && spec.pack) spec.pack.resize();
  if (name !== "graph") stopReplay();
}

function openEntity(id) { nav("explorer", { id }); }

// ---------------------------------------------------------------------------
// FIG. 01 — overview
// ---------------------------------------------------------------------------

function ledgerRow(label, value, opts = {}) {
  const bar = opts.pct !== undefined
    ? `<td class="barcell" style="color:${opts.color || "var(--dim)"}"><span class="bar" style="width:${Math.max(1, opts.pct)}%"></span></td>`
    : `<td class="lead"><i></i></td>`;
  return `<tr${opts.id ? ` data-id="${esc(opts.id)}"` : ""}${opts.cls ? ` class="${opts.cls}"` : ""}>
    <td>${label}</td>${bar}<td>${value}</td></tr>`;
}

async function renderOverview() {
  const el = $("#view-overview");
  el.innerHTML = `<div class="ov dim">reading the index…</div>`;
  const s = await api("/stats");
  const coverage = s.summarizable ? Math.round((100 * s.summarized) / s.summarizable) : 0;
  const maxType = Math.max(1, ...s.by_type.map((t) => t.count));
  const maxRel = Math.max(1, ...s.by_relation.map((r) => r.count));
  const maxLang = Math.max(1, ...s.languages.map((l) => l.files));

  const rootBanner = s.root_path_exists ? "" : `
    <div class="rootpath-banner">
      <span>Source root unknown — code is shown from stored chunks only. Absolute path of
      <b>${esc(s.display_name)}</b>:</span>
      <input id="rootpath-input" class="field" placeholder="/absolute/path/to/${esc(s.display_name)}" value="${esc(s.root_path || "")}">
      <button id="rootpath-save" class="btn accent">save</button>
    </div>`;

  el.innerHTML = `
    <div class="ov">
      <div class="specimen">
        <div>
          <div class="fig-num"><small>FIG. 01 · INDEXED SPECIMEN</small></div>
          <h1>${esc(s.display_name)}</h1>
          <div class="classline">
            <span><b>id</b>${esc(s.project)}</span>
            <span><b>root</b>${esc(s.root_path || "—")}${s.root_path_exists ? "" : " (not on disk)"}</span>
          </div>
        </div>
        <div class="fig-meta">${esc(s.languages.map((l) => `${l.language} ${l.files}`).join(" · ")) || "no files"}</div>
      </div>
      ${rootBanner}
      <div class="measures">
        <div class="figure"><span class="v">${fmt(s.entities)}</span><span class="k">ENTITIES</span></div>
        <div class="figure"><span class="v">${fmt(s.edges)}</span><span class="k">RELATIONS</span></div>
        <div class="figure"><span class="v">${fmt(s.files)}</span><span class="k">FILES</span></div>
        <div class="figure"><span class="v">${coverage}%</span><span class="k">SUMMARIZED</span>
          <span class="sub">${fmt(s.summarized)} / ${fmt(s.summarizable)}</span></div>
        <div class="figure"><span class="v">${fmt(s.chunks)}</span><span class="k">CODE CHUNKS</span>
          <span class="sub">${fmt(s.summary_embeddings)} summary embeddings</span></div>
      </div>
      <div class="ov-grid">
        <div>
          <div class="sect-head"><b>FIG. 02</b>COMPOSITION<span class="n">${s.by_type.length} types</span></div>
          <table class="ledger">${s.by_type.map((t) => ledgerRow(
            `${chip(t.type)} ${esc(t.type)}`, fmt(t.count), { pct: (100 * t.count) / maxType })).join("") ||
            `<tr><td class="dim">no entities</td></tr>`}</table>
        </div>
        <div>
          <div class="sect-head"><b>FIG. 03</b>CONNECTIONS<span class="n">${fmt(s.edges)} edges</span></div>
          <table class="ledger">${s.by_relation.map((r) => ledgerRow(
            `<span class="rel-tag rel-${esc(r.relation)}">${esc(r.relation)}</span>`, fmt(r.count),
            { pct: (100 * r.count) / maxRel, color: REL_COLORS[r.relation] || "var(--dim)" })).join("") ||
            `<tr><td class="dim">no relations</td></tr>`}</table>
          <p class="fig-cap">CONTAINS is the structural spine and is drawn as nesting on the
          specimen, not as a line.</p>
        </div>
        <div>
          <div class="sect-head"><b>FIG. 04</b>LANGUAGES<span class="n">${s.languages.length}</span></div>
          <table class="ledger">${s.languages.map((l) => ledgerRow(
            esc(l.language), fmt(l.files), { pct: (100 * l.files) / maxLang })).join("") ||
            `<tr><td class="dim">no files</td></tr>`}</table>
        </div>
        <div>
          <div class="sect-head"><b>FIG. 05</b>MOST CALLED</div>
          <table class="ledger list">${s.most_called.map((h) => ledgerRow(
            `${chip(h.type)} ${esc(h.name)}`, `${h.callers}×`, { id: h.id, cls: "click" })).join("") ||
            `<tr><td class="dim">no CALLS edges</td></tr>`}</table>
        </div>
        <div>
          <div class="sect-head"><b>FIG. 06</b>MOST IMPORTED</div>
          <table class="ledger list">${s.most_imported.map((h) => ledgerRow(
            `${chip(h.type)} ${esc(h.name)}`, `${h.importers}×`, { id: h.id, cls: "click" })).join("") ||
            `<tr><td class="dim">no IMPORTS edges</td></tr>`}</table>
        </div>
        <div>
          <div class="sect-head"><b>FIG. 07</b>LARGEST FILES</div>
          <table class="ledger list">${s.largest_files.map((f) => ledgerRow(
            `${chip("module")} ${esc(String(f.file_path).split("/").pop())}`, `${f.entities}`,
            { id: f.module_id || "", cls: f.module_id ? "click" : "" })).join("") ||
            `<tr><td class="dim">no files</td></tr>`}</table>
        </div>
      </div>
    </div>`;

  el.querySelectorAll("tr[data-id]").forEach((row) => {
    if (!row.dataset.id) return;
    row.style.cursor = "pointer";
    row.addEventListener("click", () => openEntity(row.dataset.id));
  });
  $("#rootpath-save")?.addEventListener("click", async () => {
    const path = $("#rootpath-input").value.trim();
    if (!path) return;
    const res = await api("/root", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }), allowError: true,
    });
    if (res.ok) renderOverview();
    else alert(res.error || "Failed to set root path");
  });
}

// ---------------------------------------------------------------------------
// FIG. 02 — tree
// ---------------------------------------------------------------------------

async function fetchChildren(parentId) {
  const key = parentId ?? " root";
  if (!state.treeCache.has(key)) {
    const params = parentId ? `?parent=${encodeURIComponent(parentId)}` : "";
    state.treeCache.set(key, await api(`/tree${params}`));
  }
  return state.treeCache.get(key);
}

function treeNodeEl(node) {
  const wrap = document.createElement("div");
  wrap.className = "tnode";
  wrap.dataset.id = node.id;
  const hasKids = node.child_count > 0;
  const row = document.createElement("div");
  row.className = "trow";
  row.title = node.summary || node.signature || node.name;
  row.innerHTML = `
    <span class="twisty">${hasKids ? "▸" : ""}</span>
    ${chip(node.type)}
    <span class="tname">${esc(node.name)}</span>
    ${hasKids ? `<span class="tcount">${node.child_count}</span>` : ""}`;
  wrap.appendChild(row);
  const kids = document.createElement("div");
  kids.className = "tkids hidden";
  wrap.appendChild(kids);

  row.querySelector(".twisty").addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (hasKids) toggleTreeNode(wrap, node.id);
  });
  row.addEventListener("click", () => openEntity(node.id));
  row.addEventListener("dblclick", () => hasKids && toggleTreeNode(wrap, node.id));
  return wrap;
}

async function toggleTreeNode(wrap, id, forceOpen = false) {
  const kids = wrap.querySelector(":scope > .tkids");
  const twisty = wrap.querySelector(":scope > .trow .twisty");
  const isOpen = !kids.classList.contains("hidden");
  if (isOpen && !forceOpen) {
    kids.classList.add("hidden");
    twisty.textContent = "▸";
    return;
  }
  if (!kids.dataset.loaded) {
    const data = await fetchChildren(id);
    for (const child of data.children) kids.appendChild(treeNodeEl(child));
    kids.dataset.loaded = "1";
  }
  kids.classList.remove("hidden");
  twisty.textContent = "▾";
}

async function ensureTreeRoot() {
  const tree = $("#tree");
  if (tree.childElementCount) return;
  const data = await fetchChildren(null);
  if (!data.root) {
    tree.innerHTML = `<div class="dim pad">Project has no root node.</div>`;
    return;
  }
  const rootEl = treeNodeEl({ ...data.root, child_count: data.children.length });
  tree.appendChild(rootEl);
  const kids = rootEl.querySelector(":scope > .tkids");
  for (const child of data.children) kids.appendChild(treeNodeEl(child));
  kids.dataset.loaded = "1";
  kids.classList.remove("hidden");
  rootEl.querySelector(":scope > .trow .twisty").textContent = "▾";
}

async function revealInTree(breadcrumbs, targetId) {
  await ensureTreeRoot();
  let scope = $("#tree");
  for (const crumb of breadcrumbs) {
    const wrap = scope.querySelector(`:scope > .tnode[data-id="${CSS.escape(crumb.id)}"]`)
      || scope.querySelector(`:scope > .tkids > .tnode[data-id="${CSS.escape(crumb.id)}"]`);
    if (!wrap) return;
    if (crumb.id !== targetId) await toggleTreeNode(wrap, crumb.id, true);
    scope = wrap.querySelector(":scope > .tkids") || wrap;
  }
  document.querySelectorAll("#tree .trow.active").forEach((r) => r.classList.remove("active"));
  const target = $("#tree").querySelector(`.tnode[data-id="${CSS.escape(targetId)}"] > .trow`);
  if (target) {
    target.classList.add("active");
    target.scrollIntoView({ block: "nearest" });
  }
}

// ---------------------------------------------------------------------------
// Explorer / entity detail
// ---------------------------------------------------------------------------

async function renderExplorer(entityId) {
  await ensureTreeRoot();
  if (!entityId) return;
  state.entityId = entityId;
  // Selecting an entity anywhere aims the specimen at it: if the graph is already
  // built it flies there now, otherwise it is framed the moment it loads.
  aimSpecimen(entityId);
  const pane = $("#entity-content");
  pane.classList.remove("dim", "pad");
  pane.innerHTML = `<div class="dim pad">reading…</div>`;
  let data;
  try {
    data = await api(`/entity?id=${encodeURIComponent(entityId)}`);
  } catch (e) {
    pane.innerHTML = `<div class="pad dim">Entity not found: <code>${esc(entityId)}</code></div>`;
    return;
  }
  const e = data.entity;
  revealInTree(data.breadcrumbs, entityId);

  const crumbs = data.breadcrumbs.map((c, i) =>
    i === data.breadcrumbs.length - 1
      ? `<span style="color:var(--ink)">${esc(c.name)}</span>`
      : `<a href="#" data-id="${esc(c.id)}">${esc(c.name)}</a>`
  ).join(`<span class="sep">/</span>`);

  const relGroup = (rels, dir) => {
    if (!rels.length) return `<div class="dim">none</div>`;
    const byType = {};
    rels.forEach((r) => (byType[r.relation] = byType[r.relation] || []).push(r));
    return Object.entries(byType).map(([rel, items]) => `
      <div class="rel-group">
        <span class="rel-tag rel-${esc(rel)}">${dir === "in" ? "←" + esc(rel) + "─" : "─" + esc(rel) + "→"} ${items.length}</span>
        ${items.map((r) => `
          <div class="rel-row" data-rel-id="${esc(r.id)}" title="${esc(r.summary || "")}">
            ${chip(r.type)}<span class="name" data-id="${esc(r.id)}">${esc(r.name)}</span>
            <span class="path">${esc(fileRef(r))}</span>
            <span class="see" data-see="${esc(r.id)}" title="Show on the specimen">◎</span>
          </div>`).join("")}
      </div>`).join("");
  };

  pane.innerHTML = `
    <div class="entity">
      <div class="crumbs">${crumbs}</div>
      <div class="entity-head">${chip(e.type)}<h1>${esc(e.name)}</h1></div>
      <div class="entity-meta">${esc(fileRef(e))}${e.line_end ? "–" + e.line_end : ""} · ${esc(e.id)}
        ${data.chunks ? ` · ${data.chunks} chunk${data.chunks === 1 ? "" : "s"}` : ""}</div>
      <div class="entity-actions">
        <button class="btn accent" id="btn-show-graph">show on specimen</button>
        <button class="btn" id="btn-route-from">route from here</button>
        <button class="btn" id="btn-copy-id">copy id</button>
      </div>
      ${e.summary
        ? `<div class="sect"><div class="sect-head">SUMMARY · WRITTEN BY THE INDEXER</div><div class="summary-card">${esc(e.summary)}</div></div>`
        : `<div class="sect"><div class="sect-head">SUMMARY</div><div class="dim">no summary yet — this entity has not been summarized.</div></div>`}
      ${e.signature ? `<div class="sect"><div class="sect-head">SIGNATURE</div><div class="sig-block">${esc(e.signature)}</div></div>` : ""}
      ${e.docstring ? `<div class="sect"><div class="sect-head">DOCSTRING</div><div class="doc-block">${esc(e.docstring)}</div></div>` : ""}
      <div class="sect" id="code-sect"><div class="sect-head">CODE</div><div class="dim">reading…</div></div>
      ${data.children.length ? `
        <div class="sect"><div class="sect-head">CONTAINS<span class="n">${data.children.length}</span></div>
          <div class="kids-grid">${data.children.map((c) => `
            <span class="kid-chip" data-id="${esc(c.id)}" title="${esc(c.summary || "")}">${chip(c.type)}<span>${esc(c.name)}</span></span>`).join("")}
          </div></div>` : ""}
      <div class="sect"><div class="sect-head">RELATIONS<span class="n">${data.incoming.length} in · ${data.outgoing.length} out</span></div>
        <div class="rel-grid">
          <div>${relGroup(data.outgoing, "out")}</div>
          <div>${relGroup(data.incoming, "in")}</div>
        </div>
      </div>
    </div>`;

  pane.querySelectorAll("[data-id]").forEach((el) =>
    el.addEventListener("click", (ev) => { ev.preventDefault(); openEntity(el.dataset.id); }));
  // Hovering a relation puts a crosshair on that node, so switching to the specimen
  // lands on something already marked.
  pane.querySelectorAll("[data-rel-id]").forEach((el) =>
    el.addEventListener("mouseenter", () => markOnSpecimen(el.dataset.relId)));
  pane.querySelectorAll("[data-see]").forEach((el) =>
    el.addEventListener("click", (ev) => { ev.stopPropagation(); nav("graph", { id: el.dataset.see }); }));
  $("#btn-show-graph").addEventListener("click", () => nav("graph", { id: e.id }));
  $("#btn-route-from").addEventListener("click", () => nav("graph", { id: e.id, route: "1" }));
  $("#btn-copy-id").addEventListener("click", () => navigator.clipboard.writeText(e.id));

  loadEntityCode(e, $("#code-sect"));
}

async function loadEntityCode(entity, sect, wholeFile = false) {
  const res = await api(`/code?id=${encodeURIComponent(entity.id)}${wholeFile ? "&whole_file=1" : ""}`, { allowError: true });
  const head = `<div class="sect-head">CODE</div>`;
  if (!res.content) {
    const reasons = {
      directory: "Directories have no source of their own — pick a file or an entity inside one.",
      no_root_path: "No source on disk and no stored chunks. Set the project root path on the overview.",
      no_code: "No stored code for this entity.",
      entity_not_found: "Entity not found.",
    };
    sect.innerHTML = `${head}<div class="dim">${reasons[res.reason] || "No code available."}</div>`;
    return;
  }
  const lines = res.content.split("\n");
  let highlighted;
  try {
    highlighted = res.language
      ? hljs.highlight(res.content, { language: res.language }).value
      : hljs.highlightAuto(res.content).value;
  } catch { highlighted = esc(res.content); }
  const gutter = lines.map((_, i) => res.first_line + i).join("\n");
  const isPartial = res.source === "file" && !wholeFile && entity.type !== "module" &&
    res.total_lines && lines.length < res.total_lines;
  sect.innerHTML = `
    ${head}
    <div class="code-card">
      <div class="code-head">
        <span>${esc(res.file_path || "")} · ${res.first_line}–${res.first_line + lines.length - 1}${res.total_lines ? " of " + res.total_lines : ""}</span>
        <span class="grow"></span>
        ${res.source === "chunks" ? `<span title="Reassembled from stored chunks — may be approximate">from chunks ≈</span>` : ""}
        ${isPartial ? `<button class="ghost-btn" id="btn-whole-file">whole file</button>` : ""}
        <button class="ghost-btn" id="btn-copy-code">copy</button>
      </div>
      <div class="code-scroll"><div class="code-grid">
        <div class="code-gutter">${gutter}</div>
        <pre class="code-body"><code class="hljs">${highlighted}</code></pre>
      </div></div>
    </div>`;
  sect.querySelector("#btn-copy-code").addEventListener("click", () => navigator.clipboard.writeText(res.content));
  sect.querySelector("#btn-whole-file")?.addEventListener("click", () => loadEntityCode(entity, sect, true));
}

// ---------------------------------------------------------------------------
// FIG. 03 — the specimen
//
// One Pack instance for the life of the page. Everything the toolbar, the legend, the route
// tool and the live poll do goes through it; none of them touch the canvas directly.
// ---------------------------------------------------------------------------

function aimSpecimen(id) {
  spec.wanted = id;
  if (spec.pack && spec.G && spec.G.byId.has(id)) spec.pack.focus(id, { pin: true });
}
function markOnSpecimen(id) {
  if (spec.pack && spec.G) spec.pack.mark = spec.G.byId.get(id) || null;
}

function specimenURL() {
  const q = new URLSearchParams();
  if (spec.level === "modules") q.set("level", "modules");
  if (spec.summarized) q.set("summarized", "1");
  if (spec.prefix) q.set("prefix", spec.prefix);
  const s = q.toString();
  return `/specimen${s ? "?" + s : ""}`;
}

async function loadSpecimen(opt = {}) {
  if (spec.loading) return;
  spec.loading = true;
  const first = !spec.pack;
  // Which project this payload is for. A project switch mid-flight must not install one
  // repository's graph under another's name.
  const forProject = state.project;
  if (!opt.quiet) $("#graph-loading").classList.remove("hidden");
  try {
    const raw = await api(specimenURL());
    if (forProject !== state.project) return;
    const G = window.Pack.model(raw);
    spec.meta = raw;
    spec.G = G;
    spec.error = null;
    if (!spec.pack) {
      spec.pack = new window.Pack($("#graph-canvas"), G, {
        reduced: matchMedia("(prefers-reduced-motion: reduce)").matches,
        gfx: spec.gfx,
        onSelect: (n) => onSpecimenSelect(n),
        onHover: () => {},
      });
      spec.pack.setGfx(spec.gfx);
      wireNoteActions();
    } else {
      spec.pack.setData(G, { snap: false });
    }
    for (const k of ["C", "I", "N", "M"]) spec.pack.setShow(k, !spec.off.has(k));
    if (spec.hits) spec.pack.setHits(spec.hits, spec.hitLabel, first);
    if (spec.wanted && G.byId.has(spec.wanted)) {
      spec.pack.focus(spec.wanted, { pin: true });
      onSpecimenSelect(G.byId.get(spec.wanted));
      spec.wanted = null;
    }
    buildLegend();
    updateGraphCount();
    refreshRouteOut();
  } catch (e) {
    spec.error = String(e.message || e);
    $("#graph-count").textContent = "specimen unavailable — retrying…";
  } finally {
    spec.loading = false;
    $("#graph-loading").classList.add("hidden");
  }
}

// The card is a DOM layer over the canvas, so its links are real links.
function wireNoteActions() {
  spec.pack.note.addEventListener("click", (ev) => {
    const act = ev.target.closest("[data-act]");
    if (act) {
      ev.preventDefault();
      const n = spec.pack.pinned;
      if (!n) return;
      if (act.dataset.act === "explorer") openEntity(n.id);
      else if (act.dataset.act === "route") { openRoutePanel(); setRouteEnd("a", n); }
      else if (act.dataset.act === "scope") scopeTo(n);
      return;
    }
    const hit = ev.target.closest("[data-id]");
    if (hit) { ev.preventDefault(); spec.pack.focus(hit.dataset.id, { pin: true }); onSpecimenSelect(spec.G.byId.get(hit.dataset.id)); }
  });
}

function scopeTo(n) {
  const path = n.t === "directory" ? n.p : (n.p || "").split("/").slice(0, -1).join("/");
  spec.prefix = path && path !== "." ? path : null;
  spec.wanted = n.id;
  loadSpecimen();
}

function updateGraphCount() {
  const p = spec.pack, m = spec.meta;
  if (!p || !m) return;
  const st = p.stats();
  const of = m.totals ? Object.values(m.totals).reduce((a, b) => a + b, 0) : st.nodes;
  // Say what is drawn AND what the index holds: a filtered view must not read as the whole one.
  const nodes = st.nodes < of ? `${fmt(st.nodes)} of ${fmt(of)} nodes` : `${fmt(st.nodes)} nodes`;
  const scope = m.scope ? ` · scoped to ${m.scope}` : "";  // textContent, not markup
  $("#graph-count").textContent =
    `${nodes} · ${fmt(st.edges)} edges${scope} · layout ${st.layoutMs} ms · ${st.frameMs} ms/frame`;
  $("#graph-scope").innerHTML = m.scope
    ? `SPECIMEN · <b>${esc(m.scope)}/</b>`
    : `SPECIMEN · <b>${esc(m.display_name || "")}</b>`;
  const notes = (m.notes || []).join(" ");
  $("#graph-hint").textContent = notes ||
    "drag to pan · wheel to zoom · click to pin · double-click a disc to frame it · esc";
}

function buildLegend() {
  const p = spec.pack;
  if (!p) return;
  const rows = [["C", "CALLS"], ["I", "IMPORTS"], ["N", "INHERITS"], ["M", "IMPLEMENTS"]]
    .filter(([k]) => p.counts[k])
    .map(([k, name]) => `<span class="lg${spec.off.has(k) ? " off" : ""}" data-kind="${k}"
      style="color:${REL_COLORS[name]}"><i></i>${name} ${fmt(p.counts[k])}</span>`)
    .join("");
  $("#graph-legend").innerHTML = rows +
    `<div class="marks">○ directory / file / class disc &nbsp;·&nbsp; · function &nbsp;◇ class &nbsp;□ document<br>
     nesting = CONTAINS ${fmt((spec.meta?.edge_totals || {}).CONTAINS || 0)}</div>`;
  $("#graph-legend").querySelectorAll(".lg").forEach((el) =>
    el.addEventListener("click", () => {
      const k = el.dataset.kind;
      const off = !spec.off.has(k);
      if (off) spec.off.add(k); else spec.off.delete(k);
      el.classList.toggle("off", off);
      spec.pack.setShow(k, !off);
    }));
}

async function onSpecimenSelect(n) {
  const side = $("#graph-side");
  if (!n) { side.classList.add("hidden"); return; }
  side.classList.remove("hidden");
  const box = $("#graph-side-content");
  box.innerHTML = `<div class="gs-loading">reading…</div>`;
  let detail;
  try {
    detail = await api(`/entity?id=${encodeURIComponent(n.id)}`);
  } catch {
    box.innerHTML = `<div class="gs-loading">Entity not found.</div>`;
    return;
  }
  if (!spec.pack.pinned || spec.pack.pinned.id !== n.id) return; // stale
  const e = detail.entity;
  const relGroup = (rels, dir) => {
    if (!rels.length) return "";
    const byType = {};
    rels.forEach((r) => (byType[r.relation] = byType[r.relation] || []).push(r));
    return Object.entries(byType).map(([rel, items]) => `
      <div class="rel-group">
        <span class="rel-tag rel-${esc(rel)}">${dir === "in" ? "←" + esc(rel) + "─" : "─" + esc(rel) + "→"} ${items.length}</span>
        ${items.map((r) => `
          <div class="rel-row"><span class="name" data-id="${esc(r.id)}" title="${esc(r.summary || "")}">${esc(r.name)}</span>
          <span class="path">${esc(fileRef(r))}</span></div>`).join("")}
      </div>`).join("");
  };
  box.innerHTML = `
    <div class="gs">
      <div class="gs-head">${chip(e.type)}<h3>${esc(e.name)}</h3></div>
      <div class="gs-crumbs">${detail.breadcrumbs.map((c) => esc(c.name)).join(" / ")}</div>
      <div class="gs-meta">${esc(fileRef(e))}${e.line_end ? "–" + e.line_end : ""}</div>
      <div class="gs-actions">
        <button class="btn accent" id="gs-open">open in explorer</button>
        <button class="btn" id="gs-route">route from here</button>
        <button class="btn" id="gs-scope">scope here</button>
      </div>
      ${e.summary ? `<div class="gs-sect"><h4>SUMMARY</h4><div class="gs-summary">${esc(e.summary)}</div></div>`
        : `<div class="gs-sect"><h4>SUMMARY</h4><div class="dim">no summary yet</div></div>`}
      ${e.signature ? `<div class="gs-sect"><h4>SIGNATURE</h4><div class="sig-block">${esc(e.signature)}</div></div>` : ""}
      ${detail.outgoing.length ? `<div class="gs-sect"><h4>OUTGOING ${detail.outgoing.length}</h4>${relGroup(detail.outgoing, "out")}</div>` : ""}
      ${detail.incoming.length ? `<div class="gs-sect"><h4>INCOMING ${detail.incoming.length}</h4>${relGroup(detail.incoming, "in")}</div>` : ""}
    </div>`;
  $("#gs-open").addEventListener("click", () => openEntity(n.id));
  $("#gs-route").addEventListener("click", () => { openRoutePanel(); setRouteEnd("a", n); });
  $("#gs-scope").addEventListener("click", () => scopeTo(n));
  box.querySelectorAll("[data-id]").forEach((el) =>
    el.addEventListener("click", () => {
      const target = spec.G.byId.get(el.dataset.id);
      if (target) { spec.pack.focus(target.id, { pin: true }); onSpecimenSelect(target); }
      else openEntity(el.dataset.id); // filtered out of this view — the explorer still has it
    }));
}

// ---- route: the shortest way from A to B over the real edges -----------------------------

function openRoutePanel() { $("#route-panel").classList.remove("hidden"); $("#graph-route").classList.add("on"); }
function closeRoutePanel() {
  $("#route-panel").classList.add("hidden");
  $("#graph-route").classList.remove("on");
  spec.route.arm = null;
  document.querySelectorAll(".route-pick").forEach((r) => r.classList.remove("arm"));
}

function setRouteEnd(which, node) {
  spec.route[which] = node;
  $(`#route-${which}`).value = node ? node.id : "";
  $(`#route-${which}-sugg`).classList.add("hidden");
  document.querySelectorAll(".route-pick").forEach((r) => r.classList.remove("arm"));
  spec.route.arm = null;
  computeRoute();
}

function computeRoute() {
  const { a, b } = spec.route;
  const out = $("#route-out");
  if (!a || !b) {
    out.innerHTML = `<small>PICK TWO ENTITIES</small><span class="dim">Click <b>pick</b> and then a node on the specimen, or type a name.</span>`;
    spec.pack?.clearRoute();
    return;
  }
  const R = spec.pack.routeBetween(a, b);
  if (!R) {
    out.innerHTML = `<small>NO ROUTE</small><span class="dim">Nothing in this graph connects them — not even through the tree. That is the answer, not a failure.</span>`;
    spec.pack.clearRoute();
    return;
  }
  spec.pack.setRoute(R, true);
  out.innerHTML = `<small>${R.steps.length} EDGE${R.steps.length === 1 ? "" : "S"} OF THIS GRAPH · COST ${Math.round(R.cost * 10) / 10}</small>` +
    R.steps.map((st) => `<div>
      <span data-id="${esc(st.a.id)}">${esc(spec.pack.shortName(st.a))}</span>
      <em class="rel-tag rel-${REL_KIND[st.k]}">${st.dir === ">" ? "─" + REL_KIND[st.k] + "→" : "←" + REL_KIND[st.k] + "─"}</em>
      <span data-id="${esc(st.b.id)}">${esc(spec.pack.shortName(st.b))}</span>
    </div>`).join("");
  out.querySelectorAll("[data-id]").forEach((el) =>
    el.addEventListener("click", () => { spec.pack.focus(el.dataset.id, { pin: true }); onSpecimenSelect(spec.G.byId.get(el.dataset.id)); }));
}

function refreshRouteOut() {
  // A rebuild replaces every node object; re-resolve the ends by id or drop them.
  for (const which of ["a", "b"]) {
    const n = spec.route[which];
    if (n) spec.route[which] = spec.G.byId.get(n.id) || null;
  }
  if (!$("#route-panel").classList.contains("hidden")) computeRoute();
}

function routeSuggest(which) {
  const input = $(`#route-${which}`);
  const box = $(`#route-${which}-sugg`);
  const q = input.value.trim().toLowerCase();
  if (!q || !spec.G) { box.classList.add("hidden"); return; }
  const hits = [];
  for (const n of spec.G.nodes) {
    if (n.id.toLowerCase().includes(q) || n.n.toLowerCase().includes(q)) hits.push(n);
    if (hits.length >= 40) break;
  }
  if (!hits.length) { box.innerHTML = `<div class="dim">no entity matches</div>`; box.classList.remove("hidden"); return; }
  box.innerHTML = hits.map((n) => `<div data-id="${esc(n.id)}">${chip(n.t)} ${esc(n.n)} <span class="dim">${esc(n.p)}</span></div>`).join("");
  box.classList.remove("hidden");
  box.querySelectorAll("[data-id]").forEach((el) =>
    el.addEventListener("click", () => setRouteEnd(which, spec.G.byId.get(el.dataset.id))));
}

// ---- display panel ------------------------------------------------------------------------

const GFX_SCHEMA = [
  { k: "labels", label: "labels", type: "bool", hint: "Names on the specimen, as many as fit without overlapping." },
  { k: "edgeAlpha", label: "edge ink", type: "range", min: 0, max: 2, step: .05, hint: "Multiplies the zoom-dependent opacity of the bundles. 0 leaves only the discs." },
  { k: "pulse", label: "idle pulse", type: "bool", hint: "When nothing is selected a random node breathes its links — the index is live, not a picture." },
  { k: "bloom", label: "bloom on load", type: "bool", hint: "Discs open out of their parents when the specimen is first drawn." },
  { k: "glitch", label: "row tearing", type: "bool", hint: "Rare, short, on the canvas only — never on a card you have to read." },
  { k: "grain", label: "scanlines", type: "bool", hint: "The dirt is live, not baked. Off is a clean canvas." },
  { k: "paper", label: "paper", type: "bool", hint: "The herbarium sheet: the same specimen in ink on grey paper." },
];

function renderGfxPanel() {
  const g = spec.gfx;
  $("#gfx-body").innerHTML = GFX_SCHEMA.map((it) => {
    const title = it.hint ? ` title="${esc(it.hint)}"` : "";
    if (it.type === "bool") {
      return `<label class="gfx-row"${title}><span class="gfx-k">${esc(it.label)}</span>
        <input type="checkbox" data-k="${it.k}"${g[it.k] ? " checked" : ""}></label>`;
    }
    return `<label class="gfx-row"${title}><span class="gfx-k">${esc(it.label)}</span>
      <input type="range" data-k="${it.k}" min="${it.min}" max="${it.max}" step="${it.step}" value="${g[it.k]}">
      <output class="gfx-v">${+Number(g[it.k]).toFixed(2)}</output></label>`;
  }).join("") + `<p class="gfx-note">Display only. Nothing here changes what was fetched or indexed.</p>`;
  $("#gfx-body").querySelectorAll("[data-k]").forEach((el) =>
    el.addEventListener("input", () => {
      const v = el.type === "checkbox" ? el.checked : +el.value;
      spec.gfx[el.dataset.k] = v;
      const out = el.parentElement.querySelector(".gfx-v");
      if (out) out.textContent = String(+Number(v).toFixed(2));
      saveGfx();
      spec.pack?.setGfx(spec.gfx);
    }));
}

// ---- replay: the bloom, scrubbed ----------------------------------------------------------

function startReplay() {
  if (!spec.pack) return;
  spec.replaying = true;
  spec.pack.replayStart(14 / (+$("#replay-speed").value || 1));
  $("#replay-bar").classList.remove("hidden");
  $("#graph-replay").classList.add("on");
  $("#replay-toggle").textContent = "❚❚";
  tickReplay();
}
function stopReplay() {
  if (!spec.replaying) return;
  spec.replaying = false;
  spec.pack?.replayStop();
  $("#replay-bar").classList.add("hidden");
  $("#graph-replay").classList.remove("on");
  cancelAnimationFrame(spec.replayRaf);
}
function tickReplay() {
  if (!spec.replaying || !spec.pack) return;
  const p = spec.pack.bloomProgress(performance.now());
  $("#replay-fill").style.width = `${Math.round(p * 100)}%`;
  $("#replay-label").textContent = `${Math.round(p * 100)}% of ${fmt(spec.G.nodes.length)} entities`;
  if (p >= 1 && spec.pack.bloom.playing) { setTimeout(stopReplay, 900); return; }
  spec.replayRaf = requestAnimationFrame(tickReplay);
}

function toggleRecording() {
  const canvas = $("#graph-canvas canvas");
  const btn = $("#graph-rec");
  if (!window.GraphRecorder || !window.GraphRecorder.isSupported()) {
    $("#graph-hint").textContent = "This browser cannot record a canvas (no captureStream/MediaRecorder).";
    return;
  }
  if (window.GraphRecorder.active()) {
    window.GraphRecorder.stop();
    btn.classList.remove("on");
    btn.textContent = "rec";
    return;
  }
  const ok = window.GraphRecorder.start(canvas, {
    name: `${state.project}-specimen`,
    onStop: (r) => { btn.classList.remove("on"); btn.textContent = "rec"; $("#graph-hint").textContent = `saved ${r.filename} · ${(r.bytes / 1e6).toFixed(1)} MB`; },
    onError: (msg) => { btn.classList.remove("on"); btn.textContent = "rec"; $("#graph-hint").textContent = msg; },
  });
  if (ok) { btn.classList.add("on"); btn.textContent = "■ rec"; }
}

async function renderGraph(params) {
  if (!window.Pack) {
    $("#graph-canvas").innerHTML = `<div class="pad dim">pack.js failed to load.</div>`;
    return;
  }
  const id = params.get("id");
  const prefix = params.get("prefix");
  if (prefix !== null && prefix !== spec.prefix) { spec.prefix = prefix || null; if (spec.pack) { spec.wanted = id || spec.wanted; await loadSpecimen(); } }
  if (id) spec.wanted = id;
  $("#graph-level").value = spec.level;
  $("#graph-summarized").checked = spec.summarized;
  if (!spec.pack) await loadSpecimen();
  else {
    spec.pack.resize();
    if (spec.wanted && spec.G.byId.has(spec.wanted)) {
      spec.pack.focus(spec.wanted, { pin: true });
      onSpecimenSelect(spec.G.byId.get(spec.wanted));
      spec.wanted = null;
    }
  }
  if (params.get("route") === "1") { openRoutePanel(); if (spec.pack?.pinned) setRouteEnd("a", spec.pack.pinned); }
}


// ---------------------------------------------------------------------------
// Live index: poll /status, and let the specimen grow while the indexer runs
// ---------------------------------------------------------------------------

// Nesting depends on every sibling, so a landed entity cannot be grafted onto the layout —
// the model is rebuilt from a fresh payload instead. That costs a fetch and a few hundred
// milliseconds of layout, so it is throttled; the camera, the pinned node and the route
// survive it, and everything that was not there a moment ago opens with a ring.
const LIVE_MERGE_MS = 5000;

async function liveMergeSpecimen() {
  if (state.live.merging || !spec.pack || spec.loading) return;
  if (performance.now() - state.live.lastMerge < LIVE_MERGE_MS) return;
  state.live.merging = true;
  state.live.lastMerge = performance.now();
  try {
    await loadSpecimen({ quiet: true });
  } finally {
    state.live.merging = false;
  }
}

function updateLiveBadge(s) {
  const badge = $("#live-badge");
  if (!badge) return;
  if (!s.indexing) { badge.classList.add("hidden"); return; }
  badge.classList.remove("hidden");
  $("#live-badge-text").textContent = `indexing ${fmt(s.summarized)}/${fmt(s.summarizable)}`;
  const pct = s.summarizable ? Math.round((s.summarized / s.summarizable) * 100) : 0;
  $("#live-fill").style.width = `${pct}%`;
}

// The badge answers two different questions with one control. Passive: is anything keeping this
// index current (a live daemon, a registration nobody is running, or nothing at all). On click:
// how far the index has actually drifted from disk right now — a real scan, too expensive for
// the 2 s poll, so it is asked for rather than shown by default.
function updateWatchBadge(s) {
  const badge = $("#watch-badge");
  if (!badge) return;
  const w = (s && s.watch) || null;
  if (!w) { badge.classList.add("hidden"); return; }
  badge.classList.remove("hidden", "live", "idle");
  if (w.live) {
    badge.classList.add("live");
    $("#watch-badge-text").textContent = "watched";
    badge.title = `A watch daemon is following this tree (${w.holder || "unknown holder"}) — `
      + "saves are re-indexed. Click to verify against disk.";
  } else if (w.watched) {
    badge.classList.add("idle");
    $("#watch-badge-text").textContent = "watch idle";
    badge.title = "Registered for the watch daemon, but no daemon is running it: this index is "
      + "as fresh as its last index and no fresher. Start it with `inflorescence watch run`. "
      + "Click to check drift.";
  } else {
    $("#watch-badge-text").textContent = "unwatched";
    badge.title = "Nothing is keeping this index current — it changes only when something "
      + "re-indexes it. Follow it with `inflorescence watch add <path>`. Click to check drift.";
  }
}

// Freshness is a scan, so its answer ages: drop it whenever the project changes or an index run
// ends, rather than leaving a reassuring "up to date" next to a graph that just moved.
function clearFreshness() {
  state.freshness = null;
  const el = $("#watch-fresh");
  if (el) el.textContent = "";
}

async function checkFreshness() {
  const el = $("#watch-fresh");
  if (!el || !state.project || state.freshness === "checking") return;
  state.freshness = "checking";
  el.textContent = "· checking…";
  try {
    const res = await fetch(`/api/projects/${encodeURIComponent(state.project)}/freshness`);
    const f = await res.json();
    if (!f || f.error) { el.textContent = "· check failed"; state.freshness = null; return; }
    if (!f.available) el.textContent = "· no tree on disk";
    else if (!f.pending) el.textContent = "· up to date";
    else {
      const cost = f.estimated_cost_usd ? ` ≈$${f.estimated_cost_usd.toFixed(2)}` : "";
      el.textContent = `· ${f.pending} file(s) behind${cost}`;
    }
    state.freshness = f;
  } catch (e) {
    el.textContent = "· check failed";
    state.freshness = null;
  }
}

async function refreshProjectSelect() {
  try {
    const res = await fetch("/api/projects");
    const projects = (await res.json()).projects || [];
    if (!projects.length) return;
    state.projects = projects;
    const sel = $("#project-select");
    sel.innerHTML = projects.map((p) =>
      `<option value="${esc(p.project)}">${esc(p.display_name)} · ${fmt(p.entities)}</option>`).join("");
    if (state.project) sel.value = state.project;
  } catch (e) { /* next tick retries */ }
}

// Always-on 2 s heartbeat. Cheap when idle (one lock probe + two aggregations, no specimen
// refetch unless the counts fingerprint moved). Raw fetch, not api(): a dead Memgraph must
// degrade to silence, not an error alert every two seconds.
async function pollStatus() {
  if (document.hidden) return;
  if (!state.project) {
    if (!state.projects.length) {
      await refreshProjectSelect();
      if (state.projects.length) await route();
    }
    return;
  }
  // The hash may point at a project that appeared after boot (a fresh index of a wiped or
  // brand-new repo): route() rejected it against the stale list, so refresh and re-route.
  const parsed = parseHash();
  if (parsed && parsed.project !== state.project
      && !state.projects.some((p) => p.project === parsed.project)) {
    await refreshProjectSelect();
    if (state.projects.some((p) => p.project === parsed.project)) { await route(); return; }
  }

  let s;
  try {
    const res = await fetch(`/api/projects/${encodeURIComponent(state.project)}/status`);
    s = await res.json();
  } catch (e) { return; }
  if (!s || s.error) return;

  updateLiveBadge(s);
  updateWatchBadge(s);
  $("#strip-r").textContent = `${fmt(s.nodes)} nodes · ${fmt(s.edges)} edges · ${fmt(s.summarized)}/${fmt(s.summarizable)} summarized`;

  const key = `${s.nodes}:${s.edges}:${s.summarized}`;
  const first = state.live.key === null;
  const changed = !first && key !== state.live.key;
  const finished = !s.indexing && state.live.indexing; // true→false edge
  state.live.key = key;
  state.live.indexing = s.indexing;

  if (state.view === "graph") {
    if (spec.error) await loadSpecimen({ quiet: true });
    else if (s.indexing && changed) await liveMergeSpecimen();
    else if (finished || (!s.indexing && changed)) { state.live.lastMerge = 0; await liveMergeSpecimen(); }
  } else if (finished && state.view === "overview") {
    renderOverview();
  }
  // Expire the drift answer when it stops being about the graph on screen — but never while a
  // check is in flight, and never merely because an *ongoing* index moved the counts.
  if (state.freshness !== "checking" && (finished || (changed && !s.indexing))) clearFreshness();
  if (finished) {
    state.treeCache = new Map();
    refreshProjectSelect();
  }
}

// ---------------------------------------------------------------------------
// FIG. 04 — Cypher
// ---------------------------------------------------------------------------

const QUERY_PRESETS = [
  { label: "— presets —", q: "" },
  { label: "All entities", q: `MATCH (n:Code {project: $project})\nRETURN n.name AS name, labels(n) AS labels, n.file_path AS file\nORDER BY file` },
  { label: "Most called", q: `MATCH (c:Code {project: $project})-[:CALLS]->(n:Code {project: $project})\nRETURN n.name AS name, n.file_path AS file, count(c) AS callers\nORDER BY callers DESC` },
  { label: "Call graph (as graph)", q: `MATCH (a:Code {project: $project})-[r:CALLS]->(b:Code {project: $project})\nRETURN a, r, b` },
  { label: "Import graph (as graph)", q: `MATCH (a:Code:Module {project: $project})-[r:IMPORTS]->(b:Code {project: $project})\nRETURN a, r, b` },
  { label: "Inheritance (as graph)", q: `MATCH (a:Code {project: $project})-[r:INHERITS|IMPLEMENTS]->(b:Code {project: $project})\nRETURN a, r, b` },
  { label: "Cross-file calls", q: `MATCH (a:Code {project: $project})-[:CALLS]->(b:Code {project: $project})\nWHERE a.file_path <> b.file_path\nRETURN a.name AS caller, a.file_path AS from, b.name AS callee, b.file_path AS to` },
  { label: "Entities without summary", q: `MATCH (n:Code {project: $project})\nWHERE NOT n:Directory AND (n.summary IS NULL OR n.summary = '')\nRETURN n.name AS name, labels(n) AS labels, n.file_path AS file` },
  { label: "Self-recursive functions", q: `MATCH (a:Code {project: $project})-[:CALLS]->(a)\nRETURN a.name AS name, a.file_path AS file` },
];

function renderQueryView() {
  const sel = $("#query-presets");
  if (!sel.childElementCount) {
    QUERY_PRESETS.forEach((p, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = p.label;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", () => {
      const p = QUERY_PRESETS[+sel.value];
      if (p && p.q) $("#query-input").value = p.q;
    });
    $("#query-run").addEventListener("click", runQuery);
    $("#query-input").addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") { ev.preventDefault(); runQuery(); }
    });
  }
}

async function runQuery() {
  const query = $("#query-input").value.trim();
  if (!query) return;
  const status = $("#query-status");
  status.className = "dim";
  status.textContent = "running…";
  $("#query-results").innerHTML = "";
  $("#query-graph-btn-wrap").innerHTML = "";
  const t0 = performance.now();
  const res = await api("/query", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit: 200 }), allowError: true,
  });
  if (res.error) {
    status.className = "error";
    status.textContent = res.error;
    return;
  }
  const ms = Math.round(performance.now() - t0);
  status.className = "dim";
  status.textContent = `${res.rows.length} row${res.rows.length === 1 ? "" : "s"}${res.has_more ? " (more available — refine or paginate)" : ""} · ${ms} ms`;

  // Node-bearing rows can be shown on the specimen: the answer, numbered, on the real thing.
  if (res.graph?.nodes?.length) {
    const btn = document.createElement("button");
    btn.className = "btn";
    btn.textContent = `show ${res.graph.nodes.length} on the specimen`;
    btn.addEventListener("click", () => {
      spec.hits = res.graph.nodes.slice(0, 60).map((n) => ({ id: n.id }));
      spec.hitLabel = "CYPHER RESULT";
      if (spec.pack) spec.pack.setHits(spec.hits, spec.hitLabel);
      nav("graph");
    });
    $("#query-graph-btn-wrap").appendChild(btn);
  }

  if (!res.rows.length) return;
  const cols = res.columns;
  const cell = (v) => {
    if (v === null || v === undefined) return `<span class="dim">∅</span>`;
    if (typeof v === "object" && v.__kind === "node")
      return `<span class="cell-node" data-id="${esc(v.id)}">${chip(v.type)}${esc(v.name ?? v.id)}</span>`;
    if (typeof v === "object" && v.__kind === "relationship")
      return `<span class="rel-tag rel-${esc(v.relation)}">${esc(v.relation)}</span>`;
    if (typeof v === "object" && v.__kind === "path")
      return v.nodes.map((n) => cell(n)).join(" → ");
    if (typeof v === "object") return `<span class="cell-json">${esc(JSON.stringify(v))}</span>`;
    return esc(String(v));
  };
  $("#query-results").innerHTML = `
    <table class="results">
      <thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
      <tbody>${res.rows.map((row) => `<tr>${cols.map((c) => `<td>${cell(row[c])}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>`;
  $("#query-results").querySelectorAll(".cell-node").forEach((el) =>
    el.addEventListener("click", () => openEntity(el.dataset.id)));
}


// ---------------------------------------------------------------------------
// MCP tools console
//
// The dashboard's other views ask Memgraph questions of their own design. This
// one calls the actual MCP tools — same functions, same arguments, same payload
// an agent gets back — so a human can see what the agent sees.
// ---------------------------------------------------------------------------

const CATEGORY_LABEL = { search: "Search", graph: "Graph", cypher: "Cypher", indexing: "Indexing", other: "Other" };
const SPEND_CHIP = {
  llm: { text: "$ llm", title: "Spends real money: LLM summaries and embeddings" },
  embedding: { text: "$ emb", title: "Embeds the query through the embedding API — a fraction of a cent per call" },
};

// Console-only starting values. Everything else starts empty and falls through to
// the schema default, which the field's placeholder spells out.
const TOOL_UI = {
  cypher_query: { query: { rows: 6, placeholder: "MATCH (n:Code {project: $project})\nRETURN n.name AS name LIMIT 10" } },
  // The free dry run is what index_directory's own description tells an agent to do
  // first, so it is what the form offers first. Nothing here surprise-spends.
  index_directory: { preview: { value: true } },
};

const tools = {
  loaded: false,
  catalog: [],
  error: null,
  name: null,     // selected tool
  values: {},     // tool name -> { field: raw control value }, kept while the tab lives
  run: null,      // the run shown in the result panel
  runs: [],
  tab: "result",
  poll: null,
  busy: false,
};

async function apiRoot(path, opts) {
  const res = await fetch(`/api${path}`, opts);
  const data = await res.json();
  if (data && data.error && !opts?.allowError) throw new Error(data.error);
  return data;
}

// Tool descriptions are raw Python docstrings: first line flush, the rest indented
// to the `def`, with ``literals`` in reST. Agents read them exactly like that;
// humans should not have to read around the formatting.
function toolDescHtml(text) {
  const lines = String(text || "").split("\n");
  const rest = lines.slice(1).filter((l) => l.trim());
  const pad = rest.length ? Math.min(...rest.map((l) => l.match(/^ */)[0].length)) : 0;
  const dedented = [lines[0], ...lines.slice(1).map((l) => l.slice(pad))].join("\n").trim();
  return esc(dedented).replace(/``([^`]+)``/g, (_, code) => `<code>${code}</code>`);
}

function currentTool() {
  return tools.catalog.find((t) => t.name === tools.name) || null;
}

function projectRootPath() {
  return state.projects.find((p) => p.project === state.project)?.root_path || "";
}

// A field as the form needs it: pydantic writes optionals as anyOf [T, null].
function fieldsOf(tool) {
  const schema = tool.input_schema || {};
  const required = new Set(schema.required || []);
  return Object.entries(schema.properties || {}).map(([name, raw]) => {
    let spec = raw;
    if (Array.isArray(raw.anyOf)) spec = raw.anyOf.find((s) => s.type !== "null") || {};
    return {
      name,
      type: spec.type === "array" ? "array" : spec.type || "string",
      itemType: spec.items?.type || "string",
      required: required.has(name),
      def: raw.default,
    };
  });
}

function initialValue(tool, f) {
  const ui = TOOL_UI[tool.name]?.[f.name];
  if (ui && "value" in ui) return ui.value;
  // Every tool is addressed by the directory it was indexed from, not by the
  // project id the dashboard uses internally — so hand the human that path.
  if (f.name === "directory" || f.name === "path") return projectRootPath();
  return f.type === "boolean" ? Boolean(f.def) : "";
}

function fieldValue(tool, f) {
  const saved = tools.values[tool.name];
  return saved && f.name in saved ? saved[f.name] : initialValue(tool, f);
}

function setFieldValue(tool, name, value) {
  (tools.values[tool.name] = tools.values[tool.name] || {})[name] = value;
}

function typeHint(f) {
  const base = f.type === "array" ? `${f.itemType}[]` : f.type;
  const def = f.def === undefined || f.def === null ? "" : ` · default ${JSON.stringify(f.def)}`;
  return `${base}${f.required ? " · required" : def}`;
}

function fieldHtml(tool, f) {
  const ui = TOOL_UI[tool.name]?.[f.name] || {};
  const value = fieldValue(tool, f);
  const attrs = `data-field="${esc(f.name)}"`;
  let control;
  if (f.type === "boolean") {
    control = `<input type="checkbox" ${attrs} ${value ? "checked" : ""}>`;
  } else if (ui.rows) {
    control = `<textarea ${attrs} rows="${ui.rows}" spellcheck="false" placeholder="${esc(ui.placeholder || "")}">${esc(value)}</textarea>`;
  } else {
    const numeric = f.type === "integer" || f.type === "number";
    const isPath = f.name === "directory" || f.name === "path";
    const placeholder = f.type === "array"
      ? "comma-separated"
      // No stored root path for this project: say what the tool wants instead of an empty box.
      : isPath ? "/absolute/path/the/project/was/indexed/from"
      : f.def === undefined || f.def === null ? "" : String(f.def);
    control = `<input type="${numeric ? "number" : "text"}" ${attrs} value="${esc(value)}"
      placeholder="${esc(ui.placeholder || placeholder)}" spellcheck="false" autocomplete="off">`;
  }
  return `
    <label class="tf-row${f.type === "boolean" ? " tf-bool" : ""}">
      <span class="tf-name mono">${esc(f.name)}${f.required ? `<i class="req" title="required">•</i>` : ""}</span>
      <span class="tf-control">${control}</span>
      <span class="tf-hint">${esc(typeHint(f))}</span>
    </label>`;
}

// Only what the user actually filled in is sent, so the Request tab shows the
// argument object an agent would have had to write — not a form dump.
function collectArgs(tool) {
  const args = {};
  const missing = [];
  for (const f of fieldsOf(tool)) {
    const raw = fieldValue(tool, f);
    if (f.type === "boolean") {
      if (Boolean(raw) !== Boolean(f.def)) args[f.name] = Boolean(raw);
      continue;
    }
    const text = String(raw ?? "").trim();
    if (!text) {
      if (f.required) missing.push(f.name);
      continue;
    }
    if (f.type === "array") args[f.name] = text.split(",").map((s) => s.trim()).filter(Boolean);
    else if (f.type === "integer" || f.type === "number") {
      // A number that will not parse is left out rather than sent as null: the tool
      // should see "not given", not a value the form invented.
      const parsed = f.type === "integer" ? parseInt(text, 10) : Number(text);
      if (Number.isFinite(parsed)) args[f.name] = parsed;
      else if (f.required) missing.push(f.name);
    } else args[f.name] = String(raw); // strings keep their newlines and spacing
  }
  return { args, missing };
}

// --------------------------------------------------------------- JSON viewer

function jsonValueHtml(v, ctx) {
  if (v === null || v === undefined) return `<span class="j-null">null</span>`;
  if (typeof v === "boolean") return `<span class="j-bool">${v}</span>`;
  if (typeof v === "number") return `<span class="j-num">${v}</span>`;
  if (typeof v === "string") {
    const literal = esc(JSON.stringify(v));
    // An entity id next to a name is a thing you can go look at.
    if (ctx?.key === "id" && ctx.obj?.name)
      return `<a href="#" class="j-str j-link" data-entity="${esc(v)}" title="Open in the explorer">${literal}</a>`;
    return `<span class="j-str${v.length > 320 ? " j-long" : ""}">${literal}</span>`;
  }
  if (Array.isArray(v)) {
    if (!v.length) return `<span class="j-brace">[]</span>`;
    return `<span class="j-node"><span class="j-tog">▾</span><span class="j-brace">[</span><span class="j-fold">…${v.length}</span>` +
      `<span class="j-body">${v.map((item) => `<span class="j-row">${jsonValueHtml(item, null)}</span>`).join("")}</span>` +
      `<span class="j-brace">]</span></span>`;
  }
  const keys = Object.keys(v);
  if (!keys.length) return `<span class="j-brace">{}</span>`;
  return `<span class="j-node"><span class="j-tog">▾</span><span class="j-brace">{</span><span class="j-fold">…${keys.length}</span>` +
    `<span class="j-body">${keys.map((k) =>
      `<span class="j-row"><span class="j-key">${esc(JSON.stringify(k))}</span><span class="j-colon">:</span> ${jsonValueHtml(v[k], { obj: v, key: k })}</span>`
    ).join("")}</span><span class="j-brace">}</span></span>`;
}

function wireJson(root) {
  root.querySelectorAll(".j-tog").forEach((el) =>
    el.addEventListener("click", () => el.closest(".j-node").classList.toggle("collapsed")));
  root.querySelectorAll(".j-fold").forEach((el) =>
    el.addEventListener("click", () => el.closest(".j-node").classList.remove("collapsed")));
  root.querySelectorAll(".j-long").forEach((el) =>
    el.addEventListener("click", () => el.classList.toggle("open")));
  root.querySelectorAll(".j-link").forEach((el) =>
    el.addEventListener("click", (ev) => { ev.preventDefault(); openEntity(el.dataset.entity); }));
}

// ------------------------------------------------------------------- rendering

function renderToolsSidebar() {
  const list = $("#tools-list");
  $("#tools-count").textContent = tools.catalog.length ? `${tools.catalog.length}` : "";
  if (tools.error) {
    list.innerHTML = `<div class="tool-unavailable">Tools unavailable:<br><span class="mono">${esc(tools.error)}</span></div>`;
    return;
  }
  let html = "";
  let category = null;
  for (const tool of tools.catalog) {
    if (tool.category !== category) {
      category = tool.category;
      html += `<div class="tool-cat">${esc(CATEGORY_LABEL[category] || category)}</div>`;
    }
    const spend = SPEND_CHIP[tool.spend];
    html += `
      <div class="tool-row${tool.name === tools.name ? " active" : ""}" data-tool="${esc(tool.name)}">
        <span class="tn mono">${esc(tool.name)}</span>
        ${spend ? `<span class="spend-chip" title="${esc(spend.title)}">${esc(spend.text)}</span>` : ""}
      </div>`;
  }
  list.innerHTML = html;
  list.querySelectorAll(".tool-row").forEach((el) =>
    el.addEventListener("click", () => nav("tools", { tool: el.dataset.tool })));
}

function renderRunsList() {
  const box = $("#tools-runs");
  if (!tools.runs.length) {
    box.innerHTML = `<div class="dim tool-empty">No calls yet.</div>`;
    return;
  }
  box.innerHTML = tools.runs.map((r) => `
    <div class="run-row${tools.run?.id === r.id ? " active" : ""}" data-run="${esc(r.id)}" title="${esc(r.arg_preview || "")}">
      <span class="run-dot ${esc(r.state)}"></span>
      <span class="rn mono">${esc(r.tool)}</span>
      <span class="rt">${r.duration_ms == null ? "…" : r.duration_ms + " ms"}</span>
    </div>`).join("");
  box.querySelectorAll(".run-row").forEach((el) =>
    el.addEventListener("click", () => openRun(el.dataset.run)));
}

async function refreshRuns() {
  const res = await apiRoot("/tools/runs", { allowError: true });
  tools.runs = res.runs || [];
  renderRunsList();
}

function toolIntroHtml() {
  return `
    <div class="tool-detail">
      <div class="tool-intro">
        <h1>MCP tools</h1>
        <p>These are the tools the MCP server hands to an agent — the same functions, called
        the same way. Pick one, fill in the arguments, and the panel shows the exact payload
        an agent would receive, byte for byte.</p>
        <p class="dim">Calls run in the background: a long <code>index_directory</code> survives a
        reload and can be cancelled. Indexing from here never starts a file watcher — the MCP
        server owns that.</p>
      </div>
    </div>`;
}

function renderToolDetail() {
  const pane = $("#tool-content");
  const tool = currentTool();
  if (!tool) {
    pane.classList.remove("dim", "pad");
    pane.innerHTML = tools.error
      ? `<div class="tool-detail"><div class="tool-intro"><h1>MCP tools</h1><p class="run-error mono">${esc(tools.error)}</p></div></div>`
      : toolIntroHtml();
    return;
  }
  pane.classList.remove("dim", "pad");
  const spend = SPEND_CHIP[tool.spend];
  const previewOff = tool.name === "index_directory" && !fieldValue(tool, { name: "preview", type: "boolean", def: false });
  pane.innerHTML = `
    <div class="tool-detail">
      <div class="tool-head">
        <h1 class="mono">${esc(tool.name)}</h1>
        <span class="cat-chip">${esc(CATEGORY_LABEL[tool.category] || tool.category)}</span>
        ${spend ? `<span class="spend-chip" title="${esc(spend.title)}">${esc(spend.text)}</span>` : ""}
      </div>
      <div class="tool-desc">${toolDescHtml(tool.description)}</div>
      ${previewOff ? `
        <div class="spend-warn">
          <b>This runs the real index.</b> LLM summaries and embeddings cost money and the run can
          take minutes. Leave <code>preview</code> checked for a free estimate first.
          <label class="tf-check"><input type="checkbox" id="tool-confirm"> I understand — index for real</label>
        </div>` : ""}
      <div class="sect">
        <h2>Arguments</h2>
        <div class="tf-grid">${fieldsOf(tool).map((f) => fieldHtml(tool, f)).join("")}</div>
      </div>
      <div class="tool-actions">
        <button class="tb-btn accent" id="tool-run">Run ⌘⏎</button>
        <button class="tb-btn" id="tool-reset">Reset arguments</button>
        <button class="tb-btn hidden" id="tool-cancel">Cancel</button>
        <span id="tool-status" class="dim"></span>
      </div>
      <div id="tool-result"></div>
    </div>`;

  pane.querySelectorAll("[data-field]").forEach((el) => {
    const field = el.dataset.field;
    const isCheck = el.type === "checkbox";
    el.addEventListener(isCheck ? "change" : "input", () => {
      setFieldValue(tool, field, isCheck ? el.checked : el.value);
      // `preview` decides whether this call spends money, so its warning is part of
      // the form's state, not a one-off banner.
      if (tool.name === "index_directory" && field === "preview") renderToolDetail();
    });
  });
  $("#tool-run").addEventListener("click", () => runTool());
  $("#tool-reset").addEventListener("click", () => {
    delete tools.values[tool.name];
    renderToolDetail();
  });
  $("#tool-cancel").addEventListener("click", cancelRun);
  pane.querySelectorAll("textarea, input").forEach((el) =>
    el.addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") { ev.preventDefault(); runTool(); }
    }));
  renderRunPanel();
  setBusy(tools.busy);
}

function runMeta(run) {
  const text = (run.content || []).map((b) => b.text || "").join("\n");
  const bytes = new TextEncoder().encode(text).length;
  const size = bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
  // A console for agent-facing payloads should say what a payload costs to read.
  return { text, size, tokens: Math.max(1, Math.round(text.length / 4)) };
}

function renderRunPanel() {
  const box = $("#tool-result");
  if (!box) return;
  const run = tools.run;
  const tool = currentTool();
  if (!run || !tool || run.tool !== tool.name) { box.innerHTML = ""; return; }

  const { text, size, tokens } = runMeta(run);
  const stat = run.state === "running"
    ? `<span class="run-state running"><span class="pulse-dot"></span>${run.cancel_requested ? "cancelling" : "running"}</span>`
    : `<span class="run-state ${esc(run.state)}">${esc(run.state)}</span>` +
      `<span class="dim"> · ${run.duration_ms} ms · ${size} · ≈${tokens} tok</span>`;

  const tab = (id, label) => `<button class="tab-btn${tools.tab === id ? " active" : ""}" data-tab="${id}">${label}</button>`;
  let body = "";
  if (run.state === "error") {
    body = `<div class="run-error mono">${esc(run.error)}</div>`;
  } else if (run.state === "cancelled") {
    body = `<div class="run-note">${esc(run.error || "Cancelled.")} A cancelled index keeps whatever it already persisted — nothing is rolled back.</div>`;
  } else if (run.state === "running") {
    body = `<div class="dim">${run.cancel_requested
      ? "Cancel requested — it lands at the tool's next await, so a blocking step (a filesystem scan) may still have to finish."
      : "Waiting for the tool…"}</div>`;
  } else if (tools.tab === "text") {
    body = `<pre class="run-text mono">${esc(text)}</pre>`;
  } else if (tools.tab === "request") {
    body = `<div class="json-view">${jsonValueHtml({ tool: run.tool, arguments: run.arguments })}</div>`;
  } else {
    body = (run.cancel_requested
      ? `<div class="run-note">The cancel arrived too late — the call had already finished.</div>`
      : "") +
      (run.error_envelope
      ? `<div class="run-envelope"><b class="mono">${esc(run.error_envelope.code)}</b> ${esc(run.error_envelope.message)}</div>`
      : "") +
      `<div class="json-view">${run.structured === null || run.structured === undefined
        ? `<span class="dim">No JSON payload — see the text block.</span>`
        : jsonValueHtml(run.structured)}</div>`;
  }

  box.innerHTML = `
    <div class="sect">
      <h2>Result <span class="dim run-id">#${esc(run.id)}</span></h2>
      <div class="run-bar">
        ${stat}
        <span class="grow"></span>
        ${run.state === "ok" ? `${tab("result", "Result")}${tab("text", "Text block")}${tab("request", "Request")}
          <button class="ghost-btn" id="run-copy">copy</button>` : ""}
      </div>
      <div id="run-body">${body}</div>
    </div>`;

  setBusy(tools.busy);
  box.querySelectorAll(".tab-btn").forEach((el) =>
    el.addEventListener("click", () => { tools.tab = el.dataset.tab; renderRunPanel(); }));
  $("#run-copy")?.addEventListener("click", () => navigator.clipboard.writeText(
    tools.tab === "request" ? JSON.stringify(run.arguments, null, 2) : text));
  wireJson(box);
}

// A call in flight belongs to the run, not to the click that started it: a re-render
// (hash change, project switch, reload + click on a recent call) has to find the
// Cancel button again.
function runningHere() {
  return tools.run?.state === "running" && tools.run.tool === tools.name;
}

function setBusy(on) {
  tools.busy = on;
  const busy = on || runningHere();
  $("#tool-run")?.toggleAttribute("disabled", busy);
  $("#tool-cancel")?.classList.toggle("hidden", !busy);
}

function toolStatus(message, isError = false) {
  const el = $("#tool-status");
  if (!el) return;
  el.className = isError ? "error" : "dim";
  el.textContent = message;
}

async function runTool() {
  const tool = currentTool();
  if (!tool || tools.busy) return;
  const { args, missing } = collectArgs(tool);
  if (missing.length) {
    toolStatus(`Fill in: ${missing.join(", ")}`, true);
    return;
  }
  toolStatus("");
  setBusy(true);
  const res = await apiRoot("/tools/call", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tool: tool.name, arguments: args, confirm: $("#tool-confirm")?.checked === true }),
    allowError: true,
  });
  if (res.error) {
    setBusy(false);
    toolStatus(res.error, true);
    return;
  }
  tools.run = res.run;
  tools.tab = "result";
  renderRunPanel();
  refreshRuns();
  pollRun(res.run.id);
}

function pollRun(id) {
  clearTimeout(tools.poll);
  const tick = async () => {
    if (state.view !== "tools") { setBusy(false); return; }
    const run = await apiRoot(`/tools/runs/${encodeURIComponent(id)}`, { allowError: true });
    if (run.error) { setBusy(false); toolStatus(run.error, true); return; }
    tools.run = run;
    renderRunPanel();
    renderRunsList();
    if (run.state === "running") { tools.poll = setTimeout(tick, 400); return; }
    setBusy(false);
    refreshRuns();
  };
  tools.poll = setTimeout(tick, 250);
}

async function cancelRun() {
  if (!tools.run) return;
  await apiRoot(`/tools/runs/${encodeURIComponent(tools.run.id)}/cancel`, { method: "POST", allowError: true });
  toolStatus("Cancel requested — it takes effect at the tool's next await.");
}

// Clicking a past call puts you back where it was made: same tool, same arguments,
// same output — ready to change one thing and run it again.
async function openRun(id) {
  const run = await apiRoot(`/tools/runs/${encodeURIComponent(id)}`, { allowError: true });
  if (run.error) return;
  const tool = tools.catalog.find((t) => t.name === run.tool);
  if (!tool) return;
  const values = {};
  for (const f of fieldsOf(tool)) {
    const v = run.arguments[f.name];
    if (v === undefined) values[f.name] = f.type === "boolean" ? Boolean(f.def) : "";
    else values[f.name] = f.type === "boolean" ? Boolean(v) : Array.isArray(v) ? v.join(", ") : String(v);
  }
  tools.values[tool.name] = values;
  tools.run = run;
  tools.name = tool.name;
  nav("tools", { tool: tool.name });
  renderToolsSidebar();
  renderToolDetail();
  renderRunsList();
  if (run.state === "running") { setBusy(true); pollRun(run.id); }
}

async function renderToolsView(params) {
  if (!tools.loaded) {
    const res = await apiRoot("/tools", { allowError: true });
    tools.loaded = true;
    tools.catalog = res.tools || [];
    tools.error = res.available === false ? res.error || "the MCP server could not be started" : null;
  }
  const wanted = params.get("tool");
  if (wanted && tools.catalog.some((t) => t.name === wanted)) tools.name = wanted;
  renderToolsSidebar();
  renderToolDetail();
  if (tools.busy && tools.run?.state === "running") pollRun(tools.run.id);
  refreshRuns();
}


// ---------------------------------------------------------------------------
// Search
//
// A search answers on the specimen as well as in the list: hits are numbered on the graph
// and a weak one is drawn weakly, because that is what the server said about it.
// ---------------------------------------------------------------------------

let searchTimer = null;
let lastSearchQ = "";
let lastHits = [];

function searchItem(hit, n, score) {
  return `<div class="sr-item" data-id="${esc(hit.id)}" title="${esc(hit.summary || "")}">
    <span class="num">${n}</span>${chip(hit.type)}<span class="name">${esc(hit.name)}</span>
    <span class="path">${esc(fileRef(hit))}</span>${score && hit.score ? `<span class="sc">${hit.score}</span>` : ""}
  </div>`;
}

// "weak" is the server's own word for a low-scoring vector match; carry it, don't hide it.
const WEAK_BELOW = 0.45;

async function doSearch(q, mode = "quick") {
  lastSearchQ = q;
  const box = $("#search-results");
  if (!q.trim()) { box.classList.add("hidden"); return; }
  const res = await api(`/search?q=${encodeURIComponent(q)}&mode=${mode}`, { allowError: true });
  if (lastSearchQ !== q) return; // stale
  let html = "";
  let n = 0;
  lastHits = [];
  const section = (title, items, score) => {
    if (!items?.length) return "";
    let out = `<div class="sr-section">${title}</div>`;
    for (const h of items) {
      n += 1;
      lastHits.push({ id: h.id, weak: score && h.score !== undefined && h.score < WEAK_BELOW });
      out += searchItem(h, n, score);
    }
    return out;
  };
  html += section("NAME", res.name, false);
  html += section("FULL TEXT · BM25", res.text, true);
  html += section("SEMANTIC · EMBEDDINGS", res.semantic, true);
  if (!html) html = `<div class="sr-note">No matches. The index was asked and answered nothing — that is the answer.</div>`;
  for (const note of res.notes || []) html += `<div class="sr-note">${esc(note)}</div>`;
  html += `<div class="sr-actions">
    ${mode !== "semantic" ? `<button class="btn" id="sr-semantic">semantic search (embeds the query)</button>` : ""}
    ${lastHits.length ? `<button class="btn" id="sr-specimen">show ${lastHits.length} on the specimen</button>` : ""}
  </div>`;
  box.innerHTML = html;
  box.classList.remove("hidden");
  // Hits land on the specimen straight away, so switching to it shows the answer in place.
  spec.hits = lastHits;
  spec.hitLabel = `${mode === "semantic" ? "search_semantic" : "search"} · ${q}`;
  if (spec.pack && state.view === "graph") spec.pack.setHits(spec.hits, spec.hitLabel);
  box.querySelectorAll(".sr-item").forEach((el) =>
    el.addEventListener("click", () => {
      box.classList.add("hidden");
      $("#search-input").blur();
      openEntity(el.dataset.id);
    }));
  $("#sr-semantic")?.addEventListener("click", () => doSearch(q, "semantic"));
  $("#sr-specimen")?.addEventListener("click", () => {
    box.classList.add("hidden");
    if (spec.pack) spec.pack.setHits(spec.hits, spec.hitLabel);
    nav("graph");
  });
}

function wireSearch() {
  const input = $("#search-input");
  input.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => doSearch(input.value), 250);
  });
  input.addEventListener("focus", () => { if (input.value.trim()) doSearch(input.value); });
  input.addEventListener("keydown", (ev) => {
    const box = $("#search-results");
    const items = [...box.querySelectorAll(".sr-item")];
    const sel = box.querySelector(".sr-item.sel");
    const idx = items.indexOf(sel);
    if (ev.key === "Escape") { box.classList.add("hidden"); input.blur(); }
    else if (ev.key === "ArrowDown") { ev.preventDefault(); items[Math.min(idx + 1, items.length - 1)]?.classList.add("sel"); sel?.classList.remove("sel"); }
    else if (ev.key === "ArrowUp") { ev.preventDefault(); if (idx > 0) { items[idx - 1].classList.add("sel"); sel.classList.remove("sel"); } }
    else if (ev.key === "Enter") { (sel || items[0])?.click(); }
  });
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "/" && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "TEXTAREA") {
      ev.preventDefault();
      input.focus();
      input.select();
    }
  });
  document.addEventListener("click", (ev) => {
    if (!$("#search-wrap").contains(ev.target)) $("#search-results").classList.add("hidden");
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

function wireSpecimenControls() {
  $("#graph-level").addEventListener("change", (ev) => { spec.level = ev.target.value; loadSpecimen(); });
  $("#graph-summarized").addEventListener("change", (ev) => { spec.summarized = ev.target.checked; loadSpecimen(); });
  $("#graph-overview-btn").addEventListener("click", () => {
    spec.prefix = null;
    loadSpecimen().then(() => spec.pack?.fit(0.9));
  });
  $("#graph-fit").addEventListener("click", () => spec.pack?.fit(0.9));
  $("#graph-route").addEventListener("click", () => {
    if ($("#route-panel").classList.contains("hidden")) openRoutePanel(); else closeRoutePanel();
  });
  $("#route-close").addEventListener("click", closeRoutePanel);
  $("#route-clear").addEventListener("click", () => {
    spec.route = { a: null, b: null, arm: null };
    $("#route-a").value = ""; $("#route-b").value = "";
    computeRoute();
  });
  for (const which of ["a", "b"]) {
    $(`#route-${which}`).addEventListener("input", () => routeSuggest(which));
    $(`#route-${which}`).addEventListener("focus", () => routeSuggest(which));
  }
  document.querySelectorAll("[data-pick]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const which = btn.dataset.pick;
      document.querySelectorAll(".route-pick").forEach((r) => r.classList.remove("arm"));
      $(`#route-${which}-row`).classList.add("arm");
      spec.route.arm = which;
      spec.pack?.pickNext((n) => { if (n) setRouteEnd(which, n); });
    }));
  // Chrome restores form values across a reload; a stale route end would then read as picked.
  $("#route-a").value = ""; $("#route-b").value = "";
  renderGfxPanel();
  $("#graph-gfx").addEventListener("click", () => {
    const panel = $("#gfx-panel");
    const show = panel.classList.contains("hidden");
    panel.classList.toggle("hidden", !show);
    $("#graph-gfx").classList.toggle("on", show);
  });
  $("#gfx-close").addEventListener("click", () => {
    $("#gfx-panel").classList.add("hidden");
    $("#graph-gfx").classList.remove("on");
  });
  $("#gfx-reset").addEventListener("click", () => {
    spec.gfx = { ...GFX_DEFAULTS };
    saveGfx();
    renderGfxPanel();
    spec.pack?.setGfx(spec.gfx);
  });
  $("#graph-replay").addEventListener("click", () => { if (spec.replaying) stopReplay(); else startReplay(); });
  $("#replay-toggle").addEventListener("click", () => {
    const playing = spec.pack.bloom.playing;
    spec.pack.replayPause(playing);
    $("#replay-toggle").textContent = playing ? "▶" : "❚❚";
  });
  $("#replay-stop").addEventListener("click", stopReplay);
  $("#replay-speed").addEventListener("change", (ev) => spec.pack?.replaySpeed(+ev.target.value || 1));
  $("#graph-rec").addEventListener("click", toggleRecording);
  $("#graph-png").addEventListener("click", () => {
    const canvas = $("#graph-canvas canvas");
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `${state.project}-specimen.png`;
    a.click();
  });
}

async function boot() {
  document.querySelectorAll("#nav a").forEach((a) =>
    a.addEventListener("click", (ev) => { ev.preventDefault(); if (state.project) nav(a.dataset.view); }));
  $("#brand").addEventListener("click", (ev) => { ev.preventDefault(); if (state.project) nav("overview"); });
  $("#project-select").addEventListener("change", (ev) => {
    state.project = ev.target.value;
    resetProjectState();
    nav("overview");
  });
  $("#watch-badge").addEventListener("click", () => checkFreshness());
  $("#tree-collapse").addEventListener("click", () => {
    document.querySelectorAll("#tree .tkids").forEach((k, i) => { if (i > 0) k.classList.add("hidden"); });
    document.querySelectorAll("#tree .trow .twisty").forEach((t, i) => { if (i > 0 && t.textContent === "▾") t.textContent = "▸"; });
  });
  wireSpecimenControls();
  wireSearch();

  // Esc peels one layer at a time: a panel, then the replay, then the pin.
  document.addEventListener("keydown", (ev) => {
    if (state.view !== "graph") return;
    if (ev.key === "Escape") {
      if (!$("#gfx-panel").classList.contains("hidden")) { $("#gfx-panel").classList.add("hidden"); $("#graph-gfx").classList.remove("on"); return; }
      if (spec.replaying) { stopReplay(); return; }
      if (spec.pack && spec.pack.unpin()) { $("#graph-side").classList.add("hidden"); return; }
      if (!$("#route-panel").classList.contains("hidden")) closeRoutePanel();
      return;
    }
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return;
    if (ev.key === "f" || ev.key === "F") spec.pack?.fit(0.9);
  });

  let projects = [];
  try {
    const res = await fetch("/api/projects");
    projects = (await res.json()).projects || [];
  } catch (e) {
    $("#empty-status").textContent = "Cannot reach the API — is Memgraph running? (docker compose up -d)";
  }
  state.projects = projects;
  $("#project-select").innerHTML = projects.map((p) =>
    `<option value="${esc(p.project)}">${esc(p.display_name)} · ${fmt(p.entities)}</option>`).join("");
  window.addEventListener("hashchange", route);
  await route();
  setInterval(pollStatus, 2000);
  // The toolbar's frame time is a live measurement, so it has to keep measuring.
  setInterval(() => { if (state.view === "graph" && spec.pack && !spec.loading) updateGraphCount(); }, 1500);
  pollStatus();
}

boot();
