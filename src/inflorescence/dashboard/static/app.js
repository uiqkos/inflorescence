/* Inflorescence dashboard SPA. No build step: plain DOM + cytoscape + highlight.js. */
"use strict";

// ---------------------------------------------------------------------------
// State & helpers
// ---------------------------------------------------------------------------

// Muted botanical palette — kept in sync with app.css :root tokens.
const TYPE_COLORS = {
  directory: "#9a9c8f", module: "#5a93c9", class: "#3fa588", function: "#d9a24e",
  method: "#6faf5a", interface: "#9a90de", enum: "#dd7d6b", struct: "#cd7ba3",
  trait: "#cf8455", document: "#c8bd8e", code: "#7d7f73",
};
const REL_COLORS = {
  CALLS: "#6f9fd0", IMPORTS: "#5fae86", INHERITS: "#9a90de",
  IMPLEMENTS: "#cd7ba3", CONTAINS: "#4c5044",
};
// Graph-canvas ink (matches --ink-2) and on-canvas label/halo tones.
const CANVAS = { ink: "#0c0e0b", halo: "rgba(12,14,11,0.72)", label: "#a9ac9d", accent: "#c3cf82", ring: "rgba(236,235,224,0.22)" };
const REL_TYPES = ["CALLS", "IMPORTS", "INHERITS", "IMPLEMENTS", "CONTAINS"];
const CHIP_TEXT = {
  directory: "dir", module: "mod", class: "cls", function: "fn", method: "mth",
  interface: "if", enum: "enum", struct: "st", trait: "tr", document: "doc", code: "?",
};

// Display settings for the graph view. Every one of these is presentation only —
// they change how the graph is laid out, painted and animated, never what is
// fetched or stored. Persisted per browser so a tuned view survives a reload.
const GFX_DEFAULTS = {
  // clustering / physics
  cluster: 0.7,        // pull toward the centroid of the node's own directory
  repel: 110,          // base charge; directories repel harder (see applyForces)
  containsDist: 18,    // CONTAINS is the structural spine: short and stiff
  crossDist: 240,      // CALLS/IMPORTS across modules: long and slack
  crossPull: 0.02,
  collide: true,       // force-graph ships no collision force at all
  anchors: false,      // deterministic ring slot per directory
  // paint
  hulls: true,         // soft blob behind each directory
  tint: false,         // colour nodes by directory instead of by entity type
  labels: true,        // file/entity name under every node
  edgeWidth: 0.5,      // multiplier on the weight-derived edge thickness
  edgeAlpha: 0.45,     // calls/imports outnumber nodes 5:1 — solid lines drown the graph
  // replay animation
  stems: true,         // an edge grows from the parent, then the node blooms
  glow: true,
  spring: true,        // overshoot easing instead of plain ease-out
  camera: "static",    // static | cinematic (follow the group being revealed)
  order: "structure",  // structure (CONTAINS spine first) | wave (outward ripple)
};
const GFX_KEY = "inflorescence.gfx";

// Only accept keys we know, with the type we expect: a stale or hand-edited
// localStorage entry must never be able to break the graph view.
function loadGfx() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(GFX_KEY) || "{}") || {}; } catch { saved = {}; }
  const gfx = { ...GFX_DEFAULTS };
  for (const k of Object.keys(GFX_DEFAULTS)) {
    if (typeof saved[k] === typeof GFX_DEFAULTS[k]) gfx[k] = saved[k];
  }
  return gfx;
}

const state = {
  projects: [],
  project: null,
  view: "overview",
  entityId: null,
  // All relation types are enabled by default, CONTAINS included, so the structural
  // skeleton shows alongside call/import/type edges from the start.
  // layout must be a value the #graph-layout select actually offers: "force" (no dag)
  // or a force-graph dagMode. The old "cose" default was a cytoscape leftover that
  // applyLayoutMode fed to dagMode() as a bogus DAG mode — with a cyclic CALLS graph
  // and live re-inits every merge tick this eventually killed the render loop.
  // level applies to the unrooted views: "modules" is the directory/module skeleton,
  // "all" is every indexed entity down to functions and methods. depth only means
  // anything in the rooted (neighborhood) view.
  graph: { mode: "overview", level: "modules", root: null, rootName: null, depth: 2, rels: new Set(["CALLS", "IMPORTS", "INHERITS", "IMPLEMENTS", "CONTAINS"]), layout: "force", summarizedOnly: true },
  gfx: loadGfx(),
  customGraph: null,
  treeCache: new Map(),
  // Live-index polling: `key` is the last seen nodes:edges:summarized fingerprint,
  // `indexing` the last seen lease state (for edge detection), `merging` a reentry guard.
  // `userCam` = the user grabbed the camera (wheel/drag) — live merges stop auto-fitting
  // until the next full render or an explicit Fit click.
  live: { indexing: false, key: null, merging: false, graphError: false, userCam: false },
};

// force-graph state: one instance reused across renders (keeps WebGL context and zoom)
let fg = null;
let fgData = { nodes: [], links: [] };
let fgNeighbors = new Map(); // node id -> Set of neighbor ids (for hover highlight)
let fgHoverNode = null;
let fgSelectedId = null;
let fgNeedsFit = false;
let fgLastClick = { id: null, t: 0 };

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts) {
  const res = await fetch(`/api/projects/${encodeURIComponent(state.project)}${path}`, opts);
  const data = await res.json();
  if (data && data.error && !opts?.allowError) throw new Error(data.error);
  return data;
}

function chip(type) {
  const t = TYPE_COLORS[type] ? type : "code";
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
  stopReplay({ restore: false }); // leaving the canvas: drop the animation
  $("#search-results").classList.add("hidden");
  const parsed = parseHash();
  if (parsed && state.projects.some((p) => p.project === parsed.project)) {
    if (state.project !== parsed.project) {
      state.project = parsed.project;
      state.treeCache = new Map();
      $("#tree").innerHTML = "";
      $("#entity-content").innerHTML = `<div class="dim pad">Select an entity from the tree.</div>`;
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

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
  $(`#view-${name}`)?.classList.remove("hidden");
}

function openEntity(id) {
  nav("explorer", { id });
}

// ---------------------------------------------------------------------------
// Overview
// ---------------------------------------------------------------------------

async function renderOverview() {
  const el = $("#view-overview");
  el.innerHTML = `<div class="ov-inner dim">Loading…</div>`;
  const s = await api("/stats");
  const coverage = s.summarizable ? Math.round((100 * s.summarized) / s.summarizable) : 0;
  const maxType = Math.max(1, ...s.by_type.map((t) => t.count));
  const maxRel = Math.max(1, ...s.by_relation.map((r) => r.count));
  const maxLang = Math.max(1, ...s.languages.map((l) => l.files));
  const primaryLang = s.languages[0]?.language;

  const rootBanner = s.root_path_exists
    ? ""
    : `<div class="rootpath-banner">
         <span>Source root unknown — code is shown from stored chunks only. Set the absolute path of <b>${esc(s.display_name)}</b>:</span>
         <input id="rootpath-input" class="mono" placeholder="/absolute/path/to/${esc(s.display_name)}" value="${esc(s.root_path || "")}">
         <button id="rootpath-save" class="tb-btn accent">Save</button>
       </div>`;

  const dotLabel = (name, color) =>
    `<span class="lbl"><span class="lg-dot" style="background:${color}"></span><span>${esc(name)}</span></span>`;

  el.innerHTML = `
    <div class="ov-inner">
      <div class="specimen">
        <div class="eyebrow">${primaryLang ? esc(primaryLang) + " · " : ""}indexed specimen</div>
        <h1>${esc(s.display_name)}</h1>
        <div class="classline">
          <span><b>id</b> ${esc(s.project)}</span>
          ${s.root_path_exists ? `<span><b>root</b> ${esc(s.root_path)}</span>` : ""}
        </div>
      </div>
      ${rootBanner}
      <div class="measures">
        <div class="figure"><span class="v">${s.entities.toLocaleString()}</span><span class="k">entities</span></div>
        <div class="figure"><span class="v">${s.edges.toLocaleString()}</span><span class="k">relations</span></div>
        <div class="figure"><span class="v">${s.files.toLocaleString()}</span><span class="k">files</span></div>
        <div class="figure"><span class="v">${coverage}%</span><span class="k">summarized</span>
          <span class="sub">${s.summarized.toLocaleString()} / ${s.summarizable.toLocaleString()}</span></div>
        <div class="figure"><span class="v">${s.chunks.toLocaleString()}</span><span class="k">code chunks</span>
          <span class="sub">${s.summary_embeddings.toLocaleString()} embeddings</span></div>
      </div>
      <div class="ov-grid">
        <div class="panel"><h2 class="overline">Composition</h2>${s.by_type.map((t) => `
          <div class="bar-row">
            <span class="lbl">${chip(t.type)}<span>${esc(t.type)}</span></span>
            <span class="bar-track"><span class="bar-fill" style="width:${(100 * t.count) / maxType}%;background:${TYPE_COLORS[t.type] || TYPE_COLORS.code}"></span></span>
            <span class="num">${t.count.toLocaleString()}</span>
          </div>`).join("") || `<div class="dim">no data</div>`}
        </div>
        <div class="panel"><h2 class="overline">Connections</h2>${s.by_relation.map((r) => `
          <div class="bar-row">
            ${dotLabel(r.relation.toLowerCase(), REL_COLORS[r.relation] || TYPE_COLORS.code)}
            <span class="bar-track"><span class="bar-fill" style="width:${(100 * r.count) / maxRel}%;background:${REL_COLORS[r.relation] || TYPE_COLORS.code}"></span></span>
            <span class="num">${r.count.toLocaleString()}</span>
          </div>`).join("") || `<div class="dim">no data</div>`}
        </div>
        <div class="panel"><h2 class="overline">Languages</h2>${s.languages.map((l) => `
          <div class="bar-row">
            ${dotLabel(l.language, "var(--t-class)")}
            <span class="bar-track"><span class="bar-fill" style="width:${(100 * l.files) / maxLang}%;background:var(--t-class)"></span></span>
            <span class="num">${l.files.toLocaleString()}</span>
          </div>`).join("") || `<div class="dim">no data</div>`}
        </div>
        <div class="panel"><h2 class="overline">Most called</h2>${s.most_called.map((h) => `
          <div class="list-row" data-id="${esc(h.id)}">${chip(h.type)}<span class="name">${esc(h.name)}</span>
            <span class="path">${esc(h.file_path || "")}</span><span class="n">${h.callers}×</span></div>`).join("") || `<div class="dim">no CALLS edges</div>`}
        </div>
        <div class="panel"><h2 class="overline">Most imported</h2>${s.most_imported.map((h) => `
          <div class="list-row" data-id="${esc(h.id)}">${chip(h.type)}<span class="name">${esc(h.name)}</span>
            <span class="path">${esc(h.file_path || "")}</span><span class="n">${h.importers}×</span></div>`).join("") || `<div class="dim">no IMPORTS edges</div>`}
        </div>
        <div class="panel"><h2 class="overline">Largest files</h2>${s.largest_files.map((f) => `
          <div class="list-row" ${f.module_id ? `data-id="${esc(f.module_id)}"` : ""}>${chip("module")}
            <span class="path" style="flex:1">${esc(f.file_path)}</span><span class="n">${f.entities} entities</span></div>`).join("") || `<div class="dim">no data</div>`}
        </div>
      </div>
    </div>`;

  el.querySelectorAll(".list-row[data-id]").forEach((row) =>
    row.addEventListener("click", () => openEntity(row.dataset.id)));
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
// Tree
// ---------------------------------------------------------------------------

async function fetchChildren(parentId) {
  const key = parentId ?? " root";
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
    tree.innerHTML = `<div class="dim" style="padding:12px">Project has no root node.</div>`;
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
  const pane = $("#entity-content");
  pane.innerHTML = `<div class="dim pad">Loading…</div>`;
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
      ? `<span style="color:var(--text)">${esc(c.name)}</span>`
      : `<a href="#" data-id="${esc(c.id)}">${esc(c.name)}</a>`
  ).join(`<span class="sep">/</span>`);

  const relGroup = (rels, dir) => {
    if (!rels.length) return `<div class="dim" style="font-size:12.5px">none</div>`;
    const byType = {};
    rels.forEach((r) => (byType[r.relation] = byType[r.relation] || []).push(r));
    return Object.entries(byType).map(([rel, items]) => `
      <div class="rel-group">
        <span class="rel-tag" style="--rel-c:${REL_COLORS[rel] || "#7d7f73"}">${dir === "in" ? "◂ " : "▸ "}${esc(rel)}</span>
        ${items.map((r) => `
          <div class="list-row" data-id="${esc(r.id)}" title="${esc(r.summary || "")}">
            ${chip(r.type)}<span class="name">${esc(r.name)}</span><span class="path">${esc(fileRef(r))}</span>
          </div>`).join("")}
      </div>`).join("");
  };

  pane.innerHTML = `
    <div class="entity">
      <div class="crumbs">${crumbs}</div>
      <div class="entity-head">${chip(e.type)}<h1>${esc(e.name)}</h1></div>
      <div class="entity-meta mono">${esc(fileRef(e))}${e.line_end ? "–" + e.line_end : ""} · <span title="entity id">${esc(e.id)}</span></div>
      <div class="entity-actions">
        <button class="tb-btn accent" id="btn-show-graph">Show in graph</button>
        <button class="tb-btn" id="btn-copy-id">Copy id</button>
      </div>
      ${e.summary ? `<div class="sect"><h2>Summary</h2><div class="summary-card">${esc(e.summary)}</div></div>` : ""}
      ${e.signature ? `<div class="sect"><h2>Signature</h2><div class="sig-block mono">${esc(e.signature)}</div></div>` : ""}
      ${e.docstring ? `<div class="sect"><h2>Docstring</h2><div class="doc-block">${esc(e.docstring)}</div></div>` : ""}
      <div class="sect" id="code-sect"><h2>Code</h2><div class="dim">Loading code…</div></div>
      ${data.children.length ? `
        <div class="sect"><h2>Contains (${data.children.length})</h2>
          <div class="kids-grid">${data.children.map((c) => `
            <span class="kid-chip" data-id="${esc(c.id)}" title="${esc(c.summary || "")}">${chip(c.type)}<span class="nm">${esc(c.name)}</span></span>`).join("")}
          </div></div>` : ""}
      <div class="sect"><h2>Relations</h2>
        <div class="rel-grid">
          <div><h2 style="font-size:10.5px;color:var(--muted);margin:0 0 6px">outgoing</h2>${relGroup(data.outgoing, "out")}</div>
          <div><h2 style="font-size:10.5px;color:var(--muted);margin:0 0 6px">incoming</h2>${relGroup(data.incoming, "in")}</div>
        </div>
      </div>
    </div>`;

  pane.querySelectorAll("[data-id]").forEach((el) =>
    el.addEventListener("click", (ev) => { ev.preventDefault(); openEntity(el.dataset.id); }));
  $("#btn-show-graph").addEventListener("click", () => {
    nav("graph", { root: e.id, depth: state.graph.depth });
  });
  $("#btn-copy-id").addEventListener("click", () => navigator.clipboard.writeText(e.id));

  loadEntityCode(e, $("#code-sect"));
}

async function loadEntityCode(entity, sect, wholeFile = false) {
  const res = await api(`/code?id=${encodeURIComponent(entity.id)}${wholeFile ? "&whole_file=1" : ""}`, { allowError: true });
  if (!res.content) {
    const reasons = {
      directory: "Directories have no source of their own — pick a file or entity inside.",
      no_root_path: "No source on disk and no stored chunks. Set the project root path on the Overview tab to read code from disk.",
      no_code: "No stored code for this entity.",
      entity_not_found: "Entity not found.",
    };
    sect.innerHTML = `<h2>Code</h2><div class="dim">${reasons[res.reason] || "No code available."}</div>`;
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
    <h2>Code</h2>
    <div class="code-card">
      <div class="code-head">
        <span class="mono">${esc(res.file_path || "")} · lines ${res.first_line}–${res.first_line + lines.length - 1}${res.total_lines ? " of " + res.total_lines : ""}</span>
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
// Graph
// ---------------------------------------------------------------------------

// Custom d3 forces + hull geometry, loaded from graph-forces.js ahead of this file.
const GF = globalThis.GraphForces;

// force-graph stops rendering once the layout settles (autoPauseRedraw). A change
// that only affects paint — hulls, tint, glow — would therefore not show up until
// something else moved, so nudge the loop awake for a moment.
let repaintTimer = null;
function repaintGraph() {
  if (!fg) return;
  fg.autoPauseRedraw(false);
  clearTimeout(repaintTimer);
  repaintTimer = setTimeout(() => { if (!replay.active) fg.autoPauseRedraw(true); }, 600);
}

// The colour a node is actually painted in: by entity type, or by directory when
// the viewer asked for group tinting.
function nodeInk(n) {
  return state.gfx.tint ? GF.groupColor(n.group, { alpha: 1 }) : n.color;
}

// Edges carry the weight in their thickness, so the settings scale that rather
// than replace it — a 12-call edge still reads heavier than a 1-call edge at any
// setting. Both are paint-time, so dragging the sliders is instant.
function linkGirth(l) {
  return Math.max(0.15, (l.width || 1) * state.gfx.edgeWidth);
}

function linkInk(l) {
  return state.gfx.edgeAlpha >= 1 ? l.color : withAlpha(l.color, state.gfx.edgeAlpha);
}

function groupsOf(nodes) {
  return [...new Set(nodes.map((n) => n.group))].sort();
}

// Everything physics-related lives here so the settings panel has exactly one
// place to re-apply. d3 replaces a force when it is re-registered under the same
// name, and passing null removes it — that is how the toggles switch off.
function applyForces() {
  if (!fg) return;
  const g = state.gfx;
  // Directories are the anchors of the whole picture: pushing them apart is what
  // keeps one cluster from sitting on top of the next.
  fg.d3Force("charge")
    .strength((n) => -g.repel * (n.type === "directory" ? 3.5 : 1))
    .distanceMax(520);
  // CONTAINS is a short stiff spring (a module belongs to its directory); calls
  // and imports across modules are long and slack, so they arrange the clusters
  // instead of tearing them apart.
  fg.d3Force("link")
    .distance((l) => (l.relation === "CONTAINS" ? g.containsDist : g.crossDist))
    .strength((l) => (l.relation === "CONTAINS" ? 1 : Math.min(0.6, g.crossPull * Math.sqrt(l.weight || 1))));
  fg.d3Force("cluster", g.cluster > 0 ? GF.forceCluster({ strength: g.cluster }) : null);
  // With collision on, nodes can pack tightly without overlapping, which means
  // the repulsion no longer has to do that job and clusters can stay dense.
  fg.d3Force("collide", g.collide ? GF.forceCollide((n) => nodeRadius(n) + 5, { strength: 0.8 }) : null);
  fg.d3Force("anchors", g.anchors
    ? GF.forceGroupAnchors(GF.ringAnchors(groupsOf(fgData.nodes)), 0.14)
    : null);
}

// Soft blob behind every directory, drawn under the graph. During a replay only
// the revealed nodes count, so the blobs grow with the animation.
function drawHulls(ctx, globalScale) {
  if (!state.gfx.hulls) return;
  const groups = new Map();
  for (const n of fgData.nodes) {
    if (replay.active && n.__shown !== true) continue;
    if (!Number.isFinite(n.x) || !Number.isFinite(n.y)) continue;
    if (!groups.has(n.group)) groups.set(n.group, []);
    groups.get(n.group).push(n);
  }
  for (const [group, pts] of groups) {
    if (pts.length < 2) continue;
    const hull = GF.padHull(GF.convexHull(pts), 26);
    GF.traceClosedCurve(ctx, hull, 0.4);
    ctx.fillStyle = GF.groupColor(group, { sat: 34, light: 56, alpha: 0.055 });
    ctx.fill();
    ctx.strokeStyle = GF.groupColor(group, { sat: 34, light: 62, alpha: 0.16 });
    ctx.lineWidth = 1 / globalScale;
    ctx.stroke();
    // Label the blob with the last path segment, but only when it is legible and
    // the cluster is big enough to be worth naming.
    if (globalScale > 0.35 && pts.length > 2) {
      let top = hull[0];
      for (const p of hull) if (p.y < top.y) top = p;
      const label = String(group).split("/").filter(Boolean).pop() || group;
      const size = 12 / globalScale;
      ctx.font = `500 ${size}px ${"system-ui, -apple-system, sans-serif"}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillStyle = GF.groupColor(group, { sat: 30, light: 70, alpha: 0.5 });
      ctx.fillText(label, top.x, top.y - 4 / globalScale);
    }
  }
}

// The Display panel. Every row names a key in state.gfx and how far the change
// has to propagate: "forces" re-runs the layout, "paint" only needs a repaint,
// "legend" also rebuilds the key, "replay" is read when the next replay starts.
const GFX_SCHEMA = [
  {
    title: "Clustering",
    items: [
      { k: "cluster", label: "cluster pull", type: "range", min: 0, max: 1, step: 0.05, apply: "forces",
        hint: "How hard every node is pulled toward the centre of its own folder. 0 turns clustering off." },
      { k: "collide", label: "collide", type: "bool", apply: "forces",
        hint: "Keeps nodes from overlapping, so a cluster can be dense without turning into a blob." },
      { k: "repel", label: "repulsion", type: "range", min: 40, max: 700, step: 20, apply: "forces",
        hint: "Base charge. Directories always repel 3.5× harder — that is what keeps clusters apart." },
      { k: "containsDist", label: "contains len", type: "range", min: 10, max: 90, step: 2, apply: "forces",
        hint: "Length of the structural CONTAINS spring: shorter = modules hug their directory." },
      { k: "crossDist", label: "cross len", type: "range", min: 40, max: 320, step: 10, apply: "forces",
        hint: "Length of calls/imports between modules." },
      { k: "crossPull", label: "cross pull", type: "range", min: 0, max: 0.2, step: 0.005, apply: "forces",
        hint: "Strength of those cross-folder edges. High values pull the folders back into one hairball." },
      { k: "anchors", label: "ring anchors", type: "bool", apply: "forces",
        hint: "Give every folder a fixed slot on a ring. The layout then comes out the same every run." },
    ],
  },
  {
    title: "Paint",
    items: [
      { k: "labels", label: "node labels", type: "bool", apply: "paint",
        hint: "The file / entity name under each node. Off is much easier to read on a dense graph." },
      { k: "edgeWidth", label: "edge width", type: "range", min: 0.2, max: 2, step: 0.1, apply: "paint",
        hint: "Scales every edge; heavier relations stay proportionally heavier." },
      { k: "edgeAlpha", label: "edge opacity", type: "range", min: 0.1, max: 1, step: 0.05, apply: "paint" },
      { k: "hulls", label: "folder blobs", type: "bool", apply: "paint" },
      { k: "tint", label: "colour by folder", type: "bool", apply: "legend",
        hint: "Swaps the entity-type palette for one hue per folder." },
    ],
  },
  {
    title: "Replay animation",
    items: [
      { k: "order", label: "reveal", type: "enum", options: ["structure", "wave"], apply: "replay",
        hint: "structure walks the CONTAINS tree; wave ripples outward from the centre." },
      { k: "camera", label: "camera", type: "enum", options: ["static", "cinematic"], apply: "replay",
        hint: "cinematic pushes in on each folder as it is built, then pulls back at the end." },
      { k: "stems", label: "growing stems", type: "bool", apply: "replay",
        hint: "An edge grows out of the parent first; the node opens when it arrives." },
      { k: "glow", label: "glow", type: "bool", apply: "paint" },
      { k: "spring", label: "spring easing", type: "bool", apply: "paint" },
    ],
    note: "Reveal, camera and stems take effect on the next replay.",
  },
];

function gfxRow(item, value) {
  const title = item.hint ? ` title="${esc(item.hint)}"` : "";
  const key = `<span class="gfx-k">${esc(item.label)}</span>`;
  if (item.type === "bool") {
    return `<label class="gfx-row gfx-check"${title}>${key}
      <input type="checkbox" data-k="${item.k}"${value ? " checked" : ""}></label>`;
  }
  if (item.type === "enum") {
    return `<label class="gfx-row"${title}>${key}
      <select data-k="${item.k}">${item.options.map((o) =>
        `<option value="${esc(o)}"${o === value ? " selected" : ""}>${esc(o)}</option>`).join("")}</select></label>`;
  }
  return `<label class="gfx-row"${title}>${key}
    <input type="range" data-k="${item.k}" min="${item.min}" max="${item.max}" step="${item.step}" value="${value}">
    <output class="gfx-v">${gfxFormat(value)}</output></label>`;
}

// Trailing zeros read as false precision on a slider readout: 0.7, not 0.70.
const gfxFormat = (v) => (typeof v === "number" ? String(+v.toFixed(3)) : String(v));

function renderGfxPanel() {
  const g = state.gfx;
  $("#gfx-body").innerHTML = GFX_SCHEMA.map((sect) => `
    <div class="gfx-sect">
      <h4>${esc(sect.title)}</h4>
      ${sect.items.map((it) => gfxRow(it, g[it.k])).join("")}
      ${sect.note ? `<p class="gfx-note">${esc(sect.note)}</p>` : ""}
    </div>`).join("");
  const byKey = new Map(GFX_SCHEMA.flatMap((s) => s.items).map((it) => [it.k, it]));
  $("#gfx-body").querySelectorAll("[data-k]").forEach((el) => {
    const item = byKey.get(el.dataset.k);
    el.addEventListener("input", () => {
      const value = el.type === "checkbox" ? el.checked
        : el.type === "range" ? +el.value
        : el.value;
      state.gfx[item.k] = value;
      const out = el.parentElement.querySelector(".gfx-v");
      if (out) out.textContent = gfxFormat(value);
      saveGfx();
      applyGfx(item.apply);
    });
  });
}

function saveGfx() {
  // A browser with storage disabled must not take the graph down with it.
  try { localStorage.setItem(GFX_KEY, JSON.stringify(state.gfx)); } catch { /* ignore */ }
}

// Dragging a slider fires per pixel; coalesce the work into one frame so the
// simulation is re-applied once per repaint instead of dozens of times.
// The kinds are independent, not ranked: flipping tint and then a slider inside
// one frame has to rebuild the legend AND re-run the layout, so they accumulate.
let gfxWork = { forces: false, legend: false };
let gfxQueued = false;
function applyGfx(kind) {
  if (kind === "replay" || !fg) return; // read when the next replay starts
  if (kind === "forces") gfxWork.forces = true;
  if (kind === "legend") gfxWork.legend = true;
  if (gfxQueued) return;
  gfxQueued = true;
  requestAnimationFrame(() => {
    const work = gfxWork;
    gfxWork = { forces: false, legend: false };
    gfxQueued = false;
    if (!fg) return;
    if (work.forces) { applyForces(); fg.d3ReheatSimulation(); }
    if (work.legend) buildLegend(fgData.nodes, fgData.links);
    repaintGraph();
  });
}

function toggleGfxPanel(on) {
  const panel = $("#gfx-panel");
  const show = on === undefined ? panel.classList.contains("hidden") : on;
  panel.classList.toggle("hidden", !show);
  $("#graph-gfx").classList.toggle("active", show);
}

function relCheckboxes() {
  $("#graph-rels").innerHTML = REL_TYPES.map((r) => `
    <label><input type="checkbox" data-rel="${r}" ${state.graph.rels.has(r) ? "checked" : ""}>
      <span style="color:${REL_COLORS[r]}">${r.toLowerCase()}</span></label>`).join("");
  document.querySelectorAll("#graph-rels input").forEach((cb) =>
    cb.addEventListener("change", () => {
      cb.checked ? state.graph.rels.add(cb.dataset.rel) : state.graph.rels.delete(cb.dataset.rel);
      refreshGraph();
    }));
}

async function renderGraph(params) {
  if (!window.ForceGraph) {
    $("#graph-canvas-wrap").innerHTML = `<div class="pad dim">force-graph failed to load.</div>`;
    return;
  }
  if (params.get("root")) {
    state.graph.mode = "neighborhood";
    state.graph.root = params.get("root");
    if (params.get("depth")) state.graph.depth = +params.get("depth");
  } else if (params.get("custom") && state.customGraph) {
    state.graph.mode = "custom";
  } else if (!state.graph.root) {
    state.graph.mode = "overview";
  }
  $("#graph-depth").value = String(state.graph.depth);
  $("#graph-level").value = state.graph.level;
  $("#graph-summarized").checked = state.graph.summarizedOnly;
  relCheckboxes();
  await refreshGraph();
}

function nodeRadius(n) {
  const raw = state.graph.mode === "overview" ? (n.size || 0) : (n.degree || 0);
  return Math.max(5, Math.min(16, 5 + Math.sqrt(raw) * 1.8));
}

function makeGraphElements(data) {
  const g = state.graph;
  const keepContains = g.mode !== "neighborhood"; // overview/custom keep the structural skeleton
  // "summarized" narrows the view to entities whose summary already exists. During a live
  // index summaries land batch by batch, so the visible graph grows with them. Custom
  // (query-result) graphs are exempt — their nodes may legitimately carry no summary field.
  const visible = (g.summarizedOnly && g.mode !== "custom")
    ? data.nodes.filter((n) => n.summary)
    : data.nodes;
  const nodeIds = new Set(visible.map((n) => n.id));
  const edges = data.edges.filter((e) =>
    (g.rels.has(e.relation) || (keepContains && e.relation === "CONTAINS"))
    && nodeIds.has(e.source) && nodeIds.has(e.target));
  const nodes = visible.map((n) => {
    const node = {
      id: n.id, name: n.name, type: n.type,
      color: TYPE_COLORS[n.type] || TYPE_COLORS.code,
      size: n.size, degree: n.degree, summary: n.summary || "", file: fileRef(n),
    };
    // Which directory this node belongs to. Everything group-flavoured — the
    // cluster force, the hulls, the tint, the cinematic camera — reads this.
    node.group = GF.groupKey(node);
    return node;
  });
  const links = edges.map((e) => ({
    source: e.source, target: e.target, relation: e.relation,
    color: REL_COLORS[e.relation] || "#57564f",
    width: 1 + Math.min(Math.sqrt(e.weight || 1), 4),
    weight: e.weight || 1,
  }));
  return { nodes, links };
}

function rebuildNeighbors() {
  fgNeighbors = new Map();
  const add = (a, b) => {
    if (!fgNeighbors.has(a)) fgNeighbors.set(a, new Set());
    fgNeighbors.get(a).add(b);
  };
  for (const l of fgData.links) {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    add(s, t);
    add(t, s);
  }
}

function nodeFaded(id) {
  // During the replay the graph slides under a stationary cursor, so hover
  // highlighting would randomly dim most of the show. Suppress it.
  if (!fgHoverNode || replay.active) return false;
  if (id === fgHoverNode.id) return false;
  return !(fgNeighbors.get(fgHoverNode.id)?.has(id));
}

function linkFaded(l) {
  if (!fgHoverNode || replay.active) return false;
  const s = typeof l.source === "object" ? l.source.id : l.source;
  const t = typeof l.target === "object" ? l.target.id : l.target;
  return s !== fgHoverNode.id && t !== fgHoverNode.id;
}

const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
// Overshoots past 1 and settles back — the difference between "appears" and "pops".
const easeOutBack = (t, s = 1.7) => 1 + (s + 1) * Math.pow(t - 1, 3) + s * Math.pow(t - 1, 2);
const entranceEase = (t) => (state.gfx.spring ? easeOutBack(t) : easeOutCubic(t));

// Fade a #rrggbb toward transparent — entrance animations and edge opacity.
function withAlpha(color, a) {
  const m = /^#([0-9a-f]{6})$/i.exec(String(color));
  if (!m) return color;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function drawNode(node, ctx, globalScale) {
  // A node waiting for its stem to reach it is simply not there yet.
  if (node.__bloomAt) {
    if (performance.now() < node.__bloomAt) return;
    delete node.__bloomAt;
  }
  // Two entrance animations run through here, both self-clearing so steady state
  // costs nothing:
  //   addedAt  — a node merged in by the live indexer: 400 ms scale pop.
  //   revealAt — the replay revealing a node in place: scale + fade + a ripple ring.
  let scale = 1;
  let alpha = 1;
  let ripple = -1;
  let glow = 0;
  if (node.addedAt) {
    const t = (performance.now() - node.addedAt) / 400;
    if (t < 1) { scale = 0.2 + 0.8 * entranceEase(t); glow = 1 - t; }
    else delete node.addedAt;
  }
  if (node.revealAt) {
    const dt = performance.now() - node.revealAt;
    const t = dt / 520;
    if (t < 1) {
      scale = 0.25 + 0.75 * entranceEase(t);
      alpha = Math.min(1, dt / 260);
      glow = Math.max(glow, 1 - t);
    }
    ripple = dt / 900;
    if (ripple >= 1) delete node.revealAt;
  }
  const col = nodeInk(node);
  const r = nodeRadius(node) * scale;
  const faded = nodeFaded(node.id);
  ctx.globalAlpha = faded ? 0.12 : alpha;

  ctx.beginPath();
  ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = col;
  // shadowBlur is in device pixels and ignores the canvas transform, so the halo
  // reads the same at any zoom. Must be reset or it bleeds into everything after.
  if (glow > 0 && state.gfx.glow && !faded) {
    ctx.shadowColor = col;
    ctx.shadowBlur = 26 * glow;
  }
  ctx.fill();
  ctx.shadowBlur = 0;
  if (node.id === state.graph.root || node.id === fgSelectedId) {
    ctx.lineWidth = 2 / globalScale;
    ctx.strokeStyle = node.id === state.graph.root ? CANVAS.accent : CANVAS.label;
    ctx.stroke();
  } else {
    ctx.lineWidth = 0.75 / globalScale;
    ctx.strokeStyle = CANVAS.ring;
    ctx.stroke();
  }

  // Ripple: a ring blooming out of a node the moment the replay reveals it.
  if (ripple >= 0 && ripple < 1) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + 30 * ripple, 0, 2 * Math.PI);
    ctx.strokeStyle = col;
    ctx.globalAlpha = (faded ? 0.08 : 0.55) * (1 - ripple);
    ctx.lineWidth = 1.6 / globalScale;
    ctx.stroke();
    ctx.globalAlpha = faded ? 0.12 : alpha;
  }

  // Label: constant on-screen size, with a dark halo for readability
  if (state.gfx.labels && !faded && globalScale > 0.45) {
    const label = node.name.length > 26 ? node.name.slice(0, 25) + "…" : node.name;
    const fontSize = 11 / globalScale;
    ctx.font = `${fontSize}px system-ui, -apple-system, sans-serif`;
    const w = ctx.measureText(label).width;
    const y = node.y + r + 3 / globalScale;
    ctx.fillStyle = CANVAS.halo;
    ctx.fillRect(node.x - w / 2 - 2 / globalScale, y, w + 4 / globalScale, fontSize + 2 / globalScale);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.fillStyle = CANVAS.label;
    ctx.fillText(label, node.x, y + 1 / globalScale);
  }
  ctx.globalAlpha = 1;
}

// An edge caught mid-stem: draw it from the endpoint that was already on screen
// out to the one budding off it, with a spark riding the tip. `__grow` is set
// only by the replay and clears itself, so nothing here runs in steady state.
function drawGrowingLink(link, ctx, globalScale) {
  if (!link.__grow) return;
  const from = link.__growFrom === link.target ? link.target : link.source;
  const to = from === link.source ? link.target : link.source;
  if (!from || !to || !Number.isFinite(from.x) || !Number.isFinite(to.x)) return;
  const t = easeOutCubic(Math.min(1, (performance.now() - link.__grow) / (link.__growMs || 520)));
  const hx = from.x + (to.x - from.x) * t;
  const hy = from.y + (to.y - from.y) * t;
  ctx.save();
  ctx.strokeStyle = linkInk(link);
  ctx.lineWidth = linkGirth(link) / globalScale;
  ctx.lineCap = "round";
  if (link.relation === "CONTAINS") ctx.setLineDash([3 / globalScale, 3 / globalScale]);
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(hx, hy);
  ctx.stroke();
  if (t < 1) {
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.arc(hx, hy, 2 / globalScale, 0, 2 * Math.PI);
    ctx.fillStyle = link.color;
    ctx.globalAlpha = 1 - t;
    ctx.fill();
  }
  ctx.restore();
  if (t >= 1) { delete link.__grow; delete link.__growFrom; delete link.__growMs; }
}

function ensureForceGraph() {
  if (fg) return fg;
  const container = $("#graph-canvas");
  // Real input only — programmatic zoomToFit fires neither of these, so the live
  // follow-camera keeps working until the user actually grabs the view.
  container.addEventListener("wheel", () => { state.live.userCam = true; }, { passive: true });
  container.addEventListener("pointerdown", () => { state.live.userCam = true; });
  fg = ForceGraph()(container)
    .backgroundColor(CANVAS.ink)
    .nodeId("id")
    .nodeLabel(() => "") // our own labels + card instead of the built-in tooltip
    .nodeCanvasObject(drawNode)
    // Visibility is a paint-time filter — hidden nodes still take part in the
    // physics. That is what lets the replay reveal a graph without re-laying it
    // out: nothing is inserted, so the simulation is never re-heated.
    .nodeVisibility((n) => !replay.active || n.__shown === true)
    .linkVisibility((l) => !replay.active || l.__shown === true)
    .nodePointerAreaPaint((node, color, ctx) => {
      ctx.beginPath();
      ctx.arc(node.x, node.y, nodeRadius(node) + 4, 0, 2 * Math.PI);
      ctx.fillStyle = color;
      ctx.fill();
    })
    // Edges revealed by the replay fade in and grow from a hairline to their real
    // width over 520 ms; `revealAt` is only ever set there and clears itself.
    .linkColor((l) => {
      if (linkFaded(l)) return "rgba(87,86,79,0.08)";
      if (l.revealAt) {
        const t = (performance.now() - l.revealAt) / 520;
        if (t < 1) return withAlpha(l.color, (0.1 + 0.9 * t) * state.gfx.edgeAlpha);
      }
      return linkInk(l);
    })
    .linkWidth((l) => {
      const w = linkGirth(l);
      if (!l.revealAt) return w;
      const t = (performance.now() - l.revealAt) / 520;
      if (t >= 1) { delete l.revealAt; return w; }
      return 0.2 + w * easeOutCubic(t);
    })
    // Stems: while a replayed edge is growing we paint it ourselves, from the
    // endpoint that was already on screen toward the one budding off it. Outside
    // that window the library's own (batched, much cheaper) painter does the job.
    .linkCanvasObjectMode((l) => (l.__grow ? "replace" : "after"))
    .linkCanvasObject(drawGrowingLink)
    .linkLineDash((l) => (l.relation === "CONTAINS" ? [3, 3] : null))
    // Particles are per-frame work on every visible edge — fine for a module
    // overview, ruinous for the all-entities view, which is thousands of edges.
    .linkDirectionalParticles((l) =>
      (l.relation === "CONTAINS" || linkFaded(l) || fgData.nodes.length > 400 ? 0 : 2))
    .linkDirectionalParticleWidth((l) => linkGirth(l) + 1)
    .linkDirectionalParticleSpeed(0.0045)
    .linkDirectionalParticleColor((l) => linkInk(l))
    .dagLevelDistance(70)
    .onDagError(() => {}) // cyclic graphs: keep force positions instead of throwing
    .onNodeHover((node) => {
      fgHoverNode = node || null;
      container.style.cursor = node ? "pointer" : "";
    })
    .onNodeClick((node) => {
      const now = Date.now();
      if (fgLastClick.id === node.id && now - fgLastClick.t < 350) {
        fgLastClick = { id: null, t: 0 };
        expandNode(node.id);
      } else {
        fgLastClick = { id: node.id, t: now };
        showNodePanel(node.id);
      }
    })
    .onBackgroundClick(() => hideNodePanel())
    .onRenderFramePre(drawHulls)
    .onEngineStop(() => {
      if (fgNeedsFit) {
        fgNeedsFit = false;
        fg.zoomToFit(400, 60);
      }
    });
  applyForces();
  new ResizeObserver(() => {
    const rect = container.getBoundingClientRect();
    if (rect.width && rect.height) fg.width(rect.width).height(rect.height);
  }).observe(container);
  return fg;
}

async function refreshGraph() {
  const g = state.graph;
  stopReplay({ restore: false }); // we are about to replace the canvas data anyway
  hideNodePanel();
  let data;
  // A failed fetch (Memgraph busy under heavy index writes) must not leave the view
  // dark with no force-graph instance — keep what is on canvas, flag the error, and
  // let pollStatus retry on its next tick.
  try {
    if (g.mode === "custom" && state.customGraph) {
      data = { ...state.customGraph, truncated: false };
      $("#graph-scope").textContent = "Query result";
    } else if (g.mode === "neighborhood" && g.root) {
      data = await api(`/graph?root=${encodeURIComponent(g.root)}&depth=${g.depth}&rels=${[...g.rels].join(",")}`);
      const rootNode = data.nodes.find((n) => n.id === g.root);
      g.rootName = rootNode ? rootNode.name : g.root;
      $("#graph-scope").textContent = `Around: ${g.rootName}`;
    } else if (g.level === "all") {
      g.mode = "full";
      data = await api("/graph?mode=full");
      $("#graph-scope").textContent = "Whole project · every entity";
    } else {
      g.mode = "overview";
      data = await api("/graph?mode=overview");
      $("#graph-scope").textContent = "Modules overview";
    }
    // depth is a neighborhood-only control; in the unrooted views it does nothing.
    $("#depth-wrap").classList.toggle("hidden", g.mode !== "neighborhood");
    state.live.graphError = false;
    state.live.userCam = false; // fresh render: the follow-camera takes over again
  } catch (e) {
    state.live.graphError = true;
    $("#graph-count").textContent = "graph unavailable — retrying…";
    return;
  }

  fgData = makeGraphElements(data);
  rebuildNeighbors();
  ensureForceGraph();
  applyLayoutMode();
  fgNeedsFit = true;
  fg.graphData(fgData);
  applyForces(); // after graphData: the link force recomputes strengths from the new links
  fg.d3ReheatSimulation();
  // Interim fit while the simulation is still spreading; the final fit runs on engine stop
  setTimeout(() => { if (fgNeedsFit) fg.zoomToFit(300, 60); }, 800);

  buildLegend(fgData.nodes, fgData.links);
  $("#graph-count").textContent =
    `${fgData.nodes.length} nodes · ${fgData.links.length} edges${data.truncated ? " · truncated" : ""}`;
  $("#graph-replay").disabled = fgData.nodes.length < 2;
}

function applyLayoutMode() {
  if (!fg) return;
  fg.dagMode(state.graph.layout === "force" ? null : state.graph.layout);
}

function buildLegend(nodes, links) {
  buildLegendFrom(new Set(nodes.map((n) => n.type)), new Set(links.map((e) => e.relation)));
}

function buildLegendFrom(typeSet, relSet) {
  // With folder tinting on, an entity-type key would be a lie — the dots on the
  // canvas are folders, so the key lists folders too.
  const dots = state.gfx.tint
    ? groupLegendRows()
    : [...typeSet].filter((t) => TYPE_COLORS[t]).map((t) => ({ label: t, color: TYPE_COLORS[t] }));
  const rels = [...relSet];
  $("#graph-legend").innerHTML =
    dots.map((d) => `<div class="lg-row"><span class="lg-dot" style="background:${d.color}"></span>${esc(d.label)}</div>`).join("") +
    (rels.length && dots.length ? `<div style="height:4px"></div>` : "") +
    rels.map((r) => `<div class="lg-row"><span class="lg-line" style="background:${REL_COLORS[r] || "#57564f"}"></span>${r.toLowerCase()}</div>`).join("");
}

// Folders currently on canvas, biggest first — capped, because a wide repo has
// more folders than the legend has room for.
function groupLegendRows() {
  const count = new Map();
  for (const n of fgData.nodes) {
    if (replay.active && n.__shown !== true) continue;
    count.set(n.group, (count.get(n.group) || 0) + 1);
  }
  const top = [...count.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
  const rows = top.map(([g]) => ({
    label: String(g).split("/").filter(Boolean).pop() || g,
    color: GF.groupColor(g),
  }));
  if (count.size > top.length) rows.push({ label: `+${count.size - top.length} more`, color: "transparent" });
  return rows;
}

function hideNodePanel() {
  fgSelectedId = null;
  $("#graph-side").classList.add("hidden");
}

// Add an entity to the canvas next to `fromId`, optionally with the edge that led to it.
function addEntityToCanvas(entity, fromId, relation, direction) {
  if (replay.active) return; // the replay owns the canvas; don't graft onto its staged set
  if (fgData.nodes.some((n) => n.id === entity.id)) return;
  const from = fgData.nodes.find((n) => n.id === fromId);
  const node = {
    id: entity.id, name: entity.name, type: entity.type,
    color: TYPE_COLORS[entity.type] || TYPE_COLORS.code,
    degree: 0, summary: entity.summary || "", file: fileRef(entity),
    x: (from?.x || 0) + (Math.random() - 0.5) * 40,
    y: (from?.y || 0) + (Math.random() - 0.5) * 40,
  };
  node.group = GF.groupKey(node); // or it lands outside every cluster, hull and tint
  fgData.nodes.push(node);
  if (fromId && relation) {
    // The user explicitly walked this edge — draw it even if its type is filtered out
    fgData.links.push({
      source: direction === "in" ? entity.id : fromId,
      target: direction === "in" ? fromId : entity.id,
      relation, color: REL_COLORS[relation] || "#57564f",
      width: 1.5, weight: 1,
    });
  }
  rebuildNeighbors();
  fg.graphData(fgData);
  buildLegend(fgData.nodes, fgData.links);
  $("#graph-count").textContent = `${fgData.nodes.length} nodes · ${fgData.links.length} edges`;
}

// Show full entity details in the side panel. When the entity was reached by
// clicking a relation of the previously selected node, {fromId, relation,
// direction} add it to the canvas so the walk stays visible.
async function showNodePanel(id, via = null) {
  fgSelectedId = id;
  const panel = $("#graph-side");
  const content = $("#graph-side-content");
  panel.classList.remove("hidden");
  content.innerHTML = `<div class="gs-loading">Loading…</div>`;

  let detail;
  try {
    detail = await api(`/entity?id=${encodeURIComponent(id)}`);
  } catch {
    content.innerHTML = `<div class="gs-loading">Failed to load entity.</div>`;
    return;
  }
  if (fgSelectedId !== id) return; // stale: user already clicked elsewhere
  const e = detail.entity;

  if (via) addEntityToCanvas(e, via.fromId, via.relation, via.direction);
  const node = fgData.nodes.find((n) => n.id === id);
  if (node && node.x != null && !replay.active) { // the replay owns a still camera
    fg.centerAt(node.x, node.y, 500);
    // Newly added nodes drift while the simulation settles — re-center once it has
    if (via) setTimeout(() => {
      if (fgSelectedId === id && node.x != null) fg.centerAt(node.x, node.y, 400);
    }, 800);
  }

  const inGraph = (nid) => fgData.nodes.some((n) => n.id === nid);
  const relGroup = (rels, dir) => {
    if (!rels.length) return "";
    const byType = {};
    rels.forEach((r) => (byType[r.relation] = byType[r.relation] || []).push(r));
    return Object.entries(byType).map(([rel, items]) => `
      <div class="rel-group">
        <span class="rel-tag" style="--rel-c:${REL_COLORS[rel] || "#7d7f73"}">${dir === "in" ? "◂ " : "▸ "}${esc(rel)}</span>
        ${items.map((r) => `
          <div class="gs-rel-item" data-id="${esc(r.id)}" data-rel="${esc(r.relation)}" data-dir="${dir}" title="${esc(r.summary || "")}">
            ${chip(r.type)}<span class="name">${esc(r.name)}</span><span class="path">${esc(fileRef(r))}</span>
            ${inGraph(r.id) ? "" : `<span class="plus" title="Not on canvas yet — click to add">＋</span>`}
          </div>`).join("")}
      </div>`).join("");
  };
  const childrenGroup = detail.children.length ? `
    <div class="rel-group">
      <span class="rel-tag" style="--rel-c:${REL_COLORS.CONTAINS}">▸ CONTAINS</span>
      ${detail.children.map((c) => `
        <div class="gs-rel-item" data-id="${esc(c.id)}" data-rel="CONTAINS" data-dir="out" title="${esc(c.summary || "")}">
          ${chip(c.type)}<span class="name">${esc(c.name)}</span><span class="path">${esc(fileRef(c))}</span>
          ${inGraph(c.id) ? "" : `<span class="plus" title="Not on canvas yet — click to add">＋</span>`}
        </div>`).join("")}
    </div>` : "";
  const crumbs = detail.breadcrumbs.map((c) => esc(c.name)).join(" / ");

  content.innerHTML = `
    <div class="gs-head">
      ${chip(e.type)}<h3>${esc(e.name)}</h3>
      <button class="ghost-btn" id="gs-close" title="Close">✕</button>
    </div>
    ${crumbs ? `<div class="gs-crumbs">${crumbs}</div>` : ""}
    <div class="gs-meta mono">${esc(fileRef(e))}${e.line_end ? "–" + e.line_end : ""}</div>
    <div class="gs-actions">
      <button class="tb-btn accent" id="gs-open">Open in Explorer</button>
      <button class="tb-btn" id="gs-expand">Expand</button>
      <button class="tb-btn" id="gs-focus">Focus</button>
    </div>
    ${e.summary ? `<div class="gs-sect"><h4>Summary</h4><div class="gs-summary">${esc(e.summary)}</div></div>` : ""}
    ${e.signature ? `<div class="gs-sect"><h4>Signature</h4><div class="gs-sig mono">${esc(e.signature)}</div></div>` : ""}
    <div class="gs-sect gs-code" id="gs-code-sect"></div>
    ${detail.outgoing.length || detail.children.length ? `<div class="gs-sect"><h4>Outgoing</h4>${relGroup(detail.outgoing, "out")}${childrenGroup}</div>` : ""}
    ${detail.incoming.length ? `<div class="gs-sect"><h4>Incoming</h4>${relGroup(detail.incoming, "in")}</div>` : ""}
  `;

  $("#gs-close").addEventListener("click", () => hideNodePanel());
  $("#gs-open").addEventListener("click", () => openEntity(id));
  $("#gs-expand").addEventListener("click", () => expandNode(id));
  $("#gs-focus").addEventListener("click", () => nav("graph", { root: id, depth: state.graph.depth }));
  content.querySelectorAll(".gs-rel-item").forEach((el) =>
    el.addEventListener("click", () =>
      showNodePanel(el.dataset.id, { fromId: id, relation: el.dataset.rel, direction: el.dataset.dir })));

  loadPanelCode(e, $("#gs-code-sect"));
}

async function loadPanelCode(entity, sect) {
  if (entity.type === "directory") { sect.remove(); return; }
  sect.innerHTML = `<h4>Code</h4><div class="dim" style="font-size:12px">Loading…</div>`;
  const res = await api(`/code?id=${encodeURIComponent(entity.id)}`, { allowError: true });
  if (fgSelectedId !== entity.id) return;
  if (!res.content) { sect.remove(); return; }
  const lines = res.content.split("\n");
  let highlighted;
  try {
    highlighted = res.language
      ? hljs.highlight(res.content, { language: res.language }).value
      : hljs.highlightAuto(res.content).value;
  } catch { highlighted = esc(res.content); }
  const gutter = lines.map((_, i) => res.first_line + i).join("\n");
  sect.innerHTML = `
    <h4>Code</h4>
    <div class="code-card">
      <div class="code-head"><span class="mono">lines ${res.first_line}–${res.first_line + lines.length - 1}</span>
        <span class="grow"></span>${res.source === "chunks" ? `<span title="Reassembled from stored chunks">≈</span>` : ""}</div>
      <div class="code-scroll"><div class="code-grid">
        <div class="code-gutter">${gutter}</div>
        <pre class="code-body"><code class="hljs">${highlighted}</code></pre>
      </div></div>
    </div>`;
}

async function expandNode(id) {
  if (replay.active) return; // read-only during the animation
  const data = await api(`/graph?root=${encodeURIComponent(id)}&depth=1&rels=${[...state.graph.rels].join(",")}`);
  const have = new Set(fgData.nodes.map((n) => n.id));
  const around = fgData.nodes.find((n) => n.id === id);
  const linkKey = (l) => {
    const s = typeof l.source === "object" ? l.source.id : l.source;
    const t = typeof l.target === "object" ? l.target.id : l.target;
    return `${s}→${t}→${l.relation}`;
  };
  const haveLinks = new Set(fgData.links.map(linkKey));
  let added = 0;
  for (const n of data.nodes) {
    if (have.has(n.id)) continue;
    have.add(n.id);
    added++;
    fgData.nodes.push({
      id: n.id, name: n.name, type: n.type,
      color: TYPE_COLORS[n.type] || TYPE_COLORS.code,
      size: n.size, degree: n.degree, summary: n.summary || "", file: fileRef(n),
      // spawn near the expanded node so new nodes settle in place instead of flying in
      x: (around?.x || 0) + (Math.random() - 0.5) * 30,
      y: (around?.y || 0) + (Math.random() - 0.5) * 30,
    });
  }
  for (const e of data.edges) {
    if (!state.graph.rels.has(e.relation)) continue;
    if (!have.has(e.source) || !have.has(e.target)) continue;
    const key = `${e.source}→${e.target}→${e.relation}`;
    if (haveLinks.has(key)) continue;
    haveLinks.add(key);
    added++;
    fgData.links.push({
      source: e.source, target: e.target, relation: e.relation,
      color: REL_COLORS[e.relation] || "#57564f",
      width: 1 + Math.min(Math.sqrt(e.weight || 1), 4),
      weight: e.weight || 1,
    });
  }
  if (added) {
    rebuildNeighbors();
    fg.graphData(fgData); // existing node objects keep their positions; physics absorbs the rest
    buildLegend(fgData.nodes, fgData.links);
    $("#graph-count").textContent = `${fgData.nodes.length} nodes · ${fgData.links.length} edges`;
    setTimeout(() => fg.zoomToFit(400, 60), 600);
  }
}

// ---------------------------------------------------------------------------
// Replay: "how this graph grew", Obsidian-style. Strictly read-only eye candy.
//
// It reveals the graph that is already on the canvas, in build order, WITHOUT
// touching the layout: every node keeps the position it settled into, and the
// reveal runs entirely through the nodeVisibility/linkVisibility paint filters.
// Nothing is inserted into the simulation, so it is never re-heated — no
// convulsing, no drift, and the camera can be framed once and left alone.
// Anything that would replace the canvas data (a refresh, a route change, a
// live-index merge) stops the replay first, and live merges are deferred while
// it runs.
// ---------------------------------------------------------------------------

// Motion is kinematic, never physical: the final layout is already known, so a
// node simply glides to its own spot. It starts SPROUT_FROM of the way out from
// the node that introduced it — the whole graph never scales, only the newcomer
// moves.
const REPLAY_MOTION = { sproutMs: 900, sproutFrom: 0.3, stemMs: 520 };

const replay = {
  active: false, paused: false, speed: 1,
  order: [],      // nodes, in reveal order
  anchorAt: [],   // anchorAt[i] = the node order[i] buds off from (or null)
  linksAt: [],    // linksAt[i] = links whose both endpoints are visible once order[i] lands
  i: 0, shownLinks: 0,
  types: new Set(), rels: new Set(), // legend contents so far
  raf: null, last: 0, acc: 0, rate: 1, pausedAt: 0,
  missedUpdate: false, endTimer: null,
  group: null, centroids: new Map(), baseZoom: 1, // cinematic camera
  recording: false,                                // this run is being captured to video
};

const endId = (v) => (typeof v === "object" ? v.id : v);

// Plausible build order: the CONTAINS spine first (root dir → subdirs → modules →
// their entities, breadth-first, directories before files), then anything reachable
// over CALLS/IMPORTS/… so the graph grows outward instead of jumping between islands.
function buildReplayOrder(nodes, links) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const kids = new Map();
  const parent = new Map();
  const adj = new Map();
  const push = (map, k, v) => { if (!map.has(k)) map.set(k, []); map.get(k).push(v); };

  for (const l of links) {
    const s = endId(l.source), t = endId(l.target);
    if (!byId.has(s) || !byId.has(t) || s === t) continue;
    push(adj, s, t);
    push(adj, t, s);
    if (l.relation === "CONTAINS" && !parent.has(t)) {
      parent.set(t, s);
      push(kids, s, t);
    }
  }

  const rank = { directory: 0, module: 1, class: 2, interface: 2, struct: 2, enum: 2, trait: 2 };
  const cmp = (a, b) => {
    const na = byId.get(a), nb = byId.get(b);
    return (rank[na.type] ?? 3) - (rank[nb.type] ?? 3)
      || String(na.file || "").localeCompare(String(nb.file || ""))
      || String(na.name || "").localeCompare(String(nb.name || ""));
  };

  const seeds = nodes.filter((n) => !parent.has(n.id)).map((n) => n.id).sort(cmp);
  const placed = new Set();
  const order = [];
  const anchorAt = []; // the already-revealed node each one buds off from (or null)
  const queue = [];
  let seedIdx = 0;

  while (order.length < nodes.length) {
    if (!queue.length) {
      while (seedIdx < seeds.length && placed.has(seeds[seedIdx])) seedIdx++;
      let start = seeds[seedIdx];
      if (start === undefined) {
        const left = nodes.find((n) => !placed.has(n.id));
        if (!left) break;
        start = left.id;
      }
      queue.push({ id: start, from: null });
    }
    const { id, from } = queue.shift();
    if (placed.has(id)) continue;
    placed.add(id);
    order.push(byId.get(id));
    anchorAt.push(from ? byId.get(from) : null);
    // CONTAINS children first (the tree unfolds), then whatever else this node
    // reaches, so the reveal spreads outward instead of hopping between islands.
    for (const c of (kids.get(id) || []).filter((c) => !placed.has(c)).sort(cmp))
      queue.push({ id: c, from: id });
    for (const o of (adj.get(id) || []).filter((o) => !placed.has(o) && parent.get(o) !== id).sort(cmp))
      queue.push({ id: o, from: id });
  }

  return finalizeOrder(order, anchorAt, links);
}

// An edge only exists once both of its ends do, so it lands with the later one.
function finalizeOrder(order, anchorAt, links) {
  const idx = new Map(order.map((n, i) => [n.id, i]));
  const linksAt = order.map(() => []);
  for (const l of links) {
    const s = idx.get(endId(l.source));
    const t = idx.get(endId(l.target));
    if (s === undefined || t === undefined) continue;
    linksAt[Math.max(s, t)].push(l);
  }
  return { order, anchorAt, linksAt };
}

// The other reveal order: a ripple outward from the centre of the finished
// layout. It ignores structure entirely, which is exactly why it reads as one
// object coming into existence rather than a tree being walked.
function buildWaveOrder(nodes, links) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const adj = new Map();
  for (const l of links) {
    const s = endId(l.source), t = endId(l.target);
    if (!byId.has(s) || !byId.has(t) || s === t) continue;
    if (!adj.has(s)) adj.set(s, []);
    if (!adj.has(t)) adj.set(t, []);
    adj.get(s).push(t);
    adj.get(t).push(s);
  }
  let cx = 0, cy = 0;
  for (const n of nodes) { cx += n.x || 0; cy += n.y || 0; }
  cx /= nodes.length || 1;
  cy /= nodes.length || 1;
  const dist = (n) => Math.hypot((n.x || 0) - cx, (n.y || 0) - cy);
  const order = [...nodes].sort((a, b) => dist(a) - dist(b) || String(a.id).localeCompare(String(b.id)));
  // The stem has to come from something already on screen: pick the neighbour
  // that was revealed most recently before this node, if there is one at all.
  const rank = new Map(order.map((n, i) => [n.id, i]));
  const anchorAt = order.map((n, i) => {
    let best = null, bestRank = -1;
    for (const id of adj.get(n.id) || []) {
      const r = rank.get(id);
      if (r !== undefined && r < i && r > bestRank) { bestRank = r; best = byId.get(id); }
    }
    return best;
  });
  return finalizeOrder(order, anchorAt, links);
}

function showReplayBar(on) {
  $("#replay-bar").classList.toggle("hidden", !on);
  $("#graph-canvas-wrap").classList.toggle("replaying", on);
  $("#graph-replay").classList.toggle("active", on);
  $("#graph-replay").textContent = on ? "■ Stop" : "▶ Replay";
}

function updateReplayBar() {
  const total = replay.order.length;
  $("#replay-label").textContent = `${replay.i} / ${total}`;
  $("#replay-fill").style.width = `${total ? (100 * replay.i) / total : 0}%`;
  const cur = replay.order[Math.max(0, replay.i - 1)];
  // fileRef() carries a :line suffix that is always :1 for whole modules — drop it.
  $("#replay-now").textContent = cur ? String(cur.file || cur.name || "").replace(/:\d+$/, "") : "";
}

function startReplay() {
  if (replay.active || !fg || fgData.nodes.length < 2) return;
  const { order, anchorAt, linksAt } = state.gfx.order === "wave"
    ? buildWaveOrder(fgData.nodes, fgData.links)
    : buildReplayOrder(fgData.nodes, fgData.links);
  if (order.length < 2) return;

  hideNodePanel();
  // Freeze the layout and remember where every node belongs. Pinning through
  // fx/fy makes our positions authoritative whether or not the simulation is
  // still ticking, so the animation can never be fought by the physics.
  for (const n of fgData.nodes) {
    n.__shown = false;
    delete n.revealAt;
    delete n.addedAt;
    n.__tx = n.x;
    n.__ty = n.y;
    n.__fx0 = n.fx;
    n.__fy0 = n.fy;
    n.fx = n.x;
    n.fy = n.y;
  }
  for (const l of fgData.links) { l.__shown = false; delete l.revealAt; delete l.__grow; }
  // Reveal rank, so a stem can tell which of its two ends is the older one.
  order.forEach((n, i) => { n.__ord = i; });
  // Where each directory ends up, for the camera to visit. Computed from the
  // final positions, so the camera never chases a moving target.
  replay.centroids = new Map();
  const acc = new Map();
  for (const n of fgData.nodes) {
    if (!acc.has(n.group)) acc.set(n.group, { x: 0, y: 0, n: 0 });
    const a = acc.get(n.group);
    a.x += n.__tx; a.y += n.__ty; a.n++;
  }
  for (const [g, a] of acc) replay.centroids.set(g, { x: a.x / a.n, y: a.y / a.n });
  replay.group = null;
  replay.active = true;
  replay.paused = false;
  replay.missedUpdate = false;
  replay.order = order;
  replay.anchorAt = anchorAt;
  replay.linksAt = linksAt;
  replay.i = 0;
  replay.shownLinks = 0;
  replay.types = new Set();
  replay.rels = new Set();
  replay.acc = 0;
  replay.last = performance.now();
  replay.endTimer = null;
  // Aim for a watchable run: ~7 s for a handful of nodes, capped at ~30 s.
  replay.rate = order.length / Math.min(30, Math.max(7, order.length * 0.07));

  // A settled layout stops the render loop; the reveal is time-based, so it needs
  // a frame every tick regardless of whether the physics is still running.
  fg.autoPauseRedraw(false);
  // Frame the FINAL extent once, before anything is compressed. Nothing ever
  // grows past it, so a static camera can sit still for the whole show; the
  // cinematic one uses this as the level it pushes in from and returns to.
  fg.zoomToFit(600, 70);
  // zoomToFit animates, so the level it lands on is only readable once it has.
  replay.baseZoom = fg.zoom();
  setTimeout(() => { if (replay.active) replay.baseZoom = fg.zoom(); }, 650);
  buildLegendFrom(replay.types, replay.rels);
  $("#graph-count").textContent = "0 nodes · 0 edges";
  $("#replay-toggle").textContent = "❚❚";
  showReplayBar(true);
  updateReplayBar();
  replay.raf = requestAnimationFrame(replayTick);
}

function replayTick(now) {
  if (!replay.active) return;
  const dt = Math.min(0.25, (now - replay.last) / 1000);
  replay.last = now;
  if (!replay.paused) {
    replay.acc += dt * replay.rate * replay.speed;
    let added = 0;
    let legendDirty = false;
    // Animation lengths follow the speed control, otherwise a 4x run would be a
    // blur of half-finished stems.
    const stemMs = REPLAY_MOTION.stemMs / Math.max(0.5, replay.speed);
    const stems = state.gfx.stems;
    while (replay.acc >= 1 && replay.i < replay.order.length) {
      replay.acc -= 1;
      const node = replay.order[replay.i];
      const anchor = replay.anchorAt[replay.i];
      node.__shown = true;
      if (!replay.types.has(node.type)) { replay.types.add(node.type); legendDirty = true; }
      if (stems && anchor) {
        // The stem grows first and the node opens at the end of it: hold the
        // bloom back by exactly the stem's length.
        node.__bloomAt = now + stemMs;
        node.revealAt = now + stemMs;
        node.x = node.fx = node.__tx;
        node.y = node.fy = node.__ty;
      } else {
        node.revealAt = now;
        node.__sproutAt = now;
        node.__anchor = anchor;
      }
      for (const l of replay.linksAt[replay.i]) {
        l.__shown = true;
        l.revealAt = now;
        replay.shownLinks++;
        if (stems) {
          // Grow from whichever end has been on screen longer.
          const s = l.source, t = l.target;
          l.__growFrom = (s.__ord ?? 0) <= (t.__ord ?? 0) ? s : t;
          l.__grow = now;
          l.__growMs = stemMs;
        }
        if (!replay.rels.has(l.relation)) { replay.rels.add(l.relation); legendDirty = true; }
      }
      if (state.gfx.camera === "cinematic") followGroup(node.group);
      replay.i++;
      added++;
    }
    if (added) {
      if (legendDirty) buildLegendFrom(replay.types, replay.rels);
      $("#graph-count").textContent = `${replay.i} nodes · ${replay.shownLinks} edges`;
      updateReplayBar();
    }
    if (replay.i >= replay.order.length && !replay.endTimer) {
      $("#replay-now").textContent = "done";
      // The pull-back is the closing shot: let it finish before tearing down.
      if (state.gfx.camera === "cinematic") fg.zoomToFit(1400, 70);
      replay.endTimer = setTimeout(() => stopReplay({ restore: true }), 2600);
    }
    placeReplayNodes(now);
  }
  replay.raf = requestAnimationFrame(replayTick);
}

// Cinematic camera: push in on each directory as the reveal enters it, and only
// move when the group actually changes — otherwise every node would restart the
// transition and the camera would never arrive anywhere.
function followGroup(group) {
  if (!fg || group === replay.group) return;
  replay.group = group;
  const c = replay.centroids.get(group);
  if (!c) return;
  fg.centerAt(c.x, c.y, 900);
  fg.zoom(Math.min(replay.baseZoom * 2.2, 6), 900);
}

// Glide the nodes that are still budding in toward their own spot. Reveal times
// only ever increase, so the ones in flight are the newest few — walk back from
// the tail and stop at the first one that has already landed. Writes x/y and
// fx/fy together so the result holds whether or not d3 is ticking.
function placeReplayNodes(now) {
  for (let k = replay.i - 1; k >= 0; k--) {
    const n = replay.order[k];
    if (!n.__sproutAt) break;
    const age = (now - n.__sproutAt) / REPLAY_MOTION.sproutMs;
    // Stamps are dropped only on expiry, and ages grow toward the head, so the
    // in-flight nodes always stay a contiguous tail — the break above is safe.
    if (age >= 1) {
      n.x = n.fx = n.__tx;
      n.y = n.fy = n.__ty;
      delete n.__sproutAt;
      delete n.__anchor;
      continue;
    }
    const a = n.__anchor;
    if (!a) { // a seed node: nothing to bud off, it simply appears in place
      n.x = n.fx = n.__tx;
      n.y = n.fy = n.__ty;
      continue;
    }
    const bx = a.__tx + (n.__tx - a.__tx) * REPLAY_MOTION.sproutFrom;
    const by = a.__ty + (n.__ty - a.__ty) * REPLAY_MOTION.sproutFrom;
    const e = easeOutCubic(age);
    n.x = n.fx = bx + (n.__tx - bx) * e;
    n.y = n.fy = by + (n.__ty - by) * e;
  }
}

// End the show: drop the paint filters and put every node back on the exact spot
// it held before, unpinned. Nothing was ever added to or removed from the graph,
// so there is nothing else to undo. `restore: false` is for callers about to
// replace the graph data themselves (refreshGraph, route changes).
function stopReplay({ restore = true } = {}) {
  if (!replay.active) return;
  replay.active = false;
  if (replay.raf) cancelAnimationFrame(replay.raf);
  replay.raf = null;
  clearTimeout(replay.endTimer);
  replay.endTimer = null;
  replay.order = [];
  replay.anchorAt = [];
  replay.linksAt = [];
  showReplayBar(false);
  if (!fg) return;
  fg.autoPauseRedraw(true);
  // Put every node back on its own mark and hand the layout back to d3 — pins
  // included, in case the viewer had dragged something before the show.
  for (const n of fgData.nodes) {
    n.__shown = true;
    delete n.revealAt;
    if (n.__tx != null) { n.x = n.__tx; n.y = n.__ty; }
    n.fx = n.__fx0;
    n.fy = n.__fy0;
    n.vx = 0;
    n.vy = 0;
    delete n.__tx; delete n.__ty; delete n.__fx0; delete n.__fy0;
    delete n.__sproutAt; delete n.__anchor; delete n.__bloomAt; delete n.__ord;
  }
  for (const l of fgData.links) {
    l.__shown = true;
    delete l.revealAt; delete l.__grow; delete l.__growFrom; delete l.__growMs;
  }
  // A recording exists to capture the show — when the show ends, so does it.
  if (replay.recording) {
    replay.recording = false;
    GraphRecorder.stop();
  }
  if (!restore) return;
  buildLegend(fgData.nodes, fgData.links);
  $("#graph-count").textContent = `${fgData.nodes.length} nodes · ${fgData.links.length} edges`;
  // Counts moved under us while we were animating — pick up the fresh graph.
  if (replay.missedUpdate) refreshGraph();
}

function toggleReplayPause() {
  if (!replay.active) return;
  replay.paused = !replay.paused;
  if (replay.paused) {
    replay.pausedAt = performance.now();
  } else {
    // Sprouts are wall-clock based: carry them over the pause so the nodes that
    // were mid-glide resume instead of snapping to their spot.
    const held = performance.now() - replay.pausedAt;
    for (let k = replay.i - 1; k >= 0; k--) {
      const n = replay.order[k];
      if (!n.__sproutAt) break;
      n.__sproutAt += held;
    }
  }
  $("#replay-toggle").textContent = replay.paused ? "▶" : "❚❚";
}

// ---------------------------------------------------------------------------
// Recording: capture the replay to a video file, for READMEs and demos
// ---------------------------------------------------------------------------

let recTimer = null;

function setRecUI(on) {
  const btn = $("#graph-rec");
  btn.classList.toggle("recording", on);
  clearInterval(recTimer);
  recTimer = null;
  if (!on) { btn.textContent = "⏺"; return; }
  const tick = () => {
    const s = Math.round(GraphRecorder.elapsedMs() / 1000);
    btn.textContent = `■ ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  };
  tick();
  recTimer = setInterval(tick, 500);
}

// The button exists to capture the replay, so it starts both at once and the
// capture ends when the show does. Pressing it again cuts the take short and
// saves whatever was captured.
function toggleRecording() {
  if (GraphRecorder.active()) {
    replay.recording = false;
    GraphRecorder.stop();
    setRecUI(false);
    return;
  }
  const canvas = $("#graph-canvas canvas");
  if (!canvas) return;
  if (!GraphRecorder.isSupported()) {
    $("#graph-count").textContent = "video capture is not available in this browser";
    return;
  }
  const project = state.projects.find((p) => p.project === state.project);
  const started = GraphRecorder.start(canvas, {
    name: `${project ? project.display_name : state.project || "graph"}-replay`,
    onStop: () => setRecUI(false),
    onError: (msg) => { setRecUI(false); $("#graph-count").textContent = msg; },
  });
  if (!started) return;
  setRecUI(true);
  if (!replay.active) startReplay();
  replay.recording = true;
}

// ---------------------------------------------------------------------------
// Live index mode: poll /status, merge fresh nodes into the running simulation
// ---------------------------------------------------------------------------

// Merge the current view's fresh graph into fgData without a rebuild: new nodes spawn
// near an already-placed neighbor (expandNode's pattern) and pop in; existing nodes are
// updated in place. Removals are deliberately NOT applied mid-index — the exact set is
// restored by the full refreshGraph() that runs when indexing finishes.
async function liveMergeGraph() {
  const g = state.graph;
  if (state.live.merging || g.mode === "custom" || replay.active) return;
  state.live.merging = true;
  try {
    if (!fg) {
      // The initial render never happened (its fetch failed) — recover with a full
      // refresh instead of silently skipping every merge.
      await refreshGraph();
      return;
    }
    const data = (g.mode === "neighborhood" && g.root)
      ? await api(`/graph?root=${encodeURIComponent(g.root)}&depth=${g.depth}&rels=${[...g.rels].join(",")}`)
      : await api(g.level === "all" ? "/graph?mode=full" : "/graph?mode=overview");
    const fresh = makeGraphElements(data);

    const wasEmpty = fgData.nodes.length === 0;
    const byId = new Map(fgData.nodes.map((n) => [n.id, n]));
    const linkKey = (l) => {
      const s = typeof l.source === "object" ? l.source.id : l.source;
      const t = typeof l.target === "object" ? l.target.id : l.target;
      return `${s}→${t}→${l.relation}`;
    };
    const haveLinks = new Set(fgData.links.map(linkKey));

    // Neighbor index over the fresh links, for seeding new nodes next to family.
    const freshNeighbors = new Map();
    for (const l of fresh.links) {
      if (!freshNeighbors.has(l.source)) freshNeighbors.set(l.source, []);
      if (!freshNeighbors.has(l.target)) freshNeighbors.set(l.target, []);
      freshNeighbors.get(l.source).push(l.target);
      freshNeighbors.get(l.target).push(l.source);
    }
    let cx = 0, cy = 0;
    if (fgData.nodes.length) {
      for (const n of fgData.nodes) { cx += n.x || 0; cy += n.y || 0; }
      cx /= fgData.nodes.length; cy /= fgData.nodes.length;
    }

    let added = 0;
    for (const n of fresh.nodes) {
      const existing = byId.get(n.id);
      if (existing) {
        // Radius and tooltips stay honest as summarization fills in.
        existing.size = n.size; existing.degree = n.degree; existing.summary = n.summary;
        continue;
      }
      const anchorId = (freshNeighbors.get(n.id) || []).find((id) => byId.has(id));
      const anchor = anchorId ? byId.get(anchorId) : null;
      const node = {
        ...n,
        x: (anchor ? anchor.x : cx) + (Math.random() - 0.5) * 60,
        y: (anchor ? anchor.y : cy) + (Math.random() - 0.5) * 60,
        addedAt: performance.now(),
      };
      fgData.nodes.push(node);
      byId.set(n.id, node);
      added++;
    }
    for (const l of fresh.links) {
      const key = `${l.source}→${l.target}→${l.relation}`;
      if (haveLinks.has(key) || !byId.has(l.source) || !byId.has(l.target)) continue;
      haveLinks.add(key);
      // Resolve endpoints to node OBJECTS here: force-graph resolves string ids lazily,
      // and a physics tick landing between our push and that re-init hits the raw string
      // ("Cannot create property 'vx' on string …") and kills the render loop for good.
      fgData.links.push({ ...l, source: byId.get(l.source), target: byId.get(l.target) });
      added++;
    }

    if (added) {
      rebuildNeighbors();
      fg.graphData(fgData);
      fg.d3ReheatSimulation();
      buildLegend(fgData.nodes, fgData.links);
      $("#graph-count").textContent = `${fgData.nodes.length} nodes · ${fgData.links.length} edges`;
      // Follow the growing graph: every reheat re-energizes the layout, so the cloud
      // expands and drifts — without a follow-fit it escapes a static viewport within
      // minutes (black canvas). The camera is only ours until the user grabs it.
      if (wasEmpty || !state.live.userCam) setTimeout(() => { if (fg) fg.zoomToFit(400, 60); }, 500);
    }
  } catch (e) {
    // transient API error: the next poll tick retries
  } finally {
    state.live.merging = false;
  }
}

function updateLiveBadge(s) {
  const badge = $("#live-badge");
  if (!badge) return;
  if (!s.indexing) {
    badge.classList.add("hidden");
    return;
  }
  badge.classList.remove("hidden");
  $("#live-badge-text").textContent = `indexing ${s.summarized}/${s.summarizable}`;
  const pct = s.summarizable ? Math.round((s.summarized / s.summarizable) * 100) : 0;
  $("#live-fill").style.width = `${pct}%`;
}

async function refreshProjectSelect() {
  try {
    const res = await fetch("/api/projects");
    const projects = (await res.json()).projects || [];
    if (!projects.length) return;
    state.projects = projects;
    const sel = $("#project-select");
    sel.innerHTML = projects.map((p) =>
      `<option value="${esc(p.project)}">${esc(p.display_name)} · ${p.entities}</option>`).join("");
    if (state.project) sel.value = state.project;
  } catch (e) { /* next tick retries */ }
}

// Always-on 2s heartbeat. Cheap when idle (one lock probe + two aggregations, no
// graph refetch unless the counts fingerprint moved). Raw fetch, not api(): a dead
// Memgraph must degrade to silence, not error alerts every 2 seconds.
async function pollStatus() {
  if (document.hidden) return;
  if (!state.project) {
    // A project may appear mid-poll — e.g. a wiped repo being re-indexed right now.
    if (!state.projects.length) {
      await refreshProjectSelect();
      if (state.projects.length) await route();
    }
    return;
  }
  // The hash may point at a project that appeared after boot (a fresh index of a
  // wiped or brand-new repo): route() rejected it against the stale project list,
  // so refresh the list and re-route once it shows up.
  const parsed = parseHash();
  if (parsed && parsed.project !== state.project
      && !state.projects.some((p) => p.project === parsed.project)) {
    await refreshProjectSelect();
    if (state.projects.some((p) => p.project === parsed.project)) {
      await route();
      return;
    }
  }

  let s;
  try {
    const res = await fetch(`/api/projects/${encodeURIComponent(state.project)}/status`);
    s = await res.json();
  } catch (e) {
    return;
  }
  if (!s || s.error) return;

  updateLiveBadge(s);
  const key = `${s.nodes}:${s.edges}:${s.summarized}`;
  const first = state.live.key === null; // view just booted — it fetched its own data
  const changed = !first && key !== state.live.key;
  const finished = !s.indexing && state.live.indexing; // true→false edge
  state.live.key = key;
  state.live.indexing = s.indexing;

  if (state.view === "graph") {
    // The replay owns the canvas while it runs: remember that the graph moved and
    // pick the fresh one up when it ends, instead of merging into the animation.
    if (replay.active) { if (changed || finished) replay.missedUpdate = true; }
    else if (state.live.graphError) await refreshGraph();
    else if (s.indexing && changed) await liveMergeGraph();
    // One exact refresh once the run ends (applies removals), or when an entire
    // run happened between two ticks.
    else if (finished || (!s.indexing && changed)) await refreshGraph();
  } else if (finished && state.view === "overview") {
    renderOverview();
  }
  if (finished) {
    state.treeCache = new Map();
    refreshProjectSelect();
  }
}

// ---------------------------------------------------------------------------
// Query view
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
  status.textContent = "Running…";
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

  if (res.graph?.nodes?.length) {
    const btn = document.createElement("button");
    btn.className = "tb-btn";
    btn.textContent = `Show as graph (${res.graph.nodes.length} nodes, ${res.graph.edges.length} edges)`;
    btn.addEventListener("click", () => {
      state.customGraph = res.graph;
      state.graph.mode = "custom";
      nav("graph", { custom: "1" });
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
      return `<span class="rel-tag" style="--rel-c:${REL_COLORS[v.relation] || "#7d7f73"}">${esc(v.relation)}</span>`;
    if (typeof v === "object" && v.__kind === "path")
      return v.nodes.map((n) => cell(n)).join(" → ");
    if (typeof v === "object") return `<span class="cell-json">${esc(JSON.stringify(v, null, 1))}</span>`;
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
// ---------------------------------------------------------------------------

let searchTimer = null;
let lastSearchQ = "";

function searchItem(hit, score) {
  return `<div class="sr-item" data-id="${esc(hit.id)}" title="${esc(hit.summary || "")}">
    ${chip(hit.type)}<span class="name">${esc(hit.name)}</span>
    <span class="path">${esc(fileRef(hit))}</span>${score ? `<span class="dim">${hit.score}</span>` : ""}
  </div>`;
}

async function doSearch(q, mode = "quick") {
  lastSearchQ = q;
  const box = $("#search-results");
  if (!q.trim()) { box.classList.add("hidden"); return; }
  const res = await api(`/search?q=${encodeURIComponent(q)}&mode=${mode}`, { allowError: true });
  if (lastSearchQ !== q) return; // stale
  let html = "";
  if (res.name?.length) html += `<div class="sr-section">Name matches</div>` + res.name.map((h) => searchItem(h)).join("");
  if (res.text?.length) html += `<div class="sr-section">Full-text (BM25)</div>` + res.text.map((h) => searchItem(h, true)).join("");
  if (res.semantic?.length) html += `<div class="sr-section">Semantic</div>` + res.semantic.map((h) => searchItem(h, true)).join("");
  if (!html) html = `<div class="sr-note">No matches.</div>`;
  for (const note of res.notes || []) html += `<div class="sr-note">${esc(note)}</div>`;
  if (mode !== "semantic")
    html += `<div class="sr-actions"><button class="tb-btn" id="sr-semantic">Semantic search (uses embedding API)</button></div>`;
  box.innerHTML = html;
  box.classList.remove("hidden");
  box.querySelectorAll(".sr-item").forEach((el) =>
    el.addEventListener("click", () => {
      box.classList.add("hidden");
      $("#search-input").blur();
      openEntity(el.dataset.id);
    }));
  $("#sr-semantic")?.addEventListener("click", () => doSearch(q, "semantic"));
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
    let idx = items.indexOf(sel);
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

async function boot() {
  document.querySelectorAll("#nav a").forEach((a) =>
    a.addEventListener("click", (ev) => { ev.preventDefault(); if (state.project) nav(a.dataset.view); }));
  $("#brand").addEventListener("click", (ev) => { ev.preventDefault(); if (state.project) nav("overview"); });
  $("#project-select").addEventListener("change", (ev) => {
    state.project = ev.target.value;
    state.treeCache = new Map();
    $("#tree").innerHTML = "";
    nav("overview");
  });
  $("#graph-depth").addEventListener("change", (ev) => { state.graph.depth = +ev.target.value; refreshGraph(); });
  $("#graph-level").addEventListener("change", (ev) => {
    state.graph.level = ev.target.value;
    // Switching the level is a different question about the whole project, so it
    // leaves any rooted view the same way the "Whole project" button does.
    const wasRooted = state.graph.root !== null;
    state.graph.root = null;
    state.graph.mode = "overview";
    // nav() only redraws when the hash actually changes, and on an unrooted view
    // it is already the one we would navigate to.
    if (wasRooted) nav("graph"); else refreshGraph();
  });
  $("#graph-summarized").addEventListener("change", (ev) => {
    state.graph.summarizedOnly = ev.target.checked;
    refreshGraph();
  });
  $("#graph-layout").addEventListener("change", (ev) => {
    state.graph.layout = ev.target.value;
    if (fg) {
      applyLayoutMode();
      fgNeedsFit = true;
      fg.d3ReheatSimulation();
    }
  });
  $("#graph-fit").addEventListener("click", () => {
    state.live.userCam = false; // explicit Fit re-arms the live follow-camera
    fg?.zoomToFit(400, 40);
  });
  $("#graph-replay").addEventListener("click", () => {
    if (replay.active) stopReplay({ restore: true });
    else startReplay();
  });
  renderGfxPanel();
  $("#graph-gfx").addEventListener("click", () => toggleGfxPanel());
  $("#gfx-close").addEventListener("click", () => toggleGfxPanel(false));
  $("#gfx-reset").addEventListener("click", () => {
    state.gfx = { ...GFX_DEFAULTS };
    saveGfx();
    renderGfxPanel();
    applyGfx("forces");
    buildLegend(fgData.nodes, fgData.links);
  });
  $("#graph-rec").addEventListener("click", toggleRecording);
  $("#replay-toggle").addEventListener("click", toggleReplayPause);
  $("#replay-stop").addEventListener("click", () => stopReplay({ restore: true }));
  $("#replay-speed").addEventListener("change", (ev) => { replay.speed = +ev.target.value || 1; });
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape") return;
    // Esc peels one layer at a time: the panel first, then the replay.
    if (!$("#gfx-panel").classList.contains("hidden")) { toggleGfxPanel(false); return; }
    if (replay.active) stopReplay({ restore: true });
  });
  $("#graph-png").addEventListener("click", () => {
    const canvas = $("#graph-canvas canvas");
    if (!canvas) return;
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `${state.project}-graph.png`;
    a.click();
  });
  $("#graph-overview-btn").addEventListener("click", () => {
    state.graph.mode = "overview"; state.graph.root = null;
    nav("graph");
  });
  $("#tree-collapse").addEventListener("click", () => {
    document.querySelectorAll("#tree .tkids").forEach((k, i) => { if (i > 0) k.classList.add("hidden"); });
    document.querySelectorAll("#tree .trow .twisty").forEach((t, i) => { if (i > 0 && t.textContent === "▾") t.textContent = "▸"; });
  });
  wireSearch();

  let projects = [];
  try {
    const res = await fetch("/api/projects");
    projects = (await res.json()).projects || [];
  } catch (e) {
    $("#empty-status").textContent = "Cannot reach the API — is Memgraph running? (docker compose up -d)";
  }
  state.projects = projects;
  const sel = $("#project-select");
  sel.innerHTML = projects.map((p) =>
    `<option value="${esc(p.project)}">${esc(p.display_name)} · ${p.entities}</option>`).join("");
  window.addEventListener("hashchange", route);
  await route();
  setInterval(pollStatus, 2000);
}

boot();
