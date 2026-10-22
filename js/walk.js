/* walk.js — the page is a walk through the graph. Every chapter stands on a real node of data/graph.js; scrolling flies
   the camera along the real shortest route between one chapter's node and the next, and the sheet for that chapter
   opens beside it. Nothing here is drawn from anything but the export. */
(function () {
  'use strict';
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v)), lerp = (a, b, t) => a + (b - a) * t;
  const ease = u => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
  const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const COL = { C: 'var(--pink)', I: 'var(--violet)', N: 'var(--acid)', M: 'var(--acid)', P: 'inherit' };
  const NAME = { C: 'CALLS', I: 'IMPORTS', N: 'INHERITS', M: 'IMPLEMENTS', P: 'CONTAINS' };
  const fmt = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');

  const G = window.Pack.model(window.INFLO_GRAPH);
  const stage = document.getElementById('graph');
  const pack = new window.Pack(stage, G, { reduced, onSelect: n => { if (cur === 3) showSummary(n, true); } });
  const srcNodes = G.nodes.filter(n => n.t !== 'directory');

  // ---- chapters: each stands on a real node; `fit` says which disc the camera frames
  const CH = [
    { tag: 'SPECIMEN', anchor: 'src/inflorescence/server.py', fit: 'root', frac: 0.86 },
    { tag: 'PARSE', anchor: 'src/inflorescence/code_indexer/graph_builder.py::GraphBuilder.build', fit: 'parent', frac: 0.62 },
    { tag: 'GRAPH', anchor: 'src/inflorescence/db/repository.py::GraphRepository', fit: 'self', frac: 0.62 },
    { tag: 'SUMMARY', anchor: 'src/inflorescence/code_indexer/summarizer.py::summarize_hierarchy', fit: 'parent', frac: 0.72 },
    { tag: 'SEARCH', anchor: 'src/inflorescence/tools/search.py', fit: 'root', frac: 0.84 },
    { tag: 'WATCH', anchor: 'src/inflorescence/watcher.py::_DebouncedHandler._handle', fit: 'dir:src/inflorescence', frac: 0.84 },
    { tag: 'COST', anchor: 'src/inflorescence/cost.py::estimate_index_cost', fit: 'parent', frac: 0.5 },
    { tag: 'INSTALL', anchor: 'src/inflorescence/__main__.py::_cmd_init', fit: 'parent', frac: 0.66 },
    { tag: 'TOOLS', anchor: 'src/inflorescence/server.py::create_server', fit: 'root', frac: 0.86 },
  ];
  const N = CH.length;
  for (const ch of CH) {
    ch.node = G.byId[ch.anchor] || srcNodes.find(n => n.id.startsWith(ch.anchor.split('::')[0])) || pack.root;
    const a = ch.node;
    ch.fitNode = ch.fit === 'root' ? pack.root : ch.fit === 'self' ? (a.kids.length ? a : a.par) : ch.fit === 'parent' ? (a.par || pack.root) : (G.byId[ch.fit] || pack.root);
  }
  const routes = CH.slice(1).map((ch, i) => pack.routeBetween(CH[i].node, ch.node));

  // ---- DOM
  const sheets = CH.map((_, k) => document.getElementById('ch' + k));
  const hero = document.getElementById('hero'), hint = document.getElementById('hint'), pager = document.getElementById('pager');
  const strip = document.querySelector('.strip'), corner = document.getElementById('corner');
  const markersEl = document.getElementById('markers');
  const leaders = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); leaders.setAttribute('class', 'leaders'); markersEl.appendChild(leaders);
  const markers = CH.map((ch, k) => {
    const d = document.createElement('div'); d.className = 'mk'; d.addEventListener('click', () => go(k)); markersEl.appendChild(d);
    const l = document.createElement('div'); l.className = 'mkl'; l.innerHTML = `<b>0${k}</b>${ch.tag} · ${esc(shortName(ch.node))}`; l.addEventListener('click', () => go(k)); markersEl.appendChild(l);
    const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line'); leaders.appendChild(ln);
    return { box: d, lab: l, ln };
  });
  function shortName(n) { return n.t === 'module' ? n.p.split('/').pop() : n.t === 'directory' ? (n.id === 'dir:.' ? G.project : n.p.split('/').pop()) + '/' : n.n; }
  function cardHTML(n, k) {
    return `<div class="k">${k}</div><div class="n">${esc(n.n)}</div><div class="t">${esc(n.t)} · ${esc(n.p)}${n.l && n.l[0] ? ':' + n.l[0] + '–' + n.l[1] : ''}</div>` +
      `<div class="s">${esc((n.s || '').slice(0, 360))}${(n.s || '').length > 360 ? '…' : ''}</div>` +
      `<div class="d"><i class="in">← ${n.in.length} incoming</i><i class="out">→ ${n.out.length} outgoing</i>${n.kids.length ? `<i style="opacity:.55">⊃ ${n.kids.length}</i>` : ''}</div>`;
  }
  function routeHTML(k) {
    const R = routes[k - 1]; if (!R) return `<small>no route in the graph from FIG. 0${k - 1}</small>`;
    return `<small>ROUTE FROM FIG. 0${k - 1} · ${R.steps.length} edge${R.steps.length > 1 ? 's' : ''} of this graph</small>` + R.steps.map(st =>
      `<div><span>${esc(shortName(st.a))}</span><em class="${st.k}" style="color:${COL[st.k]}">${st.dir === '>' ? '─' + NAME[st.k] + '→' : '←' + NAME[st.k] + '─'}</em><span>${esc(shortName(st.b))}</span></div>`).join('');
  }
  for (let k = 1; k < N; k++) {
    const a = document.getElementById('anchor' + k), r = document.getElementById('route' + k);
    if (a) a.innerHTML = cardHTML(CH[k].node, 'THIS PAGE STANDS ON');
    if (r) r.innerHTML = routeHTML(k);
  }
  document.getElementById('hero-counts').textContent =
    `This repository, indexed by itself: ${fmt(srcNodes.length)} nodes · ${fmt(pack.edges.length)} edges in src/ and the root documents · ` +
    `whole index incl. tests: ${fmt(Object.values(G.totals).reduce((x, y) => x + y, 0))} nodes · ${fmt(G.chunks)} chunks.`;

  // ---- 01 parse
  const gbId = 'src/inflorescence/code_indexer/graph_builder.py';
  const morph = new window.Morph(document.getElementById('morph-stage'), G, window.INFLO_SRC, { moduleId: gbId });
  document.getElementById('fig1-meta').innerHTML =
    `${gbId}<br>1 module · ${morph.imports.length} IMPORTS edges into ${morph.groups.length} files · ${morph.defs.length} definitions shown of ${G.byId[gbId].kids.length} · ${morph.exts.length} external`;

  // ---- 02 graph
  document.getElementById('fig2-meta').innerHTML =
    `src/ + root documents · ${fmt(srcNodes.length)} nodes · ${fmt(pack.edges.length)} edges<br>` +
    ['module', 'class', 'function', 'method', 'document'].map(t => `${srcNodes.filter(n => n.t === t).length} ${t}s`).join(' · ');
  const legend = document.getElementById('legend');
  for (const k of ['C', 'I', 'N']) {
    const el = document.createElement('span'); el.style.color = { C: '#ff3aa8', I: '#8b7bff', N: '#e6ff3d' }[k];
    el.innerHTML = `<i></i>${NAME[k]} ${pack.counts[k] || 0}`;
    el.addEventListener('click', () => { const on = el.classList.toggle('off'); pack.setShow(k, !on); if (k === 'N') pack.setShow('M', !on); });
    legend.appendChild(el);
  }
  const lg = document.createElement('span'); lg.style.cursor = 'default'; lg.style.opacity = '.55';
  lg.textContent = `○ directory / file / class disc  · function  ◇ class  □ document`; legend.appendChild(lg);

  // ---- 03 summary flower
  const flower = new window.WordFlower(document.getElementById('flower'), 'img/lily.jpg');
  const wanted = [CH[3].anchor, 'src/inflorescence/code_indexer/graph_builder.py', 'src/inflorescence/docker_env.py::ensure_memgraph',
    'src/inflorescence/cost.py::estimate_index_cost', 'src/inflorescence/watcher.py::_DebouncedHandler._handle', 'src/inflorescence/tools/search.py'];
  const list = document.getElementById('entities'); const items = new Map();
  function showSummary(n, fromGraph) {
    flower.setText(n.s || '(no summary yet)', { name: n.id });
    for (const [id, li] of items) li.classList.toggle('on', id === n.id);
    if (fromGraph && !items.has(n.id)) {
      const li = makeItem(n); list.prepend(li); li.classList.add('on');
      if (items.size > 6) { const last = [...items.keys()].pop(); items.get(last).remove(); items.delete(last); }
    }
  }
  function makeItem(n) {
    const li = document.createElement('li'); li.innerHTML = `${esc(n.n)}<small>${n.t} · ${esc(n.p)}</small>`;
    li.addEventListener('click', () => { pack.pinned = n; pack.updateNote(); showSummary(n, false); }); items.set(n.id, li); return li;
  }
  wanted.map(id => G.byId[id]).filter(Boolean).forEach(n => list.appendChild(makeItem(n)));

  // ---- 04 search: three real search_semantic answers recorded from the server (2026-09-10), one structural query computed here
  const QUERIES = [
    { q: 'estimate the dollar cost of indexing before spending anything', tool: 'search_semantic', hits: [
      { id: 'src/inflorescence/project_manager.py::ProjectManager.preview_directory' },
      { id: 'src/inflorescence/project_manager.py::ProjectManager._gate_reindex_cost' },
      { id: 'src/inflorescence/cost.py' },
      { id: 'src/inflorescence/cost.py::estimate_index_cost' },
      { id: 'src/inflorescence/project_manager.py::ProjectManager.index_directory', weak: true },
      { id: 'src/inflorescence/cost.py::IndexCostExceededError.__init__', weak: true }] },
    { q: 'what happens when a file changes on disk after indexing', tool: 'search_semantic', hits: [
      { id: 'src/inflorescence/code_indexer/graph_builder.py::GraphBuilder.has_pending_changes' },
      { id: 'src/inflorescence/code_indexer/graph_builder.py::GraphBuilder.pending_change_paths', weak: true },
      { id: 'src/inflorescence/dashboard/service.py::DashboardService.project_freshness', weak: true },
      { id: 'src/inflorescence/code_indexer/graph_builder.py::GraphBuilder._diff_checksums', weak: true },
      { id: 'src/inflorescence/project_manager.py::ProjectManager._gate_reindex_cost', weak: true },
      { id: 'src/inflorescence/watcher.py::_DebouncedHandler._handle', weak: true }] },
    { q: 'start the memgraph docker container if it is not running', tool: 'search_semantic', hits: [
      { id: 'src/inflorescence/__main__.py::_bootstrap_memgraph' },
      { id: 'src/inflorescence/docker_env.py::memgraph_run_argv' },
      { id: 'src/inflorescence/docker_env.py::ensure_memgraph' },
      { id: 'src/inflorescence/__main__.py::_watch_components' },
      { id: 'src/inflorescence/__main__.py::_memgraph_duplicate_containers' },
      { id: 'src/inflorescence/docker_env.py::containers_on_volume' }] },
  ];
  const most = srcNodes.filter(n => (n.t === 'method' || n.t === 'function') && !/__\w+__$/.test(n.n)).sort((a, b) => b.in.length - a.in.length)[0];
  if (most) QUERIES.push({ q: `who calls ${most.n}?`, tool: `get_related_entities · CALLS · incoming`, hits: most.in.filter(E => E.k === 'C').slice(0, 14).map(E => ({ id: E.s.id })) });
  const chips = document.getElementById('chips'); let activeChip = null;
  QUERIES.forEach(Q => {
    const el = document.createElement('div'); el.className = 'chip';
    el.innerHTML = `${esc(Q.q)}<small>${Q.tool} · ${Q.hits.length} hits</small>`;
    el.addEventListener('click', () => {
      if (activeChip === el) { el.classList.remove('on'); activeChip = null; pack.setHits(null); return; }
      if (activeChip) activeChip.classList.remove('on');
      activeChip = el; el.classList.add('on'); pack.setHits(Q.hits, `${Q.tool} · ${Q.q}`);
    });
    chips.appendChild(el);
  });
  const _setHits = pack.setHits.bind(pack);
  pack.setHits = (h, l) => { _setHits(h, l); if (!h && activeChip) { activeChip.classList.remove('on'); activeChip = null; } };

  // ---- 07 / 08 plates
  new window.Dirt(document.getElementById('install-canvas'), 'img/foxglove.jpg',
    { ascii: .5, asciiCell: 10, dots: .9, dotCell: 3.6, scan: .25, grain: .05, aber: .001, solar: 0, zoom: 1.0, shift: [0.08, 0], seed: 8.1, burstRate: .5, duo: ['#06082a', '#dfe6ff'], reduced });
  new window.Dirt(document.getElementById('end-canvas'), 'img/gloriosa-3.jpg',
    { ascii: .7, asciiCell: 9, dots: 0, scan: 0, grain: .03, aber: 0, solar: 0, zoom: 1.0, fit: 0, shift: [0, 0.02], seed: 1.3, burstRate: .4, mouseAmt: 1, thresh: ['#c6c8c3', '#111111'], reduced });
  document.getElementById('tools').innerHTML = ['index_directory', 'list_entities', 'get_entity_structure', 'get_entity_context', 'get_related_entities',
    'search_entities', 'search_code', 'search_semantic', 'search_hybrid', 'search_text', 'cypher_query', 'graph_schema', 'send_feedback']
    .map(t => `<span>${t}</span>`).join('') + `<em>13 MCP tools · stdio</em>`;

  // ---- hero jitter, as on the plates page
  const h1 = document.getElementById('h1');
  setInterval(() => { if (reduced || cur !== 0) return; h1.classList.add('jit'); setTimeout(() => h1.classList.remove('jit'), 90 + Math.random() * 120); }, 3100 + Math.random() * 2000);

  // ---- the walk
  let cur = -1, small = false, W = innerWidth, H = innerHeight;
  const spacer = document.getElementById('spacer');
  function layoutSpacer() { W = innerWidth; H = innerHeight; small = W < 720; spacer.style.height = (N * 100) + 'vh'; }
  layoutSpacer(); addEventListener('resize', layoutSpacer);
  function viewFor(sheetAmt) {
    if (small) return { x0: 0, y0: 0, w: W, h: lerp(H * 0.62, H * 0.44, sheetAmt) };
    const sw = Math.min(560, W * 0.46);
    return { x0: lerp(W * 0.3, 0, sheetAmt), y0: 0, w: lerp(W * 0.7, W - sw, sheetAmt), h: lerp(H * 0.92, H, sheetAmt) };
  }
  const Z_MAX = 16; // px per unit: a four-function file must not become four dots on an empty screen
  function camFor(ch) { return { x: ch.fitNode.x, y: ch.fitNode.y, z: Math.min(Z_MAX, pack.fitZoom(ch.fitNode, ch.frac)) }; }
  function flight(i, e) {
    const A = camFor(CH[i]), B = camFor(CH[i + 1]), R = routes[i];
    const pts = [[A.x, A.y], ...(R ? R.points : []), [B.x, B.y]];
    let total = 0; const cum = [0];
    for (let k = 1; k < pts.length; k++) { total += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); cum.push(total); }
    const L = e * total; let k = 1; while (k < pts.length - 1 && cum[k] < L) k++;
    const seg = (cum[k] - cum[k - 1]) || 1, u = clamp((L - cum[k - 1]) / seg, 0, 1);
    const x = lerp(pts[k - 1][0], pts[k][0], u), y = lerp(pts[k - 1][1], pts[k][1], u);
    // zoom: log-lerp, dipping far enough at mid-flight to hold both ends in the view
    let ext = 0; for (const p of pts) ext = Math.max(ext, Math.hypot(p[0] - (A.x + B.x) / 2, p[1] - (A.y + B.y) / 2));
    const zBox = 0.42 * Math.min(pack.view.w, pack.view.h) / Math.max(ext, 1);
    const la = Math.log(A.z), lb = Math.log(B.z), mid = (la + lb) / 2;
    const bump = Math.max(0, mid - Math.log(Math.min(zBox, A.z, B.z)));
    return { x, y, z: Math.exp(lerp(la, lb, e) - bump * Math.sin(e * Math.PI)) };
  }
  function enter(k, prev) {
    cur = k; pager.textContent = `0${k} / 0${N - 1}`;
    if (location.hash !== '#0' + k) history.replaceState(null, '', k ? '#0' + k : location.pathname);
    if (k === 1) morph.replay = performance.now() + 350;
    if (k === 3 && (!pack.pinned || prev !== 3)) showSummary(pack.pinned || CH[3].node, !!pack.pinned);
    if (prev === 4 && pack.hits) pack.setHits(null);
    pack.flashing = k === 5;
    pack.kick();
  }
  function go(k) { scrollTo({ top: k * H, behavior: reduced ? 'auto' : 'smooth' }); }
  addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'j') { e.preventDefault(); go(Math.min(N - 1, Math.round(scrollY / H) + 1)); }
    if (e.key === 'ArrowLeft' || e.key === 'k') { e.preventDefault(); go(Math.max(0, Math.round(scrollY / H) - 1)); }
    if (e.key === 'Escape') { pack.pinned = null; pack.setHits(null); pack.releasePan(); pack.updateNote(); }
  });
  let lastP = -1;
  function tick() {
    const p = clamp(scrollY / H, 0, N - 1), i = Math.min(N - 2, Math.floor(p)), s = p - i;
    const u = clamp((s - 0.3) / 0.4, 0, 1), e = ease(u);
    const k = Math.round(p); if (k !== cur) enter(k, cur);
    if (Math.abs(p - lastP) > 0.02) { pack.releasePan(); lastP = p; }
    // sheets fade per chapter; the graph's viewport gives the sheet its column once, on the first flight, and keeps it
    sheets.forEach((el, n) => {
      if (!el) return; const op = 1 - clamp((Math.abs(p - n) - 0.2) / 0.15, 0, 1);
      const off = (1 - op) * 18; el.hidden = op <= 0.001; el.style.opacity = op;
      el.style.transform = small ? `translateY(${off}px)` : `translateX(${off}px)`;
    });
    const sheetAmt = clamp((p - 0.3) / 0.4, 0, 1), sw = small ? 0 : Math.min(560, W * 0.46);
    strip.style.right = (sw * sheetAmt + 22) + 'px'; pager.style.right = (sw * sheetAmt + 22) + 'px';
    corner.style.opacity = 1 - sheetAmt; corner.style.pointerEvents = sheetAmt > 0.5 ? 'none' : '';
    const heroOp = 1 - clamp((p - 0.12) / 0.25, 0, 1); hero.style.opacity = heroOp; hint.style.opacity = heroOp * 0.55; hero.style.pointerEvents = heroOp > 0.5 ? '' : 'none';
    const paperOn = p > N - 1.55; document.body.classList.toggle('paper', paperOn); pack.paperTarget = paperOn ? 1 : 0;
    // camera
    pack.view = viewFor(sheetAmt);
    const T = u <= 0 ? camFor(CH[i]) : u >= 1 ? camFor(CH[i + 1]) : flight(i, e);
    pack.setTarget(T.x, T.y, T.z, false);
    pack.route = u > 0 && u < 1 ? routes[i] : null; pack.routeT = e;
    pack.quiet = u > 0 && u < 1;
    pack.mark = (cur > 0 && cur < N - 1 && u === 0) || (u === 1 && i + 1 < N - 1) ? CH[cur].node : null;
    // markers: all on the two overviews, the current one always, the next one as the flight approaches it
    const overview = Math.max(heroOp, paperOn ? clamp((p - (N - 1.55)) / 0.3, 0, 1) : 0);
    const v = pack.view, shown = [], dwell = u <= 0 || u >= 1;
    markers.forEach((m, n) => {
      const a = CH[n].node, [sx, sy] = pack.toScreen(a.ax === undefined ? a.x : a.ax, a.ay === undefined ? a.y : a.ay);
      const inside = sx > v.x0 + 10 && sx < v.x0 + v.w - 10 && sy > 10 && sy < v.y0 + v.h - 10;
      const pinnedHere = n === cur && dwell && !!pack.mark;
      let op = Math.max(overview, n === cur ? 1 : 0, n === i + 1 ? u : 0); if (!inside) op = 0;
      m.op = op; m.sx = sx; m.sy = sy; m.n = n; m.rs = a.kids.length ? (a.ar === undefined ? a.r : a.ar) * pack.cam.z : 0; m.pinned = pinnedHere;
      const boxOp = pinnedHere ? Math.max(overview, n === i + 1 ? u : 0) : op;
      m.box.classList.toggle('on', boxOp > 0.05); m.box.style.opacity = boxOp; m.box.classList.toggle('cur', n === cur);
      m.box.style.left = sx + 'px'; m.box.style.top = sy + 'px';
      if (op > 0.05) shown.push(m);
    });
    // labels: to the right of the box, pushed apart so none overlap, a leader back to the box
    shown.sort((a, b) => a.sy - b.sy); let lastY = -1e9;
    for (const m of shown) {
      m.lab.classList.add('on'); m.lab.classList.toggle('cur', m.n === cur); m.lab.style.opacity = m.op;
      if (m.pinned && m.rs > 0) {
        // a disc: the label sits on the crosshair's top-left corner, like the frames on the plates
        m.lab.classList.remove('flip'); m.lab.style.left = (m.sx - m.rs - 6) + 'px'; m.lab.style.top = (m.sy - m.rs - 18) + 'px'; m.ln.style.opacity = 0; continue;
      }
      const flip = m.sx > v.x0 + v.w - 250; let ly = m.sy; if (ly - lastY < 17) ly = lastY + 17; lastY = ly;
      const lx = flip ? m.sx - 30 : m.sx + 30;
      m.lab.classList.toggle('flip', flip); m.lab.style.left = lx + 'px'; m.lab.style.top = ly + 'px';
      m.ln.setAttribute('x1', flip ? m.sx - 11 : m.sx + 11); m.ln.setAttribute('y1', m.sy); m.ln.setAttribute('x2', lx); m.ln.setAttribute('y2', ly); m.ln.style.opacity = m.op * 0.7;
    }
    markers.forEach(m => { if (m.op <= 0.05) { m.lab.classList.remove('on'); m.lab.style.opacity = 0; m.ln.style.opacity = 0; } });
    requestAnimationFrame(tick);
  }
  addEventListener('hashchange', () => { const k = clamp(parseInt((location.hash || '#00').slice(1), 10) || 0, 0, N - 1); if (k !== Math.round(scrollY / H)) go(k); });
  // first frame: snap the camera to the chapter the URL points at
  const start = clamp(parseInt((location.hash || '#00').slice(1), 10) || 0, 0, N - 1);
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
  // the browser's own fragment scroll runs after this script and would reset us to the top: jump now and again on load
  if (start) { scrollTo(0, start * H); addEventListener('load', () => scrollTo(0, start * H)); }
  { pack.view = viewFor(start ? 1 : 0); const T = camFor(CH[start]); pack.setTarget(T.x, T.y, T.z, true); }
  requestAnimationFrame(tick);

  window.INFLO = { G, pack, CH, routes, morph, flower };
})();
