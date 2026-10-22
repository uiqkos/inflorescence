/* main.js — wiring. Everything on this page is drawn from data/graph.js, this repository's own index. */
(function () {
  'use strict';
  const raw = window.INFLO_GRAPH;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ---- graph model
  const byId = new Map(); raw.nodes.forEach(n => { n.kids = []; byId.set(n.id, n); });
  const root = byId.get('dir:.');
  raw.nodes.forEach(n => { if (n === root) return; const p = byId.get(n.pa) || root; n.par = p; p.kids.push(n); });
  const G = { nodes: raw.nodes, edges: raw.edges, byId: Object.fromEntries(byId), totals: raw.totals, edgeTotals: raw.edge_totals };

  // ---- 00 hero
  const hero = new window.Dirt(document.getElementById('hero-canvas'), 'img/gloriosa.jpg',
    { ascii: .6, asciiCell: 11, dots: .75, dotCell: 3.4, scan: .4, grain: .07, aber: .003, zoom: 1.0, fit: 1, shift: [0.2, 0], seed: 3.7, reduced });
  hero.fitFor = () => { const wide = innerWidth / innerHeight > 1.15; hero.o.fit = wide ? 1 : 0; hero.o.shift = wide ? [0.19, -0.01] : [0, 0]; hero.o.zoom = wide ? 1.0 : 1.02; placeBoxes(); };
  // specimen annotations, in image coordinates (u right, v down)
  const BOXES = [
    { u0: .16, v0: .04, u1: .90, v1: .60, label: 'class', side: 'tl' },
    { u0: .15, v0: .62, u1: .24, v1: .70, label: 'method', side: 'bl' },
    { u0: .45, v0: .86, u1: .53, v1: 1.0, label: 'CONTAINS', side: 'r' },
  ];
  const boxLayer = document.getElementById('boxes');
  const boxEls = BOXES.map(b => { const d = document.createElement('div'); d.className = 'box ' + b.side; d.innerHTML = `<span>${b.label}</span>`; boxLayer.appendChild(d); return d; });
  function placeBoxes(){
    if (!hero.ready) { setTimeout(placeBoxes, 120); return; }
    BOXES.forEach((b, i) => { const [x0, y0] = hero.map(b.u0, b.v0), [x1, y1] = hero.map(b.u1, b.v1);
      const el = boxEls[i]; el.style.left = x0 + 'px'; el.style.top = y0 + 'px'; el.style.width = (x1 - x0) + 'px'; el.style.height = (y1 - y0) + 'px'; });
  }
  hero.fitFor(); addEventListener('resize', hero.fitFor);
  setInterval(() => { if (reduced) return; boxLayer.classList.add('jit'); setTimeout(() => boxLayer.classList.remove('jit'), 60 + Math.random() * 90); }, 2400 + Math.random() * 2600);
  const h1 = document.getElementById('h1');
  const typelab = new window.TypeLab(hero, h1, document.getElementById('s0'));
  setInterval(() => { if (reduced) return; h1.classList.add('jit'); setTimeout(() => h1.classList.remove('jit'), 90 + Math.random() * 120); }, 3100 + Math.random() * 2000);

  // ---- 01 parse
  const gbId = 'src/inflorescence/code_indexer/graph_builder.py';
  const morph = new window.Morph(document.getElementById('morph-stage'), G, window.INFLO_SRC, { moduleId: gbId });
  document.getElementById('fig1-meta').innerHTML =
    `${gbId}<br>1 module · ${morph.imports.length} IMPORTS edges into ${morph.groups.length} files · ${morph.defs.length} definitions shown of ${G.byId[gbId].kids.length} · ${morph.exts.length} external`;

  // ---- 02 graph
  const stage = document.getElementById('radial-stage');
  const radial = new window.Radial(stage, G, { onSelect: n => showSummary(n, true) });
  const srcNodes = G.nodes.filter(n => n.t !== 'directory');
  document.getElementById('fig2-meta').innerHTML =
    `this repository · src/ + root documents · ${srcNodes.length} nodes · ${radial.edges.length} edges<br>` +
    ['module', 'class', 'function', 'method', 'document'].map(t => `${srcNodes.filter(n => n.t === t).length} ${t}s`).join(' · ') +
    `<br>whole index incl. tests: ${Object.values(raw.totals).reduce((a, b) => a + b, 0)} nodes · ${raw.chunks} chunks`;
  const legend = document.getElementById('legend');
  for (const k of ['C', 'I', 'N']) {
    const el = document.createElement('span'); el.style.color = { C: '#ff3aa8', I: '#8b7bff', N: '#e6ff3d' }[k];
    el.innerHTML = `<i></i>${{ C: 'CALLS', I: 'IMPORTS', N: 'INHERITS' }[k]} ${radial.counts[k] || 0}`;
    el.addEventListener('click', () => { const on = el.classList.toggle('off'); radial.setShow(k, !on); if (k === 'N') radial.setShow('M', !on); });
    legend.appendChild(el);
  }
  const lg = document.createElement('span'); lg.style.cursor = 'default'; lg.style.opacity = '.55';
  lg.textContent = `■ module  ◇ class  · function / method  □ document`; legend.appendChild(lg);

  // questions: three real search_semantic answers, recorded from the server, plus one structural query computed here
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
  // structural: the most-called method in the graph, and everyone who calls it
  const most = srcNodes.filter(n => (n.t === 'method' || n.t === 'function') && !/__\w+__$/.test(n.n)).sort((a, b) => b.in.length - a.in.length)[0];
  if (most) QUERIES.push({ q: `who calls ${most.n}?`, tool: `get_related_entities · CALLS · incoming`, hits: most.in.filter(E => E.k === 'C').slice(0, 14).map(E => ({ id: E.s.id })) });
  const chips = document.getElementById('chips'); let activeChip = null;
  QUERIES.forEach(Q => {
    const el = document.createElement('div'); el.className = 'chip';
    el.innerHTML = `${Q.q}<small>${Q.tool} · ${Q.hits.length} hits</small>`;
    el.addEventListener('click', () => {
      if (activeChip === el) { el.classList.remove('on'); activeChip = null; radial.setHits(null); return; }
      if (activeChip) activeChip.classList.remove('on');
      activeChip = el; el.classList.add('on'); radial.setHits(Q.hits, `${Q.tool} · ${Q.q}`);
      stage.scrollIntoView({ block: 'center', behavior: reduced ? 'auto' : 'smooth' });
    });
    chips.appendChild(el);
  });
  // clearing hits from the canvas also clears the chip
  const _setHits = radial.setHits.bind(radial);
  radial.setHits = (h, l) => { _setHits(h, l); if (!h && activeChip) { activeChip.classList.remove('on'); activeChip = null; } };

  // ---- 03 summary flower
  const flower = new window.WordFlower(document.getElementById('flower'), 'img/lily.jpg');
  const wanted = [
    'src/inflorescence/code_indexer/graph_builder.py',
    'src/inflorescence/docker_env.py::ensure_memgraph',
    'src/inflorescence/cost.py::estimate_index_cost',
    'src/inflorescence/code_indexer/graph_builder.py::GraphBuilder._diff_checksums',
    'src/inflorescence/watcher.py::_DebouncedHandler._handle',
    'src/inflorescence/tools/search.py',
  ];
  const list = document.getElementById('entities'); const items = new Map();
  function showSummary(n, fromGraph){
    flower.setText(n.s || '(no summary yet)', { name: n.id });
    for (const [id, li] of items) li.classList.toggle('on', id === n.id);
    if (fromGraph && !items.has(n.id)) {
      // the node picked on the ring joins the list at the top
      const li = makeItem(n); list.prepend(li); li.classList.add('on');
      if (items.size > 7) { const last = [...items.keys()].pop(); items.get(last).remove(); items.delete(last); }
    }
  }
  function makeItem(n){
    const li = document.createElement('li'); li.innerHTML = `${n.n}<small>${n.t} · ${n.p}</small>`;
    li.addEventListener('click', () => showSummary(n, false)); items.set(n.id, li); return li;
  }
  wanted.map(id => G.byId[id]).filter(Boolean).forEach(n => list.appendChild(makeItem(n)));
  const first = G.byId[wanted[0]] || srcNodes[0];
  if (first) showSummary(first, false);

  // ---- 04 install: cyanotype foxglove
  new window.Dirt(document.getElementById('install-canvas'), 'img/foxglove.jpg',
    { ascii: .5, asciiCell: 10, dots: .9, dotCell: 3.6, scan: .25, grain: .05, aber: .001, solar: 0, zoom: 1.02, shift: [-0.1, 0], seed: 8.1, burstRate: .5,
      duo: ['#06082a', '#dfe6ff'], reduced });

  // ---- 05 footer: ink threshold on paper
  new window.Dirt(document.getElementById('end-canvas'), 'img/gloriosa-3.jpg',
    { ascii: .7, asciiCell: 9, dots: 0, scan: 0, grain: .03, aber: 0, solar: 0, zoom: 1.0, fit: 0, shift: [0, 0.04], seed: 1.3, burstRate: .4, mouseAmt: 1,
      thresh: ['#c6c8c3', '#111111'], reduced });
  document.getElementById('tools').innerHTML = ['index_directory', 'list_entities', 'get_entity_structure', 'get_entity_context', 'get_related_entities',
    'search_entities', 'search_code', 'search_semantic', 'search_hybrid', 'search_text', 'cypher_query', 'graph_schema', 'send_feedback']
    .map(t => `<span>${t}</span>`).join('') + `<em>13 MCP tools · stdio</em>`;

  window.INFLO = { G, radial, morph, flower, hero, typelab };

  // ---- pager
  const pager = document.getElementById('pager');
  const secs = [...document.querySelectorAll('section, footer')];
  new IntersectionObserver(en => { en.forEach(e => { if (e.isIntersecting) pager.textContent = e.target.id.replace('s', '0') + ' / 05'; }); }, { threshold: 0.5 }).observe(secs[0]);
  secs.forEach(s => new IntersectionObserver(en => { if (en[0].isIntersecting) pager.textContent = s.id.replace('s', '0') + ' / 05'; }, { threshold: 0.4 }).observe(s));
})();
