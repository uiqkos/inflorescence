/* pack.js — the index as nested discs, as a tool.
 *
 * Derived from site/js/pack.js (the landing page's "specimen" view): florets — functions and
 * methods — sit in golden-angle spirals inside the file that defines them, files inside their
 * directories, so CONTAINS is drawn as nesting the way a capitulum holds its florets, and every
 * CALLS / IMPORTS / INHERITS edge is bundled through the centres of the discs it crosses.
 *
 * What is different here, because this one is driven by hand rather than by a scroll position:
 *   · data comes from /api/projects/{p}/specimen (index-addressed, see DashboardService.graph_specimen)
 *   · the camera is wheel-zoom-toward-the-cursor plus drag-pan, not a scripted flight
 *   · fit / frame / focus-by-id, double-click to frame a disc, Esc to unpin
 *   · a route tool (Dijkstra over the real edges) any two nodes can be fed to
 *   · search hits, a live-index pop for nodes that land while you watch, a scrubable replay
 *   · the layout is bounded so a directory with a thousand children still lays out in milliseconds
 *   · the canvas is repainted only when the picture actually changes — see needsDraw()
 *
 * Nothing here fetches or writes: it is given a payload and draws it.
 *
 * INVARIANT, if you add state: a canvas has no dirty regions, so this one redraws on a
 * signature of everything render() reads (needsDraw). Anything new that render() looks at
 * must join that signature — or bump one of dataV / hitsV / routeV, or set lastSig = null —
 * or the screen will keep showing the previous frame. Getting it wrong costs a stale picture;
 * over-signing only costs a redraw, so when in doubt, sign it.
 */
"use strict";

(function () {
  const TAU = Math.PI * 2, GOLD = Math.PI * (3 - Math.sqrt(5));
  const COL = { C: "#ff3aa8", I: "#8b7bff", N: "#e6ff3d", M: "#e6ff3d" };
  const OUT_C = "#4d86ff"; // an outgoing call is drawn blue, an incoming one pink
  const NAME = { C: "CALLS", I: "IMPORTS", N: "INHERITS", M: "IMPLEMENTS", P: "CONTAINS" };
  const FONT = '"IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace';
  const ORD = { directory: 0, document: 1, module: 2, class: 3, interface: 3, enum: 3, struct: 3, trait: 3, function: 4, method: 5 };
  const DARK = { bg: [7, 7, 7], ink: [235, 232, 225] }, PAPER = { bg: [198, 200, 195], ink: [17, 17, 17] };
  const isCls = (t) => t === "class" || t === "interface" || t === "struct" || t === "enum" || t === "trait";
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (u) => (u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2);
  const rnd = (i) => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const mix3 = (a, b, t) => [Math.round(lerp(a[0], b[0], t)), Math.round(lerp(a[1], b[1], t)), Math.round(lerp(a[2], b[2], t))];

  // The path a node id already encodes — the payload omits `p` whenever this reproduces it.
  // Mirrors DashboardService.derived_path; keep the two in step.
  const derivedPath = (id) => (id.startsWith("dir:") ? id.slice(4) : id.split("::")[0]);

  const GFX_DEFAULTS = {
    labels: true,      // greedy non-overlapping names, with level-of-detail
    edgeAlpha: 1,      // multiplier on the zoom-dependent alpha; 0 hides the weave
    pulse: true,       // idle: a random node breathes its links — the index is live, not a picture
    glitch: true,      // rare row-tearing, never on the cards you have to read
    grain: true,       // scanlines over the canvas (CSS class on the stage)
    bloom: true,       // open the discs out of their parents on first load
    paper: false,      // herbarium sheet: ink on grey paper
  };

  // ---- model: the specimen payload becomes a tree (CONTAINS) plus typed edges -------------
  function model(raw) {
    const nodes = raw.nodes.map((r) => ({
      id: r.i, t: r.t || "code", n: r.n || r.i,
      p: r.p !== undefined ? r.p : derivedPath(r.i),
      l: r.l || null, s: r.s || "", g: r.g || "",
      kids: [], par: null, in: [], out: [],
    }));
    const byId = new Map();
    for (const n of nodes) byId.set(n.id, n);
    let root = byId.get(raw.root) || nodes.find((n, i) => raw.nodes[i].pa === -1) || nodes[0];
    if (!root) {
      root = { id: "dir:.", t: "directory", n: raw.display_name || "root", p: ".", l: null, s: "", g: "", kids: [], par: null, in: [], out: [] };
      nodes.push(root);
      byId.set(root.id, root);
    }
    raw.nodes.forEach((r, i) => {
      const n = nodes[i];
      if (n === root) return;
      const parent = r.pa >= 0 ? nodes[r.pa] : root;
      n.par = parent && parent !== n ? parent : root;
      n.par.kids.push(n);
    });
    const edges = [];
    for (const e of raw.edges) {
      const s = nodes[e[0]], d = nodes[e[1]];
      if (!s || !d || s === d) continue;
      const E = { s, d, k: e[2], res: e[3] || "" };
      s.out.push(E); d.in.push(E); edges.push(E);
    }
    return {
      project: raw.display_name || raw.project || "", nodes, edges, byId, root,
      totals: raw.totals || {}, edgeTotals: raw.edge_totals || {},
      chunks: raw.chunks || 0, external: raw.external || [],
      scope: raw.scope || null, truncated: !!raw.truncated, notes: raw.notes || [],
    };
  }

  function Pack(stage, G, o) {
    this.stage = stage;
    this.o = Object.assign({ onSelect: null, onHover: null, onCamera: null, reduced: false }, o || {});
    this.gfx = Object.assign({}, GFX_DEFAULTS, this.o.gfx || {});
    const c = this.canvas = document.createElement("canvas");
    stage.appendChild(c);
    this.ctx = c.getContext("2d");
    this.note = document.createElement("div");
    this.note.className = "note";
    this.note.hidden = true;
    stage.appendChild(this.note);
    this.cam = { x: 0, y: 0, z: 1 };
    this.target = { x: 0, y: 0, z: 1 };
    this.view = { x0: 0, y0: 0, w: 1, h: 1 };
    this.paper = this.gfx.paper ? 1 : 0;
    this.hover = null; this.pinned = null; this.hits = null; this.hitLabel = "";
    this.route = null; this.routeT = 0; this.mark = null; this.pickMode = null;
    this.show = { C: true, I: true, N: true, M: true };
    this.pulse = null; this.glitch = null; this.nextGlitch = performance.now() + 4000;
    this.mouse = null; this.drag = null; this.dragged = false; this.last = 0;
    this.bloom = { t0: performance.now(), span: 0, open: 0.6, playing: true, pausedAt: 0, replay: false };
    // Bumped by whatever render() reads but cannot be compared cheaply (the model, the hit
    // list, the route). They are part of the redraw signature — see needsDraw().
    this.dataV = 0; this.hitsV = 0; this.routeV = 0; this.lastSig = null;
    this.W = 1; this.H = 1; this.dpr = 1;
    this.resize();
    this.setData(G, { snap: true });
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(stage);
    c.addEventListener("mousemove", (e) => this.onMove(e));
    c.addEventListener("mouseleave", () => { this.mouse = null; if (this.hover) { this.hover = null; this.updateNote(); } });
    c.addEventListener("mousedown", (e) => this.onDown(e));
    addEventListener("mouseup", () => { if (this.drag) { this.drag = null; this.canvas.style.cursor = ""; } });
    c.addEventListener("click", (e) => this.onClick(e));
    c.addEventListener("dblclick", (e) => this.onDoubleClick(e));
    c.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    this._loop = (now) => { this.frame(now); this._raf = requestAnimationFrame(this._loop); };
    this._raf = requestAnimationFrame(this._loop);
  }
  Pack.model = model;
  Pack.COL = COL;
  Pack.NAME = NAME;
  Pack.OUT_C = OUT_C;
  Pack.GFX_DEFAULTS = GFX_DEFAULTS;

  Pack.prototype.destroy = function () {
    cancelAnimationFrame(this._raf);
    if (this._ro) this._ro.disconnect();
    this.canvas.remove();
    this.note.remove();
  };

  // ---- data ------------------------------------------------------------------------------
  // A live index keeps landing new entities, so the whole model is rebuilt rather than patched:
  // nesting depends on every sibling, and a rebuild of 17k nodes is milliseconds. The camera,
  // the pinned node and the route survive it by id; nodes that were not there before pop.
  Pack.prototype.setData = function (G, opt) {
    opt = opt || {};
    const before = this.G ? new Set(this.G.nodes.map((n) => n.id)) : null;
    const pinnedId = this.pinned ? this.pinned.id : null;
    const markId = this.mark ? this.mark.id : null;
    const routeIds = this.route ? [this.route.a.id, this.route.b.id] : null;
    const hits = this.hits ? this.hits.map((h) => ({ id: h.node.id, weak: h.weak, score: h.score })) : null;
    this.G = G;
    this.root = G.root;
    this.hover = null;
    const t0 = performance.now();
    this.build();
    this.layout();
    this.layoutMs = Math.round(performance.now() - t0);
    this.pinned = pinnedId ? G.byId.get(pinnedId) || null : null;
    this.mark = markId ? G.byId.get(markId) || null : null;
    if (hits) this.setHits(hits, this.hitLabel, false);
    if (routeIds) {
      const a = G.byId.get(routeIds[0]), b = G.byId.get(routeIds[1]);
      this.route = a && b ? this.routeBetween(a, b) : null;
      if (!this.route) this.routeT = 0;
    }
    if (before && !opt.snap) {
      const now = performance.now();
      let fresh = 0;
      for (const n of G.nodes) if (!before.has(n.id)) { n.popAt = now + Math.min(600, fresh++ * 12); }
      this.freshCount = fresh;
      this.bloom.span = 0;
    } else {
      this.bloom = { t0: performance.now(), span: this.o.reduced || !this.gfx.bloom ? 0 : 2.2, open: 0.7, playing: true, pausedAt: 0, replay: false };
    }
    if (opt.snap) this.fit(0.9, true);
    this.dataV++;
    this.updateNote();
  };

  Pack.prototype.build = function () {
    const G = this.G, root = this.root;
    (function sort(n) {
      n.kids.sort((a, b) => (ORD[a.t] - ORD[b.t]) || String(a.n).localeCompare(String(b.n)));
      n.kids.forEach(sort);
    })(root);
    const order = this.order = [];
    (function dep(n, d, anc) { n.depth = d; n.anc = [n, ...anc]; order.push(n); n.kids.forEach((k) => dep(k, d + 1, n.anc)); })(root, 0, []);
    this.edges = G.edges;
    this.counts = {};
    for (const E of this.edges) this.counts[E.k] = (this.counts[E.k] || 0) + 1;
    for (const n of order) n.thin = n.t === "directory" && n.kids.length === 1 && n.kids[0].t === "directory";
    this.leaves = order.filter((n) => !n.kids.length && n.t !== "directory");
    this.containers = order.filter((n) => n.kids.length || n.t === "directory");
  };

  // ---- layout: phyllotaxis inside every container, then a bounded relaxation ---------------
  // The site relaxes every container with an O(n²) double loop for 240 iterations, which is fine
  // for 800 nodes and hopeless for a directory holding 1170 files. Here the pair search runs over
  // a uniform grid above a threshold and the iteration count falls with the child count, so the
  // whole tree lays out in well under a second even at 17k nodes.
  Pack.prototype.layout = function () {
    const GRID_FROM = 90;
    function iterations(n) { return n <= 64 ? 220 : n <= 160 ? 140 : n <= 400 ? 70 : n <= 1200 ? 40 : 24; }

    function relaxNaive(items, gap, iters) {
      const N = items.length;
      for (let it = 0; it < iters; it++) {
        let moved = false;
        for (let i = 0; i < N; i++) {
          const a = items[i];
          for (let j = i + 1; j < N; j++) {
            const b = items[j];
            let dx = b.lx - a.lx, dy = b.ly - a.ly, d = Math.hypot(dx, dy);
            const min = a.r + b.r + gap;
            if (d >= min) continue;
            moved = true;
            if (d < 1e-6) { dx = Math.cos(i * 7 + j); dy = Math.sin(i * 7 + j); d = 1; }
            const push = ((min - d) / d) * 0.52, wa = (b.r * b.r) / (a.r * a.r + b.r * b.r), wb = 1 - wa;
            a.lx -= dx * push * wa; a.ly -= dy * push * wa; b.lx += dx * push * wb; b.ly += dy * push * wb;
          }
        }
        if (it < iters * 0.7) for (const k of items) { k.lx *= 0.988; k.ly *= 0.988; }
        else if (!moved) break;
      }
    }

    // Same relaxation, but each node only meets the ones in its own cell and the eight around it.
    function relaxGrid(items, gap, iters) {
      let maxR = 0;
      for (const k of items) maxR = Math.max(maxR, k.r);
      const cell = 2 * maxR + gap;
      const buckets = new Map();
      const key = (cx, cy) => cx * 73856093 ^ cy * 19349663;
      for (let it = 0; it < iters; it++) {
        buckets.clear();
        for (const k of items) {
          const cx = Math.floor(k.lx / cell), cy = Math.floor(k.ly / cell);
          const kk = key(cx, cy);
          let b = buckets.get(kk);
          if (!b) buckets.set(kk, (b = []));
          b.push(k);
          k._cx = cx; k._cy = cy;
        }
        let moved = false;
        for (const a of items) {
          for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
            const b = buckets.get(key(a._cx + dx, a._cy + dy));
            if (!b) continue;
            for (const o of b) {
              if (o === a || o.idx <= a.idx) continue;
              let vx = o.lx - a.lx, vy = o.ly - a.ly, d = Math.hypot(vx, vy);
              const min = a.r + o.r + gap;
              if (d >= min) continue;
              moved = true;
              if (d < 1e-6) { vx = Math.cos(a.idx * 7 + o.idx); vy = Math.sin(a.idx * 7 + o.idx); d = 1; }
              const push = ((min - d) / d) * 0.5, wa = (o.r * o.r) / (a.r * a.r + o.r * o.r), wb = 1 - wa;
              a.lx -= vx * push * wa; a.ly -= vy * push * wa; o.lx += vx * push * wb; o.ly += vy * push * wb;
            }
          }
        }
        if (it < iters * 0.7) for (const k of items) { k.lx *= 0.99; k.ly *= 0.99; }
        else if (!moved) break;
      }
    }

    function pack(n) {
      n.kids.forEach(pack);
      if (!n.kids.length) {
        n.r = n.t === "document" ? 1.4 : n.t === "module" ? 1.5 : isCls(n.t) ? 1.25 : 1;
        n.lx = 0; n.ly = 0;
        return;
      }
      const gap = 0.55;
      // smallest first: florets gather in the centre, the discs of classes and files form a
      // corolla around them — unless one child holds most of the area (src/ inside the root),
      // which then sits in the centre with the rest around it
      const items = [...n.kids].sort((a, b) => (a.r - b.r) || (ORD[a.t] - ORD[b.t]) || String(a.n).localeCompare(String(b.n)));
      const areas = items.map((k) => k.r * k.r);
      let tot = 0, big = 0;
      for (const a of areas) { tot += a; if (a > big) big = a; }
      if (big > 0.45 * tot) items.reverse();
      let area = 0;
      items.forEach((k, i) => {
        const a = Math.PI * Math.pow(k.r + gap / 2, 2), rho = Math.sqrt((area + a * 0.5) / Math.PI);
        area += a;
        const th = i * GOLD + n.depth * 0.9;
        k.lx = rho * Math.cos(th); k.ly = rho * Math.sin(th); k.idx = i;
      });
      const iters = iterations(items.length);
      if (items.length >= GRID_FROM) relaxGrid(items, gap, iters);
      else relaxNaive(items, gap, iters);
      let cx = 0, cy = 0, m = 0;
      for (const k of items) { const w = k.r * k.r; cx += k.lx * w; cy += k.ly * w; m += w; }
      cx /= m; cy /= m;
      let R = 0;
      for (const k of items) { k.lx -= cx; k.ly -= cy; R = Math.max(R, Math.hypot(k.lx, k.ly) + k.r); }
      n.r = R + (n.thin ? 0.4 : n.t === "directory" ? 2.4 : n.t === "module" ? 1.5 : 0.9);
    }
    pack(this.root);
    (function abs(n, x, y) { n.x = x + (n.lx || 0); n.y = y + (n.ly || 0); n.kids.forEach((k) => abs(k, n.x, n.y)); })(this.root, 0, 0);

    // Opening order, normalised to 0..1 so one number drives both the load bloom and a replay
    // of any length: the root first, then outward and around each spiral.
    let maxDelay = 0;
    for (const n of this.order) {
      n.delay = n === this.root ? 0 : 0.25 + n.depth * 0.17 + (n.idx / Math.max(1, n.par.kids.length)) * 0.45 + rnd(n.idx + n.depth) * 0.08;
      if (n.delay > maxDelay) maxDelay = n.delay;
    }
    for (const n of this.order) n.d0 = maxDelay ? n.delay / maxDelay : 0;
    this.R = this.root.r;
    for (const E of this.edges) { const b = this.bundle(this.pathPoints(E.s, E.d), 0.86); E.path = b.path; E.len = b.len; }
    this.batch = {};
    for (const k of ["C", "I", "N", "M"]) this.batch[k] = new Path2D();
    for (const E of this.edges) this.batch[E.k].addPath(E.path);
  };

  Pack.prototype.pathPoints = function (s, d) {
    const set = new Set(s.anc);
    let l = null;
    for (const n of d.anc) if (set.has(n)) { l = n; break; }
    const up = [], down = [];
    for (const n of s.anc) { if (n === l) break; up.push(n); }
    for (const n of d.anc) { if (n === l) break; down.push(n); }
    const pts = [[s.x, s.y]];
    for (let i = 1; i < up.length; i++) pts.push([up[i].x, up[i].y]);
    if (l && l !== s && l !== d) pts.push([l.x, l.y]);
    for (let i = down.length - 1; i >= 1; i--) pts.push([down[i].x, down[i].y]);
    pts.push([d.x, d.y]);
    return pts;
  };

  // d3.curveBundle: straighten toward the chord, then a B-spline through the points
  Pack.prototype.bundle = function (pts, beta) {
    const n = pts.length - 1, [x0, y0] = pts[0], [xn, yn] = pts[n];
    const q = pts.map((p, i) => [beta * p[0] + (1 - beta) * (x0 + (i / n) * (xn - x0)), beta * p[1] + (1 - beta) * (y0 + (i / n) * (yn - y0))]);
    let len = 0;
    for (let i = 1; i < q.length; i++) len += Math.hypot(q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1]);
    const P = new Path2D();
    let px0, py0, px1, py1, st = 0;
    const pt = (x, y) => {
      switch (st) {
        case 0: st = 1; P.moveTo(x, y); break;
        case 1: st = 2; break;
        case 2: st = 3; P.lineTo((5 * px0 + px1) / 6, (5 * py0 + py1) / 6); // falls through
        default: P.bezierCurveTo((2 * px0 + px1) / 3, (2 * py0 + py1) / 3, (px0 + 2 * px1) / 3, (py0 + 2 * py1) / 3, (px0 + 4 * px1 + x) / 6, (py0 + 4 * py1 + y) / 6);
      }
      px0 = px1; px1 = x; py0 = py1; py1 = y;
    };
    q.forEach((p) => pt(p[0], p[1]));
    if (st === 3) { pt(px1, py1); P.lineTo(px1, py1); } else if (st === 2) P.lineTo(px1, py1);
    return { path: P, len: len * 0.96 };
  };

  // ---- a route between two nodes over the real edges --------------------------------------
  // Dijkstra, exactly as on the walk page: a CALLS/IMPORTS/INHERITS edge costs 1, a step up or
  // down CONTAINS costs 2.4 — so the answer prefers real relations and only walks the tree when
  // there is no other way through.
  Pack.prototype.routeBetween = function (a, b) {
    if (!a || !b || a === b) return null;
    const dist = new Map([[a, 0]]), prev = new Map();
    const open = [a], done = new Set();
    const relax = (x, y, w, k, dir) => {
      const nd = dist.get(x) + w;
      if (nd < (dist.has(y) ? dist.get(y) : Infinity)) {
        dist.set(y, nd); prev.set(y, { x, k, dir });
        if (!done.has(y)) open.push(y);
      }
    };
    while (open.length) {
      let bi = 0;
      for (let i = 1; i < open.length; i++) if (dist.get(open[i]) < dist.get(open[bi])) bi = i;
      const x = open.splice(bi, 1)[0];
      if (done.has(x)) continue;
      done.add(x);
      if (x === b) break;
      for (const E of x.out) relax(x, E.d, 1, E.k, ">");
      for (const E of x.in) relax(x, E.s, 1, E.k, "<");
      if (x.par) relax(x, x.par, 2.4, "P", "<");
      for (const k of x.kids) relax(x, k, 2.4, "P", ">");
    }
    if (!prev.has(b)) return null;
    const steps = [];
    let y = b;
    while (y !== a) { const p = prev.get(y); steps.push({ a: p.x, b: y, k: p.k, dir: p.dir }); y = p.x; }
    steps.reverse();
    let total = 0;
    for (const st of steps) {
      const pts = st.k === "P" ? [[st.a.x, st.a.y], [st.b.x, st.b.y]] : this.pathPoints(st.a, st.b);
      const bd = this.bundle(pts, 0.86);
      st.path = bd.path; st.len = Math.max(bd.len, 0.5); total += st.len;
    }
    return { a, b, steps, total, cost: dist.get(b) };
  };

  Pack.prototype.setRoute = function (route, animate) {
    this.route = route;
    this.routeV++;
    this.routeT = route && animate !== false ? 0 : 1;
    if (route) {
      this.pinned = null;
      this.frameBoth(route.a, route.b);
      this.updateNote();
    }
  };
  Pack.prototype.clearRoute = function () { this.route = null; this.routeT = 0; this.routeV++; };

  // ---- camera -----------------------------------------------------------------------------
  Pack.prototype.resize = function () {
    const r = this.stage.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const dpr = this.dpr = Math.min(2, devicePixelRatio || 1);
    // Keep the framing, not the pixel zoom: a narrower window should show the same thing
    // smaller, and a canvas that was 1x1 while its section was hidden must not leave the
    // camera fitted to a 1x1 box.
    const was = Math.min(this.W, this.H);
    const now = Math.min(r.width, r.height);
    if (was > 1 && now > 1 && this.R) {
      const k = now / was;
      this.target.z *= k; this.cam.z *= k;
    }
    this.W = r.width; this.H = r.height;
    this.view = { x0: 0, y0: 0, w: r.width, h: r.height };
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.canvas.style.width = r.width + "px";
    this.canvas.style.height = r.height + "px";
    this.lastSig = null;  // setting .width cleared the bitmap: the next frame must repaint
  };
  Pack.prototype.fitZoom = function (n, frac) { return (frac || 0.9) * Math.min(this.W, this.H) / (2 * Math.max(n.r, 0.001)); };
  Pack.prototype.zLimits = function () { return [this.fitZoom(this.root, 0.35), 64]; };
  Pack.prototype.toScreen = function (wx, wy) { const c = this.cam; return [this.W / 2 + (wx - c.x) * c.z, this.H / 2 + (wy - c.y) * c.z]; };
  Pack.prototype.toWorld = function (sx, sy, cam) { const c = cam || this.cam; return [c.x + (sx - this.W / 2) / c.z, c.y + (sy - this.H / 2) / c.z]; };
  Pack.prototype.setTarget = function (x, y, z, snap) {
    const [zmin, zmax] = this.zLimits();
    this.target = { x, y, z: clamp(z, zmin, zmax) };
    if (snap) this.cam = { ...this.target };
  };
  Pack.prototype.fit = function (frac, snap) { this.setTarget(this.root.x, this.root.y, this.fitZoom(this.root, frac || 0.9), snap); };
  Pack.prototype.frameNode = function (n, frac, snap) {
    if (!n) return;
    const target = n.kids.length ? n : n.par || n;
    this.setTarget(n.x, n.y, this.fitZoom(target, frac || 0.55), snap);
  };
  // Frame two nodes at once — what a route wants: both ends on screen with room for the curve.
  Pack.prototype.frameBoth = function (a, b) {
    const cx = (a.x + b.x) / 2, cy = (a.y + b.y) / 2;
    const span = Math.max(Math.hypot(a.x - b.x, a.y - b.y), 8);
    this.setTarget(cx, cy, (0.55 * Math.min(this.W, this.H)) / span);
  };
  Pack.prototype.focus = function (id, opt) {
    const n = this.G.byId.get(id);
    if (!n) return null;
    opt = opt || {};
    if (opt.pin !== false) this.pinned = n;
    if (opt.frame !== false) this.frameNode(n, opt.frac || (n.kids.length ? 0.55 : 0.18));
    this.updateNote();
    return n;
  };

  Pack.prototype.onWheel = function (e) {
    e.preventDefault();
    const r = this.canvas.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    // Trackpads report ctrlKey on pinch; both gestures mean the same thing here.
    const step = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    const factor = Math.exp(-clamp(step, -240, 240) * 0.0015);
    const [zmin, zmax] = this.zLimits();
    const z = clamp(this.target.z * factor, zmin, zmax);
    const [wx, wy] = this.toWorld(sx, sy, this.target);
    this.target = { z, x: wx - (sx - this.W / 2) / z, y: wy - (sy - this.H / 2) / z };
  };
  Pack.prototype.onDown = function (e) {
    if (e.button !== 0 && e.button !== 1) return;
    const r = this.canvas.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    const [wx, wy] = this.toWorld(sx, sy);
    this.drag = { sx: e.clientX, sy: e.clientY, wx, wy };
    this.dragged = false;
  };
  Pack.prototype.onMove = function (e) {
    const r = this.canvas.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    if (this.drag) {
      if (Math.hypot(e.clientX - this.drag.sx, e.clientY - this.drag.sy) > 4) this.dragged = true;
      if (this.dragged) {
        // 1:1 with the pointer: the world point grabbed stays under it, so both the live
        // camera and the target move — smoothing here would feel like drag on ice.
        const z = this.cam.z;
        this.cam.x = this.drag.wx - (sx - this.W / 2) / z;
        this.cam.y = this.drag.wy - (sy - this.H / 2) / z;
        this.target.x = this.cam.x; this.target.y = this.cam.y; this.target.z = z;
        this.canvas.style.cursor = "grabbing";
      }
      return;
    }
    this.mouse = [sx, sy];
    const n = this.pick(sx, sy);
    this.canvas.style.cursor = this.pickMode ? "crosshair" : n ? "pointer" : "grab";
    if (n !== this.hover) { this.hover = n; this.updateNote(); if (this.o.onHover) this.o.onHover(n); }
  };
  Pack.prototype.onClick = function (e) {
    if (this.dragged) { this.dragged = false; return; }
    const r = this.canvas.getBoundingClientRect();
    const n = this.pick(e.clientX - r.left, e.clientY - r.top);
    if (this.pickMode) {
      const cb = this.pickMode;
      this.pickMode = null;
      this.canvas.style.cursor = "";
      cb(n);
      return;
    }
    if (n) {
      this.pinned = this.pinned === n ? null : n;
      if (this.pinned && this.o.onSelect) this.o.onSelect(n);
    } else {
      this.pinned = null;
      if (this.hits) this.setHits(null);
      if (this.o.onSelect) this.o.onSelect(null);
    }
    this.updateNote();
  };
  Pack.prototype.onDoubleClick = function (e) {
    const r = this.canvas.getBoundingClientRect();
    const sx = e.clientX - r.left, sy = e.clientY - r.top;
    const n = this.pick(sx, sy) || this.containerAt(sx, sy);
    if (n) this.frameNode(n, n.kids.length ? 0.72 : 0.2);
    else this.fit(0.9);
  };
  Pack.prototype.unpin = function () {
    let did = false;
    if (this.pickMode) { this.pickMode = null; this.canvas.style.cursor = ""; did = true; }
    else if (this.route) { this.clearRoute(); did = true; }
    else if (this.pinned) { this.pinned = null; did = true; }
    else if (this.hits) { this.setHits(null); did = true; }
    this.updateNote();
    return did;
  };
  Pack.prototype.pickNext = function (cb) { this.pickMode = cb; this.canvas.style.cursor = "crosshair"; };

  // ---- drawing helpers ---------------------------------------------------------------------
  // The diamond closes with an explicit lineTo rather than closePath(): Skia's closePath is
  // O(subpaths) per call, so closing 1.3k class marks inside one 14k-subpath figure cost 5.5ms
  // a frame — a third of the whole render — and grew quadratically with the repository. The
  // returning segment draws identically under both fill() and stroke().
  Pack.prototype.markShape = function (ctx, n, x, y, s) {
    if (n.t === "module" || n.t === "document") ctx.rect(x - s * 1.1, y - s * 1.1, s * 2.2, s * 2.2);
    else if (isCls(n.t)) { ctx.moveTo(x, y - s * 1.5); ctx.lineTo(x + s * 1.5, y); ctx.lineTo(x, y + s * 1.5); ctx.lineTo(x - s * 1.5, y); ctx.lineTo(x, y - s * 1.5); }
    else { ctx.moveTo(x + s, y); ctx.arc(x, y, s, 0, TAU); }
  };
  // Once the discs have finished opening, the leaf marks are the same figure every frame: their
  // world coordinates are fixed by the layout and their size only follows the zoom. Assembling
  // 14.6k subpaths per frame cost ~4ms; building them once per (layout, size) and re-filling
  // the cached Path2D costs ~0.02ms. Panning never touches it — the path is in world space and
  // the camera is a transform — so only a zoom step or a reindex pays to rebuild.
  Pack.prototype.leafPaths = function (ms, animating) {
    if (animating) return null;
    const key = this.dataV + ":" + ms;
    if (this.lpKey !== key) {
      const solid = new Path2D(), hollow = new Path2D();
      for (const n of this.leaves) {
        if (n.t === "document" || isCls(n.t)) this.markShape(hollow, n, n.x, n.y, ms * 1.3);
        else this.markShape(solid, n, n.x, n.y, ms);
      }
      this.lp = { solid, hollow };
      this.lpKey = key;
    }
    return this.lp;
  };

  Pack.prototype.dirName = function (n) {
    if (n === this.root) return (this.G.project || "root") + "/";
    const base = (n.p || n.n).split("/").pop() + "/";
    return n.par && n.par.thin && n.par !== this.root ? n.par.p.split("/").pop() + "/" + base : base;
  };
  Pack.prototype.shortName = function (n) {
    if (n.t === "directory") return this.dirName(n);
    if (n.t === "module" || n.t === "document") return (n.p || n.n).split("/").pop();
    return n.n;
  };
  Pack.prototype.arcText = function (ctx, text, cx, cy, r) {
    ctx.save(); ctx.translate(cx, cy);
    const cw = ctx.measureText("M").width, total = (text.length * cw) / r;
    let ang = -Math.PI / 2 - total / 2;
    for (const ch of text) {
      ctx.save(); ctx.rotate(ang); ctx.translate(r, 0); ctx.rotate(Math.PI / 2);
      ctx.textAlign = "center"; ctx.fillText(ch, 0, 0); ctx.restore();
      ang += cw / r;
    }
    ctx.restore();
  };
  Pack.prototype.ringAt = function (ctx, n, col, w) {
    const z = this.cam.z;
    ctx.strokeStyle = col; ctx.lineWidth = (w || 1) / z;
    ctx.beginPath(); ctx.arc(n.ax, n.ay, n.kids.length ? n.ar : 6.5 / z, 0, TAU); ctx.stroke();
  };
  Pack.prototype.crosshair = function (ctx, n, col) {
    const z = this.cam.z, s = (n.kids.length ? Math.max(9, n.ar * z + 6) : 9) / z;
    ctx.strokeStyle = col; ctx.lineWidth = 1 / z;
    ctx.strokeRect(n.ax - s, n.ay - s, 2 * s, 2 * s);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      ctx.beginPath();
      ctx.moveTo(n.ax + dx * s, n.ay + dy * s);
      ctx.lineTo(n.ax + dx * (s + 6 / z), n.ay + dy * (s + 6 / z));
      ctx.stroke();
    }
  };
  // Who calls this, and what it calls — the question the hover exists to answer.
  Pack.prototype.drawLinks = function (ctx, n, alpha, labels) {
    const z = this.cam.z;
    ctx.lineWidth = 1.1 / z; ctx.lineCap = "round";
    const draw = (E, col, other) => {
      if (!this.show[E.k]) return;
      ctx.globalAlpha = alpha * 0.9; ctx.strokeStyle = col; ctx.stroke(E.path);
      ctx.globalAlpha = alpha; ctx.fillStyle = col;
      ctx.beginPath(); this.markShape(ctx, other, other.ax, other.ay, this.ms * 1.7); ctx.fill();
      if (labels && labels.length < 160) labels.push({ n: other, col });
    };
    for (const E of n.in) draw(E, E.k === "C" ? COL.C : E.k === "I" ? COL.I : COL.N, E.s);
    for (const E of n.out) draw(E, E.k === "C" ? OUT_C : E.k === "I" ? COL.I : COL.N, E.d);
    ctx.globalAlpha = 1;
  };

  // ---- frame -------------------------------------------------------------------------------
  // ---- when to redraw at all ---------------------------------------------------------------
  // A canvas has no dirty regions: every frame repaints the whole specimen, and at 17k nodes
  // that is ~6ms of path building. Doing it 80 times a second while the picture is identical
  // was burning most of a core (and warming the laptop) to show a still image.
  //
  // Returns null while something is genuinely moving — redraw unconditionally — and otherwise
  // a signature of everything render() reads. Equal signature, equal picture, so skip the
  // frame. Signing the *inputs* rather than trusting call sites to flag themselves dirty means
  // a new mutator cannot silently leave a stale picture on screen; it can only cost a redraw.
  Pack.prototype.needsDraw = function (now) {
    if (this.bloomProgress(now) < 1) return null;                 // discs still opening
    if (this.freshCount) return null;                             // pop rings on new entities
    if (this.glitch) return null;                                 // row tearing
    if (this.route && this.routeT < 1) return null;               // route still revealing
    const c = this.cam, g = this.gfx;
    const sig = [
      c.x, c.y, c.z, this.paper, this.W, this.H, this.dpr,
      this.hover ? this.hover.id : "", this.pinned ? this.pinned.id : "", this.mark ? this.mark.id : "",
      this.dataV, this.hitsV, this.routeV, this.routeT,
      this.show.C, this.show.I, this.show.N, this.show.M,
      g.labels, g.edgeAlpha, g.pulse, g.glitch, g.paper,
      this.note.hidden, this.noteList,
    ].join("|");
    if (sig !== this.lastSig) return sig;                         // something real moved
    // Nothing moved but the idle breath is mid-cycle. It is a two-second sine over a still
    // picture, so 30fps carries it and a specimen left open stays cool.
    if (this.pulse && now - this.pulse.t0 < 2200 && now - (this.pulseDrawn || 0) >= 33) {
      this.pulseDrawn = now;
      return null;
    }
    return sig;                                                   // === lastSig: frame() skips
  };

  Pack.prototype.frame = function (now) {
    const dt = Math.min(0.1, (now - (this.last || now)) / 1000);
    this.last = now;
    // Not on screen: the graph section is display:none (the canvas has no layout box) or the
    // tab is in the background. Repainting either is pure heat — showView() calls resize(),
    // which moves the signature and brings the picture straight back.
    if (document.hidden || !this.canvas.clientWidth || !this.canvas.clientHeight) { this.offscreen = true; return; }
    if (this.offscreen) { this.offscreen = false; this.lastSig = null; }
    const k = 1 - Math.exp(-dt * 7), c = this.cam, t = this.target;
    if (!this.drag || !this.dragged) {
      c.x += (t.x - c.x) * k; c.y += (t.y - c.y) * k;
      c.z = Math.exp(lerp(Math.log(c.z), Math.log(t.z), k));
      // Exponential easing only approaches its target, so land on it once the remainder is
      // under a hundredth of a pixel — otherwise the camera never stops and neither do we.
      const eps = 0.01 / c.z;
      if (Math.abs(t.x - c.x) < eps && Math.abs(t.y - c.y) < eps && Math.abs(t.z - c.z) < c.z * 1e-4) {
        c.x = t.x; c.y = t.y; c.z = t.z;
      }
    }
    const pt = this.gfx.paper ? 1 : 0;
    this.paper += (pt - this.paper) * (1 - Math.exp(-dt * 3.4));
    if (Math.abs(pt - this.paper) < 0.001) this.paper = pt;
    if (this.mouse && !this.drag) {
      const n = this.pick(this.mouse[0], this.mouse[1]);
      if (n !== this.hover) { this.hover = n; this.updateNote(); if (this.o.onHover) this.o.onHover(n); }
    }
    if (this.route && this.routeT < 1) this.routeT = Math.min(1, this.routeT + dt * 0.9);
    const focus = this.pinned || this.hover || this.hits || (this.route && this.routeT > 0) || this.freshCount;
    if (!focus && this.gfx.pulse && this.leaves.length) {
      if (!this.pulse || now - this.pulse.t0 > 3400) {
        const cand = this.leaves[Math.floor(Math.random() * this.leaves.length)];
        if (cand.in.length + cand.out.length >= 3) this.pulse = { n: cand, t0: now };
      }
    } else this.pulse = null;
    if (!this.glitch && now > this.nextGlitch && this.gfx.glitch && !this.o.reduced) {
      const kk = 1 + Math.floor(Math.random() * 3), slices = [];
      for (let i = 0; i < kk; i++) slices.push({ y: Math.random() * this.H, h: 5 + Math.random() * 26, dx: (Math.random() - 0.5) * 22 });
      this.glitch = { t0: now, dur: 55 + Math.random() * 80, slices };
      this.nextGlitch = now + 7000 + Math.random() * 12000;
    }
    if (this.glitch && (now - this.glitch.t0 > this.glitch.dur || !this.gfx.glitch)) this.glitch = null;
    const sig = this.needsDraw(now);
    if (sig !== null && sig === this.lastSig) return;
    this.lastSig = sig;
    this.render(now);
    // frame time, for the toolbar readout — an honest number, not a promise
    const ft = performance.now() - now;
    this.ftEma = this.ftEma == null ? ft : this.ftEma * 0.92 + ft * 0.08;
  };
  Pack.prototype.kick = function () { this.nextGlitch = 0; };

  // ---- replay: the bloom, scrubbed ----------------------------------------------------------
  Pack.prototype.replayStart = function (seconds) {
    const span = seconds || 12;
    this.bloom = { t0: performance.now(), span, span0: span, open: Math.max(0.5, span * 0.06), playing: true, pausedAt: 0, replay: true };
    this.pinned = null; this.hover = null;
    this.updateNote();
  };
  Pack.prototype.replayPause = function (on) {
    const b = this.bloom;
    if (on && b.playing) { b.pausedAt = performance.now(); b.playing = false; }
    else if (!on && !b.playing) { b.t0 += performance.now() - b.pausedAt; b.playing = true; }
  };
  // Changing speed mid-run keeps the progress it has already reached.
  Pack.prototype.replaySpeed = function (mult) {
    const b = this.bloom;
    if (!b.replay || !mult || !b.span0) return;
    const now = performance.now(), prog = this.bloomProgress(now);
    b.span = b.span0 / mult;
    b.open = Math.max(0.4, b.span * 0.06);
    b.t0 = now - prog * (b.span + b.open) * 1000;
    if (!b.playing) b.pausedAt = now;
  };
  Pack.prototype.replayStop = function () {
    this.bloom = { t0: performance.now() - 1e6, span: 0, open: 0.6, playing: true, pausedAt: 0, replay: false };
  };
  Pack.prototype.bloomTime = function (now) {
    const b = this.bloom;
    return ((b.playing ? now : b.pausedAt) - b.t0) / 1000;
  };
  Pack.prototype.bloomProgress = function (now) {
    const b = this.bloom;
    if (!b.span) return 1;
    return clamp(this.bloomTime(now) / (b.span + b.open), 0, 1);
  };

  Pack.prototype.render = function (now) {
    const ctx = this.ctx, W = this.W, H = this.H, dpr = this.dpr, z = this.cam.z;
    const bg = mix3(DARK.bg, PAPER.bg, this.paper), ink = mix3(DARK.ink, PAPER.ink, this.paper), paper = this.paper;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = rgba(bg, 1);
    ctx.fillRect(0, 0, W, H);

    const b = this.bloom, T = this.bloomTime(now), span = b.span, open = b.open;
    const bl = span ? clamp(T / (span + open), 0, 1) : 1;
    if (bl < 1) {
      for (const n of this.order) {
        if (n === this.root) { n.ax = n.x; n.ay = n.y; n.ae = ease(clamp(T / (open * 0.8), 0, 1)); n.ar = n.r * n.ae; continue; }
        const e = ease(clamp((T - n.d0 * span) / open, 0, 1));
        n.ax = n.par.ax + (n.x - n.par.x) * e; n.ay = n.par.ay + (n.y - n.par.y) * e; n.ar = n.r * e; n.ae = e;
      }
    } else {
      for (const n of this.order) { n.ax = n.x; n.ay = n.y; n.ar = n.r; n.ae = 1; }
    }
    const ea = bl < 1 ? clamp((T - span * 0.5) / Math.max(0.4, open), 0, 1) : 1;
    const sel = this.pinned || this.hover;
    this.placeNote();
    const focus = sel || this.hits || (this.route && this.routeT > 0);
    const dim = focus ? 0.28 : 1;
    const [ox, oy] = this.toScreen(0, 0);
    const ms = this.ms = clamp(z * 0.42, 1.2, 4.2) / z;
    const onScreen = (x, y, r) => x + r > 0 && x - r < W && y + r > 0 && y - r < H;

    // ---- world layer
    ctx.save();
    ctx.translate(ox, oy);
    ctx.scale(z, z);
    for (const n of this.containers) {
      if (n.thin) continue;
      const rs = n.ar * z;
      if (rs < 2.5) continue;
      const [sx, sy] = this.toScreen(n.ax, n.ay);
      if (!onScreen(sx, sy, rs)) continue;
      const a = (n.t === "directory" ? 0.32 : n.t === "module" ? 0.2 : 0.13) * n.ae * (n === sel ? 3 : 1);
      ctx.strokeStyle = rgba(ink, Math.min(1, a));
      ctx.lineWidth = (n.t === "directory" ? 1 : 0.8) / z;
      ctx.beginPath(); ctx.arc(n.ax, n.ay, n.ar, 0, TAU); ctx.stroke();
    }
    if (ea > 0 && this.gfx.edgeAlpha > 0) {
      // close up there are few edges on screen, so each may carry more ink; on paper the pink needs more
      ctx.lineWidth = 0.6 / z; ctx.lineCap = "round";
      const zb = clamp(Math.sqrt(z / 3), 1, 2.4);
      // Calls outnumber everything, so each one carries little ink; imports are few enough to
      // read as lines. Both are scaled by how many there actually are, or a small repository
      // comes out invisible and a large one comes out as fog.
      // Past a few thousand of a kind each one has to carry less ink or the weave turns to fog;
      // below that they all carry the same, so a small repository is not invisible.
      const thin = (k, at) => clamp(at / Math.max(at, this.counts[k] || 1), 0.45, 1);
      const A = {
        C: 0.135 * zb * thin("C", 4500) * (1 + paper * 1.4),
        I: 0.22 * Math.min(zb, 1.6) * thin("I", 1500) * (1 + paper * 0.3),
        N: 0.7, M: 0.7,
      };
      for (const kk of ["C", "I", "N", "M"]) {
        if (!this.show[kk] || !this.counts[kk]) continue;
        ctx.globalAlpha = Math.min(1, ea * dim * A[kk] * this.gfx.edgeAlpha);
        ctx.strokeStyle = COL[kk];
        ctx.stroke(this.batch[kk]);
      }
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = rgba(ink, 0.8 * dim);
    ctx.strokeStyle = rgba(ink, 0.8 * dim);
    ctx.lineWidth = 0.8 / z;
    const lp = this.leafPaths(ms, bl < 1);
    if (lp) {
      ctx.fill(lp.solid);
      ctx.stroke(lp.hollow);
    } else {
      // Mid-bloom the marks are still moving, so they are assembled per frame.
      ctx.beginPath();
      const hollow = [];
      for (const n of this.leaves) {
        if (n.ae <= 0) continue;
        const [sx, sy] = this.toScreen(n.ax, n.ay);
        if (!onScreen(sx, sy, 6)) continue;
        if (n.t === "document" || isCls(n.t)) { hollow.push(n); continue; }
        this.markShape(ctx, n, n.ax, n.ay, ms * n.ae);
      }
      ctx.fill();
      ctx.beginPath();
      for (const n of hollow) this.markShape(ctx, n, n.ax, n.ay, ms * 1.3 * n.ae);
      ctx.stroke();
    }

    // a node that landed while you were watching: it opens with a ring, once
    if (this.freshCount) {
      let alive = 0;
      ctx.lineWidth = 1.2 / z;
      for (const n of this.G.nodes) {
        if (!n.popAt) continue;
        const u = (now - n.popAt) / 900;
        if (u < 0) { alive++; continue; }
        if (u >= 1) { delete n.popAt; continue; }
        alive++;
        ctx.globalAlpha = (1 - u) * 0.9;
        ctx.strokeStyle = COL.N;
        ctx.beginPath();
        ctx.arc(n.ax, n.ay, (n.kids.length ? n.ar * z : 4) / z + (14 * u) / z, 0, TAU);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      this.freshCount = alive;
    }

    // idle pulse: a random node breathes its links — the index is live, not a snapshot
    if (this.pulse && !focus) {
      const u = (now - this.pulse.t0) / 2200;
      if (u < 1) this.drawLinks(ctx, this.pulse.n, Math.sin(u * Math.PI) * 0.6, null);
    }
    // the route, drawn in the colours of its own edges as it reveals
    const routeLabels = [];
    if (this.route && this.routeT > 0) {
      const R = this.route;
      let L = this.routeT * R.total;
      ctx.lineCap = "round";
      for (const st of R.steps) {
        if (L <= 0) break;
        const col = st.k === "P" ? rgba(ink, 0.85) : COL[st.k];
        ctx.strokeStyle = col; ctx.lineWidth = 1.5 / z;
        if (L >= st.len) {
          ctx.setLineDash(st.k === "P" ? [3 / z, 3 / z] : []);
          ctx.stroke(st.path);
          ctx.setLineDash([]);
          this.ringAt(ctx, st.b, col, 1);
          routeLabels.push({ n: st.b, col });
        } else {
          ctx.setLineDash([L, st.len + 100]);
          ctx.stroke(st.path);
          ctx.setLineDash([]);
        }
        this.ringAt(ctx, st.a, col, 1);
        routeLabels.push({ n: st.a, col });
        L -= st.len;
      }
    }
    // search hits, numbered on the specimen; a weak match is drawn weakly
    const hitLabels = [];
    if (this.hits) this.hits.forEach((h, i) => {
      const n = h.node, col = h.weak ? "rgba(255,58,168,.55)" : COL.C;
      ctx.strokeStyle = col; ctx.lineWidth = 1 / z;
      ctx.setLineDash(h.weak ? [2 / z, 2 / z] : []);
      ctx.beginPath();
      ctx.arc(n.ax, n.ay, (n.kids.length ? n.ar * z + 5 : 6.5) / z, 0, TAU);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = col;
      ctx.beginPath(); this.markShape(ctx, n, n.ax, n.ay, ms * 1.7); ctx.fill();
      hitLabels.push({ n, col, text: `${i + 1}` });
    });
    const selLabels = [];
    if (sel) {
      this.drawLinks(ctx, sel, 1, selLabels);
      ctx.fillStyle = rgba(ink, 1);
      ctx.beginPath(); this.markShape(ctx, sel, sel.ax, sel.ay, ms * 1.8); ctx.fill();
      this.crosshair(ctx, sel, rgba(ink, 1));
    }
    if (this.mark && this.mark !== sel) {
      ctx.globalAlpha = 0.9;
      this.crosshair(ctx, this.mark, COL.N);
      ctx.globalAlpha = 1;
    }
    ctx.restore();

    // ---- screen layer: labels, greedy — no two overlap
    const placed = [];
    const fits = (x, y, w, h) => {
      if (placed.length > 320) return false;
      for (const p of placed) if (x < p[0] + p[2] && x + w > p[0] && y < p[1] + p[3] && y + h > p[1]) return false;
      placed.push([x, y, w, h]);
      return true;
    };
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    const label = (n, col, text, dy, force) => {
      const [sx, sy] = this.toScreen(n.ax, n.ay);
      const fs = 9.5;
      ctx.font = `${fs}px ${FONT}`;
      const w = ctx.measureText(text).width;
      const x = sx + ms * z + 4, y = sy + (dy || 3);
      if (!onScreen(x, y, w)) return;
      if (!force && !fits(x, y - fs, w, fs + 2)) return;
      ctx.fillStyle = col;
      ctx.fillText(text, x, y);
    };
    if (this.gfx.labels) for (const L of selLabels) label(L.n, L.col, L.n.n);
    if (sel) label(sel, rgba(ink, 1), sel.n, -8, true);
    for (const L of hitLabels) label(L.n, L.col, L.text, 3, true);
    if (this.gfx.labels) for (const L of hitLabels) label(L.n, L.col, `   ${L.n.n}`, 3, false);
    for (const L of routeLabels) label(L.n, L.col, this.shortName(L.n), 3, true);
    if (this.gfx.labels) {
      for (const n of this.containers) {
        const rs = n.ar * z;
        const [sx, sy] = this.toScreen(n.ax, n.ay);
        if (!onScreen(sx, sy, rs)) continue;
        if (n.t === "directory") {
          if (rs < 46 || n.thin) continue;
          const fs = 9.5;
          ctx.font = `${fs}px ${FONT}`;
          ctx.fillStyle = rgba(ink, 0.7 * n.ae);
          const name = this.dirName(n);
          if (name.length * fs * 0.62 < rs * 1.9) this.arcText(ctx, name, sx, sy, rs - fs - 3);
          continue;
        }
        const min = n.t === "module" ? 22 : 18;
        if (rs < min) continue;
        const fs = 9.5, a = clamp((rs - min) / 14, 0, 1) * 0.85 * n.ae;
        ctx.font = `${fs}px ${FONT}`;
        const text = n.t === "module" || n.t === "document" ? this.shortName(n) : n.n;
        const w = ctx.measureText(text).width;
        const inside = w < rs * 1.5;
        const x = sx - w / 2, y = inside ? sy - rs + fs + 4 : sy - rs - 5;
        if (!fits(x, y - fs, w, fs + 2)) continue;
        ctx.fillStyle = rgba(ink, n === sel ? 1 : a);
        ctx.fillText(text, x, y);
      }
      if (z > 5.5) {
        const la = clamp((z - 5.5) / 3, 0, 1) * 0.72, fs = 9;
        ctx.font = `${fs}px ${FONT}`;
        for (const n of this.leaves) {
          if (n === sel || n === this.mark) continue;
          const [sx, sy] = this.toScreen(n.ax, n.ay);
          if (!onScreen(sx, sy, 60)) continue;
          const text = n.n.includes(".") ? n.n.slice(n.n.lastIndexOf(".") + 1) : n.n;
          const w = ctx.measureText(text).width;
          if (!fits(sx + ms * z + 4, sy - fs + 3, w, fs + 2)) continue;
          ctx.fillStyle = rgba(ink, la * n.ae);
          ctx.fillText(text, sx + ms * z + 4, sy + 3);
        }
      }
    }
    // dashed leader from the pinned / hovered node to its card
    if (sel && !this.note.hidden) {
      const nr = this.note.getBoundingClientRect(), cr = this.canvas.getBoundingClientRect();
      const [sx, sy] = this.toScreen(sel.ax, sel.ay);
      const corners = [[nr.left, nr.top], [nr.right, nr.top], [nr.left, nr.bottom], [nr.right, nr.bottom]]
        .map(([x, y]) => [x - cr.left, y - cr.top]);
      let best = corners[0], bd = Infinity;
      for (const cc of corners) { const d = Math.hypot(cc[0] - sx, cc[1] - sy); if (d < bd) { bd = d; best = cc; } }
      ctx.strokeStyle = rgba(ink, 0.7);
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(best[0], best[1]); ctx.stroke();
      ctx.setLineDash([]);
    }
    // row tearing — rare, short, and never over a card: the note is a DOM layer above this one
    if (this.glitch) {
      const g = this.glitch, cv = this.canvas;
      for (const s of g.slices) ctx.drawImage(cv, 0, s.y * dpr, W * dpr, s.h * dpr, s.dx, s.y, W, s.h);
    }
  };

  // ---- interaction --------------------------------------------------------------------------
  Pack.prototype.pick = function (sx, sy) {
    const z = this.cam.z;
    let best = null, bd = 9;
    for (const n of this.leaves) {
      const x = this.W / 2 + (n.ax - this.cam.x) * z;
      if (x < sx - 9 || x > sx + 9) continue;
      const y = this.H / 2 + (n.ay - this.cam.y) * z;
      const d = Math.hypot(x - sx, y - sy);
      if (d < bd) { bd = d; best = n; }
    }
    return best || this.containerAt(sx, sy);
  };
  Pack.prototype.containerAt = function (sx, sy) {
    const z = this.cam.z, lim = Math.min(this.W, this.H) * 0.46;
    let cont = null;
    for (const n of this.containers) {
      const rs = n.ar * z;
      if (rs < 12 || rs > lim) continue;
      const [x, y] = this.toScreen(n.ax, n.ay);
      if (Math.hypot(x - sx, y - sy) <= rs && (!cont || n.depth > cont.depth)) cont = n;
    }
    return cont;
  };

  // The card: name / type / path:lines / summary / ← incoming → outgoing. Pinned it becomes
  // clickable (the explorer link lives in it), hovered it stays out of the way.
  Pack.prototype.noteHTML = function (n) {
    const lines = n.l && n.l[0] ? `:${n.l[0]}–${n.l[1]}` : "";
    const pinned = this.pinned === n;
    return `<div class="n">${esc(n.n)}</div>` +
      `<div class="t">${esc(n.t)} · ${esc(n.p)}${lines}</div>` +
      (n.g ? `<div class="g">${esc(n.g)}</div>` : "") +
      (n.s ? `<div class="s">${esc(n.s)}</div>` : `<div class="s dim">no summary yet</div>`) +
      `<div class="d"><i class="in">← ${n.in.length} incoming</i><i class="out">→ ${n.out.length} outgoing</i>` +
      (n.kids.length ? `<i class="sub">⊃ ${n.kids.length}</i>` : "") + `</div>` +
      (pinned ? `<div class="act"><a href="#" data-act="explorer">open in explorer →</a>` +
        `<a href="#" data-act="route">route from here</a>` +
        `<a href="#" data-act="scope">scope here</a></div>` : "");
  };
  Pack.prototype.updateNote = function () {
    const n = this.pinned || this.hover, note = this.note;
    if (!n) {
      if (!this.hits || !this.hits.length) { note.hidden = true; this.noteList = false; return; }
      note.innerHTML = `<div class="t">${esc(this.hitLabel || "hits")}</div>` +
        this.hits.map((h, i) => `<div class="hit"><i>${i + 1}</i><span data-id="${esc(h.node.id)}">${esc(h.node.n)}${h.weak ? " <em>weak</em>" : ""}</span></div>`).join("");
      note.hidden = false;
      note.classList.add("pinned");
      this.noteList = true;
      // top-right: the route panel and the legend own the other corners
      const w = note.offsetWidth || 320;
      note.style.left = Math.max(8, this.W - w - 12) + "px";
      note.style.top = "12px";
      return;
    }
    note.innerHTML = this.noteHTML(n);
    note.classList.toggle("pinned", this.pinned === n);
    note.hidden = false;
    this.noteList = false;
    this.noteW = this.noteH = 0;   // re-measure: the card just changed size
    this.placeNote(true);
  };
  // The card is an annotation on a specimen, so it stays beside the node it annotates —
  // through a pan, a zoom, a flight, and a live rebuild. Measured once per content change;
  // moved only when it has actually moved.
  Pack.prototype.placeNote = function (force) {
    const n = this.pinned || this.hover;
    if (!n || this.note.hidden || this.noteList) return;
    if (!this.noteW) { this.noteW = this.note.offsetWidth || 320; this.noteH = this.note.offsetHeight || 140; }
    const nw = this.noteW, nh = this.noteH;
    const [sx, sy] = this.toScreen(n.ax, n.ay);
    let x = sx + 18, y = sy + 18;
    if (x + nw > this.W - 8) x = sx - nw - 18;
    if (y + nh > this.H - 8) y = sy - nh - 18;
    x = clamp(x, 8, Math.max(8, this.W - nw - 8));
    y = clamp(y, 8, Math.max(8, this.H - nh - 8));
    if (force || Math.abs(x - this.noteX) > 0.5 || Math.abs(y - this.noteY) > 0.5) {
      this.noteX = x; this.noteY = y;
      this.note.style.left = x + "px";
      this.note.style.top = y + "px";
    }
  };
  Pack.prototype.setHits = function (hits, label, frame) {
    this.hitsV++;
    this.hits = hits
      ? hits.map((h) => ({ node: this.G.byId.get(h.id), weak: !!h.weak, score: h.score })).filter((h) => h.node)
      : null;
    this.hitLabel = label || "";
    this.pinned = null;
    if (this.hits && this.hits.length && frame !== false) {
      // Frame everything the answer touches, so a search reads as one picture.
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const h of this.hits) { x0 = Math.min(x0, h.node.x); y0 = Math.min(y0, h.node.y); x1 = Math.max(x1, h.node.x); y1 = Math.max(y1, h.node.y); }
      const span = Math.max(Math.hypot(x1 - x0, y1 - y0), 10);
      this.setTarget((x0 + x1) / 2, (y0 + y1) / 2, (0.6 * Math.min(this.W, this.H)) / span);
    }
    this.updateNote();
    return this.hits ? this.hits.length : 0;
  };
  Pack.prototype.setShow = function (k, on) { this.show[k] = on; };
  // Every gfx flag is read by render(), so the signature below covers them one by one.
  Pack.prototype.setGfx = function (patch) {
    Object.assign(this.gfx, patch);
    if (!this.gfx.glitch) this.glitch = null;
    this.stage.classList.toggle("grain", !!this.gfx.grain);
    this.stage.classList.toggle("paper", !!this.gfx.paper);
    this.lastSig = null;
  };
  Pack.prototype.stats = function () {
    return {
      nodes: this.G.nodes.length, edges: this.edges.length,
      layoutMs: this.layoutMs, fps: this.ftEma ? Math.round(1000 / Math.max(this.ftEma, 1)) : 0,
      frameMs: this.ftEma ? Math.round(this.ftEma * 10) / 10 : 0,
    };
  };

  window.Pack = Pack;
})();
