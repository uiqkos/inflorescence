/* pack.js — the whole index as nested discs.
   Florets (functions, methods) sit in golden-angle spirals inside the file that defines them, files inside their
   directories: CONTAINS is drawn as nesting, the way a capitulum holds its florets. Every CALLS / IMPORTS / INHERITS
   edge is bundled through the centres of the discs it crosses, so a call between two files reads as a stem through both.
   The camera is a world→screen transform; the page (walk.js) drives it. */
(function () {
  'use strict';
  const TAU = Math.PI * 2, GOLD = Math.PI * (3 - Math.sqrt(5));
  const COL = { C: '#ff3aa8', I: '#8b7bff', N: '#e6ff3d', M: '#e6ff3d' };
  const NAME = { C: 'CALLS', I: 'IMPORTS', N: 'INHERITS', M: 'IMPLEMENTS', P: 'CONTAINS' };
  const FONT = '"IBM Plex Mono", Menlo, monospace';
  const ORD = { directory: 0, document: 1, module: 2, class: 3, interface: 3, enum: 3, struct: 3, trait: 3, function: 4, method: 5 };
  const DARK = { bg: [7, 7, 7], ink: [235, 232, 225] }, PAPER = { bg: [198, 200, 195], ink: [17, 17, 17] };
  const isCls = t => t === 'class' || t === 'interface' || t === 'struct' || t === 'enum' || t === 'trait';
  const esc = s => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = u => u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
  const rnd = i => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const mix3 = (a, b, t) => [Math.round(lerp(a[0], b[0], t)), Math.round(lerp(a[1], b[1], t)), Math.round(lerp(a[2], b[2], t))];

  // ---- model: the raw export becomes a tree (CONTAINS) plus typed edges
  function model(raw) {
    const byId = new Map(); raw.nodes.forEach(n => { n.kids = []; byId.set(n.id, n); });
    const root = byId.get('dir:.');
    raw.nodes.forEach(n => { if (n === root) return; const p = byId.get(n.pa) || root; n.par = p; p.kids.push(n); });
    return { project: raw.project, nodes: raw.nodes, edges: raw.edges, byId: Object.fromEntries(byId), totals: raw.totals, edgeTotals: raw.edge_totals, chunks: raw.chunks, root };
  }

  function Pack(stage, G, o) {
    this.stage = stage; this.G = G; this.o = Object.assign({ onSelect: null, reduced: false }, o || {});
    const c = this.canvas = document.createElement('canvas'); stage.appendChild(c); this.ctx = c.getContext('2d');
    this.note = document.createElement('div'); this.note.className = 'note'; this.note.hidden = true; stage.appendChild(this.note);
    this.cam = { x: 0, y: 0, z: 1 }; this.target = { x: 0, y: 0, z: 1 }; this.pan = { x: 0, y: 0 }; this.panTarget = { x: 0, y: 0 };
    this.view = { x0: 0, y0: 0, w: 1, h: 1 };
    this.paper = 0; this.paperTarget = 0;
    this.hover = null; this.pinned = null; this.hits = null; this.route = null; this.routeT = 0; this.mark = null;
    this.show = { C: true, I: true, N: true, M: true };
    this.pulse = null; this.flash = null; this.flashing = false; this.glitch = null; this.nextGlitch = 2500; this.quiet = false;
    this.mouse = null; this.drag = null; this.dragged = false; this.last = 0;
    this.t0 = performance.now(); this.bloomDur = this.o.reduced ? 0 : 2800;
    this.build(); this.layout();
    this.resize(); new ResizeObserver(() => this.resize()).observe(stage);
    c.addEventListener('mousemove', e => this.onMove(e));
    c.addEventListener('mouseleave', () => { this.mouse = null; if (this.hover) { this.hover = null; this.updateNote(); } });
    c.addEventListener('mousedown', e => { this.drag = { x: e.clientX, y: e.clientY, px: this.panTarget.x, py: this.panTarget.y }; this.dragged = false; });
    addEventListener('mouseup', () => { this.drag = null; });
    c.addEventListener('click', e => this.onClick(e));
    const loop = now => { this.frame(now); requestAnimationFrame(loop); }; requestAnimationFrame(loop);
  }
  Pack.model = model;

  Pack.prototype.build = function () {
    const G = this.G, root = this.root = G.root;
    (function sort(n) { n.kids.sort((a, b) => (ORD[a.t] - ORD[b.t]) || a.n.localeCompare(b.n)); n.kids.forEach(sort); })(root);
    const order = this.order = [];
    (function dep(n, d, anc) { n.depth = d; n.anc = [n, ...anc]; order.push(n); n.kids.forEach(k => dep(k, d + 1, n.anc)); })(root, 0, []);
    G.nodes.forEach(n => { n.out = []; n.in = []; });
    this.edges = [];
    for (const e of G.edges) {
      const s = G.byId[e[0]], d = G.byId[e[1]]; if (!s || !d || s === d) continue;
      const E = { s, d, k: e[2], res: e[3] }; s.out.push(E); d.in.push(E); this.edges.push(E);
    }
    this.counts = {}; for (const E of this.edges) this.counts[E.k] = (this.counts[E.k] || 0) + 1;
    for (const n of order) n.thin = n.t === 'directory' && n.kids.length === 1 && n.kids[0].t === 'directory';
    this.leaves = order.filter(n => !n.kids.length && n.t !== 'directory');
    this.containers = order.filter(n => n.kids.length || n.t === 'directory');
  };

  // ---- layout: phyllotaxis inside every container, then a relaxation so nothing overlaps
  Pack.prototype.layout = function () {
    function pack(n) {
      n.kids.forEach(pack);
      if (!n.kids.length) { n.r = n.t === 'document' ? 1.4 : n.t === 'module' ? 1.5 : isCls(n.t) ? 1.25 : 1; n.lx = 0; n.ly = 0; return; }
      const gap = 0.55;
      // smallest first: florets gather in the centre, the discs of classes and files form a corolla around them —
      // unless one child holds most of the area (src/ inside the root), which then sits in the centre with the rest around it
      let items = [...n.kids].sort((a, b) => (a.r - b.r) || (ORD[a.t] - ORD[b.t]) || a.n.localeCompare(b.n));
      const areas = items.map(k => k.r * k.r), tot = areas.reduce((x, y) => x + y, 0);
      if (Math.max(...areas) > 0.45 * tot) items.reverse();
      let area = 0;
      items.forEach((k, i) => {
        const a = Math.PI * Math.pow(k.r + gap / 2, 2), rho = Math.sqrt((area + a * 0.5) / Math.PI); area += a;
        const th = i * GOLD + n.depth * 0.9; k.lx = rho * Math.cos(th); k.ly = rho * Math.sin(th); k.idx = i;
      });
      const N = items.length;
      for (let it = 0; it < 240; it++) {
        let moved = false;
        for (let i = 0; i < N; i++) { const a = items[i];
          for (let j = i + 1; j < N; j++) { const b = items[j];
            let dx = b.lx - a.lx, dy = b.ly - a.ly, d = Math.hypot(dx, dy); const min = a.r + b.r + gap;
            if (d >= min) continue; moved = true;
            if (d < 1e-6) { dx = Math.cos(i * 7 + j); dy = Math.sin(i * 7 + j); d = 1; }
            const push = (min - d) / d * 0.52, wa = (b.r * b.r) / (a.r * a.r + b.r * b.r), wb = 1 - wa;
            a.lx -= dx * push * wa; a.ly -= dy * push * wa; b.lx += dx * push * wb; b.ly += dy * push * wb;
          } }
        if (it < 170) for (const k of items) { k.lx *= 0.988; k.ly *= 0.988; }
        else if (!moved) break;
      }
      let cx = 0, cy = 0, m = 0;
      for (const k of items) { const w = k.r * k.r; cx += k.lx * w; cy += k.ly * w; m += w; }
      cx /= m; cy /= m; let R = 0;
      for (const k of items) { k.lx -= cx; k.ly -= cy; R = Math.max(R, Math.hypot(k.lx, k.ly) + k.r); }
      n.r = R + (n.thin ? 0.4 : n.t === 'directory' ? 2.4 : n.t === 'module' ? 1.5 : 0.9);
    }
    pack(this.root);
    (function abs(n, x, y) { n.x = x + (n.lx || 0); n.y = y + (n.ly || 0); n.kids.forEach(k => abs(k, n.x, n.y)); })(this.root, 0, 0);
    for (const n of this.order) n.delay = n === this.root ? 0 : 0.25 + n.depth * 0.17 + (n.idx / n.par.kids.length) * 0.45 + rnd(n.idx + n.depth) * 0.08;
    this.R = this.root.r;
    for (const E of this.edges) { const b = this.bundle(this.pathPoints(E.s, E.d), 0.86); E.path = b.path; E.len = b.len; }
    this.batch = {}; for (const k of ['C', 'I', 'N', 'M']) this.batch[k] = new Path2D();
    for (const E of this.edges) this.batch[E.k].addPath(E.path);
  };
  Pack.prototype.pathPoints = function (s, d) {
    const set = new Set(s.anc); let l = null;
    for (const n of d.anc) if (set.has(n)) { l = n; break; }
    const up = [], down = [];
    for (const n of s.anc) { if (n === l) break; up.push(n); }
    for (const n of d.anc) { if (n === l) break; down.push(n); }
    const pts = [[s.x, s.y]];
    for (let i = 1; i < up.length; i++) pts.push([up[i].x, up[i].y]);
    if (l !== s && l !== d) pts.push([l.x, l.y]);
    for (let i = down.length - 1; i >= 1; i--) pts.push([down[i].x, down[i].y]);
    pts.push([d.x, d.y]);
    return pts;
  };
  // d3.curveBundle: straighten toward the chord, then a B-spline through the points
  Pack.prototype.bundle = function (pts, beta) {
    const n = pts.length - 1, [x0, y0] = pts[0], [xn, yn] = pts[n];
    const q = pts.map((p, i) => [beta * p[0] + (1 - beta) * (x0 + i / n * (xn - x0)), beta * p[1] + (1 - beta) * (y0 + i / n * (yn - y0))]);
    let len = 0; for (let i = 1; i < q.length; i++) len += Math.hypot(q[i][0] - q[i - 1][0], q[i][1] - q[i - 1][1]);
    const P = new Path2D(); let px0, py0, px1, py1, st = 0;
    const pt = (x, y) => {
      switch (st) {
        case 0: st = 1; P.moveTo(x, y); break;
        case 1: st = 2; break;
        case 2: st = 3; P.lineTo((5 * px0 + px1) / 6, (5 * py0 + py1) / 6); // falls through
        default: P.bezierCurveTo((2 * px0 + px1) / 3, (2 * py0 + py1) / 3, (px0 + 2 * px1) / 3, (py0 + 2 * py1) / 3, (px0 + 4 * px1 + x) / 6, (py0 + 4 * py1 + y) / 6);
      }
      px0 = px1; px1 = x; py0 = py1; py1 = y;
    };
    q.forEach(p => pt(p[0], p[1]));
    if (st === 3) { pt(px1, py1); P.lineTo(px1, py1); } else if (st === 2) P.lineTo(px1, py1);
    return { path: P, len: len * 0.96 };
  };

  // ---- a route between two nodes over the real edges (CONTAINS steps allowed, at a higher cost)
  Pack.prototype.routeBetween = function (a, b) {
    if (!a || !b) return null;
    const dist = new Map([[a, 0]]), prev = new Map(); const open = [a]; const done = new Set();
    const relax = (x, y, w, k, dir) => { const nd = dist.get(x) + w; if (nd < (dist.has(y) ? dist.get(y) : Infinity)) { dist.set(y, nd); prev.set(y, { x, k, dir }); if (!done.has(y)) open.push(y); } };
    while (open.length) {
      let bi = 0; for (let i = 1; i < open.length; i++) if (dist.get(open[i]) < dist.get(open[bi])) bi = i;
      const x = open.splice(bi, 1)[0]; if (done.has(x)) continue; done.add(x); if (x === b) break;
      for (const E of x.out) relax(x, E.d, 1, E.k, '>');
      for (const E of x.in) relax(x, E.s, 1, E.k, '<');
      if (x.par) relax(x, x.par, 2.4, 'P', '<');
      for (const k of x.kids) relax(x, k, 2.4, 'P', '>');
    }
    if (!prev.has(b)) return null;
    const steps = []; let y = b;
    while (y !== a) { const p = prev.get(y); steps.push({ a: p.x, b: y, k: p.k, dir: p.dir }); y = p.x; }
    steps.reverse(); let total = 0;
    for (const st of steps) { const pts = st.k === 'P' ? [[st.a.x, st.a.y], [st.b.x, st.b.y]] : this.pathPoints(st.a, st.b); const bd = this.bundle(pts, 0.86); st.path = bd.path; st.len = Math.max(bd.len, 0.5); total += st.len; }
    return { a, b, steps, total, points: [[a.x, a.y], ...steps.map(s => [s.b.x, s.b.y])] };
  };

  // ---- camera
  Pack.prototype.resize = function () {
    const r = this.stage.getBoundingClientRect(); const dpr = this.dpr = Math.min(2, devicePixelRatio || 1);
    this.W = r.width; this.H = r.height; this.canvas.width = r.width * dpr; this.canvas.height = r.height * dpr;
    this.canvas.style.width = r.width + 'px'; this.canvas.style.height = r.height + 'px';
  };
  Pack.prototype.fitZoom = function (n, frac) { return frac * Math.min(this.view.w, this.view.h) / (2 * n.r); };
  Pack.prototype.toScreen = function (wx, wy) { const v = this.view, c = this.cam; return [v.x0 + v.w / 2 + this.pan.x + (wx - c.x) * c.z, v.y0 + v.h / 2 + this.pan.y + (wy - c.y) * c.z]; };
  Pack.prototype.toWorld = function (sx, sy) { const v = this.view, c = this.cam; return [c.x + (sx - v.x0 - v.w / 2 - this.pan.x) / c.z, c.y + (sy - v.y0 - v.h / 2 - this.pan.y) / c.z]; };
  Pack.prototype.setTarget = function (x, y, z, snap) { this.target = { x, y, z }; if (snap) this.cam = { x, y, z }; };
  Pack.prototype.releasePan = function () { this.panTarget = { x: 0, y: 0 }; };

  // ---- drawing helpers
  Pack.prototype.markShape = function (ctx, n, x, y, s) {
    if (n.t === 'module') ctx.rect(x - s * 1.1, y - s * 1.1, s * 2.2, s * 2.2);
    else if (isCls(n.t)) { ctx.moveTo(x, y - s * 1.5); ctx.lineTo(x + s * 1.5, y); ctx.lineTo(x, y + s * 1.5); ctx.lineTo(x - s * 1.5, y); ctx.closePath(); }
    else { ctx.moveTo(x + s, y); ctx.arc(x, y, s, 0, TAU); }
  };
  Pack.prototype.dirName = function (n) {
    if (n.id === 'dir:.') return this.G.project + '/';
    const base = n.p.split('/').pop() + '/';
    return n.par && n.par.thin && n.par.id !== 'dir:.' ? n.par.p.split('/').pop() + '/' + base : base;
  };
  Pack.prototype.arcText = function (ctx, text, cx, cy, r) {
    ctx.save(); ctx.translate(cx, cy); const cw = ctx.measureText('M').width; const total = text.length * cw / r;
    let ang = -Math.PI / 2 - total / 2;
    for (const ch of text) { ctx.save(); ctx.rotate(ang); ctx.translate(r, 0); ctx.rotate(Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText(ch, 0, 0); ctx.restore(); ang += cw / r; }
    ctx.restore();
  };
  Pack.prototype.ringAt = function (ctx, n, col, w) {
    const z = this.cam.z; ctx.strokeStyle = col; ctx.lineWidth = (w || 1) / z; ctx.beginPath();
    ctx.arc(n.ax, n.ay, n.kids.length ? n.ar : 6.5 / z, 0, TAU); ctx.stroke();
  };
  Pack.prototype.crosshair = function (ctx, n, col) {
    const z = this.cam.z, s = (n.kids.length ? Math.max(9, n.ar * z + 6) : 9) / z; ctx.strokeStyle = col; ctx.lineWidth = 1 / z;
    ctx.strokeRect(n.ax - s, n.ay - s, 2 * s, 2 * s);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.beginPath(); ctx.moveTo(n.ax + dx * s, n.ay + dy * s); ctx.lineTo(n.ax + dx * (s + 6 / z), n.ay + dy * (s + 6 / z)); ctx.stroke(); }
  };
  Pack.prototype.drawLinks = function (ctx, n, alpha, labels) {
    const z = this.cam.z; ctx.lineWidth = 1.1 / z; ctx.lineCap = 'round';
    const draw = (E, col, other) => {
      if (!this.show[E.k]) return;
      ctx.globalAlpha = alpha * 0.9; ctx.strokeStyle = col; ctx.stroke(E.path);
      ctx.globalAlpha = alpha; ctx.fillStyle = col; ctx.beginPath(); this.markShape(ctx, other, other.ax, other.ay, this.ms * 1.7); ctx.fill();
      if (labels) labels.push({ n: other, col });
    };
    for (const E of n.in) draw(E, E.k === 'C' ? COL.C : E.k === 'I' ? COL.I : COL.N, E.s);
    for (const E of n.out) draw(E, E.k === 'C' ? '#4d86ff' : E.k === 'I' ? COL.I : COL.N, E.d);
    ctx.globalAlpha = 1;
  };

  // ---- frame
  Pack.prototype.frame = function (now) {
    const dt = Math.min(0.1, (now - (this.last || now)) / 1000); this.last = now;
    const k = 1 - Math.exp(-dt * 5.5), c = this.cam, t = this.target;
    c.x += (t.x - c.x) * k; c.y += (t.y - c.y) * k; c.z = Math.exp(lerp(Math.log(c.z), Math.log(t.z), k));
    this.pan.x += (this.panTarget.x - this.pan.x) * k; this.pan.y += (this.panTarget.y - this.pan.y) * k;
    this.paper += (this.paperTarget - this.paper) * (1 - Math.exp(-dt * 3));
    if (this.mouse && !this.drag) { const n = this.pick(this.mouse[0], this.mouse[1]); if (n !== this.hover) { this.hover = n; this.updateNote(); } }
    const focus = this.pinned || this.hover || this.hits || (this.route && this.routeT > 0);
    if (!focus && !this.quiet && !this.flashing) {
      if (!this.pulse || now - this.pulse.t0 > 3400) { const cand = this.leaves[Math.floor(Math.random() * this.leaves.length)]; if (cand.in.length + cand.out.length >= 3) this.pulse = { n: cand, t0: now }; }
    } else this.pulse = null;
    if (this.flashing && !this.o.reduced && (!this.flash || now - this.flash.t0 > 2600)) {
      const mods = this.containers.filter(n => n.t === 'module' && n.depth >= 3);
      this.flash = { n: mods[Math.floor(Math.random() * mods.length)], t0: now };
    }
    if (!this.flashing) this.flash = null;
    if (!this.glitch && now > this.nextGlitch && !this.o.reduced) {
      const kk = 2 + Math.floor(Math.random() * 4), slices = [];
      for (let i = 0; i < kk; i++) slices.push({ y: Math.random() * this.H, h: 6 + Math.random() * 40, dx: (Math.random() - 0.5) * 36 });
      this.glitch = { t0: now, dur: 70 + Math.random() * 110, slices, tint: Math.random() < 0.5 };
      this.nextGlitch = now + 3000 + Math.random() * 7000;
    }
    if (this.glitch && now - this.glitch.t0 > this.glitch.dur) this.glitch = null;
    this.render(now);
  };
  Pack.prototype.kick = function () { this.nextGlitch = 0; };

  Pack.prototype.render = function (now) {
    const ctx = this.ctx, W = this.W, H = this.H, dpr = this.dpr, z = this.cam.z, T = (now - this.t0) / 1000;
    const bg = mix3(DARK.bg, PAPER.bg, this.paper), ink = mix3(DARK.ink, PAPER.ink, this.paper), paper = this.paper;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = rgba(bg, 1); ctx.fillRect(0, 0, W, H);
    const bl = this.bloomDur ? clamp(T * 1000 / this.bloomDur, 0, 1) : 1;
    if (bl < 1) for (const n of this.order) {
      if (n === this.root) { n.ax = n.x; n.ay = n.y; n.ae = ease(clamp(T / 0.7, 0, 1)); n.ar = n.r * n.ae; continue; }
      const e = ease(clamp((T - n.delay) / 0.8, 0, 1)); n.ax = n.par.ax + (n.x - n.par.x) * e; n.ay = n.par.ay + (n.y - n.par.y) * e; n.ar = n.r * e; n.ae = e;
    } else for (const n of this.order) { n.ax = n.x; n.ay = n.y; n.ar = n.r; n.ae = 1; }
    const ea = bl < 1 ? clamp((T - 1.6) / 1.0, 0, 1) : 1;
    const sel = this.pinned || this.hover; const focus = sel || this.hits || (this.route && this.routeT > 0);
    const dim = focus ? 0.28 : 1;
    const [ox, oy] = this.toScreen(0, 0);
    const ms = this.ms = clamp(z * 0.42, 1.2, 4.2) / z;
    const onScreen = (x, y, r) => x + r > 0 && x - r < W && y + r > 0 && y - r < H;

    // ---- world layer
    ctx.save(); ctx.translate(ox, oy); ctx.scale(z, z);
    for (const n of this.containers) {
      if (n.thin) continue;
      const rs = n.ar * z; if (rs < 2.5) continue; const [sx, sy] = this.toScreen(n.ax, n.ay); if (!onScreen(sx, sy, rs)) continue;
      const a = (n.t === 'directory' ? 0.32 : n.t === 'module' ? 0.2 : 0.13) * n.ae * (n === sel ? 3 : 1);
      ctx.strokeStyle = rgba(ink, Math.min(1, a)); ctx.lineWidth = (n.t === 'directory' ? 1 : 0.8) / z; ctx.beginPath(); ctx.arc(n.ax, n.ay, n.ar, 0, TAU); ctx.stroke();
    }
    if (ea > 0) {
      // close up there are few edges on screen, so each may carry more ink; on paper the pink needs more still
      ctx.lineWidth = 0.6 / z; ctx.lineCap = 'round'; const zb = clamp(Math.sqrt(z / 3), 1, 2.4);
      const A = { C: 0.075 * zb * (1 + paper * 1.4), I: 0.2 * Math.min(zb, 1.6) * (1 + paper * 0.3), N: 0.7, M: 0.7 };
      for (const kk of ['C', 'I', 'N', 'M']) { if (!this.show[kk]) continue; ctx.globalAlpha = Math.min(1, ea * dim * A[kk]); ctx.strokeStyle = COL[kk]; ctx.stroke(this.batch[kk]); }
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = rgba(ink, 0.8 * dim); ctx.strokeStyle = rgba(ink, 0.8 * dim); ctx.lineWidth = 0.8 / z;
    ctx.beginPath(); const hollow = [];
    for (const n of this.leaves) {
      if (n.ae <= 0) continue; const [sx, sy] = this.toScreen(n.ax, n.ay); if (!onScreen(sx, sy, 6)) continue;
      if (n.t === 'document' || isCls(n.t)) { hollow.push(n); continue; }
      this.markShape(ctx, n, n.ax, n.ay, ms * n.ae);
    }
    ctx.fill(); ctx.beginPath(); for (const n of hollow) this.markShape(ctx, n, n.ax, n.ay, ms * 1.3 * n.ae); ctx.stroke();

    // idle pulse: a random node breathes its links — the graph is a live index, not a picture
    if (this.pulse && !focus) { const u = (now - this.pulse.t0) / 2200; if (u < 1) this.drawLinks(ctx, this.pulse.n, Math.sin(u * Math.PI) * 0.6, null); }
    // watch flash: a file changed → its summary and every summary up to the root are rewritten
    const flashLabels = [];
    if (this.flash) {
      const f = this.flash, el = now - f.t0; const chain = f.n.anc.filter(n => !n.thin);
      chain.forEach((n, i) => {
        const t0 = i * 260, u = (el - t0) / 900; if (u < 0 || u > 1) return;
        const a = Math.sin(u * Math.PI); ctx.globalAlpha = a; this.ringAt(ctx, n, i === 0 ? COL.N : rgba(ink, 1), i === 0 ? 1.6 : 1.1); ctx.globalAlpha = 1;
        flashLabels.push({ n, col: i === 0 ? COL.N : rgba(ink, 0.9), a, text: i === 0 ? 'changed' : 'summary ↻' });
      });
    }
    // route between chapters, revealed as the camera flies
    const routeLabels = [];
    if (this.route && this.routeT > 0) {
      const R = this.route; let L = this.routeT * R.total; ctx.lineCap = 'round';
      for (const st of R.steps) {
        if (L <= 0) break;
        const col = st.k === 'P' ? rgba(ink, 0.85) : COL[st.k]; ctx.strokeStyle = col; ctx.lineWidth = 1.5 / z;
        if (L >= st.len) { ctx.setLineDash(st.k === 'P' ? [3 / z, 3 / z] : []); ctx.stroke(st.path); ctx.setLineDash([]); this.ringAt(ctx, st.b, col, 1); routeLabels.push({ n: st.b, col }); }
        else { ctx.setLineDash([L, st.len + 100]); ctx.stroke(st.path); ctx.setLineDash([]); }
        this.ringAt(ctx, st.a, col, 1); routeLabels.push({ n: st.a, col });
        L -= st.len;
      }
    }
    // hits from a recorded search, numbered
    const hitLabels = [];
    if (this.hits) this.hits.forEach((h, i) => {
      const n = h.node, col = h.weak ? 'rgba(255,58,168,.55)' : COL.C; ctx.strokeStyle = col; ctx.lineWidth = 1 / z; ctx.setLineDash(h.weak ? [2 / z, 2 / z] : []);
      ctx.beginPath(); ctx.arc(n.ax, n.ay, (n.kids.length ? n.ar * z + 5 : 6.5) / z, 0, TAU); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = col; ctx.beginPath(); this.markShape(ctx, n, n.ax, n.ay, ms * 1.7); ctx.fill(); hitLabels.push({ n, col, text: `${i + 1}` });
    });
    const selLabels = [];
    if (sel) { this.drawLinks(ctx, sel, 1, selLabels); ctx.fillStyle = rgba(ink, 1); ctx.beginPath(); this.markShape(ctx, sel, sel.ax, sel.ay, ms * 1.8); ctx.fill(); this.crosshair(ctx, sel, rgba(ink, 1)); }
    if (this.mark && this.mark !== sel) { ctx.globalAlpha = 0.9; this.crosshair(ctx, this.mark, rgba(ink, 0.9)); ctx.globalAlpha = 1; }
    ctx.restore();

    // ---- screen layer: labels, greedy — no two overlap
    const placed = []; const fits = (x, y, w, h) => { for (const p of placed) if (x < p[0] + p[2] && x + w > p[0] && y < p[1] + p[3] && y + h > p[1]) return false; placed.push([x, y, w, h]); return true; };
    ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
    const label = (n, col, text, dy, force) => {
      const [sx, sy] = this.toScreen(n.ax, n.ay); const fs = 9.5; ctx.font = `${fs}px ${FONT}`; const w = ctx.measureText(text).width;
      const x = sx + ms * z + 4, y = sy + (dy || 3);
      if (!onScreen(x, y, w)) return; if (!force && !fits(x, y - fs, w, fs + 2)) return;
      ctx.fillStyle = col; ctx.fillText(text, x, y);
    };
    for (const L of selLabels) label(L.n, L.col, L.n.n);
    if (sel) label(sel, rgba(ink, 1), sel.n, -8, true);
    for (const L of hitLabels) { label(L.n, L.col, L.text, 3, true); }
    for (const L of hitLabels) { label(L.n, L.col, `   ${L.n.n}`, 3, false); }
    for (const L of routeLabels) label(L.n, L.col, L.n.t === 'module' ? L.n.p.split('/').pop() : L.n.n);
    for (const L of flashLabels) { ctx.globalAlpha = L.a; label(L.n, L.col, `${L.n.t === 'directory' ? this.dirName(L.n) : L.n.t === 'module' ? L.n.p.split('/').pop() : L.n.n} · ${L.text}`, -(L.n.ar * z) - 6, true); ctx.globalAlpha = 1; }
    for (const n of this.containers) {
      const rs = n.ar * z; const [sx, sy] = this.toScreen(n.ax, n.ay); if (!onScreen(sx, sy, rs)) continue;
      if (n.t === 'directory') {
        if (rs < 46 || n.thin) continue; const fs = 9.5; ctx.font = `${fs}px ${FONT}`; ctx.fillStyle = rgba(ink, 0.7 * n.ae);
        const name = this.dirName(n);
        if (name.length * fs * 0.62 < rs * 1.9) this.arcText(ctx, name, sx, sy, rs - fs - 3);
        continue;
      }
      const min = n.t === 'module' ? 22 : 18; if (rs < min) continue;
      const fs = 9.5, a = clamp((rs - min) / 14, 0, 1) * 0.85 * n.ae; ctx.font = `${fs}px ${FONT}`;
      const text = n.t === 'module' ? n.n + (n.p.endsWith('.py') ? '.py' : n.p.endsWith('.js') ? '.js' : '') : n.n;
      const w = ctx.measureText(text).width; const inside = w < rs * 1.5;
      const x = sx - w / 2, y = inside ? sy - rs + fs + 4 : sy - rs - 5;
      if (!fits(x, y - fs, w, fs + 2)) continue;
      ctx.fillStyle = rgba(n === sel ? ink : ink, n === sel ? 1 : a); ctx.fillText(text, x, y);
    }
    if (z > 5.5) {
      const la = clamp((z - 5.5) / 3, 0, 1) * 0.72; const fs = 9; ctx.font = `${fs}px ${FONT}`;
      for (const n of this.leaves) {
        if (n === sel || n === this.mark) continue; const [sx, sy] = this.toScreen(n.ax, n.ay); if (!onScreen(sx, sy, 60)) continue;
        const text = n.n.includes('.') ? n.n.slice(n.n.lastIndexOf('.') + 1) : n.n; const w = ctx.measureText(text).width;
        if (!fits(sx + ms * z + 4, sy - fs + 3, w, fs + 2)) continue;
        ctx.fillStyle = rgba(ink, la * n.ae); ctx.fillText(text, sx + ms * z + 4, sy + 3);
      }
    }
    // leader from the pinned/hovered node to its card
    if (sel && !this.note.hidden) {
      const nr = this.note.getBoundingClientRect(), cr = this.canvas.getBoundingClientRect(); const [sx, sy] = this.toScreen(sel.ax, sel.ay);
      const corners = [[nr.left, nr.top], [nr.right, nr.top], [nr.left, nr.bottom], [nr.right, nr.bottom]].map(([x, y]) => [x - cr.left, y - cr.top]);
      let best = corners[0], bd = Infinity; for (const c of corners) { const d = Math.hypot(c[0] - sx, c[1] - sy); if (d < bd) { bd = d; best = c; } }
      ctx.strokeStyle = rgba(ink, 0.7); ctx.lineWidth = 1; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(best[0], best[1]); ctx.stroke(); ctx.setLineDash([]);
    }
    // row tearing
    if (this.glitch) {
      const g = this.glitch, cv = this.canvas;
      for (const s of g.slices) ctx.drawImage(cv, 0, s.y * dpr, W * dpr, s.h * dpr, s.dx, s.y, W, s.h);
      if (g.tint) { ctx.save(); ctx.globalCompositeOperation = paper > 0.5 ? 'multiply' : 'lighter'; ctx.globalAlpha = 0.3; ctx.filter = 'hue-rotate(70deg)'; ctx.drawImage(cv, 3, 0, W, H); ctx.restore(); }
    }
  };

  // ---- interaction
  Pack.prototype.pick = function (sx, sy) {
    const z = this.cam.z; let best = null, bd = 9;
    for (const n of this.leaves) { const [x, y] = this.toScreen(n.ax, n.ay); const d = Math.hypot(x - sx, y - sy); if (d < bd) { bd = d; best = n; } }
    if (best) return best;
    let cont = null; const lim = Math.min(this.view.w, this.view.h) * 0.46;
    for (const n of this.containers) {
      const rs = n.ar * z; if (rs < 12 || rs > lim || n.t === 'directory') continue;
      const [x, y] = this.toScreen(n.ax, n.ay); if (Math.hypot(x - sx, y - sy) <= rs && (!cont || n.depth > cont.depth)) cont = n;
    }
    return cont;
  };
  Pack.prototype.onMove = function (e) {
    const r = this.canvas.getBoundingClientRect(); const sx = e.clientX - r.left, sy = e.clientY - r.top;
    if (this.drag) {
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      if (Math.hypot(dx, dy) > 4) this.dragged = true;
      this.panTarget = { x: this.drag.px + dx, y: this.drag.py + dy }; this.pan = { ...this.panTarget }; this.canvas.style.cursor = 'grabbing'; return;
    }
    this.mouse = [sx, sy]; const n = this.pick(sx, sy); this.canvas.style.cursor = n ? 'pointer' : 'default';
    if (n !== this.hover) { this.hover = n; this.updateNote(); }
  };
  Pack.prototype.onClick = function (e) {
    if (this.dragged) { this.dragged = false; return; }
    const r = this.canvas.getBoundingClientRect(); const n = this.pick(e.clientX - r.left, e.clientY - r.top);
    if (n) { this.pinned = this.pinned === n ? null : n; if (this.pinned && this.o.onSelect) this.o.onSelect(n); }
    else { this.pinned = null; if (this.hits) this.setHits(null); }
    this.updateNote();
  };
  Pack.prototype.updateNote = function () {
    const n = this.pinned || this.hover, note = this.note, v = this.view;
    if (!n) {
      if (!this.hits) { note.hidden = true; return; }
      note.innerHTML = `<div class="t">${esc(this.hitLabel || '')}</div>` + this.hits.map((h, i) => `<div class="hit"><i>${i + 1}</i><span>${esc(h.node.n)}${h.weak ? ' <em>weak</em>' : ''}</span></div>`).join('');
      note.hidden = false; note.style.left = (v.x0 + 10) + 'px'; note.style.top = (v.y0 + 44) + 'px'; return;
    }
    const cin = n.in.length, cout = n.out.length;
    note.innerHTML = `<div class="n">${esc(n.n)}</div><div class="t">${esc(n.t)} · ${esc(n.p)}${n.l && n.l[0] ? ':' + n.l[0] + '–' + n.l[1] : ''}</div>` +
      `<div class="s">${esc((n.s || '').slice(0, 300))}${(n.s || '').length > 300 ? '…' : ''}</div>` +
      `<div class="d"><i class="in">← ${cin} incoming</i><i class="out">→ ${cout} outgoing</i>${n.kids.length ? `<i style="opacity:.55">⊃ ${n.kids.length}</i>` : ''}${this.pinned === n ? '<i style="opacity:.5">pinned</i>' : ''}</div>`;
    note.hidden = false;
    const [sx, sy] = this.toScreen(n.ax, n.ay); const nw = note.offsetWidth || 300, nh = note.offsetHeight || 120;
    let x = sx + 18, y = sy + 18;
    if (x + nw > v.x0 + v.w - 8) x = sx - nw - 18; if (y + nh > v.y0 + v.h - 8) y = sy - nh - 18;
    note.style.left = clamp(x, v.x0 + 8, v.x0 + v.w - nw - 8) + 'px'; note.style.top = clamp(y, v.y0 + 8, v.y0 + v.h - nh - 8) + 'px';
  };
  Pack.prototype.setHits = function (hits, label) {
    this.hits = hits ? hits.map(h => ({ node: this.G.byId[h.id], weak: !!h.weak })).filter(h => h.node) : null; this.hitLabel = label;
    this.pinned = null; this.updateNote();
  };
  Pack.prototype.setShow = function (k, on) { this.show[k] = on; };

  window.Pack = Pack;
})();
