/* radial.js — FIG. 02: the containment tree on a ring, every CALLS / IMPORTS / INHERITS edge bundled through it. */
(function () {
  'use strict';
  const COL = { C: '#ff3aa8', I: '#8b7bff', N: '#e6ff3d', M: '#e6ff3d' };
  const NAME = { C: 'CALLS', I: 'IMPORTS', N: 'INHERITS', M: 'IMPLEMENTS' };
  const INK = '#ebe8e1', TAU = Math.PI * 2;
  const ORD = { directory: 0, document: 1, module: 2, class: 3, interface: 3, enum: 3, struct: 3, trait: 3, function: 4, method: 5 };
  const FONT = '"IBM Plex Mono", Menlo, monospace';
  const esc = (s) => String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function Radial(stage, G, o){
    this.stage = stage; this.G = G; this.o = Object.assign({ onSelect: null }, o || {});
    const c = this.canvas = document.createElement('canvas'); stage.appendChild(c);
    this.ctx = c.getContext('2d');
    this.base = document.createElement('canvas'); this.bctx = this.base.getContext('2d');
    this.note = document.createElement('div'); this.note.className = 'note'; this.note.hidden = true; stage.appendChild(this.note);
    this.hover = null; this.pinned = null; this.hits = null; this.show = { C: true, I: true, N: true, M: true };
    this.pulse = null; this.glitch = null; this.visible = false; this.nextGlitch = 0;
    this.buildTree();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(stage);
    c.addEventListener('mousemove', e => this.onMove(e));
    c.addEventListener('mouseleave', () => { if (this.hover) { this.hover = null; this.updateNote(); } });
    c.addEventListener('click', e => this.onClick(e));
    new IntersectionObserver(en => { this.visible = en[0].isIntersecting; }, { threshold: 0 }).observe(stage);
    const loop = (now) => { if (this.visible) this.frame(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  Radial.prototype.buildTree = function(){
    const G = this.G, root = this.root = G.byId['dir:.'];
    (function sort(n){ n.kids.sort((a, b) => (ORD[a.t] - ORD[b.t]) || a.n.localeCompare(b.n)); n.kids.forEach(sort); })(root);
    let maxD = 0;
    (function dep(n, d, anc){ n.depth = d; n.anc = [n, ...anc]; if (d > maxD) maxD = d; n.kids.forEach(k => dep(k, d + 1, n.anc)); })(root, 0, []);
    this.maxD = maxD;
    const slots = this.slots = [];
    (function walk(n){ if (n.t !== 'directory') { n.slot = slots.length; slots.push(n); } n.kids.forEach(walk); })(root);
    const N = slots.length, gap = 0.05, step = this.step = (TAU - gap) / N; this.a0 = -Math.PI / 2 + gap / 2;
    slots.forEach((n, i) => { n.a = this.a0 + (i + 0.5) * step; });
    (function span(n){
      let a0 = Infinity, a1 = -Infinity;
      if (n.slot !== undefined) { a0 = n.a - step / 2; a1 = n.a + step / 2; }
      n.kids.forEach(k => { span(k); a0 = Math.min(a0, k.a0); a1 = Math.max(a1, k.a1); });
      n.a0 = a0; n.a1 = a1; if (n.slot === undefined) n.a = (a0 + a1) / 2;
    })(root);
    G.nodes.forEach(n => { n.out = []; n.in = []; });
    this.edges = [];
    for (const e of G.edges) {
      const s = G.byId[e[0]], d = G.byId[e[1]]; if (!s || !d || s === d) continue;
      const E = { s, d, k: e[2], res: e[3] }; s.out.push(E); d.in.push(E); this.edges.push(E);
    }
    this.files = slots.filter(n => n.t === 'module' || n.t === 'document');
    this.dirs = G.nodes.filter(n => n.t === 'directory' && n.depth >= 2);
    this.counts = {}; for (const E of this.edges) this.counts[E.k] = (this.counts[E.k] || 0) + 1;
  };

  Radial.prototype.resize = function(){
    const r = this.stage.getBoundingClientRect(); const S = Math.floor(Math.min(r.width, r.height)); if (S < 50) return;
    const dpr = this.dpr = Math.min(2, devicePixelRatio || 1); this.S = S;
    for (const c of [this.canvas, this.base]) { c.width = S * dpr; c.height = S * dpr; c.style.width = S + 'px'; c.style.height = S + 'px'; }
    this.canvas.style.left = ((r.width - S) / 2) + 'px'; this.ox = (r.width - S) / 2;
    this.cx = S / 2; this.cy = S / 2; this.R = S / 2 - Math.max(92, S * 0.15);
    this.small = S < 640;
    this.layout(); this.drawBase();
  };

  Radial.prototype.layout = function(){
    const R = this.R, maxD = this.maxD;
    for (const n of this.G.nodes) {
      if (n.slot !== undefined) { n.x = R * Math.cos(n.a); n.y = R * Math.sin(n.a); }
      if (n.kids.length || n.t === 'directory') {
        const ri = R * Math.pow(n.depth / maxD, 1.15) * (n.t === 'directory' ? 0.85 : 0.92);
        n.xi = ri * Math.cos(n.a); n.yi = ri * Math.sin(n.a);
        if (n.slot === undefined) { n.x = n.xi; n.y = n.yi; }
      }
    }
    for (const E of this.edges) E.path = this.bundle(this.pathPoints(E.s, E.d), 0.86);
  };
  Radial.prototype.pathPoints = function(s, d){
    const set = new Set(s.anc); let l = null;
    for (const n of d.anc) if (set.has(n)) { l = n; break; }
    const up = [], down = [];
    for (const n of s.anc) { if (n === l) break; up.push(n); }
    for (const n of d.anc) { if (n === l) break; down.push(n); }
    const pts = [[s.x, s.y]];
    for (let i = 1; i < up.length; i++) pts.push([up[i].xi, up[i].yi]);
    pts.push([l.xi !== undefined ? l.xi : l.x, l.yi !== undefined ? l.yi : l.y]);
    for (let i = down.length - 1; i >= 1; i--) pts.push([down[i].xi, down[i].yi]);
    pts.push([d.x, d.y]);
    return pts;
  };
  // d3.curveBundle: straighten toward the chord, then a B-spline through the points
  Radial.prototype.bundle = function(pts, beta){
    const n = pts.length - 1, [x0, y0] = pts[0], [xn, yn] = pts[n];
    const q = pts.map((p, i) => [beta * p[0] + (1 - beta) * (x0 + i / n * (xn - x0)), beta * p[1] + (1 - beta) * (y0 + i / n * (yn - y0))]);
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
    return P;
  };

  Radial.prototype.mark = function(ctx, n, color, strong){
    ctx.fillStyle = color; ctx.strokeStyle = color; ctx.lineWidth = strong ? 1.2 : 0.8;
    const s = strong ? 1.8 : 1;
    if (n.t === 'module') ctx.fillRect(n.x - 2.4 * s, n.y - 2.4 * s, 4.8 * s, 4.8 * s);
    else if (n.t === 'document') { ctx.strokeRect(n.x - 2.2 * s, n.y - 2.2 * s, 4.4 * s, 4.4 * s); }
    else if (n.t === 'class' || n.t === 'interface' || n.t === 'struct' || n.t === 'enum' || n.t === 'trait') {
      ctx.beginPath(); ctx.moveTo(n.x, n.y - 3 * s); ctx.lineTo(n.x + 3 * s, n.y); ctx.lineTo(n.x, n.y + 3 * s); ctx.lineTo(n.x - 3 * s, n.y); ctx.closePath(); ctx.stroke();
    } else { ctx.beginPath(); ctx.arc(n.x, n.y, 1.15 * s, 0, TAU); ctx.fill(); }
  };
  Radial.prototype.radialText = function(ctx, text, a, r, align){
    ctx.save(); ctx.rotate(a); ctx.translate(r, 0);
    const flip = Math.cos(a) < 0; if (flip) ctx.rotate(Math.PI);
    ctx.textAlign = flip ? (align === 'out' ? 'right' : 'left') : (align === 'out' ? 'left' : 'right');
    ctx.fillText(text, 0, 3); ctx.restore();
  };
  Radial.prototype.arcText = function(ctx, text, a, r){
    ctx.save(); const cw = ctx.measureText('M').width; const total = text.length * cw / r;
    const up = Math.sin(a) > 0; // bottom half: flip so text reads left-to-right
    let ang = a - (up ? -total / 2 : total / 2);
    for (const ch of text) {
      ctx.save(); ctx.rotate(ang); ctx.translate(r, 0); ctx.rotate(up ? -Math.PI / 2 : Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText(ch, 0, 0); ctx.restore();
      ang += up ? -cw / r : cw / r;
    }
    ctx.restore();
  };

  Radial.prototype.drawBase = function(){
    const ctx = this.bctx, S = this.S, dpr = this.dpr, R = this.R;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, S, S); ctx.translate(this.cx, this.cy);
    ctx.lineWidth = 0.6; ctx.lineCap = 'round';
    for (const E of this.edges) {
      if (!this.show[E.k]) continue;
      ctx.strokeStyle = COL[E.k]; ctx.globalAlpha = E.k === 'C' ? 0.085 : E.k === 'I' ? 0.2 : 0.75; ctx.stroke(E.path);
    }
    ctx.globalAlpha = 1;
    for (const n of this.slots) this.mark(ctx, n, 'rgba(235,232,225,.72)', false);
    // file brackets + labels (greedy: biggest spans first, no two labels closer than 10px)
    const fs = this.small ? 8 : 9.5; ctx.font = `${fs}px ${FONT}`; ctx.fillStyle = 'rgba(235,232,225,.85)';
    const taken = []; const minGap = (fs + 1.5) / (R + 30);
    const files = [...this.files].sort((a, b) => (b.a1 - b.a0) - (a.a1 - a.a0));
    for (const f of files) {
      if (f.kids.length) { ctx.strokeStyle = 'rgba(235,232,225,.35)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(0, 0, R + 7, f.a0 + 0.002, f.a1 - 0.002); ctx.stroke(); }
      const ok = !taken.some(t => Math.abs(t - f.a) < minGap);
      if (ok) { taken.push(f.a); this.radialText(ctx, f.n === '__init__' ? f.n : f.n, f.a, R + 13, 'out'); }
      else { ctx.strokeStyle = 'rgba(235,232,225,.35)'; ctx.beginPath(); ctx.moveTo((R + 10) * Math.cos(f.a), (R + 10) * Math.sin(f.a)); ctx.lineTo((R + 13) * Math.cos(f.a), (R + 13) * Math.sin(f.a)); ctx.stroke(); }
      f.labeled = ok;
    }
    // directory arcs, nested outward by shallowness, names along the arc
    ctx.font = `${fs - 0.5}px ${FONT}`;
    for (const d of this.dirs) {
      const r = R + (this.small ? 60 : 78) + (5 - Math.min(5, d.depth)) * (this.small ? 7 : 9);
      const px = (d.a1 - d.a0) * r;
      ctx.strokeStyle = 'rgba(235,232,225,.28)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(0, 0, r, d.a0 + 0.003, d.a1 - 0.003); ctx.stroke();
      const name = d.p.split('/').pop();
      if (px > name.length * (fs - 1) + 14) { ctx.fillStyle = 'rgba(235,232,225,.7)'; this.arcText(ctx, name + '/', d.a, r - 4); }
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  };

  Radial.prototype.drawLinks = function(ctx, n, alpha, withLabels){
    ctx.lineWidth = 1.1; ctx.lineCap = 'round';
    const fs = this.small ? 8 : 9.5; ctx.font = `${fs}px ${FONT}`;
    const taken = []; const minGap = (fs + 1.5) / (this.R + 30);
    const draw = (E, col, other) => {
      if (!this.show[E.k]) return;
      ctx.globalAlpha = alpha * 0.9; ctx.strokeStyle = col; ctx.stroke(E.path);
      ctx.globalAlpha = alpha; this.mark(ctx, other, col, true);
      if (withLabels && !taken.some(t => Math.abs(t - other.a) < minGap)) { taken.push(other.a); ctx.fillStyle = col; this.radialText(ctx, other.n, other.a, this.R + 13, 'out'); }
    };
    for (const E of n.in) draw(E, E.k === 'C' ? COL.C : E.k === 'I' ? COL.I : COL.N, E.s);
    for (const E of n.out) draw(E, E.k === 'C' ? '#4d86ff' : E.k === 'I' ? COL.I : COL.N, E.d);
    ctx.globalAlpha = alpha; this.mark(ctx, n, INK, true);
    ctx.globalAlpha = 1;
  };
  Radial.prototype.crosshair = function(ctx, n, col){
    ctx.strokeStyle = col || INK; ctx.lineWidth = 1; const s = 9;
    ctx.strokeRect(n.x - s, n.y - s, 2 * s, 2 * s);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.beginPath(); ctx.moveTo(n.x + dx * s, n.y + dy * s); ctx.lineTo(n.x + dx * (s + 6), n.y + dy * (s + 6)); ctx.stroke(); }
  };

  Radial.prototype.frame = function(now){
    const focus = this.pinned || this.hover || this.hits;
    if (!focus) {
      if (!this.pulse || now - this.pulse.t0 > 3400) {
        const cand = this.slots[Math.floor(Math.random() * this.slots.length)];
        if (cand.in.length + cand.out.length >= 3) this.pulse = { n: cand, t0: now };
      }
    } else this.pulse = null;
    if (!this.glitch && now > this.nextGlitch) {
      const S = this.S, k = 2 + Math.floor(Math.random() * 4), slices = [];
      for (let i = 0; i < k; i++) slices.push({ y: Math.random() * S, h: 6 + Math.random() * 40, dx: (Math.random() - 0.5) * 36 });
      this.glitch = { t0: now, dur: 70 + Math.random() * 110, slices, tint: Math.random() < 0.5 };
      this.nextGlitch = now + 2500 + Math.random() * 6000;
    }
    if (this.glitch && now - this.glitch.t0 > this.glitch.dur) this.glitch = null;
    this.render(now);
  };

  Radial.prototype.render = function(now){
    const ctx = this.ctx, S = this.S, dpr = this.dpr, R = this.R;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, S, S);
    const sel = this.pinned || this.hover; const focus = sel || this.hits;
    ctx.globalAlpha = focus ? 0.22 : 1;
    if (this.glitch) {
      const g = this.glitch;
      ctx.drawImage(this.base, 0, 0, S, S);
      for (const s of g.slices) ctx.drawImage(this.base, 0, s.y * dpr, S * dpr, s.h * dpr, s.dx, s.y, S, s.h);
      if (g.tint) { ctx.save(); ctx.globalCompositeOperation = 'lighter'; ctx.globalAlpha *= 0.35; ctx.filter = 'hue-rotate(60deg)'; ctx.drawImage(this.base, 3, 0, S, S); ctx.restore(); }
    } else ctx.drawImage(this.base, 0, 0, S, S);
    ctx.globalAlpha = 1; ctx.translate(this.cx, this.cy);
    if (this.pulse && !focus) {
      const u = (now - this.pulse.t0) / 2200;
      if (u < 1) this.drawLinks(ctx, this.pulse.n, Math.sin(u * Math.PI) * 0.55, false);
    }
    if (this.hits) this.drawHits(ctx);
    if (sel) { this.drawLinks(ctx, sel, 1, true); this.crosshair(ctx, sel); this.leader(ctx, sel); }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  };
  Radial.prototype.drawHits = function(ctx){
    const fs = this.small ? 8 : 9.5; ctx.font = `${fs}px ${FONT}`;
    this.hits.forEach((h, i) => {
      const n = h.node, col = h.weak ? 'rgba(255,58,168,.55)' : COL.C;
      ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.setLineDash(h.weak ? [2, 2] : []);
      ctx.beginPath(); ctx.arc(n.x, n.y, 6.5, 0, TAU); ctx.stroke(); ctx.setLineDash([]);
      this.mark(ctx, n, col, true);
      // rank + name inside the ring, along the radius
      ctx.fillStyle = col;
      const r1 = this.R - 12 - (i % 2) * 11;
      ctx.beginPath(); ctx.moveTo(n.x * (this.R - 7) / this.R, n.y * (this.R - 7) / this.R); ctx.lineTo(r1 * Math.cos(n.a), r1 * Math.sin(n.a)); ctx.stroke();
      this.radialText(ctx, `${i + 1}`, n.a, r1 - 3, 'in');
    });
  };
  Radial.prototype.leader = function(ctx, n){
    if (this.note.hidden) return;
    const nr = this.note.getBoundingClientRect(), cr = this.canvas.getBoundingClientRect();
    const corners = [[nr.left, nr.top], [nr.right, nr.top], [nr.left, nr.bottom], [nr.right, nr.bottom]]
      .map(([x, y]) => [x - cr.left - this.cx, y - cr.top - this.cy]);
    let best = corners[0], bd = Infinity;
    for (const c of corners) { const d = Math.hypot(c[0] - n.x, c[1] - n.y); if (d < bd) { bd = d; best = c; } }
    ctx.strokeStyle = 'rgba(235,232,225,.7)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(n.x, n.y); ctx.lineTo(best[0], best[1]); ctx.stroke(); ctx.setLineDash([]);
  };

  Radial.prototype.pick = function(e){
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left - this.cx, y = e.clientY - r.top - this.cy;
    const d = Math.hypot(x, y); if (Math.abs(d - this.R) > 16) return null;
    let a = Math.atan2(y, x); while (a < this.a0) a += TAU; while (a >= this.a0 + TAU) a -= TAU;
    const i = Math.floor((a - this.a0) / this.step); let best = null, bd = 12;
    for (let k = i - 2; k <= i + 2; k++) { const n = this.slots[(k + this.slots.length) % this.slots.length]; const dd = Math.hypot(n.x - x, n.y - y); if (dd < bd) { bd = dd; best = n; } }
    return best;
  };
  Radial.prototype.onMove = function(e){
    const n = this.pick(e); this.canvas.style.cursor = n ? 'pointer' : 'default';
    if (n !== this.hover) { this.hover = n; this.updateNote(); }
  };
  Radial.prototype.onClick = function(e){
    const n = this.pick(e);
    if (n) { this.pinned = this.pinned === n ? null : n; if (this.pinned && this.o.onSelect) this.o.onSelect(n); }
    else { this.pinned = null; if (this.hits) this.setHits(null); }
    this.updateNote();
  };
  Radial.prototype.updateNote = function(){
    const n = this.pinned || this.hover; const note = this.note;
    if (!n) {
      if (!this.hits) { note.hidden = true; return; }
      note.innerHTML = `<div class="t">${esc(this.hitLabel || '')}</div>` + this.hits.map((h, i) =>
        `<div class="hit"><i>${i + 1}</i><span>${esc(h.node.n)}${h.weak ? ' <em>weak</em>' : ''}</span></div>`).join('');
      note.hidden = false; note.style.left = (this.ox + 8) + 'px'; note.style.right = ''; note.style.top = '8px';
      return;
    }
    const cin = n.in.length, cout = n.out.length;
    note.innerHTML = `<div class="n">${esc(n.n)}</div><div class="t">${esc(n.t)} · ${esc(n.p)}${n.l && n.l[0] ? ':' + n.l[0] + '–' + n.l[1] : ''}</div>` +
      `<div class="s">${esc((n.s || '').slice(0, 300))}${(n.s || '').length > 300 ? '…' : ''}</div>` +
      `<div class="d"><i class="in">← ${cin} incoming</i><i class="out">→ ${cout} outgoing</i>${this.pinned === n ? '<i style="opacity:.5">pinned</i>' : ''}</div>`;
    note.hidden = false;
    const onRight = n.x > 0; note.style.left = onRight ? (this.ox + 8) + 'px' : ''; note.style.right = onRight ? '' : '8px';
    note.style.top = (n.y > 0 ? 8 : Math.max(8, this.S - note.offsetHeight - 8)) + 'px';
  };
  Radial.prototype.setHits = function(hits, label){
    this.hits = hits ? hits.map(h => ({ node: this.G.byId[h.id], weak: !!h.weak })).filter(h => h.node) : null; this.hitLabel = label;
    this.pinned = null; this.updateNote();
  };
  Radial.prototype.setShow = function(k, on){ this.show[k] = on; this.drawBase(); };

  window.Radial = Radial;
})();
