/* morph.js — FIG. 01: a source file dissolves into the nodes and edges the parser makes of it. */
(function () {
  'use strict';
  const SCRAMBLE = '|/\\-_=+*#%:;<>[]{}';
  const rnd = (i) => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };
  const ease = (u) => u < 0.5 ? 4*u*u*u : 1 - Math.pow(-2*u + 2, 3)/2;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function Morph(stage, G, src, o){
    this.stage = stage; this.G = G; this.o = o || {};
    const c = this.canvas = document.createElement('canvas'); stage.appendChild(c);
    this.ctx = c.getContext('2d');
    this.lines = src.split('\n');
    this.t = 0; this.target = 0; this.replay = null;
    this.prepare();
    this.resize();
    new ResizeObserver(() => this.resize()).observe(stage);
    window.addEventListener('scroll', () => this.onScroll(), { passive: true });
    stage.addEventListener('click', () => { this.replay = performance.now(); });
    this.visible = false;
    new IntersectionObserver(en => { this.visible = en[0].isIntersecting; }, { threshold: 0 }).observe(stage);
    const loop = (now) => { if (this.visible) this.frame(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
    this.onScroll();
  }

  // Work out what every line of the excerpt turns into.
  Morph.prototype.prepare = function(){
    const G = this.G, gb = G.byId[this.o.moduleId];
    const importTargets = new Map(); // id -> node
    for (const e of G.edges) if (e[0] === gb.id && e[2] === 'I') { const n = G.byId[e[1]]; if (n) importTargets.set(n.id, n); }
    const kids = new Map(); for (const k of gb.kids) kids.set(k.n, k);

    const items = []; // {line, kind, tok:[start,end], node, ext}
    let pend = null; // multi-line "from x import ("
    this.lines.forEach((line, li) => {
      let m;
      if (pend) {
        if (line.trim() === ')') { pend = null; items.push({ li, kind: 'fade' }); return; }
        const name = line.trim().replace(',', '');
        const id = pend + '::' + name; const node = importTargets.get(id) || importTargets.get(pend);
        const s = line.indexOf(name);
        items.push({ li, kind: node ? 'import' : 'fade', tok: [s, s + name.length], node });
        return;
      }
      if ((m = /^from (inflorescence\.[\w.]+) import (.*)$/.exec(line))) {
        const file = 'src/' + m[1].replace(/\./g, '/') + '.py';
        if (m[2].trim() === '(') { pend = file; items.push({ li, kind: 'fade' }); return; }
        const names = m[2].split(',').map(s => s.trim()).filter(Boolean);
        // several names on one line: each name is its own token
        const toks = [];
        for (const name of names) {
          const node = importTargets.get(file + '::' + name) || importTargets.get(file);
          const s = line.indexOf(name, line.indexOf('import'));
          if (node) toks.push({ tok: [s, s + name.length], node });
        }
        items.push({ li, kind: toks.length ? 'import' : 'fade', toks });
        return;
      }
      if ((m = /^(?:from|import) (\w+)/.exec(line)) && m[1] !== 'inflorescence' && m[1] !== '__future__') {
        const s = line.indexOf(m[1]); items.push({ li, kind: 'ext', tok: [s, s + m[1].length], ext: m[1] }); return;
      }
      if ((m = /^(?:async )?(?:def|class) (\w+)/.exec(line))) {
        const node = kids.get(m[1]); const s = line.indexOf(m[1]);
        items.push({ li, kind: node ? 'def' : 'fade', tok: [s, s + m[1].length], node }); return;
      }
      items.push({ li, kind: 'fade' });
    });
    this.items = items;
    // visible graph
    const imports = [...new Set(items.flatMap(it => it.kind === 'import' ? (it.toks ? it.toks.map(t => t.node) : [it.node]) : []))];
    const defs = items.filter(it => it.kind === 'def').map(it => it.node);
    const exts = [...new Set(items.filter(it => it.kind === 'ext').map(it => it.ext))];
    // group imports by file, keep file order of first appearance
    const groups = []; const gmap = new Map();
    for (const n of imports) { const f = n.t === 'module' ? n.p : n.p; if (!gmap.has(f)) { gmap.set(f, []); groups.push([f, gmap.get(f)]); } gmap.get(f).push(n); }
    this.imports = imports; this.defs = defs; this.exts = exts; this.groups = groups; this.gb = gb;
    const shown = new Set(defs.map(d => d.id));
    this.calls = G.edges.filter(e => e[2] === 'C' && shown.has(e[0]) && shown.has(e[1])).map(e => [G.byId[e[0]], G.byId[e[1]]]);
  };

  Morph.prototype.resize = function(){
    const r = this.stage.getBoundingClientRect(); const dpr = this.dpr = Math.min(2, devicePixelRatio || 1);
    this.W = r.width; this.H = r.height;
    this.canvas.width = r.width * dpr; this.canvas.height = r.height * dpr;
    this.canvas.style.width = r.width + 'px'; this.canvas.style.height = r.height + 'px';
    this.layout();
  };

  Morph.prototype.layout = function(){
    const W = this.W, H = this.H, ctx = this.ctx;
    const fs = W < 720 ? 9.5 : 11; this.fs = fs; this.lh = fs * 1.16;
    ctx.font = `${fs}px "IBM Plex Mono", Menlo, monospace`;
    const cw = this.cw = ctx.measureText('M').width;
    const x0 = 6, y0 = 10;
    const maxCols = Math.floor((W * (W < 720 ? 0.98 : 0.6)) / cw);
    // graph positions
    const pos = this.pos = new Map();
    const M = [W * 0.5, H * 0.52]; pos.set(this.gb.id, M);
    const rightX = this.rightX = W < 720 ? W * 0.72 : W * 0.78;
    const nImp = this.imports.length;
    let row = 0; const top = H * 0.06, bottom = H * 0.90;
    this.groupSpans = [];
    for (const [file, nodes] of this.groups) {
      const s = row;
      row += 0.9;
      for (const n of nodes) { pos.set(n.id, [rightX, top + (row + 0.5) / (nImp + this.groups.length * 0.9) * (bottom - top)]); row++; }
      this.groupSpans.push({ file, y0: pos.get(nodes[0].id)[1], y1: pos.get(nodes[nodes.length - 1].id)[1] });
    }
    const leftX = W < 720 ? W * 0.12 : W * 0.24; const nd = this.defs.length;
    this.defs.forEach((n, i) => pos.set(n.id, [leftX, H * 0.16 + (i + 0.5) / nd * H * 0.66]));
    const ne = this.exts.length;
    this.exts.forEach((e, i) => pos.set('ext:' + e, [W * 0.22 + (i + 0.5) / ne * W * 0.52, H * (i % 2 ? 0.985 : 0.945)]));

    // particles: one per character
    const P = this.parts = [];
    const labelStart = (id, text) => { const p = pos.get(id); return [p[0] + 8, p[1]]; };
    this.items.forEach((it) => {
      const line = this.lines[it.li]; const ly = y0 + it.li * this.lh + fs;
      const toks = it.toks || (it.tok ? [{ tok: it.tok, node: it.node, ext: it.ext }] : []);
      const nCols = Math.min(line.length, maxCols);
      // which char indexes belong to a token
      const owner = new Array(nCols).fill(null);
      toks.forEach((tk, ti) => { for (let i = tk.tok[0]; i < Math.min(tk.tok[1], nCols); i++) owner[i] = tk; });
      const edgeChars = []; // chars that will be laid along the edge
      for (let i = 0; i < nCols; i++) {
        const ch = line[i]; if (ch === ' ') continue;
        const sx = x0 + i * cw, sy = ly;
        const p = { ch, sx, sy, ex: sx, ey: sy, a: 1, kind: it.kind, seed: it.li * 131 + i, node: null };
        const tk = owner[i];
        if (tk && (it.kind === 'import' || it.kind === 'def' || it.kind === 'ext')) {
          const id = tk.node ? tk.node.id : 'ext:' + tk.ext;
          const [lx, lyy] = labelStart(id); const k = i - tk.tok[0];
          p.ex = lx + k * cw; p.ey = lyy + fs * 0.35; p.a = 1; p.kind = 'label'; p.node = id;
        } else if (it.kind === 'fade' || !toks.length) {
          p.ex = sx + (rnd(p.seed) - 0.5) * 90; p.ey = sy + 30 + rnd(p.seed + 1) * 120; p.a = 0; p.kind = 'fade';
        } else { edgeChars.push(p); }
        P.push(p);
      }
      if (edgeChars.length && toks.length) {
        // spread the rest of the line along the edge(s) from the module to the token node(s)
        const per = Math.ceil(edgeChars.length / toks.length);
        edgeChars.forEach((p, k) => {
          const tk = toks[Math.min(toks.length - 1, Math.floor(k / per))];
          const id = tk.node ? tk.node.id : 'ext:' + tk.ext; const q = pos.get(id);
          const u = ((k % per) + 0.5) / per; const j = (rnd(p.seed + 3) - 0.5) * 3;
          const dx = q[0] - M[0], dy = q[1] - M[1], L = Math.hypot(dx, dy) || 1;
          p.ex = M[0] + dx * u - dy / L * j; p.ey = M[1] + dy * u + dx / L * j;
          p.a = 0.28; p.kind = 'edge';
        });
      }
    });
    this.parts.forEach(p => { p.delay = 0.08 + 0.42 * rnd(p.seed + 7) + 0.25 * (p.sy / this.H); p.bend = (rnd(p.seed + 9) - 0.5) * 120; });
  };

  Morph.prototype.onScroll = function(){
    const r = this.stage.getBoundingClientRect(); const vh = innerHeight;
    this.target = clamp((vh * 0.88 - r.top) / (r.height * 0.85), 0, 1);
  };

  Morph.prototype.frame = function(now){
    if (this.replay !== null) {
      const u = (now - this.replay) / 2600; this.t = u < 0.15 ? 0 : clamp((u - 0.15) / 0.85, 0, 1);
      if (u >= 1) this.replay = null;
    } else this.t += (this.target - this.t) * 0.08;
    this.draw(now);
  };

  Morph.prototype.draw = function(now){
    const ctx = this.ctx, dpr = this.dpr, W = this.W, H = this.H, t = this.t, fs = this.fs;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const ink = '#111';
    ctx.font = `${fs}px "IBM Plex Mono", Menlo, monospace`; ctx.textBaseline = 'alphabetic';
    // edges under everything, fading in late
    const g = clamp((t - 0.55) / 0.45, 0, 1);
    if (g > 0) {
      const M = this.pos.get(this.gb.id);
      ctx.lineWidth = 0.8;
      for (const n of this.imports) { const q = this.pos.get(n.id); ctx.strokeStyle = `rgba(90,70,220,${0.55*g})`; ctx.beginPath(); ctx.moveTo(M[0], M[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); }
      ctx.setLineDash([2, 3]);
      for (const n of this.defs) { const q = this.pos.get(n.id); ctx.strokeStyle = `rgba(17,17,17,${0.5*g})`; ctx.beginPath(); ctx.moveTo(M[0], M[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); }
      for (const e of this.exts) { const q = this.pos.get('ext:' + e); ctx.strokeStyle = `rgba(17,17,17,${0.3*g})`; ctx.beginPath(); ctx.moveTo(M[0], M[1]); ctx.lineTo(q[0], q[1]); ctx.stroke(); }
      ctx.setLineDash([]);
      ctx.strokeStyle = `rgba(255,58,168,${0.85*g})`; ctx.lineWidth = 0.9;
      for (const [a, b] of this.calls) { const p = this.pos.get(a.id), q = this.pos.get(b.id); ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.bezierCurveTo(p[0] - 60, p[1], q[0] - 60, q[1], q[0], q[1]); ctx.stroke(); }
      // marks
      const mark = (p, kind) => {
        ctx.fillStyle = ink; ctx.strokeStyle = ink; ctx.lineWidth = 1;
        if (kind === 'module') { ctx.fillRect(p[0] - 4, p[1] - 4, 8, 8); }
        else if (kind === 'class') { ctx.beginPath(); ctx.moveTo(p[0], p[1] - 4); ctx.lineTo(p[0] + 4, p[1]); ctx.lineTo(p[0], p[1] + 4); ctx.lineTo(p[0] - 4, p[1]); ctx.closePath(); ctx.stroke(); }
        else if (kind === 'ext') { ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, 7); ctx.stroke(); }
        else { ctx.beginPath(); ctx.arc(p[0], p[1], 2.2, 0, 7); ctx.fill(); }
      };
      ctx.globalAlpha = g;
      mark(this.pos.get(this.gb.id), 'module');
      for (const n of this.imports) mark(this.pos.get(n.id), n.t);
      for (const n of this.defs) mark(this.pos.get(n.id), n.t);
      for (const e of this.exts) mark(this.pos.get('ext:' + e), 'ext');
      // module label + group brackets with file names
      ctx.fillStyle = ink; ctx.textAlign = 'center';
      ctx.fillText(this.gb.n + '.py', this.pos.get(this.gb.id)[0], this.pos.get(this.gb.id)[1] - 12);
      ctx.font = `${fs * 0.82}px "IBM Plex Mono", Menlo, monospace`; ctx.textAlign = 'left'; const rightX = this.rightX;
      const bx = (W < 720 ? W * 0.72 : W * 0.78) + this.cw * 24;
      for (const gs of this.groupSpans) {
        ctx.strokeStyle = `rgba(17,17,17,.5)`; ctx.beginPath(); ctx.moveTo(bx, gs.y0 - 5); ctx.lineTo(bx + 4, gs.y0 - 5); ctx.lineTo(bx + 4, gs.y1 + 5); ctx.lineTo(bx, gs.y1 + 5); ctx.stroke();
        ctx.fillStyle = `rgba(17,17,17,.55)`; ctx.fillText(gs.file.replace('src/inflorescence/', ''), rightX + 8, gs.y0 - fs * 0.85);
      }
      ctx.textAlign = 'left'; ctx.fillStyle = `rgba(17,17,17,.6)`;
      ctx.fillText('IMPORTS', bx - this.cw * 4, H * 0.045);
      ctx.fillText('CONTAINS', (W < 720 ? W * 0.12 : W * 0.24) - 6, H * 0.13);
      ctx.fillText('External', W * 0.22 - 6, H * 0.945 - fs * 1.2);
      ctx.globalAlpha = 1;
      ctx.font = `${fs}px "IBM Plex Mono", Menlo, monospace`;
    }
    // characters
    const fr = Math.floor(now / 50);
    for (const p of this.parts) {
      const u = ease(clamp((t * 1.35 - p.delay) / 0.55, 0, 1));
      const mx = (p.sx + p.ex) / 2 + p.bend * (p.ey - p.sy) / 200, my = (p.sy + p.ey) / 2 - p.bend * (p.ex - p.sx) / 200;
      const x = (1 - u) * (1 - u) * p.sx + 2 * (1 - u) * u * mx + u * u * p.ex;
      const y = (1 - u) * (1 - u) * p.sy + 2 * (1 - u) * u * my + u * u * p.ey;
      const a = 1 + (p.a - 1) * u;
      if (a <= 0.02) continue;
      let ch = p.ch;
      if (u > 0.04 && u < 0.96 && rnd(p.seed + fr) < 0.35) ch = SCRAMBLE[Math.floor(rnd(p.seed * 3 + fr) * SCRAMBLE.length)];
      if (p.kind === 'edge' && u > 0.96) ch = '·';
      ctx.fillStyle = p.kind === 'label' && u > 0.9 ? ink : `rgba(17,17,17,${a})`;
      if (p.kind === 'label' && u > 0.5 && u < 0.96) ctx.fillStyle = `rgba(255,58,168,${a})`;
      ctx.fillText(ch, x, y);
    }
  };

  window.Morph = Morph;
})();
