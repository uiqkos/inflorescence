/* wordflower.js — FIG. 03: a flower silhouette typeset from a node's own summary. */
(function () {
  'use strict';
  const SCR = 'abcdefghijklmnopqrstuvwxyz0123456789_';
  const rnd = (i) => { const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); };

  function WordFlower(canvas, maskSrc, o){
    this.c = canvas; this.ctx = canvas.getContext('2d'); this.o = o || {};
    this.words = []; this.placed = []; this.t0 = 0; this.mouse = null; this.mask = null;
    const img = new Image(); img.crossOrigin = 'anonymous';
    img.onload = () => { this.img = img; this.resize(); };
    img.src = maskSrc;
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener('mousemove', e => { const r = canvas.getBoundingClientRect(); this.mouse = [e.clientX - r.left, e.clientY - r.top]; });
    canvas.addEventListener('mouseleave', () => { this.mouse = null; });
    this.visible = false;
    new IntersectionObserver(en => { this.visible = en[0].isIntersecting; }, { threshold: 0 }).observe(canvas);
    const loop = (now) => { if (this.visible) this.draw(now); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }
  WordFlower.prototype.resize = function(){
    const S = Math.floor(this.c.clientWidth); if (!S || !this.img) return;
    const dpr = this.dpr = Math.min(2, devicePixelRatio || 1);
    this.S = S; this.c.width = S * dpr; this.c.height = S * dpr;
    // mask at css resolution
    const m = document.createElement('canvas'); m.width = S; m.height = S;
    const mx = m.getContext('2d'); mx.drawImage(this.img, 0, 0, S, S);
    const d = mx.getImageData(0, 0, S, S).data; const mask = this.mask = new Uint8Array(S * S);
    for (let i = 0; i < S * S; i++) { const l = 0.299 * d[i*4] + 0.587 * d[i*4+1] + 0.114 * d[i*4+2]; mask[i] = l > 88 ? 1 : 0; }
    this.layout();
  };
  WordFlower.prototype.setText = function(text, meta){
    this.text = text; this.meta = meta || {};
    this.words = text.replace(/[`*]/g, '').split(/\s+/).filter(w => w.length > 1);
    this.t0 = performance.now(); this.layout();
  };
  WordFlower.prototype.layout = function(){
    if (!this.mask || !this.words.length) return;
    const S = this.S, ctx = this.ctx, mask = this.mask, placed = this.placed = [];
    let wi = 0; const lh = S / 50; let row = 0;
    for (let y = lh; y < S - lh * 0.5; y += lh, row++) {
      const fs = lh * (0.78 + 0.22 * rnd(row * 7 + 1)); ctx.font = `${fs}px "IBM Plex Mono", Menlo, monospace`;
      const yy = Math.floor(y); const off = Math.floor(rnd(row * 3) * 8);
      let x = off;
      while (x < S) {
        // find start of a run
        while (x < S && !mask[yy * S + x]) x++;
        if (x >= S) break;
        let e = x; while (e < S && mask[yy * S + e]) e++;
        // fill run
        let cx = x + 2;
        while (true) {
          const w = this.words[wi % this.words.length]; const ww = ctx.measureText(w).width;
          if (cx + ww > e - 2) break;
          placed.push({ w, x: cx, y: yy, fs, ww, i: placed.length, delay: (yy / S) * 900 + rnd(placed.length) * 260 });
          wi++; cx += ww + fs * 0.6;
        }
        x = e + 1;
      }
    }
  };
  WordFlower.prototype.draw = function(now){
    const ctx = this.ctx, dpr = this.dpr, S = this.S; if (!S) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, S, S);
    const el = now - this.t0; const fr = Math.floor(now / 60); const m = this.mouse;
    ctx.textBaseline = 'alphabetic';
    for (const p of this.placed) {
      const u = Math.min(1, Math.max(0, (el - p.delay) / 420));
      if (u <= 0) continue;
      let w = p.w; let col = `rgba(235,232,225,${0.55 + 0.4 * u})`;
      let near = false;
      if (m) { const d = Math.hypot(m[0] - (p.x + p.ww / 2), m[1] - p.y); near = d < S * 0.11; }
      if (u < 1 || near) {
        const k = u < 1 ? Math.floor(w.length * (1 - u)) : 1 + Math.floor(rnd(p.i + fr) * 2);
        let s = w.split('');
        for (let j = 0; j < k; j++) { const idx = (p.i * 7 + j * 3 + fr) % w.length; s[idx] = SCR[Math.floor(rnd(p.i + fr + j) * SCR.length)]; }
        w = s.join('');
        if (near) col = '#ff3aa8';
      }
      ctx.font = `${p.fs}px "IBM Plex Mono", Menlo, monospace`; ctx.fillStyle = col;
      ctx.fillText(w, p.x, p.y);
    }
    // specimen caption inside the canvas, bottom-left
    if (this.meta && this.meta.name) {
      ctx.font = `${Math.max(9, S / 64)}px "IBM Plex Mono", Menlo, monospace`; ctx.fillStyle = 'rgba(235,232,225,.6)';
      ctx.fillText(this.meta.name, 4, S - 6);
    }
  };
  window.WordFlower = WordFlower;
})();
