/* typelab.js — experimental treatments of the one serif word. A switcher in the hero corner cycles them;
   pick one, then delete the switcher. Shader modes draw the word into an RGBA canvas that dirt.js
   composites *before* the dots / tears / ASCII, so the type gets exactly the flower's dirt. */
(function () {
  'use strict';
  const TEXT = 'inflorescence';
  const VARIANTS = [
    { id: 'A', name: 'Instrument Serif italic · через шейдер', cls: 'v-instrument', mode: 'shader' },
    { id: 'B', name: 'Fraunces italic · wonk · плавится', cls: 'v-fraunces', mode: 'css', melt: 1, slices: 1 },
    { id: 'C', name: 'Doto · точечная матрица · через шейдер', cls: 'v-doto', mode: 'shader' },
    { id: 'D', name: 'Jacquarda Bastarda 9 · пиксельная бастарда · срезы', cls: 'v-bastarda', mode: 'css', slices: 1, stair: 1 },
    { id: 'E', name: 'Sixtyfour · CRT bleed / scan', cls: 'v-sixtyfour', mode: 'css', bleed: 1, slices: 1 },
    { id: 'F', name: 'ботанический леттеринг · сгенерирован · через шейдер', cls: 'v-instrument', mode: 'image', src: 'img/title-botanical.jpg', dirt: 0.25 },
    { id: 'G', name: 'жидкий хром · сгенерирован · через шейдер', cls: 'v-instrument', mode: 'image', src: 'img/title-chrome.jpg', dirt: 0.6 },
    { id: 'H', name: 'гербарий из лепестков · сгенерирован · через шейдер', cls: 'v-instrument', mode: 'image', src: 'img/title-herbarium.jpg', dirt: 0.45 },
  ];
  const rnd = () => Math.random();

  function TypeLab(hero, h1, host){
    this.hero = hero; this.h1 = h1; this.host = host; this.title = h1.parentElement;
    this.ov = document.createElement('canvas'); this.images = {};
    this.slices = [0, 1, 2].map(i => { const s = document.createElement('div'); s.className = 'slice'; s.textContent = TEXT; this.title.appendChild(s); return s; });
    this.melt = document.getElementById('meltmap'); this.turb = document.getElementById('meltturb');
    this.lastBurst = 0; this.sliceSeed = [];
    // switcher
    const ui = this.ui = document.createElement('div'); ui.className = 'typelab';
    ui.innerHTML = `<span class="k">TYPE</span>` + VARIANTS.map(v => `<b data-id="${v.id}">${v.id}</b>`).join('') + `<span class="n"></span>`;
    host.appendChild(ui);
    ui.addEventListener('click', e => { const b = e.target.closest('b'); if (b) this.apply(b.dataset.id); });
    addEventListener('keydown', e => { const i = 'abcdefgh'.indexOf(e.key.toLowerCase()); if (i >= 0 && !e.metaKey && !e.ctrlKey) this.apply(VARIANTS[i].id); });
    let saved = null; try { saved = localStorage.getItem('inflo-type'); } catch (e) {}
    this.apply(saved && VARIANTS.some(v => v.id === saved) ? saved : 'A');
    new ResizeObserver(() => this.redraw()).observe(hero.c);
    document.fonts.addEventListener('loadingdone', () => this.redraw());
    const loop = () => { this.frame(); requestAnimationFrame(loop); }; requestAnimationFrame(loop);
  }

  TypeLab.prototype.apply = function(id){
    const v = VARIANTS.find(x => x.id === id); if (!v) return; this.v = v;
    try { localStorage.setItem('inflo-type', id); } catch (e) {}
    this.ui.querySelectorAll('b').forEach(b => b.classList.toggle('on', b.dataset.id === id));
    this.ui.querySelector('.n').textContent = v.name;
    VARIANTS.forEach(x => { this.h1.classList.remove(x.cls); this.slices.forEach(s => s.classList.remove(x.cls)); });
    this.h1.classList.add(v.cls); this.slices.forEach(s => s.classList.add(v.cls));
    this.h1.classList.toggle('ghost', v.mode !== 'css');
    this.h1.classList.toggle('plain', v.mode !== 'css');
    this.h1.style.filter = v.melt ? 'url(#melt)' : '';
    this.hero.overDirt = v.dirt === undefined ? 1 : v.dirt;
    this.slices.forEach(s => { s.style.opacity = 0; });
    this.redraw();
  };

  // ---- shader / image modes: draw the word into the overlay canvas at the h1's own place
  TypeLab.prototype.redraw = function(){
    const v = this.v, hero = this.hero; if (!v) return;
    if (v.mode === 'css') { hero.clearOverlay(); return; }
    const c = this.ov, dpr = hero.dpr; c.width = hero.c.width; c.height = hero.c.height;
    const x = c.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, c.width, c.height);
    const r = this.h1.getBoundingClientRect(), hr = hero.c.getBoundingClientRect();
    if (v.mode === 'image') {
      this.loadImage(v.src).then(img => {
        if (this.v !== v) return;
        const W = hero.c.clientWidth; const w = Math.min(W - 44, W * 0.56); const h = w * img.height / img.width;
        x.drawImage(img, r.left - hr.left, r.bottom - hr.top - h + 10, w, h);
        hero.setOverlay(c);
      }).catch(() => { this.apply('A'); });
      return;
    }
    const cs = getComputedStyle(this.h1);
    const font = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    const draw = () => {
      x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, c.width, c.height);
      x.font = font; x.letterSpacing = cs.letterSpacing; x.fillStyle = '#ebe8e1'; x.textBaseline = 'alphabetic';
      const m = x.measureText(TEXT); const lh = parseFloat(cs.lineHeight) || r.height;
      const asc = m.fontBoundingBoxAscent, desc = m.fontBoundingBoxDescent;
      const baseline = r.top - hr.top + (lh - (asc + desc)) / 2 + asc;
      x.fillText(TEXT, r.left - hr.left, baseline);
      hero.setOverlay(c);
    };
    document.fonts.load(font, TEXT).then(draw, draw);
  };
  // generated lettering on black -> alpha from brightness, so only the letters land on the flower
  TypeLab.prototype.loadImage = function(src){
    if (this.images[src]) return this.images[src];
    return this.images[src] = new Promise((res, rej) => {
      const img = new Image(); img.crossOrigin = 'anonymous';
      img.onload = () => {
        const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
        const x = c.getContext('2d'); x.drawImage(img, 0, 0);
        const d = x.getImageData(0, 0, c.width, c.height), p = d.data;
        for (let i = 0; i < p.length; i += 4) { const a = Math.min(255, Math.max(p[i], p[i + 1], p[i + 2]) * 1.35 - 18); p[i + 3] = a < 0 ? 0 : a; }
        x.putImageData(d, 0, 0); res(c);
      };
      img.onerror = rej; img.src = src;
    });
  };

  // ---- css modes: burst-driven melt / slices / CRT bleed, fed by the shader's own burst value
  TypeLab.prototype.frame = function(){
    const v = this.v; if (!v || v.mode !== 'css') return;
    const b = this.hero.burst || 0;
    if (b > this.lastBurst + 0.15) this.sliceSeed = this.slices.map(() => [rnd(), 0.06 + rnd() * 0.25, (rnd() - 0.5) * 2]);
    this.lastBurst = b;
    if (v.melt && this.melt) { this.melt.setAttribute('scale', (b * 90).toFixed(1)); if (b > 0.02) this.turb.setAttribute('seed', String(Math.floor(performance.now() / 90) % 50)); }
    if (v.bleed) { this.h1.style.setProperty('--bled', (18 + 50 * b).toFixed(1)); this.h1.style.setProperty('--scan', (30 - 70 * b).toFixed(1)); }
    const on = b > 0.04 && v.slices;
    this.slices.forEach((s, i) => {
      if (!on) { s.style.opacity = 0; return; }
      const [y, h, d] = this.sliceSeed[i] || [0, 0, 0];
      s.style.opacity = 1; s.style.left = this.h1.offsetLeft + 'px'; s.style.top = this.h1.offsetTop + 'px'; s.style.width = this.h1.offsetWidth + 'px';
      s.style.clipPath = `inset(${(y * 100).toFixed(1)}% 0 ${Math.max(0, (1 - y - h) * 100).toFixed(1)}% 0)`;
      const step = v.stair ? 8 : 1; s.style.transform = `translateX(${Math.round(d * 34 * b / step) * step}px)`;
      s.style.color = ['#ff3aa8', '#4d86ff', '#ebe8e1'][i]; s.style.mixBlendMode = i < 2 ? 'screen' : 'normal';
    });
  };

  window.TypeLab = TypeLab;
})();
