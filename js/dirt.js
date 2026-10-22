/* dirt.js — one image, one fragment shader, several looks.
   CRT dot matrix, chroma split, pixel-stair block masks, row tearing, ASCII cells. */
(function () {
  'use strict';

  const VERT = 'attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}';
  const FRAG = `
precision highp float;
uniform sampler2D uTex, uAtlas, uOver;
uniform vec2 uRes, uMouse, uShift;
uniform float uImgAspect, uTime, uBurst, uSeed, uDpr, uZoom, uMouseAmt, uGlyphN, uFit;
uniform float uAscii, uAsciiCell, uDots, uDotCell, uScan, uGrain, uAber, uSolar, uOverAmt, uOverDirt;
uniform float uDuo; uniform vec3 uDuoA, uDuoB;
uniform float uThresh; uniform vec3 uPaper, uInk;

float hash(vec2 p){ vec3 q = fract(vec3(p.xyx) * 0.1031); q += dot(q, q.yzx + 33.33); return fract((q.x + q.y) * q.z); }
float noise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f);
  return mix(mix(hash(i), hash(i+vec2(1,0)), f.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y); }
float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
vec2 cover(vec2 uv){ float ca = uRes.x/uRes.y; vec2 s = vec2(1.0);
  bool wide = ca > uImgAspect; if (uFit > 0.5) wide = !wide;
  if (wide) s.y = uImgAspect/ca; else s.x = ca/uImgAspect; return (uv - 0.5 - uShift)*s/uZoom + 0.5; }
vec3 img(vec2 uv){ if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return vec3(0.0); return texture2D(uTex, uv).rgb; }
vec4 over(vec2 suv){ if (uOverAmt < 0.5) return vec4(0.0); return texture2D(uOver, clamp(suv, 0.0, 1.0)); }
// screen uv -> image behind + title overlay (the title lives in screen space, the image parallaxes)
vec3 scene(vec2 suv, vec2 par){ vec3 c = img(cover(suv + par)); vec4 o = over(suv); return mix(c, o.rgb, o.a); }
vec3 grade(vec3 c){
  if (uDuo > 0.5) { float l = smoothstep(0.04, 0.92, luma(c)); return mix(uDuoA, uDuoB, l); }
  if (uThresh > 0.5) { float th = 0.30 + 0.12*noise(gl_FragCoord.xy*0.07); return mix(uPaper, uInk, step(th, luma(c))); }
  return c;
}
void main(){
  vec2 px = gl_FragCoord.xy; vec2 uv = px/uRes; float t = uTime;
  vec2 par = (uMouse/uRes - 0.5) * 0.016 * uMouseAmt;

  // row tearing (only during bursts)
  float rowH = (5.0 + 44.0*hash(vec2(floor(t*6.0), uSeed))) * uDpr;
  float row = floor(px.y/rowH);
  float rr = hash(vec2(row, floor(t*11.0) + uSeed));
  float shift = 0.0;
  if (rr > 1.0 - 0.42*uBurst) shift = (hash(vec2(row*3.1, floor(t*17.0))) - 0.5) * 0.24 * uBurst;

  // pixel-stair block mask
  float bs = (9.0 + 28.0*hash(vec2(floor(t*2.0), uSeed + 2.0))) * uDpr;
  vec2 blk = floor(px/bs);
  float bm = step(0.66, noise(blk*0.23 + vec2(t*0.9, uSeed)));
  float stair = bm * step(0.12, uBurst);

  vec2 suv = uv; suv.x += shift;
  suv += stair * (vec2(hash(blk + 1.0), hash(blk + 7.0)) - 0.5) * 0.05;

  // chroma split (image and title alike)
  float ab = uAber + 0.02*uBurst + 0.014*stair;
  vec2 dir = normalize(uv - 0.5 + vec2(1e-4));
  vec3 col;
  col.r = scene(suv + dir*ab, par).r; col.g = scene(suv, par).g; col.b = scene(suv - dir*ab, par).b;

  // solarize + tint inside stair blocks
  if (stair > 0.5) { float l = luma(col); vec3 inv = (1.0 - col) * vec3(1.0, 0.45, 0.9) + vec3(0.08, 0.0, 0.14);
    col = mix(col, inv, step(0.3, l) * uSolar); }
  col = grade(col);

  // CRT dot matrix with RGB stripes, gated by a slow noise field
  if (uDots > 0.0) {
    float cs = uDotCell * uDpr; vec2 cell = floor(px/cs); vec2 cc = (cell + 0.5) * cs;
    vec2 csuv = cc/uRes; csuv.x += shift;
    vec3 cc3 = grade(scene(csuv, par)); float l = luma(cc3); float oa = over(csuv).a;
    vec2 f = fract(px/cs) - 0.5; float d = length(f);
    float rad = 0.10 + 0.50*l;
    float dot_ = 1.0 - smoothstep(rad - 0.14, rad + 0.04, d);
    float sub = fract(px.x/cs*3.0);
    vec3 stripe = vec3(step(sub, 0.333), step(0.333, sub)*step(sub, 0.666), step(0.666, sub));
    vec3 dcol = cc3 * dot_ * 1.45 * mix(vec3(1.0), stripe*2.4, 0.3);
    float dmask = smoothstep(0.38, 0.62, noise(px/(430.0*uDpr) + vec2(t*0.04, uSeed)));
    dmask = max(dmask * uDots, stair * 0.6) * (1.0 - oa * (1.0 - uOverDirt));
    col = mix(col, dcol, dmask);
  }

  // ASCII cells, gated by another noise field + mouse radius + stair blocks
  if (uAscii > 0.0) {
    float cs = uAsciiCell * uDpr; vec2 cell = floor(px/cs); vec2 cc = (cell + 0.5) * cs;
    vec2 csuv = cc/uRes; csuv.x += shift;
    vec3 raw = scene(csuv, par); vec3 cc3 = grade(raw); float l = pow(luma(cc3), 0.75); float oa = over(csuv).a;
    if (uThresh > 0.5) l = pow(luma(raw), 0.6);
    float idx = floor(l * (uGlyphN - 0.001));
    vec2 f = fract(px/cs);
    float g = texture2D(uAtlas, vec2((idx + f.x)/uGlyphN, f.y)).r;
    vec3 acol = mix(cc3*1.7, vec3(1.0), 0.12) * g;
    if (uThresh > 0.5) acol = mix(uPaper, uInk, g);
    float amask = smoothstep(0.54, 0.68, noise(px/(360.0*uDpr) + vec2(-t*0.05, uSeed + 9.0)));
    float md = distance(px, uMouse)/uDpr;
    amask = max(amask, (1.0 - smoothstep(80.0, 260.0, md)) * uMouseAmt);
    amask = max(amask * uAscii, stair) * (1.0 - oa * (1.0 - uOverDirt));
    col = mix(col, acol, amask);
  }

  col *= 1.0 - uScan * 0.5 * (0.5 + 0.5*sin(px.y/uDpr * 3.14159));
  float vig = smoothstep(1.4, 0.35, length(uv - 0.5) * 1.25);
  col *= mix(1.0, vig, uThresh > 0.5 ? 0.0 : 0.6);
  col += (hash(px + fract(t) * 100.0) - 0.5) * uGrain;
  gl_FragColor = vec4(col, 1.0);
}`;

  const GLYPHS = ' .`:-~=+*xo#%&@';
  const U = ['uTex','uAtlas','uRes','uMouse','uShift','uFit','uImgAspect','uTime','uBurst','uSeed','uDpr','uZoom','uMouseAmt','uGlyphN',
    'uAscii','uAsciiCell','uDots','uDotCell','uScan','uGrain','uAber','uSolar','uOver','uOverAmt','uOverDirt','uDuo','uDuoA','uDuoB','uThresh','uPaper','uInk'];

  const instances = [];
  let atlasCanvas = null;

  function hex(h){ const n = parseInt(h.slice(1), 16); return [((n>>16)&255)/255, ((n>>8)&255)/255, (n&255)/255]; }

  function makeAtlas(){
    const cell = 32, c = document.createElement('canvas');
    c.width = cell * GLYPHS.length; c.height = cell;
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
    x.fillStyle = '#fff'; x.font = '500 27px "IBM Plex Mono", Menlo, monospace';
    x.textAlign = 'center'; x.textBaseline = 'middle';
    for (let i = 0; i < GLYPHS.length; i++) x.fillText(GLYPHS[i], i*cell + cell/2, cell/2 + 1);
    return c;
  }

  function Dirt(canvas, src, o){
    this.c = canvas;
    this.o = Object.assign({
      ascii: .55, asciiCell: 11, dots: .7, dotCell: 3.2, scan: .35, grain: .06, aber: .0025, solar: 1,
      zoom: 1, fit: 0, shift: [0, 0], mouseAmt: 1, burstRate: 1, duo: null, thresh: null, seed: Math.random()*10, reduced: false
    }, o || {});
    const gl = this.gl = canvas.getContext('webgl', { antialias: false, alpha: false, preserveDrawingBuffer: false });
    if (!gl) { canvas.style.background = '#000'; return; }
    const prog = gl.createProgram();
    for (const [type, s] of [[gl.VERTEX_SHADER, VERT], [gl.FRAGMENT_SHADER, FRAG]]) {
      const sh = gl.createShader(type); gl.shaderSource(sh, s); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) { console.error(gl.getShaderInfoLog(sh)); return; }
      gl.attachShader(prog, sh);
    }
    gl.linkProgram(prog); gl.useProgram(prog); this.prog = prog;
    const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.u = {}; U.forEach(n => this.u[n] = gl.getUniformLocation(prog, n));

    this.tex = this.makeTex(); this.atlas = this.makeTex(); this.over = this.makeTex(); this.overAmt = 0; this.overDirt = 1; this.hold = false;
    this.ready = false; this.aspect = 1;
    this.mouse = [-1e4, -1e4]; this.mouseTarget = null; this.burst = 0; this.next = 1.2 + Math.random()*2;
    this.visible = false; this.t0 = performance.now(); this.last = this.t0;

    const img = new Image(); img.crossOrigin = 'anonymous';
    img.onload = () => { this.aspect = img.width / img.height; this.upload(this.tex, img); this.ready = true; };
    img.src = src;
    document.fonts.ready.then(() => { if (!atlasCanvas) atlasCanvas = makeAtlas(); this.upload(this.atlas, atlasCanvas); });
    if (!atlasCanvas) { atlasCanvas = makeAtlas(); this.upload(this.atlas, atlasCanvas); }

    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
    new IntersectionObserver(en => { this.visible = en[0].isIntersecting; }, { threshold: 0 }).observe(canvas);
    window.addEventListener('mousemove', e => this.onMove(e), { passive: true });
    instances.push(this);
  }
  Dirt.prototype.makeTex = function(){
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 1, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0]));
    return t;
  };
  Dirt.prototype.upload = function(t, src){
    const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, src); }
    catch (e) { console.warn('texture upload failed (serve over http, not file://)', e); }
  };
  // title / lettering overlay: an RGBA canvas the size of this canvas, composited before the dirt
  Dirt.prototype.setOverlay = function(src){
    const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, this.over);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    try { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); this.overAmt = 1; }
    catch (e) { console.warn('overlay upload failed', e); }
  };
  Dirt.prototype.clearOverlay = function(){ this.overAmt = 0; };
  Dirt.prototype.resize = function(){
    const dpr = this.dpr = Math.min(1.5, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.round(this.c.clientWidth * dpr)), h = Math.max(1, Math.round(this.c.clientHeight * dpr));
    if (this.c.width !== w || this.c.height !== h) { this.c.width = w; this.c.height = h; }
  };
  Dirt.prototype.onMove = function(e){
    const r = this.c.getBoundingClientRect();
    if (e.clientY < r.top - 200 || e.clientY > r.bottom + 200) return;
    const x = (e.clientX - r.left) * this.dpr, y = (r.height - (e.clientY - r.top)) * this.dpr;
    if (this.mouseTarget) {
      const d = Math.hypot(x - this.mouseTarget[0], y - this.mouseTarget[1]) / this.dpr;
      if (d > 30 && !this.o.reduced) this.burst = Math.max(this.burst, Math.min(0.9, d/140));
    }
    this.mouseTarget = [x, y];
  };
  Dirt.prototype.kick = function(a){ this.burst = Math.max(this.burst, a); };
  // image-normalized (u right, v down) -> canvas css px; the inverse of the shader's cover()/fit mapping
  Dirt.prototype.map = function(u, v){
    const o = this.o, W = this.c.clientWidth, H = this.c.clientHeight, ca = W / H, ia = this.aspect;
    let wide = ca > ia; if (o.fit > 0.5) wide = !wide;
    const sx = wide ? 1 : ca / ia, sy = wide ? ia / ca : 1;
    const ix = u, iy = 1 - v;
    const ux = (ix - 0.5) * o.zoom / sx + 0.5 + o.shift[0], uy = (iy - 0.5) * o.zoom / sy + 0.5 + o.shift[1];
    return [ux * W, (1 - uy) * H];
  };
  Dirt.prototype.frame = function(now){
    const gl = this.gl, o = this.o, u = this.u;
    const dt = Math.min(0.1, (now - this.last)/1000); this.last = now;
    const t = (now - this.t0)/1000;
    // bursts: scheduled + decaying
    this.next -= dt;
    if (this.next <= 0 && !o.reduced) { this.burst = Math.max(this.burst, 0.45 + Math.random()*0.55); this.next = (0.9 + Math.random()*3.6)/o.burstRate; }
    if (!this.hold) this.burst *= Math.exp(-dt*7);
    if (this.burst < 0.01) this.burst = 0;
    if (this.mouseTarget) { this.mouse[0] += (this.mouseTarget[0]-this.mouse[0])*0.12; this.mouse[1] += (this.mouseTarget[1]-this.mouse[1])*0.12; }
    gl.viewport(0, 0, this.c.width, this.c.height);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex); gl.uniform1i(u.uTex, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.atlas); gl.uniform1i(u.uAtlas, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.over); gl.uniform1i(u.uOver, 2); gl.uniform1f(u.uOverAmt, this.overAmt); gl.uniform1f(u.uOverDirt, this.overDirt);
    gl.uniform2f(u.uRes, this.c.width, this.c.height);
    gl.uniform2f(u.uMouse, this.mouse[0], this.mouse[1]);
    gl.uniform2f(u.uShift, o.shift[0], o.shift[1]); gl.uniform1f(u.uFit, o.fit);
    gl.uniform1f(u.uImgAspect, this.aspect); gl.uniform1f(u.uTime, t); gl.uniform1f(u.uBurst, this.burst);
    gl.uniform1f(u.uSeed, o.seed); gl.uniform1f(u.uDpr, this.dpr); gl.uniform1f(u.uZoom, o.zoom);
    gl.uniform1f(u.uMouseAmt, o.mouseAmt); gl.uniform1f(u.uGlyphN, GLYPHS.length);
    gl.uniform1f(u.uAscii, o.ascii); gl.uniform1f(u.uAsciiCell, o.asciiCell); gl.uniform1f(u.uDots, o.dots);
    gl.uniform1f(u.uDotCell, o.dotCell); gl.uniform1f(u.uScan, o.scan); gl.uniform1f(u.uGrain, o.grain);
    gl.uniform1f(u.uAber, o.aber); gl.uniform1f(u.uSolar, o.solar);
    gl.uniform1f(u.uDuo, o.duo ? 1 : 0);
    if (o.duo) { gl.uniform3fv(u.uDuoA, hex(o.duo[0])); gl.uniform3fv(u.uDuoB, hex(o.duo[1])); }
    gl.uniform1f(u.uThresh, o.thresh ? 1 : 0);
    if (o.thresh) { gl.uniform3fv(u.uPaper, hex(o.thresh[0])); gl.uniform3fv(u.uInk, hex(o.thresh[1])); }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  function loop(now){ for (const d of instances) if (d.visible && d.gl && d.ready) d.frame(now); requestAnimationFrame(loop); }
  requestAnimationFrame(loop);

  window.Dirt = Dirt;
})();
