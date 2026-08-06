/* Custom d3-force-compatible forces + canvas geometry helpers for the graph view's
   directory-cluster layout. force-graph@1.51.4 vendors d3-force internally but never
   puts d3 on the page, so anything registered as a custom force (fg.d3Force(name, fn))
   has to satisfy the contract by hand: a function `force(alpha)` that mutates node
   vx/vy (or x/y), carrying a `force.initialize(nodes)` that d3 calls on registration.
   No imports, no globals besides GraphForces — this file is also require()'d directly
   from a node smoke test, hence globalThis instead of window. */
"use strict";

(function () {
  // ---------------------------------------------------------------------------
  // Internal helpers
  // ---------------------------------------------------------------------------

  // Knuth multiplicative hash folded into a 32-bit range, then mapped to an angle.
  // Stands in for Math.random() everywhere a "spread these apart" nudge is needed —
  // graphs get exported to PNG/WebM, so the same input must always paint the same
  // pixels, run to run and machine to machine.
  function seededAngle(seed) {
    let h = Math.imul(seed | 0, 2654435761) >>> 0;
    h = (h ^ (h >>> 15)) >>> 0;
    return (h / 4294967295) * 2 * Math.PI;
  }

  function isPinned(n) {
    return (n.fx !== undefined && n.fx !== null) || (n.fy !== undefined && n.fy !== null);
  }

  function finitePoint(n) {
    return Number.isFinite(n.x) && Number.isFinite(n.y);
  }

  // Signed area of o->a->b, twice over. Positive means a->b turns left of o->a.
  function crossProduct(o, a, b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  }

  // ---------------------------------------------------------------------------
  // Forces
  // ---------------------------------------------------------------------------

  // Classic cluster force (cf. vasturiano/d3-force-clustering, Observable's
  // "Clustered Bubbles"): pulls every node toward the weighted centroid of its
  // own group, recomputed fresh from current positions on every tick so clusters
  // track the layout as it settles instead of anchoring to a stale snapshot.
  function forceCluster({ strength = 0.3, key = (n) => n.group, weight = (n) => 1 } = {}) {
    let nodes = [];
    const groups = new Map(); // cleared and rebuilt each tick, reused across ticks

    const force = (alpha) => {
      if (!nodes.length) return;
      groups.clear();
      for (const n of nodes) {
        if (!finitePoint(n)) continue;
        const k = key(n);
        if (k === null || k === undefined) continue;
        let g = groups.get(k);
        if (!g) { g = { sx: 0, sy: 0, sw: 0, count: 0 }; groups.set(k, g); }
        const w = weight(n) || 0;
        g.sx += n.x * w;
        g.sy += n.y * w;
        g.sw += w;
        g.count++;
      }
      for (const n of nodes) {
        if (!finitePoint(n)) continue;
        const k = key(n);
        if (k === null || k === undefined) continue;
        const g = groups.get(k);
        if (!g || g.count < 2 || g.sw === 0) continue; // lone members have no cluster to pull toward
        const cx = g.sx / g.sw;
        const cy = g.sy / g.sw;
        if (!Number.isFinite(cx) || !Number.isFinite(cy)) continue;
        n.vx = (n.vx || 0) + (cx - n.x) * alpha * strength;
        n.vy = (n.vy || 0) + (cy - n.y) * alpha * strength;
      }
    };
    force.initialize = (ns) => { nodes = ns || []; };
    return force;
  }

  // Resolves the overlap between two circles by moving x/y directly (a "positional"
  // force, unlike forceCluster's velocity nudge) — matches d3.forceCollide's own
  // convention, which is why it stays alpha-independent too.
  function resolvePair(nodes, radii, i, j, strength) {
    const a = nodes[i], b = nodes[j];
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let dist = Math.hypot(dx, dy);
    const minDist = radii[i] + radii[j];
    if (dist >= minDist) return;
    if (dist < 1e-9) {
      // exactly coincident nodes have no direction to separate along; manufacture
      // one deterministically so re-running the layout never depends on luck.
      const angle = seededAngle(i * 65599 + j);
      dx = Math.cos(angle);
      dy = Math.sin(angle);
      dist = 1;
    }
    const nx = dx / dist;
    const ny = dy / dist;
    const push = (minDist - dist) * strength;
    const pinnedA = isPinned(a);
    const pinnedB = isPinned(b);
    if (pinnedA && pinnedB) return; // neither can move; nothing to resolve
    if (pinnedA) {
      b.x += nx * push;
      b.y += ny * push;
    } else if (pinnedB) {
      a.x -= nx * push;
      a.y -= ny * push;
    } else {
      const half = push * 0.5;
      a.x -= nx * half;
      a.y -= ny * half;
      b.x += nx * half;
      b.y += ny * half;
    }
  }

  // `radius` is a number or a per-node function. Bucketing nodes into a uniform
  // grid (cell = 2 * max radius, so any colliding pair falls within one cell of
  // each other) keeps this near-O(n) instead of O(n^2) — needed because this runs
  // every simulation tick, and graphs here can run into the thousands of nodes.
  function forceCollide(radius, { strength = 0.75, iterations = 2 } = {}) {
    let nodes = [];
    const radiusOf = typeof radius === "function" ? radius : () => (radius || 0);
    let radii = [];
    const grid = new Map(); // rebuilt once per call, reused as a Map across calls

    const force = () => {
      const n = nodes.length;
      if (n < 2) return;
      if (radii.length !== n) radii = new Array(n);
      let maxR = 0;
      for (let i = 0; i < n; i++) {
        const r = radiusOf(nodes[i]) || 0;
        radii[i] = r;
        if (r > maxR) maxR = r;
      }
      const cellSize = Math.max(2 * maxR, 1e-6);

      // Grid built once per call (not per iteration) — positions drift only a
      // little between iterations, and rebuilding here would double the cost for
      // marginal accuracy. Cell size is generous enough to absorb that drift.
      grid.clear();
      for (let i = 0; i < n; i++) {
        const node = nodes[i];
        if (!finitePoint(node)) continue;
        const ck = Math.floor(node.x / cellSize) + ":" + Math.floor(node.y / cellSize);
        let bucket = grid.get(ck);
        if (!bucket) { bucket = []; grid.set(ck, bucket); }
        bucket.push(i);
      }

      for (let iter = 0; iter < iterations; iter++) {
        for (let i = 0; i < n; i++) {
          const a = nodes[i];
          if (!finitePoint(a)) continue;
          const cx = Math.floor(a.x / cellSize);
          const cy = Math.floor(a.y / cellSize);
          for (let ddx = -1; ddx <= 1; ddx++) {
            for (let ddy = -1; ddy <= 1; ddy++) {
              const bucket = grid.get((cx + ddx) + ":" + (cy + ddy));
              if (!bucket) continue;
              for (const j of bucket) {
                if (j <= i) continue; // each unordered pair resolved exactly once
                resolvePair(nodes, radii, i, j, strength);
              }
            }
          }
        }
      }
    };
    force.initialize = (ns) => { nodes = ns || []; };
    return force;
  }

  // Pulls every node toward its group's fixed slot (see ringAnchors below).
  // Groups absent from `anchors` are left alone, so a partial anchor map is safe.
  function forceGroupAnchors(anchors, strength = 0.12) {
    let nodes = [];
    const force = (alpha) => {
      if (!anchors || !anchors.size || !nodes.length) return;
      for (const n of nodes) {
        if (!finitePoint(n)) continue;
        const anchor = anchors.get(groupKey(n));
        if (!anchor) continue;
        n.vx = (n.vx || 0) + (anchor.x - n.x) * alpha * strength;
        n.vy = (n.vy || 0) + (anchor.y - n.y) * alpha * strength;
      }
    };
    force.initialize = (ns) => { nodes = ns || []; };
    return force;
  }

  // ---------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------

  // Evenly spaces group anchor slots on a circle, radius growing with sqrt(count)
  // so the ring's arc length (and so, spacing between neighbours) stays roughly
  // constant as more directories get added. Angle 0 starts at -PI/2 (screen top,
  // canvas is y-down) purely so replays are legible: the first group is always
  // where your eye lands first.
  function ringAnchors(groups, { spacing = 90, jitterFree = true } = {}) {
    const map = new Map();
    const list = groups || [];
    const n = list.length;
    if (n === 0) return map;
    if (n === 1) { map.set(list[0], { x: 0, y: 0 }); return map; }
    const radius = spacing * Math.sqrt(n);
    for (let i = 0; i < n; i++) {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / n;
      let x = radius * Math.cos(angle);
      let y = radius * Math.sin(angle);
      if (!jitterFree) {
        // small deterministic nudge so a perfectly regular ring doesn't look
        // robotic — still a pure function of (i, n), never Math.random().
        const j = seededAngle(i * 97 + n);
        x += Math.cos(j) * spacing * 0.05;
        y += Math.sin(j) * spacing * 0.05;
      }
      map.set(list[i], { x, y });
    }
    return map;
  }

  // ---------------------------------------------------------------------------
  // Geometry
  // ---------------------------------------------------------------------------

  // Andrew's monotone chain. Returns the hull in counter-clockwise order, built
  // from (and pointing at) the original point objects — callers that stash extra
  // fields on a point (color, group id) get them back on the hull for free.
  function convexHull(points) {
    const seen = new Set();
    const unique = [];
    for (const p of points || []) {
      if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
      const k = p.x + "," + p.y;
      if (seen.has(k)) continue;
      seen.add(k);
      unique.push(p);
    }
    if (unique.length < 3) return unique;

    const pts = unique.slice().sort((a, b) => a.x - b.x || a.y - b.y);
    const n = pts.length;

    const lower = [];
    for (const p of pts) {
      while (lower.length >= 2 && crossProduct(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
      lower.push(p);
    }
    const upper = [];
    for (let i = n - 1; i >= 0; i--) {
      const p = pts[i];
      while (upper.length >= 2 && crossProduct(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
      upper.push(p);
    }
    lower.pop();
    upper.pop();
    return lower.concat(upper);
  }

  // Pushes every hull vertex outward from the hull's centroid so a "blob" drawn
  // around it has breathing room past the outermost node. 1-2 point inputs have
  // no interior to push away from, so we build a small padded box instead —
  // callers always get a closed shape back, never a degenerate line or dot.
  function padHull(hull, pad) {
    const pts = (hull || []).filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y));
    if (pts.length === 0) return [];
    if (pts.length < 3) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of pts) {
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
      }
      if (maxX - minX < 1e-6) { minX -= pad; maxX += pad; }
      if (maxY - minY < 1e-6) { minY -= pad; maxY += pad; }
      return [
        { x: minX - pad, y: minY - pad },
        { x: maxX + pad, y: minY - pad },
        { x: maxX + pad, y: maxY + pad },
        { x: minX - pad, y: maxY + pad },
      ];
    }
    let cx = 0, cy = 0;
    for (const p of pts) { cx += p.x; cy += p.y; }
    cx /= pts.length;
    cy /= pts.length;
    return pts.map((p) => {
      const dx = p.x - cx, dy = p.y - cy;
      const dist = Math.hypot(dx, dy);
      if (dist < 1e-9) return { x: p.x, y: p.y }; // vertex sits on the centroid; no direction to push
      return { x: p.x + (dx / dist) * pad, y: p.y + (dy / dist) * pad };
    });
  }

  // Catmull-Rom-to-bezier through a closed loop of points, for the soft cluster
  // "blob" backgrounds. Only builds the path (beginPath..closePath) — fill/stroke
  // is the caller's call, since the same path gets filled for the blob and
  // stroked for its outline.
  function traceClosedCurve(ctx, points, tension = 0.4) {
    if (!ctx) return;
    const pts = (points || []).filter((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y));
    if (pts.length === 0) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    if (pts.length < 3) {
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      return;
    }
    const n = pts.length;
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n];
      const p1 = pts[i];
      const p2 = pts[(i + 1) % n];
      const p3 = pts[(i + 2) % n];
      const cp1x = p1.x + (p2.x - p0.x) * (tension / 6);
      const cp1y = p1.y + (p2.y - p0.y) * (tension / 6);
      const cp2x = p2.x - (p3.x - p1.x) * (tension / 6);
      const cp2y = p2.y - (p3.y - p1.y) * (tension / 6);
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
    }
    ctx.closePath();
  }

  // ---------------------------------------------------------------------------
  // Grouping & color
  // ---------------------------------------------------------------------------

  // Every force/anchor/hull in this file clusters by directory, and this is the
  // one place that decides what "directory" means for a given node. Node shape
  // is { id, name, type, file }, file possibly suffixed ":<line>" (see fileRef()
  // in app.js). A directory node is its own group — everything else groups under
  // its parent dir — so directories and their contents land in the same cluster.
  function groupKey(node) {
    try {
      if (!node || !node.file) return "·root";
      const clean = String(node.file).replace(/:\d+$/, "");
      if (node.type === "directory") {
        const norm = clean.replace(/\/+$/, "");
        return norm === "" ? "·root" : norm;
      }
      const idx = clean.lastIndexOf("/");
      const parent = idx === -1 ? "" : clean.slice(0, idx);
      const norm = parent.replace(/\/+$/, "");
      return norm === "" ? "·root" : norm;
    } catch (e) {
      return "·root"; // must never throw — this runs inside the paint loop
    }
  }

  // Deterministic FNV-1a-ish string hash, folded to a hue in [0, 360). Same
  // string always paints the same color, across sessions and across the PNG/WebM
  // export path, without keeping a color assignment table anywhere.
  function hashHue(str) {
    const s = String(str == null ? "" : str);
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0) % 360;
  }

  // Saturation/lightness stay muted by default — this paints cluster tints on
  // top of a quiet warm-botanical dark theme, and full-saturation hues would
  // fight the rest of the palette.
  function groupColor(str, { sat = 42, light = 62, alpha = 1 } = {}) {
    const hue = hashHue(str);
    return alpha >= 1 ? `hsl(${hue}, ${sat}%, ${light}%)` : `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
  }

  globalThis.GraphForces = {
    forceCluster, forceCollide, forceGroupAnchors,
    ringAnchors,
    convexHull, padHull, traceClosedCurve,
    groupKey, hashHue, groupColor,
  };
})();
