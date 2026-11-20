/* رسم — Rasm engine · core: namespace, geometry, style DNA, metrics, pen renderer.
 * Coordinates used by every recipe: (u, y) in font units.
 *   u = distance from the glyph's RIGHT edge, growing to the LEFT (reading direction).
 *   y = height above the baseline (outer bottom of the baseline stroke is y = 0).
 * Everything is converted to normal left-to-right font space only at the very end.
 */
(function (root) {
  'use strict';
  const R = root.Rasm = root.Rasm || {};

  /* ------------------------------------------------------------------ geometry */
  const G = R.geom = {};
  G.sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  G.add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  G.mul = (a, k) => [a[0] * k, a[1] * k];
  G.len = (a) => Math.hypot(a[0], a[1]);
  G.norm = (a) => { const l = Math.hypot(a[0], a[1]) || 1; return [a[0] / l, a[1] / l]; };
  G.lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

  /* sample a circular arc; tolerance ~0.2 units */
  G.arc = function (c, r, a0, sweep, out) {
    const tol = 0.2;
    const da = r > tol ? 2 * Math.acos(Math.max(-1, 1 - tol / r)) : Math.PI / 4;
    const n = Math.max(1, Math.ceil(Math.abs(sweep) / Math.max(da, 0.02)));
    for (let i = 0; i <= n; i++) {
      const a = a0 + sweep * i / n;
      out.push([c[0] + r * Math.cos(a), c[1] + r * Math.sin(a)]);
    }
    return out;
  };

  G.cubic = function (p0, p1, p2, p3, out, n) {
    n = n || 24;
    for (let i = 1; i <= n; i++) {
      const t = i / n, mt = 1 - t;
      const a = mt * mt * mt, b = 3 * mt * mt * t, c = 3 * mt * t * t, d = t * t * t;
      out.push([a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]]);
    }
    return out;
  };

  /* Rounded polyline: every interior vertex gets a circular fillet of radius r
   * (number or per-vertex array). Radii are clamped so neighbouring fillets never overlap. */
  G.rpoly = function (pts, r, closed) {
    const n = pts.length;
    if (n < 2) return pts.map(p => p.slice());
    const rad = (i) => Array.isArray(r) ? (r[i] || 0) : (r || 0);
    const info = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
      if (!closed && (i === 0 || i === n - 1)) continue;
      const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n];
      const rr = rad(i);
      if (!(rr > 0.01)) continue;
      const v1 = G.norm(G.sub(p0, p1)), v2 = G.norm(G.sub(p2, p1));
      const dot = Math.max(-1, Math.min(1, v1[0] * v2[0] + v1[1] * v2[1]));
      const th = Math.acos(dot);
      if (th < 1e-3 || Math.PI - th < 1e-3) continue;
      info[i] = { v1, v2, th, td: rr / Math.tan(th / 2) };
    }
    // clamp tangent lengths per segment
    const segCount = closed ? n : n - 1;
    const scale = new Array(n).fill(1);
    for (let s = 0; s < segCount; s++) {
      const i = s, j = (s + 1) % n;
      const L = G.len(G.sub(pts[j], pts[i]));
      const a = info[i] ? info[i].td : 0, b = info[j] ? info[j].td : 0;
      if (a + b > L * 0.999 && a + b > 0) {
        const k = (L * 0.999) / (a + b);
        if (info[i]) scale[i] = Math.min(scale[i], k);
        if (info[j]) scale[j] = Math.min(scale[j], k);
      }
    }
    const out = [];
    const emit = (i) => {
      const f = info[i];
      if (!f) { out.push(pts[i].slice()); return; }
      const td = f.td * scale[i];
      const rr = td * Math.tan(f.th / 2);
      const p1 = pts[i];
      const t1 = G.add(p1, G.mul(f.v1, td)), t2 = G.add(p1, G.mul(f.v2, td));
      const bis = G.norm(G.add(f.v1, f.v2));
      const c = G.add(p1, G.mul(bis, rr / Math.sin(f.th / 2)));
      const a0 = Math.atan2(t1[1] - c[1], t1[0] - c[0]);
      let sw = Math.atan2(t2[1] - c[1], t2[0] - c[0]) - a0;
      while (sw > Math.PI) sw -= 2 * Math.PI;
      while (sw < -Math.PI) sw += 2 * Math.PI;
      G.arc(c, rr, a0, sw, out);
    };
    for (let i = 0; i < n; i++) emit(i);
    return G.dedupe(out, closed);
  };

  G.dedupe = function (pts, closed) {
    const out = [];
    for (const p of pts) {
      const q = out[out.length - 1];
      if (!q || Math.abs(q[0] - p[0]) > 1e-6 || Math.abs(q[1] - p[1]) > 1e-6) out.push(p);
    }
    if (closed && out.length > 2) {
      const a = out[0], z = out[out.length - 1];
      if (Math.abs(a[0] - z[0]) < 1e-6 && Math.abs(a[1] - z[1]) < 1e-6) out.pop();
    }
    return out;
  };

  /* rounded rectangle centre-line, u in [u0,u1], y in [y0,y1]; radii [br, bl, tl, tr] */
  G.rrect = function (u0, y0, u1, y1, r) {
    const rr = Array.isArray(r) ? r : [r, r, r, r];
    // order: bottom-right(u0,y0) → bottom-left(u1,y0) → top-left(u1,y1) → top-right(u0,y1)
    return G.rpoly([[u0, y0], [u1, y0], [u1, y1], [u0, y1]], rr, true);
  };

  /* ------------------------------------------------------------------ style DNA */
  R.defaultDNA = function () {
    return {
      weight: 80,        // stroke thickness (font units, UPM 1000)
      contrast: 0,       // 0 = monoline … 0.8 = strong calligraphic contrast
      nibAngle: 90,      // direction of the nib's long axis (deg). 90 → horizontals thick
      nib: 'round',      // 'round' (elliptic) | 'flat' (broad nib)
      cap: 'butt',       // terminals: 'butt' | 'round' | 'square'
      roundness: 0.55,   // 0 = square kufic corners … 1 = full circles
      alif: 760,         // alif height (outer)
      xh: 470,           // tooth / x-height (outer)
      loop: 430,         // head loop height (outer)
      desc: 250,         // cup descender depth
      deep: 330,         // deep descender (ج ع م)
      width: 1,          // horizontal proportion
      spacing: 1,        // side bearings / joins
      slant: 0,          // degrees, positive leans left-to-right like italics
      dotSize: 1.3,      // × weight
      dotShape: 'square',// 'circle' | 'square' | 'diamond'
      toothStyle: 'stub' // 'stub' | 'peak' | 'bump'
    };
  };

  R.clampDNA = function (d) {
    const c = (v, a, b) => Math.max(a, Math.min(b, v));
    d.weight = c(+d.weight || 80, 20, 220);
    d.contrast = c(+d.contrast || 0, 0, 0.85);
    d.roundness = c(+d.roundness, 0, 1);
    d.alif = c(+d.alif || 760, 400, 900);
    d.xh = c(+d.xh || 470, 0.3 * d.alif, 0.85 * d.alif);
    d.loop = c(+d.loop || 430, 2.2 * d.weight, 0.9 * d.alif);
    d.desc = c(+d.desc || 250, 120, 450);
    d.deep = c(+d.deep || 330, d.desc, 520);
    d.width = c(+d.width || 1, 0.6, 1.8);
    d.spacing = c(+d.spacing || 1, 0.4, 2.5);
    d.slant = c(+d.slant || 0, -20, 20);
    d.dotSize = c(+d.dotSize || 1.3, 0.8, 2.6);
    return d;
  };

  /* ------------------------------------------------------------------ metrics */
  R.metrics = function (dna) {
    const W = dna.weight, h = W / 2, wf = dna.width;
    const capExt = dna.cap === 'butt' ? 0 : h;
    const m = { dna, W, h, wf, capExt, rho: dna.roundness, cap: dna.cap };
    m.b = h;                                   // baseline centre-line
    m.A = dna.alif; m.a = dna.alif - capExt;   // alif top centre-line
    m.X = dna.xh; m.t = dna.xh - capExt;       // tooth top centre-line
    m.L = dna.loop;
    m.D = dna.desc; m.d = -dna.desc + h;       // cup bottom centre-line
    m.DD = dna.deep; m.dd = -dna.deep + h;     // deep bottom centre-line
    const sp = dna.spacing;
    m.sb = Math.round((0.5 * W + 26) * sp);    // side bearing
    m.jl = Math.round((0.75 * W + 24) * wf * sp); // connection length on each joined side
    m.ov = Math.max(3, Math.round(W * 0.05));  // overlap beyond joined edges
    const A = dna.alif;
    m.Bl = Math.round(Math.max(0.74 * A * wf, 4.4 * W));       // basin (ب) length
    m.Cw = Math.round(Math.max(0.6 * A * wf, 3.4 * W));        // cup width
    m.Lw = Math.round(Math.max(dna.loop * 0.94 * Math.sqrt(wf), 2.3 * W)); // head loop width
    m.Ow = Math.round(Math.max(0.82 * A * wf, m.Lw * 1.35));   // big loop (ص ط)
    m.Oh = Math.round(Math.max(Math.min(dna.xh * 0.95, dna.loop * 1.05), 2.3 * W));
    m.Aw = Math.round(Math.max(0.64 * A * wf, 3.6 * W));       // arc (ح ع body)
    m.Nw = Math.round(Math.max(0.6 * A * wf, 3.4 * W));        // nose (ح head)
    m.Hw = Math.round(Math.max(0.46 * A * wf, 2.8 * W));       // ain head
    m.Tw = Math.round(Math.max(0.36 * A * wf, 2.2 * W));       // tail reach
    m.Kw = Math.round(Math.max(0.5 * A * wf, 3.1 * W));        // kaf chevron
    m.Dw = Math.round(Math.max(0.52 * A * wf, 3.0 * W));       // dal
    m.Ew = Math.round(Math.max(0.8 * A * wf, 4.2 * W));        // heh width
    m.Eh = Math.round(Math.max(Math.min(dna.loop * 1.35, A * 0.85), 3 * W)); // heh height
    m.Yw = Math.round(Math.max(0.95 * A * wf, 5 * W));         // yeh bowl width
    m.sg = Math.round(W + Math.max(1.15 * W, 0.2 * dna.xh) * wf); // tooth spacing (centre to centre)
    m.ds = Math.max(W * dna.dotSize, 28);                       // dot size
    m.dg = Math.round(m.ds * 0.32);                              // gap between dots
    m.mg = Math.round(0.5 * W + 0.35 * m.ds + 14);               // gap letter → mark
    m.mw = Math.max(18, W * 0.66);                               // pen for hamza / madda
    m.r = (rmax) => Math.max(0, m.rho * rmax);
    return m;
  };

  /* ------------------------------------------------------------------ pen renderer */
  const Pen = R.pen = {};

  function hull(points) {
    const p = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [], upper = [];
    for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
    for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
    upper.pop(); lower.pop();
    return lower.concat(upper);
  }

  /* nib polygon (font units, centred at origin) */
  Pen.nibPoly = function (dna, w) {
    const A = w / 2, B = Math.max(w * (1 - dna.contrast), w * 0.12) / 2;
    const th = dna.nibAngle * Math.PI / 180;
    const pts = [];
    if (dna.nib === 'flat') {
      for (const [x, y] of [[A, B], [-A, B], [-A, -B], [A, -B]]) pts.push([x * Math.cos(th) - y * Math.sin(th), x * Math.sin(th) + y * Math.cos(th)]);
    } else {
      const n = 20;
      for (let i = 0; i < n; i++) {
        const t = i / n * Math.PI * 2, x = A * Math.cos(t), y = B * Math.sin(t);
        pts.push([x * Math.cos(th) - y * Math.sin(th), x * Math.sin(th) + y * Math.cos(th)]);
      }
    }
    return pts;
  };

  function densify(pts, closed, step) {
    const out = [];
    const n = pts.length;
    const segs = closed ? n : n - 1;
    for (let i = 0; i < segs; i++) {
      const a = pts[i], b = pts[(i + 1) % n];
      const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const k = Math.max(1, Math.ceil(L / step));
      for (let j = 0; j < k; j++) out.push([a[0] + (b[0] - a[0]) * j / k, a[1] + (b[1] - a[1]) * j / k]);
    }
    if (!closed) out.push(pts[n - 1].slice()); else out.push(pts[0].slice());
    return out;
  }

  /* view: { map(u,y)->[X,Y], s } */
  Pen.strokePart = function (ctx, part, dna, view) {
    const w = (part.w || dna.weight) * (part.wmul || 1);
    const pts = part.pts;
    if (!pts || pts.length < 2) return;
    const useNib = dna.contrast > 0.04;
    if (!useNib) {
      ctx.beginPath();
      for (let i = 0; i < pts.length; i++) {
        const [X, Y] = view.map(pts[i][0], pts[i][1]);
        if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y);
      }
      if (part.closed) ctx.closePath();
      ctx.lineWidth = w * view.s;
      ctx.lineJoin = 'miter'; ctx.miterLimit = 4; ctx.lineCap = 'butt';
      ctx.stroke();
      if (!part.closed) {
        const caps = part.caps || [dna.cap, dna.cap];
        Pen.cap(ctx, pts[0], pts[1], caps[0], w, view);
        Pen.cap(ctx, pts[pts.length - 1], pts[pts.length - 2], caps[1], w, view);
      }
      return;
    }
    // calligraphic nib: union of convex hulls of consecutive nib stamps
    const nib = Pen.nibPoly(dna, w);
    const dense = densify(pts, part.closed, Math.max(2, w * 0.18));
    ctx.beginPath();
    for (let i = 0; i < dense.length - 1; i++) {
      const a = dense[i], b = dense[i + 1];
      const poly = hull(nib.map(p => [p[0] + a[0], p[1] + a[1]]).concat(nib.map(p => [p[0] + b[0], p[1] + b[1]])));
      for (let k = 0; k < poly.length; k++) {
        const [X, Y] = view.map(poly[k][0], poly[k][1]);
        if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y);
      }
      ctx.closePath();
    }
    ctx.fill('nonzero');
    if (!part.closed) {
      const caps = part.caps || [dna.cap, dna.cap];
      // butt joins on a nib pen: extend nothing; square: extend along direction
      if (caps[0] === 'square') Pen.cap(ctx, pts[0], pts[1], 'square', w * (1 - dna.contrast * 0.5), view);
      if (caps[1] === 'square') Pen.cap(ctx, pts[pts.length - 1], pts[pts.length - 2], 'square', w * (1 - dna.contrast * 0.5), view);
    }
  };

  /* cap at end point p, previous point q (direction q → p) */
  Pen.cap = function (ctx, p, q, type, w, view) {
    if (!type || type === 'butt' || type === 'join') return;
    const [X, Y] = view.map(p[0], p[1]);
    if (type === 'round') {
      ctx.beginPath(); ctx.arc(X, Y, w * view.s / 2, 0, Math.PI * 2); ctx.fill();
      return;
    }
    if (type === 'square') {
      const d = G.norm(G.sub(p, q)), n = [-d[1], d[0]];
      const h = w / 2;
      const c = [[p[0] + n[0] * h, p[1] + n[1] * h], [p[0] + n[0] * h + d[0] * h, p[1] + n[1] * h + d[1] * h],
        [p[0] - n[0] * h + d[0] * h, p[1] - n[1] * h + d[1] * h], [p[0] - n[0] * h, p[1] - n[1] * h]];
      ctx.beginPath();
      c.forEach((pt, i) => { const [x, y] = view.map(pt[0], pt[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fill();
    }
  };

  Pen.fillPoly = function (ctx, pts, view) {
    ctx.beginPath();
    pts.forEach((pt, i) => { const [x, y] = view.map(pt[0], pt[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.fill();
  };

  /* tinted copies of alpha-mask canvases, cached per colour */
  const tintCache = new WeakMap();
  Pen.tinted = function (img, color) {
    let m = tintCache.get(img);
    if (!m) { m = {}; tintCache.set(img, m); }
    if (m[color]) return m[color];
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const x = c.getContext('2d');
    x.drawImage(img, 0, 0);
    x.globalCompositeOperation = 'source-in';
    x.fillStyle = color; x.fillRect(0, 0, c.width, c.height);
    m[color] = c;
    return c;
  };

  /* draw a list of parts. colorOf(part) → css colour. */
  Pen.drawParts = function (ctx, parts, dna, view, colorOf) {
    for (const part of parts) {
      const col = colorOf ? colorOf(part) : '#000';
      ctx.fillStyle = col; ctx.strokeStyle = col;
      if (part.type === 'stroke') Pen.strokePart(ctx, part, dna, view);
      else if (part.type === 'poly') Pen.fillPoly(ctx, part.pts, view);
      else if (part.type === 'image') {
        const [X1, Y1] = view.map(part.u0 + part.w, part.y0 + part.h); // left, top
        const [X2, Y2] = view.map(part.u0, part.y0);                    // right, bottom
        const img = col === '#000' ? part.img : Pen.tinted(part.img, col);
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, X1, Y1, X2 - X1, Y2 - Y1);
      }
    }
  };

  /* ink bounds of parts in (u,y) */
  R.partBounds = function (parts, dna) {
    let u0 = Infinity, u1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of parts) {
      if (p.type === 'image') {
        u0 = Math.min(u0, p.u0); u1 = Math.max(u1, p.u0 + p.w); y0 = Math.min(y0, p.y0); y1 = Math.max(y1, p.y0 + p.h);
        continue;
      }
      const e = p.type === 'stroke' ? ((p.w || dna.weight) * (p.wmul || 1)) / 2 : 0;
      for (const q of p.pts) {
        u0 = Math.min(u0, q[0] - e); u1 = Math.max(u1, q[0] + e); y0 = Math.min(y0, q[1] - e); y1 = Math.max(y1, q[1] + e);
      }
    }
    return { u0, u1, y0, y1 };
  };

  R.shiftPart = function (p, du, dy) {
    if (p.type === 'image') return Object.assign({}, p, { u0: p.u0 + du, y0: p.y0 + dy });
    return Object.assign({}, p, { pts: p.pts.map(q => [q[0] + du, q[1] + dy]) });
  };
  R.scalePart = function (p, k, cu, cy) {
    if (p.type === 'image') return Object.assign({}, p, { u0: cu + (p.u0 - cu) * k, y0: cy + (p.y0 - cy) * k, w: p.w * k, h: p.h * k });
    return Object.assign({}, p, { pts: p.pts.map(q => [cu + (q[0] - cu) * k, cy + (q[1] - cy) * k]), wmul: (p.wmul || 1) * (p.keepWeight ? 1 : 1) });
  };
})(typeof window !== 'undefined' ? window : globalThis);
