/* رسم — tracer: anti-aliased bitmap → clean Bézier outlines.
 * marching squares (sub-pixel) → corner detection → least-squares cubic fitting (Schneider). */
(function (root) {
  'use strict';
  const R = root.Rasm;
  const T = R.trace = {};

  /* ---------------------------------------------------------------- contours */
  // field: Float32Array (w*h) coverage 0..1. Returns closed polygons (pixel coords, y down),
  // oriented so that the filled region is always on the same side.
  T.contours = function (field, w, h, thr) {
    thr = thr == null ? 0.5 : thr;
    const NH = w * h;
    const next = new Int32Array(2 * NH).fill(-1);
    const px = new Float32Array(2 * NH), py = new Float32Array(2 * NH);
    const v = (i, j) => field[j * w + i];
    const Hid = (i, j) => j * w + i;            // edge (i,j)-(i+1,j)
    const Vid = (i, j) => NH + j * w + i;       // edge (i,j)-(i,j+1)
    const setH = (i, j) => { const id = Hid(i, j); const a = v(i, j), b = v(i + 1, j); const t = (thr - a) / (b - a); px[id] = i + t + 0.5; py[id] = j + 0.5; return id; };
    const setV = (i, j) => { const id = Vid(i, j); const a = v(i, j), b = v(i, j + 1); const t = (thr - a) / (b - a); px[id] = i + 0.5; py[id] = j + t + 0.5; return id; };
    const starts = [];
    for (let j = 0; j < h - 1; j++) {
      for (let i = 0; i < w - 1; i++) {
        const c0 = v(i, j), c1 = v(i + 1, j), c2 = v(i + 1, j + 1), c3 = v(i, j + 1);
        const k = (c0 >= thr ? 1 : 0) | (c1 >= thr ? 2 : 0) | (c2 >= thr ? 4 : 0) | (c3 >= thr ? 8 : 0);
        if (k === 0 || k === 15) continue;
        const E = [() => setH(i, j), () => setV(i + 1, j), () => setH(i, j + 1), () => setV(i, j)];
        const corner = [[i, j, c0], [i + 1, j, c1], [i + 1, j + 1, c2], [i, j + 1, c3]];
        let pairs;
        switch (k) {
          case 1: case 14: pairs = [[3, 0]]; break;
          case 2: case 13: pairs = [[0, 1]]; break;
          case 3: case 12: pairs = [[3, 1]]; break;
          case 4: case 11: pairs = [[1, 2]]; break;
          case 6: case 9: pairs = [[0, 2]]; break;
          case 7: case 8: pairs = [[2, 3]]; break;
          case 5: pairs = ((c0 + c1 + c2 + c3) / 4 >= thr) ? [[0, 1], [2, 3]] : [[3, 0], [1, 2]]; break;
          case 10: pairs = ((c0 + c1 + c2 + c3) / 4 >= thr) ? [[3, 0], [1, 2]] : [[0, 1], [2, 3]]; break;
        }
        for (const [ea, eb] of pairs) {
          const A = E[ea](), B = E[eb]();
          // reference corner: shared corner of adjacent edges, or first corner of edge ea
          let ci;
          const s = ea < eb ? ea * 10 + eb : eb * 10 + ea;
          if (s === 1) ci = 1; else if (s === 12) ci = 2; else if (s === 23) ci = 3; else if (s === 3) ci = 0;
          else ci = ea === 0 ? 0 : ea === 1 ? 1 : ea === 2 ? 3 : 0;
          const [cx, cy, cv] = corner[ci];
          const inside = cv >= thr;
          const cross = (px[B] - px[A]) * (cy + 0.5 - py[A]) - (py[B] - py[A]) * (cx + 0.5 - px[A]);
          const keep = inside ? cross > 0 : cross < 0;
          if (keep) { next[A] = B; starts.push(A); } else { next[B] = A; starts.push(B); }
        }
      }
    }
    const seen = new Uint8Array(2 * NH);
    const polys = [];
    for (const s of starts) {
      if (seen[s]) continue;
      const poly = [];
      let e = s, guard = 0;
      while (e >= 0 && !seen[e] && guard++ < 1e6) {
        seen[e] = 1;
        poly.push([px[e], py[e]]);
        e = next[e];
      }
      if (poly.length >= 3) polys.push(poly);
    }
    return polys;
  };

  /* ---------------------------------------------------------------- fitting helpers */
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
  const mul = (a, k) => [a[0] * k, a[1] * k];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
  const len = (a) => Math.hypot(a[0], a[1]);
  const norm = (a) => { const l = len(a) || 1; return [a[0] / l, a[1] / l]; };
  const d2 = (a, b) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
  const bez = (Q, t) => {
    const mt = 1 - t, a = mt * mt * mt, b = 3 * mt * mt * t, c = 3 * mt * t * t, d = t * t * t;
    return [a * Q[0][0] + b * Q[1][0] + c * Q[2][0] + d * Q[3][0], a * Q[0][1] + b * Q[1][1] + c * Q[2][1] + d * Q[3][1]];
  };

  function chordParam(d, first, last) {
    const u = [0];
    for (let i = first + 1; i <= last; i++) u.push(u[u.length - 1] + len(sub(d[i], d[i - 1])));
    const L = u[u.length - 1] || 1;
    return u.map(x => x / L);
  }

  function generateBezier(d, first, last, u, t1, t2) {
    const p0 = d[first], p3 = d[last];
    let C00 = 0, C01 = 0, C11 = 0, X0 = 0, X1 = 0;
    for (let i = 0; i <= last - first; i++) {
      const t = u[i], mt = 1 - t;
      const b0 = mt * mt * mt, b1 = 3 * mt * mt * t, b2 = 3 * mt * t * t, b3 = t * t * t;
      const A0 = mul(t1, b1), A1 = mul(t2, b2);
      C00 += dot(A0, A0); C01 += dot(A0, A1); C11 += dot(A1, A1);
      const tmp = sub(d[first + i], add(mul(p0, b0 + b1), mul(p3, b2 + b3)));
      X0 += dot(A0, tmp); X1 += dot(A1, tmp);
    }
    const det = C00 * C11 - C01 * C01;
    let al = det === 0 ? 0 : (X0 * C11 - X1 * C01) / det;
    let ar = det === 0 ? 0 : (C00 * X1 - C01 * X0) / det;
    const seg = len(sub(p3, p0));
    const eps = 1e-6 * seg;
    if (al < eps || ar < eps || al > seg * 2 || ar > seg * 2) { al = ar = seg / 3; }
    return [p0, add(p0, mul(t1, al)), add(p3, mul(t2, ar)), p3];
  }

  function newton(Q, P, u) {
    const q = bez(Q, u);
    const q1 = [0, 1, 2].map(i => mul(sub(Q[i + 1], Q[i]), 3));
    const q2 = [0, 1].map(i => mul(sub(q1[i + 1], q1[i]), 2));
    const mt = 1 - u;
    const d1 = add(add(mul(q1[0], mt * mt), mul(q1[1], 2 * mt * u)), mul(q1[2], u * u));
    const dd = add(mul(q2[0], mt), mul(q2[1], u));
    const num = (q[0] - P[0]) * d1[0] + (q[1] - P[1]) * d1[1];
    const den = d1[0] * d1[0] + d1[1] * d1[1] + (q[0] - P[0]) * dd[0] + (q[1] - P[1]) * dd[1];
    if (Math.abs(den) < 1e-12) return u;
    return Math.min(1, Math.max(0, u - num / den));
  }

  function maxError(d, first, last, Q, u) {
    let max = 0, split = Math.floor((first + last) / 2);
    for (let i = first + 1; i < last; i++) {
      const e = d2(bez(Q, u[i - first]), d[i]);
      if (e >= max) { max = e; split = i; }
    }
    return [max, split];
  }

  function centerTangent(d, c, first, last) {
    const a = Math.max(first, c - 2), b = Math.min(last, c + 2);
    return norm(sub(d[a], d[b]));
  }

  function fitCubic(d, first, last, t1, t2, err, out, depth) {
    const n = last - first + 1;
    if (n <= 2 || depth > 24) {
      const dist = len(sub(d[last], d[first])) / 3;
      out.push([d[first], add(d[first], mul(t1, dist)), add(d[last], mul(t2, dist)), d[last]]);
      return;
    }
    let u = chordParam(d, first, last);
    let Q = generateBezier(d, first, last, u, t1, t2);
    let [e, split] = maxError(d, first, last, Q, u);
    if (e < err) { out.push(Q); return; }
    if (e < err * 6) {
      for (let k = 0; k < 12; k++) {
        const up = u.map((t, i) => newton(Q, d[first + i], t));
        Q = generateBezier(d, first, last, up, t1, t2);
        [e, split] = maxError(d, first, last, Q, up);
        if (e < err) { out.push(Q); return; }
        u = up;
      }
    }
    if (split <= first) split = first + 1;
    if (split >= last) split = last - 1;
    const tc = centerTangent(d, split, first, last);
    fitCubic(d, first, split, t1, tc, err, out, depth + 1);
    fitCubic(d, split, last, mul(tc, -1), t2, err, out, depth + 1);
  }

  /* ---------------------------------------------------------------- polygon → segments */
  function resample(poly) {
    // drop points closer than 0.3px
    const out = [];
    for (const p of poly) { const q = out[out.length - 1]; if (!q || d2(p, q) > 0.09) out.push(p); }
    if (out.length > 3 && d2(out[0], out[out.length - 1]) < 0.09) out.pop();
    return out;
  }

  function findCorners(P, k, thrDeg) {
    const n = P.length, ang = new Float32Array(n);
    const at = (i) => P[((i % n) + n) % n];
    for (let i = 0; i < n; i++) {
      // walk outwards until we are k px away on both sides
      let a = i - 1, b = i + 1, ga = 0, gb = 0;
      while (d2(at(a), P[i]) < k * k && ga++ < n) a--;
      while (d2(at(b), P[i]) < k * k && gb++ < n) b++;
      const v1 = norm(sub(P[i], at(a))), v2 = norm(sub(at(b), P[i]));
      ang[i] = Math.acos(Math.max(-1, Math.min(1, dot(v1, v2)))) * 180 / Math.PI;
    }
    const corners = [];
    const win = 3;
    for (let i = 0; i < n; i++) {
      if (ang[i] < thrDeg) continue;
      let isMax = true;
      for (let j = -win; j <= win; j++) { if (j && ang[((i + j) % n + n) % n] > ang[i]) { isMax = false; break; } }
      if (isMax && !(corners.length && i - corners[corners.length - 1] <= win)) corners.push(i);
    }
    return corners;
  }

  function smooth(P, fixed) {
    const n = P.length, out = P.slice();
    for (let i = 0; i < n; i++) {
      if (fixed.has(i)) continue;
      const a = P[(i - 1 + n) % n], b = P[(i + 1) % n];
      if (fixed.has((i - 1 + n) % n) || fixed.has((i + 1) % n)) continue;
      out[i] = [(a[0] + 2 * P[i][0] + b[0]) / 4, (a[1] + 2 * P[i][1] + b[1]) / 4];
    }
    return out;
  }

  // returns [{start:[x,y], segs:[{c1,c2,p} | {p}]}] in pixel coords
  T.fitPolygon = function (poly, tol) {
    let P = resample(poly);
    if (P.length < 4) return null;
    const n0 = P.length;
    const corners = findCorners(P, 3.2, 38);
    const fixed = new Set(corners);
    P = smooth(smooth(P, fixed), fixed);
    let breaks = corners.slice();
    const isCorner = new Set(corners);
    if (!breaks.length) {
      // smooth closed loop: break at the four extrema with axis-aligned tangents
      let ix0 = 0, ix1 = 0, iy0 = 0, iy1 = 0;
      for (let i = 1; i < n0; i++) {
        if (P[i][0] < P[ix0][0]) ix0 = i; if (P[i][0] > P[ix1][0]) ix1 = i;
        if (P[i][1] < P[iy0][1]) iy0 = i; if (P[i][1] > P[iy1][1]) iy1 = i;
      }
      breaks = Array.from(new Set([ix0, ix1, iy0, iy1])).sort((a, b) => a - b);
    }
    const axisAt = {};
    if (!corners.length) {
      for (const i of breaks) {
        // vertical tangent at x-extrema, horizontal at y-extrema
        const prev = P[(i - 3 + n0) % n0], nx = P[(i + 3) % n0];
        const dx = Math.abs(nx[0] - prev[0]), dy = Math.abs(nx[1] - prev[1]);
        axisAt[i] = dx > dy ? 'h' : 'v';
      }
    }
    const n = P.length;
    const contour = { start: P[breaks[0]].slice(), segs: [] };
    const err = tol * tol;
    for (let b = 0; b < breaks.length; b++) {
      const i0 = breaks[b], i1 = breaks[(b + 1) % breaks.length];
      // collect run points i0..i1 (wrapping)
      const run = [];
      if (breaks.length === 1) {
        for (let k = 0; k <= n; k++) run.push(P[(i0 + k) % n]);
      } else {
        let i = i0, guard = 0;
        do { run.push(P[i]); i = (i + 1) % n; } while (i !== (i1 + 1) % n && guard++ < n + 2);
      }
      if (run.length < 2) continue;
      const last = run.length - 1;
      const kk = Math.min(3, last);
      let t1 = norm(sub(run[kk], run[0]));
      let t2 = norm(sub(run[last - kk], run[last]));
      if (axisAt[i0]) t1 = axisAt[i0] === 'h' ? [Math.sign(run[kk][0] - run[0][0]) || 1, 0] : [0, Math.sign(run[kk][1] - run[0][1]) || 1];
      if (axisAt[i1]) t2 = axisAt[i1] === 'h' ? [Math.sign(run[last - kk][0] - run[last][0]) || 1, 0] : [0, Math.sign(run[last - kk][1] - run[last][1]) || 1];
      const out = [];
      fitCubic(run, 0, last, t1, t2, err, out, 0);
      for (const Q of out) {
        // straight? → line
        const ch = sub(Q[3], Q[0]), L = len(ch);
        let straight = false;
        if (L > 0.5) {
          const nrm = [-ch[1] / L, ch[0] / L];
          const e1 = Math.abs(dot(sub(Q[1], Q[0]), nrm)), e2 = Math.abs(dot(sub(Q[2], Q[0]), nrm));
          const s1 = dot(sub(Q[1], Q[0]), ch) / (L * L), s2 = dot(sub(Q[2], Q[0]), ch) / (L * L);
          straight = e1 < 0.18 && e2 < 0.18 && s1 > -0.05 && s1 < 1.05 && s2 > -0.05 && s2 < 1.05;
        } else straight = true;
        contour.segs.push(straight ? { p: Q[3] } : { c1: Q[1], c2: Q[2], p: Q[3] });
      }
    }
    return contour;
  };

  T.area = function (poly) {
    let a = 0;
    for (let i = 0, n = poly.length; i < n; i++) { const p = poly[i], q = poly[(i + 1) % n]; a += p[0] * q[1] - q[0] * p[1]; }
    return a / 2;
  };

  /* trace an alpha canvas region → contours in pixel space, outer contours CCW in y-up */
  T.traceAlpha = function (ctx, w, h, tol) {
    const data = ctx.getImageData(0, 0, w, h).data;
    const field = new Float32Array(w * h);
    for (let i = 0, j = 3; i < w * h; i++, j += 4) field[i] = data[j] / 255;
    const polys = T.contours(field, w, h, 0.5).filter(p => Math.abs(T.area(p)) > 2);
    if (!polys.length) return [];
    // orientation in y-up = mirror of y-down; make the biggest contour positive in y-up (negative in y-down)
    let big = polys[0];
    for (const p of polys) if (Math.abs(T.area(p)) > Math.abs(T.area(big))) big = p;
    const flip = T.area(big) > 0; // y-down positive → y-up negative → reverse everything
    const out = [];
    for (const p of polys) {
      const pp = flip ? p.slice().reverse() : p;
      const c = T.fitPolygon(pp, tol || 0.5);
      if (c && c.segs.length) out.push(c);
    }
    return out;
  };
})(typeof window !== 'undefined' ? window : globalThis);
