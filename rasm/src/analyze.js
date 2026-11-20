/* رسم — logo analysis (قياس الشعار): measure the logo's "DNA" and pull key shapes out of it.
 * Everything runs locally on a canvas; nothing is uploaded anywhere. */
(function (root) {
  'use strict';
  const R = root.Rasm;
  const A = R.analyze = {};
  const MAX_SIDE = 900;

  /* ---------------------------------------------------------------- loading */
  A.loadFile = function (file) {
    return new Promise((resolve, reject) => {
      const isSVG = /svg/i.test(file.type) || /\.svg$/i.test(file.name || '');
      const finish = (url, revoke) => {
        const img = new Image();
        img.onload = () => {
          let w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
          if (!w || !h) { w = MAX_SIDE; h = MAX_SIDE; }
          const k = MAX_SIDE / Math.max(w, h);
          const c = document.createElement('canvas');
          c.width = Math.max(8, Math.round(w * k)); c.height = Math.max(8, Math.round(h * k));
          const x = c.getContext('2d', { willReadFrequently: true });
          x.imageSmoothingQuality = 'high';
          x.drawImage(img, 0, 0, c.width, c.height);
          if (revoke) URL.revokeObjectURL(url);
          resolve(c);
        };
        img.onerror = () => { if (revoke) URL.revokeObjectURL(url); reject(new Error('تعذّرت قراءة الصورة')); };
        img.src = url;
      };
      if (isSVG) {
        const fr = new FileReader();
        fr.onload = () => {
          let txt = String(fr.result);
          // make sure the SVG has an intrinsic size
          const vb = txt.match(/viewBox\s*=\s*["']\s*([-\d.e]+)[\s,]+([-\d.e]+)[\s,]+([-\d.e]+)[\s,]+([-\d.e]+)/i);
          const hasW = /<svg[^>]*\swidth\s*=/i.test(txt), hasH = /<svg[^>]*\sheight\s*=/i.test(txt);
          if (vb && (!hasW || !hasH)) {
            const vw = parseFloat(vb[3]), vh = parseFloat(vb[4]);
            const k = MAX_SIDE / Math.max(vw, vh);
            txt = txt.replace(/<svg/i, `<svg width="${Math.round(vw * k)}" height="${Math.round(vh * k)}"`);
          }
          const url = URL.createObjectURL(new Blob([txt], { type: 'image/svg+xml' }));
          finish(url, true);
        };
        fr.onerror = () => reject(new Error('تعذّرت قراءة الملف'));
        fr.readAsText(file);
      } else finish(URL.createObjectURL(file), true);
    });
  };

  /* ---------------------------------------------------------------- primitives */
  function otsu(lum) {
    const hist = new Float64Array(256);
    for (let i = 0; i < lum.length; i++) hist[Math.max(0, Math.min(255, lum[i] | 0))]++;
    const total = lum.length;
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0, wB = 0, best = 0, t = 128;
    for (let i = 0; i < 256; i++) {
      wB += hist[i]; if (!wB) continue;
      const wF = total - wB; if (!wF) break;
      sumB += i * hist[i];
      const mB = sumB / wB, mF = (sum - sumB) / wF;
      const v = wB * wF * (mB - mF) * (mB - mF);
      if (v > best) { best = v; t = i; }
    }
    return t + 0.5;
  }

  function label(mask, w, h, eight, value) {
    const lab = new Int32Array(w * h).fill(-1);
    const comps = [];
    const stack = new Int32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      if (mask[i] !== value || lab[i] >= 0) continue;
      const id = comps.length;
      const c = { id, area: 0, x0: w, y0: h, x1: -1, y1: -1, sx: 0, sy: 0, border: false };
      let sp = 0; stack[sp++] = i; lab[i] = id;
      while (sp) {
        const p = stack[--sp];
        const x = p % w, y = (p / w) | 0;
        c.area++; c.sx += x; c.sy += y;
        if (x < c.x0) c.x0 = x; if (x > c.x1) c.x1 = x; if (y < c.y0) c.y0 = y; if (y > c.y1) c.y1 = y;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1) c.border = true;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          if (!eight && dx && dy) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q = ny * w + nx;
          if (mask[q] === value && lab[q] < 0) { lab[q] = id; stack[sp++] = q; }
        }
      }
      comps.push(c);
    }
    return { lab, comps };
  }

  function edt(mask, w, h) {
    const INF = 1e20, n = Math.max(w, h);
    const f = new Float64Array(n), d = new Float64Array(n), z = new Float64Array(n + 1);
    const v = new Int32Array(n);
    const g = new Float64Array(w * h);
    for (let i = 0; i < w * h; i++) g[i] = mask[i] ? INF : 0;
    const pass = (len) => {
      let k = 0; v[0] = 0; z[0] = -INF; z[1] = INF;
      for (let q = 1; q < len; q++) {
        let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        while (s <= z[k]) { k--; s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]); }
        k++; v[k] = q; z[k] = s; z[k + 1] = INF;
      }
      k = 0;
      for (let q = 0; q < len; q++) { while (z[k + 1] < q) k++; d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]; }
    };
    for (let x = 0; x < w; x++) { for (let y = 0; y < h; y++) f[y] = g[y * w + x]; pass(h); for (let y = 0; y < h; y++) g[y * w + x] = d[y]; }
    for (let y = 0; y < h; y++) { for (let x = 0; x < w; x++) f[x] = g[y * w + x]; pass(w); for (let x = 0; x < w; x++) g[y * w + x] = d[x]; }
    const out = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) out[i] = Math.sqrt(g[i]);
    return out;
  }

  function thin(mask, w, h) {
    const m = mask.slice();
    let list = [];
    for (let i = 0; i < w * h; i++) if (m[i]) list.push(i);
    let changed = true, guard = 0;
    while (changed && guard++ < 200) {
      changed = false;
      for (let pass = 0; pass < 2; pass++) {
        const del = [];
        for (const i of list) {
          if (!m[i]) continue;
          const x = i % w, y = (i / w) | 0;
          if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) { del.push(i); continue; }
          const p2 = m[i - w], p3 = m[i - w + 1], p4 = m[i + 1], p5 = m[i + w + 1], p6 = m[i + w], p7 = m[i + w - 1], p8 = m[i - 1], p9 = m[i - w - 1];
          const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (B < 2 || B > 6) continue;
          const T = (p2 === 0 && p3 === 1) + (p3 === 0 && p4 === 1) + (p4 === 0 && p5 === 1) + (p5 === 0 && p6 === 1) +
            (p6 === 0 && p7 === 1) + (p7 === 0 && p8 === 1) + (p8 === 0 && p9 === 1) + (p9 === 0 && p2 === 1);
          if (T !== 1) continue;
          if (pass === 0) { if (p2 * p4 * p6) continue; if (p4 * p6 * p8) continue; }
          else { if (p2 * p4 * p8) continue; if (p2 * p6 * p8) continue; }
          del.push(i);
        }
        for (const i of del) m[i] = 0;
        if (del.length) changed = true;
      }
      list = list.filter(i => m[i]);
    }
    return m;
  }

  const median = (a) => { if (!a.length) return 0; const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; };
  const pct = (a, p) => { if (!a.length) return 0; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.round(p * (s.length - 1))))]; };

  /* ---------------------------------------------------------------- main analysis */
  A.run = function (canvas) {
    const w = canvas.width, h = canvas.height, N = w * h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const data = ctx.getImageData(0, 0, w, h).data;
    const lum = new Float32Array(N);
    let transparent = 0;
    for (let i = 0, j = 0; i < N; i++, j += 4) {
      const a = data[j + 3] / 255;
      if (data[j + 3] < 250) transparent++;
      // composite on white for the luminance path
      lum[i] = (0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2]) * a + 255 * (1 - a);
    }
    const fg = new Uint8Array(N), cov = new Float32Array(N);
    let mode;
    if (transparent > N * 0.03) {
      mode = 'alpha';
      for (let i = 0, j = 3; i < N; i++, j += 4) { cov[i] = data[j] / 255; fg[i] = data[j] > 127 ? 1 : 0; }
    } else {
      mode = 'luminance';
      const t = otsu(lum);
      let dark = 0, nb = 0;
      for (let x = 0; x < w; x++) for (const y of [0, 1, h - 2, h - 1]) { nb++; if (lum[y * w + x] < t) dark++; }
      for (let y = 0; y < h; y++) for (const x of [0, 1, w - 2, w - 1]) { nb++; if (lum[y * w + x] < t) dark++; }
      const bgDark = dark > nb / 2;
      let s0 = 0, n0 = 0, s1 = 0, n1 = 0;
      for (let i = 0; i < N; i++) { const isFg = bgDark ? lum[i] >= t : lum[i] < t; fg[i] = isFg ? 1 : 0; if (isFg) { s1 += lum[i]; n1++; } else { s0 += lum[i]; n0++; } }
      const mb = n0 ? s0 / n0 : 255, mf = n1 ? s1 / n1 : 0;
      for (let i = 0; i < N; i++) cov[i] = Math.max(0, Math.min(1, (lum[i] - mb) / ((mf - mb) || 1)));
    }
    // remove specks
    let { lab, comps } = label(fg, w, h, true, 1);
    const minArea = Math.max(10, N * 0.00012);
    for (const c of comps) if (c.area < minArea) { for (let y = c.y0; y <= c.y1; y++) for (let x = c.x0; x <= c.x1; x++) { const i = y * w + x; if (lab[i] === c.id) { fg[i] = 0; cov[i] = 0; } } }
    ({ lab, comps } = label(fg, w, h, true, 1));
    if (!comps.length) throw new Error('لم أجد حبراً في الصورة');

    // holes (counters)
    const bgl = label(fg, w, h, false, 0);
    const holes = bgl.comps.filter(c => !c.border && c.area > 6).map(c => {
      const yMid = Math.round(c.sy / c.area);
      let xl = c.x0 - 1; while (xl > 0 && !fg[yMid * w + xl]) xl--;
      return Object.assign(c, { owner: lab[yMid * w + Math.max(0, xl)] });
    });

    const dist = edt(fg, w, h);
    const sk = thin(fg, w, h);

    // skeleton pixels: thickness, neighbours
    const skl = [];
    for (let i = 0; i < N; i++) if (sk[i]) skl.push(i);
    const thick = [];
    for (const i of skl) thick.push(Math.max(1, 2 * dist[i] - 1));
    let Wpx = median(thick) || 4;

    // direction of each skeleton pixel (PCA over a small window)
    const rr = Math.max(3, Math.round(Wpx * 0.7));
    const dir = new Float32Array(skl.length);
    for (let k = 0; k < skl.length; k++) {
      const i = skl[k], x = i % w, y = (i / w) | 0;
      let n = 0, mx = 0, my = 0, sxx = 0, syy = 0, sxy = 0;
      for (let dy = -rr; dy <= rr; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -rr; dx <= rr; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= w) continue;
          if (!sk[yy * w + xx]) continue;
          n++; mx += dx; my += dy; sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
        }
      }
      if (n < 3) { dir[k] = NaN; continue; }
      mx /= n; my /= n;
      const cxx = sxx / n - mx * mx, cyy = syy / n - my * my, cxy = sxy / n - mx * my;
      let a = 0.5 * Math.atan2(2 * cxy, cxx - cyy) * 180 / Math.PI; // principal axis, image coords
      if (a < 0) a += 180;
      dir[k] = a; // 0 = horizontal, 90 = vertical (y down)
    }

    // contrast: thickness by direction
    const bins = Array.from({ length: 6 }, () => []);
    for (let k = 0; k < skl.length; k++) if (!isNaN(dir[k])) bins[Math.min(5, Math.floor(dir[k] / 30))].push(thick[k]);
    const binMed = bins.map(b => b.length >= Math.max(8, skl.length * 0.04) ? median(b) : null);
    const valid = binMed.map((v, i) => [v, i]).filter(x => x[0] != null);
    let contrast = 0, nibAngle = 90;
    if (valid.length >= 2) {
      const mx = valid.reduce((a, b) => b[0] > a[0] ? b : a), mn = valid.reduce((a, b) => b[0] < a[0] ? b : a);
      contrast = 1 - mn[0] / mx[0];
      // thickest strokes run along direction mx; the nib's long axis is perpendicular to it
      const strokeDir = mx[1] * 30 + 15; // image coords (y down)
      nibAngle = (180 - strokeDir + 90) % 180;
      Wpx = mx[0];
      if (contrast < 0.16) { contrast = 0; Wpx = median(thick); }
    }

    // baseline: densest band of horizontal skeleton
    const hist = new Float32Array(h);
    for (let k = 0; k < skl.length; k++) {
      const a = dir[k]; if (isNaN(a)) continue;
      if (a < 14 || a > 166) hist[(skl[k] / w) | 0] += 1;
    }
    const sm = new Float32Array(h), rad = Math.max(2, Math.round(Wpx / 3));
    for (let y = 0; y < h; y++) { let s = 0; for (let d = -rad; d <= rad; d++) { const yy = y + d; if (yy >= 0 && yy < h) s += hist[yy]; } sm[y] = s; }
    let inkTop = h, inkBot = 0, inkL = w, inkR = 0;
    for (const c of comps) { inkTop = Math.min(inkTop, c.y0); inkBot = Math.max(inkBot, c.y1); inkL = Math.min(inkL, c.x0); inkR = Math.max(inkR, c.x1); }
    let by = -1, bv = 0;
    for (let y = inkTop; y <= inkBot; y++) if (sm[y] > bv) { bv = sm[y]; by = y; }
    const hasBaseline = by >= 0 && bv > Wpx * 2;
    const baseC = hasBaseline ? by : inkBot - Wpx / 2;       // centre of baseline stroke
    const baseline = baseC + Wpx / 2;                        // outer bottom of the baseline stroke

    // vertical runs → stems and teeth
    const vert = new Uint8Array(N);
    for (let k = 0; k < skl.length; k++) { const a = dir[k]; if (!isNaN(a) && a > 72 && a < 108) vert[skl[k]] = 1; }
    const vl = label(vert, w, h, true, 1);
    const runs = vl.comps.filter(c => (c.y1 - c.y0) > Wpx * 0.9).map(c => {
      const xm = Math.round(c.sx / c.area);
      let top = c.y0; while (top > 0 && fg[(top - 1) * w + xm]) top--;
      return { x: xm, y0: c.y0, y1: c.y1, top, len: c.y1 - c.y0, area: c.area, slant: 0, comp: c };
    });
    // slant per run (dx/dy across the run)
    for (const r of runs) {
      let sx0 = 0, n0 = 0, sx1 = 0, n1 = 0; const mid = (r.y0 + r.y1) / 2;
      for (let y = r.comp.y0; y <= r.comp.y1; y++) for (let x = r.comp.x0; x <= r.comp.x1; x++) {
        if (vl.lab[y * w + x] !== r.comp.id) continue;
        if (y < mid) { sx0 += x; n0++; } else { sx1 += x; n1++; }
      }
      if (n0 && n1) r.slant = Math.atan2(sx1 / n1 - sx0 / n0, (r.y1 - r.y0) / 2) * 180 / Math.PI;
    }
    const grounded = runs.filter(r => Math.abs(r.y1 - baseC) < Wpx * 1.6);
    const tallest = grounded.slice().sort((a, b) => (baseline - a.top) - (baseline - b.top))[grounded.length - 1];
    let Apx = tallest ? baseline - tallest.top : 0;
    const teeth = grounded.filter(r => (baseline - r.top) < Apx * 0.78 && (baseline - r.top) > Wpx * 1.6);
    let Xpx = teeth.length ? median(teeth.map(r => baseline - r.top)) : 0;
    let hasAscender = !!tallest && (!Xpx || Apx > Xpx * 1.3) && Apx > Wpx * 4.2;
    if (!hasAscender) {
      const above = baseline - inkTop;
      Xpx = Xpx || above;
      Apx = Math.max(Xpx / 0.62, Wpx * 6);
    }
    if (!Xpx) Xpx = Apx * 0.62;
    const Dpx = Math.max(0, inkBot - baseline);
    const slant = runs.length ? -median(runs.filter(r => r.len > Wpx * 2).map(r => r.slant)) : 0;

    // dots
    const solidity = (c) => {
      const pts = [];
      for (let y = c.y0; y <= c.y1; y++) {
        let a = -1, b = -1;
        for (let x = c.x0; x <= c.x1; x++) if (lab[y * w + x] === c.id) { if (a < 0) a = x; b = x; }
        if (a >= 0) { pts.push([a, y], [b + 1, y], [a, y + 1], [b + 1, y + 1]); }
      }
      pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
      const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
      const lo = [], up = [];
      for (const p of pts) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], p) <= 0) lo.pop(); lo.push(p); }
      for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], p) <= 0) up.pop(); up.push(p); }
      const hull = lo.slice(0, -1).concat(up.slice(0, -1));
      const ha = Math.abs(R.trace.area(hull)) || 1;
      return c.area / ha;
    };
    const dots = comps.filter(c => {
      const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1;
      const fill = c.area / (bw * bh);
      const ownsHole = holes.some(hh => hh.owner === c.id);
      return !ownsHole && c.area < Wpx * Wpx * 4.2 && Math.max(bw, bh) < Wpx * 2.8 && Math.min(bw, bh) > Wpx * 0.45 && bw / bh < 2.3 && bh / bw < 2.3 && fill > 0.42 && solidity(c) > 0.86;
    }).map(c => {
      const bw = c.x1 - c.x0 + 1, bh = c.y1 - c.y0 + 1, fill = c.area / (bw * bh);
      return Object.assign(c, { bw, bh, fill, shape: fill > 0.88 ? 'square' : fill > 0.66 ? 'circle' : 'diamond' });
    });

    // corners / roundness / terminals from the outline (ink is always on the same side of a contour)
    const polys = R.trace.contours(cov, w, h, 0.5).filter(p => p.length > 12);
    let straight = 0, curved = 0, sharp = 0, flatEnds = 0, roundEnds = 0;
    const step = Math.max(1, Math.round(Math.max(2, Wpx * 0.2)));
    const bigPolys = polys.filter(P => { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const q of P) { if (q[0] < x0) x0 = q[0]; if (q[0] > x1) x1 = q[0]; if (q[1] < y0) y0 = q[1]; if (q[1] > y1) y1 = q[1]; } return Math.max(x1 - x0, y1 - y0) > Wpx * 3; });
    const feats = bigPolys.map(P => {
      const n = P.length;
      const th = new Float64Array(n);
      for (let i = 0; i < n; i++) { const a = P[(i - step + n) % n], b = P[(i + step) % n]; th[i] = Math.atan2(b[1] - a[1], b[0] - a[0]); }
      const cum = new Float64Array(n + 1);
      for (let i = 0; i < n; i++) { let d = th[(i + 1) % n] - th[i]; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; cum[i + 1] = cum[i] + d; }
      return { P, n, cum, total: cum[n], area: Math.abs(R.trace.area(P)) };
    });
    let convexSign = 1;
    if (feats.length) { const big = feats.reduce((a, b) => b.area > a.area ? b : a); convexSign = Math.sign(big.total) || 1; }
    const deg = 180 / Math.PI;
    for (const f of feats) {
      const { P, n, cum } = f;
      const C = (j) => { const q = Math.floor(j / n); return cum[j - q * n] + f.total * q; };
      const turnAt = (i, k) => (C(i + k) - C(i - k)) * deg;
      const L = new Float64Array(n);
      for (let i = 0; i < n; i++) L[i] = turnAt(i, step);
      const corners = [];
      for (let i = 0; i < n; i++) {
        const t = Math.abs(L[i]);
        if (t < 7) straight++; else if (t < 48) curved++;
        if (t >= 48) {
          let isMax = true;
          for (let d = -step; d <= step; d++) if (d && Math.abs(L[(i + d + n) % n]) > t) { isMax = false; break; }
          if (isMax) { sharp++; corners.push(i); }
        }
      }
      // flat terminal: two convex corners ~one stroke apart, turning ~180° together
      for (let j = 0; j < corners.length; j++) {
        const i0 = corners[j], i1 = corners[(j + 1) % corners.length];
        if (i0 === i1 || Math.sign(L[i0]) !== convexSign || Math.sign(L[i1]) !== convexSign) continue;
        const d = Math.hypot(P[i1][0] - P[i0][0], P[i1][1] - P[i0][1]);
        if (d > Wpx * 0.55 && d < Wpx * 1.6 && Math.abs(L[i0]) + Math.abs(L[i1]) > 140) flatEnds++;
      }
      // round terminal: ~180° of smooth convex turning within a semicircle's length
      const span = Math.max(6, Math.round(Wpx * 1.9));
      for (let i = 0; i < n;) {
        const tot = turnAt(i + (span >> 1), span >> 1);
        let smoothOK = true;
        for (let d = 0; d < span; d++) if (Math.abs(L[(i + d) % n]) >= 60) { smoothOK = false; break; }
        if (smoothOK && Math.sign(tot) === convexSign && Math.abs(tot) > 150 && Math.abs(tot) < 220) { roundEnds++; i += span; }
        else i += Math.max(1, span >> 3);
      }
    }
    const curvedFrac = curved / Math.max(1, straight + curved);
    const roundness = Math.max(0, Math.min(1, (curvedFrac - 0.08) / 0.4));
    const cap = roundEnds > flatEnds ? 'round' : 'butt';

    // the scale that maps the logo to the font (alif = 760 units)
    const Afont = 760;
    const scale = Afont / Apx;              // font units per pixel
    const Wfont = Math.max(24, Math.min(200, Wpx * scale));

    const res = {
      canvas, w, h, mode, fg, cov, lab, comps, holes, holeLab: bgl.lab, dist, sk, skl, dir, thick,
      Wpx, contrast, nibAngle, baseline, baseC, hasBaseline, Apx, Xpx, Dpx, hasAscender, slant,
      runs, tallest, teeth, dots, roundness, cap, curvedFrac, sharp, flatEnds, roundEnds, scale,
      ink: { x0: inkL, y0: inkTop, x1: inkR, y1: inkBot }
    };
    const dotMed = dots.length ? dots.slice().sort((a, b) => a.area - b.area)[dots.length >> 1] : null;
    res.dna = {
      weight: Math.round(Wfont),
      contrast: +contrast.toFixed(2),
      nibAngle: Math.round(nibAngle),
      nib: contrast > 0 ? (cap === 'butt' ? 'flat' : 'round') : 'round',
      cap,
      roundness: +roundness.toFixed(2),
      alif: Afont,
      xh: Math.round(Math.max(0.35 * Afont, Math.min(0.82 * Afont, Xpx * scale))),
      desc: Math.round(Math.max(150, Math.min(420, Dpx > Wpx ? Dpx * scale : 250))),
      slant: Math.abs(slant) > 3 ? Math.round(slant) : 0,
      dotShape: dotMed ? dotMed.shape : 'square',
      dotSize: dotMed ? +Math.max(0.8, Math.min(2.6, Math.max(dotMed.bw, dotMed.bh) / Wpx)).toFixed(2) : 1.3
    };
    res.dna.deep = Math.round(Math.max(res.dna.desc + 40, res.dna.desc * 1.3));
    // loop size from the counters
    const goodHoles = holes.filter(hh => { const hw = hh.x1 - hh.x0 + 1, hhh = hh.y1 - hh.y0 + 1; return hw > Wpx * 0.35 && hhh > Wpx * 0.35 && hw < Wpx * 9 && hhh < Wpx * 9; });
    res.goodHoles = goodHoles;
    const roundHoles = goodHoles.filter(hh => (hh.x1 - hh.x0) / (hh.y1 - hh.y0 + 1) < 1.6);
    if (roundHoles.length) {
      const hm = median(roundHoles.map(hh => hh.y1 - hh.y0 + 1));
      res.dna.loop = Math.round(Math.max(2.3 * Wfont, Math.min(0.9 * Afont, (hm + 2 * Wpx) * scale)));
    }
    return res;
  };

  /* ---------------------------------------------------------------- piece extraction */
  // copy soft coverage of the given rect (px) into an alpha-mask canvas, optionally limited by a pixel test
  function maskCanvas(an, x0, y0, x1, y1, test) {
    const w = an.w;
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(an.w - 1, Math.ceil(x1)); y1 = Math.min(an.h - 1, Math.ceil(y1));
    // tight bounds
    let tx0 = x1, ty0 = y1, tx1 = x0, ty1 = y0;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const i = y * w + x;
      if (an.cov[i] > 0.04 && (!test || test(x, y, i))) { if (x < tx0) tx0 = x; if (x > tx1) tx1 = x; if (y < ty0) ty0 = y; if (y > ty1) ty1 = y; }
    }
    if (tx1 < tx0 || ty1 < ty0) return null;
    const cw = tx1 - tx0 + 1, ch = ty1 - ty0 + 1;
    const c = document.createElement('canvas'); c.width = cw; c.height = ch;
    const x = c.getContext('2d');
    const img = x.createImageData(cw, ch);
    for (let y = ty0; y <= ty1; y++) for (let xx = tx0; xx <= tx1; xx++) {
      const i = y * w + xx, j = ((y - ty0) * cw + (xx - tx0)) * 4;
      const ok = !test || test(xx, y, i);
      img.data[j + 3] = ok ? Math.round(an.cov[i] * 255) : 0;
    }
    x.putImageData(img, 0, 0);
    return { canvas: c, x0: tx0, y0: ty0, x1: tx1, y1: ty1 };
  }

  // turn a pixel crop into a slot override, keeping its height relative to the logo baseline
  A.toOverride = function (an, crop, slot) {
    const k = an.scale;
    const wu = (crop.x1 - crop.x0 + 1) * k, hu = (crop.y1 - crop.y0 + 1) * k;
    const o = { img: crop.canvas, wu, hu, y0: (an.baseline - (crop.y1 + 1)) * k, source: 'logo', rect: [crop.x0, crop.y0, crop.x1, crop.y1] };
    if (slot === 'dot') o.y0 = 0;
    return o;
  };

  /* user-drawn rectangle → piece for a slot */
  A.cropForSlot = function (an, rect, slot) {
    let [x0, y0, x1, y1] = rect;
    if (slot === 'stem' || slot === 'tooth' || slot === 'kaf') y1 = Math.min(y1, an.baseC);
    const crop = maskCanvas(an, x0, y0, x1, y1);
    return crop ? A.toOverride(an, crop, slot) : null;
  };

  /* automatic extraction: dot, stem, head loop, big loop */
  A.autoPieces = function (an) {
    const out = {};
    const W = an.Wpx;
    // dot: the median-size dot component
    if (an.dots.length) {
      const d = an.dots.slice().sort((a, b) => a.area - b.area)[an.dots.length >> 1];
      const crop = maskCanvas(an, d.x0 - 1, d.y0 - 1, d.x1 + 1, d.y1 + 1, (x, y, i) => an.lab[i] === d.id || an.cov[i] < 0.5);
      if (crop) out.dot = Object.assign(A.toOverride(an, crop, 'dot'), { why: 'أصغر عنصر متكرر مدمج — نقطة' });
    }
    // stem: the tallest grounded vertical run; keep its terminal, cut at the baseline centre
    if (an.hasAscender && an.tallest) {
      const r = an.tallest;
      const top = r.top, bot = an.baseC;
      const cid = an.lab[Math.round((r.y0 + r.y1) / 2) * an.w + r.x];
      const crop = maskCanvas(an, r.x - W * 1.2, top - 1, r.x + W * 1.2, bot, (x, y, i) => {
        const l = an.lab[i];
        if (l >= 0 && l !== cid) return false;
        const half = (y < top + W * 1.6) ? W * 1.2 : W * 0.53;
        return Math.abs(x - r.x) <= half;
      });
      if (crop) out.stem = Object.assign(A.toOverride(an, crop, 'stem'), { why: 'أطول خط عمودي يقف على السطر — الساق' });
    }
    // loops: counters with their ring of ink
    const ring = (hh) => {
      const pad = Math.ceil(W * 1.3);
      const cx0 = Math.max(0, hh.x0 - pad), cy0 = Math.max(0, hh.y0 - pad), cx1 = Math.min(an.w - 1, hh.x1 + pad), cy1 = Math.min(an.h - 1, hh.y1 + pad);
      const cw = cx1 - cx0 + 1, ch = cy1 - cy0 + 1;
      // distance from every pixel of the crop to the counter itself
      const notHole = new Uint8Array(cw * ch);
      for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) notHole[y * cw + x] = an.holeLab[(y + cy0) * an.w + (x + cx0)] === hh.id ? 0 : 1;
      const dd = edt(notHole, cw, ch);
      const lim = W * 1.06;
      return maskCanvas(an, cx0, cy0, cx1, cy1, (x, y, i) => {
        if (an.lab[i] >= 0 && an.lab[i] !== hh.owner) return false;
        return dd[(y - cy0) * cw + (x - cx0)] <= lim;
      });
    };
    const holes = an.goodHoles || [];
    const roundH = holes.filter(hh => (hh.x1 - hh.x0 + 1) / (hh.y1 - hh.y0 + 1) < 1.6).sort((a, b) => b.area - a.area);
    const wideH = holes.filter(hh => (hh.x1 - hh.x0 + 1) / (hh.y1 - hh.y0 + 1) >= 1.6).sort((a, b) => b.area - a.area);
    if (roundH.length) { const c = ring(roundH[0]); if (c) out.head = Object.assign(A.toOverride(an, c, 'head'), { why: 'فراغ داخلي مستدير — الرأس' }); }
    if (wideH.length) { const c = ring(wideH[0]); if (c) out.loop = Object.assign(A.toOverride(an, c, 'loop'), { why: 'فراغ داخلي ممدود — العروة' }); }
    return out;
  };

  /* uploaded piece image → tight alpha mask canvas */
  A.maskFromCanvas = function (canvas) {
    const w = canvas.width, h = canvas.height, N = w * h;
    const data = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data;
    let transparent = 0;
    const lum = new Float32Array(N);
    for (let i = 0, j = 0; i < N; i++, j += 4) { const a = data[j + 3] / 255; if (data[j + 3] < 250) transparent++; lum[i] = (0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2]) * a + 255 * (1 - a); }
    const cov = new Float32Array(N);
    if (transparent > N * 0.03) { for (let i = 0, j = 3; i < N; i++, j += 4) cov[i] = data[j] / 255; }
    else {
      const t = otsu(lum);
      let dark = 0, nb = 0;
      for (let x = 0; x < w; x++) for (const y of [0, h - 1]) { nb++; if (lum[y * w + x] < t) dark++; }
      for (let y = 0; y < h; y++) for (const x of [0, w - 1]) { nb++; if (lum[y * w + x] < t) dark++; }
      const bgDark = dark > nb / 2;
      let s0 = 0, n0 = 0, s1 = 0, n1 = 0;
      for (let i = 0; i < N; i++) { const f = bgDark ? lum[i] >= t : lum[i] < t; if (f) { s1 += lum[i]; n1++; } else { s0 += lum[i]; n0++; } }
      const mb = n0 ? s0 / n0 : 255, mf = n1 ? s1 / n1 : 0;
      for (let i = 0; i < N; i++) cov[i] = Math.max(0, Math.min(1, (lum[i] - mb) / ((mf - mb) || 1)));
    }
    const an = { w, h, cov };
    return maskCanvas(an, 0, 0, w - 1, h - 1);
  };

  /* overlay image for the UI: skeleton, baseline, detections */
  A.overlay = function (an, ctx, s, opts) {
    opts = opts || {};
    const W = an.Wpx;
    ctx.save();
    ctx.scale(s, s);
    if (opts.skeleton !== false) {
      ctx.fillStyle = opts.skColor || 'rgba(224,87,42,.9)';
      for (const i of an.skl) { const x = i % an.w, y = (i / an.w) | 0; ctx.fillRect(x - 0.5, y - 0.5, 1.6, 1.6); }
    }
    ctx.lineWidth = 1.5 / s;
    if (opts.lines !== false) {
      ctx.strokeStyle = opts.lineColor || 'rgba(25,25,25,.55)';
      ctx.setLineDash([6 / s, 4 / s]);
      const line = (y) => { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(an.w, y); ctx.stroke(); };
      line(an.baseline); line(an.baseline - an.Apx); line(an.baseline - an.Xpx); if (an.Dpx > W) line(an.baseline + an.Dpx);
      ctx.setLineDash([]);
    }
    if (opts.det !== false) {
      ctx.strokeStyle = opts.boxColor || 'rgba(59,91,253,.95)';
      for (const d of an.dots) ctx.strokeRect(d.x0 - 2, d.y0 - 2, d.x1 - d.x0 + 4, d.y1 - d.y0 + 4);
      if (an.tallest && an.hasAscender) { const r = an.tallest; ctx.strokeRect(r.x - W, r.top - 2, W * 2, an.baseline - r.top + 2); }
      for (const hh of (an.goodHoles || [])) ctx.strokeRect(hh.x0 - W, hh.y0 - W, hh.x1 - hh.x0 + 2 * W, hh.y1 - hh.y0 + 2 * W);
    }
    ctx.restore();
  };

  /* a sample logo drawn by the engine itself (so the app can be tried without a file) */
  A.sampleLogo = function (word, dna) {
    const d = R.clampDNA(Object.assign(R.defaultDNA(), dna || { weight: 118, roundness: 0.9, cap: 'round', dotShape: 'diamond', xh: 430, loop: 380, dotSize: 1.25 }));
    const m = R.metrics(d);
    const seq = R.shapeText(word || 'نور', 1);
    const gl = seq.map(g => {
      const spec = R.inventory().find(s => s.name === g.name);
      return spec ? R.buildSpec(m, spec) : null;
    }).filter(Boolean);
    const total = gl.reduce((a, g) => a + g.adv, 0);
    const s = Math.min(0.62, 760 / (total + 260), 760 / (m.A + m.DD + 560)), pad = 70;
    const c = document.createElement('canvas');
    c.width = Math.round(total * s + pad * 2); c.height = Math.round((m.A + m.DD + 300) * s + pad * 2);
    const x = c.getContext('2d');
    x.fillStyle = '#fbf6ee'; x.fillRect(0, 0, c.width, c.height);
    let px = pad + total * s;
    const base = pad + (m.A + 150) * s;
    for (const g of gl) { px -= g.adv * s; R.drawGlyph(x, g, d, { x: px, base, s, colorOf: () => '#1d2a44' }); }
    return c;
  };
})(typeof window !== 'undefined' ? window : globalThis);
