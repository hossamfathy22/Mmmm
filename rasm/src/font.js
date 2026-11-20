/* رسم — font compiler (OTF/CFF via opentype.js), Arabic shaper, logo export. */
(function (root) {
  'use strict';
  const R = root.Rasm;
  const F = R.font = {};

  /* draw a built glyph on a canvas. o: { x, base, s, colorOf, clip } */
  R.drawGlyph = function (ctx, built, dna, o) {
    const s = o.s;
    const view = { s, map: (u, y) => [o.x + (built.adv - u) * s, o.base - y * s] };
    ctx.save();
    if (o.clip !== false && built.clip && (built.clip.left != null || built.clip.right != null)) {
      const xl = built.clip.left != null ? view.map(built.clip.left, 0)[0] : -1e5;
      const xr = built.clip.right != null ? view.map(built.clip.right, 0)[0] : 1e5;
      ctx.beginPath(); ctx.rect(xl, -1e5, xr - xl, 2e5); ctx.clip();
    }
    R.pen.drawParts(ctx, built.parts, dna, view, o.colorOf);
    ctx.restore();
  };

  F.rasterize = function (built, dna, s) {
    const b = R.partBounds(built.parts, dna);
    if (!isFinite(b.u0)) return null;
    let u0 = b.u0, u1 = b.u1;
    if (built.clip && built.clip.right != null) u0 = Math.max(u0, built.clip.right);
    if (built.clip && built.clip.left != null) u1 = Math.min(u1, built.clip.left);
    const pad = 4;
    const W = Math.ceil((u1 - u0) * s) + pad * 2 + 2, H = Math.ceil((b.y1 - b.y0) * s) + pad * 2 + 2;
    const c = document.createElement('canvas'); c.width = W; c.height = H;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    const view = { s, map: (u, y) => [pad + (u1 - u) * s, pad + (b.y1 - y) * s] };
    ctx.save();
    if (built.clip && (built.clip.right != null || built.clip.left != null)) {
      const xl = built.clip.left != null ? view.map(built.clip.left, 0)[0] : -10;
      const xr = built.clip.right != null ? view.map(built.clip.right, 0)[0] : W + 10;
      ctx.beginPath(); ctx.rect(xl, -10, xr - xl, H + 20); ctx.clip();
    }
    R.pen.drawParts(ctx, built.parts, dna, view, () => '#000');
    ctx.restore();
    return { ctx, W, H, toFont: (X, Y) => [built.adv - (u1 - (X - pad) / s), b.y1 - (Y - pad) / s] };
  };

  /* outline of a built glyph as a list of contours in font units: [{start,[segs]}] */
  F.outline = function (built, dna, s) {
    const r = F.rasterize(built, dna, s);
    if (!r) return [];
    const contours = R.trace.traceAlpha(r.ctx, r.W, r.H, 0.42);
    const T = (p) => { const q = r.toFont(p[0], p[1]); return [Math.round(q[0]), Math.round(q[1])]; };
    return contours.map(c => ({ start: T(c.start), segs: c.segs.map(sg => sg.c1 ? { c1: T(sg.c1), c2: T(sg.c2), p: T(sg.p) } : { p: T(sg.p) }) }));
  };

  F.toPath = function (contours) {
    const path = new opentype.Path();
    for (const c of contours) {
      path.moveTo(c.start[0], c.start[1]);
      let cur = c.start;
      for (const sg of c.segs) {
        if (sg.c1) path.curveTo(sg.c1[0], sg.c1[1], sg.c2[0], sg.c2[1], sg.p[0], sg.p[1]);
        else if (sg.p[0] !== cur[0] || sg.p[1] !== cur[1]) path.lineTo(sg.p[0], sg.p[1]);
        cur = sg.p;
      }
      path.close();
    }
    return path;
  };

  F.svgPathData = function (contours, x, y, k) {
    // font units → svg (y down), scaled by k, origin at (x, y) = baseline-left
    const P = (p) => (x + p[0] * k).toFixed(2) + ' ' + (y - p[1] * k).toFixed(2);
    let d = '';
    for (const c of contours) {
      d += 'M' + P(c.start);
      for (const sg of c.segs) d += sg.c1 ? 'C' + P(sg.c1) + ' ' + P(sg.c2) + ' ' + P(sg.p) : 'L' + P(sg.p);
      d += 'Z';
    }
    return d;
  };

  function notdef() {
    const p = new opentype.Path();
    p.moveTo(60, 0); p.lineTo(440, 0); p.lineTo(440, 700); p.lineTo(60, 700); p.close();
    p.moveTo(120, 60); p.lineTo(120, 640); p.lineTo(380, 640); p.lineTo(380, 60); p.close();
    return new opentype.Glyph({ name: '.notdef', advanceWidth: 500, path: p });
  }

  /* build the whole font. engine: { dna, m }. Returns { font, outlines, index } */
  F.build = async function (engine, opts, onProgress) {
    opts = opts || {};
    const dna = engine.dna, m = engine.m;
    const s = Math.max(0.35, Math.min(0.9, 44 / dna.weight));
    const inv = R.inventory();
    const glyphs = [notdef()];
    const index = {}, outlines = {}, builtMap = {};
    let yMax = m.A, yMin = -m.DD;
    for (let i = 0; i < inv.length; i++) {
      const spec = inv[i];
      const built = R.buildSpec(m, spec);
      const contours = F.outline(built, dna, s);
      outlines[spec.name] = contours; builtMap[spec.name] = built;
      for (const c of contours) { for (const q of [c.start].concat(c.segs.map(x => x.p))) { yMax = Math.max(yMax, q[1]); yMin = Math.min(yMin, q[1]); } }
      const g = new opentype.Glyph({ name: spec.name, advanceWidth: Math.max(0, built.adv), path: F.toPath(contours) });
      if (spec.uni != null) g.addUnicode(spec.uni);
      index[spec.name] = glyphs.length;
      glyphs.push(g);
      if (onProgress && (i % 5 === 0 || i === inv.length - 1)) {
        if (onProgress((i + 1) / inv.length, spec.name) === false) throw Object.assign(new Error('cancelled'), { cancelled: true });
        await new Promise(r => setTimeout(r, 0));
      }
    }
    // Unicode presentation forms → positional glyphs (for apps without shaping)
    for (const name in R.presentation) {
      R.presentation[name].forEach((cp, k) => {
        const f = R.pfOrder[k]; const gn = f === 'isol' ? name : name + '.' + f;
        if (index[gn] != null) glyphs[index[gn]].addUnicode(cp);
      });
    }
    for (const lg of R.ligatures) {
      if (index[lg.name] != null) glyphs[index[lg.name]].addUnicode(lg.pf[0]);
      if (index[lg.name + '.fina'] != null) glyphs[index[lg.name + '.fina']].addUnicode(lg.pf[1]);
    }
    // Persian digits that share the Arabic shapes
    [0, 1, 2, 3, 7, 8, 9].forEach(d => { const gi = index['digit' + d]; if (gi != null) glyphs[gi].addUnicode(0x6F0 + d); });

    const family = (opts.family || 'Rasm').replace(/[^\w \-]/g, '').trim() || 'Rasm';
    const font = new opentype.Font({
      familyName: family, styleName: 'Regular', unitsPerEm: 1000, postScriptName: family.replace(/[^A-Za-z0-9-]/g, '') + '-Regular',
      ascender: Math.ceil(yMax + 20), descender: Math.floor(yMin - 20),
      designer: opts.designer || 'Rasm engine', version: '1.000',
      copyright: 'Generated on-device with Rasm', description: 'Arabic font generated from key shapes and recipes',
      glyphs
    });
    // OpenType: positional forms + lam-alef (features must be added alphabetically)
    const sub = font.substitution;
    for (const feat of ['fina', 'init', 'medi']) {
      for (const l of R.letters) {
        const a = index[l.name], b = index[l.name + '.' + feat];
        if (a != null && b != null) sub.addSingle(feat, { sub: a, by: b }, 'arab', 'dflt');
      }
    }
    for (const lg of R.ligatures) {
      const alF = index[lg.alef + '.fina'];
      if (index['lam.init'] != null && alF != null) sub.addLigature('rlig', { sub: [index['lam.init'], alF], by: index[lg.name] }, 'arab', 'dflt');
      if (index['lam.medi'] != null && alF != null) sub.addLigature('rlig', { sub: [index['lam.medi'], alF], by: index[lg.name + '.fina'] }, 'arab', 'dflt');
    }
    return { font, outlines, index, built: builtMap, family };
  };

  /* ---------------------------------------------------------------- shaper */
  const ALEFS = { 0x627: 'lamAlef', 0x623: 'lamAlefHamzaAbove', 0x625: 'lamAlefHamzaBelow', 0x622: 'lamAlefMadda' };
  const isTransparent = (cp) => (cp >= 0x64B && cp <= 0x65F) || cp === 0x670;
  R.joinType = function (cp) {
    if (cp === 0x640 || cp === 0x200D) return 'C';
    const l = R.letterByUni[cp];
    if (l) return l.join;
    if (isTransparent(cp)) return 'T';
    return 'U';
  };
  const extraByUni = {};
  (R.extras || []).forEach(e => { extraByUni[e.uni] = e.name; });

  /* text → glyph names in logical order (kashida: number of tatweels inserted inside joins) */
  R.shapeText = function (text, kashida) {
    const cps = Array.from(text || '').map(c => c.codePointAt(0)).filter(cp => !isTransparent(cp));
    const jt = cps.map(R.joinType);
    const forms = cps.map(() => 'isol');
    for (let i = 0; i < cps.length; i++) {
      if (jt[i] === 'U') continue;
      const jp = i > 0 && (jt[i - 1] === 'D' || jt[i - 1] === 'C');
      const jn = i < cps.length - 1 && (jt[i] === 'D' || jt[i] === 'C') && (jt[i + 1] === 'D' || jt[i + 1] === 'R' || jt[i + 1] === 'C');
      forms[i] = jp && jn ? 'medi' : jp ? 'fina' : jn ? 'init' : 'isol';
    }
    const out = [];
    for (let i = 0; i < cps.length; i++) {
      const cp = cps[i];
      if (cp === 0x644 && ALEFS[cps[i + 1]]) {
        out.push({ name: ALEFS[cps[i + 1]] + (forms[i] === 'medi' || forms[i] === 'fina' ? '.fina' : ''), joinsNext: false });
        i++; continue;
      }
      const l = R.letterByUni[cp];
      if (l) {
        const f = forms[i];
        out.push({ name: f === 'isol' ? l.name : l.name + '.' + f, joinsNext: f === 'init' || f === 'medi' });
      } else if (cp === 0x640) out.push({ name: 'tatweel', joinsNext: true });
      else if (extraByUni[cp]) out.push({ name: extraByUni[cp], joinsNext: false });
      else if (cp >= 0x30 && cp <= 0x39) out.push({ name: 'digit' + (cp - 0x30), joinsNext: false });
      else if (cp >= 0x6F0 && cp <= 0x6F9) out.push({ name: 'digit' + (cp - 0x6F0), joinsNext: false });
      else if (cp === 0x2C) out.push({ name: 'commaArabic', joinsNext: false });
      else if (cp === 0x3F) out.push({ name: 'questionArabic', joinsNext: false });
      else if (cp === 0x3B) out.push({ name: 'semicolonArabic', joinsNext: false });
      else out.push({ name: 'space', joinsNext: false });
      if (kashida > 0 && out[out.length - 1].joinsNext && i < cps.length - 1) for (let k = 0; k < kashida; k++) out.push({ name: 'tatweel', joinsNext: true });
    }
    return out;
  };

  /* SVG of a word using built outlines (right-to-left layout) */
  R.textToSVG = function (result, text, opts) {
    opts = opts || {};
    const seq = R.shapeText(text, opts.kashida || 0);
    const k = (opts.size || 200) / 1000;
    const advOf = (n) => result.built[n] ? result.built[n].adv : 0;
    const total = seq.reduce((a, g) => a + advOf(g.name), 0);
    const pad = opts.pad != null ? opts.pad : 0.12;
    const asc = result.font.ascender, desc = result.font.descender;
    const W = total * k, H = (asc - desc) * k;
    const px = W * pad + 4, py = H * pad + 4;
    let x = px + W, d = '';
    for (const g of seq) {
      const a = advOf(g.name) * k;
      x -= a;
      const c = result.outlines[g.name];
      if (c) d += R.font.svgPathData(c, x, py + asc * k, k);
    }
    const w = W + px * 2, h = H + py * 2;
    const fg = opts.color || '#191919', bg = opts.bg;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w.toFixed(1)} ${h.toFixed(1)}" width="${w.toFixed(0)}" height="${h.toFixed(0)}">` +
      (bg ? `<rect width="100%" height="100%" fill="${bg}"/>` : '') + `<path fill="${fg}" d="${d}"/></svg>`;
  };
})(typeof window !== 'undefined' ? window : globalThis);
