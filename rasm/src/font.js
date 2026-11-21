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
    // mark positioning: where each harakah sits on each letter (GDEF + GPOS mark, mkmk)
    const layout = F.markLayout(inv, builtMap, index, dna, m, glyphs.length);
    const res = { font, outlines, index, built: builtMap, family, layout };
    res.toArrayBuffer = () => F.withTables(font.toArrayBuffer(), { GDEF: F.gdef(layout), GPOS: F.gpos(layout) });
    return res;
  };

  /* ---------------------------------------------------------------- GDEF / GPOS (written here: opentype.js does not) */
  F.markLayout = function (inv, built, index, dna, m, nGlyphs) {
    const marks = [], bases = [], anchors = {}, markAnchor = {};
    const classes = new Array(nGlyphs).fill(1); classes[0] = 0;
    for (const spec of inv) {
      const gid = index[spec.name], g = built[spec.name];
      if (gid == null || !g) continue;
      if (spec.kind === 'mark') {
        classes[gid] = 3;
        const side = spec.mark.side === 'below' ? 1 : 0;
        marks.push({ gid, cls: side, x: g.markAnchor[0], y: Math.round(g.markAnchor[1]), name: spec.name, top: Math.round(g.markTop) });
        markAnchor[spec.name] = { cls: side, x: g.markAnchor[0], y: Math.round(g.markAnchor[1]) };
        continue;
      }
      if (spec.kind === 'letter' || spec.kind === 'lig') {
        const a = R.forms.anchorsFor(g, dna, m);
        if (a && (a.top || a.bottom)) { bases.push({ gid, top: a.top, bottom: a.bottom }); anchors[spec.name] = a; }
      }
    }
    // a harakah over a shadda sits on the shadda's top
    const gap = Math.round(R.forms.markGap(m) * 0.5);
    const shadda = marks.find((mk) => mk.name === 'shadda');
    const mkmk = shadda ? { mark1: marks.filter((mk) => mk.cls === 0 && mk.name !== 'shadda'), mark2: [{ gid: shadda.gid, top: [0, shadda.top + gap] }] } : null;
    return { marks, bases, mkmk, classes, anchors, markAnchor };
  };

  const B = {
    u16: (a, v) => { a.push((v >> 8) & 255, v & 255); return a; },
    s16: (a, v) => B.u16(a, (Math.round(v) + 65536) & 0xFFFF),
    tag: (a, t) => { for (let i = 0; i < 4; i++) a.push(t.charCodeAt(i)); return a; }
  };
  // a structure whose children are laid out after its header; offsets are written from its own start
  function withChildren(head, kids) {
    // head: byte array with 0-placeholders; kids: [{at: index in head, bytes}] (null bytes → offset 0)
    const out = head.slice();
    for (const k of kids) {
      if (!k.bytes) continue;
      const off = out.length;
      out[k.at] = (off >> 8) & 255; out[k.at + 1] = off & 255;
      for (const b of k.bytes) out.push(b);
    }
    return out;
  }
  const coverage = (gids) => { const a = []; B.u16(a, 1); B.u16(a, gids.length); gids.forEach((g) => B.u16(a, g)); return a; };
  const anchor = (x, y) => { const a = []; B.u16(a, 1); B.s16(a, x); B.s16(a, y); return a; };
  function markArray(marks) {
    const head = []; B.u16(head, marks.length);
    const kids = [];
    marks.forEach((mk) => { B.u16(head, mk.cls); kids.push({ at: head.length, bytes: anchor(mk.x, mk.y) }); B.u16(head, 0); });
    return withChildren(head, kids);
  }
  function anchorRows(rows, nCls) {
    // BaseArray / Mark2Array: per row nCls anchors (null → offset 0)
    const head = []; B.u16(head, rows.length);
    const kids = [];
    rows.forEach((row) => { for (let c = 0; c < nCls; c++) { kids.push({ at: head.length, bytes: row[c] ? anchor(row[c][0], row[c][1]) : null }); B.u16(head, 0); } });
    return withChildren(head, kids);
  }
  function markAttach(fmtMarks, fmtBases, nCls, baseRows) {
    // MarkBasePos / MarkMarkPos format 1 (same layout)
    const ms = fmtMarks.slice().sort((p, q) => p.gid - q.gid);
    const order = baseRows.map((r, i) => i).sort((p, q) => fmtBases[p] - fmtBases[q]);
    const head = []; B.u16(head, 1); B.u16(head, 0); B.u16(head, 0); B.u16(head, nCls); B.u16(head, 0); B.u16(head, 0);
    return withChildren(head, [
      { at: 2, bytes: coverage(ms.map((x) => x.gid)) },
      { at: 4, bytes: coverage(order.map((i) => fmtBases[i])) },
      { at: 8, bytes: markArray(ms) },
      { at: 10, bytes: anchorRows(order.map((i) => baseRows[i]), nCls) }
    ]);
  }
  function lookup(type, subtable) {
    const head = []; B.u16(head, type); B.u16(head, 0); B.u16(head, 1); B.u16(head, 0);
    return withChildren(head, [{ at: 6, bytes: subtable }]);
  }
  F.gpos = function (L) {
    if (!L.marks.length || !L.bases.length) return null;
    const lookups = [lookup(4, markAttach(L.marks, L.bases.map((b) => b.gid), 2, L.bases.map((b) => [b.top, b.bottom])))];
    const feats = [['mark', [0]]];
    if (L.mkmk && L.mkmk.mark1.length) {
      lookups.push(lookup(6, markAttach(L.mkmk.mark1, L.mkmk.mark2.map((x) => x.gid), 1, L.mkmk.mark2.map((x) => [x.top]))));
      feats.push(['mkmk', [1]]);
    }
    const lookupList = (() => { const h = []; B.u16(h, lookups.length); const kids = []; lookups.forEach((lk) => { kids.push({ at: h.length, bytes: lk }); B.u16(h, 0); }); return withChildren(h, kids); })();
    const featureList = (() => {
      const h = []; B.u16(h, feats.length); const kids = [];
      feats.forEach(([tag, idx]) => { B.tag(h, tag); const f = []; B.u16(f, 0); B.u16(f, idx.length); idx.forEach((i) => B.u16(f, i)); kids.push({ at: h.length, bytes: f }); B.u16(h, 0); });
      return withChildren(h, kids);
    })();
    const langSys = (() => { const a = []; B.u16(a, 0); B.u16(a, 0xFFFF); B.u16(a, feats.length); feats.forEach((f, i) => B.u16(a, i)); return a; })();
    const script = () => { const h = []; B.u16(h, 0); B.u16(h, 0); return withChildren(h, [{ at: 0, bytes: langSys }]); };
    const scriptList = (() => {
      const h = []; B.u16(h, 2); const kids = [];
      ['DFLT', 'arab'].forEach((t) => { B.tag(h, t); kids.push({ at: h.length, bytes: script() }); B.u16(h, 0); });
      return withChildren(h, kids);
    })();
    const head = []; B.u16(head, 1); B.u16(head, 0); B.u16(head, 0); B.u16(head, 0); B.u16(head, 0);
    return new Uint8Array(withChildren(head, [{ at: 4, bytes: scriptList }, { at: 6, bytes: featureList }, { at: 8, bytes: lookupList }]));
  };
  F.gdef = function (L) {
    const cd = []; B.u16(cd, 1); B.u16(cd, 1); B.u16(cd, L.classes.length - 1);
    for (let g = 1; g < L.classes.length; g++) B.u16(cd, L.classes[g]);
    const head = []; B.u16(head, 1); B.u16(head, 0); B.u16(head, 0); B.u16(head, 0); B.u16(head, 0); B.u16(head, 0);
    return new Uint8Array(withChildren(head, [{ at: 4, bytes: cd }]));
  };

  /* add or replace tables in an sfnt (recomputing offsets and checksums) */
  F.withTables = function (buf, add) {
    const src = new DataView(buf);
    const n = src.getUint16(4);
    const tables = {};
    for (let i = 0; i < n; i++) {
      const o = 12 + i * 16;
      const tag = String.fromCharCode(src.getUint8(o), src.getUint8(o + 1), src.getUint8(o + 2), src.getUint8(o + 3));
      const off = src.getUint32(o + 8), len = src.getUint32(o + 12);
      tables[tag] = new Uint8Array(buf, off, len).slice();
    }
    for (const tag in add) if (add[tag]) tables[tag] = add[tag];
    const tags = Object.keys(tables).sort();
    const num = tags.length;
    let es = 0; while ((1 << (es + 1)) <= num) es++;
    const sr = (1 << es) * 16;
    let size = 12 + num * 16;
    const offs = {};
    for (const t of tags) { offs[t] = size; size += (tables[t].length + 3) & ~3; }
    const out = new Uint8Array(size), dv = new DataView(out.buffer);
    dv.setUint32(0, src.getUint32(0)); dv.setUint16(4, num); dv.setUint16(6, sr); dv.setUint16(8, es); dv.setUint16(10, num * 16 - sr);
    const sum = (bytes) => { let s = 0; for (let i = 0; i < bytes.length; i += 4) s = (s + (((bytes[i] << 24) | ((bytes[i + 1] || 0) << 16) | ((bytes[i + 2] || 0) << 8) | (bytes[i + 3] || 0)) >>> 0)) >>> 0; return s; };
    if (tables.head) { tables.head = tables.head.slice(); tables.head[8] = tables.head[9] = tables.head[10] = tables.head[11] = 0; }
    tags.forEach((t, i) => {
      const o = 12 + i * 16;
      for (let k = 0; k < 4; k++) out[o + k] = t.charCodeAt(k);
      dv.setUint32(o + 4, sum(tables[t])); dv.setUint32(o + 8, offs[t]); dv.setUint32(o + 12, tables[t].length);
      out.set(tables[t], offs[t]);
    });
    if (tables.head) dv.setUint32(offs.head + 8, (0xB1B0AFBA - sum(out)) >>> 0);
    return out.buffer;
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
  R.shapeText = function (text, kashida, opts) {
    const withMarks = !!(opts && opts.marks);
    const all = Array.from(text || '').map(c => c.codePointAt(0));
    const cps = [], marksAfter = [];
    for (const cp of all) {
      if (isTransparent(cp)) {
        const mk = R.forms && R.forms.markByUni[cp];
        if (withMarks && mk && cps.length) marksAfter[cps.length - 1].push(mk.id);
        continue;
      }
      cps.push(cp); marksAfter.push([]);
    }
    const jt = cps.map(R.joinType);
    const forms = cps.map(() => 'isol');
    for (let i = 0; i < cps.length; i++) {
      if (jt[i] === 'U') continue;
      const jp = i > 0 && (jt[i - 1] === 'D' || jt[i - 1] === 'C');
      const jn = i < cps.length - 1 && (jt[i] === 'D' || jt[i] === 'C') && (jt[i + 1] === 'D' || jt[i + 1] === 'R' || jt[i + 1] === 'C');
      forms[i] = jp && jn ? 'medi' : jp ? 'fina' : jn ? 'init' : 'isol';
    }
    const out = [];
    const pushMarks = (...idx) => { for (const i of idx) for (const id of (marksAfter[i] || [])) out.push({ name: id, mark: true, joinsNext: false }); };
    for (let i = 0; i < cps.length; i++) {
      const cp = cps[i];
      if (cp === 0x644 && ALEFS[cps[i + 1]]) {
        out.push({ name: ALEFS[cps[i + 1]] + (forms[i] === 'medi' || forms[i] === 'fina' ? '.fina' : ''), joinsNext: false });
        pushMarks(i, i + 1);
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
      const base = out[out.length - 1];
      pushMarks(i);
      if (kashida > 0 && base.joinsNext && i < cps.length - 1) for (let k = 0; k < kashida; k++) out.push({ name: 'tatweel', joinsNext: true });
    }
    return out;
  };

  /* SVG of a word using built outlines (right-to-left layout) */
  R.textToSVG = function (result, text, opts) {
    opts = opts || {};
    const seq = R.shapeText(text, opts.kashida || 0, { marks: true });
    const k = (opts.size || 200) / 1000;
    const advOf = (n) => result.built[n] ? result.built[n].adv : 0;
    const total = seq.reduce((a, g) => a + (g.mark ? 0 : advOf(g.name)), 0);
    const pad = opts.pad != null ? opts.pad : 0.12;
    const asc = result.font.ascender, desc = result.font.descender;
    const W = total * k, H = (asc - desc) * k;
    const px = W * pad + 4, py = H * pad + 4;
    let x = px + W, d = '', baseX = x, baseName = null;
    const L = result.layout || {};
    for (const g of seq) {
      const c = result.outlines[g.name];
      if (g.mark) {
        const ma = L.markAnchor && L.markAnchor[g.name], ba = baseName && L.anchors && L.anchors[baseName];
        const at = ma && ba ? (ma.cls === 1 ? ba.bottom : ba.top) : null;
        if (c && at) d += R.font.svgPathData(c, baseX + (at[0] - ma.x) * k, py + asc * k - (at[1] - ma.y) * k, k);
        continue;
      }
      const a = advOf(g.name) * k;
      x -= a;
      baseX = x; baseName = g.name;
      if (c) d += R.font.svgPathData(c, x, py + asc * k, k);
    }
    const w = W + px * 2, h = H + py * 2;
    const fg = opts.color || '#191919', bg = opts.bg;
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w.toFixed(1)} ${h.toFixed(1)}" width="${w.toFixed(0)}" height="${h.toFixed(0)}">` +
      (bg ? `<rect width="100%" height="100%" fill="${bg}"/>` : '') + `<path fill="${fg}" d="${d}"/></svg>`;
  };
})(typeof window !== 'undefined' ? window : globalThis);
