/* رسم — قالب الأشكال: every skeleton in each of its positions drawn whole, with lam-alef, the dot,
 * hamza and madda, and the harakat. What is drawn here is the glyph; the twelve pieces only fill
 * what is not drawn, and give the first draft that fills this template. */
(function (root) {
  'use strict';
  const R = root.Rasm, G = R.geom, IO = R.svg;
  const FM = R.forms = { drawn: {}, use: {}, kash: null };

  /* ---------------------------------------------------------------- template geometry */
  const SC = 0.5;                       // template px per font unit (keeps the board under Illustrator's limit)
  const FWU = 1900, FHU = 1700;         // a frame, in font units
  const FW = FWU * SC, FH = FHU * SC;
  const COLS = 8, GAP = 60, HEAD = 300, ROWS = 11;
  const BASEU = 1080, REFU = 300;       // baseline below the frame top, join line left of the frame's right side
  FM.SC = SC;
  FM.SIZE = { W: COLS * FW + (COLS + 1) * GAP, H: HEAD + ROWS * FH + (ROWS + 1) * GAP };

  const FORMS = ['isol', 'init', 'medi', 'fina'];
  const FORM_AR = { isol: 'منفصل', init: 'أوّل', medi: 'وسط', fina: 'آخر' };
  FM.FORM_AR = FORM_AR;
  const joinsR = (f) => f === 'fina' || f === 'medi';
  const joinsL = (f) => f === 'init' || f === 'medi';
  FM.joinsR = joinsR; FM.joinsL = joinsL;

  FM.MARKS = [
    { id: 'fatha', uni: 0x64E, ar: 'الفتحة', side: 'above' },
    { id: 'damma', uni: 0x64F, ar: 'الضمّة', side: 'above' },
    { id: 'kasra', uni: 0x650, ar: 'الكسرة', side: 'below' },
    { id: 'sukun', uni: 0x652, ar: 'السكون', side: 'above' },
    { id: 'shadda', uni: 0x651, ar: 'الشدّة', side: 'above' },
    { id: 'fathatan', uni: 0x64B, ar: 'تنوين الفتح', side: 'above' },
    { id: 'dammatan', uni: 0x64C, ar: 'تنوين الضم', side: 'above' },
    { id: 'kasratan', uni: 0x64D, ar: 'تنوين الكسر', side: 'below' }
  ];
  FM.markById = {}; FM.markByUni = {};
  FM.MARKS.forEach((mk) => { FM.markById[mk.id] = mk; FM.markByUni[mk.uni] = mk; });

  const joinOf = (rasm) => R.letters.some((l) => l.rasm === rasm && l.join === 'D') ? 'D' : 'R';
  const familyOf = (rasm) => R.letters.filter((l) => l.rasm === rasm && l.uni < 0x660).map((l) => String.fromCodePoint(l.uni)).join(' ');
  const shown = (ch, f) => f === 'isol' ? ch : f === 'init' ? ch + 'ـ' : f === 'medi' ? 'ـ' + ch + 'ـ' : 'ـ' + ch;

  FM.frames = function () {
    if (FM._frames) return FM._frames;
    const out = [];
    const place = (row, col, key, kind, extra) => {
      const x0 = GAP + (COLS - 1 - col) * (FW + GAP), y0 = HEAD + GAP + row * (FH + GAP);
      out.push(Object.assign({ key, kind, row, col, x0, y0, x1: x0 + FW, y1: y0 + FH, base: y0 + BASEU * SC, ref: x0 + FW - REFU * SC }, extra || {}));
    };
    R.rasm18.forEach(([rasm, ch], i) => {
      const row = Math.floor(i / 2), half = i % 2;
      const forms = R.formsOf(joinOf(rasm));
      FORMS.forEach((f, fi) => place(row, half * 4 + fi, rasm + '.' + f, forms.includes(f) ? 'form' : 'none', { rasm, form: f, ch }));
    });
    const extras = [['lamAlef.isol', 'lig', { form: 'isol', ch: 'لا' }], ['lamAlef.fina', 'lig', { form: 'fina', ch: 'لا' }],
      ['dot', 'small'], ['hamza', 'small'], ['madda', 'small']].concat(FM.MARKS.map((mk) => [mk.id, 'mark', { side: mk.side }]));
    extras.forEach(([key, kind, extra], i) => place(9 + Math.floor(i / COLS), i % COLS, key, kind, extra));
    return (FM._frames = out);
  };
  FM.frameByKey = (key) => FM.frames().find((f) => f.key === key);
  FM.formKeys = () => FM.frames().filter((f) => f.kind === 'form').map((f) => f.key);

  /* ---------------------------------------------------------------- the drawn store */
  FM.has = (key) => !!FM.drawn[key] && FM.use[key] !== false;
  FM.get = (key) => FM.has(key) ? FM.drawn[key] : null;
  FM.count = () => FM.formKeys().filter(FM.has).length;
  FM.kashFor = () => (FM.kash && (FM._force || Object.keys(FM.drawn).length)) ? FM.kash : null;

  /* ---------------------------------------------------------------- harakat drawn by the pen (until you draw yours) */
  FM.genMark = function (id, m) {
    const W = m.W, ds = m.ds;
    const wm = Math.max(24, Math.round(W * 0.42));
    const L = Math.max(ds * 1.55, wm * 3.6), sl = 0.32;
    const st = (pts, r, closed) => ({ type: 'stroke', pts: G.rpoly(pts, r || 0, !!closed), closed: !!closed, caps: ['round', 'round'], w: wm, slot: 'mark' });
    const fatha = (dy) => st([[wm / 2, L * sl + wm / 2 + dy], [L, wm / 2 + dy]]);
    const ring = (cu, cy, rr) => { const pts = []; for (let k = 0; k < 24; k++) { const a = k / 24 * Math.PI * 2; pts.push([cu + Math.cos(a) * rr, cy + Math.sin(a) * rr]); } return { type: 'stroke', pts, closed: true, caps: ['butt', 'butt'], w: wm * 0.85, slot: 'mark' }; };
    const damma = (du) => {
      const r = Math.max(ds * 0.3, wm * 0.9), h = r * 2 + wm + ds * 0.55;
      return [ring(du + r + wm / 2, h - r - wm / 2, r), st([[du + r * 1.9 + wm * 0.2, h - r * 1.2], [du + r * 2.2, h * 0.3], [du + r * 1.2, wm / 2]], [0, r * 0.8, 0])];
    };
    let parts = [];
    switch (id) {
      case 'fatha': case 'kasra': parts = [fatha(0)]; break;
      case 'fathatan': case 'kasratan': parts = [fatha(0), fatha(L * sl * 0.55 + wm * 1.2)]; break;
      case 'damma': parts = damma(0); break;
      case 'dammatan': parts = damma(0).concat(damma(ds * 0.95 + wm)); break;
      case 'sukun': { const r = Math.max(ds * 0.36, wm); parts = [ring(r + wm / 2, r + wm / 2, r)]; break; }
      case 'shadda': {
        const w = L * 1.05, h = Math.max(ds * 0.55, wm * 2);
        parts = [st([[wm / 2, h], [w * 0.2, wm / 2], [w * 0.5, h * 0.8], [w * 0.8, wm / 2], [w - wm / 2, h]], [0, wm * 0.6, wm * 0.6, wm * 0.6, 0])];
        break;
      }
    }
    return normPiece({ parts }, m);
  };
  function normPiece(p, m) {
    const b = R.partBounds(p.parts, m.dna);
    const parts = p.parts.map((pt) => R.shiftPart(pt, -b.u0, -b.y0));
    return { w: b.u1 - b.u0, h: b.y1 - b.y0, parts };
  }
  const shapesParts = (d, slot) => d.shapes.map((sh) => ({ type: 'shape', cmds: sh.cmds, rule: sh.rule, slot }));
  FM.markPiece = function (id, m) {
    const d = FM.get(id);
    if (d) return { w: d.w, h: d.h, parts: shapesParts(d, 'mark'), drawn: true };
    return FM.genMark(id, m);
  };

  /* a mark as a glyph: zero advance, centred on x = 0, resting at a default place (GPOS moves it) */
  FM.markGap = (m) => Math.round(m.W * 0.3 + m.ds * 0.12 + 14);
  FM.buildMark = function (m, mk) {
    const p = FM.markPiece(mk.id, m);
    const gap = FM.markGap(m);
    const yA = m.X + m.mg + gap, yB = -(gap + p.h);
    const y0 = mk.side === 'below' ? yB : yA;
    const parts = p.parts.map((pt) => R.shiftPart(pt, -p.w / 2, y0));
    return { adv: 0, parts, clip: {}, anc: {}, form: 'isol', mark: mk.id, markAnchor: mk.side === 'below' ? [0, y0 + p.h] : [0, y0], markTop: y0 + p.h };
  };

  /* where marks attach on a base glyph: above and below its ink, near where its dots go (font units, x from the left) */
  FM.anchorsFor = function (built, dna, m) {
    if (!built || !built.parts || !built.parts.length) return null;
    const s = 0.12;
    const r = R.font.rasterize(built, dna, s);
    if (!r) return null;
    const { W, H } = r;
    const data = r.ctx.getImageData(0, 0, W, H).data;
    const top = new Float32Array(W).fill(NaN), bot = new Float32Array(W).fill(NaN);
    for (let x = 0; x < W; x++) {
      for (let y = 0; y < H; y++) if (data[(y * W + x) * 4 + 3] > 90) { top[x] = y; break; }
      for (let y = H - 1; y >= 0; y--) if (data[(y * W + x) * 4 + 3] > 90) { bot[x] = y; break; }
    }
    const toFont = r.toFont;
    const xOf = (X) => toFont(X, 0)[0];
    const Xof = (fx) => { const a = toFont(0, 0)[0], b = toFont(1, 0)[0]; return (fx - a) / (b - a); };
    const yOf = (Y) => toFont(0, Y)[1];
    const gap = FM.markGap(m);
    const hw = Math.max(m.ds * 0.8, m.W * 1.2);
    const ink = R.partBounds(built.parts, dna);
    const cx = built.adv - (ink.u0 + ink.u1) / 2;
    const pick = (hint, fn) => {
      const fx = hint != null ? hint : cx;
      const X0 = Math.max(0, Math.floor(Xof(fx - hw))), X1 = Math.min(W - 1, Math.ceil(Xof(fx + hw)));
      let best = null;
      for (let X = X0; X <= X1; X++) { const v = fn === 'top' ? top[X] : bot[X]; if (isNaN(v)) continue; if (best == null || (fn === 'top' ? v < best : v > best)) best = v; }
      if (best == null) return null;
      const y = fn === 'top' ? Math.max(yOf(best) + gap, m.X + gap) : Math.min(yOf(best) - gap, -gap);
      return [Math.round(fx), Math.round(y)];
    };
    const ha = built.anc && built.anc.above ? built.adv - built.anc.above[0] : null;
    const hb = built.anc && built.anc.below ? built.adv - built.anc.below[0] : null;
    return { top: pick(ha, 'top'), bottom: pick(hb != null ? hb : ha, 'bottom') };
  };

  /* ---------------------------------------------------------------- reading a forms file */
  FM.isForms = (parsed) => { const b = parsed.box; return b[2] / b[3] < 1; };
  FM.read = function (parsed) {
    const report = { matched: 0, ignored: 0, open: 0, outlineOnly: 0, outside: 0, issues: parsed.issues.slice(), kind: 'forms' };
    const [bx, by, bw, bh] = parsed.box;
    const kx = FM.SIZE.W / bw, ky = FM.SIZE.H / bh;
    if (Math.abs(kx - ky) / kx > 0.02) report.issues.push('size');
    const toT = (q) => [(q[0] - bx) * kx, (q[1] - by) * ky];
    const frames = FM.frames().filter((f) => f.kind !== 'none');
    const per = {};
    for (const sh of IO.inkShapes(parsed, toT, report)) {
      const byFrame = {};
      for (const sub of sh.subs) {
        const fr = frames.find((f) => sub.cx >= f.x0 && sub.cx <= f.x1 && sub.cy >= f.y0 && sub.cy <= f.y1);
        if (!fr) { report.outside++; continue; }
        (byFrame[fr.key] = byFrame[fr.key] || { cmds: [], fr }).cmds.push(...sub.cmds);
      }
      for (const key in byFrame) { (per[key] = per[key] || []).push({ cmds: byFrame[key].cmds, rule: sh.rule, fr: byFrame[key].fr }); report.matched++; }
    }
    const forms = {};
    for (const key in per) {
      const fr = per[key][0].fr;
      let shapes = per[key].map((s) => ({ rule: s.rule, cmds: IO.xcmds(s.cmds, (q) => [(fr.ref - q[0]) / SC, (fr.base - q[1]) / SC]) }));
      if (fr.kind === 'small' || fr.kind === 'mark') {
        const bb = IO.bboxOf([].concat(...shapes.map((s) => G.flattenCmds(s.cmds, 10))));
        shapes = shapes.map((s) => ({ rule: s.rule, cmds: IO.xcmds(s.cmds, (q) => [q[0] - bb.u0, q[1] - bb.y0]) }));
      }
      const d = IO.makePiece(key, shapes, null, 0);
      Object.assign(d, { kind: fr.kind, rasm: fr.rasm, form: fr.form });
      forms[key] = d;
    }
    return { forms, report };
  };

  /* ---------------------------------------------------------------- measuring the joins */
  const vspans = (d, u) => G.spanAt(d.polys, u, true);
  function bandSpan(d, u, W) {
    // the vertical section at u nearest to the joint band
    const sp = vspans(d, u).filter((s) => s[0] < W * 1.6 && s[1] > -W * 0.6);
    if (!sp.length) return null;
    sp.sort((a, b) => Math.abs((a[0] + a[1]) / 2 - W / 2) - Math.abs((b[0] + b[1]) / 2 - W / 2));
    return [sp[0][0], sp[0][1]];
  }
  FM.measure = function (d, m) {
    const W = m.W, ms = {};
    const f = d.form;
    if (d.kind !== 'form' && d.kind !== 'lig') return ms;
    if (joinsR(f)) {
      ms.entryGap = d.bbox.u0;                                       // > 0: the join stops short of the line
      ms.entry = bandSpan(d, Math.max(d.bbox.u0, 0) + 2, W);
    }
    if (joinsL(f)) {
      let left = null;
      for (const k of [0.15, 0.3, 0.5, 0.7]) {
        const hs = G.spanAt(d.polys, W * k);
        if (hs.length) { const l = Math.max(...hs.map((s) => s[1])); if (left == null || l > left) left = l; }
      }
      ms.exitU = left;
      ms.exit = left != null ? bandSpan(d, left - 2, W) : null;
    }
    return ms;
  };
  // the kashida: the section most joins share
  FM.measureKash = function () {
    const y0 = [], y1 = [];
    for (const key in FM.drawn) {
      const ms = FM.drawn[key].meas || {};
      for (const s of [ms.entry, ms.exit]) if (s) { y0.push(s[0]); y1.push(s[1]); }
    }
    if (y0.length < 2) return null;
    const med = (a) => { a = a.slice().sort((p, q) => p - q); return a[a.length >> 1]; };
    const k0 = Math.round(Math.max(-40, med(y0))), k1 = Math.round(med(y1));
    return k1 - k0 > 8 ? [k0, k1] : null;
  };

  FM.check = function (key, d, m) {
    const items = [], W = m.W, ms = d.meas || {}, k = FM.kash;
    const add = (level, msg) => items.push({ level, msg });
    const AR = (v) => String(Math.round(v)).replace(/[0-9]/g, (x) => '٠١٢٣٤٥٦٧٨٩'[x]).replace('-', '−');
    if (d.kind === 'small' || d.kind === 'mark') {
      if (d.w < 8 || d.h < 8) add('warn', 'الشكل صغير جداً');
      return items;
    }
    const f = d.form;
    if (joinsR(f)) {
      if (!ms.entry) add('error', 'الوصلة اليمنى لا تصل إلى الشريط عند خط الحافة');
      else if (ms.entryGap > 4) add('warn', 'الوصلة اليمنى تقف قبل خط الحافة بـ' + AR(ms.entryGap) + ' وحدة؛ سيصلها المحرك بوصلة قصيرة');
    }
    if (joinsL(f) && !ms.exit) add('error', 'لا وصلة يسرى على الشريط');
    const off = (s) => k && s && (Math.abs(s[0] - k[0]) > Math.max(6, (k[1] - k[0]) * 0.18) || Math.abs(s[1] - k[1]) > Math.max(6, (k[1] - k[0]) * 0.18));
    if (off(ms.entry) || off(ms.exit)) {
      const s = off(ms.entry) ? ms.entry : ms.exit;
      add('warn', 'سماكة الوصلة هنا ' + AR(s[1] - s[0]) + ' (من ' + AR(s[0]) + ' إلى ' + AR(s[1]) + ') والكشيدة ' + AR(k[1] - k[0]) + '؛ ستظهر درجة عند الاتصال');
    }
    if (d.bbox.y1 > m.A * 1.35) add('warn', 'الشكل أعلى بكثير من خط الألف');
    return items;
  };

  /* the pen and the grid, read from the drawn forms */
  FM.dnaFrom = function () {
    const out = {}, get = (k) => FM.get(k);
    const al = get('alef.isol');
    if (al) {
      const top = al.bbox.y1;
      const hs = G.spanAt(al.polys, top * 0.5);
      if (hs.length) { const w = Math.max(...hs.map((s) => s[1] - s[0])); if (w > 10) out.weight = Math.round(w); }
      out.alif = Math.round(top);
    }
    const bi = get('beh.init');
    if (bi) out.xh = Math.round(bi.bbox.y1);
    const fi = get('feh.init');
    if (fi) out.loop = Math.round(fi.bbox.y1);
    const ni = get('noon.isol');
    if (ni && ni.bbox.y0 < 0) out.desc = Math.round(-ni.bbox.y0);
    const hi = get('hah.isol');
    if (hi && hi.bbox.y0 < 0) out.deep = Math.round(-hi.bbox.y0);
    return out;
  };

  /* ---------------------------------------------------------------- the template */
  FM.template = function (m, opts) {
    opts = opts || {};
    if (opts.kash && !FM.kash) {
      FM.kash = opts.kash; FM._force = true;
      try { return FM.template(m, Object.assign({}, opts, { kash: null })); } finally { FM.kash = null; FM._force = false; }
    }
    const dna = m.dna, esc = IO.esc, rtl = IO.rtl, f1 = IO.f1;
    const { W: TW, H: TH } = FM.SIZE;
    const out = [];
    out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${TW}" height="${TH}" viewBox="0 0 ${TW} ${TH}">`);
    out.push(`<title>Rasm — forms template</title>`);
    out.push(`<rect id="paper" x="0" y="0" width="${TW}" height="${TH}" fill="#f4f2ef"/>`);
    out.push(`<g id="guides">`);
    out.push(`<text x="${TW - GAP}" y="${GAP + 80}" font-family="Arial" font-size="72" fill="#191919" text-anchor="end">${rtl('رسم — قالب الأشكال: كل حرف في كل موضع')}</text>`);
    [
      '١ في كل إطار شكل مرسوم مسبقاً من قطعك. صحّحه بأداة القلم أو ارسمه من جديد: شكلاً مغلقاً مملوءاً بالأسود.',
      '٢ ما يتصل من اليمين يلمس خط الحافة الأزرق. وما يتصل من اليسار يمتد على الشريط وينتهي بقطع عمودي.',
      '٣ سماكة الوصلة عند الحافة هي الكشيدة في الخط كله: اجعلها واحدة في كل الأشكال. لا تغيّر حجم اللوحة.',
      '٤ النقاط تُوضع آلياً فلا ترسمها هنا. صدّر: File › Export › Export As › SVG، ثم ارفع الملف في المحرك.'
    ].forEach((t, i) => out.push(`<text x="${TW - GAP}" y="${GAP + 140 + i * 34}" font-family="Arial" font-size="26" fill="#5f5a55" text-anchor="end">${rtl(t)}</text>`));
    const k = FM.kash || [0, m.W];
    const draw = [];
    const glyphPath = (built, fr) => {
      const contours = R.font.outline(built, dna, 0.6);
      return IO.contoursD(contours, (x) => fr.ref - (built.adv - x) * SC, (y) => fr.base - y * SC);
    };
    const drawnPath = (d, fr, small) => d.shapes.map((sh) => `<path d="${IO.cmdsD(sh.cmds, small ? small.X : (u) => fr.ref - u * SC, small ? small.Y : (y) => fr.base - y * SC)}" fill="#000000"${sh.rule === 'evenodd' ? ' fill-rule="evenodd"' : ''}/>`).join('');
    // a light beh for the harakat frames: their size is judged against a letter
    let behBuilt = null;
    try { behBuilt = R.buildLetter(m, 'beh', 'isol', null); } catch (e) { /* none */ }
    for (const fr of FM.frames()) {
      const X = (u) => fr.ref - u * SC, Y = (y) => fr.base - y * SC;
      const id = 'f-' + fr.key.replace('.', '-');
      if (fr.kind === 'none') {
        out.push(`<g id="${id}"><rect x="${fr.x0}" y="${fr.y0}" width="${FW}" height="${FH}" rx="18" fill="none" stroke="#e2ddd7" stroke-width="2" stroke-dasharray="8 10"/>`);
        out.push(`<text x="${fr.x0 + FW / 2}" y="${fr.y0 + FH / 2}" font-family="Arial" font-size="28" fill="#b9b3ac" text-anchor="middle">${rtl('لا يتصل بما بعده')}</text></g>`);
        continue;
      }
      out.push(`<g id="${id}">`);
      out.push(`<rect x="${fr.x0}" y="${fr.y0}" width="${FW}" height="${FH}" rx="18" fill="#ffffff" stroke="#d9d4ce" stroke-width="2"/>`);
      let title = '', sub = '';
      if (fr.kind === 'form') { title = shown(fr.ch, fr.form) + '  ' + FORM_AR[fr.form]; sub = familyOf(fr.rasm); }
      else if (fr.kind === 'lig') { title = (fr.form === 'fina' ? 'ـلا' : 'لا') + '  ' + FORM_AR[fr.form]; sub = 'لا لأ لإ لآ'; }
      else if (fr.kind === 'small') { title = R.slots.byId[fr.key].ar; sub = fr.key === 'dot' ? 'تُنسخ إلى نقطتين وثلاث' : 'اختيارية'; }
      else { title = FM.markById[fr.key].ar; sub = fr.side === 'below' ? 'تحت الحرف' : 'فوق الحرف'; }
      out.push(`<text x="${fr.x1 - 24}" y="${fr.y0 + 58}" font-family="Arial" font-size="46" fill="#191919" text-anchor="end">${rtl(title)}</text>`);
      out.push(`<text x="${fr.x1 - 24}" y="${fr.y0 + 98}" font-family="Arial" font-size="24" fill="#8a847d" text-anchor="end">${rtl(sub)}</text>`);
      out.push(`<text x="${fr.x0 + 20}" y="${fr.y0 + 44}" font-family="Courier New, monospace" font-size="20" fill="#a59f98">${esc(fr.key.toUpperCase())}</text>`);
      if (fr.kind === 'form' || fr.kind === 'lig') {
        out.push(`<rect x="${fr.x0 + 14}" y="${f1(Y(k[1]))}" width="${FW - 28}" height="${f1((k[1] - k[0]) * SC)}" fill="#ffd8c2"/>`);
        const ml = [[0, '#8a847d', ''], [m.X, '#c9c3bc', '8 8'], [m.A, '#c9c3bc', '8 8'], [-m.D, '#c9c3bc', '8 8']];
        for (const [y, col, dash] of ml) out.push(`<line x1="${fr.x0 + 14}" y1="${f1(Y(y))}" x2="${fr.x1 - 14}" y2="${f1(Y(y))}" stroke="${col}" stroke-width="2"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`);
        out.push(`<line x1="${fr.ref}" y1="${f1(Y(m.A + 60))}" x2="${fr.ref}" y2="${f1(Y(-m.DD - 40))}" stroke="#8fa1ff" stroke-width="2.5" stroke-dasharray="4 8"/>`);
        if (joinsR(fr.form)) out.push(`<text x="${fr.ref + 8}" y="${f1(Y(-m.DD - 30))}" font-family="Arial" font-size="20" fill="#6f84ff">${rtl('تتصل هنا')}</text>`);
      }
      // the draft (or what was drawn before) on the drawing layer
      try {
        const prev = FM.drawn[fr.key];
        if (fr.kind === 'form' || fr.kind === 'lig') {
          if (prev) draw.push(drawnPath(prev, fr));
          else {
            const built = fr.kind === 'form' ? R.buildLetter(m, fr.rasm, fr.form, null) : R.buildLigature(m, R.ligatures[0], fr.form);
            draw.push(`<path d="${glyphPath(built, fr)}" fill="#000000"/>`);
            if (joinsL(fr.form)) {
              const ex = X(built.adv);
              out.push(`<line x1="${f1(ex)}" y1="${f1(Y(m.W * 2.6))}" x2="${f1(ex)}" y2="${f1(Y(-m.W * 1.6))}" stroke="#8fa1ff" stroke-width="2" stroke-dasharray="3 7"/>`);
              out.push(`<text x="${f1(ex - 6)}" y="${f1(Y(-m.W * 1.6) + 22)}" font-family="Arial" font-size="20" fill="#6f84ff" text-anchor="end">${rtl('مخرج الوصلة')}</text>`);
            }
          }
        } else if (fr.kind === 'small') {
          // the dot, hamza and madda as the font has them now (yours if you drew them in the pieces)
          const piece = prev ? { w: prev.w, h: prev.h, parts: shapesParts(prev, fr.key) } : R.slots.piece(fr.key, m, fr.key === 'hamza' ? { scale: 0.5 } : {});
          const gx = fr.x0 + FW / 2 + piece.w * SC / 2, gy = fr.base - 120 * SC;
          if (prev) draw.push(drawnPath(prev, fr, { X: (u) => gx - u * SC, Y: (y) => gy - y * SC }));
          else {
            const c = R.font.outline({ adv: piece.w, parts: piece.parts, clip: {} }, dna, 0.6);
            draw.push(`<path d="${IO.contoursD(c, (x) => gx - piece.w * SC + x * SC, (y) => gy - y * SC)}" fill="#000000"/>`);
          }
        } else if (fr.kind === 'mark') {
          // a light letter for scale, the mark above or below it
          let topY = m.X, botY = 0, bx0 = fr.x0 + FW / 2, bw = 0;
          if (behBuilt) {
            const c = R.font.outline(behBuilt, dna, 0.6);
            bw = behBuilt.adv * SC; bx0 = fr.x0 + FW / 2 - bw / 2;
            out.push(`<path d="${IO.contoursD(c, (x) => bx0 + x * SC, (y) => fr.base - y * SC)}" fill="#e6e1db"/>`);
            const ink = R.partBounds(behBuilt.parts, dna);
            topY = ink.y1; botY = ink.y0;
          }
          const piece = prev ? { w: prev.w, h: prev.h, parts: shapesParts(prev, 'mark') } : FM.genMark(fr.key, m);
          const gap = FM.markGap(m);
          const my = fr.side === 'below' ? botY - gap - piece.h : topY + gap;
          const gx = fr.x0 + FW / 2 + piece.w * SC / 2, gy = fr.base - my * SC;
          if (prev) draw.push(drawnPath(prev, fr, { X: (u) => gx - u * SC, Y: (y) => gy - y * SC }));
          else {
            const c = R.font.outline({ adv: piece.w, parts: piece.parts, clip: {} }, dna, 0.6);
            draw.push(`<path d="${IO.contoursD(c, (x) => gx - piece.w * SC + x * SC, (y) => gy - y * SC)}" fill="#000000"/>`);
          }
        }
      } catch (e) { console.error(fr.key, e); }
      out.push(`</g>`);
    }
    out.push(`</g>`);
    out.push(`<g id="draw">${draw.join('\n')}</g>`);
    out.push(`</svg>`);
    return out.join('\n');
  };

  /* ---------------------------------------------------------------- import / clear / project */
  FM.adopt = function (forms, m) {
    for (const k in FM.drawn) delete FM.drawn[k];
    for (const k in FM.use) delete FM.use[k];
    Object.assign(FM.drawn, forms);
    FM.remeasure(m);
  };
  FM.remeasure = function (m) {
    FM.kash = null;
    for (const key in FM.drawn) FM.drawn[key].meas = FM.measure(FM.drawn[key], m);
    FM.kash = FM.measureKash();
  };
  FM.clear = function () { for (const k in FM.drawn) delete FM.drawn[k]; for (const k in FM.use) delete FM.use[k]; FM.kash = null; };
  FM.save = function () {
    const out = {};
    for (const key in FM.drawn) out[key] = { shapes: FM.drawn[key].shapes, kind: FM.drawn[key].kind, rasm: FM.drawn[key].rasm, form: FM.drawn[key].form };
    return { drawn: out, use: Object.assign({}, FM.use) };
  };
  FM.load = function (data, m) {
    FM.clear();
    if (!data) return;
    for (const key in (data.drawn || {})) {
      const o = data.drawn[key];
      const d = IO.makePiece(key, o.shapes, null, 0);
      Object.assign(d, { kind: o.kind, rasm: o.rasm, form: o.form });
      FM.drawn[key] = d;
    }
    Object.assign(FM.use, data.use || {});
    FM.remeasure(m);
  };
})(typeof window !== 'undefined' ? window : globalThis);
