/* رسم — Illustrator round-trip: the drawing template (SVG with one frame per piece and its guides)
 * and the reader for the SVG the designer exports back. Drawn pieces come back as exact Bézier
 * outlines in font units. */
(function (root) {
  'use strict';
  const R = root.Rasm;
  const G = R.geom;
  const IO = R.svg = {};

  /* ---------------------------------------------------------------- template geometry */
  IO.ORDER = ['stem', 'tooth', 'basin', 'head', 'arc', 'tail', 'cup', 'loop', 'nose', 'ain', 'kaf', 'heh', 'dot', 'hamza', 'madda'];
  const COLS = 5, FW = 1500, FH = 1900, GAP = 120, HEAD = 640, BASE = 1220, REF = 330;
  IO.SIZE = { W: COLS * FW + (COLS + 1) * GAP, H: HEAD + 3 * FH + 4 * GAP };
  IO.frame = function (i) {
    const row = Math.floor(i / COLS), col = COLS - 1 - (i % COLS);      // first frame at the top-right (Arabic order)
    const x0 = GAP + col * (FW + GAP), y0 = HEAD + GAP + row * (FH + GAP);
    return { id: IO.ORDER[i], x0, y0, x1: x0 + FW, y1: y0 + FH, base: y0 + BASE, ref: x0 + FW - REF };
  };
  IO.frames = () => IO.ORDER.map((_, i) => IO.frame(i));
  const SMALL = { dot: 1, hamza: 1, madda: 1 };

  /* what each frame asks for (shown on the template and in the report) */
  IO.GUIDE = {
    stem: ['ارسم الألف كاملة: من السطر إلى خط الألف.', 'عرضها هو سماكة القلم للخط كله.', 'اترك وسطها مستقيماً: هنا يُمطّ.'],
    tooth: ['نتوء يقف على السطر، وقدمه داخل الشريط.', 'ارتفاعه يحدد ارتفاع الأسنان كلها.'],
    basin: ['حوض الباء كاملاً بطرفيه، وقاعدته داخل الشريط.', 'اترك وسط القاعدة مستقيماً: هنا يُمدّ للكاف والفاء.'],
    head: ['عروة صغيرة مغلقة تجلس على السطر.', 'رأس الفاء والقاف والميم والواو.'],
    arc: ['الحاء المنفردة كاملة: ذراع علوية، وقوس ينزل تحت السطر.', 'جسمه يُستعمل أيضاً للعين.'],
    tail: ['الراء المنفردة: نزول يعبر الشريط وينعطف يساراً.'],
    cup: ['النون المنفردة بلا نقطة: ذراعها اليمنى تعبر الشريط إلى ارتفاع السن.', 'اترك قاع الكأس أفقياً في وسطه: هنا تُوسَّع للياء واللام.'],
    loop: ['عروة الصاد والطاء: مغلقة، ممدودة، تجلس على السطر.'],
    nose: ['رأس الحاء الأولى «حـ»: خط علوي، وجانب أيمن ينزل، وساق قصيرة على الشريط إلى اليسار.'],
    ain: ['رأس العين الأولى «عـ»: قوس صغير مفتوح لليمين، أسفله على الشريط.', 'منه تُصنع الهمزة إن لم ترسمها.'],
    kaf: ['الكاف الأولى «كـ»: الشَّرطة المائلة والقائم النازل إلى الشريط.', 'نصفها الأسفل يصير الدال.'],
    heh: ['الهاء المنفردة: قبة على السطر، وفي داخلها عين صغيرة.'],
    dot: ['نقطة واحدة. تُنسخ إلى نقطتين وثلاث.'],
    hamza: ['اختيارية: إن تركتها تُصنع من رأس العين.'],
    madda: ['اختيارية: موجة المدّ فوق الألف.']
  };

  /* ---------------------------------------------------------------- template */
  const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // Arabic lines: an explicit right-to-left embedding, anchored at their right end. This reads the same in
  // renderers that ignore the SVG direction attribute (the anchor then still means the right edge)
  const rtl = (t) => '\u202B' + esc(t) + '\u202C';
  const wrap = (t, n) => {
    const out = []; let cur = '';
    for (const w of String(t).split(' ')) { if (cur && (cur + ' ' + w).length > n) { out.push(cur); cur = w; } else cur = cur ? cur + ' ' + w : w; }
    if (cur) out.push(cur);
    return out;
  };
  const f1 = (v) => (Math.round(v * 10) / 10).toString();

  function contoursD(contours, X, Y) {
    // contours from R.font.outline (font coords) → path data
    let d = '';
    for (const c of contours) {
      d += 'M' + f1(X(c.start[0])) + ' ' + f1(Y(c.start[1]));
      for (const sg of c.segs) {
        d += sg.c1 ? 'C' + f1(X(sg.c1[0])) + ' ' + f1(Y(sg.c1[1])) + ' ' + f1(X(sg.c2[0])) + ' ' + f1(Y(sg.c2[1])) + ' ' + f1(X(sg.p[0])) + ' ' + f1(Y(sg.p[1]))
          : 'L' + f1(X(sg.p[0])) + ' ' + f1(Y(sg.p[1]));
      }
      d += 'Z';
    }
    return d;
  }
  function cmdsD(cmds, X, Y) {
    let d = '';
    for (const c of cmds) {
      if (c[0] === 'Z') d += 'Z';
      else if (c[0] === 'C') d += 'C' + [X(c[1]), Y(c[2]), X(c[3]), Y(c[4]), X(c[5]), Y(c[6])].map(f1).join(' ');
      else d += c[0] + f1(X(c[1])) + ' ' + f1(Y(c[2]));
    }
    return d;
  }

  /* the template SVG. m: current metrics (guides follow the pen); drawn: pieces already drawn (kept, editable) */
  IO.template = function (m, drawn) {
    drawn = drawn || {};
    const { W: TW, H: TH } = IO.SIZE;
    const out = [];
    out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${TW}" height="${TH}" viewBox="0 0 ${TW} ${TH}">`);
    out.push(`<title>Rasm — piece template</title>`);
    out.push(`<rect id="paper" x="0" y="0" width="${TW}" height="${TH}" fill="#f4f2ef"/>`);
    out.push(`<g id="guides">`);
    out.push(`<text x="${TW - GAP}" y="${GAP + 150}" font-family="Arial" font-size="130" fill="#191919" text-anchor="end">${rtl('رسم — قالب القطع')}</text>`);
    const lines = [
      '١ ارسم كل قطعة داخل إطارها بأداة القلم: شكلاً مغلقاً مملوءاً بالأسود.',
      '٢ اجعل أسفل القطعة على السطر، وما يتصل منها داخل الشريط البرتقالي.',
      '٣ الشكل الرمادي اقتراح من المحرك: ارسم فوقه أو خالفه. لا تغيّر حجم اللوحة.',
      '٤ صدّر: File › Export › Export As › SVG، ثم ارفع الملف في المحرك.'
    ];
    lines.forEach((t, i) => out.push(`<text x="${TW - GAP}" y="${GAP + 270 + i * 70}" font-family="Arial" font-size="48" fill="#5f5a55" text-anchor="end">${rtl(t)}</text>`));
    const W = m.W;
    IO.frames().forEach((fr) => {
      const id = fr.id, meta = R.slots.byId[id];
      const X = (u) => fr.ref - u, Y = (y) => fr.base - y;
      out.push(`<g id="frame-${id}">`);
      out.push(`<rect x="${fr.x0}" y="${fr.y0}" width="${FW}" height="${FH}" rx="36" fill="#ffffff" stroke="#d9d4ce" stroke-width="4"/>`);
      const num = meta.n ? meta.n + ' ' : '';
      out.push(`<text x="${fr.x1 - 50}" y="${fr.y0 + 110}" font-family="Arial" font-size="86" fill="#191919" text-anchor="end">${rtl(num + meta.ar)}</text>`);
      out.push(`<text x="${fr.x0 + 50}" y="${fr.y0 + 100}" font-family="Courier New, monospace" font-size="40" fill="#9a948d">${esc(id.toUpperCase())}${meta.role ? ' · ' + (meta.role === 'root' ? 'ROOT' : 'DERIVED') : ''}</text>`);
      (IO.GUIDE[id] || []).reduce((a, t) => a.concat(wrap(t, 56)), []).forEach((t, i) => out.push(`<text x="${fr.x1 - 50}" y="${fr.y0 + 185 + i * 54}" font-family="Arial" font-size="40" fill="#6f6a64" text-anchor="end">${rtl(t)}</text>`));
      if (!SMALL[id]) {
        // the joint band and the metric lines
        out.push(`<rect x="${fr.x0 + 30}" y="${f1(Y(W))}" width="${FW - 60}" height="${f1(W)}" fill="#ffd8c2"/>`);
        // guide colours stay light so the reader never mistakes them for ink
        const ml = [[0, 'BASE 0', '#8a847d', ''], [m.X, 'TOOTH ' + Math.round(m.X), '#a59f98', '14 10'], [m.A, 'ALIF ' + Math.round(m.A), '#a59f98', '14 10'],
          [m.L, 'HEAD ' + Math.round(m.L), '#c9c3bc', '6 10'], [-m.D, 'DESC −' + Math.round(m.D), '#a59f98', '14 10'], [-m.DD, 'DEEP −' + Math.round(m.DD), '#c9c3bc', '6 10']];
        for (const [y, lab, col, dash] of ml) {
          out.push(`<line x1="${fr.x0 + 30}" y1="${f1(Y(y))}" x2="${fr.x1 - 30}" y2="${f1(Y(y))}" stroke="${col}" stroke-width="3"${dash ? ` stroke-dasharray="${dash}"` : ''}/>`);
          // secondary lines (head, deep) label at the far end so they never sit on a main label
          const far = col === '#c9c3bc';
          out.push(`<text x="${far ? fr.x1 - 40 : fr.x0 + 40}" y="${f1(Y(y) - 12)}" font-family="Courier New, monospace" font-size="30" fill="${col}"${far ? ' text-anchor="end"' : ''}>${lab}</text>`);
        }
        // the piece's right join line
        out.push(`<line x1="${fr.ref}" y1="${f1(Y(m.A + 60))}" x2="${fr.ref}" y2="${f1(Y(-m.DD - 40))}" stroke="#8fa1ff" stroke-width="3" stroke-dasharray="4 12"/>`);
        out.push(`<text x="${fr.ref + 14}" y="${f1(Y(-m.DD - 30))}" font-family="Arial" font-size="30" fill="#6f84ff">حافة القطعة اليمنى</text>`);
      }
      // the engine's suggestion, as a light ghost
      try {
        const piece = IO.suggestion(id, m);
        if (piece) {
          const pseudo = { adv: piece.w, parts: piece.parts, clip: {} };
          const contours = R.font.outline(pseudo, m.dna, 0.6);
          const gx = SMALL[id] ? (fr.x0 + FW / 2 - piece.w / 2) : fr.ref - piece.w;
          const gy = SMALL[id] ? (fr.base - 200) : fr.base;
          const d = contoursD(contours, (x) => gx + x, (y) => gy - y);
          out.push(`<path d="${d}" fill="#e6e1db"/>`);
        }
      } catch (e) { /* no ghost */ }
      out.push(`</g>`);
    });
    out.push(`</g>`);
    // the drawing layer: pieces already drawn come back here, editable
    out.push(`<g id="draw">`);
    IO.frames().forEach((fr) => {
      const d = drawn[fr.id];
      if (!d) return;
      const gx = SMALL[fr.id] ? fr.x0 + FW / 2 - d.w / 2 : fr.ref + (d.placeU || 0);
      const gy = SMALL[fr.id] ? fr.base - 200 : fr.base;
      for (const sh of d.shapes) out.push(`<path d="${cmdsD(sh.cmds, (u) => (SMALL[fr.id] ? gx + d.w - u : gx - u), (y) => gy - y)}" fill="#000000"${sh.rule === 'evenodd' ? ' fill-rule="evenodd"' : ''}/>`);
    });
    out.push(`</g>`);
    out.push(`</svg>`);
    return out.join('\n');
  };

  // the engine's own version of a piece (pen + current metrics), used as the ghost
  IO.suggestion = function (id, m) {
    const saved = R.slots.drawn[id], savedImg = R.slots.overrides[id];
    delete R.slots.drawn[id]; delete R.slots.overrides[id];
    try {
      const p = {};
      if (id === 'stem') p.free = true;
      if (id === 'basin') p.full = true;
      if (id === 'cup') { p.top = m.t; p.topCap = m.cap; }
      if (id === 'ain') { p.botEnd = m.h; p.botCap = m.cap; }
      if (id === 'hamza') p.scale = 0.5;
      return R.slots.piece(id, m, p);
    } finally {
      if (saved) R.slots.drawn[id] = saved;
      if (savedImg) R.slots.overrides[id] = savedImg;
    }
  };

  IO.contoursD = contoursD;
  IO.cmdsD = cmdsD;
  IO.esc = esc;
  IO.rtl = rtl;
  IO.wrap = wrap;
  IO.f1 = f1;

  /* ---------------------------------------------------------------- SVG reading */
  const NAMED = { black: '#000000', white: '#ffffff', red: '#ff0000', none: 'none', transparent: 'none' };
  function parseColor(v) {
    if (v == null) return null;
    v = String(v).trim().toLowerCase();
    if (!v || v === 'none' || v === 'transparent') return 'none';
    if (NAMED[v]) return NAMED[v];
    if (v[0] === '#') {
      if (v.length === 4) return '#' + v[1] + v[1] + v[2] + v[2] + v[3] + v[3];
      return v.slice(0, 7);
    }
    const m = v.match(/rgba?\(\s*([\d.]+)%?[\s,]+([\d.]+)%?[\s,]+([\d.]+)%?/);
    if (m) return '#' + [m[1], m[2], m[3]].map(x => Math.max(0, Math.min(255, Math.round(+x)))).map(x => x.toString(16).padStart(2, '0')).join('');
    if (v.startsWith('url(')) return '#000000';
    return '#000000';
  }
  function lum(hex) {
    if (!hex || hex === 'none') return 1;
    const r = parseInt(hex.slice(1, 3), 16) / 255, g = parseInt(hex.slice(3, 5), 16) / 255, b = parseInt(hex.slice(5, 7), 16) / 255;
    return 0.299 * r + 0.587 * g + 0.114 * b;
  }

  function parseCSS(text, rules) {
    text = text.replace(/\/\*[\s\S]*?\*\//g, '');
    const re = /([^{}]+)\{([^}]*)\}/g;
    let mm;
    while ((mm = re.exec(text))) {
      const decl = {};
      mm[2].split(';').forEach((d) => { const i = d.indexOf(':'); if (i > 0) decl[d.slice(0, i).trim().toLowerCase()] = d.slice(i + 1).trim(); });
      mm[1].split(',').forEach((sel) => { sel = sel.trim(); if (sel) rules[sel] = Object.assign(rules[sel] || {}, decl); });
    }
  }

  // matrices [a b c d e f]: x' = a x + c y + e, y' = b x + d y + f
  const mul = (A, B) => [A[0] * B[0] + A[2] * B[1], A[1] * B[0] + A[3] * B[1], A[0] * B[2] + A[2] * B[3], A[1] * B[2] + A[3] * B[3], A[0] * B[4] + A[2] * B[5] + A[4], A[1] * B[4] + A[3] * B[5] + A[5]];
  const ID = [1, 0, 0, 1, 0, 0];
  function parseTransform(t) {
    let M = ID.slice();
    if (!t) return M;
    const re = /(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)/g;
    let mm;
    while ((mm = re.exec(t))) {
      const a = mm[2].split(/[\s,]+/).filter(Boolean).map(Number);
      let T = ID;
      if (mm[1] === 'matrix') T = a.slice(0, 6);
      else if (mm[1] === 'translate') T = [1, 0, 0, 1, a[0] || 0, a[1] || 0];
      else if (mm[1] === 'scale') T = [a[0], 0, 0, a.length > 1 ? a[1] : a[0], 0, 0];
      else if (mm[1] === 'rotate') {
        const r = (a[0] || 0) * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
        T = [c, s, -s, c, 0, 0];
        if (a.length >= 3) T = mul(mul([1, 0, 0, 1, a[1], a[2]], T), [1, 0, 0, 1, -a[1], -a[2]]);
      } else if (mm[1] === 'skewX') T = [1, 0, Math.tan(a[0] * Math.PI / 180), 1, 0, 0];
      else if (mm[1] === 'skewY') T = [1, Math.tan(a[0] * Math.PI / 180), 0, 1, 0, 0];
      M = mul(M, T);
    }
    return M;
  }

  // path data → absolute commands (M L C Z), arcs and quadratics turned into cubics
  function parseD(d) {
    const out = [];
    const reNum = /[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?/y;
    let i = 0;
    const n = d.length;
    const skip = () => { while (i < n && (d[i] === ' ' || d[i] === ',' || d[i] === '\n' || d[i] === '\t' || d[i] === '\r')) i++; };
    const num = () => { skip(); reNum.lastIndex = i; const m = reNum.exec(d); if (!m) return null; i = reNum.lastIndex; return parseFloat(m[0]); };
    const flag = () => { skip(); const c = d[i]; if (c === '0' || c === '1') { i++; return +c; } return null; };
    let cmd = null, x = 0, y = 0, sx = 0, sy = 0, pc = null, pq = null, lastCmd = '';
    const isCmd = (c) => /[MmLlHhVvCcSsQqTtAaZz]/.test(c);
    while (i < n) {
      skip(); if (i >= n) break;
      if (isCmd(d[i])) { cmd = d[i++]; }
      else if (!cmd) break;
      const rel = cmd === cmd.toLowerCase(), C = cmd.toUpperCase();
      if (C === 'Z') { out.push(['Z']); x = sx; y = sy; pc = pq = null; lastCmd = 'Z'; cmd = null; continue; }
      if (C === 'M') {
        const a = num(), b = num(); if (a == null || b == null) break;
        x = rel ? x + a : a; y = rel ? y + b : b; sx = x; sy = y;
        out.push(['M', x, y]); cmd = rel ? 'l' : 'L'; pc = pq = null; lastCmd = 'M'; continue;
      }
      if (C === 'L') { const a = num(), b = num(); if (a == null || b == null) break; x = rel ? x + a : a; y = rel ? y + b : b; out.push(['L', x, y]); pc = pq = null; lastCmd = 'L'; continue; }
      if (C === 'H') { const a = num(); if (a == null) break; x = rel ? x + a : a; out.push(['L', x, y]); pc = pq = null; lastCmd = 'L'; continue; }
      if (C === 'V') { const a = num(); if (a == null) break; y = rel ? y + a : a; out.push(['L', x, y]); pc = pq = null; lastCmd = 'L'; continue; }
      if (C === 'C' || C === 'S') {
        let x1, y1;
        if (C === 'C') { const a = num(), b = num(); if (a == null) break; x1 = rel ? x + a : a; y1 = rel ? y + b : b; }
        else { x1 = pc ? 2 * x - pc[0] : x; y1 = pc ? 2 * y - pc[1] : y; }
        const c = num(), e = num(), f = num(), g = num(); if (g == null) break;
        const x2 = rel ? x + c : c, y2 = rel ? y + e : e, x3 = rel ? x + f : f, y3 = rel ? y + g : g;
        out.push(['C', x1, y1, x2, y2, x3, y3]); pc = [x2, y2]; pq = null; x = x3; y = y3; lastCmd = 'C'; continue;
      }
      if (C === 'Q' || C === 'T') {
        let qx, qy;
        if (C === 'Q') { const a = num(), b = num(); if (a == null) break; qx = rel ? x + a : a; qy = rel ? y + b : b; }
        else { qx = pq ? 2 * x - pq[0] : x; qy = pq ? 2 * y - pq[1] : y; }
        const c = num(), e = num(); if (e == null) break;
        const x3 = rel ? x + c : c, y3 = rel ? y + e : e;
        out.push(['C', x + 2 / 3 * (qx - x), y + 2 / 3 * (qy - y), x3 + 2 / 3 * (qx - x3), y3 + 2 / 3 * (qy - y3), x3, y3]);
        pq = [qx, qy]; pc = null; x = x3; y = y3; lastCmd = 'Q'; continue;
      }
      if (C === 'A') {
        const rx = num(), ry = num(), rot = num(), la = flag(), sw = flag(), a = num(), b = num();
        if (b == null || la == null || sw == null) break;
        const x3 = rel ? x + a : a, y3 = rel ? y + b : b;
        arcToCubic(x, y, rx, ry, rot, la, sw, x3, y3, out);
        x = x3; y = y3; pc = pq = null; lastCmd = 'A'; continue;
      }
      break;
    }
    return out;
  }

  function arcToCubic(x1, y1, rx, ry, phiDeg, fa, fs, x2, y2, out) {
    if (!rx || !ry) { out.push(['L', x2, y2]); return; }
    rx = Math.abs(rx); ry = Math.abs(ry);
    const phi = phiDeg * Math.PI / 180, cp = Math.cos(phi), sp = Math.sin(phi);
    const dx = (x1 - x2) / 2, dy = (y1 - y2) / 2;
    const x1p = cp * dx + sp * dy, y1p = -sp * dx + cp * dy;
    let lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if (lam > 1) { const s = Math.sqrt(lam); rx *= s; ry *= s; }
    const num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p;
    const den = rx * rx * y1p * y1p + ry * ry * x1p * x1p;
    let co = Math.sqrt(Math.max(0, num / den));
    if (fa === fs) co = -co;
    const cxp = co * rx * y1p / ry, cyp = -co * ry * x1p / rx;
    const cx = cp * cxp - sp * cyp + (x1 + x2) / 2, cy = sp * cxp + cp * cyp + (y1 + y2) / 2;
    const ang = (ux, uy, vx, vy) => { const a = Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy); return a; };
    let t1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry);
    let dt = ang((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry);
    if (!fs && dt > 0) dt -= 2 * Math.PI; else if (fs && dt < 0) dt += 2 * Math.PI;
    const segs = Math.max(1, Math.ceil(Math.abs(dt) / (Math.PI / 2)));
    const d = dt / segs, k = 4 / 3 * Math.tan(d / 4);
    const pt = (t) => [cx + rx * Math.cos(t) * cp - ry * Math.sin(t) * sp, cy + rx * Math.cos(t) * sp + ry * Math.sin(t) * cp];
    const dv = (t) => [-rx * Math.sin(t) * cp - ry * Math.cos(t) * sp, -rx * Math.sin(t) * sp + ry * Math.cos(t) * cp];
    for (let s = 0; s < segs; s++) {
      const a = t1 + s * d, b = a + d;
      const p0 = pt(a), p3 = pt(b), d0 = dv(a), d3 = dv(b);
      out.push(['C', p0[0] + k * d0[0], p0[1] + k * d0[1], p3[0] - k * d3[0], p3[1] - k * d3[1], p3[0], p3[1]]);
    }
  }

  const K = 0.5522847498;
  function ellipseCmds(cx, cy, rx, ry) {
    return [['M', cx + rx, cy], ['C', cx + rx, cy + K * ry, cx + K * rx, cy + ry, cx, cy + ry], ['C', cx - K * rx, cy + ry, cx - rx, cy + K * ry, cx - rx, cy],
      ['C', cx - rx, cy - K * ry, cx - K * rx, cy - ry, cx, cy - ry], ['C', cx + K * rx, cy - ry, cx + rx, cy - K * ry, cx + rx, cy], ['Z']];
  }
  function rectCmds(x, y, w, h, rx, ry) {
    if (!rx && !ry) return [['M', x, y], ['L', x + w, y], ['L', x + w, y + h], ['L', x, y + h], ['Z']];
    rx = Math.min(rx || ry, w / 2); ry = Math.min(ry || rx, h / 2);
    const kx = K * rx, ky = K * ry;
    return [['M', x + rx, y], ['L', x + w - rx, y], ['C', x + w - rx + kx, y, x + w, y + ry - ky, x + w, y + ry], ['L', x + w, y + h - ry],
      ['C', x + w, y + h - ry + ky, x + w - rx + kx, y + h, x + w - rx, y + h], ['L', x + rx, y + h], ['C', x + rx - kx, y + h, x, y + h - ry + ky, x, y + h - ry],
      ['L', x, y + ry], ['C', x, y + ry - ky, x + rx - kx, y, x + rx, y], ['Z']];
  }

  /* parse an SVG file: every visible shape with its colours, in the file's own coordinates */
  IO.parse = function (text) {
    const doc = new DOMParser().parseFromString(text, 'image/svg+xml');
    const svg = doc.documentElement;
    if (!svg || svg.nodeName.toLowerCase() !== 'svg' || doc.getElementsByTagName('parsererror').length) throw new Error('الملف ليس SVG صالحاً');
    const vb = (svg.getAttribute('viewBox') || '').trim().split(/[\s,]+/).map(Number);
    let box;
    if (vb.length === 4 && vb.every(isFinite) && vb[2] > 0) box = vb;
    else box = [0, 0, parseFloat(svg.getAttribute('width')) || IO.SIZE.W, parseFloat(svg.getAttribute('height')) || IO.SIZE.H];
    const css = {};
    for (const st of Array.from(doc.getElementsByTagName('style'))) parseCSS(st.textContent || '', css);
    const shapes = [], issues = new Set();
    const SKIP = { defs: 1, clippath: 1, mask: 1, symbol: 1, style: 1, title: 1, desc: 1, metadata: 1, text: 1, tspan: 1, lineargradient: 1, radialgradient: 1, pattern: 1, filter: 1, marker: 1 };
    const walk = (el, M, inh) => {
      for (const node of Array.from(el.children)) {
        const tag = node.nodeName.toLowerCase().replace(/^svg:/, '');
        if (SKIP[tag]) continue;
        if (node.getAttribute('id') === 'guides') continue;
        if (tag === 'image') { issues.add('image'); continue; }
        if (tag === 'use') { issues.add('use'); continue; }
        const st = Object.assign({}, inh);
        const attrs = ['fill', 'stroke', 'stroke-width', 'fill-rule', 'opacity', 'fill-opacity', 'display', 'visibility'];
        for (const a of attrs) { const v = node.getAttribute(a); if (v != null) st[a] = v; }
        const cls = (node.getAttribute('class') || '').split(/\s+/).filter(Boolean);
        for (const c of cls) { const r = css['.' + c]; if (r) for (const a of attrs) if (r[a] != null) st[a] = r[a]; }
        const tagRule = css[tag]; if (tagRule) for (const a of attrs) if (tagRule[a] != null) st[a] = tagRule[a];
        const sa = node.getAttribute('style');
        if (sa) sa.split(';').forEach((d) => { const i = d.indexOf(':'); if (i > 0) { const k = d.slice(0, i).trim().toLowerCase(); if (attrs.includes(k)) st[k] = d.slice(i + 1).trim(); } });
        if (st.display === 'none' || st.visibility === 'hidden') continue;
        if (node.getAttribute('clip-path') || node.getAttribute('mask')) issues.add('clip');
        const M2 = mul(M, parseTransform(node.getAttribute('transform')));
        if (tag === 'g' || tag === 'svg' || tag === 'a' || tag === 'switch') { walk(node, M2, st); continue; }
        let cmds = null;
        const num = (k) => parseFloat(node.getAttribute(k)) || 0;
        if (tag === 'path') cmds = parseD(node.getAttribute('d') || '');
        else if (tag === 'rect') cmds = rectCmds(num('x'), num('y'), num('width'), num('height'), num('rx'), num('ry'));
        else if (tag === 'circle') cmds = ellipseCmds(num('cx'), num('cy'), num('r'), num('r'));
        else if (tag === 'ellipse') cmds = ellipseCmds(num('cx'), num('cy'), num('rx'), num('ry'));
        else if (tag === 'polygon' || tag === 'polyline') {
          const v = (node.getAttribute('points') || '').trim().split(/[\s,]+/).map(Number);
          cmds = [];
          for (let i = 0; i + 1 < v.length; i += 2) cmds.push([i ? 'L' : 'M', v[i], v[i + 1]]);
          if (tag === 'polygon') cmds.push(['Z']);
        } else if (tag === 'line') cmds = [['M', num('x1'), num('y1')], ['L', num('x2'), num('y2')]];
        if (!cmds || !cmds.length) continue;
        // apply the transform
        const T = (x, y) => [M2[0] * x + M2[2] * y + M2[4], M2[1] * x + M2[3] * y + M2[5]];
        const tc = cmds.map((c) => {
          if (c[0] === 'Z') return c;
          const r = [c[0]];
          for (let i = 1; i < c.length; i += 2) { const q = T(c[i], c[i + 1]); r.push(q[0], q[1]); }
          return r;
        });
        // open/closed sub-paths
        const subs = [];
        let cur = null;
        for (const c of tc) { if (c[0] === 'M') { cur = { cmds: [c], closed: false }; subs.push(cur); } else if (cur) { cur.cmds.push(c); if (c[0] === 'Z') cur.closed = true; } }
        const fill = parseColor(st.fill == null ? '#000000' : st.fill), stroke = parseColor(st.stroke == null ? 'none' : st.stroke);
        const op = (st.opacity != null ? +st.opacity : 1) * (st['fill-opacity'] != null ? +st['fill-opacity'] : 1);
        shapes.push({ subs, fill, stroke, opacity: isFinite(op) ? op : 1, rule: st['fill-rule'] === 'evenodd' ? 'evenodd' : 'nonzero', tag });
      }
    };
    walk(svg, ID, {});
    return { box, shapes, issues: Array.from(issues) };
  };

  /* ---------------------------------------------------------------- from an exported file to pieces */
  const bboxOf = (polys) => {
    let u0 = Infinity, u1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const P of polys) for (const q of P) { if (q[0] < u0) u0 = q[0]; if (q[0] > u1) u1 = q[0]; if (q[1] < y0) y0 = q[1]; if (q[1] > y1) y1 = q[1]; }
    return { u0, u1, y0, y1 };
  };
  const xcmds = (cmds, fn) => cmds.map((c) => { if (c[0] === 'Z') return c; const r = [c[0]]; for (let i = 1; i < c.length; i += 2) { const q = fn([c[i], c[i + 1]]); r.push(q[0], q[1]); } return r; });

  IO.toPieces = function (parsed) {
    const report = { matched: 0, ignored: 0, open: 0, outlineOnly: 0, outside: 0, issues: parsed.issues.slice(), scale: 1 };
    const [bx, by, bw, bh] = parsed.box;
    const kx = IO.SIZE.W / bw, ky = IO.SIZE.H / bh;
    if (Math.abs(kx - ky) / kx > 0.02) report.issues.push('size');
    report.scale = kx;
    const toT = (x, y) => [(x - bx) * kx, (y - by) * ky];
    const frames = IO.frames();
    const per = {};
    for (const sh of parsed.shapes) {
      const darkFill = sh.fill !== 'none' && lum(sh.fill) < 0.45 && sh.opacity > 0.4;
      const darkStroke = sh.stroke !== 'none' && lum(sh.stroke) < 0.45 && sh.opacity > 0.4;
      if (!darkFill && !darkStroke) { report.ignored++; continue; }
      // the sub-paths of one shape stay together (a counter is a sub-path of its loop)
      const byFrame = {};
      for (const sub of sh.subs) {
        if (!sub.closed) {
          if (darkStroke || darkFill) report.open++;
          if (!darkFill) continue;
        }
        if (!darkFill) report.outlineOnly++;
        const cmds = xcmds(sub.cmds, (q) => toT(q[0], q[1]));
        if (!sub.closed) cmds.push(['Z']);
        const polys = G.flattenCmds(cmds, 8);
        const bb = bboxOf(polys);
        const cx = (bb.u0 + bb.u1) / 2, cy = (bb.y0 + bb.y1) / 2;
        const fr = frames.find((f) => cx >= f.x0 && cx <= f.x1 && cy >= f.y0 && cy <= f.y1);
        if (!fr) { report.outside++; continue; }
        (byFrame[fr.id] = byFrame[fr.id] || { cmds: [], fr }).cmds.push(...cmds);
      }
      for (const id in byFrame) { (per[id] = per[id] || []).push({ cmds: byFrame[id].cmds, rule: sh.rule, fr: byFrame[id].fr }); report.matched++; }
    }
    const pieces = {};
    for (const id in per) {
      const fr = per[id][0].fr;
      // template coords → (u, y): u grows to the left of the frame's join line, y above the frame's baseline
      let shapes = per[id].map((s) => ({ rule: s.rule, cmds: xcmds(s.cmds, (q) => [fr.ref - q[0], fr.base - q[1]]) }));
      let polys = [];
      for (const s of shapes) polys = polys.concat(G.flattenCmds(s.cmds, 10));
      const bb = bboxOf(polys);
      // u = 0 at the piece's right edge; small pieces also sit on y = 0
      const du = -bb.u0, dy = SMALL[id] ? -bb.y0 : 0;
      shapes = shapes.map((s) => ({ rule: s.rule, cmds: xcmds(s.cmds, (q) => [q[0] + du, q[1] + dy]) }));
      polys = polys.map((P) => P.map((q) => [q[0] + du, q[1] + dy]));
      pieces[id] = IO.makePiece(id, shapes, polys, -bb.u0);
    }
    return { pieces, report };
  };

  IO.lum = lum;
  IO.xcmds = xcmds;
  IO.bboxOf = bboxOf;
  // the dark shapes of a parsed file, each with its closed sub-paths in template coordinates (toT maps file → template)
  IO.inkShapes = function (parsed, toT, report) {
    const out = [];
    for (const sh of parsed.shapes) {
      const darkFill = sh.fill !== 'none' && lum(sh.fill) < 0.45 && sh.opacity > 0.4;
      const darkStroke = sh.stroke !== 'none' && lum(sh.stroke) < 0.45 && sh.opacity > 0.4;
      if (!darkFill && !darkStroke) { report.ignored++; continue; }
      const subs = [];
      for (const sub of sh.subs) {
        if (!sub.closed) { report.open++; if (!darkFill) continue; }
        if (!darkFill) report.outlineOnly++;
        const cmds = xcmds(sub.cmds, toT);
        if (!sub.closed) cmds.push(['Z']);
        const bb = bboxOf(G.flattenCmds(cmds, 8));
        subs.push({ cmds, cx: (bb.u0 + bb.u1) / 2, cy: (bb.y0 + bb.y1) / 2 });
      }
      if (subs.length) out.push({ subs, rule: sh.rule });
    }
    return out;
  };
  IO.makePiece = function (id, shapes, polys, placeU) {
    polys = polys || shapes.reduce((a, s) => a.concat(G.flattenCmds(s.cmds, 10)), []);
    const bb = bboxOf(polys);
    return { id, shapes, polys, bbox: bb, w: bb.u1 - bb.u0, h: bb.y1 - bb.y0, placeU: placeU || 0 };
  };

  /* ---------------------------------------------------------------- measuring a drawn piece */
  const spans = (d, y) => G.spanAt(d.polys, y);
  const vspans = (d, u) => G.spanAt(d.polys, u, true);
  IO.measure = function (d, W) {
    const ms = {};
    const bb = d.bbox;
    const right = (sp) => sp.slice().sort((a, b) => a[0] - b[0])[0];
    const left = (sp) => sp.slice().sort((a, b) => b[1] - a[1])[0];
    const widest = (sp) => sp.slice().sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]))[0];
    switch (d.id) {
      case 'stem': {
        const sp = spans(d, bb.y1 * 0.5); const s = widest(sp) || [0, W];
        ms.cx = (s[0] + s[1]) / 2; ms.width = s[1] - s[0]; ms.top = bb.y1;
        // a round end narrows fast near the top (a half circle is 65% as wide at 12% of its width down); a flat one does not
        const near = spans(d, bb.y1 - ms.width * 0.12); const nt = widest(near);
        ms.capRound = nt ? (nt[1] - nt[0]) < ms.width * 0.82 : false;
        break;
      }
      case 'tooth': { const s = right(spans(d, Math.min(bb.y1 * 0.7, W * 1.6))) || [0, W]; ms.cx = (s[0] + s[1]) / 2; ms.width = s[1] - s[0]; ms.top = bb.y1; break; }
      case 'basin': { const s = right(spans(d, bb.y1 * 0.6)) || [0, W]; ms.tipEnd = s[1]; break; }
      case 'cup': {
        const s = right(spans(d, Math.max(W * 1.8, bb.y1 * 0.6))) || [0, W];
        ms.armW = s[1] - s[0]; ms.armTop = 0;
        for (const P of d.polys) for (const q of P) if (q[0] <= s[1] + W * 0.3 && q[1] > ms.armTop) ms.armTop = q[1];
        ms.depth = -bb.y0;
        break;
      }
      case 'nose': { const s = right(spans(d, W * 0.5)); ms.legEnd = s ? s[1] : d.w * 0.4; break; }
      case 'ain': { const s = left(spans(d, bb.y1 * 0.55)); ms.leftU = s ? (s[0] + s[1]) / 2 : d.w - W / 2; break; }
      case 'arc': {
        const s = left(spans(d, bb.y0 * 0.4)); ms.sideU = s ? (s[0] + s[1]) / 2 : d.w - W / 2; ms.depth = -bb.y0; break;
      }
      case 'kaf': {
        let vx = -Infinity, vy = 0;
        for (const P of d.polys) for (const q of P) if (q[0] > vx) { vx = q[0]; vy = q[1]; }
        ms.vertexY = vy;
        const s = right(spans(d, W * 1.6)) || [0, W]; ms.cx = (s[0] + s[1]) / 2;
        break;
      }
      case 'head': case 'loop': case 'heh': ms.height = bb.y1 - bb.y0; break;
    }
    return ms;
  };

  /* rendered stroke thickness (median of 2·distance on the medial axis) */
  IO.thickness = function (d) {
    const s = Math.min(0.5, 600 / Math.max(d.w, d.h, 1));
    const pad = 4;
    const Wc = Math.ceil(d.w * s) + pad * 2, Hc = Math.ceil(d.h * s) + pad * 2;
    const c = document.createElement('canvas'); c.width = Wc; c.height = Hc;
    const x = c.getContext('2d', { willReadFrequently: true });
    const view = { s, map: (u, y) => [pad + (d.bbox.u1 - u) * s, pad + (d.bbox.y1 - y) * s] };
    x.fillStyle = '#000';
    for (const sh of d.shapes) R.pen.fillShape(x, { cmds: sh.cmds, rule: sh.rule }, view);
    const data = x.getImageData(0, 0, Wc, Hc).data;
    const mask = new Uint8Array(Wc * Hc);
    for (let i = 0, j = 3; i < mask.length; i++, j += 4) mask[i] = data[j] > 127 ? 1 : 0;
    const dist = R.analyze._edt(mask, Wc, Hc), sk = R.analyze._thin(mask, Wc, Hc);
    const v = [];
    for (let i = 0; i < mask.length; i++) if (sk[i]) v.push(2 * dist[i] - 1);
    if (!v.length) return null;
    v.sort((a, b) => a - b);
    return v[v.length >> 1] / s;
  };

  /* ---------------------------------------------------------------- the guide: checks per piece */
  // level: 'ok' | 'warn' | 'error'. fix: an action id the app can apply
  IO.check = function (id, d, ctx) {
    const W = ctx.W, items = [];
    const add = (level, msg, fix) => items.push({ level, msg, fix });
    const bb = d.bbox;
    const AR = (v) => String(Math.round(v)).replace(/[0-9]/g, (x) => '٠١٢٣٤٥٦٧٨٩'[x]);
    if (SMALL[id]) {
      if (d.w < W * 0.3 || d.h < W * 0.3) add('warn', 'القطعة صغيرة جداً مقارنة بسماكة القلم');
      return items;
    }
    const sits = ['stem', 'tooth', 'basin', 'head', 'loop', 'nose', 'ain', 'kaf', 'heh'];
    const hangs = ['cup', 'arc', 'tail'];
    if (sits.includes(id)) {
      // a free-standing piece sits on the line; a joining one may stand anywhere inside the band
      const lo = -W * 0.12, hi = (id === 'stem' || id === 'head' || id === 'loop' || id === 'heh' || id === 'basin') ? W * 0.12 : W * 0.75;
      if (bb.y0 < lo || bb.y0 > hi) add('warn', 'أسفل القطعة ' + (bb.y0 > 0 ? 'أعلى' : 'أدنى') + ' من السطر بـ' + AR(Math.abs(bb.y0)) + ' وحدة', 'snap');
      if (!spans(d, W * 0.5).length) add('error', 'لا شيء من القطعة داخل شريط الوصلة، فلن تتصل بما حولها');
    }
    if (hangs.includes(id)) {
      if (bb.y0 > -W) add('error', 'يجب أن تنزل القطعة تحت السطر');
      if (!spans(d, W * 0.5).length) add('error', 'يجب أن تعبر القطعة شريط الوصلة');
      const want = id === 'arc' ? ctx.DD : ctx.D;
      if (Math.abs(-bb.y0 - want) > want * 0.3) add('warn', 'النزول هنا ' + AR(-bb.y0) + ' والدليل ' + AR(want) + '؛ سيُعتمد رسمك');
    }
    if (id === 'stem' && d.h < W * 3) add('error', 'الساق أقصر من أن تكون ألفاً');
    if (id === 'tooth' && bb.y1 < W * 1.6) add('warn', 'السن قصير جداً، قد يختفي في الوصلة');
    // stroke weight against the stem
    if (ctx.stemW && id !== 'stem') {
      const t = IO.thickness(d);
      if (t && Math.abs(t - ctx.stemW) > ctx.stemW * 0.16) add('warn', 'سماكة القطعة ' + AR(t) + ' وسماكة الساق ' + AR(ctx.stemW));
    }
    // zones that get stretched must stay straight
    // straight zone: the cross-section is the same all along it (within a small tolerance)
    const zone = (axis, from, to, at) => {
      const tol = Math.max(2, W * 0.04);
      let ref = null;
      for (let k = 0; k <= 4; k++) {
        const v = from + (to - from) * k / 4;
        const sp = axis === 'y' ? spans(d, v) : vspans(d, v);
        if (!ref) { ref = sp; continue; }
        if (sp.length !== ref.length) return false;
        for (let j = 0; j < sp.length; j++) {
          if (at && (Math.abs(sp[j][0] - ref[j][0]) > tol || Math.abs(sp[j][1] - ref[j][1]) > tol)) return false;
          if (!at && Math.abs((sp[j][1] - sp[j][0]) - (ref[j][1] - ref[j][0])) > tol) return false;
        }
      }
      return true;
    };
    if (id === 'stem' && !zone('y', bb.y1 * 0.35, bb.y1 * 0.65, true)) add('warn', 'وسط الساق ليس مستقيماً؛ مدّها قد يُظهر كسراً');
    if (id === 'basin' && !zone('u', d.w * 0.4, d.w * 0.6, true)) add('warn', 'وسط قاعدة الحوض ليس مستقيماً؛ مدّه قد يُظهر كسراً');
    if (id === 'cup' && !zone('u', d.w * 0.46, d.w * 0.54, false)) add('warn', 'قاع الكأس ليس أفقياً في وسطه؛ توسيعه للياء قد يُظهر كسراً');
    return items;
  };

  /* a drawn piece as an alpha mask (so the same fitting as for logo pieces applies) */
  IO.asMask = function (d, s) {
    s = s || 0.5;
    const pad = 2;
    const c = document.createElement('canvas');
    c.width = Math.max(4, Math.ceil(d.w * s) + pad * 2); c.height = Math.max(4, Math.ceil(d.h * s) + pad * 2);
    const x = c.getContext('2d', { willReadFrequently: true });
    const view = { s, map: (u, y) => [pad + (d.bbox.u1 - u) * s, pad + (d.bbox.y1 - y) * s] };
    x.fillStyle = '#000';
    for (const sh of d.shapes) R.pen.fillShape(x, { cmds: sh.cmds, rule: sh.rule }, view);
    return c;
  };
  /* every drawn piece's own measurements → piece parameters, so derived pieces inherit from drawn parents */
  IO.fitAll = function (pieces, m) {
    const pp = {}, rs = [];
    for (const id in pieces) {
      if (!R.slots.params[id]) continue;
      const d = pieces[id];
      const s = 0.5;
      const c = IO.asMask(d, s);
      const o = { img: c, wu: c.width / s, hu: c.height / s, y0: d.bbox.y0 - 2 / s };
      const f = R.slots.fitParams(id, o, m);
      if (id === 'basin' || id === 'cup' || id === 'head' || id === 'loop' || id === 'arc' || id === 'tail' || id === 'heh' || id === 'nose' || id === 'ain' || id === 'kaf') {
        if (f.w != null) f.w = Math.round(d.w); if (f.len != null) f.len = Math.round(d.w); if (f.reach != null) f.reach = Math.round(d.w);
        if (f.h != null) f.h = Math.round(d.h);
      }
      pp[id] = f;
      if (f.r != null && R.slots.byId[id].role === 'root') rs.push(f.r);
    }
    return { pp, roundness: rs.length ? +(rs.reduce((a, b) => a + b, 0) / rs.length).toFixed(2) : null };
  };

  /* the pen and the vertical metrics read off the drawings */
  IO.dnaFrom = function (drawn) {
    const out = {};
    const st = drawn.stem;
    if (st) {
      const ms = st.meas || IO.measure(st, 80);
      out.weight = Math.round(ms.width);
      out.alif = Math.round(ms.top);
      out.cap = ms.capRound ? 'round' : 'butt';
    }
    if (drawn.tooth) out.xh = Math.round(drawn.tooth.bbox.y1);
    if (drawn.head) out.loop = Math.round(drawn.head.h);
    if (drawn.cup) out.desc = Math.round(-drawn.cup.bbox.y0);
    if (drawn.arc) out.deep = Math.round(-drawn.arc.bbox.y0);
    if (drawn.dot && out.weight) out.dotSize = +(Math.max(drawn.dot.w, drawn.dot.h) / out.weight).toFixed(2);
    return out;
  };
})(typeof window !== 'undefined' ? window : globalThis);
