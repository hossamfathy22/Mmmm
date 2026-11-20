/* رسم — the 12 key shapes (الأشكال المفتاحية) + 4 small pieces.
 * Every slot has: metadata, a vector generator driven by the style DNA, and an optional
 * image override (taken from the logo or uploaded by the user). Pieces are returned in local
 * (u, y) coordinates: u = 0 is the piece's right edge, u grows to the left. */
(function (root) {
  'use strict';
  const R = root.Rasm;
  const G = R.geom;

  const S = R.slots = {};

  S.list = [
    { id: 'stem',  n: '01', ar: 'الساق',      en: 'Stem',       letters: 'ا ل ك ط',     rule: 'خط عمودي بطول الألف — يُقاس من الشعار مباشرة', fit: { bottom: 0, top: 'A' } },
    { id: 'tooth', n: '02', ar: 'السنّ',      en: 'Tooth',      letters: 'ب ت ث ن ي س ش', rule: 'نتوء قصير — الساق مقصوصة إلى ارتفاع السن', fit: { bottom: 0, top: 'X' } },
    { id: 'basin', n: '03', ar: 'الحوض',      en: 'Basin',      letters: 'ب ت ث ف ك',    rule: 'منحنى مفتوح على السطر — طرفاه بارتفاع السن', fit: { bottom: 0, top: 'X' } },
    { id: 'cup',   n: '04', ar: 'الكأس',      en: 'Cup',        letters: 'ن ق ل س ص ي',  rule: 'حوض ينزل تحت السطر بعمق النزول', fit: { bottom: '-D', top: 'X' } },
    { id: 'head',  n: '05', ar: 'الرأس',      en: 'Head',       letters: 'ف ق م و',      rule: 'عروة صغيرة — من البياض الداخلي وسماكة القلم', fit: { bottom: 0, top: 'L' } },
    { id: 'loop',  n: '06', ar: 'العروة',     en: 'Loop',       letters: 'ص ض ط ظ',      rule: 'الرأس ممدوداً أفقياً', fit: { bottom: 0, top: 'Oh' } },
    { id: 'arc',   n: '07', ar: 'القوس',      en: 'Arc',        letters: 'ج ح خ ع غ',    rule: 'قوس مفتوح لليمين ينزل تحت السطر', fit: { bottom: '-DD', top: 'X' } },
    { id: 'nose',  n: '08', ar: 'الأنف',      en: 'Nose',       letters: 'ج ح خ',        rule: 'رأس الحاء — خط علوي يعود إلى السطر', fit: { bottom: 0, top: 'X' } },
    { id: 'ain',   n: '09', ar: 'رأس العين',  en: 'Ain head',   letters: 'ع غ ء',        rule: 'قوس صغير مفتوح لليمين — ومنه الهمزة', fit: { bottom: 0, top: 'X' } },
    { id: 'tail',  n: '10', ar: 'الذيل',      en: 'Tail',       letters: 'ر ز و م',      rule: 'نزول تحت السطر ينعطف يساراً', fit: { bottom: '-D', top: 'X' } },
    { id: 'kaf',   n: '11', ar: 'خط الكاف',   en: 'Kaf stroke', letters: 'ك د',          rule: 'شَرطة مائلة — ومنها الدال', fit: { bottom: 0, top: 'A' } },
    { id: 'heh',   n: '12', ar: 'الهاء',      en: 'Heh',        letters: 'ه ة',          rule: 'قبة على السطر وفي داخلها عين صغيرة', fit: { bottom: 0, top: 'Eh' } }
  ];
  S.small = [
    { id: 'dot',   ar: 'النقطة',  en: 'Dot',     rule: 'تُنسخ: نقطة، نقطتان، ثلاث', fit: { bottom: 0, top: 'ds' } },
    { id: 'hamza', ar: 'الهمزة',  en: 'Hamza',   rule: 'رأس العين مصغّراً', fit: { bottom: 0, top: 'hz' } },
    { id: 'madda', ar: 'المدّة',  en: 'Madda',   rule: 'موجة بقلم رفيع', fit: { bottom: 0, top: 'md' } },
    { id: 'conn',  ar: 'الوصلة',  en: 'Joint',   rule: 'خط السطر بسماكة القلم — الكشيدة', fit: null }
  ];
  S.byId = {};
  S.list.concat(S.small).forEach(s => { S.byId[s.id] = s; });

  const stroke = (pts, caps, slot, extra) => Object.assign({ type: 'stroke', pts, caps, slot }, extra || {});
  const closedStroke = (pts, slot, extra) => Object.assign({ type: 'stroke', pts, closed: true, slot }, extra || {});

  /* ------------------------------------------------------------ vector generators */
  const gen = S.gen = {};

  gen.stem = (m, p) => {
    const top = (p.top != null ? p.top : m.A) - m.capExt;
    const bot = p.free ? m.capExt : (p.bottom != null ? p.bottom : m.b);
    return { w: m.W, cx: m.h, parts: [stroke([[m.h, bot], [m.h, top]], [p.free ? m.cap : 'butt', m.cap], 'stem')] };
  };

  gen.tooth = (m, p) => {
    const top = (p.top != null ? p.top : m.X) - m.capExt;
    const st = m.dna.toothStyle;
    if (st === 'peak') {
      const s = Math.max(m.W * 0.8, (top - m.b) * 0.42), cx = s + m.h;
      return { w: 2 * s + m.W, cx, parts: [stroke(G.rpoly([[cx - s, m.b], [cx, top], [cx + s, m.b]], [0, m.r(m.W * 0.5), 0]), ['butt', 'butt'], 'tooth')] };
    }
    if (st === 'bump') {
      const s = Math.max(m.W * 0.75, (top - m.b) * 0.5), cx = s + m.h;
      return { w: 2 * s + m.W, cx, parts: [stroke(G.rpoly([[cx - s, m.b], [cx - s, top], [cx + s, top], [cx + s, m.b]], [0, s, s, 0]), ['butt', 'butt'], 'tooth')] };
    }
    return { w: m.W, cx: m.h, parts: [stroke([[m.h, m.b], [m.h, top]], ['butt', m.cap], 'tooth')] };
  };

  gen.basin = (m, p) => {
    const len = p.len != null ? p.len : m.Bl;
    const tipL = p.tipL != null ? p.tipL : m.b + (m.t - m.b) * 0.72;
    const rB = m.r(Math.min(len * 0.32, tipL - m.b));
    if (p.full) {
      const tipR = p.tipR != null ? p.tipR : m.t;
      const rR = m.r(Math.min(len * 0.3, tipR - m.b) * 0.9);
      const pts = G.rpoly([[m.h, tipR], [m.h, m.b], [len - m.h, m.b], [len - m.h, tipL]], [0, rR, rB, 0]);
      return { w: len, cx: len / 2, parts: [stroke(pts, [m.cap, m.cap], 'basin')] };
    }
    const pts = G.rpoly([[0, m.b], [len - m.h, m.b], [len - m.h, tipL]], [0, rB, 0]);
    return { w: len, cx: len / 2, parts: [stroke(pts, ['butt', m.cap], 'basin')] };
  };

  gen.cup = (m, p) => {
    const w = p.w != null ? p.w : m.Cw;
    const top = p.top != null ? p.top : m.b;
    const tip = p.tip != null ? p.tip : m.b + (m.t - m.b) * 0.5;
    const d = p.d != null ? p.d : m.d;
    const rr = m.r((w - m.W) / 2);
    const pts = G.rpoly([[m.h, top], [m.h, d], [w - m.h, d], [w - m.h, tip]], [0, rr, rr, 0]);
    return { w, cx: w / 2, parts: [stroke(pts, [p.topCap || 'butt', m.cap], 'cup')] };
  };

  gen.yeh = (m, p) => {
    // ى bowl: a short arm from the top-right, along the baseline, then a wide cup below it
    const w = p.w != null ? p.w : m.Yw;
    const top = p.top != null ? p.top : m.t;
    const k = Math.max(w * 0.34, m.W * 2.4);
    const tip = m.b + (m.t - m.b) * 0.42;
    const rr = m.r(Math.min((w - k) / 2, (m.b - m.d)) * 0.95);
    const rs = m.r(m.W * 1.4);
    const pts = G.rpoly([[m.h, top], [m.h, m.b], [k, m.b], [k, m.d], [w - m.h, m.d], [w - m.h, tip]], [0, rs, rs, rr, rr, 0]);
    return { w, cx: k + (w - k) / 2, parts: [stroke(pts, [p.topCap || m.cap, m.cap], 'cup')] };
  };

  gen.head = (m, p) => {
    const w = p.w != null ? p.w : m.Lw, H = p.h != null ? p.h : m.L, y0 = p.y0 || 0;
    const rr = m.r((Math.min(w, H) - m.W) / 2);
    return { w, cx: w / 2, parts: [closedStroke(G.rrect(m.h, y0 + m.h, w - m.h, y0 + H - m.h, rr), 'head')] };
  };

  gen.loop = (m, p) => {
    const w = p.w != null ? p.w : m.Ow, H = p.h != null ? p.h : m.Oh;
    const rr = m.r((H - m.W) / 2);
    // right end fully round, left end a little firmer — like ص
    return { w, cx: w / 2, parts: [closedStroke(G.rrect(m.h, m.h, w - m.h, H - m.h, [rr, rr * 0.75, rr * 0.75, rr]), 'loop')] };
  };

  gen.arc = (m, p) => {
    const w = p.w != null ? p.w : m.Aw;
    const top = p.top != null ? p.top : m.t;
    const bot = p.bot != null ? p.bot : m.dd;
    const endU = p.endU != null ? p.endU : m.h + w * 0.05;
    const rBot = m.r((w - m.W) / 2);
    const rTop = m.r(Math.min(w * 0.3, (top - bot) * 0.22));
    if (p.noTop) {
      const pts = G.rpoly([[w - m.h, top], [w - m.h, bot], [endU, bot]], [0, rBot, 0]);
      return { w, parts: [stroke(pts, ['butt', m.cap], 'arc')] };
    }
    const startU = p.startU != null ? p.startU : m.h;
    const pts = G.rpoly([[startU, top], [w - m.h, top], [w - m.h, bot], [endU, bot]], [0, rTop, rBot, 0]);
    return { w, parts: [stroke(pts, [p.startCap || m.cap, m.cap], 'arc')] };
  };

  gen.nose = (m, p) => {
    // "コ": top bar with a small down-tick on its free (left) end, right side down to the baseline, short leg left
    const w = p.w != null ? p.w : m.Nw;
    const top = m.t;
    const tick = Math.min(m.W * 0.9, (top - m.b) * 0.3);
    const leg = Math.max(m.W * 1.2, m.r(m.W * 1.6) + m.W * 0.6);
    const rc = m.r(Math.min((top - m.b) * 0.45, m.W * 2));
    const pts = G.rpoly([[w - m.h, top - tick], [w - m.h, top], [m.h, top], [m.h, m.b], [m.h + leg, m.b]], [0, m.r(m.W * 0.5), rc, rc, 0]);
    return { w, exitU: m.h + leg, parts: [stroke(pts, [m.cap, 'butt'], 'nose')] };
  };

  gen.ain = (m, p) => {
    const w = p.w != null ? p.w : m.Hw;
    const top = p.top != null ? p.top : m.t;
    const bot = m.b;
    if (p.closed) {
      const cw = w * 0.86;
      const rr = m.r(Math.min(cw, top - bot) * 0.35);
      const pts = G.rpoly([[m.h, top], [cw - m.h, top], [cw / 2, bot]], [rr, rr, m.r(m.W * 0.3)], true);
      return { w: cw, cx: cw / 2, parts: [closedStroke(pts, 'ain')] };
    }
    const rr = m.r((top - bot) / 2);
    const botEnd = p.botEnd != null ? p.botEnd : m.h + w * 0.08;
    const pts = G.rpoly([[m.h, top], [w - m.h, top], [w - m.h, bot], [botEnd, bot]], [0, rr, rr, 0]);
    return { w, cx: w / 2, parts: [stroke(pts, [m.cap, p.botCap || m.cap], 'ain')] };
  };

  gen.tail = (m, p) => {
    const reach = p.reach != null ? p.reach : m.Tw;
    const top = p.top != null ? p.top : m.t;
    const bot = p.bot != null ? p.bot : m.d;
    if (reach <= m.W * 1.05) {
      return { w: m.W, parts: [stroke([[m.h, top], [m.h, bot + (m.cap === 'butt' ? -m.h : 0)]], [p.topCap || 'butt', m.cap], 'tail')] };
    }
    const rr = m.r(Math.min(reach - m.W, top - bot) * 0.98);
    const pts = G.rpoly([[m.h, top], [m.h, bot], [reach - m.h, bot]], [0, rr, 0]);
    return { w: reach, parts: [stroke(pts, [p.topCap || m.cap, m.cap], 'tail')] };
  };

  gen.kaf = (m, p) => {
    const w = p.w != null ? p.w : m.Kw;
    const top = m.a, low = m.t, mid = low + (top - low) * 0.42;
    const rv = m.r(m.W * 0.9);
    const pts = G.rpoly([[m.h + w * 0.06, top], [w - m.h, mid], [m.h, low], [m.h, m.b]], [0, rv, rv, 0]);
    return { w, cx: m.h, parts: [stroke(pts, [m.cap, 'butt'], 'kaf')] };
  };

  gen.heh = (m, p) => {
    if (p.fina) {
      // "ɑ" — head loop with its right side rising a little above it
      const w = m.Lw, H = Math.max(m.L * 0.92, m.W * 2.3);
      const rr = m.r((Math.min(w, H) - m.W) / 2);
      const loop = G.rrect(m.h, m.h, w - m.h, H - m.h, [m.r(m.W * 0.4), rr, rr, m.r(m.W * 0.4)]);
      const stemTop = H + Math.max(m.W * 0.8, H * 0.22) - m.capExt;
      return { w, cx: w / 2, parts: [closedStroke(loop, 'heh'), stroke([[m.h, m.b], [m.h, stemTop]], ['butt', m.cap], 'heh')] };
    }
    const w = p.w != null ? p.w : m.Ew, H = p.h != null ? p.h : m.Eh;
    const top = H - m.h;
    const rt = m.r((Math.min(w, H * 1.6) - m.W) / 2 * 0.9), rb = m.r(m.W * 0.3);
    const outer = G.rrect(m.h, m.b, w - m.h, top, [rb, rb, rt, rt]);
    const iw = Math.max(w * 0.32, m.W * 2.4), ih = Math.max(H * 0.44, m.W * 2.6);
    const c = w / 2, l = c - iw / 2 + m.h, r = c + iw / 2 - m.h;
    const ri = m.r((iw - m.W) / 2);
    const inner = G.rpoly([[l, m.b], [l, ih - m.h], [r, ih - m.h], [r, m.b]], [0, ri, ri, 0]);
    return { w, cx: c, parts: [closedStroke(outer, 'heh'), stroke(inner, ['butt', 'butt'], 'heh')] };
  };

  /* small pieces ---------------------------------------------------------------- */
  gen.dot = (m, p) => {
    const s = (p.size || m.ds);
    const shape = m.dna.dotShape;
    let pts = [];
    if (shape === 'circle') {
      for (let i = 0; i < 40; i++) { const a = i / 40 * Math.PI * 2; pts.push([s / 2 + s / 2 * Math.cos(a), s / 2 + s / 2 * Math.sin(a)]); }
    } else if (shape === 'diamond') {
      const d = s * 1.18 / 2;
      pts = [[s / 2, s / 2 - d], [s / 2 + d, s / 2], [s / 2, s / 2 + d], [s / 2 - d, s / 2]];
    } else {
      const k = s * 0.92, o = (s - k) / 2;
      const rr = Math.min(k / 2, m.r(k * 0.25));
      pts = G.rpoly([[o, o], [o + k, o], [o + k, o + k], [o, o + k]], rr, true);
    }
    return { w: s, h: s, parts: [{ type: 'poly', pts, slot: 'dot' }] };
  };

  gen.hamza = (m, p) => {
    // derived: the ain head at ~half size, drawn with the thinner mark pen, with a short base stroke
    const k = p.scale || 0.5;
    const w = Math.max(m.Hw * k, m.mw * 3.2), H = Math.max((m.X) * k, m.mw * 3.2);
    const h = m.mw / 2;
    const rr = m.r((H - m.mw) / 2);
    const pts = G.rpoly([[h, H - h], [w * 0.78 - h, H - h], [w * 0.78 - h, h], [h, h]], [0, rr, rr, 0]);
    const base = [[w * 0.78 - h, h], [w - h, h]];
    return { w, h: H, parts: [stroke(pts, ['butt', 'butt'], 'hamza', { w: m.mw }), stroke(base, ['butt', 'butt'], 'hamza', { w: m.mw })] };
  };

  gen.madda = (m, p) => {
    const w = Math.max(m.ds * 2.1, m.mw * 5), H = Math.max(m.ds * 0.75, m.mw * 2.2);
    const h = m.mw / 2;
    const pts = [[h, H * 0.35]];
    G.cubic([h, H * 0.35], [w * 0.3, H - h + H * 0.25], [w * 0.55, h - H * 0.2], [w - h, H - h], pts, 32);
    return { w, h: H, parts: [stroke(pts, [m.cap === 'butt' ? 'butt' : 'round', m.cap === 'butt' ? 'butt' : 'round'], 'madda', { w: m.mw })] };
  };

  /* ------------------------------------------------------------ image overrides */
  // override = { img: canvas (alpha mask), wu, hu (size in units), y0 (bottom, units), source }
  S.overrides = {};

  // vertical 3-slice stretch keeping the terminals intact
  function stretchV(img, newH, topFrac, botFrac) {
    const h = img.height, w = img.width;
    const t = Math.round(h * topFrac), b = Math.round(h * botFrac);
    const midSrc = Math.max(1, h - t - b);
    const H = Math.max(t + b + 1, Math.round(newH));
    const c = document.createElement('canvas'); c.width = w; c.height = H;
    const x = c.getContext('2d');
    x.drawImage(img, 0, 0, w, t, 0, 0, w, t);
    x.drawImage(img, 0, t, w, midSrc, 0, t, w, H - t - b);
    x.drawImage(img, 0, h - b, w, b, 0, H - b, w, b);
    return c;
  }

  S.imagePiece = function (id, m, p) {
    const o = S.overrides[id];
    if (!o) return null;
    let img = o.img, wu = o.wu, hu = o.hu, y0 = o.y0;
    if (id === 'stem' || id === 'kaf') {
      // keep the terminal, stretch the shaft to the wanted height
      const wantTop = p.top != null ? p.top : m.A;
      const wantBot = p.free ? 0 : (p.bottom != null ? p.bottom : m.b);
      const want = wantTop - wantBot;
      if (Math.abs(want - hu) > 2) {
        const k = img.height / hu;
        img = stretchV(o.img, want * k, 0.3, p.free ? 0.2 : 0.08);
        hu = want;
      }
      y0 = wantBot;
    } else if (id === 'tooth') {
      y0 = m.b - (o.baseIn || 0);
    }
    const part = { type: 'image', img, u0: 0, y0, w: wu, h: hu, slot: id };
    const out = { w: wu, h: hu, cx: wu / 2, parts: [part], image: true };
    if (id === 'nose') out.exitU = wu - m.h;
    return out;
  };

  // which (slot, variant) combinations may use the image
  const vectorOnly = {
    basin: (p) => !p.full,
    arc: (p) => !!p.noTop || p.startU != null,
    tail: (p) => !!p.joinTop,
    heh: (p) => !!p.fina,
    ain: (p) => !!p.closed
  };

  S.piece = function (id, m, p) {
    p = p || {};
    if (id === 'hamza' && !S.overrides.hamza && S.overrides.ain) {
      // declared rule: الهمزة = رأس العين مصغّراً
      const o = S.overrides.ain, k = p.scale || 0.5;
      return { w: o.wu * k, h: o.hu * k, parts: [{ type: 'image', img: o.img, u0: 0, y0: 0, w: o.wu * k, h: o.hu * k, slot: 'hamza' }], image: true };
    }
    if (id === 'dot' && S.overrides.dot) {
      const o = S.overrides.dot;
      return { w: o.wu, h: o.hu, parts: [{ type: 'image', img: o.img, u0: 0, y0: 0, w: o.wu, h: o.hu, slot: 'dot' }], image: true };
    }
    if (S.overrides[id] && !(vectorOnly[id] && vectorOnly[id](p)) && !p.vector) {
      const ip = S.imagePiece(id, m, p);
      if (ip) return ip;
    }
    const g = gen[id];
    if (!g) throw new Error('unknown slot ' + id);
    return g(m, p);
  };

  /* fit an alpha-mask canvas to a slot when the user uploads a standalone image */
  S.fitUpload = function (id, canvas, m) {
    const meta = S.byId[id];
    const f = meta.fit || { bottom: 0, top: 'X' };
    const val = (k) => {
      if (typeof k === 'number') return k;
      const neg = k[0] === '-'; const key = neg ? k.slice(1) : k;
      const map = { A: m.A, X: m.X, L: m.L, D: m.D, DD: m.DD, Oh: m.Oh, Eh: m.Eh, ds: m.ds, hz: m.X * 0.5, md: m.ds * 0.8 };
      return (neg ? -1 : 1) * map[key];
    };
    const bottom = val(f.bottom), top = val(f.top);
    const hu = top - bottom;
    const wu = canvas.width * hu / canvas.height;
    return { img: canvas, wu, hu, y0: bottom, source: 'upload' };
  };
})(typeof window !== 'undefined' ? window : globalThis);
