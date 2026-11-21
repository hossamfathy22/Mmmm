/* رسم — the 12 key shapes (الأشكال المفتاحية): 6 roots + 6 derived by declared rules,
 * plus 4 small pieces. They are the only source of the font: every letter in every position is
 * assembled from them (see recipes.js).
 *
 * Each piece = a skeleton (centre-line) drawn with the pen, sized by its own parameters.
 * A root takes its parameters from the logo, an upload, the user or the pen DNA.
 * A derived piece inherits from its parent unless it is set itself.
 * Pieces are returned in local (u, y) coordinates: u = 0 is the piece's right edge, u grows left. */
(function (root) {
  'use strict';
  const R = root.Rasm;
  const G = R.geom;

  const S = R.slots = {};

  /* role: 'root' | 'derived'. parent: where a derived piece inherits from. */
  S.list = [
    { id: 'stem',  n: '01', role: 'root', ar: 'الساق', short: 'ساق', en: 'Stem', letters: 'ا ل ك ط', rule: 'خط عمودي بطول الألف، يؤخذ طرفه من الشعار', fit: { bottom: 0, top: 'A' } },
    { id: 'tooth', n: '02', role: 'root', ar: 'السنّ', short: 'سنّ', en: 'Tooth', letters: 'ب ت ث ن ي س ش', rule: 'نتوء قصير يقف على السطر', fit: { bottom: 0, top: 'X' } },
    { id: 'basin', n: '03', role: 'root', ar: 'الحوض', short: 'حوض', en: 'Basin', letters: 'ب ت ث ف ك', rule: 'منحنى مفتوح على السطر، طرفاه إلى أعلى', fit: { bottom: 0, top: 'X' } },
    { id: 'head',  n: '04', role: 'root', ar: 'الرأس', short: 'رأس', en: 'Head', letters: 'ف ق م و', rule: 'عروة صغيرة: بياضها وسماكة القلم', fit: { bottom: 0, top: 'L' } },
    { id: 'arc',   n: '05', role: 'root', ar: 'القوس', short: 'قوس', en: 'Arc', letters: 'ج ح خ ع غ', rule: 'قوس مفتوح لليمين ينزل تحت السطر', fit: { bottom: '-DD', top: 'X' } },
    { id: 'tail',  n: '06', role: 'root', ar: 'الذيل', short: 'ذيل', en: 'Tail', letters: 'ر ز و م', rule: 'نزول تحت السطر ينعطف يساراً', fit: { bottom: '-D', top: 'X' } },
    { id: 'cup',   n: '07', role: 'derived', parent: 'basin', ar: 'الكأس', short: 'كأس', en: 'Cup', letters: 'ن ق ل س ص ي', rule: 'الحوض نازلاً تحت السطر بعمق النزول، يرث استدارته', fit: { bottom: '-D', top: 'X' } },
    { id: 'loop',  n: '08', role: 'derived', parent: 'head', ar: 'العروة', short: 'عروة', en: 'Loop', letters: 'ص ض ط ظ', rule: 'الرأس ممدوداً أفقياً بالاستدارة نفسها', fit: { bottom: 0, top: 'Oh' } },
    { id: 'nose',  n: '09', role: 'derived', parent: 'arc', ar: 'الأنف', short: 'أنف', en: 'Nose', letters: 'ج ح خ', rule: 'أعلى القوس يعود إلى السطر: رأس الحاء', fit: { bottom: 0, top: 'X' } },
    { id: 'ain',   n: '10', role: 'derived', parent: 'arc', ar: 'رأس العين', short: 'عين', en: 'Ain head', letters: 'ع غ ء', rule: 'القوس مصغّراً فوق السطر، ومنه الهمزة', fit: { bottom: 0, top: 'X' } },
    { id: 'kaf',   n: '11', role: 'derived', parent: 'stem', ar: 'خط الكاف', short: 'كاف', en: 'Kaf stroke', letters: 'ك د', rule: 'الساق بشَرطة مائلة، ومنها الدال', fit: { bottom: 0, top: 'A' } },
    { id: 'heh',   n: '12', role: 'derived', parent: 'head', ar: 'الهاء', short: 'هاء', en: 'Heh', letters: 'ه ة', rule: 'قبة من الرأس وفي داخلها عين صغيرة', fit: { bottom: 0, top: 'Eh' } }
  ];
  S.small = [
    { id: 'dot',   ar: 'النقطة', short: 'نقطة', en: 'Dot', rule: 'تُنسخ: نقطة، نقطتان، ثلاث', fit: { bottom: 0, top: 'ds' } },
    { id: 'hamza', ar: 'الهمزة', short: 'همزة', en: 'Hamza', rule: 'رأس العين مصغّراً بقلم رفيع', fit: { bottom: 0, top: 'hz' } },
    { id: 'madda', ar: 'المدّة', short: 'مدّة', en: 'Madda', rule: 'موجة بقلم رفيع', fit: { bottom: 0, top: 'md' } },
    { id: 'conn',  ar: 'الوصلة', short: 'وصلة', en: 'Joint', rule: 'خط السطر بسماكة القلم: الكشيدة', fit: null }
  ];
  S.byId = {};
  S.list.concat(S.small).forEach(s => { S.byId[s.id] = s; });
  S.children = {};
  S.list.forEach(s => { if (s.parent) (S.children[s.parent] = S.children[s.parent] || []).push(s.id); });

  /* ------------------------------------------------------------ piece parameters */
  // ranges are functions of the metrics so they scale with the font
  const PCT = (a, b, st) => ({ min: () => a, max: () => b, step: st || 0.01, fmt: (v) => Math.round(v * 100) + '٪', unit: '' });
  const UNITS = (fa, fb) => ({ min: (m) => Math.round(fa(m)), max: (m) => Math.round(fb(m)), step: 1, fmt: (v) => String(Math.round(v)) });
  const ROUND = { min: () => 0, max: () => 1, step: 0.01, fmt: (v) => (+v).toFixed(2) };
  S.params = {
    stem: [Object.assign({ k: 'lean', ar: 'الميل', min: () => -14, max: () => 18, step: 1, fmt: (v) => Math.round(v) + '°' })],
    tooth: [Object.assign({ k: 'hf', ar: 'الارتفاع' }, PCT(0.5, 1.15)), Object.assign({ k: 'gap', ar: 'التباعد في السين' }, UNITS((m) => m.W * 1.6, (m) => m.W * 5)),
      { k: 'style', ar: 'الشكل', options: [['stub', 'قائم'], ['peak', 'مدبّب'], ['bump', 'مقوّس']] }],
    basin: [Object.assign({ k: 'len', ar: 'الطول' }, UNITS((m) => m.W * 3.2, (m) => m.A * 1.3)), Object.assign({ k: 'tipL', ar: 'الطرف الأيسر' }, PCT(0.2, 1.1)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    head: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 2.2, (m) => m.A * 0.95)), Object.assign({ k: 'h', ar: 'الارتفاع' }, UNITS((m) => m.W * 2.2, (m) => m.A * 0.9)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    arc: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 3, (m) => m.A * 1.1)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    tail: [Object.assign({ k: 'reach', ar: 'الامتداد' }, UNITS((m) => m.W * 1.1, (m) => m.A * 0.8)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    cup: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 3, (m) => m.A * 1.1)), Object.assign({ k: 'tip', ar: 'الطرف الأيسر' }, PCT(0.1, 1.1)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    loop: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 3, (m) => m.A * 1.4)), Object.assign({ k: 'h', ar: 'الارتفاع' }, UNITS((m) => m.W * 2.2, (m) => m.A * 0.85)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    nose: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 2.6, (m) => m.A * 1.1)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    ain: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 2.4, (m) => m.A * 0.8)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    kaf: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 2.6, (m) => m.A * 0.9)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)],
    heh: [Object.assign({ k: 'w', ar: 'العرض' }, UNITS((m) => m.W * 3.6, (m) => m.A * 1.3)), Object.assign({ k: 'h', ar: 'الارتفاع' }, UNITS((m) => m.W * 3, (m) => m.A * 0.95)), Object.assign({ k: 'r', ar: 'الاستدارة' }, ROUND)]
  };

  /* resolve every piece's parameters: explicit value → else the declared rule (parent or pen DNA) */
  S.resolve = function (m, pp) {
    pp = pp || {};
    const W = m.W, A = m.A, X = m.X, wf = m.wf, rho = m.rho, dna = m.dna;
    const pc = {}, src = {};
    const put = (id, k, def, from) => {
      pc[id] = pc[id] || {}; src[id] = src[id] || {};
      const v = pp[id] ? pp[id][k] : undefined;
      if (v != null && v !== '' && !(typeof v === 'number' && !isFinite(v))) { pc[id][k] = v; src[id][k] = 'set'; }
      else { pc[id][k] = def; src[id][k] = from; }
    };
    // roots — from the pen DNA
    put('stem', 'lean', 0, 'dna');
    put('tooth', 'hf', 1, 'dna');
    put('tooth', 'gap', Math.round(W + Math.max(1.15 * W, 0.2 * dna.xh) * wf), 'dna');
    put('tooth', 'style', dna.toothStyle || 'stub', 'dna');
    put('basin', 'len', Math.round(Math.max(0.74 * A * wf, 4.4 * W)), 'dna');
    put('basin', 'tipL', 0.72, 'dna');
    put('basin', 'r', rho, 'dna');
    put('head', 'h', Math.round(dna.loop), 'dna');
    put('head', 'w', Math.round(Math.max(dna.loop * 0.94 * Math.sqrt(wf), 2.3 * W)), 'dna');
    put('head', 'r', rho, 'dna');
    put('arc', 'w', Math.round(Math.max(0.64 * A * wf, 3.6 * W)), 'dna');
    put('arc', 'r', rho, 'dna');
    put('tail', 'reach', Math.round(Math.max(0.36 * A * wf, 2.2 * W)), 'dna');
    put('tail', 'r', rho, 'dna');
    // derived — declared rules
    put('cup', 'w', Math.round(Math.max(pc.basin.len * 0.81, 3.4 * W)), 'basin');
    put('cup', 'tip', +(pc.basin.tipL * 0.7).toFixed(3), 'basin');
    put('cup', 'r', pc.basin.r, 'basin');
    put('loop', 'h', Math.round(Math.max(Math.min(pc.head.h * 1.04, X * 0.95), 2.3 * W)), 'head');
    put('loop', 'w', Math.round(Math.max(pc.head.w * 1.55, pc.loop.h * 1.35)), 'head');
    put('loop', 'r', pc.head.r, 'head');
    put('nose', 'w', Math.round(Math.max(pc.arc.w * 0.94, 3.4 * W)), 'arc');
    put('nose', 'r', pc.arc.r, 'arc');
    put('ain', 'w', Math.round(Math.max(pc.arc.w * 0.72, 2.8 * W)), 'arc');
    put('ain', 'r', pc.arc.r, 'arc');
    put('kaf', 'w', Math.round(Math.max(pc.basin.len * 0.68, 3.1 * W)), 'stem');
    put('kaf', 'r', rho, 'stem');
    put('heh', 'w', Math.round(Math.max(pc.head.w * 1.5, 4.2 * W)), 'head');
    put('heh', 'h', Math.round(Math.max(Math.min(pc.head.h * 1.35, A * 0.85), 3 * W)), 'head');
    put('heh', 'r', pc.head.r, 'head');
    return { pc, src };
  };

  const stroke = (pts, caps, slot, extra) => Object.assign({ type: 'stroke', pts, caps, slot }, extra || {});
  const closedStroke = (pts, slot, extra) => Object.assign({ type: 'stroke', pts, closed: true, slot }, extra || {});
  const rp = (ctrl, r, closed) => G.rpoly(ctrl, r, closed);

  /* ------------------------------------------------------------ vector generators */
  const gen = S.gen = {};

  gen.stem = (m, p) => {
    const top = (p.top != null ? p.top : m.A) - m.capExt;
    const bot = p.free ? m.capExt : (p.bottom != null ? p.bottom : m.b);
    const lean = (p.lean != null ? p.lean : 0) + m.pc.stem.lean;
    const dx = Math.tan(lean * Math.PI / 180) * (top - bot);
    const ctrl = [[m.h, bot], [m.h + dx, top]];
    return { w: m.W + Math.max(0, dx), cx: m.h, top: [m.h + dx, top], parts: [stroke(ctrl, [p.free ? m.cap : 'butt', m.cap], 'stem', { ctrl })] };
  };

  gen.tooth = (m, p) => {
    const pt = m.pc.tooth;
    const top = (p.top != null ? p.top : m.X * pt.hf) - m.capExt;
    const st = pt.style;
    if (st === 'peak') {
      const s = Math.max(m.W * 0.8, (top - m.b) * 0.42), cx = s + m.h;
      const ctrl = [[cx - s, m.b], [cx, top], [cx + s, m.b]];
      return { w: 2 * s + m.W, cx, parts: [stroke(rp(ctrl, [0, m.rr('tooth', m.W * 0.5), 0]), ['butt', 'butt'], 'tooth', { ctrl })] };
    }
    if (st === 'bump') {
      const s = Math.max(m.W * 0.75, (top - m.b) * 0.5), cx = s + m.h;
      const ctrl = [[cx - s, m.b], [cx - s, top], [cx + s, top], [cx + s, m.b]];
      return { w: 2 * s + m.W, cx, parts: [stroke(rp(ctrl, [0, s, s, 0]), ['butt', 'butt'], 'tooth', { ctrl })] };
    }
    const ctrl = [[m.h, m.b], [m.h, top]];
    return { w: m.W, cx: m.h, parts: [stroke(ctrl, ['butt', m.cap], 'tooth', { ctrl })] };
  };

  gen.basin = (m, p) => {
    const pb = m.pc.basin;
    const len = p.len != null ? p.len : pb.len;
    const tipL = m.b + (m.t - m.b) * pb.tipL;
    const rB = m.rr('basin', Math.min(len * 0.32, tipL - m.b));
    if (p.full) {
      const tipR = p.tipR != null ? p.tipR : m.X * m.pc.tooth.hf - m.capExt;
      const rR = m.rr('basin', Math.min(len * 0.3, tipR - m.b) * 0.9);
      const ctrl = [[m.h, tipR], [m.h, m.b], [len - m.h, m.b], [len - m.h, tipL]];
      return { w: len, cx: len / 2, parts: [stroke(rp(ctrl, [0, rR, rB, 0]), [m.cap, m.cap], 'basin', { ctrl })] };
    }
    const ctrl = [[0, m.b], [len - m.h, m.b], [len - m.h, tipL]];
    return { w: len, cx: len / 2, parts: [stroke(rp(ctrl, [0, rB, 0]), ['butt', m.cap], 'basin', { ctrl })] };
  };

  gen.cup = (m, p) => {
    const pcup = m.pc.cup;
    if (p.variant === 'yeh') {
      // ى: a short arm from the top-right, along the baseline, then a wide cup below it
      const w = m.Yw;
      const top = p.top != null ? p.top : m.t;
      const k = Math.max(w * 0.34, m.W * 2.4);
      const tip = m.b + (m.t - m.b) * Math.max(0.15, pcup.tip * 0.84);
      const rr = m.rr('cup', Math.min((w - k) / 2, (m.b - m.d)) * 0.95);
      const rs = m.rr('cup', m.W * 1.4);
      const ctrl = [[m.h, top], [m.h, m.b], [k, m.b], [k, m.d], [w - m.h, m.d], [w - m.h, tip]];
      return { w, cx: k + (w - k) / 2, parts: [stroke(rp(ctrl, [0, rs, rs, rr, rr, 0]), [p.topCap || m.cap, m.cap], 'cup', { ctrl })] };
    }
    const w = p.w != null ? p.w : pcup.w;
    const top = p.top != null ? p.top : m.b;
    const tip = m.b + (m.t - m.b) * pcup.tip;
    const d = p.d != null ? p.d : m.d;
    const rr = m.rr('cup', (w - m.W) / 2);
    const ctrl = [[m.h, top], [m.h, d], [w - m.h, d], [w - m.h, tip]];
    return { w, cx: w / 2, parts: [stroke(rp(ctrl, [0, rr, rr, 0]), [p.topCap || 'butt', m.cap], 'cup', { ctrl })] };
  };

  gen.head = (m, p) => {
    const ph = m.pc.head;
    const w = p.w != null ? p.w : ph.w, H = p.h != null ? p.h : ph.h, y0 = p.y0 || 0;
    const rr = m.rr('head', (Math.min(w, H) - m.W) / 2);
    const ctrl = [[m.h, y0 + m.h], [w - m.h, y0 + m.h], [w - m.h, y0 + H - m.h], [m.h, y0 + H - m.h]];
    return { w, cx: w / 2, parts: [closedStroke(rp(ctrl, rr, true), 'head', { ctrl })] };
  };

  gen.loop = (m, p) => {
    const pl = m.pc.loop;
    const w = p.w != null ? p.w : pl.w, H = p.h != null ? p.h : pl.h;
    const rr = m.rr('loop', (H - m.W) / 2);
    // right end fully round, left end a little firmer — like ص
    const ctrl = [[m.h, m.h], [w - m.h, m.h], [w - m.h, H - m.h], [m.h, H - m.h]];
    return { w, cx: w / 2, parts: [closedStroke(rp(ctrl, [rr, rr * 0.75, rr * 0.75, rr], true), 'loop', { ctrl })] };
  };

  /* القوس: the C under the line. variants:
   *   (default) full C with a top arm (ح isolated)        startU: where the top arm starts
   *   noTop: from its top-left corner down                 from: 'vertex' (ع medial/final head), 'side' (ع isolated) */
  gen.arc = (m, p) => {
    const w = p.w != null ? p.w : m.pc.arc.w;
    const top = p.top != null ? p.top : m.t;
    const bot = p.bot != null ? p.bot : m.dd;
    const endU = p.endU != null ? p.endU : m.h + w * 0.05;
    const rBot = m.rr('arc', (w - m.W) / 2);
    const rTop = m.rr('arc', Math.min(w * 0.3, (top - bot) * 0.22));
    let ctrl, radii, caps;
    if (p.from === 'vertex') {
      ctrl = [[p.vx, m.b], [w - m.h, m.b - (m.b - bot) * 0.32], [w - m.h, bot], [endU, bot]];
      radii = [0, m.rr('arc', m.W * 1.2), rBot, 0]; caps = ['butt', m.cap];
    } else if (p.from === 'side') {
      const mid = m.b - (m.b - bot) * 0.42;
      ctrl = [[p.su, m.b + p.leg], [p.su, m.b], [w - m.h, mid], [w - m.h, bot], [endU, bot]];
      radii = [0, m.rr('arc', m.W * 1.1), m.rr('arc', m.W * 1.6), rBot, 0]; caps = ['butt', m.cap];
    } else if (p.noTop) {
      ctrl = [[w - m.h, top], [w - m.h, bot], [endU, bot]];
      radii = [0, rBot, 0]; caps = ['butt', m.cap];
    } else {
      const startU = p.startU != null ? p.startU : m.h;
      ctrl = [[startU, top], [w - m.h, top], [w - m.h, bot], [endU, bot]];
      radii = [0, rTop, rBot, 0]; caps = [p.startCap || m.cap, m.cap];
    }
    return { w, parts: [stroke(rp(ctrl, radii), caps, 'arc', { ctrl })] };
  };

  /* الأنف: "コ" — top bar with a small down-tick, right side down to the line, a short leg left.
   * variant 'up': the right side rising from the line into the top bar (ح final), ends at u = p.to */
  gen.nose = (m, p) => {
    const w = p.w != null ? p.w : m.pc.nose.w;
    const top = m.t;
    const rc = m.rr('nose', Math.min((top - m.b) * 0.45, m.W * 2));
    if (p.variant === 'up') {
      const ctrl = [[m.h, m.b], [m.h, top], [p.to, top]];
      return { w: p.to, parts: [stroke(rp(ctrl, [0, rc, 0]), ['butt', 'butt'], 'nose', { ctrl })] };
    }
    const tick = Math.min(m.W * 0.9, (top - m.b) * 0.3);
    const leg = Math.max(m.W * 1.2, m.r(m.W * 1.6) + m.W * 0.6);
    const ctrl = [[w - m.h, top - tick], [w - m.h, top], [m.h, top], [m.h, m.b], [m.h + leg, m.b]];
    return { w, exitU: m.h + leg, parts: [stroke(rp(ctrl, [0, m.rr('nose', m.W * 0.5), rc, rc, 0]), [m.cap, 'butt'], 'nose', { ctrl })] };
  };

  gen.ain = (m, p) => {
    const w = p.w != null ? p.w : m.pc.ain.w;
    const top = p.top != null ? p.top : m.t;
    const bot = m.b;
    if (p.closed) {
      const cw = w * 0.86;
      const rr = m.rr('ain', Math.min(cw, top - bot) * 0.35);
      const ctrl = [[m.h, top], [cw - m.h, top], [cw / 2, bot]];
      return { w: cw, cx: cw / 2, parts: [closedStroke(rp(ctrl, [rr, rr, m.rr('ain', m.W * 0.3)], true), 'ain', { ctrl })] };
    }
    const rr = m.rr('ain', (top - bot) / 2);
    const botEnd = p.botEnd != null ? p.botEnd : m.h + w * 0.08;
    const ctrl = [[m.h, top], [w - m.h, top], [w - m.h, bot], [botEnd, bot]];
    return { w, cx: w / 2, parts: [stroke(rp(ctrl, [0, rr, rr, 0]), [m.cap, p.botCap || m.cap], 'ain', { ctrl })] };
  };

  gen.tail = (m, p) => {
    const reach = p.reach != null ? p.reach : m.pc.tail.reach;
    const top = p.top != null ? p.top : m.t;
    const bot = p.bot != null ? p.bot : m.d;
    if (reach <= m.W * 1.05) {
      const ctrl = [[m.h, top], [m.h, bot + (m.cap === 'butt' ? -m.h : 0)]];
      return { w: m.W, parts: [stroke(ctrl, [p.topCap || 'butt', m.cap], 'tail', { ctrl })] };
    }
    const rr = m.rr('tail', Math.min(reach - m.W, top - bot) * 0.98);
    const ctrl = [[m.h, top], [m.h, bot], [reach - m.h, bot]];
    return { w: reach, parts: [stroke(rp(ctrl, [0, rr, 0]), [p.topCap || m.cap, m.cap], 'tail', { ctrl })] };
  };

  /* خط الكاف: the chevron of كـ. variant 'dal': its lower half on the line (د) */
  gen.kaf = (m, p) => {
    if (p.variant === 'dal') {
      const w = m.Dw;
      const ctrl = [[w * 0.44, m.t], [m.h, m.b], [w - m.h, m.b]];
      return { w, cx: m.h, top: [w * 0.44, m.t], parts: [stroke(rp(ctrl, [0, m.rr('kaf', m.W * 1.4), 0]), [m.cap, m.cap], 'kaf', { ctrl })] };
    }
    const w = p.w != null ? p.w : m.pc.kaf.w;
    const lean = Math.tan(m.pc.stem.lean * Math.PI / 180);
    const top = m.a, low = m.t, mid = low + (top - low) * 0.42;
    const rv = m.rr('kaf', m.W * 0.9);
    const ctrl = [[m.h + w * 0.06 + lean * (top - m.b), top], [w - m.h, mid], [m.h + lean * (low - m.b), low], [m.h, m.b]];
    return { w, cx: m.h, parts: [stroke(rp(ctrl, [0, rv, rv, 0]), [m.cap, 'butt'], 'kaf', { ctrl })] };
  };

  gen.heh = (m, p) => {
    if (p.fina) {
      // "ɑ": the head loop with its right side rising a little above it
      const w = m.pc.head.w, H = Math.max(m.pc.head.h * 0.92, m.W * 2.3);
      const rr = m.rr('heh', (Math.min(w, H) - m.W) / 2);
      const ctrl = [[m.h, m.h], [w - m.h, m.h], [w - m.h, H - m.h], [m.h, H - m.h]];
      const loop = rp(ctrl, [m.rr('heh', m.W * 0.4), rr, rr, m.rr('heh', m.W * 0.4)], true);
      const stemTop = H + Math.max(m.W * 0.8, H * 0.22) - m.capExt;
      return { w, cx: w / 2, parts: [closedStroke(loop, 'heh', { ctrl }), stroke([[m.h, m.b], [m.h, stemTop]], ['butt', m.cap], 'heh', { ctrl: [[m.h, m.b], [m.h, stemTop]] })] };
    }
    const w = p.w != null ? p.w : m.pc.heh.w, H = p.h != null ? p.h : m.pc.heh.h;
    const top = H - m.h;
    const rt = m.rr('heh', (Math.min(w, H * 1.6) - m.W) / 2 * 0.9), rb = m.rr('heh', m.W * 0.3);
    const octrl = [[m.h, m.b], [w - m.h, m.b], [w - m.h, top], [m.h, top]];
    const outer = rp(octrl, [rb, rb, rt, rt], true);
    const iw = Math.max(w * 0.32, m.W * 2.4), ih = Math.max(H * 0.44, m.W * 2.6);
    const c = w / 2, l = c - iw / 2 + m.h, r = c + iw / 2 - m.h;
    const ri = m.rr('heh', (iw - m.W) / 2);
    const ictrl = [[l, m.b], [l, ih - m.h], [r, ih - m.h], [r, m.b]];
    const inner = rp(ictrl, [0, ri, ri, 0]);
    return { w, cx: c, parts: [closedStroke(outer, 'heh', { ctrl: octrl }), stroke(inner, ['butt', 'butt'], 'heh', { ctrl: ictrl })] };
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
    const w = Math.max(m.pc.ain.w * k, m.mw * 3.2), H = Math.max((m.X) * k, m.mw * 3.2);
    const h = m.mw / 2;
    const rr = m.rr('ain', (H - m.mw) / 2);
    const ctrl = [[h, H - h], [w * 0.78 - h, H - h], [w * 0.78 - h, h], [h, h]];
    const base = [[w * 0.78 - h, h], [w - h, h]];
    return { w, h: H, parts: [stroke(rp(ctrl, [0, rr, rr, 0]), ['butt', 'butt'], 'hamza', { w: m.mw, ctrl }), stroke(base, ['butt', 'butt'], 'hamza', { w: m.mw })] };
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
  S.useImage = {};   // false → the piece keeps its fitted vector form even though an image exists

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

  // the variants that always use the vector form (an image cannot bend into them)
  const vectorOnly = {
    basin: (p) => !p.full,
    arc: (p) => !!p.noTop || p.startU != null || !!p.from,
    tail: (p) => !!p.joinTop,
    heh: (p) => !!p.fina,
    ain: (p) => !!p.closed,
    nose: (p) => p.variant === 'up',
    kaf: (p) => p.variant === 'dal',
    cup: (p) => p.variant === 'yeh'
  };
  S.imageUsed = (id, p) => !!S.overrides[id] && S.useImage[id] !== false && !(vectorOnly[id] && vectorOnly[id](p || {}));

  S.piece = function (id, m, p) {
    p = p || {};
    if (id === 'hamza' && !S.overrides.hamza && S.overrides.ain && S.useImage.ain !== false) {
      // declared rule: الهمزة = رأس العين مصغّراً
      const o = S.overrides.ain, k = p.scale || 0.5;
      return { w: o.wu * k, h: o.hu * k, parts: [{ type: 'image', img: o.img, u0: 0, y0: 0, w: o.wu * k, h: o.hu * k, slot: 'hamza' }], image: true };
    }
    if (id === 'dot' && S.overrides.dot && S.useImage.dot !== false) {
      const o = S.overrides.dot;
      return { w: o.wu, h: o.hu, parts: [{ type: 'image', img: o.img, u0: 0, y0: 0, w: o.wu, h: o.hu, slot: 'dot' }], image: true };
    }
    if (S.imageUsed(id, p) && !p.vector) {
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

  /* ------------------------------------------------------------ fitting: image → piece parameters */
  // measure an alpha mask: roundness of its outline and of its counters, ink tops near the edges, lean
  S.measureMask = function (img, strokePx) {
    const w = img.width, h = img.height;
    const d = img.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data;
    const cov = new Float32Array(w * h);
    for (let i = 0, j = 3; i < w * h; i++, j += 4) cov[i] = d[j] / 255;
    // pad by one pixel so every contour closes
    const W = w + 2, H = h + 2, pf = new Float32Array(W * H);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pf[(y + 1) * W + x + 1] = cov[y * w + x];
    const polys = R.trace.contours(pf, W, H, 0.5).filter(p => p.length > 8);
    const step = Math.max(2, Math.round(strokePx * 0.2));
    let straight = 0, curved = 0;
    for (const P of polys) {
      const n = P.length;
      for (let i = 0; i < n; i++) {
        const a = P[(i - step + n) % n], b = P[i], c = P[(i + step) % n];
        const v1 = [b[0] - a[0], b[1] - a[1]], v2 = [c[0] - b[0], c[1] - b[1]];
        const t = Math.abs(Math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])) * 180 / Math.PI;
        if (t < 7) straight++; else if (t < 48) curved++;
      }
    }
    const cf = curved / Math.max(1, straight + curved);
    const round = Math.max(0, Math.min(1, (cf - 0.08) / 0.4));
    // counters: background not connected to the border
    const bg = new Uint8Array(w * h), seen = new Uint8Array(w * h), stack = [];
    for (let i = 0; i < w * h; i++) bg[i] = cov[i] < 0.5 ? 1 : 0;
    const flood = (s) => { const out = []; stack.length = 0; stack.push(s); seen[s] = 1; let border = false;
      while (stack.length) { const q = stack.pop(); out.push(q); const x = q % w, y = (q / w) | 0; if (!x || !y || x === w - 1 || y === h - 1) border = true;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue; const r = ny * w + nx; if (bg[r] && !seen[r]) { seen[r] = 1; stack.push(r); } } }
      return { px: out, border }; };
    let holeFill = null, best = 0;
    for (let i = 0; i < w * h; i++) {
      if (!bg[i] || seen[i]) continue;
      const f = flood(i);
      if (f.border || f.px.length < 6) continue;
      let x0 = w, y0 = h, x1 = 0, y1 = 0;
      for (const q of f.px) { const x = q % w, y = (q / w) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      if (f.px.length > best) { best = f.px.length; holeFill = f.px.length / ((x1 - x0 + 1) * (y1 - y0 + 1)); }
    }
    const counterRound = holeFill == null ? null : Math.max(0, Math.min(1, (1 - holeFill) / (1 - Math.PI / 4)));
    // ink tops in the left/right 15% of columns (as heights from the bottom, px)
    const colTop = (x0, x1) => { for (let y = 0; y < h; y++) for (let x = x0; x <= x1; x++) if (cov[y * w + x] > 0.5) return h - y; return 0; };
    const band = Math.max(1, Math.round(w * 0.15));
    const leftTop = colTop(0, band - 1), rightTop = colTop(w - band, w - 1);
    // lean: centroid shift between the lower and upper thirds
    const rowC = (y0, y1) => { let s = 0, n = 0; for (let y = y0; y < y1; y++) for (let x = 0; x < w; x++) { const v = cov[y * w + x]; if (v > 0.5) { s += x; n++; } } return n ? s / n : w / 2; };
    const t1 = Math.round(h * 0.2), t2 = Math.round(h * 0.45), b1 = Math.round(h * 0.55), b2 = Math.round(h * 0.8);
    const leanDeg = Math.atan2(rowC(b1, b2) - rowC(t1, t2), (b1 + b2 - t1 - t2) / 2) * 180 / Math.PI;
    return { round, counterRound, leftTop, rightTop, leanDeg, w, h };
  };

  // parameters of a piece read off an image override (units), clamped to the piece's ranges
  S.fitParams = function (id, o, m) {
    if (!o || !S.params[id]) return {};
    const k = o.wu / o.img.width;                      // units per pixel
    const g = S.measureMask(o.img, m.W / k);
    const out = {};
    const top = o.y0 + o.hu;
    const tipF = (px) => Math.max(0.05, ((o.y0 + px * k) - m.b) / Math.max(1, m.t - m.b));
    const r = g.counterRound != null ? g.counterRound : g.round;
    switch (id) {
      case 'stem': out.lean = Math.round(g.leanDeg); break;
      case 'tooth': out.hf = (top + m.capExt * 0) / m.X; break;
      case 'basin': out.len = o.wu; out.tipL = tipF(g.leftTop); out.r = g.round; break;
      case 'cup': out.w = o.wu; out.tip = tipF(g.leftTop); out.r = g.round; break;
      case 'head': out.w = o.wu; out.h = o.hu; out.r = r; break;
      case 'loop': out.w = o.wu; out.h = o.hu; out.r = r; break;
      case 'arc': case 'nose': case 'ain': case 'kaf': out.w = o.wu; out.r = g.round; break;
      case 'tail': out.reach = o.wu; out.r = g.round; break;
      case 'heh': out.w = o.wu; out.h = o.hu; out.r = r; break;
    }
    for (const def of S.params[id]) {
      if (out[def.k] == null || def.options) continue;
      const lo = def.min(m), hi = def.max(m);
      out[def.k] = Math.max(lo, Math.min(hi, out[def.k]));
      out[def.k] = def.step >= 1 ? Math.round(out[def.k]) : +(+out[def.k]).toFixed(2);
    }
    return out;
  };
})(typeof window !== 'undefined' ? window : globalThis);
