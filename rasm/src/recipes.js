/* رسم — recipes (الوصفات): every letter in every position = key shapes in a declared order.
 * A recipe draws ink starting at u = 0 (right) and declares where the joining strokes attach:
 *   entry → connection to the previous letter (right side), exit → to the next letter (left side).
 * The builder adds side bearings or connections and computes the advance width. */
(function (root) {
  'use strict';
  const R = root.Rasm;
  const G = R.geom;

  /* ---------------------------------------------------------------- builder */
  class Builder {
    constructor(m, form) {
      this.m = m; this.form = form; this.parts = [];
      this.entry = null; this.exit = null; this.anc = {};
    }
    /* rounded polyline. opts: { slot, caps, closed, tags (per segment), w } */
    S(pts, r, opts) {
      opts = opts || {};
      const m = this.m;
      const caps = opts.caps || [m.cap, m.cap];
      const base = { type: 'stroke', w: opts.w };
      if (!opts.tags || opts.closed) {
        this.parts.push(Object.assign({}, base, { pts: G.rpoly(pts, r, !!opts.closed), closed: !!opts.closed, caps, slot: opts.slot || 'conn' }));
        return this;
      }
      // split the stroke where its tag changes (in the middle of a straight segment → seamless)
      const tags = opts.tags;
      const rad = (i) => Array.isArray(r) ? (r[i] || 0) : (r || 0);
      const P = [pts[0]], RR = [rad(0)], marks = [];
      for (let i = 1; i < pts.length; i++) {
        if (i - 1 > 0 && tags[i - 1] !== tags[i - 2]) {
          const segLen = G.len(G.sub(pts[i], pts[i - 1])) || 1;
          const mid = G.lerp(pts[i - 1], pts[i], Math.min(0.5, (m.W * 2.2) / segLen));
          P.push(mid); RR.push(0); marks.push(mid);
        }
        P.push(pts[i]); RR.push(rad(i));
      }
      const flat = G.rpoly(P, RR, false);
      const chunks = [];
      let cur = [], mi = 0;
      for (const q of flat) {
        cur.push(q);
        const mk = marks[mi];
        if (mk && Math.abs(q[0] - mk[0]) < 1e-6 && Math.abs(q[1] - mk[1]) < 1e-6) { chunks.push(cur); cur = [q]; mi++; }
      }
      chunks.push(cur);
      const segTags = [tags[0]];
      for (let i = 1; i < tags.length; i++) if (tags[i] !== tags[i - 1]) segTags.push(tags[i]);
      chunks.forEach((c, k) => {
        if (c.length < 2) return;
        this.parts.push(Object.assign({}, base, {
          pts: c, caps: [k === 0 ? caps[0] : 'butt', k === chunks.length - 1 ? caps[1] : 'butt'], slot: segTags[k] || 'conn'
        }));
      });
      return this;
    }
    P(id, u, y, p) {
      const piece = R.slots.piece(id, this.m, p || {});
      for (const part of piece.parts) this.parts.push(R.shiftPart(part, u, y || 0));
      return piece;
    }
    /* baseline with an optional rounded L-corner going up into a vertical at u0 */
    base(u0, u1, wrapUp) {
      const m = this.m;
      if (wrapUp) {
        const rr = Math.min(m.r(m.W * 1.6), Math.abs(u1 - u0) * 0.5), leg = Math.max(m.W * 0.8, rr + m.W * 0.3);
        return this.S([[u0, m.b + leg], [u0, m.b], [u1, m.b]], [0, rr, 0], { caps: ['butt', 'butt'], slot: 'conn' });
      }
      return this.S([[u0, m.b], [u1, m.b]], 0, { caps: ['butt', 'butt'], slot: 'conn' });
    }
    conn(at, toU) {
      const m = this.m;
      const pts = [], rr = [];
      if (at.dir === 'up' || at.dir === 'down') {
        const rad = Math.min(m.r(m.W * 1.6), m.jl * 0.55), leg = Math.max(m.W * 0.8, rad + m.W * 0.3);
        pts.push([at.u, m.b + (at.dir === 'up' ? leg : -leg)]); rr.push(0);
        pts.push([at.u, m.b]); rr.push(rad);
      } else { pts.push([at.u, m.b]); rr.push(0); }
      pts.push([toU, m.b]); rr.push(0);
      this.parts.push({ type: 'stroke', pts: G.rpoly(pts, rr, false), caps: ['butt', 'butt'], slot: 'conn' });
    }
    dots(n, at, side) {
      if (!at || !n) return;
      const m = this.m;
      const piece = R.slots.piece('dot', m, {});
      const s = piece.w, hs = piece.h || piece.w, g = m.dg;
      const [u, y] = at;
      const cs = n === 1 ? [[u, 0]] : n === 2 ? [[u - (s + g) / 2, 0], [u + (s + g) / 2, 0]]
        : [[u - (s + g) / 2, 0], [u + (s + g) / 2, 0], [u, 1]];
      for (const [cu, row] of cs) {
        const off = row * (hs + g * 0.75);
        const yy = side === 'below' ? y - off - hs : y + off;
        for (const part of piece.parts) this.parts.push(R.shiftPart(part, cu - s / 2, yy));
      }
    }
    mark(kind, at, side, scale) {
      if (!at) return;
      const m = this.m;
      const piece = R.slots.piece(kind, m, { scale: scale || 0.5 });
      const hh = piece.h || m.X * 0.5;
      const yy = side === 'below' ? at[1] - hh : at[1];
      for (const part of piece.parts) this.parts.push(R.shiftPart(part, at[0] - piece.w / 2, yy));
      return piece;
    }
  }
  R.Builder = Builder;

  /* ---------------------------------------------------------------- rasm recipes */
  const RX = R.recipes = {};
  const joinsR = (f) => f === 'fina' || f === 'medi';
  const joinsL = (f) => f === 'init' || f === 'medi';

  RX.alef = (b, m, f) => {
    const fin = f === 'fina';
    const s = b.P('stem', 0, 0, { free: !fin });
    if (fin) b.entry = { u: s.cx, dir: 'up' };
    b.anc.above = [s.cx, m.A + m.mg * 0.7];
    b.anc.below = [s.cx, -m.mg * 0.7];
  };

  function tooth(b, m, f) {
    const t = b.P('tooth', 0, 0);
    const dir = m.dna.toothStyle === 'stub' && !t.image ? 'up' : 'flat';
    b.exit = { u: t.cx, dir };
    if (f === 'medi') b.entry = { u: t.cx, dir };
    b.anc.above = [t.cx, m.X + m.mg];
    b.anc.below = [t.cx, -m.mg];
    b.anc.inside = b.anc.below;
    return t;
  }

  RX.beh = (b, m, f) => {
    if (joinsL(f)) return tooth(b, m, f);
    const p = b.P('basin', 0, 0, { full: true });
    if (f === 'fina') b.entry = { u: p.image ? p.w * 0.1 : m.h, dir: p.image ? 'flat' : 'up' };
    b.anc.above = [p.w * 0.5, m.W + m.mg * 0.55];
    b.anc.below = [p.w * 0.5, -m.mg];
  };

  /* cup whose right arm rises to `top`; image cups get a vector extension when needed */
  function cupArm(b, m, u, top, opts) {
    const c = b.P('cup', u, 0, Object.assign({ top, topCap: m.cap }, opts || {}));
    if (c.image) {
      const part = b.parts[b.parts.length - 1];
      const imgTop = part.y0 + part.h;
      if (top > imgTop - m.W * 0.5) b.S([[u + m.h, imgTop - m.W], [u + m.h, top]], 0, { slot: 'cup', caps: ['butt', m.cap] });
    }
    return c;
  }

  RX.noon = (b, m, f) => {
    if (joinsL(f)) return tooth(b, m, f);
    const fin = f === 'fina';
    const c = cupArm(b, m, 0, fin ? m.b + (m.t - m.b) * 0.62 : m.t);
    if (fin) b.entry = { u: m.h, dir: 'up' };
    b.anc.above = [c.w * 0.5, m.W + m.mg * 0.45];
    b.anc.below = [c.w * 0.5, m.d - m.h - m.mg];
  };

  RX.yeh = (b, m, f) => {
    if (joinsL(f)) return tooth(b, m, f);
    const fin = f === 'fina';
    const y = b.P('yeh', 0, 0, { top: fin ? m.b + (m.t - m.b) * 0.55 : m.t, topCap: m.cap });
    if (fin) b.entry = { u: m.h, dir: 'up' };
    b.anc.above = [m.h + m.W * 0.3, m.X + m.mg * 0.6];
    b.anc.below = [y.cx, m.d - m.h - m.mg * 0.8];
  };

  RX.seen = (b, m, f) => {
    const sg = m.sg;
    const t0 = b.P('tooth', 0, 0);
    const c0 = t0.cx, c1 = c0 + sg, c2 = c0 + 2 * sg;
    b.P('tooth', c1 - t0.cx, 0);
    b.base(c0, c2, !joinsR(f) && m.dna.toothStyle === 'stub' && !t0.image);
    if (f === 'isol' || f === 'fina') cupArm(b, m, c2 - m.h, m.t);
    else { b.P('tooth', c2 - t0.cx, 0); b.exit = { u: c2, dir: 'flat' }; }
    if (joinsR(f)) b.entry = { u: c0, dir: 'flat' };
    b.anc.above = [c1, m.X + m.mg];
    b.anc.below = [c1, -m.mg];
  };

  RX.sad = (b, m, f) => {
    const o = b.P('loop', 0, 0);
    if (joinsR(f)) b.entry = { u: o.w * 0.5, dir: 'flat' };
    if (joinsL(f)) b.exit = { u: o.w * 0.5, dir: 'flat' };
    else cupArm(b, m, o.w - m.W, m.Oh * 0.5, { topCap: 'butt' });
    b.anc.above = [o.w * 0.5, m.Oh + m.mg];
  };

  RX.tah = (b, m, f) => {
    const o = b.P('loop', 0, 0);
    const su = o.w * 0.72;
    const s = b.P('stem', su - m.h, 0, { bottom: m.Oh - m.h });
    if (joinsR(f)) b.entry = { u: o.w * 0.5, dir: 'flat' };
    if (joinsL(f)) b.exit = { u: o.w * 0.5, dir: 'flat' };
    b.anc.above = [o.w * 0.34, m.Oh + m.mg];
    return s;
  };

  RX.ain = (b, m, f) => {
    if (f === 'init') {
      const a = b.P('ain', 0, 0, { botEnd: m.h, botCap: m.cap });
      b.exit = { u: a.w - m.h, dir: a.image ? 'flat' : 'up' };
      b.anc.above = [a.w * 0.5, m.X + m.mg];
      return;
    }
    if (f === 'medi' || f === 'fina') {
      const a = b.P('ain', 0, 0, { closed: true });
      b.entry = { u: a.cx, dir: 'flat' };
      if (f === 'medi') b.exit = { u: a.cx, dir: 'flat' };
      else {
        const w = m.Aw;
        b.S([[a.cx, m.b], [w - m.h, m.b - (m.b - m.dd) * 0.32], [w - m.h, m.dd], [m.h + w * 0.04, m.dd]],
          [0, m.r(m.W * 1.2), m.r((w - m.W) / 2), 0], { slot: 'arc', caps: ['butt', m.cap] });
      }
      b.anc.above = [a.cx, m.X + m.mg];
      return;
    }
    // isolated: the head (a small c) and, growing out of its left side, the big arc under the line (ε-like)
    const hw = m.Hw, w = Math.max(m.Aw, hw + m.W * 1.2), top = m.t;
    const mid = m.b - (m.b - m.dd) * 0.42;
    const rh = m.r((top - m.b) / 2);
    const a = b.P('ain', 0, 0, { botEnd: m.h + hw * 0.22, botCap: m.cap });
    const leg = Math.max(m.W * 0.9, rh * 0.6 + m.W * 0.5);
    b.S([[a.w - m.h, m.b + leg], [a.w - m.h, m.b], [w - m.h, mid], [w - m.h, m.dd], [m.h + w * 0.04, m.dd]],
      [0, m.r(m.W * 1.1), m.r(m.W * 1.6), m.r((w - m.W) / 2), 0], { slot: 'arc', caps: ['butt', m.cap] });
    b.anc.above = [a.w * 0.5, m.X + m.mg];
  };

  RX.hah = (b, m, f) => {
    if (f === 'isol') {
      const a = b.P('arc', 0, 0, { top: m.t });
      b.anc.above = [a.w * 0.42, m.X + m.mg];
      b.anc.inside = [a.w * 0.52, m.b - m.ds * 0.15];
      return;
    }
    if (joinsL(f)) {
      const n = b.P('nose', 0, 0);
      b.exit = { u: n.exitU, dir: 'flat' };
      if (f === 'medi') b.entry = { u: m.h, dir: 'flat' };
      b.anc.above = [n.w * 0.55, m.X + m.mg];
      b.anc.inside = [n.w * 0.5, -m.mg];
      return;
    }
    const w = m.Aw, top = m.t;
    const rc = m.r(Math.min((top - m.b) * 0.45, m.W * 2));
    b.S([[m.h, m.b], [m.h, top], [w - m.h, top], [w - m.h, m.dd], [m.h + w * 0.04, m.dd]],
      [0, rc, m.r(Math.min(w * 0.3, (top - m.dd) * 0.22)), m.r((w - m.W) / 2), 0],
      { tags: ['nose', 'arc', 'arc', 'arc'], caps: ['butt', m.cap] });
    b.entry = { u: m.h, dir: 'flat' };
    b.anc.above = [w * 0.5, m.X + m.mg];
    b.anc.inside = [w * 0.55, m.b - m.ds * 0.15];
  };

  RX.dal = (b, m, f) => {
    const w = m.Dw;
    b.S([[w * 0.44, m.t], [m.h, m.b], [w - m.h, m.b]], [0, m.r(m.W * 1.4), 0], { slot: 'kaf', caps: [m.cap, m.cap] });
    if (f === 'fina') b.entry = { u: m.h, dir: 'flat' };
    b.anc.above = [w * 0.46, m.X + m.mg];
  };

  RX.reh = (b, m, f) => {
    if (f === 'fina') {
      b.P('tail', 0, 0, { top: m.b, topCap: 'butt', joinTop: true });
      b.entry = { u: m.h, dir: 'down' };
      b.anc.above = [m.h + m.W * 0.3, m.W + m.mg];
    } else {
      b.P('tail', 0, 0, { top: m.t });
      b.anc.above = [m.h, m.X + m.mg];
    }
  };

  RX.waw = (b, m, f) => {
    const hd = b.P('head', 0, 0);
    b.P('tail', 0, 0, { top: m.L * 0.5, topCap: 'butt', joinTop: true, reach: Math.max(m.Tw, hd.w * 0.95) });
    if (f === 'fina') b.entry = { u: hd.w * 0.5, dir: 'flat' };
    b.anc.above = [hd.w * 0.5, m.L + m.mg];
  };

  RX.meem = (b, m, f) => {
    const hd = b.P('head', 0, 0);
    if (joinsR(f)) b.entry = { u: hd.w * 0.5, dir: 'flat' };
    if (joinsL(f)) b.exit = { u: hd.w * 0.5, dir: 'flat' };
    else b.P('tail', hd.w - m.W, 0, { top: m.L * 0.5, topCap: 'butt', joinTop: true, reach: m.W, bot: m.dd });
  };

  RX.feh = (b, m, f) => {
    const hd = b.P('head', 0, 0);
    if (joinsR(f)) b.entry = { u: hd.w * 0.5, dir: 'flat' };
    if (joinsL(f)) b.exit = { u: hd.w * 0.5, dir: 'flat' };
    else b.P('basin', hd.w * 0.5, 0, { len: hd.w * 0.5 + m.Bl * 0.7 });
    b.anc.above = [hd.w * 0.5, m.L + m.mg];
  };

  RX.qaf = (b, m, f) => {
    if (joinsL(f)) return RX.feh(b, m, f);
    const hd = b.P('head', 0, 0);
    if (f === 'fina') b.entry = { u: hd.w * 0.5, dir: 'flat' };
    cupArm(b, m, hd.w - m.W, m.L * 0.5, { topCap: 'butt' });
    b.anc.above = [hd.w * 0.5, m.L + m.mg];
  };

  RX.kaf = (b, m, f) => {
    if (joinsL(f)) {
      const k = b.P('kaf', 0, 0);
      b.exit = { u: k.cx, dir: k.image ? 'flat' : 'up' };
      if (f === 'medi') b.entry = { u: k.cx, dir: 'flat' };
      return;
    }
    const s = b.P('stem', 0, 0);
    const lu = s.cx;
    const rr = m.r(m.W * 1.6), leg = Math.max(m.W, rr + m.W * 0.3);
    b.S([[lu, m.b + leg], [lu, m.b], [lu + leg, m.b]], [0, rr, 0], { slot: 'conn', caps: ['butt', 'butt'] });
    const len = m.Bl * 0.9;
    b.P('basin', lu + leg - 1, 0, { len });
    if (f === 'fina') b.entry = { u: lu, dir: 'flat' };
    b.mark('hamza', [lu + m.h + (len + leg) * 0.48, m.t * 0.62 + m.A * 0.18], 'above', 0.4);
  };

  RX.lam = (b, m, f) => {
    const s = b.P('stem', 0, 0);
    if (f === 'isol' || f === 'fina') b.P('cup', s.cx - m.h, 0, { top: m.b, w: m.Cw * 1.04 });
    if (joinsL(f)) b.exit = { u: s.cx, dir: 'up' };
    if (joinsR(f)) b.entry = { u: s.cx, dir: f === 'fina' ? 'flat' : 'up' };
    b.anc.above = [s.cx, m.A + m.mg];
  };

  RX.heh = (b, m, f) => {
    if (f === 'fina') {
      const e = b.P('heh', 0, 0, { fina: true });
      b.entry = { u: m.h, dir: 'up' };
      b.anc.above = [e.w * 0.62, m.L + m.mg * 0.9];
      return;
    }
    const e = b.P('heh', 0, 0);
    if (joinsL(f)) b.exit = { u: e.w - m.h, dir: e.image ? 'flat' : 'up' };
    if (f === 'medi') b.entry = { u: m.h, dir: e.image ? 'flat' : 'up' };
    b.anc.above = [e.w * 0.5, m.Eh + m.mg];
  };

  RX.hamza = (b, m) => {
    b.mark('hamza', [m.Hw * 0.42, 0], 'above', 0.78);
  };

  RX.lamalef = (b, m, f) => {
    const La = Math.max(0.46 * m.A * m.wf, 3.8 * m.W);
    const foot = La * 0.42;
    b.S([[m.h, m.a], [m.h, m.b], [foot, m.b], [La - m.h, m.a]], [0, m.r(m.W * 1.4), m.r(m.W * 1.1), 0],
      { tags: ['stem', 'conn', 'stem'], caps: [m.cap, m.cap] });
    if (f === 'fina') b.entry = { u: m.h, dir: 'flat' };
    b.anc.above = [La - m.h, m.A + m.mg * 0.7];
    b.anc.below = [foot + (La - foot) * 0.25, -m.mg * 0.7];
  };

  /* ---------------------------------------------------------------- digits & punctuation */
  const DG = R.digits = {};
  function digitBox(m) {
    const w = Math.round(Math.max(0.5 * m.A, 3.6 * m.W));
    return { w, c: w / 2, top: m.A * 0.84 - m.capExt, bot: m.capExt, fw: Math.max(w * 0.62, m.W * 2.6) };
  }
  DG[0] = (b, m, d) => { const s = R.slots.piece('dot', m, {}); for (const p of s.parts) b.parts.push(R.shiftPart(p, d.c - s.w / 2, m.A * 0.36 - s.h / 2)); };
  DG[1] = (b, m, d) => { b.S([[d.c, d.bot], [d.c, d.top]], 0, { slot: 'stem' }); };
  DG[2] = (b, m, d) => {
    const q = d.fw * 0.3, fh = (d.top - d.bot) * 0.26;
    b.S([[d.c + q, d.bot], [d.c + q, d.top - fh], [d.c + q - d.fw * 0.75, d.top]], [0, m.r(fh * 0.8), 0], { tags: ['stem', 'tooth'] });
  };
  DG[3] = (b, m, d) => {
    const q = d.fw * 0.36, fh = (d.top - d.bot) * 0.22, u0 = d.c + q, u1 = u0 - d.fw * 0.42, u2 = u0 - d.fw * 0.84;
    b.S([[u0, d.bot], [u0, d.top - fh], [u2, d.top - fh], [u2, d.top]], [0, m.r(m.W), m.r(m.W * 0.8), 0], { tags: ['stem', 'tooth', 'tooth'] });
    b.S([[u1, d.top - fh], [u1, d.top]], 0, { slot: 'tooth', caps: ['butt', m.cap] });
  };
  DG[4] = (b, m, d) => {
    const top = d.top, bot = Math.max(d.bot, m.b), mid = bot + (top - bot) * 0.52, u0 = d.c - d.fw / 2, u1 = d.c + d.fw / 2;
    const uw = d.c - d.fw * 0.12;
    b.S([[u0 + m.W * 0.4, top], [u1 - m.W * 0.2, top], [u1 - m.W * 0.2, mid], [uw, mid], [u1, mid - (mid - bot) * 0.4], [u1, bot], [u0, bot]],
      [0, m.r(m.W * 1.2), m.r(m.W * 0.8), m.r(m.W * 0.6), m.r(m.W * 1.2), m.r(m.W * 1.4), 0], { tags: ['ain', 'ain', 'ain', 'arc', 'arc', 'arc'] });
  };
  DG[5] = (b, m, d) => {
    const H = (d.top - d.bot) * 0.62 + m.W, w = Math.min(d.fw, H * 0.9);
    const hd = R.slots.piece('head', m, { w, h: H, y0: 0 });
    for (const p of hd.parts) b.parts.push(R.shiftPart(p, d.c - w / 2, 0));
  };
  DG[6] = (b, m, d) => {
    const q = d.fw * 0.34;
    b.S([[d.c - q, d.bot], [d.c - q, d.top], [d.c - q + d.fw * 0.8, d.top]], [0, m.r(m.W * 1.2), 0], { tags: ['stem', 'kaf'] });
  };
  DG[7] = (b, m, d) => { b.S([[d.c - d.fw / 2, d.top], [d.c, d.bot + m.h * 0.2], [d.c + d.fw / 2, d.top]], [0, m.r(m.W * 0.6), 0], { slot: 'kaf' }); };
  DG[8] = (b, m, d) => { b.S([[d.c - d.fw / 2, d.bot], [d.c, d.top - m.h * 0.2], [d.c + d.fw / 2, d.bot]], [0, m.r(m.W * 0.6), 0], { slot: 'kaf' }); };
  DG[9] = (b, m, d) => {
    const q = d.fw * 0.34, su = d.c - q;
    b.S([[su, d.bot], [su, d.top]], 0, { slot: 'stem' });
    const H = (d.top - d.bot) * 0.5, w = Math.min(d.fw * 0.85, H * 1.05);
    const hd = R.slots.piece('head', m, { w, h: H });
    for (const p of hd.parts) b.parts.push(R.shiftPart(p, su - m.h, d.top + m.capExt - H));
  };

  const PU = R.punct = {};
  PU.period = (b, m) => { const s = R.slots.piece('dot', m, {}); for (const p of s.parts) b.parts.push(R.shiftPart(p, 0, 0)); };
  PU.colon = (b, m) => { const s = R.slots.piece('dot', m, {}); for (const p of s.parts) { b.parts.push(R.shiftPart(p, 0, 0)); b.parts.push(R.shiftPart(p, 0, m.X * 0.62)); } };
  function commaShape(b, m, y0) {
    const s = R.slots.piece('dot', m, {});
    for (const p of s.parts) b.parts.push(R.shiftPart(p, 0, y0));
    const w = m.mw;
    b.S([[s.w * 0.25, y0 + s.h * 0.6], [s.w * 0.25, y0 + s.h * 1.25], [s.w * 0.1 - s.w * 0.55, y0 + s.h * 2.0]], [0, s.w * 0.7, 0], { slot: 'dot', w, caps: ['butt', m.cap] });
  }
  PU.comma = (b, m) => commaShape(b, m, m.capExt * 0 + 0);
  PU.semicolon = (b, m) => {
    const s = R.slots.piece('dot', m, {});
    for (const p of s.parts) b.parts.push(R.shiftPart(p, 0, 0));
    commaShape(b, m, m.X * 0.5);
  };
  PU.question = (b, m) => {
    const fw = Math.max(m.W * 3, m.X * 0.62), top = m.A * 0.78 - m.capExt, mid = top - (top - m.X * 0.2) * 0.5, low = m.ds + m.dg * 2;
    b.S([[m.h, top - fw * 0.35], [m.h, top], [fw - m.h, top], [fw - m.h, mid], [fw / 2, mid], [fw / 2, low]],
      [0, m.r(fw * 0.4), m.r(fw * 0.4), m.r(m.W), m.r(m.W), 0], { slot: 'ain' });
    const s = R.slots.piece('dot', m, {});
    for (const p of s.parts) b.parts.push(R.shiftPart(p, fw / 2 - s.w / 2, 0));
  };
  PU.exclam = (b, m) => {
    const s = R.slots.piece('dot', m, {});
    for (const p of s.parts) b.parts.push(R.shiftPart(p, 0, 0));
    b.S([[s.w / 2, s.h + m.dg * 2 + m.capExt], [s.w / 2, m.A * 0.78 - m.capExt]], 0, { slot: 'stem' });
  };
  PU.hyphen = (b, m) => { b.S([[m.h, m.X * 0.42], [m.h + Math.max(m.W * 2.5, m.X * 0.5), m.X * 0.42]], 0, { slot: 'conn' }); };

  /* ---------------------------------------------------------------- inventory */
  // [name, codepoint, joining, rasm, marks]
  const L = [
    ['alef', 0x627, 'R', 'alef'],
    ['alefHamzaAbove', 0x623, 'R', 'alef', { hamza: 'above' }],
    ['alefHamzaBelow', 0x625, 'R', 'alef', { hamza: 'below' }],
    ['alefMadda', 0x622, 'R', 'alef', { madda: 1 }],
    ['beh', 0x628, 'D', 'beh', { dots: [1, 'below'] }],
    ['teh', 0x62A, 'D', 'beh', { dots: [2, 'above'] }],
    ['theh', 0x62B, 'D', 'beh', { dots: [3, 'above'] }],
    ['jeem', 0x62C, 'D', 'hah', { dots: [1, 'inside'] }],
    ['hah', 0x62D, 'D', 'hah'],
    ['khah', 0x62E, 'D', 'hah', { dots: [1, 'above'] }],
    ['dal', 0x62F, 'R', 'dal'],
    ['thal', 0x630, 'R', 'dal', { dots: [1, 'above'] }],
    ['reh', 0x631, 'R', 'reh'],
    ['zain', 0x632, 'R', 'reh', { dots: [1, 'above'] }],
    ['seen', 0x633, 'D', 'seen'],
    ['sheen', 0x634, 'D', 'seen', { dots: [3, 'above'] }],
    ['sad', 0x635, 'D', 'sad'],
    ['dad', 0x636, 'D', 'sad', { dots: [1, 'above'] }],
    ['tah', 0x637, 'D', 'tah'],
    ['zah', 0x638, 'D', 'tah', { dots: [1, 'above'] }],
    ['ain', 0x639, 'D', 'ain'],
    ['ghain', 0x63A, 'D', 'ain', { dots: [1, 'above'] }],
    ['feh', 0x641, 'D', 'feh', { dots: [1, 'above'] }],
    ['qaf', 0x642, 'D', 'qaf', { dots: [2, 'above'] }],
    ['kaf', 0x643, 'D', 'kaf'],
    ['lam', 0x644, 'D', 'lam'],
    ['meem', 0x645, 'D', 'meem'],
    ['noon', 0x646, 'D', 'noon', { dots: [1, 'above'] }],
    ['heh', 0x647, 'D', 'heh'],
    ['tehMarbuta', 0x629, 'R', 'heh', { dots: [2, 'above'] }],
    ['waw', 0x648, 'R', 'waw'],
    ['wawHamza', 0x624, 'R', 'waw', { hamza: 'above' }],
    ['alefMaksura', 0x649, 'D', 'yeh'],
    ['yeh', 0x64A, 'D', 'yeh', { dots: [2, 'below'] }],
    ['yehHamza', 0x626, 'D', 'yeh', { hamza: 'above' }],
    ['hamza', 0x621, 'U', 'hamza'],
    // the dotless skeletons themselves, so the rasm can be typed
    ['dotlessBeh', 0x66E, 'D', 'beh'],
    ['dotlessFeh', 0x6A1, 'D', 'feh'],
    ['dotlessQaf', 0x66F, 'D', 'qaf'],
    ['noonGhunna', 0x6BA, 'D', 'noon']
  ];
  R.letters = L.map(([name, uni, join, rasm, marks]) => ({ name, uni, join, rasm, marks: marks || {} }));
  R.letterByName = {}; R.letterByUni = {};
  R.letters.forEach(l => { R.letterByName[l.name] = l; R.letterByUni[l.uni] = l; });

  R.formsOf = (join) => join === 'D' ? ['isol', 'init', 'medi', 'fina'] : join === 'R' ? ['isol', 'fina'] : ['isol'];

  // the 18 dotless skeletons (الرسم) shown in the UI
  R.rasm18 = [
    ['alef', 'ا'], ['beh', 'ٮ'], ['hah', 'ح'], ['dal', 'د'], ['reh', 'ر'], ['seen', 'س'], ['sad', 'ص'], ['tah', 'ط'], ['ain', 'ع'],
    ['feh', 'ڡ'], ['qaf', 'ٯ'], ['kaf', 'ك'], ['lam', 'ل'], ['meem', 'م'], ['noon', 'ں'], ['heh', 'ه'], ['waw', 'و'], ['yeh', 'ى']
  ];

  R.ligatures = [
    { name: 'lamAlef', alef: 'alef', marks: {}, pf: [0xFEFB, 0xFEFC] },
    { name: 'lamAlefHamzaAbove', alef: 'alefHamzaAbove', marks: { hamza: 'above' }, pf: [0xFEF7, 0xFEF8] },
    { name: 'lamAlefHamzaBelow', alef: 'alefHamzaBelow', marks: { hamza: 'below' }, pf: [0xFEF9, 0xFEFA] },
    { name: 'lamAlefMadda', alef: 'alefMadda', marks: { madda: 1 }, pf: [0xFEF5, 0xFEF6] }
  ];

  R.extras = [
    { name: 'space', uni: 0x20, kind: 'space' },
    { name: 'nbsp', uni: 0xA0, kind: 'space' },
    { name: 'tatweel', uni: 0x640, kind: 'tatweel' },
    { name: 'period', uni: 0x2E, kind: 'punct', fn: 'period' },
    { name: 'colon', uni: 0x3A, kind: 'punct', fn: 'colon' },
    { name: 'exclam', uni: 0x21, kind: 'punct', fn: 'exclam' },
    { name: 'hyphen', uni: 0x2D, kind: 'punct', fn: 'hyphen' },
    { name: 'commaArabic', uni: 0x60C, kind: 'punct', fn: 'comma' },
    { name: 'semicolonArabic', uni: 0x61B, kind: 'punct', fn: 'semicolon' },
    { name: 'questionArabic', uni: 0x61F, kind: 'punct', fn: 'question' }
  ];
  for (let i = 0; i < 10; i++) R.extras.push({ name: 'digit' + i, uni: 0x660 + i, kind: 'digit', d: i, alt: 0x6F0 + (i === 4 || i === 5 || i === 6 ? -1 : i) });

  /* Unicode presentation forms (isol, fina, init, medi) for compatibility with apps that do no shaping */
  R.presentation = {
    hamza: [0xFE80], alefMadda: [0xFE81, 0xFE82], alefHamzaAbove: [0xFE83, 0xFE84], wawHamza: [0xFE85, 0xFE86],
    alefHamzaBelow: [0xFE87, 0xFE88], yehHamza: [0xFE89, 0xFE8A, 0xFE8B, 0xFE8C], alef: [0xFE8D, 0xFE8E],
    beh: [0xFE8F, 0xFE90, 0xFE91, 0xFE92], tehMarbuta: [0xFE93, 0xFE94], teh: [0xFE95, 0xFE96, 0xFE97, 0xFE98],
    theh: [0xFE99, 0xFE9A, 0xFE9B, 0xFE9C], jeem: [0xFE9D, 0xFE9E, 0xFE9F, 0xFEA0], hah: [0xFEA1, 0xFEA2, 0xFEA3, 0xFEA4],
    khah: [0xFEA5, 0xFEA6, 0xFEA7, 0xFEA8], dal: [0xFEA9, 0xFEAA], thal: [0xFEAB, 0xFEAC], reh: [0xFEAD, 0xFEAE],
    zain: [0xFEAF, 0xFEB0], seen: [0xFEB1, 0xFEB2, 0xFEB3, 0xFEB4], sheen: [0xFEB5, 0xFEB6, 0xFEB7, 0xFEB8],
    sad: [0xFEB9, 0xFEBA, 0xFEBB, 0xFEBC], dad: [0xFEBD, 0xFEBE, 0xFEBF, 0xFEC0], tah: [0xFEC1, 0xFEC2, 0xFEC3, 0xFEC4],
    zah: [0xFEC5, 0xFEC6, 0xFEC7, 0xFEC8], ain: [0xFEC9, 0xFECA, 0xFECB, 0xFECC], ghain: [0xFECD, 0xFECE, 0xFECF, 0xFED0],
    feh: [0xFED1, 0xFED2, 0xFED3, 0xFED4], qaf: [0xFED5, 0xFED6, 0xFED7, 0xFED8], kaf: [0xFED9, 0xFEDA, 0xFEDB, 0xFEDC],
    lam: [0xFEDD, 0xFEDE, 0xFEDF, 0xFEE0], meem: [0xFEE1, 0xFEE2, 0xFEE3, 0xFEE4], noon: [0xFEE5, 0xFEE6, 0xFEE7, 0xFEE8],
    heh: [0xFEE9, 0xFEEA, 0xFEEB, 0xFEEC], waw: [0xFEED, 0xFEEE], alefMaksura: [0xFEEF, 0xFEF0], yeh: [0xFEF1, 0xFEF2, 0xFEF3, 0xFEF4]
  };
  R.pfOrder = ['isol', 'fina', 'init', 'medi'];

  /* ---------------------------------------------------------------- build one glyph */
  function finalize(b, opts) {
    const m = b.m, f = b.form;
    const jr = joinsR(f) && b.entry, jl = joinsL(f) && b.exit;
    if (jr && jl && Math.abs(b.entry.u - b.exit.u) < 1) { b.entry.dir = 'flat'; b.exit.dir = 'flat'; }
    const ink = R.partBounds(b.parts, m.dna);
    if (!isFinite(ink.u0)) { ink.u0 = 0; ink.u1 = 0; }
    const uR = jr ? Math.min(b.entry.u - m.jl, ink.u0 - m.jl * 0.4) : ink.u0 - (opts.sb != null ? opts.sb : m.sb);
    const uL = jl ? Math.max(b.exit.u + m.jl, ink.u1 + m.jl * 0.4) : ink.u1 + (opts.sb != null ? opts.sb : m.sb);
    if (jr) b.conn(b.entry, uR - m.ov * 3);
    if (jl) b.conn(b.exit, uL + m.ov * 3);
    const adv = Math.round(uL - uR);
    const shift = -uR;
    let parts = b.parts.map(p => R.shiftPart(p, shift, 0));
    // slant: shear (tops lean to the left for positive values)
    const k = Math.tan((m.dna.slant || 0) * Math.PI / 180);
    if (k) parts = parts.map(p => p.type === 'image' ? Object.assign({}, p, { skew: k }) : Object.assign({}, p, { pts: p.pts.map(q => [q[0] + q[1] * k, q[1]]) }));
    const anc = {};
    for (const key in b.anc) anc[key] = [b.anc[key][0] + shift, b.anc[key][1]];
    return { adv, parts, clip: { right: jr ? -m.ov : null, left: jl ? adv + m.ov : null }, anc, form: f };
  }

  function applyMarks(b, marks) {
    if (marks.dots) b.dots(marks.dots[0], b.anc[marks.dots[1]] || b.anc.above, marks.dots[1] === 'below' ? 'below' : 'above');
    if (marks.hamza) {
      const at = b.anc[marks.hamza];
      if (at) b.mark('hamza', marks.hamza === 'below' ? at : [at[0], at[1]], marks.hamza);
    }
    if (marks.madda) b.mark('madda', b.anc.above, 'above');
  }

  R.buildLetter = function (m, rasm, form, marks, opts) {
    const b = new Builder(m, form);
    RX[rasm](b, m, form);
    if (marks) applyMarks(b, marks);
    return finalize(b, opts || {});
  };

  R.buildLigature = function (m, lig, form) {
    const b = new Builder(m, form);
    RX.lamalef(b, m, form);
    applyMarks(b, lig.marks);
    return finalize(b, {});
  };

  R.buildExtra = function (m, ex) {
    const b = new Builder(m, 'isol');
    if (ex.kind === 'space') return { adv: Math.round(m.A * 0.26 + m.W * 0.4), parts: [], clip: {}, anc: {} };
    if (ex.kind === 'tatweel') {
      const adv = Math.round(Math.max(m.jl * 2, m.W * 1.6));
      return { adv, parts: [{ type: 'stroke', pts: [[-m.ov * 3, m.b], [adv + m.ov * 3, m.b]], caps: ['butt', 'butt'], slot: 'conn' }], clip: { right: -m.ov, left: adv + m.ov }, anc: {} };
    }
    if (ex.kind === 'digit') {
      const d = digitBox(m);
      DG[ex.d](b, m, d);
      const r = finalize(b, { sb: 0 });
      // tabular: centre in a fixed box
      const ink = R.partBounds(b.parts, m.dna);
      const adv = Math.round(d.w + m.sb * 1.2);
      const inkW = isFinite(ink.u0) ? ink.u1 - ink.u0 : 0;
      const shift = (adv - inkW) / 2 - (isFinite(ink.u0) ? ink.u0 : 0);
      return { adv, parts: b.parts.map(p => R.shiftPart(p, shift, 0)), clip: {}, anc: {}, tabular: true, _r: r };
    }
    PU[ex.fn](b, m);
    return finalize(b, {});
  };

  /* full glyph inventory for the font */
  R.inventory = function () {
    const out = [];
    for (const l of R.letters) {
      for (const f of R.formsOf(l.join)) {
        out.push({ name: f === 'isol' ? l.name : l.name + '.' + f, kind: 'letter', letter: l, form: f, uni: f === 'isol' ? l.uni : null });
      }
    }
    for (const lg of R.ligatures) {
      out.push({ name: lg.name, kind: 'lig', lig: lg, form: 'isol' });
      out.push({ name: lg.name + '.fina', kind: 'lig', lig: lg, form: 'fina' });
    }
    for (const ex of R.extras) out.push({ name: ex.name, kind: 'extra', extra: ex, uni: ex.uni, form: 'isol' });
    return out;
  };

  R.buildSpec = function (m, spec) {
    if (spec.kind === 'letter') return R.buildLetter(m, spec.letter.rasm, spec.form, spec.letter.marks);
    if (spec.kind === 'lig') return R.buildLigature(m, spec.lig, spec.form);
    return R.buildExtra(m, spec.extra);
  };
})(typeof window !== 'undefined' ? window : globalThis);
