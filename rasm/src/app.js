/* رسم — application: the Illustrator workflow wired to the engine.
 * 01 template & upload · 02 pieces & their checks · 03 recipes · 04 build & tests · 05 tester · 06 logos · 07 export */
(function () {
  'use strict';
  const R = window.Rasm, IO = R.svg;
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
  const AR = (v) => String(v).replace(/[0-9]/g, (d) => '٠١٢٣٤٥٦٧٨٩'[d]);
  const DPR = () => Math.min(2, window.devicePixelRatio || 1);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const el = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };

  /* ---------------------------------------------------------------- state */
  const S = {
    dna: R.defaultDNA(), measured: {}, drawnDNA: {}, pp: {}, ppSrc: {},
    an: null, logo: null, src: {}, hl: null, ver: 0,
    font: null, fontVer: -1, fontFace: null, fontBuf: null, stat: 'idle',
    file: null, checks: {}, report: null, collisions: [],
    ov: { sk: true, lines: true, det: true }, crop: null, sel: 'beh.init',
    mkColor: 'ink', feat: { ctx: true, rlig: true, kash: true, rasm: true }
  };
  let M = R.metrics(S.dna, S.pp);
  const INV = R.inventory();
  const SPEC = {}; INV.forEach((s) => { SPEC[s.name] = s; });
  let cache = new Map(), rcache = new Map();
  function G(name) {
    let g = cache.get(name);
    if (g === undefined) {
      const sp = SPEC[name];
      try { g = sp ? R.buildSpec(M, sp) : null; } catch (e) { console.error(name, e); g = null; }
      cache.set(name, g);
    }
    return g;
  }
  function RG(key, form) {
    const k = key + '|' + form;
    let g = rcache.get(k);
    if (g === undefined) { try { g = R.buildLetter(M, key, form, null); } catch (e) { console.error(k, e); g = null; } rcache.set(k, g); }
    return g;
  }
  const css = () => getComputedStyle(document.documentElement);
  const T = (n) => css().getPropertyValue(n).trim() || '#191919';
  const META = (id) => R.slots.byId[id];

  /* ---------------------------------------------------------------- drawing helpers */
  function sizeCanvas(cv, fixedH) {
    const r = cv.getBoundingClientRect();
    const w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(fixedH || r.height));
    const k = DPR();
    if (cv.width !== Math.round(w * k) || cv.height !== Math.round(h * k)) { cv.width = Math.round(w * k); cv.height = Math.round(h * k); }
    const ctx = cv.getContext('2d');
    ctx.setTransform(k, 0, 0, k, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }
  const VSPAN = () => M.A + M.DD + 300;
  const baseAt = (h, s) => (h - VSPAN() * s) / 2 + (M.A + 170) * s;
  function drawCentered(ctx, g, w, h, s, colorOf, cx) {
    if (!g) return;
    const b = R.partBounds(g.parts, S.dna);
    const uc = isFinite(b.u0) ? (b.u0 + b.u1) / 2 : g.adv / 2;
    let ss = s;
    if (isFinite(b.u0) && (b.u1 - b.u0) * ss > w * 0.92) ss = w * 0.92 / (b.u1 - b.u0);
    const x = (cx != null ? cx : w / 2) - (g.adv - uc) * ss;
    R.drawGlyph(ctx, g, S.dna, { x, base: baseAt(h, ss), s: ss, colorOf });
  }
  const wordGlyphs = (text, kashida) => R.shapeText(text, kashida || 0).map((x) => G(x.name)).filter(Boolean);
  function drawRun(ctx, glyphs, box, color, align, sMax) {
    const total = glyphs.reduce((a, g) => a + g.adv, 0) || 1;
    let s = Math.min(box.h * 0.94 / VSPAN(), box.w * 0.96 / total);
    if (sMax) s = Math.min(s, sMax);
    const base = box.y + baseAt(box.h, s);
    let x = align === 'right' ? box.x + box.w : box.x + (box.w + total * s) / 2;
    const colorOf = typeof color === 'function' ? color : () => color;
    for (const g of glyphs) { x -= g.adv * s; R.drawGlyph(ctx, g, S.dna, { x, base, s, colorOf }); }
    return s;
  }

  /* ---------------------------------------------------------------- toast & downloads */
  let toastT = 0;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg; t.hidden = false;
    clearTimeout(toastT); toastT = setTimeout(() => { t.hidden = true; }, 3400);
  }
  let dlP = null;
  function downloadsNS() {
    if (!dlP) dlP = (window.claude && typeof window.claude.use === 'function') ? Promise.resolve(window.claude.use('downloads')).catch(() => null) : Promise.resolve(null);
    return dlP;
  }
  async function saveFile(filename, data, mime) {
    const dl = await downloadsNS();
    if (dl) {
      try {
        const r = await dl.save({ filename, data });
        toast(r && r.status === 'delivered' ? 'أُرسل الملف' : 'حُفظ الملف: ' + filename);
      } catch (e) {
        const c = e && e.code;
        if (c === 'declined') return;
        toast(c === 'rate_limited' ? 'نافذة حفظ مفتوحة بالفعل. أكملها ثم أعد المحاولة.' : c === 'rejected_extension' ? 'هذه الصيغة غير مسموحة هنا.' : 'تعذّر الحفظ في هذا العرض.');
      }
      return;
    }
    const blob = data instanceof Blob ? data : new Blob([data], { type: mime || 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = filename;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2500);
    toast('بدأ تحميل ' + filename);
  }
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(u8) { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
  function makeZip(files) {
    const enc = new TextEncoder(), d = new Date();
    const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
    const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    const parts = [], central = [];
    let off = 0;
    for (const f of files) {
      const name = enc.encode(f.name), data = f.data, crc = crc32(data);
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true);
      lh.setUint16(10, time, true); lh.setUint16(12, date, true); lh.setUint32(14, crc, true);
      lh.setUint32(18, data.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true);
      parts.push(new Uint8Array(lh.buffer), name, data);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true);
      ch.setUint16(12, time, true); ch.setUint16(14, date, true); ch.setUint32(16, crc, true); ch.setUint32(20, data.length, true); ch.setUint32(24, data.length, true);
      ch.setUint16(28, name.length, true); ch.setUint32(42, off, true);
      central.push(new Uint8Array(ch.buffer), name);
      off += 30 + name.length + data.length;
    }
    const size = central.reduce((a, c) => a + c.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true); end.setUint16(8, files.length, true); end.setUint16(10, files.length, true);
    end.setUint32(12, size, true); end.setUint32(16, off, true);
    return new Blob(parts.concat(central, [new Uint8Array(end.buffer)]), { type: 'application/zip' });
  }

  /* ---------------------------------------------------------------- font building */
  let buildTimer = 0, buildSeq = 0;
  const famName = () => (($('#famName').value || '').replace(/[^A-Za-z0-9 \-]/g, '').trim()) || 'Rasm';
  function setStat(st, p) {
    S.stat = st;
    const stat = $('#buildStat'), track = $('#buildTrack'), btn = $('#buildBtn'), note = $('#testerNote');
    if (st === 'building') {
      track.style.width = Math.round((p || 0) * 100) + '%';
      stat.textContent = 'BUILDING ' + Math.round((p || 0) * 100) + '%';
      btn.textContent = 'يُبنى…'; note.textContent = 'يُبنى الخط الآن…';
    } else if (st === 'ready' && S.font) {
      track.style.width = '100%';
      stat.textContent = S.font.font.glyphs.length + ' GLYPHS · ' + (S.fontBuf.byteLength / 1024).toFixed(1) + ' KB · fina init medi rlig';
      btn.textContent = 'أعد البناء';
      $('#glyphCount').textContent = AR(S.font.font.glyphs.length);
      note.textContent = 'الخط الحالي: ' + S.font.family + '، ' + AR(S.font.font.glyphs.length) + ' شكلاً، بُني على جهازك.';
    } else if (st === 'dirty') {
      stat.textContent = 'CHANGED · REBUILDING…'; btn.textContent = 'ابنِ الآن';
    } else if (st === 'nolib') {
      stat.textContent = 'opentype.js not loaded';
      note.textContent = 'تعذّر تحميل مكتبة كتابة الخطوط. المعاينة تعمل، أما ملف الخط فيحتاج اتصالاً.';
    } else if (st === 'error') {
      stat.textContent = 'ERROR · ' + (p || ''); btn.textContent = 'أعد المحاولة';
    }
  }
  function scheduleBuild(ms) { clearTimeout(buildTimer); setStat('dirty'); buildTimer = setTimeout(buildNow, ms == null ? 900 : ms); }
  async function buildNow() {
    clearTimeout(buildTimer);
    if (!window.opentype) { setStat('nolib'); return null; }
    const seq = ++buildSeq, ver = S.ver;
    const dna = Object.assign({}, S.dna), m = R.metrics(dna, JSON.parse(JSON.stringify(S.pp)));
    setStat('building', 0);
    try {
      const res = await R.font.build({ dna, m }, { family: famName() }, (p) => {
        if (seq !== buildSeq || ver !== S.ver) return false;
        setStat('building', p); return true;
      });
      if (seq !== buildSeq || ver !== S.ver) return null;
      const buf = res.font.toArrayBuffer();
      const fam = 'RasmLive' + seq;
      const ff = new FontFace(fam, buf.slice(0));
      await ff.load();
      if (seq !== buildSeq) return null;
      document.fonts.add(ff);
      if (S.fontFace) { try { document.fonts.delete(S.fontFace); } catch (e) { /* ignore */ } }
      Object.assign(S, { font: res, fontBuf: buf, fontFace: ff, fontVer: ver });
      document.documentElement.style.setProperty('--live', '"' + fam + '"');
      setStat('ready');
      setTimeout(() => runFontChecks(res, buf, fam, m), 30);
      return res;
    } catch (e) {
      if (e && e.cancelled) return null;
      console.error(e); setStat('error', e && e.message); return null;
    }
  }
  async function ensureFont() {
    if (S.font && S.fontVer === S.ver) return S.font;
    toast('يُبنى الخط…');
    return await buildNow();
  }

  /* ---------------------------------------------------------------- change propagation */
  let rq = 0, slowT = 0;
  function changed(now) {
    S.ver++;
    R.clampDNA(S.dna);
    M = R.metrics(S.dna, S.pp);
    cache = new Map(); rcache = new Map(); usageVer = -1;
    requestRender();
    scheduleBuild(now ? 40 : 900);
  }
  function requestRender() {
    if (rq) return;
    rq = requestAnimationFrame(() => { rq = 0; renderAll(); });
  }
  function renderAll() {
    renderHero(); renderAxesValues(); renderImport(); renderPieces(); renderMatrix(); renderPreview(); renderMaker(); renderFoot(); renderLogo(); renderFound();
    clearTimeout(slowT); slowT = setTimeout(() => { renderTiles(); renderThumb(); }, 160);
  }

  /* ---------------------------------------------------------------- hero */
  function renderHero() {
    const { ctx, w, h } = sizeCanvas($('#heroTitle'));
    drawRun(ctx, wordGlyphs('رسم', 0), { x: 0, y: 0, w, h }, T('--hero-ink'), 'right');
    const f = sizeCanvas($('#heroForms'));
    const gl = ['beh', 'beh.init', 'beh.medi', 'beh.fina'].map(G).filter(Boolean);
    const sum = gl.reduce((a, g) => a + g.adv, 0);
    const s = Math.min(f.h * 0.98 / VSPAN(), f.w / (sum + M.A * 0.9));
    const gap = (f.w - sum * s) / (gl.length + 1);
    let x = f.w - gap;
    const col = T('--hero-ink');
    for (const g of gl) { x -= g.adv * s; R.drawGlyph(f.ctx, g, S.dna, { x, base: baseAt(f.h, s), s, colorOf: () => col }); x -= gap; }
  }
  function renderFoot() {
    const { ctx, w, h } = sizeCanvas($('#footMark'));
    drawRun(ctx, wordGlyphs('رسم', 1), { x: 0, y: 0, w, h }, T('--lav-ink'), 'right');
  }

  /* ---------------------------------------------------------------- 01 template & upload */
  let thumbURL = null, thumbVer = -1;
  function renderThumb() {
    if (thumbVer === S.ver) return;
    thumbVer = S.ver;
    try {
      const svg = IO.template(M, R.slots.drawn);
      if (thumbURL) URL.revokeObjectURL(thumbURL);
      thumbURL = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
      $('#tplThumb').src = thumbURL;
    } catch (e) { console.error(e); }
  }
  async function downloadTemplate() {
    const svg = IO.template(M, R.slots.drawn);
    await saveFile('rasm-template.svg', svg, 'image/svg+xml');
  }

  function clearPP(src) {
    for (const id in S.ppSrc) for (const k in S.ppSrc[id]) if (S.ppSrc[id][k] === src) { delete S.ppSrc[id][k]; if (S.pp[id]) delete S.pp[id][k]; }
  }
  function setPP(id, k, v, src) {
    S.pp[id] = S.pp[id] || {}; S.ppSrc[id] = S.ppSrc[id] || {};
    S.pp[id][k] = v; S.ppSrc[id][k] = src;
  }

  function importSVG(text, name, isDemo) {
    let parsed;
    try { parsed = IO.parse(text); } catch (e) { toast(e.message || 'تعذّرت قراءة الملف'); return; }
    const { pieces, report } = IO.toPieces(parsed);
    const n = Object.keys(pieces).length;
    S.file = { name, report, count: n, demo: !!isDemo };
    if (!n) { renderImport(); toast('لم أجد أشكالاً سوداء داخل الإطارات.'); return; }
    // the file is the whole truth: it replaces every drawing
    for (const k in R.slots.drawn) delete R.slots.drawn[k];
    for (const k in R.slots.useDrawn) delete R.slots.useDrawn[k];
    Object.assign(R.slots.drawn, pieces);
    adoptDrawings();
    changed(true);
    runPieceChecks();
    if (!isDemo) toast('وصلت ' + AR(n) + ' قطعة. الفحص في القسم الثاني.');
  }
  // the pen and the metrics come from the drawings; each drawing's measurements become its piece's
  // parameters, so the pieces derived from it (and the ones the engine draws) inherit them
  function adoptDrawings() {
    const pieces = R.slots.drawn;
    for (const id in pieces) pieces[id].meas = IO.measure(pieces[id], S.dna.weight);
    S.drawnDNA = IO.dnaFrom(pieces);
    Object.assign(S.dna, S.drawnDNA); R.clampDNA(S.dna);
    for (const id in pieces) pieces[id].meas = IO.measure(pieces[id], S.dna.weight);
    clearPP('drawn');
    const fit = IO.fitAll(pieces, R.metrics(S.dna, {}));
    for (const id in fit.pp) for (const k in fit.pp[id]) if (!(S.ppSrc[id] && S.ppSrc[id][k] === 'user')) setPP(id, k, fit.pp[id][k], 'drawn');
    if (fit.roundness != null) S.dna.roundness = fit.roundness;
    S.measured = Object.assign({}, S.dna);
  }
  function runPieceChecks() {
    S.checks = {};
    const stem = R.slots.drawn.stem;
    const ctx = { W: M.W, D: M.D, DD: M.DD, stemW: stem && stem.meas ? stem.meas.width : null };
    for (const id in R.slots.drawn) {
      try { S.checks[id] = IO.check(id, R.slots.drawn[id], ctx); } catch (e) { S.checks[id] = [{ level: 'warn', msg: 'تعذّر الفحص: ' + e.message }]; }
    }
    requestRender();
  }
  function renderImport() {
    const box = $('#importSummary');
    const chip = $('#svgName');
    if (!S.file) { chip.textContent = 'لم يُرفع ملف'; chip.className = 'chip'; box.innerHTML = ''; return; }
    chip.textContent = S.file.demo ? 'ملف تجريبي رسمه المحرك' : S.file.name;
    chip.className = 'chip ' + (S.file.demo ? '' : 'drawn');
    const r = S.file.report, drawn = R.slots.drawn;
    const missing = IO.ORDER.filter((id) => !drawn[id]);
    const rows = [];
    rows.push(['ok', 'وصلت ' + AR(S.file.count) + ' قطعة من ١٥']);
    if (missing.length) rows.push(['info', 'يولّدها المحرك: ' + missing.map((id) => META(id).ar).join('، ')]);
    if (r.open) rows.push(['warn', AR(r.open) + ' مسار مفتوح لم يُقرأ. أغلقه أو حوّل الخط إلى شكل (Outline Stroke).']);
    if (r.outlineOnly) rows.push(['warn', AR(r.outlineOnly) + ' شكل بلا تعبئة قُرئ من خطه الخارجي. الأفضل أن تملأه بالأسود.']);
    if (r.outside) rows.push(['warn', AR(r.outside) + ' شكل خارج الإطارات تُرك.']);
    if (r.issues.includes('size')) rows.push(['error', 'حجم اللوحة تغيّر عن القالب، فقد تقع القطع في غير إطاراتها.']);
    if (r.issues.includes('image')) rows.push(['warn', 'في الملف صورة نقطية، والمحرك يقرأ الأشكال المتجهة فقط.']);
    if (r.issues.includes('clip')) rows.push(['warn', 'في الملف قناع قص يُتجاهل. حرّر الأشكال منه قبل التصدير.']);
    box.innerHTML = '';
    const ul = el('ul', 'checks');
    for (const [lv, t] of rows) { const li = el('li', lv); li.textContent = t; ul.appendChild(li); }
    box.appendChild(ul);
  }

  /* demo: a designer's export, simulated so the whole flow can be tried without Illustrator */
  function demoSVG() {
    const style = R.clampDNA(Object.assign(R.defaultDNA(), { weight: 96, roundness: 0.85, cap: 'round', dotShape: 'circle', xh: 450, loop: 410, desc: 260 }));
    const ms = R.metrics(style);
    const savedD = Object.assign({}, R.slots.drawn), savedO = Object.assign({}, R.slots.overrides);
    for (const k in R.slots.drawn) delete R.slots.drawn[k];
    for (const k in R.slots.overrides) delete R.slots.overrides[k];
    let body = '';
    try {
      IO.frames().forEach((fr) => {
        if (fr.id === 'hamza' || fr.id === 'madda') return;
        const piece = IO.suggestion(fr.id, ms);
        const contours = R.font.outline({ adv: piece.w, parts: piece.parts, clip: {} }, style, 0.6);
        const small = fr.id === 'dot';
        const gx = small ? fr.x0 + 750 - piece.w / 2 : fr.ref - piece.w, gy = small ? fr.base - 200 : fr.base;
        body += '<path fill="#000000" d="' + IO.contoursD(contours, (x) => gx + x, (y) => gy - y) + '"/>';
      });
    } finally {
      Object.assign(R.slots.drawn, savedD); Object.assign(R.slots.overrides, savedO);
    }
    return IO.template(ms, {}).replace('</svg>', '<g id="demo-drawing">' + body + '</g></svg>');
  }

  function initStart() {
    $('#tplBtn').addEventListener('click', downloadTemplate);
    $('#svgFile').addEventListener('change', async (e) => {
      const f = e.target.files[0]; e.target.value = '';
      if (f) importSVG(await f.text(), f.name, false);
    });
    const drop = $('#svgDrop');
    ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('drag'); }));
    drop.addEventListener('drop', async (e) => {
      const f = e.dataTransfer && e.dataTransfer.files[0];
      if (!f) return;
      if (!/svg/i.test(f.type) && !/\.svg$/i.test(f.name)) { toast('ارفع ملف SVG المصدَّر من Illustrator.'); return; }
      importSVG(await f.text(), f.name, false);
    });
    $('#demoBtn').addEventListener('click', () => { importSVG(demoSVG(), 'demo', true); toast('حُمّل ملف تجريبي'); });
    $('#clearDrawn').addEventListener('click', () => {
      for (const k in R.slots.drawn) delete R.slots.drawn[k];
      clearPP('drawn'); S.file = null; S.checks = {}; S.drawnDNA = {};
      changed(true); toast('مُسحت الرسوم. كل القطع يولّدها القلم الآن.');
    });
  }

  /* ---------------------------------------------------------------- 01 pen & grid */
  const AXES = [
    ['weight', 'الوزن', 'WGHT', 24, 200, 1], ['contrast', 'التباين', 'CNTR', 0, 0.85, 0.01], ['nibAngle', 'زاوية القلم', 'ANGL', 0, 180, 1],
    ['roundness', 'الاستدارة', 'ROND', 0, 1, 0.01], ['alif', 'الألف', 'ALIF', 400, 900, 1], ['xh', 'السنّ', 'XHGT', 200, 720, 1],
    ['loop', 'الرأس', 'LOOP', 150, 700, 1], ['desc', 'النزول', 'DESC', 120, 450, 1], ['width', 'العرض', 'WDTH', 0.6, 1.8, 0.01],
    ['spacing', 'التباعد', 'SPAC', 0.4, 2.5, 0.01], ['slant', 'الميل', 'SLNT', -20, 20, 1], ['dotSize', 'النقطة', 'DOTS', 0.8, 2.6, 0.01]
  ];
  const CHOICES = [
    ['cap', 'الأطراف', [['butt', 'مسطّحة'], ['round', 'مدوّرة'], ['square', 'مربّعة']]],
    ['dotShape', 'شكل النقطة', [['square', 'مربّعة'], ['circle', 'دائرية'], ['diamond', 'معيّن']]],
    ['nib', 'رأس القلم', [['round', 'مستدير'], ['flat', 'عريض']]]
  ];
  const fmt = (k, v) => (['contrast', 'roundness', 'width', 'spacing', 'dotSize'].includes(k)) ? (+v).toFixed(2) : String(Math.round(v));
  function buildAxes() {
    const box = $('#axes'); box.innerHTML = '';
    for (const [k, nm, tg, mn, mx, st] of AXES) {
      const row = el('div', 'axis'); row.dataset.k = k;
      row.innerHTML = '<span class="nm"></span><span class="tg mono"></span><input type="range"><output class="num"></output>';
      row.querySelector('.nm').textContent = nm; row.querySelector('.tg').textContent = tg;
      const inp = row.querySelector('input');
      Object.assign(inp, { min: mn, max: mx, step: st, id: 'ax-' + k });
      inp.setAttribute('aria-label', nm); inp.value = S.dna[k];
      inp.addEventListener('input', () => {
        S.dna[k] = +inp.value;
        if (k === 'desc') S.dna.deep = Math.round(Math.max(S.dna.desc + 40, S.dna.desc * 1.3));
        changed(false);
      });
      box.appendChild(row);
    }
    const cbox = $('#choices'); cbox.innerHTML = '';
    for (const [k, nm, opts] of CHOICES) {
      const row = el('div', 'choice');
      const seg = el('div', 'seg'); seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', nm); seg.dataset.k = k;
      for (const [v, t] of opts) { const b = el('button', '', t); b.type = 'button'; b.dataset.v = v; b.addEventListener('click', () => { S.dna[k] = v; changed(false); }); seg.appendChild(b); }
      row.append(el('span', 'nm', nm), seg);
      cbox.appendChild(row);
    }
    renderAxesValues();
  }
  function renderAxesValues() {
    for (const row of $$('#axes .axis')) {
      const k = row.dataset.k, v = S.dna[k], inp = row.querySelector('input');
      if (document.activeElement !== inp) inp.value = v;
      row.querySelector('output').textContent = fmt(k, v);
      const mv = S.measured[k];
      row.classList.toggle('measured', mv != null && Math.abs(mv - v) < 1e-6 + (Math.abs(mv) > 2 ? 0.5 : 0.005));
    }
    for (const seg of $$('#choices .seg')) for (const b of seg.children) b.setAttribute('aria-pressed', String(S.dna[seg.dataset.k] === b.dataset.v));
  }

  /* ---------------------------------------------------------------- 01 logo assistant (optional) */
  async function loadLogo(canvas, name) {
    try {
      const an = R.analyze.run(canvas);
      S.an = an; S.logo = canvas;
      Object.assign(S.dna, an.dna, S.drawnDNA);        // the drawings keep the last word on their own measures
      for (const id in R.slots.overrides) if (S.src[id] === 'logo') { delete R.slots.overrides[id]; delete S.src[id]; }
      const pieces = R.analyze.autoPieces(an);
      for (const id in pieces) { R.slots.overrides[id] = pieces[id]; S.src[id] = 'logo'; }
      $('#logoName').textContent = name;
      $('#logoFold').open = true;
      S.measured = Object.assign({}, S.dna);
      changed(true);
      toast('قيس الشعار: الأشكال الرمادية في القالب على مقاس قلمه الآن.');
    } catch (e) { console.error(e); toast('تعذّر قياس الصورة: ' + (e.message || e)); }
  }
  function renderLogo(rect) {
    const cv = $('#logoCanvas');
    $('#stage').hidden = !S.an; $('#overlaySeg').hidden = !S.an;
    if (!S.an || !$('#logoFold').open) return;
    const stage = $('#stage');
    const W = Math.max(10, Math.round(stage.getBoundingClientRect().width));
    const H = Math.round(W * S.an.h / S.an.w), k = DPR();
    cv.width = Math.round(W * k); cv.height = Math.round(H * k); cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(k, 0, 0, k, 0, 0); ctx.clearRect(0, 0, W, H);
    ctx.drawImage(S.logo, 0, 0, W, H);
    const s = W / S.an.w;
    R.analyze.overlay(S.an, ctx, s, { skeleton: S.ov.sk, lines: S.ov.lines, det: S.ov.det, skColor: T('--accent'), boxColor: T('--blue'), lineColor: T('--muted') });
    if (S.crop) {
      ctx.fillStyle = 'rgba(0,0,0,.28)'; ctx.fillRect(0, 0, W, H);
      if (rect) {
        const [x0, y0, x1, y1] = rect.map((v) => v * s);
        const rx = Math.min(x0, x1), ry = Math.min(y0, y1), rw = Math.abs(x1 - x0), rh = Math.abs(y1 - y0);
        ctx.clearRect(rx, ry, rw, rh);
        ctx.drawImage(S.logo, rx / s, ry / s, rw / s, rh / s, rx, ry, rw, rh);
        ctx.strokeStyle = T('--accent'); ctx.lineWidth = 2; ctx.setLineDash([6, 4]); ctx.strokeRect(rx, ry, rw, rh); ctx.setLineDash([]);
      }
    }
  }
  function renderFound() {
    const an = S.an, box = $('#found');
    if (!an) { box.innerHTML = ''; return; }
    const items = [['السطر', an.hasBaseline], ['الساق', S.src.stem === 'logo'], ['النقطة', an.dots.length > 0], ['الرأس', S.src.head === 'logo'], ['العروة', S.src.loop === 'logo'],
      ['التباين ' + AR(Math.round(an.contrast * 100)) + '٪', an.contrast > 0], ['الأطراف ' + (an.cap === 'round' ? 'مدوّرة' : 'مسطّحة'), true]];
    box.innerHTML = '';
    for (const [t, ok] of items) box.appendChild(el('span', ok ? 'ok' : '', (ok ? '✓ ' : '— ') + t));
  }
  function initLogo() {
    $('#logoFile').addEventListener('change', async (e) => {
      const f = e.target.files[0]; e.target.value = '';
      if (!f) return;
      try { loadLogo(await R.analyze.loadFile(f), f.name); } catch (err) { toast(err.message || 'تعذّرت قراءة الملف'); }
    });
    $('#logoFold').addEventListener('toggle', () => renderLogo());
    $$('#overlaySeg button').forEach((b) => b.addEventListener('click', () => { const k = b.dataset.k; S.ov[k] = !S.ov[k]; b.setAttribute('aria-pressed', String(S.ov[k])); renderLogo(); }));
    const cv = $('#logoCanvas');
    let start = null;
    const pt = (e) => { const r = cv.getBoundingClientRect(); return [clamp((e.clientX - r.left) / r.width, 0, 1) * S.an.w, clamp((e.clientY - r.top) / r.height, 0, 1) * S.an.h]; };
    cv.addEventListener('pointerdown', (e) => { if (!S.crop || !S.an) return; e.preventDefault(); cv.setPointerCapture(e.pointerId); start = pt(e); renderLogo([...start, ...start]); });
    cv.addEventListener('pointermove', (e) => { if (S.crop && start) renderLogo([...start, ...pt(e)]); });
    cv.addEventListener('pointerup', (e) => {
      if (!S.crop || !start) return;
      const cur = pt(e);
      const rect = [Math.min(start[0], cur[0]), Math.min(start[1], cur[1]), Math.max(start[0], cur[0]), Math.max(start[1], cur[1])];
      start = null;
      if (rect[2] - rect[0] < 4 || rect[3] - rect[1] < 4) { renderLogo(); return; }
      const id = S.crop, o = R.analyze.cropForSlot(S.an, rect, id);
      endCrop();
      if (!o) { toast('لا يوجد حبر داخل المستطيل.'); return; }
      R.slots.overrides[id] = o; S.src[id] = 'logo';
      const fit = R.slots.fitParams(id, o, M);
      for (const k in fit) if (!(S.ppSrc[id] && S.ppSrc[id][k] === 'drawn')) setPP(id, k, fit[k], 'logo');
      changed(true);
      toast('«' + META(id).ar + '» أُخذت من الشعار' + (R.slots.drawn[id] ? '، ورسمك ما زال مقدَّماً عليها' : ''));
      const card = $('[data-slot="' + id + '"]'); if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
    $('#cropCancel').addEventListener('click', endCrop);
  }
  function startCrop(id) {
    if (!S.an) { toast('افتح «مساعد الشعار» وارفع شعاراً أولاً.'); return; }
    S.crop = id; $('#logoFold').open = true;
    $('#cropText').textContent = 'ارسم مستطيلاً حول «' + META(id).ar + '» في الشعار';
    $('#cropBar').hidden = false; $('#stage').classList.add('cropping');
    renderLogo(); $('#logoFold').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function endCrop() { S.crop = null; $('#cropBar').hidden = true; $('#stage').classList.remove('cropping'); renderLogo(); }

  /* ---------------------------------------------------------------- 02 pieces */
  function canonPiece(id) {
    const m = M, p = {};
    if (id === 'stem') p.free = true;
    if (id === 'basin') p.full = true;
    if (id === 'cup') { p.top = m.t; p.topCap = m.cap; }
    if (id === 'ain') { p.botEnd = m.h; p.botCap = m.cap; }
    if (id === 'hamza') p.scale = 0.9;
    return R.slots.piece(id, m, p);
  }
  let usageVer = -1, usage = {};
  function getUsage() {
    if (usageVer === S.ver) return usage;
    usage = {};
    for (const spec of INV) {
      const g = G(spec.name); if (!g) continue;
      const seen = new Set();
      for (const p of g.parts) if (p.slot && !seen.has(p.slot)) { seen.add(p.slot); (usage[p.slot] = usage[p.slot] || []).push(spec.name); }
    }
    usageVer = S.ver;
    return usage;
  }
  function sampleOf(id, n) {
    const out = [], bases = new Set();
    for (const name of (getUsage()[id] || [])) {
      const base = name.split('.')[0];
      if (bases.has(base) || /^(digit|period|colon|comma|semicolon|question|exclam|hyphen|dotless|noonGhunna)/.test(name)) continue;
      bases.add(base); out.push(name);
      if (out.length >= n) break;
    }
    return out;
  }
  function statusOf(id) {
    const d = R.slots.drawn[id], meta = META(id);
    if (d && R.slots.useDrawn[id] !== false) {
      const items = S.checks[id] || [];
      if (items.some((x) => x.level === 'error')) return ['error', 'رسمك: خطأ'];
      if (items.some((x) => x.level === 'warn')) return ['warn', 'رسمك: تنبيه'];
      return ['drawn', 'رسمك ✓'];
    }
    if (d) return ['gen', 'رسمك موقوف'];
    if (R.slots.overrides[id] && S.src[id] === 'logo') return ['logo', 'من الشعار'];
    if (id === 'hamza' && R.slots.drawn.ain) return ['derived', 'من رأس العين'];
    if (meta.role === 'derived') return ['derived', 'مشتقة من ' + META(meta.parent).ar];
    return ['gen', 'يولّدها القلم'];
  }
  function buildTree() {
    const box = $('#tree'); box.innerHTML = '';
    const roots = R.slots.list.filter((s) => s.role === 'root');
    for (const r of roots) {
      const br = el('div', 'branch');
      const node = (s, root) => { const b = el('button', 'node' + (root ? ' root' : ''), s.ar); b.type = 'button'; b.dataset.slot = s.id; b.addEventListener('click', () => { const c = $('#slots [data-slot="' + s.id + '"]'); if (c) c.scrollIntoView({ behavior: 'smooth', block: 'center' }); }); return b; };
      br.appendChild(node(r, true));
      const kids = (R.slots.children[r.id] || []).map((k) => META(k));
      if (r.id === 'arc') kids.push(META('hamza'));
      if (kids.length) { br.appendChild(el('span', 'arrow', '←')); kids.forEach((k) => br.appendChild(node(k, false))); }
      box.appendChild(br);
    }
  }
  function buildPieces() {
    buildTree();
    const box = $('#slots'); box.innerHTML = '';
    for (const id of IO.ORDER) {
      const meta = META(id);
      const card = el('article', 'slot'); card.dataset.slot = id;
      card.innerHTML = '<div class="card-top" style="margin:0"><span class="label"><b></b><span class="mono"></span></span><span class="chip"></span></div>' +
        '<canvas class="piece-cv" role="img"></canvas><p class="lineage"></p><ul class="checks"></ul><canvas class="use-cv" role="img"></canvas><p class="usage mono"></p><div class="acts"></div>';
      card.querySelector('.label b').textContent = (meta.n ? AR(meta.n) + ' ' : '') + meta.ar;
      card.querySelector('.label .mono').textContent = meta.en;
      card.querySelector('.piece-cv').setAttribute('aria-label', 'القطعة: ' + meta.ar);
      card.querySelector('.use-cv').setAttribute('aria-label', 'حروف تستعمل ' + meta.ar);
      const acts = card.querySelector('.acts');
      const mk = (t, fn, key) => { const b = el('button', 'btn small', t); b.type = 'button'; b.dataset.act = key; b.addEventListener('click', fn); acts.appendChild(b); return b; };
      mk('أنزلها إلى السطر', () => snapPiece(id), 'snap');
      mk('استعمل المولّدة', () => { R.slots.useDrawn[id] = R.slots.useDrawn[id] === false; changed(true); }, 'toggle');
      mk('قصّ من الشعار', () => startCrop(id), 'crop');
      mk('أين تُستعمل', () => { setHL(id); $('#recipes').scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 'where');
      box.appendChild(card);
    }
  }
  function snapPiece(id) {
    const d = R.slots.drawn[id]; if (!d) return;
    const dy = -d.bbox.y0;
    const shapes = d.shapes.map((sh) => ({ rule: sh.rule, cmds: sh.cmds.map((c) => c[0] === 'Z' ? c : [c[0]].concat(c.slice(1).map((v, i) => i % 2 ? v + dy : v))) }));
    R.slots.drawn[id] = IO.makePiece(id, shapes, null, d.placeU);
    adoptDrawings();
    changed(true); runPieceChecks();
    toast('أُنزلت «' + META(id).ar + '» إلى السطر');
  }
  function renderPieces() {
    const ink = T('--ink'), accent = T('--accent'), rest = T('--rest'), line = T('--line'), band = T('--band');
    for (const card of $$('#slots .slot')) {
      const id = card.dataset.slot, meta = META(id);
      const d = R.slots.drawn[id], using = d && R.slots.useDrawn[id] !== false;
      const [cls, txt] = statusOf(id);
      const chip = card.querySelector('.chip'); chip.className = 'chip ' + cls; chip.textContent = txt;
      card.classList.toggle('is-hl', S.hl === id);
      // the piece
      const { ctx, w, h } = sizeCanvas(card.querySelector('.piece-cv'));
      let piece;
      try { piece = canonPiece(id); } catch (e) { console.error(e); continue; }
      const b = R.partBounds(piece.parts, S.dna);
      if (isFinite(b.u0)) {
        const small = !meta.role;
        const pw = Math.max(1, b.u1 - b.u0), ph = Math.max(1, b.y1 - b.y0);
        const s = Math.min(w * 0.8 / pw, h * 0.8 / ph, h / (M.A * (small ? 0.42 : 1.05)));
        const base = small ? h / 2 + ((b.y0 + b.y1) / 2) * s : h / 2 + ((b.y0 + b.y1) / 2) * s;
        if (!small) {
          ctx.fillStyle = band; ctx.fillRect(0, base - M.W * s, w, M.W * s);
          ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(0, Math.round(base) + 0.5); ctx.lineTo(w, Math.round(base) + 0.5); ctx.stroke();
        }
        const col = using ? ink : (R.slots.overrides[id] && S.src[id] === 'logo' ? accent : rest);
        R.drawGlyph(ctx, { adv: piece.w, parts: piece.parts, clip: {} }, S.dna, { x: w / 2 - (piece.w - (b.u0 + b.u1) / 2) * s, base, s, colorOf: () => col });
      }
      // lineage
      let lin = meta.role === 'root' ? 'جذر' : meta.role === 'derived' ? 'مشتقة من «' + META(meta.parent).ar + '»' : 'قطعة صغيرة';
      if (meta.role === 'derived' && !using && R.slots.drawn[meta.parent]) lin += '، وترث مقاسات رسمك لها';
      if (meta.role === 'root' && R.slots.children[id] && using) lin += '، ومنها: ' + R.slots.children[id].map((k) => META(k).ar).join('، ');
      card.querySelector('.lineage').textContent = lin + '. ' + meta.rule + '.';
      // checks
      const ul = card.querySelector('.checks'); ul.innerHTML = '';
      if (using) {
        const items = S.checks[id] || [];
        if (!items.length) ul.appendChild(el('li', 'ok', !meta.role ? 'سليمة: حجمها مناسب لسماكة القلم'
          : ['cup', 'arc', 'tail'].includes(id) ? 'سليمة: تعبر الشريط وتنزل تحت السطر، والسماكة متسقة'
          : 'سليمة: على السطر، داخل الشريط، والسماكة متسقة'));
        for (const it of items) ul.appendChild(el('li', it.level, it.msg));
      } else if (!d) {
        ul.appendChild(el('li', 'info', 'غير مرسومة. ارسمها في إطار «' + meta.ar + '» في القالب، أو اتركها للمحرك.'));
      }
      // letters using it
      const u = sizeCanvas(card.querySelector('.use-cv'));
      const names = sampleOf(id, 5), gl = names.map(G).filter(Boolean);
      const cell = u.w / 5;
      const sf = Math.min(u.h * 0.98 / VSPAN(), cell * 0.85 / Math.max(1, ...gl.map((g) => g.adv)));
      gl.forEach((g, i) => drawCentered(u.ctx, g, u.w, u.h, sf, (p) => p.slot === id ? accent : rest, u.w - cell * (i + 0.5)));
      const count = (getUsage()[id] || []).length;
      card.querySelector('.usage').textContent = count ? 'USED IN ' + count + ' GLYPHS' : '';
      // actions
      const show = (key, v) => { const bt = card.querySelector('[data-act="' + key + '"]'); if (bt) bt.hidden = !v; return bt; };
      show('snap', using && (S.checks[id] || []).some((x) => x.fix === 'snap'));
      const tg = show('toggle', !!d); if (tg) tg.textContent = using ? 'استعمل المولّدة' : 'استعمل رسمي';
      show('crop', !!S.an && !!meta.role);
      show('where', !!meta.role);
    }
  }

  /* ---------------------------------------------------------------- 03 recipes */
  const FORMS = ['isol', 'init', 'medi', 'fina'];
  const FORM_AR = { isol: 'منفصل', init: 'أوّل', medi: 'وسط', fina: 'آخر' };
  function formulaOf(built) {
    return R.recipeOf(built).map((it) => (META(it.id) ? META(it.id).short : it.id) + (it.n > 1 ? '×' + AR(it.n) : '')).join(' + ');
  }
  // the recipe as text with a joint mark on each side that connects (right = to the letter before)
  function setFormula(node, built, form) {
    node.textContent = '';
    if (!built) return;
    const jr = form === 'fina' || form === 'medi', jl = form === 'init' || form === 'medi';
    const mark = (side) => { const i = el('i', 'j'); i.title = side; i.setAttribute('aria-hidden', 'true'); return i; };
    if (jr) node.appendChild(mark('يتصل بما قبله'));
    node.appendChild(document.createTextNode(formulaOf(built)));
    if (jl) node.appendChild(mark('يتصل بما بعده'));
  }
  function buildMatrix() {
    const seg = $('#hlSeg'); seg.innerHTML = '';
    for (const [id, t] of [[null, 'الكل']].concat(R.slots.list.map((s) => [s.id, s.ar]))) {
      const b = el('button', '', t); b.type = 'button'; b.dataset.id = id || '';
      b.addEventListener('click', () => setHL(id)); seg.appendChild(b);
    }
    const box = $('#matrix'); box.innerHTML = '';
    const head = el('div', 'mx-row mx-head'); head.setAttribute('role', 'row');
    head.appendChild(el('span', '', ''));
    FORMS.forEach((f) => head.appendChild(el('span', 'mono', FORM_AR[f])));
    box.appendChild(head);
    for (const [key, ch] of R.rasm18) {
      const l = R.letters.find((x) => x.rasm === key);
      const fam = R.letters.filter((x) => x.rasm === key && x.uni < 0x660).map((x) => String.fromCodePoint(x.uni)).join(' ');
      const row = el('div', 'mx-row'); row.dataset.key = key; row.setAttribute('role', 'row');
      const nm = el('div', 'mx-name'); nm.append(el('b', '', ch), el('span', '', fam)); row.appendChild(nm);
      const forms = R.formsOf(l.join);
      for (const f of FORMS) {
        const c = el('div', 'mx-cell'); c.dataset.form = f; c.setAttribute('role', 'cell');
        if (forms.includes(f)) { const cv = el('canvas'); cv.setAttribute('aria-label', ch + ' ' + FORM_AR[f]); c.append(cv, el('p', 'mx-f')); }
        else { c.classList.add('none'); c.appendChild(el('p', 'mx-f', 'لا يتصل بما بعده')); }
        row.appendChild(c);
      }
      box.appendChild(row);
    }
  }
  function setHL(id) { S.hl = id || null; renderMatrix(); renderPieces(); renderPreview(); }
  function renderMatrix() {
    for (const b of $$('#hlSeg button')) b.setAttribute('aria-pressed', String((b.dataset.id || null) === S.hl));
    const accent = T('--accent'), rest = T('--rest'), ink = T('--ink');
    const colorOf = (p) => !S.hl ? ink : (p.slot === S.hl ? accent : rest);
    let total = 0, uses = 0;
    for (const row of $$('#matrix .mx-row[data-key]')) {
      const key = row.dataset.key;
      for (const c of $$('.mx-cell', row)) {
        if (c.classList.contains('none')) continue;
        const f = c.dataset.form, g = RG(key, f);
        total++;
        const cv = c.querySelector('canvas');
        const { ctx, w, h } = sizeCanvas(cv);
        drawCentered(ctx, g, w, h, h * 0.98 / VSPAN(), colorOf);
        const items = g ? R.recipeOf(g) : [];
        const has = !!S.hl && items.some((it) => it.id === S.hl);
        if (has) uses++;
        c.classList.toggle('uses', has);
        setFormula(c.querySelector('.mx-f'), g, f);
      }
    }
    $('#coverage').textContent = S.hl ? '«' + META(S.hl).ar + '» في ' + AR(uses) + ' وصفة من ' + AR(total) + '.'
      : AR(total) + ' وصفة، كلها من القطع الاثنتي عشرة والوصلة. الشَّرطة على جانب الوصفة تعني أن الحرف يتصل من تلك الجهة.';
  }

  /* ---------------------------------------------------------------- 04 tests */
  function contoursToPolys(contours) {
    const polys = [];
    for (const c of contours) {
      const P = [c.start.slice()];
      let cur = c.start;
      for (const sg of c.segs) {
        if (sg.c1) R.geom.cubic(cur, sg.c1, sg.c2, sg.p, P, 8); else P.push(sg.p.slice());
        cur = sg.p;
      }
      polys.push(P);
    }
    return polys;
  }
  function joinCheck(res, m) {
    const bad = [];
    const W = m.W;
    for (const spec of INV) {
      const name = spec.name, f = spec.form;
      const jr = f === 'fina' || f === 'medi', jl = f === 'init' || f === 'medi';
      if (!jr && !jl && name !== 'tatweel') continue;
      const built = res.built[name], polys = contoursToPolys(res.outlines[name] || []);
      if (!built || !polys.length) continue;
      const covered = (x) => R.geom.spanAt(polys, x, true).some((s) => s[0] <= W * 0.2 && s[1] >= W * 0.8);
      if ((jr || name === 'tatweel') && !covered(built.adv - 3)) bad.push(name);
      else if ((jl || name === 'tatweel') && !covered(3)) bad.push(name);
    }
    return bad;
  }
  function shapingCheck(fam, res) {
    const c = document.createElement('canvas').getContext('2d');
    c.font = '100px "' + fam + '"';
    const words = ['بسم', 'سلام', 'محمد', 'لا', 'عين', 'كتب'];
    const off = [];
    for (const wd of words) {
      const expect = R.shapeText(wd, 0).reduce((a, g) => a + (res.built[g.name] ? res.built[g.name].adv : 0), 0) / 10;
      const got = c.measureText(wd).width;
      if (Math.abs(got - expect) > Math.max(1.5, expect * 0.01)) off.push(wd);
    }
    return off.length ? ['warn', 'المتصفح شكّل ' + off.join('، ') + ' بعرض غير المتوقع'] : ['ok', 'التشكيل في المتصفح: الكلمات المرجعية تطابق الوصفات'];
  }
  function collisionCheck(res, m) {
    const s = 0.06, top = m.A + 420, bottom = -(m.DD + 420), PAD = Math.round(m.A * 0.8 * s);
    const H = Math.ceil((top - bottom) * s) + 2;
    const masks = {};
    const maskOf = (name) => {
      if (masks[name]) return masks[name];
      const g = res.built[name]; if (!g) return null;
      const w = Math.max(2, Math.ceil(g.adv * s) + PAD * 2);
      const cv = document.createElement('canvas'); cv.width = w; cv.height = H;
      const ctx = cv.getContext('2d', { willReadFrequently: true });
      R.drawGlyph(ctx, g, m.dna, { x: PAD, base: top * s + 1, s, colorOf: () => '#000' });
      const d = ctx.getImageData(0, 0, w, H).data, a = new Uint8Array(w * H);
      for (let i = 0, j = 3; i < a.length; i++, j += 4) a[i] = d[j] > 110 ? 1 : 0;
      return (masks[name] = { a, w, adv: g.adv * s });
    };
    const bandTop = Math.floor((top - m.W * 1.2) * s), bandBot = Math.ceil((top + m.W * 0.2) * s) + 2;
    const base = R.letters.filter((l) => l.uni < 0x660);
    const hits = [];
    let pairs = 0;
    for (const a of base) for (const b of base) {
      const txt = String.fromCodePoint(a.uni) + String.fromCodePoint(b.uni);
      const seq = R.shapeText(txt, 0);
      if (seq.length !== 2) continue;
      const L = maskOf(seq[1].name), Rm = maskOf(seq[0].name);   // visual: second glyph on the left
      if (!L || !Rm) continue;
      pairs++;
      const shift = Math.round(L.adv);                             // right glyph's origin = left glyph's origin + its advance
      const joined = /\.(init|medi)$/.test(seq[0].name) && /\.(fina|medi)$/.test(seq[1].name);
      let n = 0;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < Rm.w; x++) {
          const lx = x + shift;
          if (lx >= L.w) break;
          if (!Rm.a[y * Rm.w + x] || !L.a[y * L.w + lx]) continue;
          if (joined && Math.abs(x - PAD) <= 2 && y >= bandTop && y <= bandBot) continue;   // the joint itself
          n++;
        }
      }
      if (n > 2) hits.push({ txt, n });
    }
    hits.sort((p, q) => q.n - p.n);
    return { list: hits, pairs };
  }
  async function runFontChecks(res, buf, fam, m) {
    const items = [];
    try {
      const empty = INV.filter((sp) => !(sp.kind === 'extra' && sp.extra.kind === 'space') && !(res.outlines[sp.name] || []).length).map((sp) => sp.name);
      items.push(empty.length ? ['error', 'أشكال بلا رسم: ' + empty.slice(0, 6).join('، ')] : ['ok', 'كل الأشكال مرسومة: ' + AR(INV.length) + ' شكلاً']);
      const parsed = opentype.parse(buf.slice(0));
      const missing = R.letters.filter((l) => !parsed.charToGlyphIndex(String.fromCodePoint(l.uni))).map((l) => String.fromCodePoint(l.uni));
      items.push(missing.length ? ['error', 'حروف بلا خريطة يونيكود: ' + missing.join(' ')] : ['ok', 'خريطة يونيكود: ' + AR(R.letters.length) + ' حرفاً، وأشكال العرض القديمة']);
      const feats = parsed.tables.gsub ? Array.from(new Set(parsed.tables.gsub.features.map((f) => f.tag))) : [];
      const lack = ['fina', 'init', 'medi', 'rlig'].filter((f) => !feats.includes(f));
      items.push(lack.length ? ['error', 'ميزات ناقصة: ' + lack.join(', ')] : ['ok', 'البرمجة: المواضع الأربعة ولام ألف (fina · init · medi · rlig)']);
      const bad = joinCheck(res, m);
      items.push(bad.length ? ['warn', 'وصل غير مكتمل عند الحافة: ' + bad.slice(0, 6).join('، ')] : ['ok', 'الوصل: كل شكل متصل يغطي الشريط عند حافته']);
      items.push(shapingCheck(fam, res));
      const hits = collisionCheck(res, m);
      S.collisions = hits.list;
      items.push(hits.list.length ? ['warn', 'تداخل في ' + AR(hits.list.length) + ' زوجاً من ' + AR(hits.pairs)] : ['ok', 'التداخل: لا يتداخل أي حرفين متجاورين (' + AR(hits.pairs) + ' زوجاً)']);
    } catch (e) { console.error(e); items.push(['warn', 'تعذّر إكمال الفحص: ' + e.message]); }
    S.report = items;
    renderReport();
  }
  function renderReport() {
    const box = $('#report'); box.innerHTML = '';
    if (!S.report) return;
    const ul = el('ul', 'checks big');
    for (const [lv, t] of S.report) ul.appendChild(el('li', lv, t));
    box.appendChild(ul);
    const cbox = $('#collide'); cbox.innerHTML = '';
    if (!S.collisions.length) return;
    cbox.appendChild(el('p', 'hint', 'أكثر الأزواج تداخلاً:'));
    const row = el('div', 'pairs');
    for (const h of S.collisions.slice(0, 8)) {
      const fig = el('figure', 'pair');
      const cv = el('canvas'); fig.appendChild(cv); fig.appendChild(el('figcaption', '', h.txt));
      row.appendChild(fig);
      requestAnimationFrame(() => { const { ctx, w, h: hh } = sizeCanvas(cv); drawRun(ctx, wordGlyphs(h.txt, 0), { x: 0, y: 0, w, h: hh }, T('--ink'), 'center'); });
    }
    cbox.appendChild(row);
  }

  /* glyph set */
  function charOf(spec) {
    if (spec.kind === 'letter') return String.fromCodePoint(spec.letter.uni);
    if (spec.kind === 'lig') return 'لا';
    if (spec.kind === 'extra') return spec.extra.uni === 0x20 ? '␣' : spec.extra.uni === 0xA0 ? '⍽' : String.fromCodePoint(spec.extra.uni);
    return '';
  }
  function buildTiles() {
    const box = $('#tiles'); box.innerHTML = '';
    for (const spec of INV) {
      if (spec.kind === 'extra' && spec.extra.kind === 'space') continue;
      const b = el('button', 'tile'); b.type = 'button'; b.dataset.name = spec.name; b.setAttribute('role', 'listitem');
      b.setAttribute('aria-label', charOf(spec) + ' ' + (FORM_AR[spec.form] || '') + '، ' + spec.name);
      b.appendChild(el('canvas'));
      b.addEventListener('click', () => { S.sel = spec.name; renderPreview(); for (const t of $$('.tile')) t.setAttribute('aria-pressed', String(t.dataset.name === S.sel)); });
      box.appendChild(b);
    }
  }
  function renderTiles() {
    const ink = T('--ink');
    for (const t of $$('.tile')) {
      const { ctx, w, h } = sizeCanvas(t.querySelector('canvas'));
      drawCentered(ctx, G(t.dataset.name), w, h, h * 0.92 / (VSPAN() + 200), () => ink);
      t.setAttribute('aria-pressed', String(t.dataset.name === S.sel));
    }
  }
  function renderPreview() {
    const spec = SPEC[S.sel] || SPEC.beh, g = G(spec.name);
    $('#pvTitle').textContent = charOf(spec);
    $('#pvName').textContent = spec.name;
    $('#pvForm').textContent = FORM_AR[spec.form] || spec.form;
    const { ctx, w, h } = sizeCanvas($('#pvCanvas'));
    if (!g) return;
    const s = Math.min(h * 0.9 / (VSPAN() + 120), w * 0.7 / Math.max(g.adv, 1));
    const base = baseAt(h, s), x = (w - g.adv * s) / 2;
    const line = T('--line'), muted = T('--muted');
    ctx.font = '10.5px "IBM Plex Mono", monospace'; ctx.direction = 'ltr'; ctx.textAlign = 'left';
    ctx.fillStyle = muted; ctx.strokeStyle = line; ctx.lineWidth = 1;
    const Hl = (y, label) => { const Y = Math.round(base - y * s) + 0.5; ctx.beginPath(); ctx.moveTo(0, Y); ctx.lineTo(w, Y); ctx.stroke(); ctx.fillText(label, 6, Y - 4); };
    Hl(0, 'BASE 0'); Hl(M.X, 'TOOTH ' + Math.round(M.X)); Hl(M.A, 'ALIF ' + Math.round(M.A)); Hl(-M.D, 'DESC −' + Math.round(M.D));
    ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, h); ctx.moveTo(Math.round(x + g.adv * s) + 0.5, 0); ctx.lineTo(Math.round(x + g.adv * s) + 0.5, h);
    ctx.stroke(); ctx.setLineDash([]);
    const ink = T('--ink'), accent = T('--accent');
    R.drawGlyph(ctx, g, S.dna, { x, base, s, colorOf: (p) => (S.hl && p.slot === S.hl) ? accent : ink });
    const pf = (() => {
      if (spec.kind === 'letter') { const arr = R.presentation[spec.letter.name]; return arr ? arr[R.pfOrder.indexOf(spec.form)] : null; }
      if (spec.kind === 'lig') return spec.form === 'fina' ? spec.lig.pf[1] : spec.lig.pf[0];
      return null;
    })();
    const uni = spec.kind === 'letter' ? spec.letter.uni : spec.kind === 'extra' ? spec.extra.uni : null;
    const hex = (v) => 'U+' + v.toString(16).toUpperCase().padStart(4, '0');
    const rows = [['العرض', String(g.adv)], ['يونيكود', uni != null ? hex(uni) + (pf ? ' · ' + hex(pf) : '') : (pf ? hex(pf) : '—')]];
    const info = $('#pvInfo'); info.innerHTML = '';
    for (const [k, v] of rows) info.append(el('dt', '', k), el('dd', 'mono', v));
    const fd = el('dd', 'formula'); setFormula(fd, g, spec.form);
    if (!fd.textContent) fd.textContent = '—';
    info.append(el('dt', '', 'الوصفة'), fd);
  }

  /* OpenType feature cards */
  const RASM_MAP = { 'ب': 'ٮ', 'ت': 'ٮ', 'ث': 'ٮ', 'ن': 'ں', 'ي': 'ى', 'ئ': 'ى', 'ج': 'ح', 'خ': 'ح', 'ذ': 'د', 'ز': 'ر', 'ش': 'س', 'ض': 'ص', 'ظ': 'ط', 'غ': 'ع', 'ف': 'ڡ', 'ق': 'ٯ', 'ة': 'ه', 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و' };
  const FEATS = [
    { id: 'ctx', ar: 'الأشكال السياقية', tag: 'init · medi · fina', on: 'بيت سعيد', ff: '"init" 0, "medi" 0, "fina" 0' },
    { id: 'rlig', ar: 'لام ألف', tag: 'rlig', on: 'سلام لا', ff: '"rlig" 0' },
    { id: 'kash', ar: 'الكشيدة', tag: 'U+0640', on: 'مـحـمـد', off: 'محمد' },
    { id: 'rasm', ar: 'الهيكل بلا نقاط', tag: 'U+066E · U+06A1', on: 'تشبيه' },
    { id: 'digits', ar: 'الأرقام والعلامات', tag: 'U+0660–0669', on: '١٤٤٧ ، ؟', fixed: true },
    { id: 'pf', ar: 'أشكال العرض', tag: 'U+FE70–FEFC', on: 'ﺭﺳﻢ', fixed: true }
  ];
  function buildFeats() {
    const box = $('#feats'); box.innerHTML = '';
    for (const f of FEATS) {
      const card = el('div', 'feat'); card.dataset.id = f.id;
      card.innerHTML = '<div class="card-top" style="margin:0"><span class="label"><b></b><span class="mono" style="text-transform:none"></span></span></div><div class="sample" dir="rtl"></div>';
      card.querySelector('b').textContent = f.ar; card.querySelector('.mono').textContent = f.tag;
      if (!f.fixed) {
        const sw = el('button', 'switch'); sw.type = 'button'; sw.setAttribute('role', 'switch'); sw.setAttribute('aria-checked', 'true'); sw.setAttribute('aria-label', f.ar);
        sw.addEventListener('click', () => { S.feat[f.id] = !S.feat[f.id]; sw.setAttribute('aria-checked', String(S.feat[f.id])); renderFeats(); });
        card.querySelector('.card-top').appendChild(sw);
      }
      box.appendChild(card);
    }
    renderFeats();
  }
  function renderFeats() {
    for (const f of FEATS) {
      const smp = $('.feat[data-id="' + f.id + '"] .sample');
      const on = f.fixed ? true : S.feat[f.id];
      let txt = f.on;
      if (!on && f.off) txt = f.off;
      if (f.id === 'rasm' && !on) txt = Array.from(f.on).map((c) => RASM_MAP[c] || c).join('');
      smp.textContent = txt;
      smp.style.fontFeatureSettings = (!on && f.ff) ? f.ff : 'normal';
    }
  }

  /* ---------------------------------------------------------------- 05 tester, 06 maker */
  function initTester() {
    const r = $('#sizeRange'), out = $('#sizeOut'), t = $('#testerText');
    const apply = () => { t.style.fontSize = r.value + 'px'; out.textContent = r.value; };
    r.addEventListener('input', apply); apply();
    for (const b of $$('#presets button')) b.addEventListener('click', () => { t.textContent = b.dataset.t; });
  }
  const MK = { ink: () => [T('--ink'), T('--field')], accent: () => [T('--accent'), T('--field')], hero: () => ['#191919', '#ffcf96'], night: () => ['#f3efe9', '#191919'] };
  function renderMaker() {
    const cv = $('#mkCanvas'), r = cv.getBoundingClientRect();
    const h = Math.round(Math.max(160, r.width * 0.46));
    const { ctx, w } = sizeCanvas(cv, h);
    cv.style.height = h + 'px';
    const [fg, bg] = MK[S.mkColor]();
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
    drawRun(ctx, wordGlyphs($('#mkWord').value.trim() || 'رسم', +$('#mkKash').value), { x: w * 0.06, y: h * 0.06, w: w * 0.88, h: h * 0.88 }, fg, 'center');
  }
  function initMaker() {
    $('#mkWord').addEventListener('input', renderMaker);
    $('#mkKash').addEventListener('input', () => { $('#mkKashOut').textContent = $('#mkKash').value; renderMaker(); });
    for (const b of $$('#mkColor button')) b.addEventListener('click', () => { S.mkColor = b.dataset.c; for (const x of $$('#mkColor button')) x.setAttribute('aria-pressed', String(x === b)); renderMaker(); });
    $('#mkSvg').addEventListener('click', async () => {
      const res = await ensureFont(); if (!res) return;
      const [fg, bg] = MK[S.mkColor]();
      const svg = R.textToSVG(res, $('#mkWord').value.trim() || 'رسم', { size: 400, kashida: +$('#mkKash').value, color: fg, bg: S.mkColor === 'hero' || S.mkColor === 'night' ? bg : null });
      await saveFile(famName() + '-logo.svg', svg, 'image/svg+xml');
    });
    $('#mkPng').addEventListener('click', async () => {
      const gl = wordGlyphs($('#mkWord').value.trim() || 'رسم', +$('#mkKash').value);
      const total = gl.reduce((a, g) => a + g.adv, 0), s = 0.9, pad = 120;
      const c = document.createElement('canvas');
      c.width = Math.round(total * s + pad * 2); c.height = Math.round(VSPAN() * s + pad * 2);
      const ctx = c.getContext('2d');
      const [fg, bg] = MK[S.mkColor]();
      if (S.mkColor === 'hero' || S.mkColor === 'night') { ctx.fillStyle = bg; ctx.fillRect(0, 0, c.width, c.height); }
      drawRun(ctx, gl, { x: pad, y: pad, w: c.width - pad * 2, h: c.height - pad * 2 }, fg, 'center', s);
      const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
      if (blob) await saveFile(famName() + '-logo.png', blob, 'image/png');
    });
  }

  /* ---------------------------------------------------------------- 07 export */
  async function downloadFont() {
    const res = await ensureFont();
    if (!res || !S.fontBuf) return;
    const fam = res.family.replace(/\s+/g, '');
    const otf = new Uint8Array(S.fontBuf.slice(0));
    const dl = await downloadsNS();
    if (dl) {
      const readme = 'Rasm — ' + res.family + '\r\n\r\nهذا خط OpenType ‏(.otf) بُني على جهازك بمحرك «رسم».\r\nالتثبيت: افتح الملف ' + fam + '-Regular.otf ثم اختر «تثبيت».\r\n\r\nInstall: open ' + fam + '-Regular.otf and choose Install.\r\nGlyphs: ' + res.font.glyphs.length + ' · Features: init, medi, fina, rlig (script arab).\r\n';
      await saveFile(fam + '-font.zip', makeZip([{ name: fam + '-Regular.otf', data: otf }, { name: 'README.txt', data: new TextEncoder().encode(readme) }]), 'application/zip');
    } else await saveFile(fam + '-Regular.otf', new Blob([otf], { type: 'font/otf' }), 'font/otf');
  }
  async function saveProject() {
    const drawn = {};
    for (const id in R.slots.drawn) drawn[id] = { shapes: R.slots.drawn[id].shapes, placeU: R.slots.drawn[id].placeU };
    const slots = {};
    for (const id in R.slots.overrides) { const o = R.slots.overrides[id]; try { slots[id] = { png: o.img.toDataURL('image/png'), wu: o.wu, hu: o.hu, y0: o.y0, source: S.src[id] || 'upload' }; } catch (e) { /* skip */ } }
    const data = JSON.stringify({ app: 'rasm', v: 2, family: famName(), dna: S.dna, measured: S.measured, drawnDNA: S.drawnDNA, pp: S.pp, ppSrc: S.ppSrc, useDrawn: R.slots.useDrawn, drawn, slots });
    await saveFile(famName() + '-rasm.json', data, 'application/json');
  }
  async function loadProject(file) {
    try {
      const j = JSON.parse(await file.text());
      if (!j || j.app !== 'rasm') throw new Error('هذا ليس ملف مشروع من «رسم».');
      S.dna = R.clampDNA(Object.assign(R.defaultDNA(), j.dna || {}));
      S.measured = j.measured || {}; S.drawnDNA = j.drawnDNA || {}; S.pp = j.pp || {}; S.ppSrc = j.ppSrc || {};
      if (j.family) $('#famName').value = j.family;
      for (const k in R.slots.drawn) delete R.slots.drawn[k];
      for (const k in R.slots.useDrawn) delete R.slots.useDrawn[k];
      Object.assign(R.slots.useDrawn, j.useDrawn || {});
      for (const id in (j.drawn || {})) { const d = IO.makePiece(id, j.drawn[id].shapes, null, j.drawn[id].placeU); d.meas = IO.measure(d, S.dna.weight); R.slots.drawn[id] = d; }
      for (const id in R.slots.overrides) delete R.slots.overrides[id];
      S.src = {};
      await Promise.all(Object.entries(j.slots || {}).map(([id, o]) => new Promise((res) => {
        const img = new Image();
        img.onload = () => { const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight; c.getContext('2d').drawImage(img, 0, 0); R.slots.overrides[id] = { img: c, wu: o.wu, hu: o.hu, y0: o.y0, source: o.source }; S.src[id] = o.source === 'logo' ? 'logo' : 'upload'; res(); };
        img.onerror = () => res(); img.src = o.png;
      })));
      S.file = Object.keys(R.slots.drawn).length ? { name: file.name, report: { open: 0, outlineOnly: 0, outside: 0, issues: [] }, count: Object.keys(R.slots.drawn).length } : null;
      buildAxes(); changed(true); runPieceChecks();
      toast('فُتح المشروع');
    } catch (e) { toast(e.message || 'تعذّر فتح الملف'); }
  }
  function initExport() {
    $('#nameForm').addEventListener('submit', (e) => { e.preventDefault(); downloadFont(); });
    $('#navDownload').addEventListener('click', downloadFont);
    $('#famName').addEventListener('change', () => scheduleBuild(300));
    $('#saveProject').addEventListener('click', saveProject);
    $('#loadProject').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) loadProject(f); });
    $('#buildBtn').addEventListener('click', () => buildNow());
    $('#resetDNA').addEventListener('click', () => { Object.assign(S.dna, S.measured); changed(true); });
    downloadsNS().then((dl) => {
      $('#exportNote').textContent = dl ? 'اسم الخط بالحروف اللاتينية. في هذه الصفحة يصلك الخط داخل ملف ZIP، وفيه ملف الخط بصيغة OTF.' : 'اسم الخط بالحروف اللاتينية. يُحفظ الخط مباشرة بصيغة OTF.';
    });
  }

  /* ---------------------------------------------------------------- theme & boot */
  function initTheme() {
    $('#themeBtn').addEventListener('click', () => {
      const root = document.documentElement;
      const cur = root.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      root.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
    });
    try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { thumbVer = -1; requestRender(); }); } catch (e) { /* old browsers */ }
    new MutationObserver(() => { requestRender(); clearTimeout(slowT); slowT = setTimeout(renderTiles, 60); renderReport(); }).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
  function boot() {
    $('#pipelineCount').textContent = '١٥ ← ٦٤ ← ' + AR(INV.length + 1);
    $('#glyphCount').textContent = AR(INV.length + 1);
    buildAxes(); buildPieces(); buildMatrix(); buildTiles(); buildFeats();
    initStart(); initLogo(); initTester(); initMaker(); initExport(); initTheme();
    let rt = 0;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(requestRender, 120); });
    if (!window.opentype) setStat('nolib');
    importSVG(demoSVG(), 'demo', true);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => requestRender());
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
