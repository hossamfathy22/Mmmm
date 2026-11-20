/* رسم — application: wires the engine to the page. */
(function () {
  'use strict';
  const R = window.Rasm;
  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
  const AR = (v) => String(v).replace(/[0-9]/g, (d) => '٠١٢٣٤٥٦٧٨٩'[d]);
  const DPR = () => Math.min(2, window.devicePixelRatio || 1);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  /* ---------------------------------------------------------------- state */
  const S = {
    dna: R.defaultDNA(), measured: {}, an: null, logo: null, logoName: '', isSample: true,
    src: {}, hl: null, ver: 0, font: null, fontVer: -1, fontFace: null, fontBuf: null,
    ov: { sk: true, lines: true, det: true }, crop: null, uploadSlot: null, sel: 'beh',
    mkColor: 'ink', feat: { ctx: true, rlig: true, kash: true, rasm: true }, stat: 'idle'
  };
  let M = R.metrics(S.dna);
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
  const VSPAN = () => M.A + M.DD + 300;            // vertical span of a glyph box in units
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
  function wordGlyphs(text, kashida) {
    return R.shapeText(text, kashida || 0).map((x) => G(x.name)).filter(Boolean);
  }
  /* draw a run right-to-left; align 'right' | 'center' */
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

  /* ---------------------------------------------------------------- toast */
  let toastT = 0;
  function toast(msg) {
    const el = $('#toast');
    el.textContent = msg; el.hidden = false;
    clearTimeout(toastT); toastT = setTimeout(() => { el.hidden = true; }, 3200);
  }

  /* ---------------------------------------------------------------- downloads */
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

  // minimal ZIP (stored) — the page's save surface accepts .zip but not .otf
  const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
  function crc32(u8) { let c = 0xFFFFFFFF; for (let i = 0; i < u8.length; i++) c = CRC[(c ^ u8[i]) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; }
  function makeZip(files) {
    const enc = new TextEncoder();
    const d = new Date();
    const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
    const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
    const parts = [], central = [];
    let off = 0;
    for (const f of files) {
      const name = enc.encode(f.name), data = f.data, crc = crc32(data);
      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true); lh.setUint16(4, 20, true); lh.setUint16(6, 0x0800, true); lh.setUint16(8, 0, true);
      lh.setUint16(10, time, true); lh.setUint16(12, date, true); lh.setUint32(14, crc, true);
      lh.setUint32(18, data.length, true); lh.setUint32(22, data.length, true); lh.setUint16(26, name.length, true); lh.setUint16(28, 0, true);
      parts.push(new Uint8Array(lh.buffer), name, data);
      const ch = new DataView(new ArrayBuffer(46));
      ch.setUint32(0, 0x02014b50, true); ch.setUint16(4, 20, true); ch.setUint16(6, 20, true); ch.setUint16(8, 0x0800, true); ch.setUint16(10, 0, true);
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
  function famName() {
    const v = ($('#famName').value || '').replace(/[^A-Za-z0-9 \-]/g, '').trim();
    return v || 'Rasm';
  }
  function setStat(st, p) {
    S.stat = st;
    const stat = $('#buildStat'), track = $('#buildTrack'), btn = $('#buildBtn'), note = $('#testerNote');
    if (st === 'building') {
      track.style.width = Math.round((p || 0) * 100) + '%';
      stat.textContent = 'BUILDING ' + Math.round((p || 0) * 100) + '%';
      btn.textContent = 'يُبنى…';
      note.textContent = 'يُبنى الخط الآن…';
    } else if (st === 'ready' && S.font) {
      const kb = (S.fontBuf.byteLength / 1024).toFixed(1);
      track.style.width = '100%';
      stat.textContent = S.font.font.glyphs.length + ' GLYPHS · ' + kb + ' KB · fina init medi rlig';
      btn.textContent = 'أعد البناء';
      $('#glyphCount').textContent = AR(S.font.font.glyphs.length);
      note.textContent = 'الخط الحالي: ' + S.font.family + ' — ' + AR(S.font.font.glyphs.length) + ' شكلاً، بُني على جهازك.';
    } else if (st === 'dirty') {
      stat.textContent = 'CHANGED · REBUILDING…';
      btn.textContent = 'ابنِ الآن';
    } else if (st === 'nolib') {
      stat.textContent = 'opentype.js not loaded';
      note.textContent = 'تعذّر تحميل مكتبة كتابة الخطوط. المعاينة تعمل، أما ملف الخط فيحتاج اتصالاً.';
    } else if (st === 'error') {
      stat.textContent = 'ERROR · ' + (p || '');
      btn.textContent = 'أعد المحاولة';
    }
  }
  function scheduleBuild(ms) { clearTimeout(buildTimer); setStat('dirty'); buildTimer = setTimeout(buildNow, ms == null ? 900 : ms); }
  async function buildNow() {
    clearTimeout(buildTimer);
    if (!window.opentype) { setStat('nolib'); return null; }
    const seq = ++buildSeq, ver = S.ver;
    const dna = Object.assign({}, S.dna), m = R.metrics(dna);
    setStat('building', 0);
    try {
      const res = await R.font.build({ dna, m }, { family: famName() }, (p) => {
        if (seq !== buildSeq || ver !== S.ver) return false;
        setStat('building', p);
        return true;
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
      return res;
    } catch (e) {
      if (e && e.cancelled) return null;
      console.error(e);
      setStat('error', e && e.message);
      return null;
    }
  }
  async function ensureFont() {
    if (S.font && S.fontVer === S.ver) return S.font;
    toast('يُبنى الخط…');
    return await buildNow();
  }

  /* ---------------------------------------------------------------- change propagation */
  let rq = 0, tilesT = 0;
  function changed(now) {
    S.ver++;
    R.clampDNA(S.dna);
    M = R.metrics(S.dna);
    cache = new Map(); rcache = new Map();
    requestRender();
    scheduleBuild(now ? 40 : 900);
  }
  function requestRender() {
    if (rq) return;
    rq = requestAnimationFrame(() => { rq = 0; renderAll(); });
  }
  function renderAll() {
    renderHero(); renderAxesValues(); renderSlots(); renderRasm(); renderPreview(); renderMaker(); renderFoot(); renderLogo(); renderFound();
    clearTimeout(tilesT); tilesT = setTimeout(renderTiles, 140);
  }

  /* ---------------------------------------------------------------- hero */
  function renderHero() {
    const cv = $('#heroTitle');
    const { ctx, w, h } = sizeCanvas(cv);
    drawRun(ctx, wordGlyphs('رسم', 0), { x: 0, y: 0, w, h }, T('--hero-ink'), 'right');
    const fv = $('#heroForms');
    const f = sizeCanvas(fv);
    const names = ['beh', 'beh.init', 'beh.medi', 'beh.fina'];
    const gl = names.map(G).filter(Boolean);
    const s = Math.min(f.h * 0.98 / VSPAN(), f.w / (gl.reduce((a, g) => a + g.adv, 0) + M.A * 0.9));
    const gap = (f.w - gl.reduce((a, g) => a + g.adv * s, 0)) / (gl.length + 1);
    let x = f.w - gap;
    const col = T('--hero-ink');
    for (const g of gl) { x -= g.adv * s; R.drawGlyph(f.ctx, g, S.dna, { x, base: baseAt(f.h, s), s, colorOf: () => col }); x -= gap; }
  }
  function renderFoot() {
    const cv = $('#footMark');
    const { ctx, w, h } = sizeCanvas(cv);
    drawRun(ctx, wordGlyphs('رسم', 1), { x: 0, y: 0, w, h }, T('--lav-ink'), 'right');
  }

  /* ---------------------------------------------------------------- 01 logo */
  async function loadLogo(canvas, name, isSample) {
    try {
      const an = R.analyze.run(canvas);
      S.an = an; S.logo = canvas; S.logoName = name; S.isSample = !!isSample;
      // measured style DNA
      S.measured = Object.assign({}, an.dna);
      S.dna = R.clampDNA(Object.assign(R.defaultDNA(), an.dna));
      // pieces found in the logo
      for (const id in R.slots.overrides) if (S.src[id] === 'logo') { delete R.slots.overrides[id]; delete S.src[id]; }
      const pieces = R.analyze.autoPieces(an);
      for (const id in pieces) { if (S.src[id] === 'upload') continue; R.slots.overrides[id] = pieces[id]; S.src[id] = 'logo'; }
      $('#logoName').textContent = isSample ? 'شعار تجريبي — رسمه المحرك' : name;
      if (!isSample) {
        const base = (name || '').replace(/\.[a-z0-9]+$/i, '').replace(/[^A-Za-z0-9 \-]/g, '').trim();
        if (base) $('#famName').value = base.slice(0, 28);
      }
      buildAxes();
      changed(true);
      if (!isSample) toast('قيس الشعار: ' + Object.keys(pieces).length + ' قطع أُخذت منه');
    } catch (e) {
      console.error(e);
      toast('تعذّر قياس الصورة: ' + (e.message || e));
    }
  }
  function renderLogo(rect) {
    const cv = $('#logoCanvas');
    if (!S.an) return;
    const stage = $('#stage');
    const W = Math.max(10, Math.round(stage.getBoundingClientRect().width));
    const H = Math.round(W * S.an.h / S.an.w);
    const k = DPR();
    cv.width = Math.round(W * k); cv.height = Math.round(H * k);
    cv.style.height = H + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(k, 0, 0, k, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(S.logo, 0, 0, W, H);
    const s = W / S.an.w;
    R.analyze.overlay(S.an, ctx, s, { skeleton: S.ov.sk, lines: S.ov.lines, det: S.ov.det, skColor: T('--accent'), boxColor: T('--blue'), lineColor: T('--muted') });
    if (S.crop) {
      ctx.fillStyle = 'rgba(0,0,0,.28)';
      ctx.fillRect(0, 0, W, H);
      if (rect) {
        const [x0, y0, x1, y1] = rect.map((v) => v * s);
        ctx.clearRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
        ctx.drawImage(S.logo, Math.min(x0, x1) / s, Math.min(y0, y1) / s, Math.abs(x1 - x0) / s, Math.abs(y1 - y0) / s, Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
        ctx.strokeStyle = T('--accent'); ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
        ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
        ctx.setLineDash([]);
      }
    }
  }
  function renderFound() {
    const an = S.an, el = $('#found');
    if (!an) { el.innerHTML = ''; return; }
    const items = [
      ['السطر', an.hasBaseline],
      ['الساق', !!R.slots.overrides.stem && S.src.stem === 'logo'],
      ['النقطة' + (an.dots.length ? ' (' + ({ square: 'مربّعة', circle: 'دائرية', diamond: 'معيّن' }[an.dna.dotShape]) + ')' : ''), an.dots.length > 0],
      ['الرأس', S.src.head === 'logo'],
      ['العروة', S.src.loop === 'logo'],
      ['التباين ' + AR(Math.round(an.contrast * 100)) + '٪', an.contrast > 0],
      ['الأطراف ' + (an.cap === 'round' ? 'مدوّرة' : 'مسطّحة'), true]
    ];
    el.innerHTML = '';
    for (const [t, ok] of items) { const s = document.createElement('span'); s.className = ok ? 'ok' : ''; s.textContent = (ok ? '✓ ' : '— ') + t; el.appendChild(s); }
  }

  // drag & drop / file input
  async function onLogoFile(file) {
    if (!file) return;
    try { const c = await R.analyze.loadFile(file); await loadLogo(c, file.name, false); }
    catch (e) { toast(e.message || 'تعذّرت قراءة الملف'); }
  }
  function initLogoUI() {
    $('#logoFile').addEventListener('change', (e) => { onLogoFile(e.target.files[0]); e.target.value = ''; });
    $('#sampleBtn').addEventListener('click', () => loadSample());
    const stage = $('#stage');
    ['dragenter', 'dragover'].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add('drag'); }));
    ['dragleave', 'drop'].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove('drag'); }));
    stage.addEventListener('drop', (e) => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) onLogoFile(f); });
    $$('#overlaySeg button').forEach((b) => b.addEventListener('click', () => {
      const k = b.dataset.k; S.ov[k] = !S.ov[k]; b.setAttribute('aria-pressed', String(S.ov[k])); renderLogo();
    }));
    // crop interaction
    const cv = $('#logoCanvas');
    let start = null, cur = null;
    const pt = (e) => { const r = cv.getBoundingClientRect(); return [clamp((e.clientX - r.left) / r.width, 0, 1) * S.an.w, clamp((e.clientY - r.top) / r.height, 0, 1) * S.an.h]; };
    cv.addEventListener('pointerdown', (e) => { if (!S.crop || !S.an) return; e.preventDefault(); cv.setPointerCapture(e.pointerId); start = pt(e); cur = start; renderLogo([...start, ...cur]); });
    cv.addEventListener('pointermove', (e) => { if (!S.crop || !start) return; cur = pt(e); renderLogo([...start, ...cur]); });
    cv.addEventListener('pointerup', (e) => {
      if (!S.crop || !start) return;
      cur = pt(e);
      const rect = [Math.min(start[0], cur[0]), Math.min(start[1], cur[1]), Math.max(start[0], cur[0]), Math.max(start[1], cur[1])];
      start = null;
      if (rect[2] - rect[0] < 4 || rect[3] - rect[1] < 4) { renderLogo(); return; }
      const id = S.crop;
      const o = R.analyze.cropForSlot(S.an, rect, id);
      endCrop();
      if (!o) { toast('لا يوجد حبر داخل المستطيل.'); return; }
      R.slots.overrides[id] = o; S.src[id] = 'logo';
      changed(true);
      toast('«' + R.slots.byId[id].ar + '» صار من الشعار');
      const card = document.querySelector('[data-slot="' + id + '"]');
      if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
    $('#cropCancel').addEventListener('click', endCrop);
  }
  function startCrop(id) {
    if (!S.an) { toast('ارفع شعاراً أولاً.'); return; }
    S.crop = id;
    $('#cropText').textContent = 'ارسم مستطيلاً حول «' + R.slots.byId[id].ar + '» في الشعار';
    $('#cropBar').hidden = false;
    $('#stage').classList.add('cropping');
    renderLogo();
    $('#logo').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function endCrop() { S.crop = null; $('#cropBar').hidden = true; $('#stage').classList.remove('cropping'); renderLogo(); }

  function loadSample() {
    const c = R.analyze.sampleLogo('نور فلك');
    loadLogo(c, 'sample', true);
  }

  /* ---------------------------------------------------------------- 01 axes */
  const AXES = [
    ['weight', 'الوزن', 'WGHT', 24, 200, 1],
    ['contrast', 'التباين', 'CNTR', 0, 0.85, 0.01],
    ['nibAngle', 'زاوية القلم', 'ANGL', 0, 180, 1],
    ['roundness', 'الاستدارة', 'ROND', 0, 1, 0.01],
    ['alif', 'الألف', 'ALIF', 500, 900, 1],
    ['xh', 'السنّ', 'XHGT', 240, 720, 1],
    ['loop', 'الرأس', 'LOOP', 150, 700, 1],
    ['desc', 'النزول', 'DESC', 120, 450, 1],
    ['width', 'العرض', 'WDTH', 0.6, 1.8, 0.01],
    ['spacing', 'التباعد', 'SPAC', 0.4, 2.5, 0.01],
    ['slant', 'الميل', 'SLNT', -20, 20, 1],
    ['dotSize', 'النقطة', 'DOTS', 0.8, 2.6, 0.01]
  ];
  const CHOICES = [
    ['cap', 'الأطراف', [['butt', 'مسطّحة'], ['round', 'مدوّرة'], ['square', 'مربّعة']]],
    ['dotShape', 'شكل النقطة', [['square', 'مربّعة'], ['circle', 'دائرية'], ['diamond', 'معيّن']]],
    ['toothStyle', 'السنّ', [['stub', 'قائم'], ['peak', 'مدبّب'], ['bump', 'مقوّس']]],
    ['nib', 'رأس القلم', [['round', 'مستدير'], ['flat', 'عريض']]]
  ];
  const fmt = (k, v) => (k === 'contrast' || k === 'roundness' || k === 'width' || k === 'spacing' || k === 'dotSize') ? (+v).toFixed(2) : String(Math.round(v));
  function buildAxes() {
    const box = $('#axes');
    box.innerHTML = '';
    for (const [k, nm, tg, mn, mx, st] of AXES) {
      const row = document.createElement('div');
      row.className = 'axis'; row.dataset.k = k;
      row.innerHTML = '<span class="nm"></span><span class="tg mono"></span><input type="range"><output class="num"></output>';
      row.querySelector('.nm').textContent = nm;
      row.querySelector('.tg').textContent = tg;
      const inp = row.querySelector('input');
      Object.assign(inp, { min: mn, max: mx, step: st, id: 'ax-' + k });
      inp.setAttribute('aria-label', nm);
      inp.value = S.dna[k];
      inp.addEventListener('input', () => {
        S.dna[k] = +inp.value;
        if (k === 'desc') S.dna.deep = Math.round(Math.max(S.dna.desc + 40, S.dna.desc * 1.3));
        changed(false);
      });
      box.appendChild(row);
    }
    const cbox = $('#choices');
    cbox.innerHTML = '';
    for (const [k, nm, opts] of CHOICES) {
      const row = document.createElement('div');
      row.className = 'choice';
      const lab = document.createElement('span'); lab.className = 'nm'; lab.textContent = nm;
      const seg = document.createElement('div'); seg.className = 'seg'; seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', nm);
      for (const [v, t] of opts) {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.dataset.v = v;
        b.addEventListener('click', () => { S.dna[k] = v; changed(false); });
        seg.appendChild(b);
      }
      seg.dataset.k = k;
      row.append(lab, seg);
      cbox.appendChild(row);
    }
    renderAxesValues();
  }
  function renderAxesValues() {
    for (const row of $$('#axes .axis')) {
      const k = row.dataset.k, v = S.dna[k];
      const inp = row.querySelector('input');
      if (document.activeElement !== inp) inp.value = v;
      row.querySelector('output').textContent = fmt(k, v);
      const mv = S.measured[k];
      row.classList.toggle('measured', mv != null && Math.abs(mv - v) < 1e-6 + (Math.abs(mv) > 2 ? 0.5 : 0.005));
    }
    for (const seg of $$('#choices .seg')) {
      const k = seg.dataset.k;
      for (const b of seg.children) b.setAttribute('aria-pressed', String(S.dna[k] === b.dataset.v));
    }
  }

  /* ---------------------------------------------------------------- 02 slots */
  function canonPiece(id) {
    const m = M;
    if (id === 'conn') return { w: m.W * 5, parts: [{ type: 'stroke', pts: [[0, m.b], [m.W * 5, m.b]], caps: ['butt', 'butt'], slot: 'conn' }] };
    const p = {};
    if (id === 'stem') p.free = true;
    if (id === 'basin') p.full = true;
    if (id === 'cup') { p.top = m.t; p.topCap = m.cap; }
    if (id === 'ain') { p.botEnd = m.h; p.botCap = m.cap; }
    if (id === 'hamza') p.scale = 0.9;
    return R.slots.piece(id, m, p);
  }
  function statusOf(id) {
    if (id === 'conn') return ['gen', 'من القلم'];
    if (id === 'hamza' && !R.slots.overrides.hamza && R.slots.overrides.ain) return ['logo', 'من رأس العين'];
    const s = S.src[id];
    if (R.slots.overrides[id] && s === 'logo') return ['logo', 'من الشعار'];
    if (R.slots.overrides[id] && s === 'upload') return ['upload', 'مرفوع'];
    return ['gen', 'مولّد'];
  }
  function slotCard(meta, small) {
    const el = document.createElement('article');
    el.className = 'slot'; el.dataset.slot = meta.id;
    el.innerHTML = '<div class="card-top" style="margin:0"><span class="label"><b></b><span class="mono"></span></span><span class="chip"></span></div>' +
      '<canvas role="img"></canvas><div class="meta"><span class="letters"></span></div><p class="rule"></p><div class="acts"></div>';
    el.querySelector('.label b').textContent = (meta.n ? AR(meta.n) + ' ' : '') + meta.ar;
    el.querySelector('.label .mono').textContent = meta.en;
    el.querySelector('canvas').setAttribute('aria-label', 'الشكل: ' + meta.ar);
    el.querySelector('.letters').textContent = meta.letters || '';
    el.querySelector('.rule').textContent = meta.rule;
    const acts = el.querySelector('.acts');
    if (meta.id !== 'conn') {
      const mk = (t, fn, cls) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'btn small' + (cls ? ' ' + cls : ''); b.textContent = t; b.addEventListener('click', fn); acts.appendChild(b); return b; };
      mk('قصّ من الشعار', () => startCrop(meta.id));
      mk('رفع صورة', () => { S.uploadSlot = meta.id; $('#slotFile').click(); });
      const rb = mk('إعادة التوليد', () => { delete R.slots.overrides[meta.id]; delete S.src[meta.id]; changed(true); });
      rb.dataset.reset = '1';
    }
    if (!small) el.querySelector('canvas').addEventListener('click', () => { setHL(meta.id); $('#rasm').scrollIntoView({ behavior: 'smooth', block: 'start' }); });
    return el;
  }
  function buildSlots() {
    const a = $('#slots'), b = $('#smallSlots');
    a.innerHTML = ''; b.innerHTML = '';
    for (const meta of R.slots.list) a.appendChild(slotCard(meta, false));
    for (const meta of R.slots.small) b.appendChild(slotCard(meta, true));
    $('#slotFile').addEventListener('change', async (e) => {
      const f = e.target.files[0]; e.target.value = '';
      const id = S.uploadSlot; if (!f || !id) return;
      try {
        const c = await R.analyze.loadFile(f);
        const crop = R.analyze.maskFromCanvas(c);
        if (!crop) { toast('لا يوجد حبر في الصورة.'); return; }
        R.slots.overrides[id] = R.slots.fitUpload(id, crop.canvas, M);
        S.src[id] = 'upload';
        changed(true);
        toast('رُفع «' + R.slots.byId[id].ar + '»');
      } catch (err) { toast(err.message || 'تعذّرت قراءة الصورة'); }
    });
  }
  function renderSlots() {
    const accent = T('--accent'), ink = T('--ink');
    for (const el of $$('.slot')) {
      const id = el.dataset.slot;
      const small = el.parentElement.id === 'smallSlots';
      const cv = el.querySelector('canvas');
      const { ctx, w, h } = sizeCanvas(cv);
      // baseline
      let piece;
      try { piece = canonPiece(id); } catch (e) { console.error(e); continue; }
      const g = { adv: piece.w, parts: piece.parts, clip: {} };
      const b = R.partBounds(piece.parts, S.dna);
      if (!isFinite(b.u0)) continue;
      const pw = Math.max(1, b.u1 - b.u0), ph = Math.max(1, b.y1 - b.y0);
      // as large as the card allows, but never larger than a stem at ~60% of the card
      const s = Math.min(w * 0.8 / pw, h * 0.8 / ph, h / (M.A * (small ? 0.42 : 1.05)));
      const base = h / 2 + ((b.y0 + b.y1) / 2) * s;
      ctx.strokeStyle = T('--line'); ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(0, Math.round(base) + 0.5); ctx.lineTo(w, Math.round(base) + 0.5); ctx.stroke(); ctx.setLineDash([]);
      const uc = (b.u0 + b.u1) / 2;
      R.drawGlyph(ctx, g, S.dna, { x: w / 2 - (piece.w - uc) * s, base, s, colorOf: (p) => (p.slot === id || p.slot === 'conn' && id === 'conn') ? accent : ink });
      const [cls, txt] = statusOf(id);
      const chip = el.querySelector('.chip'); chip.className = 'chip ' + cls; chip.textContent = txt;
      const rb = el.querySelector('[data-reset]'); if (rb) rb.disabled = !R.slots.overrides[id];
      el.classList.toggle('is-hl', S.hl === id);
    }
  }

  /* ---------------------------------------------------------------- 03 rasm */
  function buildRasm() {
    const seg = $('#hlSeg');
    seg.innerHTML = '';
    const opts = [[null, 'الكل']].concat(R.slots.list.map((s) => [s.id, s.ar]));
    for (const [id, t] of opts) {
      const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.dataset.id = id || '';
      b.addEventListener('click', () => setHL(id));
      seg.appendChild(b);
    }
    const grid = $('#rasmGrid');
    grid.innerHTML = '';
    for (const [key, ch] of R.rasm18) {
      const l = R.letters.find((x) => x.rasm === key);
      const fam = R.letters.filter((x) => x.rasm === key && x.uni < 0x660).map((x) => String.fromCodePoint(x.uni)).join(' ');
      const el = document.createElement('div');
      el.className = 'rcard'; el.dataset.key = key; el.dataset.join = l.join;
      el.innerHTML = '<canvas class="big" role="img"></canvas><canvas class="forms" role="img"></canvas><div class="cap"><b></b><span></span></div>';
      el.querySelector('.cap b').textContent = ch;
      el.querySelector('.cap span').textContent = fam;
      el.querySelector('canvas.big').setAttribute('aria-label', 'الهيكل ' + ch);
      el.querySelector('canvas.forms').setAttribute('aria-label', 'مواضع ' + ch);
      grid.appendChild(el);
    }
  }
  function setHL(id) {
    S.hl = id || null;
    for (const b of $$('#hlSeg button')) b.setAttribute('aria-pressed', String((b.dataset.id || null) === S.hl));
    renderRasm(); renderSlots();
  }
  function renderRasm() {
    const accent = T('--accent'), rest = T('--rest'), ink = T('--ink');
    const colorOf = (p) => !S.hl ? ink : (p.slot === S.hl ? accent : rest);
    for (const el of $$('.rcard')) {
      const key = el.dataset.key, join = el.dataset.join;
      const big = sizeCanvas(el.querySelector('canvas.big'));
      const s = big.h * 0.98 / VSPAN();
      drawCentered(big.ctx, RG(key, 'isol'), big.w, big.h, s, colorOf);
      const fv = sizeCanvas(el.querySelector('canvas.forms'));
      const forms = R.formsOf(join);
      const gl = forms.map((f) => RG(key, f)).filter(Boolean);
      const cell = fv.w / 4;
      const sf = Math.min(fv.h * 0.98 / VSPAN(), cell * 0.8 / Math.max(...gl.map((g) => g.adv), 1));
      gl.forEach((g, i) => drawCentered(fv.ctx, g, fv.w, fv.h, sf, colorOf, fv.w - cell * (i + 0.5)));
    }
    for (const b of $$('#hlSeg button')) b.setAttribute('aria-pressed', String((b.dataset.id || null) === S.hl));
  }

  /* ---------------------------------------------------------------- 04 glyph set */
  const FORM_AR = { isol: 'منفصل', init: 'أوّل', medi: 'وسط', fina: 'آخر' };
  function charOf(spec) {
    if (spec.kind === 'letter') return String.fromCodePoint(spec.letter.uni);
    if (spec.kind === 'lig') return 'لا';
    if (spec.kind === 'extra') return spec.extra.uni === 0x20 ? '␣' : spec.extra.uni === 0xA0 ? '⍽' : String.fromCodePoint(spec.extra.uni);
    return '';
  }
  function buildTiles() {
    const box = $('#tiles');
    box.innerHTML = '';
    for (const spec of INV) {
      if (spec.kind === 'extra' && (spec.extra.kind === 'space')) continue;
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'tile'; b.dataset.name = spec.name;
      b.setAttribute('role', 'listitem');
      b.setAttribute('aria-label', charOf(spec) + ' ' + (FORM_AR[spec.form] || '') + ' — ' + spec.name);
      b.appendChild(document.createElement('canvas'));
      b.addEventListener('click', () => { S.sel = spec.name; renderPreview(); for (const t of $$('.tile')) t.setAttribute('aria-pressed', String(t.dataset.name === S.sel)); });
      box.appendChild(b);
    }
  }
  function renderTiles() {
    const ink = T('--ink');
    for (const t of $$('.tile')) {
      const { ctx, w, h } = sizeCanvas(t.querySelector('canvas'));
      const g = G(t.dataset.name);
      drawCentered(ctx, g, w, h, h * 0.92 / (VSPAN() + 200), () => ink);
      t.setAttribute('aria-pressed', String(t.dataset.name === S.sel));
    }
  }
  function recipeText(g) {
    const seen = [];
    for (const p of g.parts) if (p.slot && !seen.includes(p.slot)) seen.push(p.slot);
    return seen.map((id) => (R.slots.byId[id] ? R.slots.byId[id].ar : id)).join(' + ');
  }
  function renderPreview() {
    const spec = SPEC[S.sel] || SPEC.beh;
    const g = G(spec.name);
    $('#pvTitle').textContent = charOf(spec);
    $('#pvName').textContent = spec.name;
    $('#pvForm').textContent = FORM_AR[spec.form] || spec.form;
    const cv = $('#pvCanvas');
    const { ctx, w, h } = sizeCanvas(cv);
    if (!g) return;
    let s = h * 0.9 / (VSPAN() + 120);
    s = Math.min(s, w * 0.7 / Math.max(g.adv, 1));
    const base = baseAt(h, s);
    const x = (w - g.adv * s) / 2;
    const line = T('--line'), muted = T('--muted');
    ctx.font = '10.5px "IBM Plex Mono", monospace';
    ctx.direction = 'ltr'; ctx.textAlign = 'left';
    ctx.fillStyle = muted; ctx.strokeStyle = line; ctx.lineWidth = 1;
    const H = (y, label) => { const Y = Math.round(base - y * s) + 0.5; ctx.beginPath(); ctx.moveTo(0, Y); ctx.lineTo(w, Y); ctx.stroke(); ctx.fillText(label, 6, Y - 4); };
    H(0, 'BASE 0'); H(M.X, 'TOOTH ' + M.X); H(M.A, 'ALIF ' + M.A); H(-M.D, 'DESC −' + M.D);
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, h); ctx.moveTo(Math.round(x + g.adv * s) + 0.5, 0); ctx.lineTo(Math.round(x + g.adv * s) + 0.5, h); ctx.stroke();
    ctx.setLineDash([]);
    const ink = T('--ink'), accent = T('--accent');
    R.drawGlyph(ctx, g, S.dna, { x, base, s, colorOf: (p) => (S.hl && p.slot === S.hl) ? accent : ink });
    const info = $('#pvInfo');
    const pf = (() => {
      if (spec.kind === 'letter') { const arr = R.presentation[spec.letter.name]; if (!arr) return null; const i = R.pfOrder.indexOf(spec.form); return arr[i]; }
      if (spec.kind === 'lig') return spec.form === 'fina' ? spec.lig.pf[1] : spec.lig.pf[0];
      return null;
    })();
    const uni = spec.kind === 'letter' ? spec.letter.uni : spec.kind === 'extra' ? spec.extra.uni : null;
    const hex = (v) => 'U+' + v.toString(16).toUpperCase().padStart(4, '0');
    const rows = [
      ['العرض', String(g.adv)],
      ['يونيكود', uni != null ? hex(uni) + (pf ? ' · ' + hex(pf) : '') : (pf ? hex(pf) : '—')],
      ['الوصفة', recipeText(g) || '—']
    ];
    info.innerHTML = '';
    for (const [k, v] of rows) { const dt = document.createElement('dt'); dt.textContent = k; const dd = document.createElement('dd'); dd.textContent = v; if (k !== 'الوصفة') dd.className = 'mono'; info.append(dt, dd); }
  }

  /* feature cards */
  const RASM_MAP = { 'ب': 'ٮ', 'ت': 'ٮ', 'ث': 'ٮ', 'ن': 'ں', 'ي': 'ى', 'ئ': 'ى', 'ج': 'ح', 'خ': 'ح', 'ذ': 'د', 'ز': 'ر', 'ش': 'س', 'ض': 'ص', 'ظ': 'ط', 'غ': 'ع', 'ف': 'ڡ', 'ق': 'ٯ', 'ة': 'ه', 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ؤ': 'و' };
  const FEATS = [
    { id: 'ctx', ar: 'الأشكال السياقية', tag: 'init · medi · fina', on: 'بيت سعيد', ff: '"init" 0, "medi" 0, "fina" 0' },
    { id: 'rlig', ar: 'لام ألف', tag: 'rlig', on: 'سلام لا', ff: '"rlig" 0' },
    { id: 'kash', ar: 'الكشيدة', tag: 'U+0640', on: 'مـحـمـد', off: 'محمد' },
    { id: 'rasm', ar: 'الهيكل بلا نقاط', tag: 'U+066E · U+06A1', on: 'تشبيه', off: null },
    { id: 'digits', ar: 'الأرقام والعلامات', tag: 'U+0660–0669', on: '١٤٤٧ ، ؟', fixed: true },
    { id: 'pf', ar: 'أشكال العرض', tag: 'U+FE70–FEFC', on: 'ﺭﺳﻢ', fixed: true }
  ];
  function buildFeats() {
    const box = $('#feats');
    box.innerHTML = '';
    for (const f of FEATS) {
      const el = document.createElement('div');
      el.className = 'feat'; el.dataset.id = f.id;
      el.innerHTML = '<div class="card-top" style="margin:0"><span class="label"><b></b><span class="mono" style="text-transform:none"></span></span></div><div class="sample" dir="rtl"></div>';
      el.querySelector('b').textContent = f.ar;
      el.querySelector('.mono').textContent = f.tag;
      if (!f.fixed) {
        const sw = document.createElement('button');
        sw.type = 'button'; sw.className = 'switch'; sw.setAttribute('role', 'switch'); sw.setAttribute('aria-checked', 'true');
        sw.setAttribute('aria-label', f.ar);
        sw.addEventListener('click', () => { S.feat[f.id] = !S.feat[f.id]; sw.setAttribute('aria-checked', String(S.feat[f.id])); renderFeats(); });
        el.querySelector('.card-top').appendChild(sw);
      }
      box.appendChild(el);
    }
    renderFeats();
  }
  function renderFeats() {
    for (const f of FEATS) {
      const el = $('.feat[data-id="' + f.id + '"] .sample');
      const on = f.fixed ? true : S.feat[f.id];
      let txt = f.on;
      if (!on && f.off) txt = f.off;
      if (f.id === 'rasm' && !on) txt = Array.from(f.on).map((c) => RASM_MAP[c] || c).join('');
      el.textContent = txt;
      el.style.fontFeatureSettings = (!on && f.ff) ? f.ff : 'normal';
    }
  }

  /* ---------------------------------------------------------------- 05 tester */
  function initTester() {
    const r = $('#sizeRange'), out = $('#sizeOut'), t = $('#testerText');
    const apply = () => { t.style.fontSize = r.value + 'px'; out.textContent = r.value; };
    r.addEventListener('input', apply); apply();
    for (const b of $$('#presets button')) b.addEventListener('click', () => { t.textContent = b.dataset.t; });
  }

  /* ---------------------------------------------------------------- 06 logo maker */
  const MK = {
    ink: () => [T('--ink'), T('--field')], accent: () => [T('--accent'), T('--field')],
    hero: () => ['#191919', '#ffcf96'], night: () => ['#f3efe9', '#191919']
  };
  function renderMaker() {
    const cv = $('#mkCanvas');
    const r = cv.getBoundingClientRect();
    const h = Math.round(Math.max(160, r.width * 0.46));
    const { ctx, w } = sizeCanvas(cv, h);
    cv.style.height = h + 'px';
    const [fg, bg] = MK[S.mkColor]();
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
    const word = $('#mkWord').value.trim() || 'رسم';
    const k = +$('#mkKash').value;
    drawRun(ctx, wordGlyphs(word, k), { x: w * 0.06, y: h * 0.06, w: w * 0.88, h: h * 0.88 }, fg, 'center');
  }
  function initMaker() {
    $('#mkWord').addEventListener('input', renderMaker);
    $('#mkKash').addEventListener('input', () => { $('#mkKashOut').textContent = $('#mkKash').value; renderMaker(); });
    for (const b of $$('#mkColor button')) b.addEventListener('click', () => {
      S.mkColor = b.dataset.c;
      for (const x of $$('#mkColor button')) x.setAttribute('aria-pressed', String(x === b));
      renderMaker();
    });
    $('#mkSvg').addEventListener('click', async () => {
      const res = await ensureFont(); if (!res) return;
      const [fg, bg] = MK[S.mkColor]();
      const word = $('#mkWord').value.trim() || 'رسم';
      const svg = R.textToSVG(res, word, { size: 400, kashida: +$('#mkKash').value, color: fg, bg: S.mkColor === 'ink' || S.mkColor === 'accent' ? null : bg });
      await saveFile(famName() + '-logo.svg', svg, 'image/svg+xml');
    });
    $('#mkPng').addEventListener('click', async () => {
      const word = $('#mkWord').value.trim() || 'رسم';
      const gl = wordGlyphs(word, +$('#mkKash').value);
      const total = gl.reduce((a, g) => a + g.adv, 0);
      const s = 0.9, pad = 120;
      const c = document.createElement('canvas');
      c.width = Math.round(total * s + pad * 2); c.height = Math.round(VSPAN() * s + pad * 2);
      const ctx = c.getContext('2d');
      const [fg, bg] = MK[S.mkColor]();
      if (S.mkColor === 'hero' || S.mkColor === 'night') { ctx.fillStyle = bg; ctx.fillRect(0, 0, c.width, c.height); }
      drawRun(ctx, gl, { x: pad, y: pad, w: c.width - pad * 2, h: c.height - pad * 2 }, fg, 'center', s);
      const blob = await new Promise((res) => c.toBlob(res, 'image/png'));
      if (blob) await saveFile(famName() + '-logo.png', blob, 'image/png');
    });
  }

  /* ---------------------------------------------------------------- export */
  async function downloadFont() {
    const res = await ensureFont();
    if (!res || !S.fontBuf) return;
    const fam = res.family.replace(/\s+/g, '');
    const otf = new Uint8Array(S.fontBuf.slice(0));
    const dl = await downloadsNS();
    if (dl) {
      const readme = 'Rasm — ' + res.family + '\r\n\r\n' +
        'هذا خط OpenType ‏(.otf) وُلّد على جهازك بمحرك «رسم».\r\n' +
        'التثبيت: افتح الملف ' + fam + '-Regular.otf ثم اختر «تثبيت».\r\n\r\n' +
        'Install: open ' + fam + '-Regular.otf and choose Install.\r\n' +
        'Glyphs: ' + res.font.glyphs.length + ' · Features: init, medi, fina, rlig (script arab).\r\n';
      const zip = makeZip([{ name: fam + '-Regular.otf', data: otf }, { name: 'README.txt', data: new TextEncoder().encode(readme) }]);
      await saveFile(fam + '-font.zip', zip, 'application/zip');
    } else {
      await saveFile(fam + '-Regular.otf', new Blob([otf], { type: 'font/otf' }), 'font/otf');
    }
  }
  async function saveProject() {
    const slots = {};
    for (const id in R.slots.overrides) {
      const o = R.slots.overrides[id];
      try { slots[id] = { png: o.img.toDataURL('image/png'), wu: o.wu, hu: o.hu, y0: o.y0, source: S.src[id] || 'upload' }; } catch (e) { /* skip */ }
    }
    const data = JSON.stringify({ app: 'rasm', v: 1, family: famName(), dna: S.dna, measured: S.measured, slots }, null, 1);
    await saveFile(famName() + '-rasm.json', data, 'application/json');
  }
  async function loadProject(file) {
    try {
      const j = JSON.parse(await file.text());
      if (!j || j.app !== 'rasm') throw new Error('هذا ليس ملف مشروع من «رسم».');
      S.dna = R.clampDNA(Object.assign(R.defaultDNA(), j.dna || {}));
      S.measured = j.measured || {};
      if (j.family) $('#famName').value = j.family;
      for (const id in R.slots.overrides) delete R.slots.overrides[id];
      S.src = {};
      const jobs = Object.entries(j.slots || {}).map(([id, o]) => new Promise((res) => {
        const img = new Image();
        img.onload = () => {
          const c = document.createElement('canvas'); c.width = img.naturalWidth; c.height = img.naturalHeight;
          c.getContext('2d').drawImage(img, 0, 0);
          R.slots.overrides[id] = { img: c, wu: o.wu, hu: o.hu, y0: o.y0, source: o.source };
          S.src[id] = o.source === 'logo' ? 'logo' : 'upload';
          res();
        };
        img.onerror = () => res();
        img.src = o.png;
      }));
      await Promise.all(jobs);
      buildAxes();
      changed(true);
      toast('فُتح المشروع');
    } catch (e) { toast(e.message || 'تعذّر فتح الملف'); }
  }
  function initExport() {
    $('#nameForm').addEventListener('submit', (e) => { e.preventDefault(); downloadFont(); });
    $('#navDownload').addEventListener('click', downloadFont);
    $('#famName').addEventListener('change', () => { scheduleBuild(300); });
    $('#saveProject').addEventListener('click', saveProject);
    $('#loadProject').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) loadProject(f); });
    $('#buildBtn').addEventListener('click', () => buildNow());
    downloadsNS().then((dl) => {
      $('#exportNote').textContent = dl
        ? 'اسم الخط بالحروف اللاتينية. في هذه الصفحة يُحفظ الخط داخل ملف ZIP فيه ملف ‎.otf‎.'
        : 'اسم الخط بالحروف اللاتينية. يُحفظ الخط مباشرة بصيغة ‎.otf‎.';
    });
  }

  /* ---------------------------------------------------------------- theme */
  function initTheme() {
    $('#themeBtn').addEventListener('click', () => {
      const root = document.documentElement;
      const cur = root.getAttribute('data-theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      root.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
      requestRender(); clearTimeout(tilesT); tilesT = setTimeout(renderTiles, 60);
    });
    try { matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { requestRender(); }); } catch (e) { /* old browsers */ }
    new MutationObserver(() => requestRender()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  /* ---------------------------------------------------------------- boot */
  function boot() {
    const nGlyphs = INV.length + 1;
    $('#pipelineCount').textContent = '١٢ ← ١٨ ← ' + AR(nGlyphs);
    $('#glyphCount').textContent = AR(nGlyphs);
    buildAxes(); buildSlots(); buildRasm(); buildTiles(); buildFeats();
    initLogoUI(); initTester(); initMaker(); initExport(); initTheme();
    setHL(null);
    let rt = 0;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { requestRender(); }, 120); });
    if (!window.opentype) setStat('nolib');
    loadSample();
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => requestRender());
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
