// Logo Studio — the eight steps we followed on Women Tech, for any logo.
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const NS = 'http://www.w3.org/2000/svg';
const S = {                       // everything the page knows about the open project
  id: null, state: null, step: 1, tool: null, cats: {}, cat: 'h_mid',
  marks: [], measure: null, ref: null, glyphs: [], design: { strokes: [], connections: [], erase: [], hide: [], copies: [] },
  sel: { target: null, match: null }, active: null, vb: null,
};
const STEP_NAMES = { 1: 'الشعار والمنحنيات', 2: 'الوحدة وخط القاعدة', 3: 'تعليم الضربات', 4: 'القياس والتوحيد', 5: 'الاختبار', 6: 'الخط المرجعي', 7: 'بناء الحرف', 8: 'التصدير' };

// ---------- server
async function api(path, opts = {}) {
  $('#busy').hidden = false;
  try {
    const r = await fetch(path, opts);
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (e) {} throw new Error(m); }
    return await r.json();
  } finally { $('#busy').hidden = true; }
}
const post = (p, body) => api(p, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
function fail(e) { alert('خطأ: ' + e.message); }

// ---------- viewer (pan / zoom / pick points in logo coordinates)
const view = $('#view');
function el(tag, attrs = {}, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function clear(id) { const g = $(id); while (g.firstChild) g.removeChild(g.firstChild); return g; }
function setVB(x, y, w, h) { S.vb = [x, y, w, h]; view.setAttribute('viewBox', `${x} ${y} ${w} ${h}`); }
function fit(w, h) { const pad = Math.max(w, h) * 0.04; setVB(-pad, -pad, w + 2 * pad, h + 2 * pad); }
function pt(ev) { const p = view.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY; const q = p.matrixTransform(view.getScreenCTM().inverse()); return [q.x, q.y]; }
const U = () => (S.state && S.state.U) || 10;
view.addEventListener('wheel', ev => {
  ev.preventDefault(); if (!S.vb) return;
  const [mx, my] = pt(ev); const k = ev.deltaY > 0 ? 1.15 : 1 / 1.15; const [x, y, w, h] = S.vb;
  setVB(mx - (mx - x) * k, my - (my - y) * k, w * k, h * k);
}, { passive: false });
let drag = null;
view.addEventListener('pointerdown', ev => {
  const p = pt(ev);
  if (ev.button === 1 || ev.button === 2 || ev.shiftKey || !S.tool) { drag = { pan: true, sx: ev.clientX, sy: ev.clientY, vb: S.vb.slice() }; view.setPointerCapture(ev.pointerId); return; }
  drag = { p0: p, p1: p }; view.setPointerCapture(ev.pointerId);
  if (TOOLS[S.tool] && TOOLS[S.tool].down) TOOLS[S.tool].down(p);
});
view.addEventListener('pointermove', ev => {
  if (!drag) return;
  if (drag.pan) { const r = view.getBoundingClientRect(); const [x, y, w, h] = drag.vb; const s = w / r.width; setVB(x - (ev.clientX - drag.sx) * s, y - (ev.clientY - drag.sy) * s, w, h); return; }
  drag.p1 = pt(ev); if (TOOLS[S.tool] && TOOLS[S.tool].move) TOOLS[S.tool].move(drag.p0, drag.p1);
});
view.addEventListener('pointerup', ev => {
  if (!drag) return; const d = drag; drag = null;
  if (d.pan) return;
  if (TOOLS[S.tool] && TOOLS[S.tool].up) TOOLS[S.tool].up(d.p0, pt(ev));
});
view.addEventListener('contextmenu', e => e.preventDefault());

function tempLine(a, b, color, w) { const g = clear('#L-temp'); el('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: color, 'stroke-width': w, 'stroke-linecap': 'round', opacity: .7 }, g); }
function tempRect(a, b, color) { const g = clear('#L-temp'); el('rect', { x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]), width: Math.abs(b[0] - a[0]), height: Math.abs(b[1] - a[1]), fill: color, 'fill-opacity': .15, stroke: color, 'stroke-width': U() * 0.02 }, g); }

// ---------- tools: what a click / drag does in each step
const TOOLS = {
  crop: { move: (a, b) => tempRect(a, b, '#1f6fe0'), up: (a, b) => { S.box = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])]; tempRect(a, b, '#1f6fe0'); hint('تم تحديد منطقة الشعار. اضغط «حوّل إلى منحنيات».'); } },
  unit: { move: (a, b) => tempLine(a, b, '#d62828', U() * 0.03), up: (a, b) => { const u = Math.abs(b[1] - a[1]); if (u > 1) { $('#inU').value = u.toFixed(2); hint('الوحدة = ' + u.toFixed(2) + ' بكسل. احفظ.'); } } },
  baseline: { up: (a, b) => { $('#inBase').value = b[1].toFixed(2); drawSettings(+$('#inU').value, b[1]); } },
  mark: { move: (a, b) => tempLine(a, b, S.cats[S.cat].color, U() * 0.08), up: (a, b) => {
      clear('#L-temp'); if (Math.hypot(b[0] - a[0], b[1] - a[1]) < U() * 0.05) return;
      const id = String(1 + Math.max(0, ...S.marks.map(m => +m.id || 0)));
      S.marks.push({ id, cat: S.cat, p0: a, p1: b }); drawMarks(); renderMarkList(); } },
  height: { up: (a, b) => { const h = (S.state.baseline - b[1]) / U(); $('#inMH').value = h.toFixed(3); hint('ارتفاع النقطة فوق خط القاعدة = ' + h.toFixed(3) + ' وحدة'); } },
  rightx: { up: (a, b) => { $('#inRX').value = b[0].toFixed(2); hint('سيقع الطرف الأيمن للحرف عند x = ' + b[0].toFixed(1)); } },
  extend: { up: (a, b) => { const st = S.design.strokes[S.active.i]; const key = S.active.where === 'before' ? 'extend_before' : 'extend_after'; (st[key] = st[key] || []).push(b); drawDesign(); renderStrokes(); } },
  conn: { up: (a, b) => { const c = S.design.connections[S.active.i]; c.points.push(b); drawDesign(); renderConnections(); } },
  erase: { move: (a, b) => tempRect(a, b, '#c62828'), up: (a, b) => { clear('#L-temp'); const part = $('#erasePart').value; S.design.erase.push({ part, rect: [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])] }); drawDesign(); renderErase(); } },
};
function setTool(t, label) {
  S.tool = t; document.querySelectorAll('#toolbar button[data-tool]').forEach(b => b.classList.toggle('on', b.dataset.tool === t));
  hint(label || ''); clear('#L-temp');
}
function hint(t) { $('#hint').textContent = t || ''; }

// ---------- drawing layers
function drawPicture() {
  const g = clear('#L-pic'); clear('#L-logo');
  const [W, H] = S.state.picture || [1000, 1000];
  el('image', { href: `/api/projects/${S.id}/picture`, x: 0, y: 0, width: W, height: H }, g);
  if (!S.vb) fit(W, H);
}
async function drawLogo(highlight) {
  clear('#L-pic');
  const parts = S.parts || (S.parts = await api(`/api/projects/${S.id}/parts`));
  const g = clear('#L-logo');
  for (const p of parts) {
    const path = el('path', { d: p.d, fill: (S.step === 7 && S.design.hide.includes(p.name)) ? '#e6e6e6' : '#222', 'fill-rule': 'evenodd', 'data-part': p.name }, g);
    el('title', {}, path).textContent = p.name;
  }
}
function drawSettings(u, base) {
  const g = clear('#L-over'); if (!S.state.size) return; const [W] = S.state.size;
  el('line', { x1: -5, y1: base, x2: W + 5, y2: base, stroke: '#1e8e3e', 'stroke-width': u * 0.02, 'stroke-dasharray': `${u * .1} ${u * .05}` }, g);
  el('rect', { x: 2, y: base - u, width: u * 0.12, height: u, fill: '#d62828', opacity: .6 }, g);
  const t = el('text', { x: 2 + u * .18, y: base - u / 2, 'font-size': u * 0.35, fill: '#d62828' }, g); t.textContent = '1 U';
}
function drawMarks(results) {
  const g = clear('#L-over');
  for (const m of S.marks) {
    const c = (S.cats[m.cat] || {}).color || '#888';
    el('line', { x1: m.p0[0], y1: m.p0[1], x2: m.p1[0], y2: m.p1[1], stroke: c, 'stroke-width': U() * 0.08, 'stroke-linecap': 'round', opacity: .65 }, g);
    const t = el('text', { x: m.p1[0] + U() * .05, y: m.p1[1] - U() * .05, 'font-size': U() * .22, fill: '#000' }, g); t.textContent = m.id;
    const r = results && results[m.id];
    if (r && r.seg) { const s = r.seg; el('line', { x1: s[0][0], y1: s[0][1], x2: s[s.length - 1][0], y2: s[s.length - 1][1], stroke: '#000', 'stroke-width': U() * 0.02 }, g);
      const tt = el('text', { x: r.at[0] + U() * .08, y: r.at[1] + U() * .3, 'font-size': U() * .2, fill: '#c62828' }, g); tt.textContent = r.width.toFixed(3); }
  }
}
function polyline(P, attrs, g) { return el('polyline', Object.assign({ points: P.map(p => p[0].toFixed(2) + ',' + p[1].toFixed(2)).join(' '), fill: 'none' }, attrs), g); }
function drawReference() {
  const g = clear('#L-ref'); if (!S.ref) return;
  el('path', { d: S.ref.word_path, fill: '#c62828', 'fill-opacity': .12, stroke: '#c62828', 'stroke-width': U() * .01 }, g);
  el('path', { d: S.ref.glyph_path, fill: '#c62828', 'fill-opacity': .25, stroke: '#c62828', 'stroke-width': U() * .02 }, g);
  const colors = ['#1f6fe0', '#1e8e3e', '#a347c2', '#e09100', '#0097a7', '#6d4c41', '#d81b60', '#5e35b1'];
  for (const e of S.ref.edges) {
    const col = colors[e.id % colors.length];
    polyline(e.P, { stroke: col, 'stroke-width': U() * .04 }, g);
    const m = e.P[Math.floor(e.P.length / 2)];
    const t = el('text', { x: m[0], y: m[1] - U() * .08, 'font-size': U() * .28, fill: col, 'font-weight': 'bold' }, g); t.textContent = e.id;
  }
}
function drawDesign() {
  const g = clear('#L-over');
  S.design.strokes.forEach((st, i) => { ['extend_before', 'extend_after'].forEach(k => (st[k] || []).forEach(p => el('circle', { cx: p[0], cy: p[1], r: U() * .05, fill: '#1f6fe0' }, g))); });
  S.design.connections.forEach((c, i) => { if (c.points.length) polyline(c.points, { stroke: '#1e8e3e', 'stroke-width': U() * .03, 'stroke-dasharray': `${U() * .06}` }, g); c.points.forEach(p => el('circle', { cx: p[0], cy: p[1], r: U() * .05, fill: '#1e8e3e' }, g)); });
  S.design.erase.forEach(e => { const r = e.rect; el('rect', { x: r[0], y: r[1], width: r[2] - r[0], height: r[3] - r[1], fill: '#c62828', 'fill-opacity': .08, stroke: '#c62828', 'stroke-width': U() * .02 }, g); });
}
function drawResult(show) {
  const g = clear('#L-result'); const r = S.state.result; if (!r || !show) { $('#L-logo').style.opacity = 1; return; }
  $('#L-logo').style.opacity = .12;
  el('path', { d: r.path, fill: '#111', 'fill-rule': 'evenodd' }, g);
}

// ---------- steps
function setStep(n) {
  S.step = n; document.querySelectorAll('#steps button[data-step]').forEach(b => { b.classList.toggle('on', +b.dataset.step === n); b.classList.toggle('done', S.state && +b.dataset.step < (S.state.step || 1)); });
  setTool(null); const tb = $('#toolbar'); tb.innerHTML = ''; $('#panel').innerHTML = '';
  clear('#L-over'); clear('#L-temp'); clear('#L-ref'); clear('#L-result'); $('#L-logo').style.opacity = 1;
  if (!S.id && n !== 1) { $('#panel').innerHTML = '<p>ابدأ برفع شعار في الخطوة الأولى.</p>'; return; }
  STEPS[n]();
}
function tool(label, t, help) { const b = document.createElement('button'); b.textContent = label; b.dataset.tool = t; b.onclick = () => setTool(S.tool === t ? null : t, help); $('#toolbar').appendChild(b); return b; }
function btn(label, fn, primary, parent) { const b = document.createElement('button'); b.className = 'btn' + (primary ? ' primary' : ''); b.textContent = label; b.onclick = fn; (parent || $('#toolbar')).appendChild(b); return b; }
const fmt = v => (v === null || v === undefined || Number.isNaN(v)) ? '—' : (+v).toFixed(3);

const STEPS = {
  1() {
    const p = $('#panel');
    p.innerHTML = `<h2>١. الشعار والمنحنيات</h2>
      <p>ارفع صورة الشعار. حدّد بالسحب منطقة الشعار إن كانت الصورة فيها أشياء أخرى، ثم حوّله إلى منحنيات بيزيه. الحافة تقع في منتصف المسافة بين لون الخلفية ولون الحرف، والزوايا الحادة فقط حيث ينكسر المحيط فعلاً.</p>
      <div class="row"><input type="file" id="file" accept="image/*"> <span class="muted">PNG أو JPG، والأكبر أدق.</span></div>
      <div id="rep"></div>`;
    $('#file').onchange = async ev => {
      const f = ev.target.files[0]; if (!f) return; const fd = new FormData(); fd.append('file', f);
      try { const st = await api('/api/projects', { method: 'POST', body: fd }); await openProject(st.id, 1); } catch (e) { fail(e); }
    };
    if (!S.id) return;
    tool('حدّد منطقة الشعار', 'crop', 'اسحب مستطيلاً حول الشعار.');
    btn('حوّل إلى منحنيات', async () => {
      try { const r = await post(`/api/projects/${S.id}/vectorize`, { box: S.box || null }); S.parts = null; await reload(); S.vb = null; fit(...S.state.size); await drawLogo(); showReport(); setStep(1); } catch (e) { fail(e); }
    }, true);
    if (S.state.parts) { if (!S.vb || S.vbFor !== 'logo') { fit(...S.state.size); S.vbFor = 'logo'; } drawLogo(); btn('اعرض الصورة الأصلية', () => { S.vb = null; S.vbFor = 'pic'; drawPicture(); }); showReport(); }
    else { drawPicture(); }
  },
  2() {
    drawLogo(); const st = S.state;
    $('#panel').innerHTML = `<h2>٢. الوحدة وخط القاعدة</h2>
      <p>كل الأرقام تُكتب بوحدة واحدة. في Women Tech كانت الوحدة ارتفاع الحرف اللاتيني الصغير. اختر ما يناسب هذا الشعار: ارسم خطاً عمودياً بطول الوحدة، أو اكتب الرقم. ثم حدّد خط القاعدة الذي تجلس عليه الحروف العربية.</p>
      <div class="row">الوحدة (بكسل): <input type="number" step="0.01" id="inU" value="${(st.U || 0).toFixed(2)}">
      خط القاعدة (y): <input type="number" step="0.01" id="inBase" value="${(st.baseline || 0).toFixed(2)}"></div>
      <p class="muted">البكسل الواحد في الصورة = <b id="pxU">${(1 / (st.U || 1)).toFixed(3)}</b> من الوحدة. أي فرق أصغر من ذلك ليس حقيقياً.</p>`;
    tool('ارسم الوحدة', 'unit', 'اسحب من أعلى الحرف إلى أسفله.');
    tool('حدّد خط القاعدة', 'baseline', 'انقر على خط القاعدة.');
    btn('احفظ', async () => { try { const r = await post(`/api/projects/${S.id}/settings`, { U: +$('#inU').value, baseline: +$('#inBase').value }); S.state.U = r.U; S.state.baseline = r.baseline; $('#pxU').textContent = (1 / r.U).toFixed(3); drawSettings(r.U, r.baseline); hint('حُفظت.'); } catch (e) { fail(e); } }, true);
    $('#inU').oninput = $('#inBase').oninput = () => drawSettings(+$('#inU').value, +$('#inBase').value);
    drawSettings(st.U, st.baseline);
  },
  3() {
    drawLogo(); drawMarks();
    const p = $('#panel');
    p.innerHTML = `<h2>٣. تعليم الضربات</h2>
      <p>أنت تقرر ما الضربة وأين تبدأ وأين تنتهي. ارسم خطاً قصيراً عبر الضربة في كل موضع، واختر نوع المكان قبل الرسم. البرنامج يقيس فقط.</p>
      <div id="palette" class="row"></div>
      <p class="muted">التكبير بعجلة الفأرة، والتحريك بالزر الأيمن أو بـ Shift مع السحب.</p>
      <div id="markList"></div>`;
    const pal = $('#palette');
    for (const k in S.cats) { const b = document.createElement('span'); b.className = 'chip' + (k === S.cat ? ' on' : ''); b.innerHTML = `<span class="sw" style="background:${S.cats[k].color}"></span>${S.cats[k].ar}`; b.onclick = () => { S.cat = k; pal.querySelectorAll('.chip').forEach(c => c.classList.remove('on')); b.classList.add('on'); setTool('mark', 'ارسم خطاً عبر الضربة: ' + S.cats[k].ar); }; pal.appendChild(b); }
    tool('علّم', 'mark', 'ارسم خطاً عبر الضربة.');
    btn('تراجع', () => { S.marks.pop(); drawMarks(); renderMarkList(); });
    renderMarkList(); setTool('mark', 'اختر النوع ثم ارسم خطاً عبر الضربة.');
  },
  4() {
    drawLogo(); drawMarks(S.measure && S.measure.results);
    $('#panel').innerHTML = `<h2>٤. القياس والتوحيد</h2>
      <p>عند كل علامة: النقطة التي يعبر فيها خطك منتصف الضربة، والعرض هناك عمودياً على الضربة. ثم تُقرأ كل ضربة كاملة بين علاماتها. بعدها كل نوع مكان يُوحَّد بالمتوسط، وكل الشعيرات برقم واحد إن أردت.</p>
      <div class="row"><label><input type="checkbox" id="hair1" ${(!S.measure || S.measure.hairline_one !== false) ? 'checked' : ''}> وحّد كل الشعيرات برقم واحد</label></div>
      <div id="mres"></div>`;
    btn('قِس عند علاماتي', () => runMeasure(), true);
    if (S.measure) renderMeasure();
  },
  5() {
    drawLogo();
    const g = S.measure && S.measure.gate;
    $('#panel').innerHTML = `<h2>٥. الاختبار</h2><p>قبل رسم أي حرف جديد تُعاد كل ضربة قيست من الأرقام الموحّدة وحدها، على مسارها هي، ثم تُقارن بالأصل. إن لم تطابق، فالأرقام أو التعليم يحتاجان مراجعة.</p><div id="gres"></div>`;
    if (!g) { $('#gres').innerHTML = '<p>نفّذ القياس أولاً.</p>'; return; }
    const kind = { H: 'عرضي', V: 'عمودي' };
    let h = `<p>متوسط التطابق: <b class="${g.mean_iou >= .85 ? 'good' : 'bad'}">${(100 * g.mean_iou).toFixed(1)}%</b> <span class="muted">(في Women Tech: 81% لكل ضربة وحدها، و87% للنون كاملة)</span></p>
      <table><tr><th>الضربة</th><th>النوع</th><th>التطابق</th><th>الأصل: بداية / أتخن / نهاية</th><th>المعاد: بداية / أتخن / نهاية</th></tr>`;
    for (const s of g.strokes) h += `<tr><td>${s.start} → ${s.end}</td><td>${kind[s.kind]}</td><td class="${s.iou >= .85 ? 'good' : 'bad'}">${(100 * s.iou).toFixed(1)}%</td><td class="num">${s.real.map(fmt).join(' / ')}</td><td class="num">${s.rebuilt.map(fmt).join(' / ')}</td></tr>`;
    $('#gres').innerHTML = h + '</table>';
    const gg = clear('#L-over'); for (const s of S.measure.spans) polyline(s.P, { stroke: s.kind === 'H' ? '#1f6fe0' : s.kind === 'V' ? '#e09100' : '#a347c2', 'stroke-width': U() * .04 }, gg);
  },
  6() {
    drawLogo(); drawReference();
    const r = S.ref || {};
    $('#panel').innerHTML = `<h2>٦. الخط المرجعي</h2>
      <p>الخط المرجعي من عائلة الشعار نفسها (نسخ، رقعة…)، ويعطي <b>الشكل والحجم والمكان فقط</b>. اكتب كلمة فيها الحرف الجديد، واختر الحرف الجديد، واختر حرفاً موجوداً في الشعار نضبط عليه الحجم.</p>
      <div class="row">ملف الخط: <input type="file" id="font" accept=".ttf,.otf"> ${S.state.font ? '<span class="good">✓ محمّل</span>' : ''}</div>
      <div class="row">الكلمة: <input type="text" id="word" value="${r.text || ''}" style="width:200px"> <button class="btn" id="bShape">شكّل</button></div>
      <div id="glyphs"></div>
      <div class="row">ارتفاع الحرف المطابق في الشعار (وحدة): <input type="number" step="0.001" id="inMH" value="${r.match_height_u || ''}">
        الطرف الأيمن للحرف الجديد (x): <input type="number" step="0.1" id="inRX" value="${r.right_x || ''}"></div>
      <div id="edges"></div>`;
    $('#font').onchange = async ev => { const f = ev.target.files[0]; if (!f) return; const fd = new FormData(); fd.append('file', f); try { await api(`/api/projects/${S.id}/font`, { method: 'POST', body: fd }); S.state.font = f.name; hint('حُمّل الخط.'); } catch (e) { fail(e); } };
    $('#bShape').onclick = async () => { try { S.glyphs = await post(`/api/projects/${S.id}/shape`, { text: $('#word').value }); renderGlyphs(); } catch (e) { fail(e); } };
    tool('اقرأ ارتفاعاً من الشعار', 'height', 'انقر على قمة الحرف المطابق في الشعار.');
    tool('حدّد الطرف الأيمن', 'rightx', 'انقر حيث يجب أن يقع الطرف الأيمن للحرف الجديد.');
    btn('ضع الحرف', async () => {
      try { S.ref = await post(`/api/projects/${S.id}/reference`, { text: $('#word').value, match_index: S.sel.match, match_height_u: +$('#inMH').value, target_index: S.sel.target, right_x: +$('#inRX').value });
        S.state.reference = S.ref; drawReference(); renderEdges(); } catch (e) { fail(e); }
    }, true);
    if (r.glyphs) { S.glyphs = r.glyphs; S.sel.target = r.target_index; S.sel.match = r.match_index; renderGlyphs(); renderEdges(); }
  },
  7() {
    drawLogo(); drawReference(); drawDesign();
    $('#panel').innerHTML = `<h2>٧. بناء الحرف</h2>
      <p>قرّر أنت تقسيم الحرف: كل ضربة مجموعة من خطوط المرجع المرقّمة. اذكر نوعها، وماذا يلتقي بطرفيها: شعيرة أم جذع تخين. السُّمك يأتي كله من الأرقام الموحّدة: العرضي أتخن عند أخفض نقطة فيه، والعمودي يرقّ من أعلاه إلى أسفله. ما يصل الحرف بحرف آخر ارسمه وصلةً مستقلة.</p>
      <h3>الضربات</h3><div id="strokes"></div>
      <h3>الوصلات بالحروف المجاورة</h3><div id="conns"></div>
      <h3>ما يُحذف من الشعار</h3><div id="erase"></div>
      <h3>أجزاء تُخفى، ونقاط تُنسخ</h3><div id="hide"></div><div id="copies"></div>
      <div id="checks"></div>`;
    btn('ابنِ الحرف', () => runBuild(), true);
    btn('أظهر / أخفِ النتيجة', () => { S.showResult = !S.showResult; drawResult(S.showResult); });
    renderStrokes(); renderConnections(); renderErase(); renderHide(); renderCopies(); if (S.state.result) { renderChecks(); drawResult(true); S.showResult = true; }
  },
  8() {
    drawLogo(); if (S.state.result) drawResult(true);
    $('#panel').innerHTML = `<h2>٨. التصدير</h2>
      <p>منحنيات الشعار، والشعار مع الحرف الجديد كمنحنيات بيزيه قابلة للتعديل، وتقرير بكل الأرقام وقرارات التصميم.</p>
      <div class="cards">
        <a class="card" href="/api/projects/${S.id}/logo.svg" download>منحنيات الشعار (SVG)</a>
        ${S.state.result ? `<a class="card" href="/api/projects/${S.id}/result.svg" download>الشعار مع الحرف الجديد (SVG)</a>
        <a class="card" href="/api/projects/${S.id}/result.png" download>صورة (PNG)</a>` : '<span class="card muted">ابنِ الحرف أولاً</span>'}
        <a class="card" href="/api/projects/${S.id}/report.json" download>التقرير (JSON)</a>
      </div>`;
  },
};

// ---------- panels
function showReport() {
  const r = S.state.report; if (!r) return;
  $('#rep').innerHTML = `<table><tr><th>الأجزاء</th><th>قطع المنحنى</th><th>الزوايا الحادة</th><th>التطابق مع الصورة</th><th>بُعد المنحنى عن الحافة: متوسط / أقصى (بكسل)</th></tr>
    <tr><td>${r.parts}</td><td>${r.curve_segments}</td><td>${r.sharp_corners}</td><td class="${r.match_vs_picture >= .95 ? 'good' : 'bad'}">${(100 * r.match_vs_picture).toFixed(1)}%</td><td class="num">${r.edge_error_mean_px} / ${r.edge_error_max_px}</td></tr></table>
    <p class="muted">في Women Tech: 97.3% و0.06 بكسل.</p>`;
}
function renderMarkList() {
  const by = {}; S.marks.forEach(m => (by[m.cat] = by[m.cat] || []).push(m));
  let h = `<p>${S.marks.length} علامة.</p><table><tr><th>النوع</th><th>العلامات (انقر لحذف)</th></tr>`;
  for (const k in S.cats) h += `<tr><td><span class="sw" style="background:${S.cats[k].color}"></span>${S.cats[k].ar}</td><td>${(by[k] || []).map(m => `<span class="chip" data-del="${m.id}">${m.id}</span>`).join('')}</td></tr>`;
  $('#markList').innerHTML = h + '</table>';
  $('#markList').querySelectorAll('[data-del]').forEach(c => c.onclick = () => { S.marks = S.marks.filter(m => m.id !== c.dataset.del); drawMarks(); renderMarkList(); });
}
async function runMeasure(overrides) {
  try {
    S.measure = await post(`/api/projects/${S.id}/measure`, { marks: S.marks, overrides: overrides || (S.measure && S.measure.overrides) || null, hairline_one: $('#hair1') ? $('#hair1').checked : true });
    S.state.measure = S.measure; S.state.step = Math.max(S.state.step, 5); drawMarks(S.measure.results); renderMeasure();
  } catch (e) { fail(e); }
}
const SPEC_NAMES = { H_MID: 'أتخن الخط العرضي', H_HAIR: 'الشعيرة (طرف عرضي، أسفل عمودي، وصلة)', H_STEM: 'طرف عرضي عند جذع', HEAD_TOP: 'أعلى الرأس', V_TOP: 'أعلى عمودي (قمة مقطوعة)', V_MID: 'وسط عمودي', V_BOT: 'أسفل عمودي', JOINT: 'وصلة', THICK_POS: 'موضع الأتخن إن لم تكن للضربة نقطة أخفض (نسبة من الطول)' };
function renderMeasure() {
  const m = S.measure; if (!m) return;
  let h = `<h3>عند كل علامة</h3><table><tr><th>علامة</th><th>النوع</th><th>العرض (وحدة)</th><th>كيف قيست</th></tr>`;
  for (const mk of S.marks) { const r = m.results[mk.id]; if (!r) continue; h += `<tr><td>${mk.id}</td><td>${(S.cats[mk.cat] || {}).ar || mk.cat}</td><td class="num">${fmt(r.width)}</td><td>${r.how === 'across the stroke' ? 'عمودياً على الضربة' : r.how === 'along your line' ? 'على امتداد خطك' : 'خارج الشعار'}</td></tr>`; }
  h += `</table><h3>التوحيد بالمتوسط</h3><table><tr><th>المكان</th><th>المتوسط</th><th>العدد</th><th>من – إلى</th><th>أبعد قياس عن المتوسط</th></tr>`;
  for (const k in m.table) { const t = m.table[k]; const name = k === 'hair' ? 'الشعيرة (كل الأماكن الرفيعة)' : (S.cats[k] || {}).ar || k;
    h += `<tr><td>${name}</td><td class="num"><b>${fmt(t.value)}</b></td><td>${t.n}</td><td class="num">${fmt(t.min)} – ${fmt(t.max)}</td><td>${t.farthest ? t.farthest + ' (' + (t.farthest_gap > 0 ? '+' : '') + fmt(t.farthest_gap) + ')' : ''}</td></tr>`; }
  const sp = m.spec; const lr = sp.lowest_point_rule;
  h += `</table><p>دقة الصورة: البكسل = <b>${fmt(sp.pixel)}</b> وحدة. قاعدة «الأتخن عند أخفض نقطة» صحيحة في <b>${lr.holds} من ${lr.of}</b> خطوط عرضية.</p>
    <h3>الأرقام الموحّدة التي سيُرسم بها (يمكنك تعديلها)</h3><table>`;
  for (const k in SPEC_NAMES) h += `<tr><td>${SPEC_NAMES[k]}</td><td><input type="number" step="0.001" data-spec="${k}" value="${sp[k] == null ? '' : sp[k]}"></td></tr>`;
  h += '</table>';
  $('#mres').innerHTML = h;
  btn('أعد الحساب بالأرقام المعدّلة', () => { const o = {}; document.querySelectorAll('[data-spec]').forEach(i => { if (i.value !== '') o[i.dataset.spec] = +i.value; }); runMeasure(o); }, false, $('#mres'));
}
function renderGlyphs() {
  let h = `<table><tr><th>#</th><th>الحرف</th><th>اسم الشكل</th><th>الحرف الجديد</th><th>نطابق عليه الحجم</th></tr>`;
  for (const g of S.glyphs) h += `<tr><td>${g.index}</td><td>${g.char}</td><td style="direction:ltr">${g.name}</td><td><input type="radio" name="tg" value="${g.index}" ${S.sel.target === g.index ? 'checked' : ''}></td><td><input type="radio" name="mg" value="${g.index}" ${S.sel.match === g.index ? 'checked' : ''}></td></tr>`;
  $('#glyphs').innerHTML = h + '</table>';
  document.querySelectorAll('[name=tg]').forEach(i => i.onchange = () => S.sel.target = +i.value);
  document.querySelectorAll('[name=mg]').forEach(i => i.onchange = () => S.sel.match = +i.value);
}
function renderEdges() {
  if (!S.ref) return;
  let h = `<p>المقياس: ${S.ref.scale.toFixed(4)}. خطوط الحرف المرجعي المرقّمة (تراها ملوّنة على الرسم):</p><table><tr><th>#</th><th>الطول (وحدة)</th><th>الطرفان</th></tr>`;
  for (const e of S.ref.edges) h += `<tr><td>${e.id}</td><td class="num">${e.length}</td><td>${e.ends.map(x => x === 'end' ? 'طرف حر' : 'ملتقى').join(' – ')}</td></tr>`;
  $('#edges').innerHTML = h + '</table>';
}
function sel(opts, val) { return opts.map(([v, t]) => `<option value="${v}" ${v === val ? 'selected' : ''}>${t}</option>`).join(''); }
function renderStrokes() {
  const box = $('#strokes'); if (!box) return;
  let h = `<table><tr><th>#</th><th>خطوط المرجع بالترتيب</th><th>النوع</th><th>البداية</th><th>النهاية</th><th>أعلى العمودي</th><th>اعكس</th><th>امتداد</th><th></th></tr>`;
  S.design.strokes.forEach((st, i) => {
    h += `<tr><td>${i}</td><td><input type="text" data-i="${i}" data-k="edges" value="${st.edges.join(',')}"></td>
      <td><select data-i="${i}" data-k="type">${sel([['H', 'عرضي'], ['V', 'عمودي']], st.type)}</select></td>
      <td><select data-i="${i}" data-k="start">${sel([['hair', 'شعيرة'], ['stem', 'جذع']], st.start)}</select></td>
      <td><select data-i="${i}" data-k="end">${sel([['hair', 'شعيرة'], ['stem', 'جذع']], st.end)}</select></td>
      <td><select data-i="${i}" data-k="top">${sel([['cut', 'قمة مقطوعة'], ['head', 'رأس']], st.top || 'cut')}</select></td>
      <td><input type="checkbox" data-i="${i}" data-k="reverse" ${st.reverse ? 'checked' : ''}></td>
      <td><button class="btn" data-ext="${i}" data-where="before">قبل (${(st.extend_before || []).length})</button><button class="btn" data-ext="${i}" data-where="after">بعد (${(st.extend_after || []).length})</button><button class="btn" data-clr="${i}">امسح</button></td>
      <td><button class="btn" data-rm="${i}">✕</button></td></tr>`;
  });
  box.innerHTML = h + '</table>';
  btn('أضف ضربة', () => { S.design.strokes.push({ edges: [], type: 'H', start: 'hair', end: 'hair' }); renderStrokes(); }, false, box);
  box.querySelectorAll('[data-k]').forEach(inp => inp.onchange = () => { const st = S.design.strokes[+inp.dataset.i]; const k = inp.dataset.k;
    st[k] = k === 'edges' ? inp.value.split(/[,،\s]+/).filter(x => x !== '').map(Number) : inp.type === 'checkbox' ? inp.checked : inp.value; });
  box.querySelectorAll('[data-ext]').forEach(b => b.onclick = () => { S.active = { i: +b.dataset.ext, where: b.dataset.where }; setTool('extend', 'انقر نقاطاً يمتد عليها مسار الضربة ' + b.dataset.ext + (b.dataset.where === 'before' ? ' قبل بدايتها' : ' بعد نهايتها') + '.'); });
  box.querySelectorAll('[data-clr]').forEach(b => b.onclick = () => { const st = S.design.strokes[+b.dataset.clr]; delete st.extend_before; delete st.extend_after; drawDesign(); renderStrokes(); });
  box.querySelectorAll('[data-rm]').forEach(b => b.onclick = () => { S.design.strokes.splice(+b.dataset.rm, 1); renderStrokes(); drawDesign(); });
}
function renderConnections() {
  const box = $('#conns'); if (!box) return;
  let h = `<table><tr><th>#</th><th>النقاط</th><th>البداية</th><th>النهاية</th><th></th></tr>`;
  S.design.connections.forEach((c, i) => {
    h += `<tr><td>${i}</td><td>${c.points.length} <button class="btn" data-pts="${i}">انقر النقاط</button><button class="btn" data-cl="${i}">امسح</button></td>
      <td><select data-i="${i}" data-k="start">${sel([['hair', 'شعيرة'], ['stem', 'جذع']], c.start)}</select></td>
      <td><select data-i="${i}" data-k="end">${sel([['hair', 'شعيرة'], ['stem', 'جذع']], c.end)}</select></td><td><button class="btn" data-rm="${i}">✕</button></td></tr>`;
  });
  box.innerHTML = h + '</table><p class="muted">ابدأ النقاط من الحرف الذي تخرج منه الوصلة. أول نقطة وآخر نقطة هما نقطتا الالتقاء، وبينهما أخفض نقطة.</p>';
  btn('أضف وصلة', () => { S.design.connections.push({ points: [], start: 'hair', end: 'hair' }); renderConnections(); }, false, box);
  box.querySelectorAll('[data-k]').forEach(inp => inp.onchange = () => { S.design.connections[+inp.dataset.i][inp.dataset.k] = inp.value; });
  box.querySelectorAll('[data-pts]').forEach(b => b.onclick = () => { S.active = { i: +b.dataset.pts }; setTool('conn', 'انقر نقاط مسار الوصلة ' + b.dataset.pts + ' بالترتيب.'); });
  box.querySelectorAll('[data-cl]').forEach(b => b.onclick = () => { S.design.connections[+b.dataset.cl].points = []; drawDesign(); renderConnections(); });
  box.querySelectorAll('[data-rm]').forEach(b => b.onclick = () => { S.design.connections.splice(+b.dataset.rm, 1); renderConnections(); drawDesign(); });
}
function partOptions(v) { return (S.parts || []).map(p => `<option ${p.name === v ? 'selected' : ''}>${p.name}</option>`).join(''); }
function renderErase() {
  const box = $('#erase'); if (!box) return;
  box.innerHTML = `<div class="row">الجزء: <select id="erasePart">${partOptions()}</select> <button class="btn" id="bErase">ارسم مستطيل حذف</button> ${S.design.erase.map((e, i) => `<span class="chip" data-rm="${i}">${e.part} ✕</span>`).join('')}</div>
    <p class="muted">الحذف داخل الجزء المختار فقط، مثل حذف سنّ الياء والكشيدة القديمة من كلمة واحدة دون المساس بغيرها. مرّر الفأرة على الرسم لترى اسم كل جزء.</p>`;
  $('#bErase').onclick = () => setTool('erase', 'اسحب مستطيلاً على ما يُحذف من الجزء ' + $('#erasePart').value + '.');
  box.querySelectorAll('[data-rm]').forEach(c => c.onclick = () => { S.design.erase.splice(+c.dataset.rm, 1); renderErase(); drawDesign(); });
}
function renderHide() {
  const box = $('#hide'); if (!box) return;
  box.innerHTML = '<div class="row">' + (S.parts || []).map(p => `<label class="chip"><input type="checkbox" data-h="${p.name}" ${S.design.hide.includes(p.name) ? 'checked' : ''}> ${p.name}</label>`).join('') + '</div><p class="muted">الأجزاء المؤشَّر عليها تُخفى، مثل نقاط الياء وعلامتها.</p>';
  box.querySelectorAll('[data-h]').forEach(i => i.onchange = () => { const n = i.dataset.h; S.design.hide = S.design.hide.filter(x => x !== n); if (i.checked) S.design.hide.push(n); drawLogo(); });
}
function renderCopies() {
  const box = $('#copies'); if (!box) return;
  box.innerHTML = `<div class="row">انسخ الجزء <select id="cpPart">${partOptions()}</select> فوق أتخن موضع في الضربة <input type="number" id="cpStroke" value="0" min="0" style="width:60px"> بمسافة <input type="number" id="cpGap" value="0.2" step="0.05" style="width:70px"> وحدة <button class="btn" id="bCopy">أضف</button>
    ${S.design.copies.map((c, i) => `<span class="chip" data-rm="${i}">${c.part} → ${c.above_stroke} ✕</span>`).join('')}</div><p class="muted">مثل نقطة الخاء: معيّن نقطة النون من الشعار نفسه.</p>`;
  $('#bCopy').onclick = () => { S.design.copies.push({ part: $('#cpPart').value, above_stroke: +$('#cpStroke').value, gap: +$('#cpGap').value }); renderCopies(); };
  box.querySelectorAll('[data-rm]').forEach(c => c.onclick = () => { S.design.copies.splice(+c.dataset.rm, 1); renderCopies(); });
}
async function runBuild() {
  try { const r = await post(`/api/projects/${S.id}/build`, S.design); S.state.result = r; S.state.step = Math.max(S.state.step, 8); S.showResult = true; drawResult(true); renderChecks(); }
  catch (e) { fail(e); }
}
function renderChecks() {
  const r = S.state.result; if (!r) return; const sp = (S.measure || S.state.measure || {}).spec || {};
  let h = `<h3>القياس على الحرف بعد رسمه</h3><table><tr><th>#</th><th>النوع</th><th>الطول</th><th>المطلوب: بداية / أتخن / نهاية</th><th>المرسوم: بداية / أتخن (موضعه) / نهاية</th><th>قطوع نظيفة</th></tr>`;
  for (const c of r.checks) { if (!c.clean) { h += `<tr><td>${c.i}</td><td colspan="5">داخل ضربة أخرى، فلا يُقاس وحده</td></tr>`; continue; }
    h += `<tr><td>${c.i}</td><td>${c.kind === 'connection' ? 'وصلة' : (c.type === 'V' ? 'عمودي' : 'عرضي')}</td><td class="num">${c.length}</td><td class="num">${c.designed.map(fmt).join(' / ')}</td><td class="num">${fmt(c.start)} / ${fmt(c.thickest)} (${c.thickest_at}%) / ${fmt(c.end)}</td><td>${c.clean}/21</td></tr>`; }
  $('#checks').innerHTML = h + '</table><p class="muted">الطرف الذي يقع داخل حرف آخر يُقاس أعرض، لأن القطع يعبر الحرف المجاور.</p>';
}

// ---------- project open / reload
async function reload() { S.state = await api(`/api/projects/${S.id}`); }
async function openProject(id, step) {
  S.id = id; location.hash = id; S.parts = null; S.vb = null; S.vbFor = null; await reload();
  const st = S.state; S.marks = st.marks || []; S.measure = st.measure || null; S.ref = st.reference || null;
  S.design = Object.assign({ strokes: [], connections: [], erase: [], hide: [], copies: [] }, st.design || {});
  if (S.ref) { S.sel.target = S.ref.target_index; S.sel.match = S.ref.match_index; }
  $('#pid').textContent = id;
  if (st.size) { fit(...st.size); S.vbFor = 'logo'; } else if (st.picture) fit(...st.picture);
  setStep(step || Math.min(st.step || 1, 8)); listProjects();
}
async function listProjects() {
  const ps = await api('/api/projects');
  $('#projects').innerHTML = '<b>مشاريع سابقة</b>' + ps.map(p => `<a href="#${p.id}">${p.id}</a>`).join('');
}
window.addEventListener('hashchange', () => { const id = location.hash.slice(1); if (id && id !== S.id) openProject(id); });
document.querySelectorAll('#steps button[data-step]').forEach(b => b.onclick = () => setStep(+b.dataset.step));
(async () => {
  S.cats = await api('/api/categories'); listProjects();
  const id = location.hash.slice(1); if (id) { try { await openProject(id); return; } catch (e) {} }
  setStep(1);
})();
