"""The rules of a professional Arabic typeface (khatt.anatomy.RULES), measured on what the engine draws
in a style: each rule passed or not, by how much, and what is missing.

A typeface a foundry ships (see anatomy: iwantype, Glyphs, Microsoft, W3C) draws every letter in each of
its forms with لا, the marks, digits and punctuation; keeps its levels (teeth, heads, ascenders) at one
height; draws one stroke; joins every letter on one joining line as thick as the stroke; lengthens a
join for the kashida; keeps dots one size and clear of the letters, counters open, each letter unlike
its neighbours, the white between letters even and the ink even along a line. Each rule is measured
here the way it can be measured on the engine's drawing; what cannot be drawn yet is listed, not hidden.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import anatomy
from .style import Style

FORMS = {"i": "{}", "I": "{}ـ", "M": "ـ{}ـ", "F": "ـ{}"}
FORM_AR = {"i": "منفصلة", "I": "أولى", "M": "وسطى", "F": "أخيرة"}
# the letters whose tops make each level (khatt.anatomy.PARTS: ascender, head, tooth). Measured on eight
# typefaces (medians of their tops, in x-heights): ascenders 1.91 and one level (spread 0.04); joined
# teeth 1.00 and one level (0.00; only Naskh's contextual teeth differ); heads are two levels, not one:
# the head of ف ق raised on its neck (1.20-1.27) and the head of م و on the line (1.02-1.09), each
# within 0.07 (29LT: Arabic has several levels instead of one x-height)
LEVEL_LETTERS = {
    "الصاعد": [("ا", "i"), ("ا", "F"), ("ل", "I"), ("ل", "M"), ("ك", "I"), ("ط", "i"), ("ط", "I")],
    "الرأس على العنق": [("ف", "I"), ("ف", "M"), ("ق", "I"), ("ق", "M")],
    "الرأس على السطر": [("م", "I"), ("م", "M"), ("و", "i"), ("و", "F")],
    "السن": [("ب", "I"), ("ن", "I"), ("ي", "I"), ("ت", "M"), ("ث", "M")],
}
LEVEL_TOL = {"الصاعد": 0.05, "الرأس على العنق": 0.10, "الرأس على السطر": 0.10, "السن": 0.05}   # of the x-height
JOIN_WORDS = ["بسم", "محمد", "سلام", "كتاب", "عين", "فهم", "ليلة", "شمس", "نجح", "صبر", "طبع", "قلم", "هيكل", "مكة"]
GAP_WORDS = ["دار", "ورد", "زاد", "ودود", "أراد", "وزارة", "دروس"]
PANGRAM = "نص حكيم له سر قاطع وذو شأن عظيم مكتوب على ثوب أخضر ومغلف بجلد أزرق"
# how even eight open typefaces are (Noto Naskh, Amiri, Vazirmatn, Noto Kufi, Reem Kufi, Cairo, Tajawal,
# Kufam), measured the same way: the white between letters that do not join varies by 27-64% (median
# 43%), the ink along the pangram in windows of an x-height by 30-41% (median 32%). A style passes when
# it is as even as the median typeface.
SPACING_CV, COLOR_CV = 0.43, 0.32
# how often a real typeface's letter is read as itself among another typeface's letters (the bench's 53
# styles, each read against 8 typefaces of other families): the reader's own limit, the yardstick of
# identity. ص is read as س by it in half the typefaces: the engine is not asked to do better than that.
REAL_READ = {"ا.i": 1.0, "ا.F": 0.95, "ب.i": 0.55, "ب.I": 0.82, "ب.M": 0.92, "ب.F": 0.56, "ح.i": 0.38, "ح.I": 0.43, "ح.M": 0.38, "ح.F": 0.33, "د.i": 0.62, "د.F": 0.61, "ر.i": 0.88, "ر.F": 0.81, "س.i": 0.64, "س.I": 0.79, "س.M": 0.74, "س.F": 0.65, "ص.i": 0.5, "ص.I": 0.52, "ص.M": 0.54, "ص.F": 0.51, "ط.i": 0.78, "ط.I": 0.68, "ط.M": 0.68, "ط.F": 0.83, "ع.i": 0.43, "ع.I": 0.39, "ع.M": 0.33, "ع.F": 0.43, "ف.i": 0.58, "ف.I": 0.42, "ف.M": 0.48, "ف.F": 0.68, "ق.i": 0.53, "ق.I": 0.44, "ق.M": 0.48, "ق.F": 0.65, "ك.i": 0.63, "ك.I": 0.43, "ك.M": 0.42, "ك.F": 0.61, "ل.i": 0.65, "ل.I": 0.79, "ل.M": 0.87, "ل.F": 0.7, "م.i": 0.7, "م.I": 0.65, "م.M": 0.62, "م.F": 0.67, "ن.i": 0.52, "ن.I": 0.79, "ن.M": 0.87, "ن.F": 0.54, "ه.i": 0.71, "ه.I": 0.56, "ه.M": 0.54, "ه.F": 0.87, "و.i": 0.82, "و.F": 0.88, "ي.i": 0.44, "ي.I": 0.86, "ي.M": 0.84, "ي.F": 0.56}


def _rule(key: str, ok: bool, measure: str, detail: Optional[List[str]] = None) -> Dict:
    ar = next((a for k, a, _ in anatomy.RULES if k == key), key)
    what = next((w for k, _, w in anatomy.RULES if k == key), "")
    return {"rule": key, "name": ar, "what": what, "ok": bool(ok), "measure": measure, "detail": detail or []}


def _is_mark(sh, st: Style) -> bool:
    from .typeset import _is_mark as im
    return im(sh, st)


def _body_of(glyph, st: Style):
    """The letter's body without its marks, as one geometry in the glyph's own coordinates."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .typeset import flat
    acc = None
    for sh in flat(glyph.shapes):
        if _is_mark(sh, st):
            continue
        part = None
        for ring in sh:
            if len(ring) >= 3:
                q = Polygon(ring).buffer(0)
                part = q if part is None else part.symmetric_difference(q)
        if part is not None and not part.is_empty:
            acc = part if acc is None else acc.union(part)
    return acc


def _glyph(c: str, form: str, st: Style):
    from .typeset import typeset
    lay = typeset(FORMS[form].format(c), st)
    got = [p for p in lay.placed if p.char == c]
    return got[0].glyph if got and not lay.skipped else None


def _ink(lay, st: Style, marks: bool = False):
    """The drawn word's ink as the layout holds it, letters and the joins between them, upright (the
    slant is the renderer's), marks left out unless asked: by name, and any piece wholly above the
    teeth or under the baseline (a block layout keeps its dots in one piece of ink)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .typeset import flat
    parts = []
    for sh in flat(lay.shapes):
        if not marks and _is_mark(sh, st):
            continue
        acc = None
        for ring in sh:
            if len(ring) >= 3:
                q = Polygon(ring).buffer(0)
                acc = q if acc is None else acc.symmetric_difference(q)
        if acc is not None and not acc.is_empty:
            parts.append(acc)
    if not parts:
        return None
    ink = unary_union(parts)
    if not marks:
        X = st.x_height
        keep = [q for q in getattr(ink, "geoms", [ink]) if not (q.bounds[1] > 1.02 * X or q.bounds[3] < -0.05 * X)]
        ink = unary_union(keep) if keep else None
    return ink


def _runs(lay) -> List[List[int]]:
    """The letters of a word in runs that join: a run ends at a letter that does not join the next."""
    runs, cur = [], []
    for i, p in enumerate(lay.placed):
        cur.append(i)
        if p.form in ("i", "F") or i == len(lay.placed) - 1:
            runs.append(cur); cur = []
    return runs


# ---------------------------------------------------------------- the rules
def coverage(st: Style) -> Dict:
    from .typeset import typeset
    cs = anatomy.charset("extended")
    core = anatomy.ARABIC_CORE + "ٱ"
    missing, wrong_forms = [], []
    for c in core:
        g = _glyph(c, "i", st)
        if g is None:
            missing.append(c)
            continue
        for f in anatomy.forms(c):
            if _glyph(c, f, st) is None:
                wrong_forms.append(f"{c}.{f}")
    ext = [c for c in anatomy.PERSIAN + anatomy.URDU + anatomy.KURDISH if _glyph(c, "i", st) is None]
    digits = [d for d in cs["digits"] if typeset(d, st).skipped or not typeset(d, st).placed]
    punct = [d for d in cs["punctuation"] if typeset(d, st).skipped or not typeset(d, st).placed]
    lay = typeset("بَ", st)
    harakat = not any(len(sh) for p in lay.placed for sh in p.glyph.shapes if "حركة" in (getattr(sh, "name", "") or ""))
    ok = not missing and not wrong_forms and not digits and not punct and not harakat
    measure = (f"الحروف الأساسية {len(core) - len(missing)}/{len(core)} بكل صورها"
               + (f" (ناقصة صور: {' '.join(wrong_forms)})" if wrong_forms else "")
               + f"؛ الفارسية والأردية والكردية {18 - len(ext)}/18؛ الأرقام {20 - len(digits)}/20؛"
               + f" علامات الترقيم {len(cs['punctuation']) - len(punct)}/{len(cs['punctuation'])}؛ الحركات "
               + ("غير مرسومة" if harakat else "مرسومة"))
    detail = ([f"حروف لا يرسمها: {' '.join(missing)}"] if missing else []) + \
             ([f"حروف موسعة لا يرسمها: {' '.join(ext)}"] if ext else [])
    return _rule("coverage", ok, measure, detail)


def levels(st: Style) -> Dict:
    X = st.x_height
    out, ok = [], True
    for name, letters in LEVEL_LETTERS.items():
        tops = {}
        for c, f in letters:
            g = _glyph(c, f, st)
            b = _body_of(g, st) if g is not None else None
            if b is not None:
                tops[f"{c}.{f}"] = b.bounds[3]
        if len(tops) < 2:
            continue
        spread = (max(tops.values()) - min(tops.values())) / X
        good = spread <= LEVEL_TOL[name]
        ok &= good
        hi, lo = max(tops, key=tops.get), min(tops, key=tops.get)
        out.append(f"{name}: تفاوت {spread:.2f} من ارتفاع الحروف (الحد {LEVEL_TOL[name]:.2f})"
                   + ("" if good else f"، الأعلى {hi} والأدنى {lo}"))
    return _rule("levels", ok, "؛ ".join(out))


def _defects(st: Style) -> List:
    from .critic import specimen_defects
    return specimen_defects(st)


def stroke(st: Style, defects: List) -> Dict:
    # a block logo's slits between its blocks are its design (blocks.py: «slit»), not a pen's seam
    kinds = ("thin", "blob") if st.construction == "block" else ("thin", "blob", "crack")
    bad = [d for d in defects if d.kind in kinds]
    note = "؛ الشقوق بين الكتل من تصميم الشعار فلا تُعد" if st.construction == "block" else ""
    return _rule("stroke", not bad, f"عيوب القلم في الحروف كلها بصورها: {len(bad)}{note}", [d.text() for d in bad[:12]])


def joins(st: Style) -> Dict:
    """Every run of joined letters is one piece, and every join sits on the baseline as thick as the
    hairline (the connection is the stroke on the joining line)."""
    from shapely.geometry import LineString
    from .typeset import typeset
    broken, off, n_joins = [], [], 0
    for w in JOIN_WORDS:
        lay = typeset(w, st)
        if lay.skipped:
            continue
        ink = _ink(lay, st)
        if ink is None:
            continue
        pieces = len(getattr(ink, "geoms", [ink]))
        if pieces > len(_runs(lay)):
            broken.append(w)
        for x0, x1 in lay.joins:
            if x1 - x0 < 0.3 * st.hair:
                continue
            n_joins += 1
            x = 0.5 * (x0 + x1)
            cut = ink.intersection(LineString([(x, -st.x_height), (x, 0.6 * st.x_height)]))
            segs = [s for s in getattr(cut, "geoms", [cut]) if not s.is_empty and s.length > 0]
            if not segs:
                off.append(f"{w}: وصلة بلا حبر"); continue
            # the ink at the baseline (a bowl or a tail may pass under a join)
            low = min(segs, key=lambda s: abs(s.bounds[1]))
            y0, y1 = low.bounds[1], low.bounds[3]
            if abs(y0) > 0.05 * st.x_height or not (0.7 * st.hair <= y1 - y0 <= 1.45 * max(st.hair, 1e-6)):
                off.append(f"{w}: وصلة من {y0:.1f} إلى {y1:.1f} (الخط الشعري {st.hair:.1f})")
    ok = not broken and not off
    line = (f"{n_joins - len(off)}/{n_joins} وصلة على السطر بسمك الخط الشعري" if n_joins else
            "الوصلات أحزمة من الكتلة نفسها (تقاس في blocks.py)")
    return _rule("joins", ok, f"{len(JOIN_WORDS) - len(broken)}/{len(JOIN_WORDS)} كلمة موصولة بلا انقطاع؛ {line}",
                 ([f"منقطعة: {' '.join(broken)}"] if broken else []) + off[:8])


def kashida(st: Style) -> Dict:
    from .typeset import typeset
    w, key = "كتاب", "ك.I"
    if st.construction == "block":
        plain = typeset(w, st)
        long = typeset(w, Style(**{**st.__dict__, "block": {**(st.block or {}), "kashida": st.pitch}}))
    else:
        plain = typeset(w, st)
        long = typeset(w, Style(**{**st.__dict__, "stretch": {**st.stretch, key: st.pitch}}))
    a, b = _ink(plain, st), _ink(long, long.style)
    if a is None or b is None:
        return _rule("kashida", False, "لم تُرسم الكلمة")
    grew = (b.bounds[2] - b.bounds[0]) - (a.bounds[2] - a.bounds[0])
    whole = len(getattr(b, "geoms", [b])) <= len(_runs(long))
    ok = grew > 0.7 * st.pitch and whole
    return _rule("kashida", ok, f"مدّ وصلة «{w}» بخطوة: طالت {grew / st.pitch:.2f} خطوة"
                                + ("، والكلمة موصولة" if whole else "، وانقطعت الكلمة"))


def lam_alef(st: Style) -> Dict:
    from .typeset import typeset
    out, ok = [], True
    for la in anatomy.LAM_ALEFS:
        lay = typeset(la, st)
        one = len(lay.placed) == 1
        ink = _ink(lay, st)
        joined = ink is not None and len(getattr(ink, "geoms", [ink])) == 1
        marks_ok = True
        if la != "لا":
            withm = _ink(lay, st, marks=True)
            marks_ok = withm is not None and ink is not None and withm.area > ink.area * 1.02
        good = (one or joined) and marks_ok
        ok &= good
        out.append(f"{la}: " + ("حرف واحد" if one else "ل وا موصولتان (صورة U)" if joined else "حرفان منفصلان")
                   + ("" if marks_ok else "، بلا همزتها أو مدتها"))
    return _rule("lam-alef", ok, "؛ ".join(out),
                 ["الصورة المتقاطعة لـ«لا» (نصف الخطوط المفتوحة المقيسة) بديل لم يُرسم بعد"] if ok else [])


def marks(st: Style, defects: List) -> Dict:
    """The single dot of ب ج خ ذ ز ض ظ غ ف ن, wherever it is drawn: one size (a block logo may draw
    two dots as one bar, so only single dots are compared), and no mark touching its letter."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .typeset import flat
    X = st.x_height
    sizes = []
    for c in "بجخذزضظغفن":
        for f in anatomy.forms(c):
            g = _glyph(c, f, st)
            if g is None:
                continue
            pieces = []
            for sh in flat(g.shapes):
                acc = None
                for ring in sh:
                    if len(ring) >= 3:
                        q = Polygon(ring).buffer(0)
                        acc = q if acc is None else acc.symmetric_difference(q)
                if acc is not None and not acc.is_empty:
                    pieces.append(acc)
            ink = unary_union(pieces) if pieces else None
            if ink is None:
                continue
            for q in getattr(ink, "geoms", [ink]):
                if q.bounds[1] > 1.02 * X or q.bounds[3] < -0.05 * X:
                    x0, y0, x1, y1 = q.bounds
                    sizes.append(max(x1 - x0, y1 - y0))
    touch = [d for d in defects if d.kind == "touch"]
    if not sizes:
        return _rule("marks", False, "لم توجد نقطة منفصلة عن حرفها", [d.text() for d in touch[:8]])
    spread = max(sizes) / min(sizes) - 1.0
    ok = spread <= 0.1 and not touch
    return _rule("marks", ok, f"{len(sizes)} نقطة مفردة، تفاوت أحجامها {spread:.0%}؛ نقاط تلمس الحروف: {len(touch)}",
                 [d.text() for d in touch[:8]])


def counters(st: Style, defects: List) -> Dict:
    bad = [d for d in defects if d.kind == "counter"]
    return _rule("counters", not bad, f"فراغات داخلية أضيق من البياض: {len(bad)}", [d.text() for d in bad[:8]])


def identity(st: Style, defects: List, fonts: bool = True) -> Dict:
    conf = [d for d in defects if d.kind == "confusable"]
    detail = [d.text() for d in conf]
    measure = f"أزواج متشابهة داخل الخط: {len(conf)}"
    ok = not conf
    if fonts:
        from .legible import reads_as_real
        from .reference import faces
        if [f for f in faces() if "vazirmatn-400" in f]:
            from .critic import SKELETONS
            right, wrong = 0, []
            up = Style(**{**st.__dict__, "slant": 0.0})
            for c in SKELETONS:
                for f in anatomy.forms(c):
                    g = _glyph(c, f, up)
                    if g is None:
                        continue
                    got = reads_as_real(g, f, up)
                    if got is None or got == c or anatomy.same_body(got, c, f):
                        right += 1
                    else:
                        wrong.append(f"{c}.{f}←{got}")
            n = right + len(wrong)
            # as real typefaces are read by the same reader: the count they would read right, less two
            expect = sum(REAL_READ.get(f"{'ي' if c == 'ی' else c}.{f}", 0.6) for c in SKELETONS for f in anatomy.forms(c))
            measure += f"؛ تُقرأ بين حروف خطوط حقيقية {right}/{n} (الخطوط الحقيقية نفسها نحو {expect:.0f}/{n})"
            ok &= right >= expect - 2
            detail += [f"تُقرأ حرفًا آخر: {' '.join(wrong)}"] if wrong else []
    return _rule("identity", ok, measure, detail)


def spacing(st: Style) -> Dict:
    """The white between letters that do not join, word by word: as even as the white between teeth."""
    from .typeset import typeset
    from shapely.geometry import box
    X = st.x_height
    band = box(-1e6, 0.2 * X, 1e6, 0.9 * X)       # the letters' bodies, tails under the line left out
    gaps = []
    for w in GAP_WORDS:
        lay = typeset(w, st)
        if lay.skipped:
            continue
        spans = []
        for run in _runs(lay):
            xs = []
            for i in run:
                b = _body_of(lay.placed[i].glyph, st)
                if b is not None:
                    b = b.intersection(box(-1e6, band.bounds[1], 1e6, band.bounds[3]))
                    if not b.is_empty:
                        xs += [lay.placed[i].x + b.bounds[0], lay.placed[i].x + b.bounds[2]]
            if xs:
                spans.append((min(xs), max(xs)))
        for (l0, l1), (r0, r1) in zip(spans[1:], spans[:-1]):   # right to left: each run left of the one before
            gaps.append(r0 - l1)
    if not gaps:
        return _rule("spacing", False, "لا كلمات")
    g = np.array(gaps)
    head = f"البياض بين الحروف المنفصلة: متوسطه {g.mean() / st.white:.2f} من البياض، أدناه {g.min() / st.white:.2f}"
    if g.mean() <= 0.2 * st.white:
        return _rule("spacing", False, head + "؛ الحروف تتلامس (تصميم شعار، لا خط نص)")
    cv = float(g.std() / g.mean())
    ok = cv <= SPACING_CV and g.min() > 0.2 * st.white
    return _rule("spacing", ok, head + f"، تفاوته {cv:.0%} (الخط الحقيقي الوسيط {SPACING_CV:.0%})")


def color(st: Style) -> Dict:
    """The ink along a line, in windows of an x-height (word spaces left out): how even it is."""
    from .critic import raster
    from .typeset import typeset
    lay = typeset(PANGRAM, st)
    body, s, _, _ = raster(lay, px_per_stem=6.0, frame_marks=False)
    cols = body.sum(axis=0).astype(float)
    win = max(2, int(round(st.x_height * s)))
    vals = [cols[i:i + win].sum() for i in range(0, len(cols) - win, win)]
    vals = np.array([v for v in vals if v > 0.15 * max(vals)]) if vals else np.array([])
    if not len(vals):
        return _rule("color", False, "لا حبر")
    cv = float(vals.std() / vals.mean())
    ok = cv <= COLOR_CV
    return _rule("color", ok, f"تفاوت الحبر على سطر كامل في نوافذ بارتفاع الحروف: {cv:.0%} (الخط الحقيقي الوسيط {COLOR_CV:.0%})")


def exam(st: Style, fonts: bool = True) -> List[Dict]:
    """Every rule of a professional typeface, measured on the style."""
    d = _defects(st)
    return [coverage(st), levels(st), stroke(st, d), joins(st), kashida(st), lam_alef(st), marks(st, d),
            counters(st, d), identity(st, d, fonts), spacing(st), color(st)]


def report(results: List[Dict]) -> str:
    lines = [f"قواعد الخط الاحترافي: {sum(r['ok'] for r in results)}/{len(results)} محققة"]
    for r in results:
        lines.append(f"{'✓' if r['ok'] else '✗'} {r['name']}: {r['measure']}")
        lines += [f"    {x}" for x in r["detail"]]
    return "\n".join(lines)
