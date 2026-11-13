"""Stage 3: understanding — parts, script, letters, and which parts are copies of each other.

No machine learning: every decision is a geometric rule on the measured line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .measure import Line, Piece, Stem, _cross, _runs

ROLE_AR = {"tooth": "سن", "lowered": "سن منخفض", "tall": "جذع صاعد", "short": "جذع قصير", "descending": "جذع نازل"}


@dataclass
class Join:
    right: int                 # index of the stem on the right
    left: int
    white: float
    bottom: float              # thickness of the join at the bottom of the counter
    kind: str = "join"         # join (two teeth) / bowl (a wide cup such as the body of ک)


@dataclass
class Tail:
    comp: int
    x0: float; x1: float; y0: float
    stem: Optional[int]        # descending stem it hangs from
    direction: str             # left / right / both


@dataclass
class Bar:
    x0: float; x1: float; y: float
    thick: float
    attached: bool


@dataclass
class Unit:
    """One recognised letter."""
    letter: str
    stems: List[int] = field(default_factory=list)
    marks: List[int] = field(default_factory=list)   # indexes into line.pieces
    sure: bool = True


@dataclass
class Reading:
    script: str                                  # arabic / latin / unknown
    joins: List[Join] = field(default_factory=list)
    tails: List[Tail] = field(default_factory=list)
    bars: List[Bar] = field(default_factory=list)
    units: List[Unit] = field(default_factory=list)
    text_guess: str = ""
    text_given: str = ""
    mark_side: Dict[int, str] = field(default_factory=dict)      # piece index -> above/below


# ------------------------------------------------------------------ parts
def _joins(L: Line) -> List[Join]:
    out = []
    for i, (a, b) in enumerate(zip(L.stems, L.stems[1:])):
        if a.comp != b.comp or a.x - b.x > max(6 * L.stem, 1.6 * L.x_height):
            continue
        mx = ((a.x - a.w / 2) + (b.x + b.w / 2)) / 2
        c = int(round(L.col_of_x(mx)))
        col = L.G[:, max(0, c - 1):c + 2].mean(axis=1)
        base_r = L.row_of_y(0)
        X = L.x_height
        bottom_ink = any(abs((r0 + n - 1) - base_r) <= 0.4 * L.stem for r0, n in _runs(col >= L.level))
        if not bottom_ink:
            continue                       # open at the bottom: a Latin n/m arch, not an Arabic join
        thick = float("nan")
        for r0, n in _runs(col >= L.level):
            if abs((r0 + n - 1) - base_r) <= 0.4 * L.stem:
                p, q = _cross(col, L.level, r0 + n // 2, -1), _cross(col, L.level, r0 + n // 2, +1)
                if p is not None and q is not None:
                    thick = q - p
        out.append(Join(right=i, left=i + 1, white=(a.x - a.w / 2) - (b.x + b.w / 2), bottom=thick,
                        kind="join" if a.x - b.x < max(3.2 * L.stem, 0.85 * L.x_height) else "bowl"))
    return out


def _tails(L: Line) -> List[Tail]:
    out = []
    lim = L.row_of_y(-0.3 * L.stem)
    for p in L.pieces:
        if p.kind != "body" or p.y0 > -0.3 * L.stem:
            continue
        m = L.lab == p.comp
        m[: int(np.ceil(lim))] = False
        if m.sum() < 0.3 * L.stem * L.stem:
            continue
        cols = np.flatnonzero(m.any(axis=0))
        x0, x1 = L.x_of_col(cols[0]), L.x_of_col(cols[-1] + 1)
        rows = np.flatnonzero(m.any(axis=1))
        y0 = L.y_of_row(rows[-1] + 1)
        stem_i = None
        for i, s in enumerate(L.stems):
            if s.comp == p.comp and s.role == "descending" and x0 - L.stem <= s.x <= x1 + L.stem:
                stem_i = i
        ref = L.stems[stem_i].x if stem_i is not None else (x0 + x1) / 2
        cx = float(np.mean(np.nonzero(m)[1])) - L.off
        direction = "left" if cx < ref - 0.3 * L.stem else ("right" if cx > ref + 0.3 * L.stem else "both")
        out.append(Tail(comp=p.comp, x0=x0, x1=x1, y0=y0, stem=stem_i, direction=direction))
    return out


def _bars(L: Line) -> List[Bar]:
    """Thin horizontal strokes above the x-height (the lines of گ, flags)."""
    out: List[Bar] = []
    for p in L.dashes:
        if p.cy > L.x_height * 0.9:
            out.append(Bar(x0=p.x0, x1=p.x1, y=p.cy, thick=p.thick, attached=False))
    lim = L.hair * 1.9 if L.hair == L.hair else 0.3 * L.stem
    body = np.isin(L.lab, [p.comp for p in L.pieces if p.kind == "body"])
    seg: Dict[int, List[Tuple[int, float]]] = {}
    for c in range(body.shape[1]):
        for r0, n in _runs(body[:, c]):
            y = L.y_of_row(r0 + n / 2)
            if n <= lim and y > L.x_height + 0.15 * L.stem:
                key = int(round(y / max(1.0, L.hair)))
                seg.setdefault(key, []).append((c, float(n)))
    for key, cols in seg.items():
        cols.sort()
        run = [cols[0]]
        for c in cols[1:] + [(10 ** 9, 0.0)]:
            if c[0] - run[-1][0] <= 1:
                run.append(c)
                continue
            if len(run) >= 1.2 * L.stem:
                out.append(Bar(x0=L.x_of_col(run[0][0]), x1=L.x_of_col(run[-1][0] + 1), y=key * max(1.0, L.hair),
                               thick=float(np.median([n for _, n in run])), attached=True))
            run = [c]
    out.sort(key=lambda b: -b.y)
    return out


def _script(L: Line, joins: List[Join]) -> str:
    if not L.stems:
        return "unknown"
    joined = len(joins) / max(1, len(L.stems) - 1)
    marks_below = any(p.cy < -0.2 * L.stem for p in L.pieces if p.kind in ("dot", "dash"))
    bodies = sum(p.kind == "body" for p in L.pieces)
    score = joined + (0.3 if marks_below else 0) - (0.4 if bodies >= 0.7 * len(L.stems) else 0)
    return "arabic" if score >= 0.35 else "latin"


# ------------------------------------------------------------------ marks
def _assign_marks(L: Line) -> Dict[int, Tuple[int, str]]:
    """Each dot/dash goes to the nearest stem (by x); side = above / below."""
    out = {}
    for i, p in enumerate(L.pieces):
        if p.kind not in ("dot", "dash") or not L.stems:
            continue
        if p.kind == "dash" and p.cy > L.x_height * 0.9 and p.size > 2.2 * L.stem:
            continue          # a long upper line is a bar of گ, not a pair of dots
        side = "above" if p.cy > L.x_height / 2 else "below"
        j = int(np.argmin([abs(s.x - p.cx) for s in L.stems]))
        out[i] = (j, side)
    return out


def _mark_code(L: Line, idxs: List[int], marks: Dict[int, Tuple[int, str]]) -> str:
    """'a1', 'b2' ... from the marks attached to a group of stems. A hairline dash counts as two dots."""
    up = sum(2 if L.pieces[i].kind == "dash" else 1 for i in idxs if marks[i][1] == "above")
    down = sum(2 if L.pieces[i].kind == "dash" else 1 for i in idxs if marks[i][1] == "below")
    return ("a%d" % up if up else "") + ("b%d" % down if down else "")


# ------------------------------------------------------------------ recognition
BEH_BY_MARK = {"b1": "ب", "a1": "ن", "a2": "ت", "a3": "ث", "b2": "ي", "b3": "پ", "": "ٮ"}


def _recognise(L: Line, R: Reading) -> None:
    marks = _assign_marks(L)
    by_stem: Dict[int, List[int]] = {}
    for i, (j, side) in marks.items():
        by_stem.setdefault(j, []).append(i)
    tails_by_stem = {t.stem: t for t in R.tails if t.stem is not None}
    loops_by_comp: Dict[int, List] = {}
    for c in L.counters:
        loops_by_comp.setdefault(c.comp, []).append(c)
    bars_upper = [b for b in R.bars]
    joined_pairs = {(j.right, j.left) for j in R.joins}
    # group stems into connected runs (letters joined at the baseline), right to left
    groups: List[List[int]] = []
    for i in range(len(L.stems)):
        if groups and (i - 1, i) in joined_pairs:
            groups[-1].append(i)
        else:
            groups.append([i])
    units: List[Unit] = []
    for g in groups:
        k = 0
        while k < len(g):
            i = g[k]
            s = L.stems[i]
            last = k == len(g) - 1
            mk = by_stem.get(i, [])
            code = _mark_code(L, mk, marks)
            if s.role == "descending":
                t = tails_by_stem.get(i)
                letter = "ز" if code.startswith("a1") else "ر"
                units.append(Unit(letter, [i], mk, sure=t is not None and t.direction == "left"))
                k += 1
                continue
            if s.role == "tall":
                units.append(Unit("ا" if last and k > 0 or len(g) == 1 else "ل", [i], mk))
                k += 1
                continue
            # two stems closing a group under one or two upper bars: ک / گ
            bars_here = [b for b in bars_upper if b.x0 - L.stem <= s.x <= b.x1 + 2 * L.stem]
            if k == len(g) - 2 and bars_here:
                units.append(Unit("گ" if len(bars_here) >= 2 else "ک", g[k:], mk + by_stem.get(g[k + 1], []), sure=True))
                k += 2
                continue
            # three teeth without dots on the first two = س / ش
            if k + 2 < len(g) + 0 and k + 2 <= len(g) - 1:
                trio = g[k:k + 3]
                trio_marks = sum((by_stem.get(j, []) for j in trio), [])
                codes = _mark_code(L, trio_marks, marks)
                lows = sum(L.stems[j].role == "lowered" for j in trio)
                if all(L.stems[j].role in ("tooth", "lowered") for j in trio) and (codes in ("", "a3") and (lows >= 1 or not trio_marks)):
                    units.append(Unit("ش" if codes == "a3" else "س", trio, trio_marks))
                    k += 3
                    continue
            units.append(Unit(BEH_BY_MARK.get(code, "ٮ"), [i], mk, sure=code in BEH_BY_MARK and code != ""))
            k += 1
    R.units = units
    R.text_guess = "".join(u.letter for u in units)


def align_text(L: Line, R: Reading, text: str) -> None:
    """When the user types the word that is in the logo, label every stem with its letter by
    aligning the stems the letter engine expects with the stems that were found."""
    from .style import Style
    from .typeset import typeset
    st = Style(stem=L.stem, x_height=L.x_height, white=L.white if L.white == L.white else L.stem,
               ascender=max(L.ascender, 0.3 * L.x_height), descender=max(L.descender, 0.3 * L.x_height), slant=0)
    lay = typeset(text, st)
    # expected stems with their place along the word (0 = right end, 1 = left end)
    exp = []
    for p in lay.placed:
        for xc, top, bottom in sorted(p.glyph.stems, key=lambda v: -v[0]):
            role = "tall" if top > 1.2 * st.x_height else ("tooth" if top > 0.7 * st.x_height else "short")
            exp.append((p.char, role, p.x + xc))
    if not exp or not L.stems:
        R.text_given = text
        return
    ex0, ex1 = max(e[2] for e in exp), min(e[2] for e in exp)
    fx0, fx1 = max(s.x for s in L.stems), min(s.x for s in L.stems)
    epos = [(ex0 - e[2]) / max(1e-6, ex0 - ex1) for e in exp]
    fpos = [(fx0 - s.x) / max(1e-6, fx0 - fx1) for s in L.stems]
    expected = [(e[0], e[1]) for e in exp]
    found = [s.role for s in L.stems]
    n, m = len(expected), len(found)
    INF = 1e9
    D = np.full((n + 1, m + 1), INF)
    D[0, :] = np.arange(m + 1)
    D[:, 0] = np.arange(n + 1)

    def cost(i, j):
        e, f = expected[i][1], found[j]
        c = 0.0 if (e == f or (e == "tooth" and f == "lowered") or (e == "tall" and f in ("tooth", "descending"))
                    or (f == "descending")) else 0.6
        return c + 1.5 * abs(epos[i] - fpos[j])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + cost(i - 1, j - 1), D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j = n, m
    while i > 0 and j > 0:
        if abs(D[i, j] - (D[i - 1, j - 1] + cost(i - 1, j - 1))) < 1e-9:
            L.stems[j - 1].letter = expected[i - 1][0]
            i, j = i - 1, j - 1
        elif abs(D[i, j] - (D[i - 1, j] + 1)) < 1e-9:
            i -= 1
        else:
            j -= 1
    R.text_given = text
    units: List[Unit] = []
    for k, s in enumerate(L.stems):
        if units and units[-1].letter == s.letter and s.letter and units[-1].stems[-1] == k - 1 and _same_letter_run(text, units, s.letter):
            units[-1].stems.append(k)
        else:
            units.append(Unit(s.letter or "?", [k]))
    marks = _assign_marks(L)
    for i, (j, side) in marks.items():
        for u in units:
            if j in u.stems:
                u.marks.append(i)
    R.units = units


def _same_letter_run(text: str, units: List[Unit], letter: str) -> bool:
    """Teeth of the same letter (س has three) stay together; two different ب ب do not."""
    return letter in "سشصضگکك" or letter in "ط"


def read_line(L: Line, text: Optional[str] = None) -> Reading:
    joins = _joins(L)
    R = Reading(script=_script(L, joins), joins=joins)
    joined = {(j.right, j.left) for j in joins}
    for s in L.stems:
        s.joined_left = s.joined_right = False
    for a, b in joined:
        L.stems[a].joined_left = True
        L.stems[b].joined_right = True
    R.tails = _tails(L)
    R.bars = _bars(L)
    for i, (j, side) in _assign_marks(L).items():
        R.mark_side[i] = side
    if R.script == "arabic":
        if text:
            align_text(L, R, text)
        else:
            _recognise(L, R)
    return R


# ------------------------------------------------------------------ the observations
def _close(a: float, b: float, tol: float = 0.06) -> bool:
    return a == a and b == b and a > 0 and b > 0 and abs(a / b - 1) <= tol


def _scaled_line(L: Line, k: float):
    """A light copy of the measured numbers in original-image pixels (k = 1 / working scale)."""
    from types import SimpleNamespace
    sc = lambda v: v * k
    return SimpleNamespace(
        idx=L.idx, slant=L.slant, monoline=L.monoline, power=L.power,
        pen_fit={**L.pen_fit, "median_error": L.pen_fit.get("median_error", 0) * k} if L.pen_fit else {},
        stem=sc(L.stem), hair=sc(L.hair), white=sc(L.white), x_height=sc(L.x_height), ascender=sc(L.ascender),
        descender=sc(L.descender), width=sc(L.width),
        stems=[SimpleNamespace(w=sc(s.w)) for s in L.stems],
        levels=[SimpleNamespace(y=sc(v.y), n=v.n, name=v.name) for v in L.levels],
        dots=[SimpleNamespace(size=sc(p.size), y1=sc(p.y1)) for p in L.dots],
        dashes=[SimpleNamespace(size=sc(p.size), thick=sc(p.thick)) for p in L.dashes],
        counters=[SimpleNamespace(x0=sc(c.x0), x1=sc(c.x1), y0=sc(c.y0), y1=sc(c.y1), area=c.area * k * k) for c in L.counters])


def observations(lines: List[Line], readings: List[Reading], k: float = 1.0) -> List[Dict]:
    """Rules read off the numbers, each with its evidence. kind: pen / grid / rhythm / part / match."""
    out: List[Dict] = []

    def add(kind: str, text: str, **ev):
        out.append({"kind": kind, "text": text, "evidence": ev})

    from copy import copy
    lines = [_scaled_line(L, k) for L in lines]
    for L, R in zip(lines, readings):
        if not L.stems:
            continue
        tag = f"السطر {L.idx + 1}"
        ws = [s.w for s in L.stems]
        same = sum(abs(w / L.stem - 1) <= 0.05 for w in ws)
        add("pen", f"{tag}: {same} من {len(ws)} جذوع بنفس السمك ({L.stem:.2f}).", stem=L.stem, widths=ws)
        if not L.monoline:
            add("pen", f"{tag}: قلمان فقط — جذع {L.stem:.2f} وخط شعري {L.hair:.2f}، والتباين {L.stem / L.hair:.1f} : 1.",
                stem=L.stem, hair=L.hair)
            if L.pen_fit:
                add("pen", f"{tag}: السمك يتبع اتجاه الخط بقاعدة: شعري + (جذع − شعري) × |جيب الزاوية|^{L.power:.2f} (خطأ وسيط {L.pen_fit['median_error']:.2f}).",
                    **L.pen_fit)
        lows = [v for v in L.levels if v.name == "lowered"]
        for v in lows:
            add("grid", f"{tag}: {v.n} أسنان أخفض من ارتفاع الحروف بـ{L.x_height - v.y:.1f}.", drop=L.x_height - v.y, count=v.n)
        if _close(L.white, L.stem, 0.15):
            add("rhythm", f"{tag}: الأبيض بين الجذوع ({L.white:.2f}) يساوي الجذع تقريبًا — إيقاع 1:1.", white=L.white, stem=L.stem)
        if _close(L.descender, L.stem, 0.1):
            add("grid", f"{tag}: النزول ({L.descender:.1f}) يساوي سمك الجذع.", descender=L.descender, stem=L.stem)
        for p in L.dots:
            add("part", f"{tag}: النقطة {'مربعة' if getattr(p, 'squareness', 0.0) > 0.9 else 'دائرة'} ⌀{p.size:.2f} = {p.size / L.stem:.2f} من الجذع، وقمتها عند {p.y1 - L.x_height:+.1f} من ارتفاع الحروف.",
                diameter=p.size)
        for p in L.dashes:
            if _close(p.thick, L.hair, 0.25):
                add("part", f"{tag}: شرطة {p.size:.1f} × {p.thick:.2f} بسمك الخط الشعري — بديل عن نقطتين أو علامة.", length=p.size, thick=p.thick)
        for b in R.bars:
            pass
        if len(L.counters) >= 2:
            c = L.counters
            for a in range(len(c)):
                for b in range(a + 1, len(c)):
                    if _close(c[a].area, c[b].area, 0.05):
                        add("match", f"{tag}: فراغان مغلقان متطابقان ({c[a].x1 - c[a].x0:.1f}×{c[a].y1 - c[a].y0:.1f}) — نفس الكأس مستعمل مرتين.")
    if len(lines) > 1:
        A = [L for L in lines if L.stems]
        if len(A) > 1:
            pairs = [("stem", "سمك الجذع"), ("x_height", "ارتفاع الحروف"), ("ascender", "الصعود"), ("descender", "النزول"),
                     ("width", "العرض")]
            for attr, name in pairs:
                vals = [getattr(L, attr) for L in A]
                if all(_close(v, vals[0], 0.04) for v in vals):
                    add("match", f"{name} متساوٍ في كل الأسطر ({' / '.join(f'{v:.1f}' for v in vals)}).", values=vals)
            if all(abs(L.slant - A[0].slant) <= 0.6 for L in A):
                add("match", f"الميل نفسه في كل الأسطر ({' / '.join(f'{L.slant:.1f}°' for L in A)}).")
            ds = [[p.size for p in L.dots] for L in A]
            if all(ds) and all(_close(d[0], ds[0][0], 0.04) for d in ds):
                add("match", f"النقطة نفسها في كل الأسطر (⌀{' / '.join(f'{d[0]:.1f}' for d in ds)}).")
            hs = [L.hair for L in A if not L.monoline]
            dash_t = [p.thick for L in A for p in L.dashes]
            if hs and dash_t and _close(float(np.median(dash_t)), float(np.median(hs)), 0.25):
                add("match", "الشرطات بسمك الخط الشعري في الأسطر الأخرى — نفس القلم الرفيع.")
    return out
