"""Arabic shaping and right-to-left layout."""
from __future__ import annotations

import numpy as np

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import geometry as g
from .letters import JOIN, LETTERS, Glyph, top_joins
from .library import Compound, Tagged
from .recipes import BUILDING, PARAMS, READS, REFERENCE, SKELETON, siblings, stretch_of, value
from .style import Style

TOOTH_END = set("بتثپنيیئسشصضطظلكکگ")     # letters whose joining forms end with a tooth or a stem on the left
HARAKAT = {chr(c) for c in range(0x064B, 0x0653)} | {"ٰ"}


@dataclass
class Placed:
    char: str
    form: str
    x: float          # where the glyph's right edge sits
    glyph: Glyph
    parts: List[Tuple[str, str]] = field(default_factory=list)   # (source, part name)
    join_out: Optional[Tuple[float, float]] = None   # the join leaving this letter to the left: (x left, x right)


@dataclass
class Layout:
    text: str
    style: Style
    shapes: List[g.Shape] = field(default_factory=list)
    placed: List[Placed] = field(default_factory=list)
    joins: List[Tuple[float, float]] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def bbox(self) -> Tuple[float, float, float, float]:
        """Upright bounding box (x0, y0, x1, y1)."""
        xs = [p[0] for s in self.shapes for poly in s for p in poly]
        ys = [p[1] for s in self.shapes for poly in s for p in poly]
        return min(xs), min(ys), max(xs), max(ys)


def _join(c: str) -> str:
    return LETTERS[c][0] if c in LETTERS else "U"


def shape_forms(word: str) -> List[Tuple[str, str]]:
    """Return (letter, form) pairs; letters the engine cannot draw come back with form '?'."""
    chars = [c for c in unicodedata.normalize("NFC", word) if c not in HARAKAT]
    out: List[Tuple[str, str]] = []
    for i, c in enumerate(chars):
        if c.isspace():
            out.append((" ", " "))
            continue
        if c not in LETTERS:
            out.append((c, "?"))
            continue
        prev = chars[i - 1] if i > 0 else None
        nxt = chars[i + 1] if i + 1 < len(chars) else None
        jp = prev is not None and _join(prev) == "D" and _join(c) in "DR"
        jn = nxt is not None and _join(c) == "D" and _join(nxt) in "DR"
        out.append((c, {(False, False): "i", (False, True): "I", (True, True): "M", (True, False): "F"}[(jp, jn)]))
    return out


def flat(shapes):
    """Compound parts (several unioned shapes, possibly compound themselves) become separate shapes."""
    for s in shapes:
        if isinstance(s, Compound):
            yield from flat(s)
        else:
            yield s


def _below_range(glyph: Glyph):
    xs = [x for s in flat(glyph.shapes) if max(y for poly in s for _, y in poly) < 0 for poly in s for x, _ in poly]
    return (min(xs), max(xs)) if xs else None


def glyph_hint(glyph: Glyph) -> float:
    ys = [y for s in flat(glyph.shapes) for poly in s for _, y in poly]
    return max(ys) - min(ys) if ys else 1.0


def _has_floating_below(glyph: Glyph) -> bool:
    """True when some separate shape (a dot or dash) lies wholly under the baseline."""
    return any(max(y for poly in s for _, y in poly) < 0 for s in flat(glyph.shapes))


def _shift(shape: g.Shape, dx: float) -> g.Shape:
    return [[(x + dx, y) for x, y in poly] for poly in shape]


def _strip_marks(polys, x_height: float):
    """The body of a letter cut from the logo, without its dots and dashes: the small pieces wholly
    above the teeth or under the baseline."""
    from shapely.geometry import Polygon
    acc = None
    for ring in polys:
        if len(ring) >= 3:
            q = Polygon(ring).buffer(0)
            acc = q if acc is None else acc.symmetric_difference(q)
    if acc is None:
        return polys
    pieces = [q for q in getattr(acc, "geoms", [acc]) if q.geom_type == "Polygon" and not q.is_empty]
    if not pieces:
        return polys
    big = max(q.area for q in pieces)
    keep = [q for q in pieces if not (q.area < 0.35 * big and (q.bounds[1] >= 0.8 * x_height or q.bounds[3] <= 0.05 * x_height))]
    out = []
    for q in keep:
        out.append(list(q.exterior.coords)[:-1])
        out += [list(h.coords)[:-1] for h in q.interiors]
    return out


def from_logo(built: Glyph, cut: dict, form: str = "M", borrowed: Optional[str] = None, stem: float = 0.0,
              marks: Optional[list] = None, x_height: float = 0.0) -> Glyph:
    """The letter as the logo draws it, standing in for the engine's: the engine's letter is kept
    only for where joins arrive and leave. A letter nothing joins from the right (isolated, initial)
    keeps its right edge where the engine's is, so it does not reach into the white before it.
    `borrowed`: the cut was taken in another form of the letter (final for isolated, medial for
    initial, and back); a join that arrived from the right is cut off when this form has none.
    `marks`: the cut is another letter with the same body (ب for ن): its own dots are taken off and
    these (the engine's marks for this letter) put on."""
    polys = [[(float(x), float(y)) for x, y in ring] for ring in cut["polys"]]
    if marks is not None:
        polys = _strip_marks(polys, x_height)
    from shapely.geometry import Polygon as _P

    def area(rings):
        acc = None
        for ring in rings:
            if len(ring) >= 3:
                q = _P(ring).buffer(0)
                acc = q if acc is None else acc.symmetric_difference(q)
        return acc.area if acc is not None else 0.0
    full = area(polys)
    # a cut far from the size of the letter it stands for is a fragment or a swallowed neighbour
    mine = sum(area(sh) for sh in flat(built.shapes) if getattr(sh, "name", "") not in ("نقطة", "شرطة", "همزة"))
    # (the engine's letter is the one that differs from the logo, so only a clear misfit is refused)
    if mine > 0 and not (0.15 <= full / mine <= 6.0):
        return built
    if borrowed in ("F", "M") and form in ("i", "I"):
        from shapely.geometry import Polygon, box
        acc = None
        for ring in polys:
            q = Polygon(ring).buffer(0)
            acc = q if acc is None else acc.symmetric_difference(q)
        if acc is not None:
            # the incoming join is the thin ink right of the letter's own body
            right = max(x for ring in polys for x, _ in ring)
            body_right = right
            for xc in np.arange(right, right - 6 * stem, -0.25 * stem if stem else -1.0):
                band = acc.intersection(box(xc - 0.25 * stem, -1e4, xc, 1e4))
                if band.area > 0.6 * stem * stem:              # thicker than a hairline: the body begins
                    body_right = xc
                    break
            acc = acc.intersection(box(-1e6, -1e6, body_right, 1e6))
            polys = []
            for poly in getattr(acc, "geoms", [acc]):
                if poly.geom_type == "Polygon" and not poly.is_empty:
                    polys.append(list(poly.exterior.coords)[:-1])
                    polys += [list(h.coords)[:-1] for h in poly.interiors]
    if not polys or (borrowed and area(polys) < 0.5 * full):
        return built                                     # trimming the join left too little of the letter
    xs = [x for ring in polys for x, _ in ring]
    if form in ("i", "I") and max(xs) > 0:
        over = max(xs)
        polys = [[(x - over, y) for x, y in ring] for ring in polys]
        xs = [x - over for x in xs]
    # the next letter joins at the cut's own left end, not where the engine's letter would have ended
    reach = min(0.0, min(xs) - built.left)
    exit_x = built.exit_x + reach if built.exit_x is not None else None
    exit_edge = built.exit_edge + reach if built.exit_edge is not None else None
    return Glyph([Tagged(polys, "logo", "من الشعار")] + list(marks or []), left=min(min(xs), built.left), entry_x=built.entry_x,
                 entry_edge=built.entry_edge, exit_x=exit_x, exit_edge=exit_edge, stems=built.stems,
                 tail=built.tail)


MARK_NAMES = ("نقطة", "شرطة", "همزة")


def _is_mark(sh, st: Style) -> bool:
    name = getattr(sh, "name", "") or ""
    if any(m in name for m in MARK_NAMES):
        return True
    ys = [y for poly in sh for _, y in poly]
    return bool(ys) and (max(ys) < -0.05 * st.x_height or min(ys) > 1.02 * st.x_height)


def _section(shape, x: float) -> List[Tuple[float, float]]:
    """The y intervals an even-odd shape covers in the column at x."""
    ys = []
    for poly in shape:
        n = len(poly)
        for j in range(n):
            (x1, y1), (x2, y2) = poly[j], poly[(j + 1) % n]
            if (x1 <= x) != (x2 <= x):
                ys.append(y1 + (x - x1) * (y2 - y1) / (x2 - x1))
    ys.sort()
    return [(ys[i], ys[i + 1]) for i in range(0, len(ys) - 1, 2)]


def _merged(spans: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def stretch_run(glyph: Glyph, st: Style) -> Optional[Tuple[float, float]]:
    """Where a letter can be drawn longer: the longest stretch of columns in which its body is one
    flat band lying on the baseline (the floor of ب، ي، ك، س ...), as a kashida lengthens it.
    None when the letter has no such floor (ا، د، ر، و، م ...)."""
    body = [sh for sh in flat(glyph.shapes) if not _is_mark(sh, st)]
    xs = [x for sh in body for poly in sh for x, _ in poly]
    if not xs:
        return None
    pen = max(st.stem, st.hair)
    step = max(0.05 * st.stem, 0.25)
    lo, hi = min(xs) + 0.3 * st.stem, max(xs) - 0.3 * st.stem
    best, run = None, None
    prev = None
    for x in np.arange(lo, hi, step):
        sec = _merged([iv for sh in body for iv in _section(sh, x)])
        ok = (len(sec) == 1 and abs(sec[0][0]) <= 0.4 * pen and sec[0][1] - sec[0][0] <= 1.35 * pen
              and sec[0][1] <= 0.6 * st.x_height
              and (prev is None or (abs(prev[0] - sec[0][0]) < 0.06 * pen and abs(prev[1] - sec[0][1]) < 0.06 * pen)))
        prev = sec[0] if len(sec) == 1 else None
        if ok:
            run = (run[0], x) if run else (x, x)
            if best is None or run[1] - run[0] > best[1] - best[0]:
                best = run
        else:
            run = None
    if best is None or best[1] - best[0] < 0.3 * st.stem:
        return None
    return best


def stretch_glyph(glyph: Glyph, e: float, st: Style) -> Optional[Glyph]:
    """The letter drawn longer by e along its floor (see stretch_run); None if it has no floor."""
    run = stretch_run(glyph, st)
    if run is None or e <= 0:
        return None
    xc = 0.5 * (run[0] + run[1])

    def move(sh):
        if _is_mark(sh, st):
            cx = 0.5 * (min(x for poly in sh for x, _ in poly) + max(x for poly in sh for x, _ in poly))
            d = -e if cx < run[0] else (-0.5 * e if cx <= run[1] else 0.0)    # a mark over the floor stays over its middle
            new = [[(x + d, y) for x, y in poly] for poly in sh]
        else:
            new = [[(x - e if x < xc else x, y) for x, y in poly] for poly in sh]
        return Tagged(new, getattr(sh, "source", "engine"), getattr(sh, "name", "")) if isinstance(sh, Tagged) else new
    def deep(sh):
        return Compound(deep(sub) for sub in sh) if isinstance(sh, Compound) else move(sh)
    shapes = [deep(s) for s in glyph.shapes]
    at = lambda v: (v - e if v < xc else v) if v is not None else None
    return Glyph(shapes, left=at(glyph.left), entry_x=glyph.entry_x, entry_edge=glyph.entry_edge,
                 exit_x=at(glyph.exit_x), exit_edge=at(glyph.exit_edge),
                 stems=[(at(x), t, b) for x, t, b in glyph.stems], tail=at(glyph.tail))


def extension(st: Style, key: str) -> float:
    """How much longer the logo draws this letter form: the designer's value, else the fitted one."""
    return stretch_of(st, key)


def find_cut(st: Style, c: str, form: str):
    """The letter as the logo draws it, for `c` in `form`: cut in this form, else in the other form
    that differs only by the join from the right (isolated/final, initial/medial), else the same of a
    letter with the same body (ب for ن). Returns (cut, the form it was cut in if another, whether it
    is another letter's)."""
    other = {"i": "F", "F": "i", "I": "M", "M": "I"}[form]
    # a letter form the logo has keeps its own drawing (the engine's, or its own cut); another letter's
    # body serves only letters the logo does not have
    kin_ok = f"{c}.{form}" not in st.seen.split()
    for letter, kin in [(c, False)] + [(q, True) for q in (siblings(c, form) if kin_ok else [])]:
        if f"{letter}.{form}" in st.taken:
            return st.taken[f"{letter}.{form}"], None, kin
        if f"{letter}.{other}" in st.taken and (not kin or letter in siblings(c, other) or c in siblings(letter, other)):
            return st.taken[f"{letter}.{other}"], other, kin
    return None, None, False


def _dep_value(st: Style, dep: str):
    """The current value of something a letter read while it was built (see recipes.READS)."""
    if dep.startswith("v:"):
        k = dep[2:]
        return st.variants.get(k) or st.chosen.get(k)
    return value(st, dep) if dep in PARAMS else None


def typeset(word: str, st: Style, cache: Optional[dict] = None) -> Layout:
    """Lay out `word` right to left. `cache` (a dict kept by the caller) reuses the letters built for
    an earlier layout of the same logo whose parameters and variants they read are unchanged: for
    trying many values on the same word. (Only the designer's and the measured values are compared:
    the style's other measures must not change between calls that share a cache.)"""
    lay = Layout(text=word, style=st)
    x = 0.0
    prev: Optional[Placed] = None
    group_gap = st.group_gap if st.group_gap > 0 else 1.15 * st.white
    forms = shape_forms(word)
    # letters of the ر family that this logo joins from the top (see letters.top_joins)
    # In a curved logo the bar needs a tooth or a stem at the left end of the letter before it.
    top = {i for i, (c, f) in enumerate(forms) if f == "F" and c in "رزژ" and top_joins(st)
           and (st.angular >= 0.5 or forms[i - 1][0] in TOOTH_END)}
    extra = 0.0                          # the white or join length the letter before asks to leave after it
    for i, (c, form) in enumerate(forms):
        if form == " ":
            extra = 0.0
            if prev is not None:
                x = prev.x + prev.glyph.left - 3 * st.pitch
            prev = None
            continue
        if form == "?":
            lay.skipped.append(c)
            continue
        flags = frozenset((["no_exit"] if i + 1 in top else []) + (["top_entry"] if i in top else []))
        glyph = None
        ltok = BUILDING.set(c)
        if cache is not None and (c, form, flags) in cache:
            kept, deps = cache[(c, form, flags)]
            if all(_dep_value(st, d) == v for d, v in deps.items()):
                glyph = kept
        if glyph is None:
            tok, rtok = JOIN.set(flags), READS.set(set()) if cache is not None else None
            try:
                glyph = LETTERS[c][1](st, form)
                if cache is not None:
                    cache[(c, form, flags)] = (glyph, {d: _dep_value(st, d) for d in READS.get()})
            finally:
                JOIN.reset(tok)
                if rtok is not None:
                    READS.reset(rtok)
        BUILDING.reset(ltok)
        if st.reference and (st.ref_all or (f"{c}.{form}" not in st.seen.split()
                                             and REFERENCE in (st.variants.get(f"{c}.{form}"),
                                                               st.variants.get(f"{SKELETON.get(c, c)}.{form}")))):
            # a letter the logo lacks, asked of the logo's reference typeface: built on its structure
            # with the logo's pen (the examination found this no better than the engine on average,
            # so it is the designer's choice, not the default)
            from .reference import ref_glyph
            got = ref_glyph(st.reference, c, form, st, st.ref_wide, engine=glyph)
            if got is not None:
                glyph = got
        cut, borrowed, kin = find_cut(st, c, form)
        ext = extension(st, f"{c}.{form}")
        if cut and st.variants.get(f"{c}.{form}") != "المحرك":
            marks = [sh for sh in flat(glyph.shapes) if _is_mark(sh, st)] if kin else None
            glyph = from_logo(glyph, cut, form, borrowed, st.stem, marks, st.x_height)
            ext = 0.0                    # the letter as the logo draws it already has the logo's length
        if ext > 0 and form in ("i", "F"):
            longer = stretch_glyph(glyph, ext, st)
            if longer is not None:
                glyph, ext = longer, 0.0
        if form in ("I", "M"):
            ext = max(0.0, ext)          # a join is lengthened, never shortened past the white between teeth
        else:
            ext = max(-0.8 * group_gap, ext)
        glyph.below = _below_range(glyph)
        joined = prev is not None and form in ("M", "F")
        if prev is not None:
            if i in top:
                # joined at the top: the bar reaches the left edge of the previous tooth
                x = prev.x + prev.glyph.left - st.white - glyph.entry_edge
                joined = False
            elif joined and prev.glyph.exit_edge is not None:
                x = prev.x + prev.glyph.exit_edge - st.white - glyph.entry_edge
            else:
                gap = group_gap * (st.round_gap if glyph.entry_x < glyph.entry_edge - 1e-6 else 1.0)   # optical: rounds need more
                x = prev.x + prev.glyph.left - gap - glyph.entry_edge
                if prev.glyph.tail is not None and glyph.below is not None:
                    # a dot under this letter must not sit on the tail of the letter before
                    x = min(x, prev.x + prev.glyph.tail - 0.4 * st.dot - glyph.below[1])
            x -= extra
        here = Placed(c, form, x, glyph)
        if joined and prev.glyph.exit_x is not None:
            a, b = x + glyph.entry_x, prev.x + prev.glyph.exit_x
            lay.shapes.insert(0, Tagged(g.connector(st, a, b), "engine", "وصلة"))
            lay.joins.append((a, b))
            prev.join_out = (min(a, b), max(a, b))
        for s in glyph.shapes:
            for sub in flat([s]):
                src, name = getattr(sub, "source", "engine"), getattr(sub, "name", "") or "قطعة"
                here.parts.append((src, name))
                lay.shapes.append(Tagged(_shift(sub, x), src, name))
        lay.placed.append(here)
        prev = here
        extra = ext
    return lay
