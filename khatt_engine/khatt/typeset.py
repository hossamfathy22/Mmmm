"""Arabic shaping and right-to-left layout."""
from __future__ import annotations

import numpy as np

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import geometry as g
from .letters import JOIN, LETTERS, Glyph, top_joins
from .library import Compound, Tagged
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
    """Compound parts (several unioned shapes) become separate shapes."""
    for s in shapes:
        if isinstance(s, Compound):
            yield from s
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


def from_logo(built: Glyph, cut: dict, form: str = "M", borrowed: Optional[str] = None, stem: float = 0.0) -> Glyph:
    """The letter as the logo draws it, standing in for the engine's: the engine's letter is kept
    only for where joins arrive and leave. A letter nothing joins from the right (isolated, initial)
    keeps its right edge where the engine's is, so it does not reach into the white before it.
    `borrowed`: the cut was taken in another form of the letter (final for isolated, medial for
    initial, and back); a join that arrived from the right is cut off when this form has none."""
    polys = [[(float(x), float(y)) for x, y in ring] for ring in cut["polys"]]
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
    return Glyph([Tagged(polys, "logo", "من الشعار")], left=min(min(xs), built.left), entry_x=built.entry_x,
                 entry_edge=built.entry_edge, exit_x=exit_x, exit_edge=exit_edge, stems=built.stems,
                 tail=built.tail)


def typeset(word: str, st: Style) -> Layout:
    lay = Layout(text=word, style=st)
    x = 0.0
    prev: Optional[Placed] = None
    group_gap = st.group_gap if st.group_gap > 0 else 1.15 * st.white
    forms = shape_forms(word)
    # letters of the ر family that this logo joins from the top (see letters.top_joins)
    # In a curved logo the bar needs a tooth or a stem at the left end of the letter before it.
    top = {i for i, (c, f) in enumerate(forms) if f == "F" and c in "رزژ" and top_joins(st)
           and (st.angular >= 0.5 or forms[i - 1][0] in TOOTH_END)}
    for i, (c, form) in enumerate(forms):
        if form == " ":
            if prev is not None:
                x = prev.x + prev.glyph.left - 3 * st.pitch
            prev = None
            continue
        if form == "?":
            lay.skipped.append(c)
            continue
        flags = frozenset((["no_exit"] if i + 1 in top else []) + (["top_entry"] if i in top else []))
        tok = JOIN.set(flags)
        try:
            glyph = LETTERS[c][1](st, form)
        finally:
            JOIN.reset(tok)
        cut, borrowed = st.taken.get(f"{c}.{form}"), None
        if cut is None:
            # the same body in the other form that differs only by the join from the right
            other = {"i": "F", "F": "i", "I": "M", "M": "I"}[form]
            if f"{c}.{other}" in st.taken:
                cut, borrowed = st.taken[f"{c}.{other}"], other
        if cut and st.variants.get(f"{c}.{form}") != "المحرك":
            glyph = from_logo(glyph, cut, form, borrowed, st.stem)
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
        here = Placed(c, form, x, glyph)
        if joined and prev.glyph.exit_x is not None:
            a, b = x + glyph.entry_x, prev.x + prev.glyph.exit_x
            lay.shapes.insert(0, Tagged(g.connector(st, a, b), "engine", "وصلة"))
            lay.joins.append((a, b))
        for s in glyph.shapes:
            for sub in (s if isinstance(s, Compound) else [s]):
                src, name = getattr(sub, "source", "engine"), getattr(sub, "name", "") or "قطعة"
                here.parts.append((src, name))
                lay.shapes.append(Tagged(_shift(sub, x), src, name))
        lay.placed.append(here)
        prev = here
    return lay
