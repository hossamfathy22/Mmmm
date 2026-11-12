"""Arabic shaping and right-to-left layout."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import geometry as g
from .letters import JOIN, LETTERS, Glyph, top_joins
from .library import Compound, Tagged
from .style import Style

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


def typeset(word: str, st: Style) -> Layout:
    lay = Layout(text=word, style=st)
    x = 0.0
    prev: Optional[Placed] = None
    group_gap = st.group_gap if st.group_gap > 0 else 1.15 * st.white
    forms = shape_forms(word)
    # letters of the ر family that this logo joins from the top (see letters.top_joins)
    top = {i for i, (c, f) in enumerate(forms) if f == "F" and c in "رزژ" and top_joins(st)}
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
