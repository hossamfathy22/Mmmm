"""Arabic shaping and right-to-left layout."""
from __future__ import annotations

import numpy as np

import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import geometry as g
from .letters import JOIN, LETTERS, Glyph, top_joins
from .library import Compound, Tagged
from .recipes import BUILDING, ON_REH, PARAMS, TEETH_CHOICE, READS, REFERENCE, SKELETON, siblings, stretch_of, value
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
        if cut.get("marks_dx"):
            dx = float(cut["marks_dx"])
            marks = [Tagged([[(x + dx, y) for x, y in poly] for poly in sh], getattr(sh, "source", "engine"),
                            getattr(sh, "name", "")) for sh in marks]
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
    body = Tagged(polys, "logo", "من الشعار")
    body.piece = bool(cut.get("piece"))                  # read from the logo's own pieces (selfcheck._by_pieces)
    return Glyph([body] + list(marks or []), left=min(min(xs), built.left), entry_x=built.entry_x,
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


PART_SAME = 0.6          # how alike the engine's two parts must be (IoU) for the logo's part to pass between them


def _same_part(a, b) -> float:
    """IoU of two engine parts drawn in the same frame (right ends together, on the same baseline)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def geom(rings):
        return unary_union([Polygon(r).buffer(0) for r in rings if len(r) >= 3])
    A, B = geom(a), geom(b)
    if A.is_empty or B.is_empty:
        return 0.0
    u = A.union(B).area
    return A.intersection(B).area / u if u else 0.0


TRANSFORMS = {"id": (1, 1), "mx": (-1, 1), "my": (1, -1), "r180": (-1, -1)}
PART_DERIVED = 0.55      # a part made from another (mirrored or turned) must match this well in the engine's own construction


def _turn(rings, t: str):
    """Mirror or turn outlines (a turn by 180 degrees is both mirrors): a pen thick on verticals and
    thin on horizontals stays so under all of these; a quarter turn would swap them, and is not used."""
    a, b = TRANSFORMS[t]
    return [[(a * x, b * y) for x, y in ring] for ring in rings]


def _placement(engine_src, target, t: str):
    """How to move the source part, once turned, onto the target: right ends together, and (when
    turned) bottoms together. Returns (dx, dy, the source engine part placed) or None."""
    src = _turn(engine_src, t)
    if not src or not target:
        return None
    dx = -max(x for r in src for x, _ in r)
    dy = 0.0 if t == "id" else min(y for r in target for _, y in r) - min(y for r in src for _, y in r)
    return dx, dy, [[(x + dx, y + dy) for x, y in r] for r in src]


def with_parts(glyph: Glyph, st: Style, key: str = "") -> Glyph:
    """The letter with each of its parts the logo draws (st.parts) put in place of the engine's: the
    logo's part keeps where it stood against the engine's part in the letter it was cut from. A part
    the logo lacks may be made from another of its parts, mirrored or turned, when the engine's own
    two parts match so (the loop of ص from the head of و, the back cup of ع from the cup of ن)."""
    used = [False]

    real = st.derive_parts == "real"
    subs = None
    if real:
        from .reference import substitutions
        subs = substitutions(st.ref_skip)

    def best_source(name, target):
        best = None
        for kind, part in st.parts.items():
            if not part.get("engine"):
                continue
            for t in TRANSFORMS:
                same = kind == name and t == "id"
                if not same and not st.derive_parts:
                    continue
                if kind == name and t != "id":
                    continue                       # the part itself is used as it is
                got = _placement(part["engine"], target, t)
                if got is None:
                    continue
                v = _same_part(got[2], target)
                if same:
                    ok = v >= PART_SAME
                elif real:
                    # a part made from another only where real typefaces draw the two alike so
                    r = subs.get((f"{part.get('from')}|{kind}", f"{key}|{name}", t, ""),
                                 subs.get((kind, name, t, "kind"), 0.0))
                    v, ok = r, r >= PART_DERIVED
                else:
                    ok = v >= PART_DERIVED
                if ok and (best is None or v > best[0]):
                    best = (v, kind, t, got[0], got[1])
        return best

    def swap(sh):
        if isinstance(sh, Compound):
            return Compound(swap(sub) for sub in sh)
        name = getattr(sh, "name", "")
        rings = [p for p in sh if len(p) >= 3]
        if not name or not rings or (name not in st.parts and not st.derive_parts):
            return sh
        xr = max(x for poly in rings for x, _ in poly)
        target = [[(x - xr, y) for x, y in poly] for poly in rings]
        found = best_source(name, target)
        if found is None:
            return sh
        _, kind, t, dx, dy = found
        polys = None
        if st.part_fit and t == "id":
            # the logo's part on the size of this letter's part, with the logo's own widths
            from .reference import fit_part
            got = fit_part(st.parts[kind], target)
            if got:
                polys = [[(x + xr, y) for x, y in ring] for ring in got]
        if polys is None:
            polys = [[(x + dx + xr, y + dy) for x, y in ring] for ring in _turn(st.parts[kind]["polys"], t)]
        # the designer's hand on the part: longer or shorter, taller or lower, thicker or thinner; the
        # part is drawn again from its centre line, so a longer part keeps its pen unless asked otherwise
        kl, kh, kw = (float(st.params.get(f"قطعة.{kind}.{w}", 1.0)) for w in ("طول", "ارتفاع", "سمك"))
        # and, before the designer's hand, the difference real typefaces make between the letter the
        # part was cut from and this one
        if st.part_ref_fit and key and t == "id" and kind == name and st.parts[kind].get("from") and st.parts[kind]["from"] != key:
            from .reference import part_ratio
            got = part_ratio(st.parts[kind]["from"], key, kind, st.ref_skip)
            if got:
                kl *= min(1.6, max(0.6, got[0]))
                kh *= min(1.6, max(0.6, got[1]))
                if abs(kl - 1) < 0.04:
                    kl = 1.0
                if abs(kh - 1) < 0.04:
                    kh = 1.0
        if (kl, kh, kw) != (1.0, 1.0, 1.0):
            from .reference import draw_strokes, part_strokes
            strokes = part_strokes(polys)
            if strokes:
                x1 = max(x for r in polys for x, _ in r)
                y0 = min(y for r in polys for _, y in r)
                redrawn = draw_strokes([[(x, y, w * kw) for x, y, w in st_] for st_ in strokes],
                                       lambda x, y: (x1 + (x - x1) * kl, y0 + (y - y0) * kh))
                if redrawn:
                    polys = redrawn
        used[0] = True
        return Tagged(polys, "logo", name)
    shapes = [swap(s) for s in glyph.shapes]
    if not used[0]:
        return glyph
    xs = [x for sh in flat(shapes) for poly in sh for x, _ in poly]
    return Glyph(shapes, left=min(glyph.left, min(xs)) if xs else glyph.left, entry_x=glyph.entry_x,
                 entry_edge=glyph.entry_edge, exit_x=glyph.exit_x, exit_edge=glyph.exit_edge, stems=glyph.stems,
                 tail=glyph.tail)


def _cut_span(glyph: Glyph) -> Optional[Tuple[float, float]]:
    """Where the body cut from the logo's own pieces lies in a letter (its left and right ends), or None
    when the letter is the engine's own, or cut where the engine's letter stood (its joins are the
    engine's)."""
    xs = [x for sh in glyph.shapes if getattr(sh, "source", "") == "logo" and getattr(sh, "piece", False)
          for poly in sh for x, _ in poly]
    return (min(xs), max(xs)) if xs else None


# A logo that draws its ر as a stem (an upright stroke from the x-height down past the baseline, its
# foot curled) shows how its designer draws every stem: ا is that ر set on the baseline, ل joined to
# the next letter the same (its curl is the join), and ل alone or last the ر drawn up to ل's height;
# ك the same stems, joined with the engine's arm across the top, alone or last with its small mark.
# What a designer does by hand with Mootanroo's ر. The designer's call (style.stem_kin, or the
# variant ON_REH letter by letter): on the hidden ا of the other logos whose ر is a stem it fitted
# worse than the engine's own ا three times in four (karashab's ا from 0.92 to 0.45).
STEM_KIN = {"ا": "iF", "ل": "iIMF", "ك": "iIMF"}
_STEMS: dict = {}


def _sections(shape, y: float, w: float) -> List[Tuple[float, float]]:
    from shapely.geometry import box
    cut = shape.intersection(box(-1e5, y - 0.02 * w, 1e5, y + 0.02 * w))
    spans = sorted((g.bounds[0], g.bounds[2]) for g in getattr(cut, "geoms", [cut]) if not g.is_empty and g.area > 0)
    return _merged(spans)


def stem_from_reh(st: Style, c: str, form: str) -> Optional[dict]:
    """`c` (ا or ل) in `form` on the stroke of the logo's ر (see STEM_KIN), or None when the logo has no
    ر cut, or draws it otherwise than as a stem."""
    src = next((st.taken[q] for q in ("ر.F", "ر.i", "ز.F", "ز.i") if q in st.taken), None)
    if src is None:
        return None
    polys = [[(float(x), float(y)) for x, y in ring] for ring in src["polys"]]
    key = (c, form, round(st.x_height, 2), round(st.stem, 2), len(polys), tuple(polys[0][:3]))
    if key in _STEMS:
        return _STEMS[key]
    from shapely.geometry import Polygon
    X = st.x_height
    acc = None
    for ring in polys:
        if len(ring) >= 3:
            q = Polygon(ring).buffer(0)
            acc = q if acc is None else acc.symmetric_difference(q)
    out = None
    if acc is not None and not acc.is_empty:
        y0, y1 = acc.bounds[1], acc.bounds[3]
        mid, low = _sections(acc, 0.55 * X, X), _sections(acc, 0.25 * X, X)
        upright = (y1 >= 0.85 * X and y0 < -0.1 * X and len(mid) == 1 and len(low) == 1
                   and all(b - a <= 2.2 * st.stem for a, b in mid + low)
                   and abs((mid[0][0] + mid[0][1]) - (low[0][0] + low[0][1])) / 2 <= 0.6 * st.stem)
        if upright:
            built = LETTERS[c][1](st, form)
            top = max(y for sh in flat(built.shapes) for poly in sh for _, y in poly)
            dy = -y0 if (c == "ا" or form in ("I", "M")) else 0.0     # set on the baseline, or left hanging
            yc = 0.55 * X + dy
            grow = max(0.0, top - (y1 + dy))
            rings = [[(round(x, 2), round(y + dy + (grow if y + dy > yc else 0.0), 2)) for x, y in ring] for ring in polys]
            out = {"polys": rings, "from": "ر", "piece": bool(src.get("piece"))}
            if c == "ك":
                # the stem's left side, high up and half way: where the arm leaves it, and the mark stands by
                stem_left = mid[0][0]
                arm = [sh for sh in flat(built.shapes) if getattr(sh, "name", "") == "ذراع الكاف"]
                if form in ("I", "M") and arm:
                    from shapely.ops import unary_union
                    ax = max(x for sh in arm for poly in sh for x, _ in poly)
                    ay = max(y for sh in arm for poly in sh for _, y in poly)
                    top_y = max(y for ring in rings for _, y in ring)
                    pieces = [Polygon(r).buffer(0) for r in rings if len(r) >= 3]
                    pieces += [Polygon([(x - ax + stem_left + 0.5 * st.stem, y - ay + top_y) for x, y in poly]).buffer(0)
                               for sh in arm for poly in sh if len(poly) >= 3]
                    whole = unary_union(pieces)
                    rings = []
                    for poly in getattr(whole, "geoms", [whole]):
                        if poly.geom_type == "Polygon" and not poly.is_empty:
                            rings.append([(round(x, 2), round(y, 2)) for x, y in list(poly.exterior.coords)[:-1]])
                            rings += [[(round(x, 2), round(y, 2)) for x, y in list(h.coords)[:-1]] for h in poly.interiors]
                    out["polys"] = rings
                mark = [x for sh in flat(built.shapes) if getattr(sh, "name", "") == "شرطة" for poly in sh for x, _ in poly]
                if mark:
                    out["marks_dx"] = round(stem_left - 0.25 * st.stem - max(mark), 2)
    if len(_STEMS) > 512:
        _STEMS.clear()
    _STEMS[key] = out
    return out


def _upright(glyph: Glyph, st: Style, side: str) -> Optional[Tuple[float, float]]:
    """The upright stroke a cut letter begins (right) or ends (left) with, standing on the baseline and
    reaching the x-height: its left and right sides at half the x-height; None when the letter has none
    there (a curl, a hook, a join that comes in low)."""
    from shapely.geometry import Polygon
    rings = [poly for sh in glyph.shapes if getattr(sh, "source", "") == "logo" for poly in sh if len(poly) >= 3]
    if not rings:
        return None
    acc = None
    for ring in rings:
        q = Polygon(ring).buffer(0)
        acc = q if acc is None else acc.symmetric_difference(q)
    X = st.x_height
    mid = _sections(acc, 0.5 * X, X)
    if not mid:
        return None
    a, b = mid[-1] if side == "right" else mid[0]
    if b - a > 2.2 * st.stem:
        return None
    # at the letter's edge (a hook inside a bowl is no arm the next letter can share)
    x0, _, x1, _ = acc.bounds
    if (side == "right" and x1 - b > 0.6 * st.stem) or (side == "left" and a - x0 > 0.6 * st.stem):
        return None
    c = 0.5 * (a + b)
    # straight: the same stroke at the same place higher up (the side of a bowl leans as it rises and
    # curls over: ن's is no arm)
    for f in (0.65, 0.8):
        row = _sections(acc, f * X, X)
        near = [sp for sp in row if sp[0] - 0.5 * st.stem <= c <= sp[1] + 0.5 * st.stem]
        if not near or abs(0.5 * (near[0][0] + near[0][1]) - c) > 0.4 * st.stem:
            return None
    ys = _merged([(lo, hi) for lo, hi in _section([list(r) for r in rings], c)])
    span = next(((lo, hi) for lo, hi in ys if lo <= 0.5 * X <= hi), None)
    if span is None or span[0] > 0.15 * X or span[1] < 0.85 * X:
        return None
    return a, b


# A letter of teeth (س ش) the logo lacks is built of the logo's own tooth, where the logo shows one
# (ب ت ث ي in a joined form): three of them in a row, joined as the logo joins its teeth (Mootanroo
# hangs them as cups sharing an arm), the engine's bowl after them where the letter ends in one.
TEETH = {"س": 3, "ش": 3}
TOOTH_FROM = [f"{c}.{f}" for f in "MI" for c in "بتثپيی"] + ["ن.M", "ن.I"]
_TEETH: dict = {}


def teeth_from_logo(st: Style, c: str, form: str, cups_only: bool = False) -> Optional[dict]:
    """`c` (س or ش) in `form` built of the logo's tooth (see TEETH), as a cut; None when the logo shows
    no tooth. `cups_only`: only when the logo's tooth is a cup read from its pieces, an upright on
    either side, the teeth joined by the arm they share (where a tooth is cut with the joins on either
    side of it, three of them make a long س: on the hidden س of three logos that were so, the word
    matched worse every time)."""
    src_key = next((q for q in TOOTH_FROM if q in st.taken), None)
    if src_key is None or c not in TEETH or (cups_only and not st.taken[src_key].get("piece")):
        return None
    src = st.taken[src_key]
    key = (c, form, src_key, cups_only, round(st.x_height, 2), round(st.stem, 2), len(src["polys"]), tuple(map(tuple, src["polys"][0][:3])))
    if key in _TEETH:
        return _TEETH[key]
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import affinity
    X = st.x_height
    body = _strip_marks([[(float(x), float(y)) for x, y in ring] for ring in src["polys"]], X)
    acc = None
    for ring in body:
        if len(ring) >= 3:
            q = Polygon(ring).buffer(0)
            acc = q if acc is None else acc.symmetric_difference(q)
    out = None
    if acc is not None and not acc.is_empty and 0.5 * X <= acc.bounds[3] - acc.bounds[1] <= 1.8 * X:
        tooth = Glyph([Tagged([list(r) for r in body], "logo", "سن")], left=acc.bounds[0])
        tooth.shapes[0].piece = True
        a, b = _upright(tooth, st, "left"), _upright(tooth, st, "right")
        if cups_only and not (a and b):
            if len(_TEETH) > 512:
                _TEETH.clear()
            _TEETH[key] = None
            return None
        x0, _, x1, _ = acc.bounds
        # the step from one tooth to the next: the arms they share, or the teeth side by side
        step = (0.5 * (b[0] + b[1]) - 0.5 * (a[0] + a[1])) if (a and b) else (x1 - x0)
        teeth = [affinity.translate(acc, xoff=-x1 - k * step) for k in range(TEETH[c])]
        parts = list(teeth)
        built = LETTERS[c][1](st, form)
        bowl = [sh for sh in flat(built.shapes) if getattr(sh, "name", "") == "كأس سفلي"]
        if bowl:
            # the engine's bowl, its right end at the foot of the last tooth's left side
            left = min(t.bounds[0] for t in teeth)
            bx = max(x for sh in bowl for poly in sh for x, _ in poly)
            for sh in bowl:
                for poly in sh:
                    if len(poly) >= 3:
                        parts.append(Polygon([(x - bx + left + 0.5 * st.stem, y) for x, y in poly]).buffer(0))
        whole = unary_union(parts)
        rings = []
        for poly in getattr(whole, "geoms", [whole]):
            if poly.geom_type == "Polygon" and not poly.is_empty:
                rings.append([(round(x, 2), round(y, 2)) for x, y in list(poly.exterior.coords)[:-1]])
                rings += [[(round(x, 2), round(y, 2)) for x, y in list(h.coords)[:-1]] for h in poly.interiors]
        # the engine's marks (ش) over the middle of the new teeth
        eng = [x for sh in flat(built.shapes) if getattr(sh, "name", "") in ("جذع", "كأس الوصل") for poly in sh for x, _ in poly]
        mid_new = 0.5 * (min(t.bounds[0] for t in teeth) + max(t.bounds[2] for t in teeth))
        mid_eng = 0.5 * (min(eng) + max(eng)) if eng else mid_new
        out = {"polys": rings, "from": src_key, "piece": True, "marks_dx": round(mid_new - mid_eng, 2)}
    if len(_TEETH) > 512:
        _TEETH.clear()
    _TEETH[key] = out
    return out


# The letters few logos show, and the engine builds worst. Hidden in 324 words set in open typefaces
# standing in for logos, the reference typeface's outline at the logo's weight (the typeface chosen
# by the word's other letters) matched the logo's letter 0.461, the engine's construction 0.396, the
# engine's with the logo's parts 0.407 (better 194, worse 99); every one of them gained (خ 0.22 to
# 0.42, ف 0.35 to 0.51). On the simple letters (teeth, stems, ر و) the engine stays the better.
REF_LETTERS = set("حصطعفقه")
HEADED = set("فق")          # letters of a head on a neck: on the reference, the head is the logo's (reference.with_logo_head)


def _logo_source(st: Style, c: str, form: str, glyph: Glyph, parts: bool = True) -> bool:
    """Whether the logo gives this letter anything of its own: a cut (its own, a sibling's, its teeth,
    its stem), or (`parts`) one of its parts."""
    if find_cut(st, c, form)[0] is not None:
        return True
    if not parts:
        return False
    names = {getattr(sh, "name", "") for sh in flat(glyph.shapes)}
    return bool(st.parts) and bool(names & set(st.parts))


def find_cut(st: Style, c: str, form: str):
    """The letter as the logo draws it, for `c` in `form`: cut in this form, else in the other form
    that differs only by the join from the right (isolated/final, initial/medial), else the same of a
    letter with the same body (ب for ن). Returns (cut, the form it was cut in if another, whether it
    is another letter's)."""
    other = {"i": "F", "F": "i", "I": "M", "M": "I"}[form]
    # a letter form the logo has keeps its own drawing (the engine's, or its own cut); another letter's
    # body serves only letters the logo does not have
    kin_ok = f"{c}.{form}" not in st.seen.split()
    want = st.variants.get(f"{c}.{form}")
    if (c in TEETH and kin_ok and (want == TEETH_CHOICE or (st.teeth_kin and want is None))
            and not any(f"{c}.{f}" in st.taken for f in (form, other))):
        got = teeth_from_logo(st, c, form, cups_only=want != TEETH_CHOICE)
        if got is not None:
            return got, None, True
    if ((want == ON_REH or (st.stem_kin and want is None)) and kin_ok and c in STEM_KIN and form in STEM_KIN[c]
            and not any(f"{c}.{f}" in st.taken for f in (form, other))):
        got = stem_from_reh(st, c, form)
        if got is not None:
            return got, None, True
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
        ltok = BUILDING.set(f"{c}.{form}")
        if cache is not None and (c, form, flags) in cache:
            kept, deps = cache[(c, form, flags)]
            if all(_dep_value(st, d) == v for d, v in deps.items()):
                glyph = kept
        if glyph is None:
            tok, rtok = JOIN.set(flags), READS.set(set()) if cache is not None else None
            try:
                glyph = LETTERS[c][1](st, form)
                if cache is not None:
                    # sorted: a set's order changes from run to run, and the fits try values in this order
                    cache[(c, form, flags)] = (glyph, {d: _dep_value(st, d) for d in sorted(READS.get())})
            finally:
                JOIN.reset(tok)
                if rtok is not None:
                    READS.reset(rtok)
        BUILDING.reset(ltok)
        # a letter of few logos (ح ص ط ع ف ق ه ...) the logo lacks and has no letter of its own for (no
        # cut, no sibling's body): on the reference typeface's outline (see REF_LETTERS)
        auto = (st.ref_auto and st.seen and SKELETON.get(c, c) in REF_LETTERS and f"{c}.{form}" not in st.seen.split()
                and st.variants.get(f"{c}.{form}") is None and not _logo_source(st, c, form, glyph, parts=False))
        if st.reference and (st.ref_all or auto or (f"{c}.{form}" not in st.seen.split()
                                                    and REFERENCE in (st.variants.get(f"{c}.{form}"),
                                                                      st.variants.get(f"{SKELETON.get(c, c)}.{form}")))):
            # a letter the logo lacks, asked of the logo's reference typeface: built on its structure
            # with the logo's pen (the examination found this no better than the engine on average,
            # so it is the designer's choice, not the default)
            from .reference import ref_glyph, with_logo_head
            got = ref_glyph(st.reference, c, form, st, st.ref_wide, engine=glyph)
            if got is not None and SKELETON.get(c, c) in HEADED and st.parts.get("رأس"):
                # the logo's own head on the reference's neck (ف ق): the logo's part where it has one
                got = with_logo_head(got, st) or got
            if got is not None:
                glyph = got
        if (st.legible_fallback and st.seen and f"{c}.{form}" not in st.seen.split()
                and not (cut_here := st.taken.get(f"{c}.{form}"))):
            # a letter the logo lacks that real typefaces would not read as itself: on a reference
            # structure that they do read, with the logo's pen
            from .legible import readable_reference
            better = readable_reference(glyph, c, form, st)
            if better is not None:
                glyph = better
        if st.parts and f"{c}.{form}" not in st.seen.split():
            # a letter the logo lacks: its parts as the logo draws them, where the logo has them, as
            # long as it still reads as the letter it is
            built = with_parts(glyph, st, f"{c}.{form}")
            if built is not glyph:
                from .legible import keeps_reading
                # a part the designer has set by hand is their judgement, not the reader's
                touched = {n.split(".")[1] for n in st.params if n.startswith("قطعة.")}
                # and so is a part the designer drew or chose for the kit
                touched |= {n for n, v in st.parts.items() if isinstance(v, dict) and v.get("source") in ("designer", "reference")}
                mine = {getattr(sh, "name", "") for sh in flat(built.shapes) if getattr(sh, "source", "") == "logo"}
                if not st.part_legible or (touched & mine) or keeps_reading(glyph, built, c, form, st):
                    glyph = built
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
            pc, gc = _cut_span(prev.glyph), _cut_span(glyph)
            if joined and pc and gc:
                # two letters as the logo draws them: the logo's join is in them (the hairline of م,
                # the arm two cups share), so they meet as they meet in the logo, with no join of the
                # engine's between them
                x = prev.x + pc[0] - gc[1]
                # and where the one before ends in an upright arm and this one begins with an upright,
                # they are one stroke, as two cups share an arm in the logo (and the arm of a cup rises
                # into the ا after it)
                a, b = _upright(prev.glyph, st, "left"), _upright(glyph, st, "right")
                if a and b:
                    x = prev.x + 0.5 * (a[0] + a[1]) - 0.5 * (b[0] + b[1])
                joined = False
            elif i in top:
                # joined at the top: the bar reaches the left edge of the previous tooth
                x = prev.x + prev.glyph.left - st.white - glyph.entry_edge
                joined = False
            elif joined and prev.glyph.exit_edge is not None:
                x = prev.x + prev.glyph.exit_edge - (st.join_white or st.white) - glyph.entry_edge
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
