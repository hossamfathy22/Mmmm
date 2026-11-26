"""The parts kit: the few parts every Arabic letter is built of, as the logo draws them.

A logo's word holds a part of the alphabet's letters but more of its parts: the cup of its ن builds the
cups of ي ل ق س, the head of its و the heads of ف ق م. A dozen parts build every letter. The kit says,
for one logo, where each part comes from:

  logo      cut out of the logo's own letters (selfcheck.take_parts)
  designer  drawn or chosen by the designer, once, for this logo
  engine    neither: the engine's construction, with the logo's measures

and gives the designer what is needed to complete it: each part in a letter that shows it, the
letters it builds, suggestions from real typefaces drawn again with the logo's pen, and a pen that
turns a line the designer draws into a part in the logo's manner.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .letters import LETTERS
from .style import Style

# name, what a designer calls it, the letter form it is shown and drawn in
KIT: List[Tuple[str, str, str]] = [
    ("رأس", "الرأس (و ف ق م)", "و.i"),
    ("عنق", "عنق الفاء والقاف", "ف.I"),
    ("ذيل", "الذيل (ر ز و)", "ر.i"),
    ("ذيل الميم", "ذيل الميم", "م.i"),
    ("كأس سفلي", "الكأس (ن ي ل ق س ص)", "ن.i"),
    ("كأس أيمن", "الكأس الخلفية (ع)", "ع.F"),
    ("عروة", "العروة (ص ض ط ظ)", "ص.I"),
    ("أنف الحاء", "أنف الحاء (ح ج خ)", "ح.I"),
    ("الحاء", "الحاء المنفصلة والأخيرة", "ح.i"),
    ("عين", "عين العين الوسطى", "ع.M"),
    ("ذراع الكاف", "ذراع الكاف", "ك.I"),
    ("راية الدال", "راية الدال", "د.i"),
    ("قاعدة الدال", "قاعدة الدال", "د.i"),
    ("قاطع الهاء", "قاطع الهاء الأولى", "ه.I"),
    ("عين سفلى", "عين الهاء الوسطى", "ه.M"),
    ("الياء", "الياء المنفصلة", "ي.i"),
]
LABEL = {n: label for n, label, _ in KIT}
# parts of the kit the logo's part library (library.py) gives the letters' builders directly
LIBRARY_OF = {"ذيل": "tail", "رأس": "loop"}
SAMPLE = {n: sample for n, _, sample in KIT}


def _forms(c: str) -> str:
    return "iIMF" if LETTERS[c][0] == "D" else "iF"


def _plain(st: Style) -> Style:
    return Style(**{**st.__dict__, "slant": 0.0})


def engine_part(st: Style, name: str, sample: Optional[str] = None) -> Optional[Dict]:
    """The engine's own drawing of a part in a letter that shows it: its outlines relative to the
    part's right end, and the letter's outlines, right end of the letter at 0."""
    from .typeset import flat
    c, f = (sample or SAMPLE[name]).split(".")
    up = _plain(Style(**{**st.__dict__, "parts": {}, "taken": {}}))
    g = LETTERS[c][1](up, f)
    part = [sh for sh in flat(g.shapes) if getattr(sh, "name", "") == name]
    if not part:
        return None
    rings = [p for sh in part for p in sh if len(p) >= 3]
    xr = max(x for r in rings for x, _ in r)
    letter = [[list(map(float, pt)) for pt in p] for sh in flat(g.shapes) for p in sh if len(p) >= 3]
    return {"part": [[(x - xr, y) for x, y in r] for r in rings], "xr": xr, "letter": letter}


def users(st: Style) -> Dict[str, List[str]]:
    """Which letter forms each part of the kit builds, in the engine's construction."""
    from .typeset import flat
    up = _plain(Style(**{**st.__dict__, "parts": {}, "taken": {}}))
    out: Dict[str, List[str]] = {n: [] for n, _, _ in KIT}
    for c in LETTERS:
        for f in _forms(c):
            try:
                g = LETTERS[c][1](up, f)
            except Exception:
                continue
            for n in {getattr(sh, "name", "") for sh in flat(g.shapes)}:
                if n in out:
                    out[n].append(f"{c}.{f}")
    return out


def status(st: Style) -> List[Dict]:
    """Every part of the kit for this logo: where it comes from, and the letters it builds."""
    by = users(st)
    lib = (st.library or {}).get("parts") or {}
    out = []
    for name, label, sample in KIT:
        p = st.parts.get(name)
        src = (p.get("source") or "logo") if p else "engine"
        if src == "engine" and LIBRARY_OF.get(name) in lib:
            src = "logo"                     # the letters take it from the logo's part library (library.py)
        out.append({"name": name, "label": label, "sample": sample, "source": src,
                    "from": p.get("from", "") if p else "", "letters": by.get(name, [])})
    return out


# ------------------------------------------------------------------ the logo's pen
def _smooth(points: List[Tuple[float, float]], closed: bool, per: int = 8) -> List[Tuple[float, float]]:
    """A Catmull-Rom curve through the designer's points (a closed one when the ring is closed)."""
    pts = list(points)
    if len(pts) < 3:
        return pts
    n = len(pts)
    out = []
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = pts[(i - 1) % n] if closed else pts[max(0, i - 1)]
        p1, p2 = pts[i], pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if closed else pts[min(n - 1, i + 2)]
        for k in range(per):
            t = k / per
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    if not closed:
        out.append(pts[-1])
    return out


def pen_line(points: List[Tuple[float, float]], st: Style, closed: bool = False, thickness: float = 1.0,
             smooth: bool = True, ends: str = "round") -> List[List[Tuple[float, float]]]:
    """A line drawn with the logo's pen: at every point as thick as the pen rule gives for the line's
    direction there (thick across verticals, thin across horizontals), times `thickness`. `ends`:
    round, flat (cut square to the line) or diamond (the logo's rhombic dot at each end)."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    pts = _smooth(points, closed) if smooth else list(points)
    if len(pts) < 2:
        return []
    n = len(pts)
    seq = pts + [pts[0]] if closed else pts
    widths = []
    for i in range(len(seq)):
        a, b = seq[max(0, i - 1)], seq[min(len(seq) - 1, i + 1)]
        if closed:
            a, b = seq[(i - 1) % n], seq[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        widths.append(st.width_at(dy / d) * thickness / 2)
    pieces = []
    for (x0, y0), (x1, y1), r0, r1 in zip(seq, seq[1:], widths, widths[1:]):
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d < 1e-6:
            continue
        nx, ny = -dy / d, dx / d
        pieces.append(Polygon([(x0 + nx * r0, y0 + ny * r0), (x1 + nx * r1, y1 + ny * r1),
                               (x1 - nx * r1, y1 - ny * r1), (x0 - nx * r0, y0 - ny * r0)]).buffer(0))
    inner = seq[1:-1] if not closed else seq
    pieces += [Point(x, y).buffer(r, 12) for (x, y), r in zip(inner, widths[1:-1] if not closed else widths)]
    if not closed:
        for (x, y), r, (xo, yo) in ((seq[0], widths[0], seq[1]), (seq[-1], widths[-1], seq[-2])):
            if ends == "round":
                pieces.append(Point(x, y).buffer(r, 12))
            elif ends == "diamond":
                s = max(r * 1.35, st.dot / 2)
                pieces.append(Polygon([(x, y + s), (x + s, y), (x, y - s), (x - s, y)]))
    body = unary_union(pieces)
    rings = []
    for poly in getattr(body, "geoms", [body]):
        if poly.geom_type == "Polygon" and not poly.is_empty:
            rings.append([(round(x, 2), round(y, 2)) for x, y in list(poly.exterior.coords)[:-1]])
            rings += [[(round(x, 2), round(y, 2)) for x, y in list(h.coords)[:-1]] for h in poly.interiors]
    return rings


def centre_line(polys, points: int = 14) -> List[Tuple[float, float]]:
    """The longest stroke of a part's centre line, as a few points the designer can drag."""
    from .reference import part_strokes
    strokes = part_strokes(polys)
    if not strokes:
        return []
    s = max(strokes, key=len)
    idx = np.linspace(0, len(s) - 1, min(points, len(s))).round().astype(int)
    return [(round(s[i][0], 2), round(s[i][1], 2)) for i in idx]


# ------------------------------------------------------------------ parts the designer makes
def designer_part(st: Style, name: str, points: List[Tuple[float, float]], closed: bool = False,
                  thickness: float = 1.0, ends: str = "round") -> Optional[Dict]:
    """A part drawn by the designer: a line through `points` (in the frame of the part's letter as
    the engine shows it: its right end at 0, the baseline at 0), drawn with the logo's pen. Returns
    a part as the kit keeps it (relative to the engine's part, so that it goes into every letter
    that has this part)."""
    eng = engine_part(st, name)
    if eng is None or len(points) < 2:
        return None
    rings = pen_line(points, st, closed, thickness, ends=ends)
    if not rings:
        return None
    xr = eng["xr"]
    return {"polys": [[(x - xr, y) for x, y in r] for r in rings], "engine": eng["part"], "from": SAMPLE[name],
            "source": "designer", "line": [(float(x), float(y)) for x, y in points], "closed": bool(closed),
            "thickness": float(thickness), "ends": ends}


def candidates(st: Style, name: str, limit: int = 6) -> List[Dict]:
    """Suggestions for a part the logo lacks: the part as real typefaces draw it in the same letter
    (or, failing that, in another letter that has it), sized to the logo's x-height and drawn again
    along its centre line with the logo's pen."""
    from .legible import READERS
    from .reference import _SHAPES, family, part_library, part_strokes
    part_library()
    eng = engine_part(st, name)
    if eng is None:
        return []
    sample = SAMPLE[name]
    X = st.x_height
    found = []
    fonts = sorted(_SHAPES, key=lambda f: (family(f) not in READERS, f))
    for font in fonts:
        shapes = _SHAPES[font]
        key = f"{sample}|{name}" if f"{sample}|{name}" in shapes else next((k for k in shapes if k.endswith(f"|{name}")), None)
        if key is None:
            continue
        polys = [[(x * X, y * X) for x, y in ring] for ring in shapes[key]]
        strokes = part_strokes(polys)
        if not strokes:
            continue
        rings = []
        for s in strokes:
            pts = [(x, y) for x, y, _ in s]
            closed = len(pts) > 6 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.15 * X
            rings += pen_line(pts, st, closed=closed, smooth=False)
        if not rings:
            continue
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        body = unary_union([Polygon(r).buffer(0) for r in rings])
        rings = []
        for poly in getattr(body, "geoms", [body]):
            if poly.geom_type == "Polygon" and not poly.is_empty:
                rings.append([(round(x, 2), round(y, 2)) for x, y in list(poly.exterior.coords)[:-1]])
                rings += [[(round(x, 2), round(y, 2)) for x, y in list(h.coords)[:-1]] for h in poly.interiors]
        found.append({"polys": rings, "engine": eng["part"], "from": key.split("|")[0], "source": "reference",
                      "font": family(font), "line": centre_line(polys)})
        if len(found) >= limit:
            break
    return found
