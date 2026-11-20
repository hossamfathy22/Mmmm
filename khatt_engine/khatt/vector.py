"""A logo drawn in Illustrator, read as the designer built it: shapes of Bezier curves, not pixels.

A logo drawn in a vector program is a construction: rectangles and curves, copied and pasted, joined
(Pathfinder), sheared. Everything the engine estimates from a picture is exact in its file: the lines
of the grid are where anchor points share a height, a stem is the distance between two parallel straight
edges, the slant is the angle of the straight upright edges, the feel of the curves is the length of
their handles. And a file kept before its shapes are joined holds the parts themselves, one object each.

  read_svg(data)          the shapes of an SVG file (one per drawn object), y up, Bezier segments kept;
                          every transform (matrix, translate, scale, rotate, skew; on groups too) applied
  word_shapes(face, text) a word set in a typeface as shapes, one per letter (the examination's logos:
                          a typeface is a set of vector drawings whose every letter is known)
  dna(shapes)             the logo's style read from its geometry: the lines of the grid, the stem, the
                          hairline, the slant, the tension of its curves, how many of its turns are
                          corners, its dots
  raster(shapes, ...)     a mask of the shapes, when pixels are needed (matching the typed word)

Segments are tuples: ("L", p0, p1) or ("C", p0, c1, c2, p1); quadratic curves become cubic.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Pt = Tuple[float, float]
Seg = tuple
Contour = List[Seg]


@dataclass
class VShape:
    """One drawn object: its contours (a hole is a contour too), and where it came from."""
    contours: List[Contour]
    source: str = ""                    # the element's id, or the letter it belongs to
    index: Optional[int] = None         # the index in the typed text of the letter it belongs to, if known

    def points(self) -> np.ndarray:
        pts = [p for c in self.contours for s in c for p in s[1:]]
        return np.array(pts, float) if pts else np.zeros((0, 2))

    def bounds(self) -> Tuple[float, float, float, float]:
        p = self.points()
        return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())) if len(p) else (0, 0, 0, 0)


# ------------------------------------------------------------------ a pen that keeps the curves
class _SegPen:
    """A fontTools pen recording segments (quadratics raised to cubics), transformed by `m`."""

    def __init__(self, m=(1, 0, 0, 1, 0, 0)):
        self.m = m
        self.contours: List[Contour] = []
        self.cur: Contour = []
        self.start: Optional[Pt] = None
        self.last: Optional[Pt] = None

    def _t(self, p) -> Pt:
        a, b, c, d, e, f = self.m
        return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)

    def moveTo(self, p):
        self._flush()
        self.start = self.last = self._t(p)

    def lineTo(self, p):
        q = self._t(p)
        if self.last is not None and q != self.last:
            self.cur.append(("L", self.last, q))
        self.last = q

    def curveTo(self, *pts):
        ps = [self._t(p) for p in pts]
        # a run of off-curve points (the fontTools convention): split into cubic pieces
        while len(ps) > 3:
            a, b, c = ps[0], ps[1], ps[2]
            mid = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
            self.cur.append(("C", self.last, a, b, mid)); self.last = mid
            ps = ps[2:]
        if len(ps) == 3:
            self.cur.append(("C", self.last, ps[0], ps[1], ps[2])); self.last = ps[2]
        elif len(ps) == 2:
            self._quad(ps[0], ps[1])
        elif len(ps) == 1:
            self.lineTo_t(ps[0])

    def lineTo_t(self, q):
        if self.last is not None and q != self.last:
            self.cur.append(("L", self.last, q))
        self.last = q

    def _quad(self, c, p):
        p0 = self.last
        c1 = (p0[0] + 2 / 3 * (c[0] - p0[0]), p0[1] + 2 / 3 * (c[1] - p0[1]))
        c2 = (p[0] + 2 / 3 * (c[0] - p[0]), p[1] + 2 / 3 * (c[1] - p[1]))
        self.cur.append(("C", p0, c1, c2, p)); self.last = p

    def qCurveTo(self, *pts):
        ps = [self._t(p) if p is not None else None for p in pts]
        if ps and ps[-1] is None:                         # a closed contour of off-curve points only
            ps = ps[:-1]
            first = ((ps[-1][0] + ps[0][0]) / 2, (ps[-1][1] + ps[0][1]) / 2)
            self.start = self.last = first
            ps = ps + [first]
        offs, end = ps[:-1], ps[-1]
        for i, c in enumerate(offs):
            nxt = end if i == len(offs) - 1 else ((c[0] + offs[i + 1][0]) / 2, (c[1] + offs[i + 1][1]) / 2)
            self._quad(c, nxt)

    def closePath(self):
        if self.last is not None and self.start is not None and self.last != self.start:
            self.cur.append(("L", self.last, self.start))
        self._flush()

    def endPath(self):
        self._flush()

    def _flush(self):
        if len(self.cur) >= 2:
            self.contours.append(self.cur)
        self.cur = []

    def addComponent(self, glyphName, transformation):   # components are drawn by the glyph set itself
        raise NotImplementedError


# ------------------------------------------------------------------ SVG
def _mul(m, n):
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2, a * e2 + c * f2 + e, b * e2 + d * f2 + f)


def parse_transform(s: str):
    """An SVG transform list as one matrix (a, b, c, d, e, f)."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "matrix" and len(v) == 6:
            n = tuple(v)
        elif name == "translate":
            n = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            n = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "rotate":
            a = math.radians(v[0]); co, si = math.cos(a), math.sin(a)
            n = (co, si, -si, co, 0, 0)
            if len(v) == 3:
                n = _mul(_mul((1, 0, 0, 1, v[1], v[2]), n), (1, 0, 0, 1, -v[1], -v[2]))
        elif name == "skewX":
            n = (1, 0, math.tan(math.radians(v[0])), 1, 0, 0)
        elif name == "skewY":
            n = (1, math.tan(math.radians(v[0])), 0, 1, 0, 0)
        else:
            continue
        m = _mul(m, n)
    return m


SKIP = {"defs", "clipPath", "mask", "symbol", "metadata", "title", "desc", "style", "pattern", "marker"}


def read_svg(data) -> List[VShape]:
    """The shapes of an SVG file, one per drawn element, in a y-up frame (y = - the SVG's y)."""
    from fontTools.misc import etree
    from fontTools.svgLib.path import parse_path
    from fontTools.svgLib.path.shapes import PathBuilder
    if isinstance(data, (bytes, bytearray)):
        root = etree.fromstring(bytes(data), parser=etree.XMLParser())
    else:
        root = etree.parse(str(data), parser=etree.XMLParser()).getroot()
    out: List[VShape] = []
    flip = (1.0, 0.0, 0.0, -1.0, 0.0, 0.0)

    def tag(el):
        return el.tag.split("}")[-1] if isinstance(el.tag, str) else ""

    def hidden(el):
        st = (el.attrib.get("style") or "") + ";display:" + (el.attrib.get("display") or "")
        return "display:none" in st.replace(" ", "") or el.attrib.get("visibility") == "hidden"

    def walk(el, m):
        t = tag(el)
        if t in SKIP or hidden(el):
            return
        m = _mul(m, parse_transform(el.attrib.get("transform", "")))
        if t in ("path", "rect", "circle", "ellipse", "polygon", "polyline", "line"):
            if t in ("polyline", "line") and (el.attrib.get("fill", "") in ("none",)):
                return
            pb = PathBuilder()
            attrib = {k: v for k, v in el.attrib.items() if k != "transform"}
            clean = etree.Element(el.tag, attrib)
            if pb.add_path_from_element(clean) and pb.paths and pb.paths[0]:
                pen = _SegPen(_mul(flip, m))
                parse_path(pb.paths[0], pen)
                pen._flush()
                if pen.contours:
                    out.append(VShape(pen.contours, el.attrib.get("id", t)))
            return
        for ch in el:
            walk(ch, m)

    walk(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
    return out


def to_svg(shapes: Sequence[VShape], pad: float = 20.0, fill: str = "#111") -> str:
    """The shapes as an SVG file (each its own path, with its source as id), y down as SVG wants."""
    pts = np.vstack([s.points() for s in shapes if len(s.points())])
    x0, y0, x1, y1 = pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad

    def d_of(s: VShape) -> str:
        out = []
        for c in s.contours:
            p = c[0][1]
            out.append(f"M{p[0]:.2f},{-p[1]:.2f}")
            for seg in c:
                if seg[0] == "L":
                    out.append(f"L{seg[2][0]:.2f},{-seg[2][1]:.2f}")
                else:
                    out.append("C" + " ".join(f"{q[0]:.2f},{-q[1]:.2f}" for q in seg[2:]))
            out.append("Z")
        return "".join(out)
    body = "".join(f'<path id="{s.source or i}" fill-rule="evenodd" d="{d_of(s)}"/>' for i, s in enumerate(shapes))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.1f} {-y1:.1f} {x1 - x0:.1f} {y1 - y0:.1f}">'
            f'<g fill="{fill}">{body}</g></svg>')


# ------------------------------------------------------------------ typefaces as logos
def word_shapes(face, text: str) -> List[VShape]:
    """`text` set in a typeface (bench.fonts.Face), one shape per glyph, each knowing the index in the
    text of the letter it draws (HarfBuzz clusters), in font units, y up."""
    import uharfbuzz as hb
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(face.hb_font, buf, {})
    gs = face.tt.getGlyphSet()
    out: List[VShape] = []
    x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        name = face.tt.getGlyphName(info.codepoint)
        pen = _GlyphPen(gs, (1, 0, 0, 1, x + pos.x_offset, pos.y_offset))
        gs[name].draw(pen)
        pen._flush()
        if pen.contours:
            out.append(VShape(pen.contours, name, info.cluster))
        x += pos.x_advance
    return out


class _GlyphPen(_SegPen):
    """_SegPen that draws composite glyphs' components through the glyph set."""

    def __init__(self, glyphset, m):
        super().__init__(m)
        self.gs = glyphset

    def addComponent(self, glyphName, transformation):
        sub = _GlyphPen(self.gs, _mul(self.m, tuple(transformation)))
        self.gs[glyphName].draw(sub)
        sub._flush()
        self.contours += sub.contours


# ------------------------------------------------------------------ geometry
def flatten(c: Contour, tol: float = 0.5) -> np.ndarray:
    """A contour as a polygon, curves cut into pieces no longer than `tol` from the curve."""
    pts = []
    for s in c:
        if s[0] == "L":
            pts.append(s[1])
        else:
            p0, c1, c2, p1 = map(np.array, s[1:])
            n = max(2, int(math.ceil(math.sqrt((np.abs(p0 - 2 * c1 + c2).max() + np.abs(c1 - 2 * c2 + p1).max()) * 0.75 / max(tol, 1e-6)))))
            for t in np.linspace(0, 1, n, endpoint=False):
                m = 1 - t
                pts.append(tuple(m ** 3 * p0 + 3 * m * m * t * c1 + 3 * m * t * t * c2 + t ** 3 * p1))
    return np.array(pts, float)


def area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def raster(shapes: Sequence[VShape], scale: float, pad: int = 10, box=None):
    """A mask of the shapes (rows down): each shape filled even-odd, the shapes OR-ed (shapes drawn over
    one another are joined, as the eye sees them). Returns (mask, (x0, y1, scale)) to map back."""
    from PIL import Image, ImageDraw
    pts = np.vstack([s.points() for s in shapes if len(s.points())])
    x0, y0, x1, y1 = box or (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    W, H = int((x1 - x0) * scale) + 2 * pad + 1, int((y1 - y0) * scale) + 2 * pad + 1
    out = np.zeros((H, W), bool)
    for s in shapes:
        acc = np.zeros((H, W), bool)
        for c in s.contours:
            p = flatten(c, 0.35 / scale)
            if len(p) < 3:
                continue
            im = Image.new("L", (W, H), 0)
            ImageDraw.Draw(im).polygon([((x - x0) * scale + pad, (y1 - y) * scale + pad) for x, y in p], fill=1)
            acc ^= np.asarray(im, bool)
        out |= acc
    return out, (x0 - pad / scale, y1 + pad / scale, scale)


# ------------------------------------------------------------------ the style read from the geometry
def _cluster(vals: Sequence[float], w: Sequence[float], tol: float) -> List[Tuple[float, float, int]]:
    """1-D clusters: [(weighted centre, total weight, count)], heaviest first."""
    order = np.argsort(vals)
    out, cur = [], []
    for i in order:
        if cur and vals[i] - vals[cur[-1]] > tol:
            out.append(cur); cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    res = []
    for g in out:
        ww = np.array([w[i] for i in g]); vv = np.array([vals[i] for i in g])
        res.append((float((vv * ww).sum() / ww.sum()), float(ww.sum()), len(g)))
    return sorted(res, key=lambda r: -r[1])


def _is_line(s: Seg, tol: float) -> bool:
    if s[0] == "L":
        return True
    p0, c1, c2, p1 = map(np.array, s[1:])
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1e-9:
        return False
    n = np.array([-d[1], d[0]]) / L
    return abs(float(np.dot(c1 - p0, n))) < tol and abs(float(np.dot(c2 - p0, n))) < tol


def geometry(s: VShape, tol: float = 0.5):
    """A shape as a shapely geometry (its contours filled even-odd)."""
    from shapely.geometry import Polygon
    g = None
    for c in s.contours:
        p = flatten(c, tol)
        if len(p) < 3:
            continue
        q = Polygon(p).buffer(0)
        g = q if g is None else g.symmetric_difference(q)
    return g


def _turns(body: Sequence[np.ndarray]):
    """The heights where the outline turns: its bottoms (lower than the outline either side) and its tops."""
    lows, tops = [], []
    for p in body:
        y = p[:, 1]; n = len(p)
        if n < 5:
            continue
        k = max(2, n // 60)
        idx = np.arange(n)
        win = np.stack([y[(idx + j) % n] for j in range(-k, k + 1)])
        lo = y <= win.min(axis=0) + 1e-9
        hi = y >= win.max(axis=0) - 1e-9
        # a flat stretch is one turn, not one per point
        for mask, out in ((lo, lows), (hi, tops)):
            run = False
            for i in range(n):
                if mask[i] and not run:
                    out.append(float(y[i]))
                run = bool(mask[i])
    return lows, tops


TEETH = set("بتثنيیئپ")          # a tooth in their initial and medial forms
SEEN = set("سشصض")              # teeth in every form
TALL = set("اأإآلكکگطظ")        # rise above the x-height
HEADS = set("فقموؤهةعغ")         # a head as tall as the x-height, where there is no tooth


def _runs(g, x0, y0, x1, y1):
    """The pieces of ink a line from (x0, y0) to (x1, y1) crosses in geometry g: [(start, end)] along it."""
    from shapely.geometry import LineString
    seg = g.intersection(LineString([(x0, y0), (x1, y1)]))
    out = []
    for q in getattr(seg, "geoms", [seg]):
        if q.geom_type == "LineString" and not q.is_empty:
            (a, b), (c, d) = q.coords[0], q.coords[-1]
            out.append(tuple(sorted((a, c))) if x0 != x1 else tuple(sorted((b, d))))
    return out


def dna(shapes: Sequence[VShape], text: Optional[str] = None) -> Dict[str, float]:
    """The style of a logo from its geometry (its own units). The baseline is the height most of the
    strokes' bottoms share (the joins rest on it); the hairline the step from it to the tops of the joins;
    the x-height the height most of the other tops share (the teeth), below the tall letters; the stem and
    the slant are read off the straight upright edges that face each other across ink."""
    polys = [flatten(c, 0.5) for s in shapes for c in s.contours]
    polys = [p for p in polys if len(p) >= 3]
    allp = np.vstack(polys)
    H = float(np.ptp(allp[:, 1])) or 1.0
    areas = [abs(area(p)) for p in polys]
    amax = max(areas)
    small = [a < 0.03 * amax for a in areas]
    dots = [p for p, sm in zip(polys, small) if sm and 0.5 < (np.ptp(p[:, 0]) + 1e-9) / (np.ptp(p[:, 1]) + 1e-9) < 2.0]
    body = [p for p, sm in zip(polys, small) if not sm]
    out: Dict[str, float] = {"height": H}
    lows, tops = _turns(body)
    # the baseline: the most shared bottom (the descending bowls and tails are each alone at their depth)
    cl = _cluster(lows, [1.0] * len(lows), 0.02 * H)
    base = max(cl, key=lambda c: (c[2], c[0]))[0] if cl else float(np.percentile(allp[:, 1], 25))
    out["baseline"] = base
    ct = _cluster(tops, [1.0] * len(tops), 0.02 * H) if tops else []
    above = [c for c in ct if c[0] > base + 0.01 * H]
    # the hairline: the tops of the joins, the nearest shared top above the baseline
    joins = [c for c in above if c[0] < base + 0.2 * H and c[2] >= 2]
    if joins:
        out["hair"] = min(joins, key=lambda c: c[0])[0] - base
    hair = out.get("hair", 0.04 * H)
    tall = float(allp[:, 1].max()) if not body else float(max(p[:, 1].max() for p in body))
    # where the typed word says which letter each shape draws: the x-height is the top of its teeth (or of
    # its heads, where it has no tooth), the ascender the top of its tall letters (as the hand method did)
    known = {}
    if text:
        from .typeset import shape_forms
        forms = shape_forms(text)
        for sh in shapes:
            if sh.index is None or sh.index >= len(forms):
                continue
            c, f = forms[sh.index]
            bp = [q for q in (flatten(cc, 0.5) for cc in sh.contours) if len(q) >= 3 and abs(area(q)) >= 0.03 * amax]
            if not bp:
                continue
            top = float(max(q[:, 1].max() for q in bp))
            kind = ("tooth" if (c in TEETH and f in "IM") or c in SEEN else "tall" if c in TALL
                    else "head" if c in HEADS else None)
            if kind:
                known.setdefault(kind, []).append(top)
    if known.get("tooth") or known.get("head"):
        out["x_height"] = float(np.median(known.get("tooth") or known["head"])) - base
        out["x_from"] = "teeth" if known.get("tooth") else "heads"
    else:
        teeth = [c for c in above if c[0] > base + max(2.5 * hair, 0.12 * H) and c[0] < tall - 0.04 * H]
        if teeth:
            x = max(teeth, key=lambda c: (c[2], -c[0]))
            out["x_height"] = x[0] - base
            out["x_from"] = "tops"
    if known.get("tall"):
        out["ascender"] = float(np.median(known["tall"])) - base - out.get("x_height", 0.0)
    else:
        out["ascender"] = tall - base - out.get("x_height", 0.0)
    out["descender"] = base - float(min(p[:, 1].min() for p in body)) if body else 0.0
    # slant and stem: straight upright edges; a stem is two of them facing each other across ink
    X = out.get("x_height", 0.3 * H)
    ups = []
    for si, s in enumerate(shapes):
        for c in s.contours:
            for seg in c:
                if not _is_line(seg, 0.01 * H):
                    continue
                p0, p1 = np.array(seg[1]), np.array(seg[-1])
                d = p1 - p0
                L = float(np.hypot(*d))
                ang = (math.degrees(math.atan2(d[0], d[1])) + 90) % 180 - 90
                if L >= 0.15 * X and abs(ang) < 30:
                    ups.append((ang, L, p0, p1, si))
    if ups:
        angs = np.array([u[0] for u in ups]); ws = np.array([u[1] for u in ups])
        o = np.argsort(angs); cw = np.cumsum(ws[o])
        out["slant"] = round(float(angs[o][np.searchsorted(cw, cw[-1] / 2)]), 2)
    geos = {}
    widths = []
    for i, a in enumerate(ups):
        for b in ups[i + 1:]:
            if a[4] != b[4]:
                continue
            ya = sorted((a[2][1], a[3][1])); yb = sorted((b[2][1], b[3][1]))
            lo_, hi_ = max(ya[0], yb[0]), min(ya[1], yb[1])
            if hi_ - lo_ < 0.5 * min(ya[1] - ya[0], yb[1] - yb[0]):
                continue
            ym = 0.5 * (lo_ + hi_)
            xa = a[2][0] + (ym - a[2][1]) * (a[3][0] - a[2][0]) / ((a[3][1] - a[2][1]) or 1e-9)
            xb = b[2][0] + (ym - b[2][1]) * (b[3][0] - b[2][0]) / ((b[3][1] - b[2][1]) or 1e-9)
            d = abs(xa - xb)
            if not (0.01 * H < d < 0.35 * H):
                continue
            from shapely.geometry import Point
            g = geos.get(id(shapes[a[4]])) or geos.setdefault(id(shapes[a[4]]), geometry(shapes[a[4]]))
            if g is not None and g.contains(Point(0.5 * (xa + xb), ym)):
                widths.append(d)
    if widths:
        c = _cluster(widths, [1.0] * len(widths), 0.01 * H)
        out["stem"] = max(c, key=lambda q: (q[2], -q[0]))[0]
        out["stem_votes"] = float(max(c, key=lambda q: (q[2], -q[0]))[2])
    # the hairline: how thick the ink is where it rests on the baseline and stays thin (the joins), read
    # along upright lines through the whole word; and the stem across the middle of the x-height where the
    # straight edges did not give it (a calligraphic hand curves its stems)
    X = out.get("x_height", 0.3 * H)
    whole = None
    for sh in shapes:
        g = geos.get(id(sh)) or geometry(sh)
        if g is not None and not g.is_empty:
            whole = g if whole is None else whole.union(g)
    if whole is not None:
        x0, _, x1, _ = whole.bounds
        thin = []
        for x in np.linspace(x0, x1, 400):
            for a, b in _runs(whole, x, base - 0.5 * H, x, base + 1.5 * H):
                if abs(a - base) < 0.03 * H and b - a < 0.45 * X:
                    thin.append(b - a)
        if len(thin) >= 5:
            out["hair_scan"] = float(np.median(thin))
            if "hair" not in out or out["hair"] <= 0:
                out["hair"] = out["hair_scan"]
        if "stem" not in out:
            ym = base + 0.5 * X
            runs = [b - a for a, b in _runs(whole, x0 - 1, ym, x1 + 1, ym) if b - a > 0.01 * H]
            if runs:
                c = _cluster(runs, [1.0] * len(runs), 0.03 * X)
                out["stem"] = max(c, key=lambda q: (q[2], -q[0]))[0]
                out["stem_from"] = "scan"
    # the feel of the curves: handle lengths over the chord, and the share of turns that are corners
    ten, corners, turns = [], 0, 0
    for s in shapes:
        for c in s.contours:
            for seg in c:
                if seg[0] == "C" and not _is_line(seg, 0.01 * H):
                    p0, c1, c2, p1 = map(np.array, seg[1:])
                    ch = float(np.hypot(*(p1 - p0)))
                    if ch > 0.02 * H:
                        ten.append((float(np.hypot(*(c1 - p0))) + float(np.hypot(*(p1 - c2)))) / ch)
            for a, b in zip(c, c[1:] + c[:1]):
                ta = np.array(a[-1]) - np.array(a[-2]); tb = np.array(b[2]) - np.array(b[1])
                if np.hypot(*ta) < 1e-9 or np.hypot(*tb) < 1e-9:
                    continue
                turns += 1
                if float(np.dot(ta, tb) / (np.hypot(*ta) * np.hypot(*tb))) < math.cos(math.radians(30)):
                    corners += 1
    if ten:
        out["tension"] = float(np.median(ten))
    if turns:
        out["corners"] = corners / turns
    if dots:
        out["dot"] = float(np.median([math.sqrt(abs(area(p))) for p in dots]))
        out["dot_fill"] = float(np.median([abs(area(p)) / ((np.ptp(p[:, 0]) * np.ptp(p[:, 1])) or 1) for p in dots]))
    return out
