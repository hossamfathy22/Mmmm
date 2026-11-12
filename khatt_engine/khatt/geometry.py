"""Drawing primitives. Coordinates are upright (y up, baseline at 0); slant is applied at render time.

Every primitive returns a *shape*: a list of closed polygons filled with the even-odd rule,
so a ring is [outer, inner].
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .style import Style

Pt = Tuple[float, float]
Poly = List[Pt]
Shape = List[Poly]


def arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int = 10) -> Poly:
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def rect(x0: float, x1: float, y0: float, y1: float) -> Shape:
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]


def stem(st: Style, xc: float, top: float, bottom: float = 0.0, *, entry: bool = False,
         exit: bool = False, round_br: bool = False, round_bl: bool = False) -> Shape:
    """A vertical stem. `entry`/`exit` add the concave fillet that flows into a baseline join
    on the right/left; `round_*` round an outer bottom corner.
    When a logo part library is active, the stem is assembled from the logo's own pieces."""
    from .library import ACTIVE
    lib = ACTIVE.get()
    if lib is not None and abs(bottom) < 1e-6:
        built = lib.build_stem(xc, top, entry, exit,
                         lambda t: _engine_stem(st, xc, t, 0.0, entry, exit, round_br, round_bl))
        if built is not None:
            return built
    return _engine_stem(st, xc, top, bottom, entry, exit, round_br, round_bl)


def clip_x(poly: Poly, c: float, keep_right: bool) -> Poly:
    """Sutherland-Hodgman clip of one polygon against the half-plane x >= c (or x <= c)."""
    inside = (lambda p: p[0] >= c) if keep_right else (lambda p: p[0] <= c)
    out: Poly = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        pin, qin = inside(p), inside(q)
        if pin:
            out.append(p)
        if pin != qin:
            t = (c - p[0]) / (q[0] - p[0]) if q[0] != p[0] else 0.0
            out.append((c, p[1] + t * (q[1] - p[1])))
    return out


def top_edge(st: Style, xc: float, top: float) -> Poly:
    """The top of a stem, left to right: cut at the logo's angle, corners rounded by top_round."""
    u = st.stem / 2
    ta = math.tan(math.radians(st.top_angle))
    r = max(0.0, min(1.0, st.top_round)) * u
    pts: Poly = []
    if r < 0.3:
        pts = [(xc - u, top), (xc + u, top)]
    else:
        pts += [(xc - u + r - r * math.cos(math.radians(a)), top - r + r * math.sin(math.radians(a))) for a in range(0, 91, 10)]
        pts += [(xc + u - r + r * math.cos(math.radians(a)), top - r + r * math.sin(math.radians(a))) for a in range(90, -1, -10)]
    return [(x, y + ta * (x - xc)) for x, y in pts]


def bowl_top(st: Style) -> float:
    return max(st.join_in, st.join_out) * st.x_height


def bowl(st: Style, xr: float, xl: float) -> Shape:
    """The join between two teeth (stem centres xr > xl): thick sides, hairline at the bottom.
    The outer curve starts at join_out x X on the outer edges and reaches the baseline (minus the
    overshoot) in the middle; the inner curve starts at join_in x X and reaches the hairline."""
    u, h, U, X = st.stem / 2, st.hair, st.stem, st.x_height
    hin, hout, ov = st.join_in * X, st.join_out * X, st.overshoot
    xm = (xr + xl) / 2
    pts: Poly = []
    for seg in (cubic((xr + u, hout), (xr + u, 0.3 * hout - ov), (xr + 0.087 * U, -ov), (xm, -ov)),
                cubic((xm, -ov), (xl - 0.087 * U, -ov), (xl - u, 0.3 * hout - ov), (xl - u, hout)),
                line((xl - u, hout), (xl + u, hin), 2),
                cubic((xl + u, hin), (xl + u, 0.43 * hin), (xl + u + 0.174 * U, h), (xm, h)),
                cubic((xm, h), (xr - u - 0.174 * U, h), (xr - u, 0.43 * hin), (xr - u, hin)),
                line((xr - u, hin), (xr + u, hout), 2)):
        pts += [(p[0], p[1]) for p in seg]
    return [pts]


def _engine_stem(st: Style, xc: float, top: float, bottom: float, entry: bool, exit: bool,
                 round_br: bool, round_bl: bool) -> Shape:
    """Engine stem. A stem joined at the baseline draws its own half of the join bowl on each
    joined side; two neighbours' halves meet exactly in the middle of the counter."""
    u = st.stem / 2
    on_base = abs(bottom) < 1e-6
    if on_base and (entry or exit) and top > bowl_top(st) + 0.2 * st.stem:
        cut = min(st.join_in, st.join_out) * st.x_height - 0.1 * st.stem   # the stem reaches into the bowl
        polys: Shape = [top_edge(st, xc, top) + [(xc + u, cut), (xc - u, cut)]]
        P = st.pitch
        if exit:                      # joined on the left: right half of bowl(xc, xc - P)
            b = bowl(st, xc, xc - P)[0]
            polys.append(clip_x(b, xc - P / 2 - 0.04 * P, True))
        if entry:                     # joined on the right: left half of bowl(xc + P, xc)
            b = bowl(st, xc + P, xc)[0]
            polys.append(clip_x(b, xc + P / 2 + 0.04 * P, False))
        from .library import Compound, Tagged
        names = ["جذع"] + ["كأس الوصل"] * (len(polys) - 1)
        return Compound([Tagged([p], "engine", n) for p, n in zip(polys, names) if len(p) >= 3])   # unioned
    return _fillet_stem(st, xc, top, bottom, entry, exit, round_br, round_bl)


def _fillet_stem(st: Style, xc: float, top: float, bottom: float, entry: bool, exit: bool,
                 round_br: bool, round_bl: bool) -> Shape:
    u, h, F = st.stem / 2, st.hair, st.fillet
    R = u
    xl, xr = xc - u, xc + u
    on_base = abs(bottom) < 1e-6
    p: Poly = top_edge(st, xc, top)
    if on_base and entry:
        p.append((xr, h + F)); p += arc(xr + F, h + F, F, 180, 270)[1:]; p.append((xr + F, 0))
    elif on_base and round_br:
        p.append((xr, R)); p += arc(xr - R, R, R, 0, -90)[1:]
    else:
        p.append((xr, bottom))
    if on_base and exit:
        p.append((xl - F, 0)); p.append((xl - F, h)); p += arc(xl - F, h + F, F, 270, 360)[1:]
    elif on_base and round_bl:
        p.append((xl + R, 0)); p += arc(xl + R, R, R, 270, 180)[1:]
    else:
        p.append((xl, bottom))
    return [p]


def connector(st: Style, x_left: float, x_right: float) -> Shape:
    return rect(min(x_left, x_right), max(x_left, x_right), 0, st.hair)


def _super(cx, cy, a, b, n=2.3, N=180) -> Poly:
    out = []
    for i in range(N):
        t = 2 * math.pi * i / N
        c, s = math.cos(t), math.sin(t)
        out.append((cx + a * math.copysign(abs(c) ** (2 / n), c), cy + b * math.copysign(abs(s) ** (2 / n), s)))
    return out


def ring(st: Style, xl: float, xr: float, yb: float, yt: float, side: float | None = None) -> Shape:
    """A closed head (و ف ق ه م ...): sides as thick as the stem, top and bottom hairline."""
    from .library import ACTIVE
    lib = ACTIVE.get()
    if lib is not None:
        got = lib.loop(xl, xr, yb, yt)
        if got is not None:
            return got
    side = st.stem if side is None else side
    ov = st.overshoot
    if abs(yb) < 1e-6:
        yb -= ov
    if yt >= 0.99 * st.x_height:
        yt += ov
    cx, cy, a, b = (xl + xr) / 2, (yb + yt) / 2, (xr - xl) / 2, (yt - yb) / 2
    n = max(1.5, st.ring_n)
    outer = _super(cx, cy, a, b, n)
    ai, bi = a - side, b - st.hair
    if ai < 1 or bi < 1:
        return [outer]
    return [outer, _super(cx, cy, ai, bi, n)]


# ---- variable-width strokes (tails, bowls) -------------------------------------------------
Sample = Tuple[float, float, float, float]  # x, y, unit tangent


def cubic(p0: Pt, c1: Pt, c2: Pt, p1: Pt, n: int = 40) -> List[Sample]:
    out = []
    for i in range(n + 1):
        t = i / n; m = 1 - t
        x = m**3 * p0[0] + 3*m*m*t * c1[0] + 3*m*t*t * c2[0] + t**3 * p1[0]
        y = m**3 * p0[1] + 3*m*m*t * c1[1] + 3*m*t*t * c2[1] + t**3 * p1[1]
        dx = 3*m*m*(c1[0]-p0[0]) + 6*m*t*(c2[0]-c1[0]) + 3*t*t*(p1[0]-c2[0])
        dy = 3*m*m*(c1[1]-p0[1]) + 6*m*t*(c2[1]-c1[1]) + 3*t*t*(p1[1]-c2[1])
        L = math.hypot(dx, dy) or 1.0
        out.append((x, y, dx / L, dy / L))
    return out


def line(a: Pt, b: Pt, n: int = 12) -> List[Sample]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1.0
    return [(a[0] + dx * i / n, a[1] + dy * i / n, dx / L, dy / L) for i in range(n + 1)]


def path(*segments: Sequence[Sample]) -> List[Sample]:
    pts: List[Sample] = []
    for s in segments:
        pts += list(s) if not pts else list(s)[1:]
    return pts


def stroke(st: Style, pts: List[Sample]) -> Shape:
    """Offset a centreline by the pen rule: width depends on direction only."""
    L, R = [], []
    for x, y, tx, ty in pts:
        w = st.width_at(ty) / 2
        nx, ny = -ty, tx
        L.append((x + nx * w, y + ny * w)); R.append((x - nx * w, y - ny * w))
    return [L + R[::-1]]


def open_arc(st: Style, cx: float, cy: float, a: float, b: float, a0: float, a1: float, scale: float = 1.0) -> Shape:
    """Stroke along part of an ellipse (used for the head of ع and for hamza)."""
    pts = []
    n = 48
    for i in range(n + 1):
        t = math.radians(a0 + (a1 - a0) * i / n)
        x, y = cx + a * math.cos(t), cy + b * math.sin(t)
        dx, dy = -a * math.sin(t), b * math.cos(t)
        if a1 < a0:
            dx, dy = -dx, -dy
        L = math.hypot(dx, dy) or 1.0
        pts.append((x, y, dx / L, dy / L))
    if scale == 1.0:
        return stroke(st, pts)
    small = Style(**{**st.__dict__, "stem": st.stem * scale, "hair": max(1.0, st.hair * scale)})
    return stroke(small, pts)


def circle(cx: float, cy: float, r: float, tan: float = 0.0, n: int = 48, shape_n: float = 2.0, rot: float = 0.0) -> Shape:
    """A dot: a superellipse (2 = circle, larger = square), optionally turned (45 = diamond).
    Pre-compensated for the slant so it keeps its shape after shearing."""
    pts = []
    e = 2.0 / max(1.2, shape_n)
    cr, sr = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    for i in range(n):
        c, s = math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)
        x, y = r * math.copysign(abs(c) ** e, c), r * math.copysign(abs(s) ** e, s)
        x, y = x * cr - y * sr, x * sr + y * cr
        pts.append((cx + x - y * tan, cy + y))
    return [pts]



def angular_stroke(st: Style, pts: Sequence[Pt], r_out: float = 0.0) -> Shape:
    """Straight strokes meeting at corners, the way a geometric logo draws its tails and cups.
    Each stroke takes its pen width from its direction (vertical = stem, horizontal = hairline),
    strokes overlap fully at the corners, and only the outer corner of each turn is rounded with
    the logo's radius `r_out`; the stroke ends stay square."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    segs = []
    for a, b in zip(pts, pts[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dx, dy)
        if ln < 1e-6:
            continue
        u = (dx / ln, dy / ln)
        segs.append((a, b, u, st.width_at(u[1])))
    if not segs:
        return []
    polys = []
    for i, (a, b, u, w) in enumerate(segs):
        ea = segs[i - 1][3] / 2 if i > 0 else 0.0
        eb = segs[i + 1][3] / 2 if i < len(segs) - 1 else 0.0
        A = (a[0] - u[0] * ea, a[1] - u[1] * ea)
        B = (b[0] + u[0] * eb, b[1] + u[1] * eb)
        n = (-u[1] * w / 2, u[0] * w / 2)
        polys.append(Polygon([(A[0] + n[0], A[1] + n[1]), (B[0] + n[0], B[1] + n[1]),
                              (B[0] - n[0], B[1] - n[1]), (A[0] - n[0], A[1] - n[1])]))
    # each joint is filled (a bevel between the two stroke ends), so a sharp turn leaves no wedge
    for i in range(1, len(segs)):
        (_, b, u1, w1), (_, _, u2, w2) = segs[i - 1], segs[i]
        n1 = (-u1[1] * w1 / 2, u1[0] * w1 / 2)
        n2 = (-u2[1] * w2 / 2, u2[0] * w2 / 2)
        joint = Polygon([(b[0] + n1[0], b[1] + n1[1]), (b[0] + n2[0], b[1] + n2[1]),
                         (b[0] - n1[0], b[1] - n1[1]), (b[0] - n2[0], b[1] - n2[1])]).convex_hull
        polys.append(joint)
    shape = unary_union(polys)
    for i in range(1, len(segs)):
        (_, b, u1, w1), (_, _, u2, w2) = segs[i - 1], segs[i]
        r = min(r_out, w1 + 0.5 * w2, w2 + 0.5 * w1)
        if r <= 0.5:
            continue
        # the outer corner: past the turn along the incoming stroke, behind it along the outgoing one
        O = (b[0] + u1[0] * w2 / 2 - u2[0] * w1 / 2, b[1] + u1[1] * w2 / 2 - u2[1] * w1 / 2)
        C = (O[0] - u1[0] * r + u2[0] * r, O[1] - u1[1] * r + u2[1] * r)
        cutter = Polygon([O, (O[0] - u1[0] * r, O[1] - u1[1] * r), C, (O[0] + u2[0] * r, O[1] + u2[1] * r)])
        shape = shape.difference(cutter.difference(Point(C).buffer(r, 32)))
    out: Shape = []
    for poly in getattr(shape, "geoms", [shape]):
        if poly.is_empty:
            continue
        out.append(list(poly.exterior.coords)[:-1])
        out += [list(h.coords)[:-1] for h in poly.interiors]
    return out
