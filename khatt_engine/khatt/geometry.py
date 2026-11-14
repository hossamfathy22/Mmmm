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


# ---- key shapes drawn as explicit outlines (outer edge and counter) ---------------------------
def _pts(seg) -> Poly:
    return [(q[0], q[1]) for q in seg]


def loop_length(st: Style) -> float:
    """Length of the loop of ص ط, from the foot of its stem to its round end: the slope needs 0.8 of
    the x-height to come down at about 45 degrees, the round end 1.35 stems. (recipes: loop.length)"""
    from .recipes import P
    return P(st, "loop.length")


def teardrop(st: Style, x0: float, xr: float) -> Shape:
    """The loop of ص ض ط ظ. `x0` is the right edge of the tooth (or tall stem) that stands at its left
    point, `xr` its right end. Outer edge: flat on the baseline, a round end on the right, the top at
    the x-height, then a slope down to the foot of the stem. Counter: flat bottom one hairline up,
    upright right side one stem in, a slope parallel to the outer one, a point at the stem.
    The widths are the pen rule's: hairline below and at the top, the stem at the round end, the
    width of a leaning stroke on the slope."""
    U, h, X = st.stem, st.hair, st.x_height
    L = xr - x0
    ax = xr - 1.35 * U                                   # top of the round end
    S = max(ax - x0, 0.3 * L)                            # horizontal run of the slope
    th = math.atan2(0.8 * (X - 2 * h), max(S - 0.22 * U, 1e-6))
    w = st.width_at(math.sin(th))
    taper = 1.0 - 0.39 * (1.0 - h / U) / (1.0 - 4.5 / 23.0) if U > h else 1.0   # a contrast pen thins at the foot
    yf = min(h + taper * w / math.cos(th), 0.6 * X)       # where the slope meets the stem
    outer = ([(x0 - U / 2, 0.0)] + _pts(cubic((xr - 1.3 * U, 0), (xr - 0.565 * U, 0), (xr, 0.2 * X), (xr, 0.5 * X)))
             + _pts(cubic((xr, 0.5 * X), (xr, 0.8 * X), (xr - 0.565 * U, X), (ax, X)))[1:]
             + _pts(cubic((ax, X), (ax - 0.34 * S, X), (x0 + 0.45 * S, yf + 0.47 * S * math.tan(th)), (x0, yf)))[1:]
             + [(x0 - U / 2, yf)])
    tip = (x0 + 0.09 * U, h)
    full = (X - 2 * h) / math.sin(th)                    # the counter's slope, if it ran straight to the top
    d = (math.cos(th), math.sin(th))
    e = (tip[0] + d[0] * 0.47 * full, tip[1] + d[1] * 0.47 * full)
    c = (tip[0] + d[0] * 0.64 * full, tip[1] + d[1] * 0.64 * full)
    top_in = (ax - 0.13 * U, X - h)
    if xr - U - 0.3 * U <= tip[0] or X - 2 * h < 0.15 * U:
        return [outer]                                   # too heavy for a counter
    inner = ([tip] + _pts(cubic((xr - 1.48 * U, h), (xr - 1.13 * U, h), (xr - U, 0.21 * X), (xr - U, 0.455 * X)))
             + _pts(cubic((xr - U, 0.455 * X), (xr - U, 0.7 * X), (xr - 1.13 * U, X - h), top_in))[1:]
             + _pts(cubic(top_in, (top_in[0] - 0.19 * S, X - h), c, e))[1:])
    from shapely.geometry import Polygon
    o, i = Polygon(outer).buffer(0), Polygon(inner).buffer(0)
    i = i.intersection(o.buffer(-0.5 * h))               # the counter never breaks the outer edge
    if i.is_empty or i.area < 0.2 * U * U:
        return [outer]
    big = max(getattr(i, "geoms", [i]), key=lambda q: q.area)
    return [outer, list(big.exterior.coords)[:-1]]


def nose_top(st: Style, d0: float, xa: float) -> Poly:
    """The upper half of ح ج خ: a hairline bar from `xa` at the x-height that turns down into the nose,
    whose right end is `d0`. It stops where the pen is upright (0.636 of the x-height): the lower arm
    takes over from that cut, d0 - stem .. d0."""
    U, h, X = st.stem, st.hair, st.x_height
    yn = 0.636 * X
    return ([(xa, X)] + _pts(cubic((d0 - 1.13 * U, X), (d0 - 0.435 * U, X), (d0, 0.864 * X), (d0, yn)))
            + _pts(cubic((d0 - U, yn), (d0 - U, 0.803 * X), (d0 - 1.3 * U, X - h), (d0 - 1.65 * U, X - h)))
            + [(xa, X - h)])


def nose_arm(st: Style, d0: float, xj: float, ex: float) -> Poly:
    """The lower arm of حـ ـحـ: from the nose it sweeps back down to the baseline at `xj`, thinning as it
    leans, and leaves as a hairline to `ex`."""
    U, h, X = st.stem, st.hair, st.x_height
    yn = 0.636 * X
    run = d0 - xj
    return ([(d0, yn + 0.01)] + _pts(cubic((d0, yn), (d0, 0.303 * X), (d0 - 0.387 * run, 0), (xj, 0)))
            + [(ex, 0), (ex, h)]
            + _pts(cubic((xj + 0.065 * run, h), (d0 - 0.6 * run, h), (d0 - U, 0.333 * X), (d0 - U, yn)))
            + [(d0 - U, yn + 0.01)])


def nose_turn(st: Style, d0: float, m: Pt, beta: float) -> Poly:
    """The lower arm of ح ـح where it leaves the nose: from the cut d0 - stem .. d0 it turns to the left
    and reaches the cross-section of a pen stroke centred on `m` and leaning `beta` degrees below the
    horizontal; the pen stroke of the bowl goes on from there. The outer edge is a round turn, the
    inner edge drops straight and then turns, so the white between bar and arm ends round, not in a
    sharp wedge."""
    U, X = st.stem, st.x_height
    yn = 0.636 * X
    b = math.radians(beta)
    t = (-math.cos(b), -math.sin(b))
    n = (math.sin(b), -math.cos(b))                      # towards the outer (lower right) edge
    w = st.width_at(math.sin(b))
    lo = (m[0] + n[0] * w / 2, m[1] + n[1] * w / 2)
    hi = (m[0] - n[0] * w / 2, m[1] - n[1] * w / 2)
    run = d0 - m[0]
    return ([(d0, yn + 0.01)] + _pts(cubic((d0, yn), (d0, yn - 0.3 * X), (lo[0] - t[0] * 0.7 * run, lo[1] - t[1] * 0.7 * run), lo))
            + _pts(cubic(hi, (hi[0] - t[0] * 0.5 * run, hi[1] - t[1] * 0.5 * run), (d0 - U, yn - 0.16 * X), (d0 - U, yn)))
            + [(d0 - U, yn + 0.01)])


def soften(shape: Shape, r: float) -> Shape:
    """Round the outward corners of a shape with radius r (its counters keep theirs)."""
    from shapely.geometry import Polygon
    if r <= 0 or not shape:
        return shape
    outer = Polygon(shape[0]).buffer(0)
    rounded = outer.buffer(-r, 12).buffer(r, 12)
    if rounded.is_empty or rounded.geom_type != "Polygon":
        return shape
    return [list(rounded.exterior.coords)[:-1]] + list(shape[1:])


def fuse(shapes: Sequence[Shape], r: float) -> Shape:
    """Merge shapes into one and fill the sharp white wedges where two of them meet at a shallow angle
    (narrower than 2r), the way ink closes them."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    geoms = []
    for sh in shapes:
        acc = None
        for poly in sh:
            if len(poly) >= 3:
                q = Polygon(poly).buffer(0)
                acc = q if acc is None else acc.symmetric_difference(q)
        if acc is not None:
            geoms.append(acc)
    body = unary_union(geoms)
    if r > 0:
        body = body.buffer(r, 12).buffer(-r, 12)
    out: Shape = []
    for poly in getattr(body, "geoms", [body]):
        if poly.is_empty:
            continue
        out.append(list(poly.exterior.coords)[:-1])
        out += [list(q.coords)[:-1] for q in poly.interiors]
    return out


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


def solid_stroke(st: Style, pts: List[Sample]) -> Shape:
    """The same pen stroke, built piece by piece and merged: where the path turns more tightly than
    half the pen (the inside of a sharp turn) the offset edge would fold over itself and leave a notch;
    the merged pieces fill it, as ink does."""
    from shapely.geometry import MultiPoint
    from shapely.ops import unary_union
    L, R = [], []
    for x, y, tx, ty in pts:
        w = st.width_at(ty) / 2
        L.append((x - ty * w, y + tx * w)); R.append((x + ty * w, y - tx * w))
    quads = [MultiPoint([L[i], L[i + 1], R[i + 1], R[i]]).convex_hull for i in range(len(pts) - 1)]
    body = unary_union([q for q in quads if q.geom_type == "Polygon"]).buffer(0)
    out: Shape = []
    for poly in getattr(body, "geoms", [body]):
        if poly.is_empty:
            continue
        out.append(list(poly.exterior.coords)[:-1])
        out += [list(r.coords)[:-1] for r in poly.interiors]
    return out


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
