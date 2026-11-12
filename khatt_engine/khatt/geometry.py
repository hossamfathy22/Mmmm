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


def _engine_stem(st: Style, xc: float, top: float, bottom: float, entry: bool, exit: bool,
                 round_br: bool, round_bl: bool) -> Shape:
    u, h, F = st.stem / 2, st.hair, st.fillet
    R = u
    xl, xr = xc - u, xc + u
    on_base = abs(bottom) < 1e-6
    p: Poly = [(xl, top), (xr, top)]
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
    cx, cy, a, b = (xl + xr) / 2, (yb + yt) / 2, (xr - xl) / 2, (yt - yb) / 2
    outer = _super(cx, cy, a, b)
    ai, bi = a - side, b - st.hair
    if ai < 1 or bi < 1:
        return [outer]
    return [outer, _super(cx, cy, ai, bi)]


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


def circle(cx: float, cy: float, r: float, tan: float = 0.0, n: int = 48) -> Shape:
    """A dot. Pre-compensated for the slant so it stays perfectly round after shearing."""
    pts = []
    for i in range(n):
        c, s = math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)
        pts.append((cx + r * c - r * s * tan, cy + r * s))
    return [pts]
