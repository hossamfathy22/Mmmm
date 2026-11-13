"""Letter forms built from six parts: stem, baseline join, head (ring), tail/bowl, dot, dash.

Each builder draws one letter in local coordinates: the right edge of the letter body is x = 0,
the letter grows to the left (negative x), baseline is y = 0.
`form` is one of: "i" isolated, "I" initial, "M" medial, "F" final.
"""
from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import geometry as g
from .library import ACTIVE, Compound, Tagged
from .style import Style


def _lib():
    return ACTIVE.get()


def _named(shape, name: str):
    """Tag an engine-drawn shape with the name of the part it plays."""
    if isinstance(shape, (Tagged, Compound)):
        return shape
    return Tagged(shape, "engine", name)


def _ring(st, xl, xr, yb, yt):
    return _named(g.ring(st, xl, xr, yb, yt), "رأس")


@dataclass
class Glyph:
    shapes: List[g.Shape]
    left: float                      # leftmost x of the body (tails excluded), used for spacing
    entry_x: float = 0.0             # where a join coming from the right must reach
    entry_edge: float = 0.0          # visible right edge at the baseline
    exit_x: Optional[float] = None   # where a join leaving to the left starts
    exit_edge: Optional[float] = None
    stems: List[Tuple[float, float, float]] = field(default_factory=list)  # (x centre, top, bottom) for drawings
    tail: Optional[float] = None     # leftmost x of a tail that runs under the next letter
    below: Optional[Tuple[float, float]] = None  # x range of marks under the baseline


def _e(form: str) -> bool:
    return form in ("M", "F")


def _x(form: str) -> bool:
    return form in ("I", "M") and "no_exit" not in JOIN.get()


# How a letter joins, when the logo joins some letters differently from the baseline join:
# "no_exit"   the letter keeps its joining shape but leaves nothing at the baseline for the next one
# "top_entry" the letter is reached by a bar at the top of the previous tooth (ر ز ژ in a logo whose
#             tails rest on the baseline, as in a geometric kufi)
JOIN: ContextVar = ContextVar("khatt_join", default=frozenset())


def top_joins(st: Style) -> bool:
    """ر is joined from the top when the logo itself does so (measured: `top_join`), or when it draws
    with corners and its ر tails rest on the baseline."""
    return st.top_join >= 0.5 or (st.angular >= 0.5 and st.tail_drop < 0.3)


# ---------------------------------------------------------------- marks (dots, dashes, hamza)
def marks(st: Style, spec: Optional[str], cx: float, above_ref: Optional[float] = None,
          below_ref: float = 0.0, at: Optional[float] = None) -> List[g.Shape]:
    """spec: a1/a2/a3 dots above, b1/b2/b3 below, ha/hb hamza, md madda.
    `at` forces the centre height (used for dots inside a bowl)."""
    if not spec:
        return []
    d, h = st.dot, st.hair
    gap = 0.44 * d
    above = spec[0] in "ah" and spec != "hb"
    ref = st.x_height if above_ref is None else above_ref
    cy = at if at is not None else (ref + gap + d / 2 if above else below_ref - gap - d / 2)
    out: List[g.Shape] = []
    kind = spec[1:] if spec[0] in "ab" else spec
    lib = _lib()

    def dot(x, y):
        got = lib.dot(x, y) if lib is not None else None
        return got if got is not None else _named(g.circle(x, y, d / 2, st.tan, shape_n=st.dot_n, rot=st.dot_rot), "نقطة")

    def dash(x, y, length):
        got = lib.dash(x, y, length) if lib is not None else None
        return got if got is not None else _named(g.rect(x - length / 2, x + length / 2, y - h / 2, y + h / 2), "شرطة")
    if kind == "1":
        out.append(dot(cx, cy))
    elif kind == "2":
        if st.dash_marks:
            out.append(dash(cx, cy, 1.5 * d))
        else:
            out += [dot(cx - 0.6 * d, cy), dot(cx + 0.6 * d, cy)]
    elif kind == "3":
        step = 0.95 * d if above else -0.95 * d
        if st.dash_marks:
            out.append(dash(cx, cy, 1.5 * d))
            out.append(dot(cx, cy + step * 0.75 + (d / 2 - h / 2) * (1 if above else -1) * 0.2))
        else:
            out += [dot(cx - 0.6 * d, cy), dot(cx + 0.6 * d, cy), dot(cx, cy + step)]
    elif spec in ("ha", "hb"):
        hz = g.open_arc(st, cx, cy, 0.42 * d, 0.5 * d, 40, 320, scale=0.55)
        if at is None:
            # measured, not guessed: the drawn hamza keeps a white of 0.3 stem from its letter
            ys = [y for poly in hz for _, y in poly]
            dy = (ref + 0.3 * st.stem - min(ys)) if above else (below_ref - 0.3 * st.stem - max(ys))
            if (above and dy > 0) or (not above and dy < 0):
                hz = [[(x, y + dy) for x, y in poly] for poly in hz]
        out.append(_named(hz, "همزة"))
    elif spec == "md":
        out.append(dash(cx, cy, 1.8 * d))
    return out


def clear_marks(st: Style, body: List[g.Shape], spec: Optional[str], spots: List[Tuple[float, float]]) -> List[g.Shape]:
    """Put the marks at the first spot where they keep their own white (a fifth of the stem)
    from the letter: the critic's 'touch' test, applied while drawing."""
    if not spec:
        return []
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .typeset import flat
    polys = [Polygon(p).buffer(0) for s in flat(body) for p in s if len(p) >= 3]
    ink = unary_union(polys).buffer(0.2 * st.stem) if polys else None
    best = None
    for x, y in spots:
        m = marks(st, spec, x, at=y)
        mp = unary_union([Polygon(p).buffer(0) for s in m for p in s if len(p) >= 3])
        if ink is None or not mp.intersects(ink):
            return m
        if best is None:
            best = m
    return best or []


# ---------------------------------------------------------------- shared parts
def teeth(st: Style, n: int, entry: bool, exit: bool, top: Optional[float] = None,
          x0: float = 0.0, drop_inner: bool = False) -> Tuple[List[g.Shape], List[float]]:
    """n teeth joined at the baseline, right to left, starting at right edge x0."""
    u, P = st.stem / 2, st.pitch
    top = st.x_height if top is None else top
    xs = [x0 - u - i * P for i in range(n)]
    shapes: List[g.Shape] = []
    for i, xc in enumerate(xs):
        ent = entry if i == 0 else True
        ex = exit if i == n - 1 else True
        t = top - st.seen_drop if (drop_inner and i > 0) else top
        shapes.append(g.stem(st, xc, t, entry=ent, exit=ex,
                             round_br=(i == 0 and not entry), round_bl=(i == n - 1 and not exit)))
        if i > 0:
            shapes.append(g.connector(st, xc + u, xs[i - 1] - u))
    return shapes, xs


def bowl_down(st: Style, r: float, top: float, width: float, left_top: float) -> g.Shape:
    """Arm at x = r goes down below the baseline, across, and up on the left (ن ل ق س ص ي)."""
    D, h, X = st.descender, st.hair, st.x_height
    Lx, yb, k = r - width, -D + h / 2, 0.08 * X
    if st.angular >= 0.5:      # a logo that turns with corners: a square cup
        return _named(g.angular_stroke(st, [(r, top), (r, yb), (Lx, yb), (Lx, left_top)], st.corner_r), "كأس سفلي")
    return _named(g.stroke(st, g.path(
        g.line((r, top), (r, k)),
        g.cubic((r, k), (r, -0.72 * D), (r - 0.2 * width, yb), (r - 0.5 * width, yb)),
        g.cubic((r - 0.5 * width, yb), (Lx + 0.2 * width, yb), (Lx, -0.72 * D), (Lx, k)),
        g.line((Lx, k), (Lx, left_top)))), "كأس سفلي")


def tail(st: Style, r: float, top: float) -> g.Shape:
    """Stem that drops below the baseline and sweeps left as a hairline (ر و).
    With a logo library: the logo's own shaft and tail."""
    lib = _lib()
    if lib is not None and lib.has("tail"):
        parts = Compound([lib.tail(r)])
        sh = lib.shaft(r, top)
        if sh is not None:
            parts.insert(0, sh)
        return parts
    D, h = st.descender * st.tail_drop, st.hair
    if st.angular >= 0.5:      # down, a corner, and a flat foot to the left (as far as the logo's)
        yb = -D + h / 2 if D > h else h / 2
        foot = r - st.tail_reach * st.stem
        return _named(g.angular_stroke(st, [(r, top), (r, yb), (foot, yb)], st.corner_r), "ذيل")
    U = st.stem * st.tail_reach / 3.05           # horizontal scale of the sweep, from the logo
    yb = -D + h / 2
    return _named(g.stroke(st, g.path(
        g.line((r, top), (r, 0)),
        g.cubic((r, 0), (r, -0.6 * D), (r - 0.52 * U, yb), (r - 1.48 * U, yb)),
        g.cubic((r - 1.48 * U, yb), (r - 2.17 * U, yb), (r - 2.6 * U, -0.9 * D), (r - 3.05 * U, -0.8 * D)))), "ذيل")


def sweep_right(st: Style, x: float, top: float, reach: float) -> g.Shape:
    """Down on the left, then a bowl that swings back to the right (ح ع isolated/final)."""
    D, h, X = st.descender, st.hair, st.x_height
    yb = -D + h / 2
    if st.angular >= 0.5:
        return _named(g.angular_stroke(st, [(x, top), (x, yb), (x + reach, yb)], st.corner_r), "كأس أيمن")
    return _named(g.stroke(st, g.path(
        g.line((x, top), (x, 0.05 * X)),
        g.cubic((x, 0.05 * X), (x, -0.75 * D), (x + 0.3 * reach, yb), (x + 0.6 * reach, yb)),
        g.cubic((x + 0.6 * reach, yb), (x + 0.85 * reach, yb), (x + 0.95 * reach, -0.85 * D), (x + reach, -0.6 * D)))), "كأس أيمن")


def head_box(st: Style, counter: Optional[float] = None) -> Tuple[float, float]:
    """Left edge and width of a full head (و ف ق ه): two stems around one counter."""
    c = st.white if counter is None else counter
    w = 2 * st.stem + c
    return -w, w


# ---------------------------------------------------------------- letter families
def alef(st: Style, form: str, mark: Optional[str]) -> Glyph:
    u, X, A = st.stem / 2, st.x_height, st.ascender
    shapes = [g.stem(st, -u, X + A, entry=_e(form))]
    if mark == "ha":
        shapes += marks(st, "ha", -u, above_ref=X + A)
    elif mark == "hb":
        shapes += marks(st, "hb", -u)
    elif mark == "md":
        shapes += marks(st, "md", -u, above_ref=X + A)
    return Glyph(shapes, left=-st.stem, stems=[(-u, X + A, 0)])


def beh(st: Style, form: str, mark: Optional[str]) -> Glyph:
    u, P, X = st.stem / 2, st.pitch, st.x_height
    if form in ("I", "M"):
        shapes, xs = teeth(st, 1, _e(form), _x(form))
        shapes += marks(st, mark, xs[0])
        return Glyph(shapes, left=-st.stem, exit_x=-st.stem, exit_edge=-st.stem, stems=[(xs[0], X, 0)])
    post = 0.5 * X
    xr, xl = -u, -u - 2.2 * P
    shapes = [g.stem(st, xr, post, entry=_e(form), exit=True, round_br=not _e(form)),
              g.connector(st, xl + u, xr - u),
              g.stem(st, xl, post, entry=True, round_bl=True)]
    shapes += marks(st, mark, (xr + xl) / 2, above_ref=post)
    return Glyph(shapes, left=xl - u, stems=[(xr, post, 0), (xl, post, 0)])


def noon(st: Style, form: str, mark: Optional[str]) -> Glyph:
    if form in ("I", "M"):
        return beh(st, form, mark)
    U, u, X = st.stem, st.stem / 2, st.x_height
    r, w = -u, 3.4 * U
    shapes = [g.stem(st, r, 0.62 * X, entry=_e(form)), bowl_down(st, r, 0.4 * X, w, 0.45 * X)]
    shapes += marks(st, mark, r - w / 2, at=0.3 * X + 0.44 * st.dot + st.dot / 2)
    return Glyph(shapes, left=r - w - u, stems=[(r, 0.62 * X, 0)])


def yeh(st: Style, form: str, mark: Optional[str], final_mark: Optional[str]) -> Glyph:
    if form in ("I", "M"):
        return beh(st, form, mark)
    U, u, X, D = st.stem, st.stem / 2, st.x_height, st.descender
    r, w = -u, 4.0 * U
    shapes = [g.stem(st, r, 0.45 * X, entry=_e(form)), bowl_down(st, r, 0.3 * X, w, 0.3 * X)]
    if final_mark == "ha":
        shapes += marks(st, "ha", r - w / 2, above_ref=0.2 * X)
    else:
        d = st.dot
        shapes += marks(st, final_mark, r - w / 2, at=-D + st.hair + 0.4 * d + d / 2)
    return Glyph(shapes, left=r - w - u, stems=[(r, 0.45 * X, 0)])


def _pen(st: Style, curved, corners) -> g.Shape:
    """One continuous pen stroke: the curved path for a curved logo, straight strokes meeting at
    corners for an angular one (see skeleton.py)."""
    if st.angular >= 0.5:
        return g.angular_stroke(st, corners, st.corner_r)
    return g.stroke(st, curved)


def _jeem_square(st: Style, form: str, mark: Optional[str]) -> Glyph:
    """ح ج خ چ in a logo that draws with corners (square kufi): a hairline bar at the x-height and, from its right end, one stroke that turns back
    and runs down to the left: to the baseline join (حـ ـحـ) or into a deep bowl that swings back
    to the right under the letter (ح ـح). Bar and spine are one path, so they cannot part."""
    U, u, P, X, h, D = st.stem, st.stem / 2, st.pitch, st.x_height, st.hair, st.descender
    d0 = -u if not _e(form) else -0.55 * P - u      # the apex: right end of the bar
    xa = d0 - 1.25 * P                              # left end of the bar
    yt = X - h / 2                                  # centre line of the bar
    shapes: List[g.Shape] = []
    if _e(form):      # the join from the right climbs to the apex
        shapes.append(_named(_pen(st, g.cubic((0, h / 2), (d0 * 0.4, h / 2), (d0, 0.35 * X), (d0, yt)),
                                  [(0.0, h / 2), (d0 - u, yt)]), "صعود الوصل"))
    turn = g.path(g.line((xa, yt), (d0 - 0.25 * U, yt)),
                  g.cubic((d0 - 0.25 * U, yt), (d0 + 0.4 * U, yt), (d0 + 0.15 * U, 0.6 * X), (d0 - 0.45 * P, 0.4 * X)))
    if form in ("I", "M"):
        foot = xa - 0.05 * P
        ex = foot - 0.35 * P
        curved = g.path(turn, g.cubic((d0 - 0.45 * P, 0.4 * X), (d0 - 0.85 * P, 0.18 * X), (foot + 0.15 * P, h / 2), (ex, h / 2)))
        # square kufi: from the apex the bar runs left, turns down the left side, and leaves on the baseline
        shapes.append(_named(_pen(st, curved, [(d0, yt), (xa + u, yt), (xa + u, h / 2), (ex, h / 2)]), "رأس الحاء"))
        if mark and mark[0] == "b":
            shapes += marks(st, mark, (xa + d0) / 2 + 0.15 * P)
        else:
            shapes += marks(st, mark, (xa + d0) / 2)
        return Glyph(shapes, left=min(xa, ex), exit_x=ex + 1, exit_edge=ex)
    xl, reach = xa + 0.05 * P, 2.1 * P
    yb = -D + h / 2
    curved = g.path(turn,
                    g.cubic((d0 - 0.45 * P, 0.4 * X), (xl - 0.05 * P, 0.25 * X), (xl - 0.1 * P, -0.5 * D), (xl + 0.45 * reach, yb)),
                    g.cubic((xl + 0.45 * reach, yb), (xl + 0.8 * reach, yb), (xl + 0.95 * reach, -0.85 * D), (xl + reach, -0.6 * D)))
    corners = [(d0, yt), (xl + u, yt), (xl + u, yb), (xl + reach, yb)]
    shapes.append(_named(_pen(st, curved, corners), "الحاء"))
    if mark and mark[0] == "b":
        d = st.dot
        base_y = yb + h / 2 + 0.45 * d + d / 2
        spots = [(xl + f * reach, base_y + k * 0.25 * d) for k in (0, 1, 2) for f in (0.62, 0.55, 0.48, 0.7, 0.4)]
        # a shallow bowl (short descender) has no room inside: then under it
        spots += [(xl + f * reach, yb - h / 2 - 0.45 * d - d / 2) for f in (0.55, 0.45)]
        shapes += clear_marks(st, shapes, mark, spots)
    else:
        shapes += marks(st, mark, (xa + d0) / 2)
    return Glyph(shapes, left=xa)


def jeem(st: Style, form: str, mark: Optional[str]) -> Glyph:
    """ح ج خ چ. The skeleton of the reference typefaces: an upper arm, a nose at the far right above the
    middle of the letter, and a lower arm that sweeps back to the left: to the baseline join (حـ ـحـ)
    or on down into a deep bowl that swings back to the right under the letter (ح ـح).
    The upper arm is a hairline bar, the nose is where the pen is upright (a full stem), the lower arm
    thins as it leans. A join from the right arrives on the baseline, under the nose."""
    U, u, P, X, h, D = st.stem, st.stem / 2, st.pitch, st.x_height, st.hair, st.descender
    d0 = 0.0                                        # the nose: right end of the letter
    xa = d0 - 1.78 * P                              # left end of the bar
    xj = d0 - 1.38 * P                              # where the lower arm reaches the baseline
    yt = X - h / 2
    if st.angular >= 0.5:
        return _jeem_square(st, form, mark)
    top = [g.nose_top(st, d0, xa)]
    if form in ("I", "M"):
        ex = xj - 0.45 * P
        run = d0 - xj
        arm = [g.nose_arm(st, d0, xj, ex)]
        if _e(form):
            # the join from the right runs on the baseline under the nose and into the arm; the wedge
            # between the two is filled
            # the stub runs right until the arm has risen well clear of it (2.6 hairlines)
            rise = [q[0] for q in g.cubic((d0, 0.636 * X), (d0, 0.303 * X), (d0 - 0.387 * run, 0), (xj, 0)) if q[1] <= 2.6 * h]
            entry = min(max(rise) if rise else xj + 0.55 * run, d0 - 0.05 * run)
            shapes = [_named(g.fuse([top, arm, g.rect(xj, entry, 0, h)], 0.6 * h), "أنف الحاء")]
        else:
            entry = 0.0
            shapes = [_named(g.fuse([top, arm], 0.0), "أنف الحاء")]
        if mark and mark[0] == "b":
            shapes += marks(st, mark, (xa + d0) / 2 - 0.1 * P)
        else:
            shapes += marks(st, mark, (xa + d0) / 2)
        # the next letter keeps its white from the bar's left end; the baseline join passes under it
        return Glyph(shapes, left=xa, entry_x=entry, entry_edge=0, exit_x=ex + 1, exit_edge=xa)
    # ح ـح: from the nose the stroke runs down to the left, through the baseline, into the bowl
    xl, reach = xa - 0.05 * P, 2.2 * P
    yb = -D + h / 2
    run = d0 - xl
    beta = 32.0
    m = (d0 - 0.5 * run, 0.2 * X)
    t = (-math.cos(math.radians(beta)), -math.sin(math.radians(beta)))
    bowl = g.path(g.line((m[0] - t[0] * 0.05 * run, m[1] - t[1] * 0.05 * run), m),
                  g.cubic(m, (m[0] + t[0] * 0.33 * run, m[1] + t[1] * 0.33 * run), (xl + 0.03 * P, -0.02 * D), (xl, -0.3 * D)),
                  g.cubic((xl, -0.3 * D), (xl - 0.03 * P, -0.8 * D), (xl + 0.2 * reach, yb), (xl + 0.5 * reach, yb)),
                  g.cubic((xl + 0.5 * reach, yb), (xl + 0.8 * reach, yb), (xl + 0.95 * reach, -0.85 * D), (xl + reach, -0.6 * D)))
    parts = [top, [g.nose_turn(st, d0, m, beta)], g.solid_stroke(st, bowl)]
    if _e(form):
        parts.append(g.rect(d0 - 0.78 * run, 0.0, 0, h))      # the join from the right meets the arm on the baseline
    shapes = [_named(g.fuse(parts, 0.6 * h), "الحاء")]
    if mark and mark[0] == "b":
        d = st.dot
        base_y = yb + h / 2 + 0.45 * d + d / 2
        spots = [(xl + f * reach, base_y + k * 0.25 * d) for k in (0, 1, 2) for f in (0.62, 0.55, 0.48, 0.7, 0.4)]
        spots += [(xl + f * reach, yb - h / 2 - 0.45 * d - d / 2) for f in (0.55, 0.45)]   # a shallow bowl: under it
        shapes += clear_marks(st, shapes, mark, spots)
    else:
        shapes += marks(st, mark, (xa + d0) / 2)
    return Glyph(shapes, left=xa, entry_x=0.0, entry_edge=0)


def dal(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, h = st.stem, st.stem / 2, st.x_height, st.hair
    shapes = [g.stem(st, -u, X, entry=_e(form), exit=True),
              _named(g.rect(-U - 0.57 * U, -U, X - h, X), "راية الدال"),
              _named(g.connector(st, -U - 1.75 * U, -U), "قاعدة الدال")]
    shapes += marks(st, mark, -u)
    return Glyph(shapes, left=-U - 1.75 * U, stems=[(-u, X, 0)])


def reh(st: Style, form: str, mark: Optional[str]) -> Glyph:
    u, X = st.stem / 2, st.x_height
    if "top_entry" in JOIN.get() and st.angular < 0.5:
        # as the logo joins its own ر: a hairline bar from the top of the previous tooth, a rounded
        # corner, a full stem, and the tail under the line
        h, U = st.hair, st.stem
        r = 0.48 * U
        bar = st.white + 0.5 * U
        body = [(bar, X), (bar, X - h), (0.0, X - h), (0.0, 0.0), (-U, 0.0), (-U, X - r)] \
            + [(q[0], q[1]) for q in g.cubic((-U, X - r), (-U, X - 0.45 * r), (-U + 0.45 * r, X), (-U + r, X), 10)]
        shapes = [_named([body], "راء موصولة من أعلى"), tail(st, -u, 0.3 * X)]
        shapes += marks(st, mark, -u, above_ref=X)
        return Glyph(shapes, left=-U, stems=[(-u, X, 0)], tail=-u - 3.05 * st.stem - st.hair)
    if "top_entry" in JOIN.get():
        # a bar from the top of the previous tooth, down the stem, and a flat foot on the baseline
        h, U = st.hair, st.stem
        foot = -u - st.tail_reach * U + u
        yb = h / 2
        path = [(st.white + 0.5 * U, X - h / 2), (-u, X - h / 2), (-u, yb), (foot, yb)]
        shapes = [_named(g.angular_stroke(st, path, st.corner_r), "راء موصولة من أعلى")]
        shapes += marks(st, mark, -u, above_ref=X)
        return Glyph(shapes, left=-U, stems=[(-u, X, 0)], tail=foot - h)
    lib = _lib()
    top = lib.parts["tail"].info.get("top", 0.62 * X) if lib is not None and lib.has("tail") else 0.62 * X
    shapes = [tail(st, -u, top)]
    if _e(form):
        half = lib.half("r", True, -u, lib.white) if lib is not None else None
        shapes.append(half if half is not None else g.stem(st, -u, top, entry=True))
    shapes += marks(st, mark, -u, above_ref=top)
    return Glyph(shapes, left=-st.stem, stems=[(-u, top, 0)], tail=-u - 3.05 * st.stem - st.hair)


def seen(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, P, X = st.stem, st.stem / 2, st.pitch, st.x_height
    if form in ("I", "M"):
        shapes, xs = teeth(st, 3, _e(form), _x(form), drop_inner=True)
        shapes += marks(st, mark, xs[1])
        return Glyph(shapes, left=xs[-1] - u, exit_x=xs[-1] - u, exit_edge=xs[-1] - u, stems=[(x, X, 0) for x in xs])
    shapes, xs = teeth(st, 2, _e(form), True, drop_inner=True)
    r = xs[-1] - P
    w = 3.4 * U
    shapes += [g.connector(st, r + u, xs[-1] - u), g.stem(st, r, X, entry=True), bowl_down(st, r, 0.4 * X, w, 0.45 * X)]
    shapes += marks(st, mark, xs[1])
    return Glyph(shapes, left=r - w - u, stems=[(x, X, 0) for x in xs] + [(r, X, 0)])


def _loop_square(st: Style) -> Tuple[float, float]:
    """The long loop of ص ط. Like every head it fills the x-height (all letters lie between the
    baseline and the x-height line); it is longer than a head: its counter is 1.6 whites wide."""
    w = 2 * st.stem + 1.6 * st.white
    return -w, st.x_height


def _sad_square(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X = st.stem, st.stem / 2, st.x_height
    xl, ht = _loop_square(st)
    shapes = [_ring(st, xl, 0, 0, ht)]
    xt = xl + u
    if form in ("I", "M"):
        # the tooth of ص stands one pitch after the loop, joined by the same u-bowl as two teeth:
        # the loop's left side gives the right half of the bowl, the tooth the left half
        xs = xt - st.pitch
        shapes.append(g.stem(st, xt, 0.5 * X, exit=True))
        shapes.append(g.stem(st, xs, X, entry=True, exit=_x(form), round_bl=not _x(form)))
        shapes += marks(st, mark, xl / 2, above_ref=ht)
        return Glyph(shapes, left=xs - u, entry_x=-U, entry_edge=0, exit_x=xs - u, exit_edge=xs - u,
                     stems=[(xs, X, 0)])
    w = 3.4 * U
    shapes.append(bowl_down(st, xt, 0.4 * ht, w, 0.45 * X))
    shapes += marks(st, mark, xl / 2, above_ref=ht)
    return Glyph(shapes, left=xt - w - u, entry_x=-U, entry_edge=0)


def _tah_square(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, A = st.stem, st.stem / 2, st.x_height, st.ascender
    xl, ht = _loop_square(st)
    xt = xl + u
    # the stem stands on the loop; a join to the next letter leaves from the loop's own bottom
    shapes = [_ring(st, xl, 0, 0, ht), g.stem(st, xt, X + A)]
    shapes += marks(st, mark, xl * 0.38, above_ref=ht)
    return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0, exit_x=xl if _x(form) else None,
                 exit_edge=xl if _x(form) else None, stems=[(xt, X + A, 0)])


def _loop(st: Style) -> Tuple[List[g.Shape], float, float]:
    """The loop of ص ط with its right end at x = 0: the teardrop, the centre of the stem that stands at
    its left point, and its length."""
    L = g.loop_length(st)
    return [_named(g.teardrop(st, -L, 0.0), "عروة")], -L - st.stem / 2, L


def sad(st: Style, form: str, mark: Optional[str]) -> Glyph:
    """ص ض: the loop, and a tooth standing at its left point (the reference typefaces put it there,
    not a pitch away). The tooth carries the join, or goes down into the cup (ص ـص)."""
    if st.angular >= 0.5:
        return _sad_square(st, form, mark)
    U, u, X = st.stem, st.stem / 2, st.x_height
    shapes, xs, L = _loop(st)
    common = dict(entry_x=-1.3 * U, entry_edge=0)
    if form in ("I", "M"):
        shapes.append(g.stem(st, xs, X, exit=_x(form), round_bl=not _x(form)))
        shapes += marks(st, mark, -0.43 * L, above_ref=X)
        return Glyph(shapes, left=xs - u, exit_x=xs - u, exit_edge=xs - u, stems=[(xs, X, 0)], **common)
    w = 3.4 * U
    shapes += [g.stem(st, xs, X), bowl_down(st, xs, 0.4 * X, w, 0.45 * X)]
    shapes += marks(st, mark, -0.43 * L, above_ref=X)
    return Glyph(shapes, left=xs - w - u, stems=[(xs, X, 0)], **common)


def tah(st: Style, form: str, mark: Optional[str]) -> Glyph:
    """ط ظ: the same loop, and a stem to the ascender standing at its left point."""
    if st.angular >= 0.5:
        return _tah_square(st, form, mark)
    U, u, X, A = st.stem, st.stem / 2, st.x_height, st.ascender
    shapes, xs, L = _loop(st)
    shapes.append(g.stem(st, xs, X + A, exit=_x(form), round_bl=not _x(form)))
    shapes += marks(st, mark, -0.4 * L, above_ref=X)
    return Glyph(shapes, left=xs - u, entry_x=-1.3 * U, entry_edge=0, exit_x=(xs - u) if _x(form) else None,
                 exit_edge=(xs - u) if _x(form) else None, stems=[(xs, X + A, 0)])


def _ain_eye(st: Style, xl: float, top: float) -> List[g.Shape]:
    """The eye of ـعـ ـع: a head on the baseline whose top dips in a V. The counter stays closed
    (an open top would read as two teeth of ـسـ). The V is the band between the cut and the same
    cut moved down by the pen, so it shares its edges with the head and cannot leave a seam."""
    from shapely.affinity import translate
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    U, u, h = st.stem, st.stem / 2, st.hair
    inner = top - 2 * h                             # height of the counter
    depth = h + 0.35 * inner                        # the V reaches a third of the way into it
    ring = g.ring(st, xl, 0, 0, top)
    shp, outer = None, None
    for poly in ring:
        P_ = Polygon(poly).buffer(0)
        outer = P_ if outer is None else outer.union(P_)
        shp = P_ if shp is None else shp.symmetric_difference(P_)
    xm = xl / 2
    notch = Polygon([(xl + u, top + 2), (-u, top + 2), (-u, top - h / 2), (xm, top - depth), (xl + u, top - h / 2)])
    t = 1.2 * h                                     # a diagonal is a little heavier than the hairline
    band = notch.difference(translate(notch, 0, -t)).intersection(outer)
    shp = unary_union([shp.difference(notch), band])
    out: g.Shape = []
    for poly in getattr(shp, "geoms", [shp]):
        if poly.geom_type == "Polygon" and not poly.is_empty:
            out.append(list(poly.exterior.coords)[:-1])
            out += [list(i.coords)[:-1] for i in poly.interiors]
    return [_named(out, "عين")]


def ain(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, P, X, h, D = st.stem, st.stem / 2, st.pitch, st.x_height, st.hair, st.descender
    if form in ("M", "F"):
        xl = -(2 * U + 0.7 * st.white)
        top = 0.85 * X
        shapes = _ain_eye(st, xl, top) + [g.connector(st, xl, 0)]
        shapes += marks(st, mark, xl / 2, above_ref=top)
        if form == "M":
            return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0, exit_x=xl + U, exit_edge=xl)
        shapes.append(sweep_right(st, xl + u, 0.3 * top, 1.8 * P))
        return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0)
    a, b = 0.55 * P, 0.3 * X
    cx, cy = -a * 0.95, X - b
    xs = cx - a
    if st.angular >= 0.5:      # a square c: top bar, the left side, the lower arm
        shapes = [_named(g.angular_stroke(st, [(cx + a, X - h / 2), (xs, X - h / 2), (xs, cy - b + h / 2), (cx + 0.6 * a, cy - b + h / 2)],
                                          st.corner_r), "رأس العين")]
    else:
        shapes = [g.open_arc(st, cx, cy, a, b, 35, 325)]
    shapes += marks(st, mark, cx)
    if form == "I":
        shapes.append(g.stem(st, xs, cy, exit=True))
        return Glyph(shapes, left=xs - u, exit_x=xs - u, exit_edge=xs - u, stems=[(xs, cy, 0)])
    shapes.append(sweep_right(st, xs, cy, 2.0 * P))
    return Glyph(shapes, left=xs - u)


def feh(st: Style, form: str, mark: Optional[str], qaf: bool = False) -> Glyph:
    U, u, P, X = st.stem, st.stem / 2, st.pitch, st.x_height
    xl, w = head_box(st)
    shapes = [_ring(st, xl, 0, 0, X)]
    shapes += marks(st, mark, xl / 2)
    common = dict(entry_x=-U, entry_edge=0)
    if form in ("I", "M"):
        return Glyph(shapes, left=xl, exit_x=xl + U, exit_edge=xl, **common)
    if qaf:
        # the cup's left arm keeps half a white from the head (12 against a white of 22 in the
        # hand-made بيدوناق), never against it
        bw = max(3.5 * U, -xl + 0.5 * st.white)
        shapes.append(bowl_down(st, -u, X / 2, bw, 0.6 * X))
        return Glyph(shapes, left=xl, **common)
    post = xl - 2.0 * P
    shapes += [g.connector(st, post + u, xl + U), g.stem(st, post, 0.5 * X, entry=True, round_bl=True)]
    return Glyph(shapes, left=post - u, stems=[(post, 0.5 * X, 0)], **common)


def kaf(st: Style, form: str, mark: Optional[str], variant: str) -> Glyph:
    """ك ک گ: a stem to the ascender with a hairline arm to the left at its top (the gaf has a
    second, floating bar above the arm; both stay under the ascender line). Final and isolated
    forms add the baseline and a short post on the left, like ب; Arabic ك carries a small dash."""
    U, u, P, X, A, h = st.stem, st.stem / 2, st.pitch, st.x_height, st.ascender, st.hair
    top = X + A
    arm_top = top - 2.6 * h if variant == "gaf" else top
    arm_left = -U - 1.3 * P
    shapes: List[g.Shape] = []
    if form in ("I", "M"):
        shapes.append(g.stem(st, -u, arm_top, entry=_e(form), exit=_x(form), round_br=not _e(form)))
    else:
        shapes.append(g.stem(st, -u, arm_top, entry=_e(form), exit=True, round_br=not _e(form)))
    shapes.append(_named(g.rect(arm_left, -u, arm_top - h, arm_top), "ذراع الكاف"))
    if variant == "gaf":
        shapes.append(_named(g.rect(arm_left + 0.15 * P, -U - 0.1 * P, top - h, top), "شرطة الگاف"))
    if form in ("I", "M"):
        return Glyph(shapes, left=min(arm_left, -U), exit_x=-U, exit_edge=-U, stems=[(-u, arm_top, 0)])
    post = -U - 2.2 * P
    shapes += [g.connector(st, post + u, -U), g.stem(st, post, 0.45 * X, entry=True, round_bl=True)]
    if variant == "arabic":
        cx = (post + u - U) / 2
        shapes.append(_named(g.rect(cx - 0.35 * P, cx + 0.35 * P, 0.42 * X - h / 2, 0.42 * X + h / 2), "شرطة"))
    return Glyph(shapes, left=min(arm_left, post - u), stems=[(-u, arm_top, 0), (post, 0.45 * X, 0)])


def lam(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, A = st.stem, st.stem / 2, st.x_height, st.ascender
    if form in ("I", "M"):
        shapes = [g.stem(st, -u, X + A, entry=_e(form), exit=True, round_br=not _e(form))]
        return Glyph(shapes, left=-U, exit_x=-U, exit_edge=-U, stems=[(-u, X + A, 0)])
    w = 3.4 * U
    shapes = [g.stem(st, -u, X + A, entry=_e(form)), bowl_down(st, -u, 0.4 * X, w, 0.45 * X)]
    return Glyph(shapes, left=-u - w - u, stems=[(-u, X + A, 0)])


def meem(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, D, h = st.stem, st.stem / 2, st.x_height, st.descender, st.hair
    xl = -(2 * U + 0.55 * st.white)
    top = 0.6 * X
    shapes = [_ring(st, xl, 0, 0, top)]
    common = dict(entry_x=-U, entry_edge=0)
    if form in ("I", "M"):
        return Glyph(shapes, left=xl, exit_x=xl + U, exit_edge=xl, **common)
    x = xl + u
    # the tail drops from the left side of the head; in a curved logo it hooks to the left and thins
    curved = g.path(g.line((x, top / 2), (x, -0.4 * D)),
                    g.cubic((x, -0.4 * D), (x, -0.85 * D), (x - 0.3 * U, -D + h / 2), (x - 1.0 * U, -D + h / 2)))
    shapes.append(_named(_pen(st, curved, [(x, top / 2), (x, -D)]), "ذيل الميم"))
    return Glyph(shapes, left=xl - (1.0 * U if st.angular < 0.5 else 0), stems=[(x, top / 2, -D)], **common)


def heh(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, A, h = st.stem, st.stem / 2, st.x_height, st.ascender, st.hair
    xl, w = head_box(st)
    if form == "I":
        # هـ: a lower head whose right side rises as a stem to the x-height (like a mirrored ط)
        ht = 0.68 * X
        shapes = [_ring(st, xl, 0, 0, ht), g.stem(st, -u, X, round_br=True)]
        shapes += marks(st, mark, xl / 2, above_ref=X)
        return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0, exit_x=xl + U, exit_edge=xl, stems=[(-u, X, 0)])
    if form == "M":
        # ـهـ in the reference typefaces: an eye above the baseline and a smaller one hanging under it.
        # The upper eye is the head; the lower one is the same ring, 0.62 as wide, down to the descender
        side = 0.63 * U
        wl = max(0.62 * w, 2 * side + 0.45 * st.white)
        cx = xl / 2
        shapes = [_ring(st, xl, 0, 0, X),
                  _named(g.ring(st, cx - wl / 2, cx + wl / 2, -st.descender, h, side=side), "عين سفلى")]
    else:
        shapes = [_ring(st, xl, 0, 0, X)]
    shapes += marks(st, mark, xl / 2)
    return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0,
                 exit_x=(xl + U) if _x(form) else None, exit_edge=xl if _x(form) else None)


def waw(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X = st.stem, st.stem / 2, st.x_height
    xl, w = head_box(st)
    if st.tail_drop < 0.3:
        # the tail rests on the baseline: the head rises above it, or it would read as ه with a foot
        hb = 0.4 * X
        shapes = [_ring(st, xl, 0, hb, X), tail(st, -u, hb + st.hair)]
    else:
        shapes = [_ring(st, xl, 0, 0, X), tail(st, -u, X / 2)]
    shapes += marks(st, mark, xl / 2)
    return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0, tail=-u - 3.05 * U - st.hair)


def hamza(st: Style, form: str, mark: Optional[str]) -> Glyph:
    X, U = st.x_height, st.stem
    shapes = [g.open_arc(st, -0.7 * U, 0.42 * X, 0.55 * U, 0.3 * X, 40, 320, scale=0.8)]
    return Glyph(shapes, left=-1.5 * U)


def tatweel(st: Style, form: str, mark: Optional[str]) -> Glyph:
    P = st.pitch
    return Glyph([g.connector(st, -P, 0)], left=-P, entry_x=0, exit_x=-P, exit_edge=-P)


# ---------------------------------------------------------------- the table
# join: "D" joins both sides, "R" joins only the letter before it, "U" never joins.
Builder = Callable[[Style, str], Glyph]
LETTERS: Dict[str, Tuple[str, Builder]] = {
    "ا": ("R", lambda s, f: alef(s, f, None)),
    "أ": ("R", lambda s, f: alef(s, f, "ha")),
    "إ": ("R", lambda s, f: alef(s, f, "hb")),
    "آ": ("R", lambda s, f: alef(s, f, "md")),
    "ٱ": ("R", lambda s, f: alef(s, f, None)),
    "ب": ("D", lambda s, f: beh(s, f, "b1")),
    "ت": ("D", lambda s, f: beh(s, f, "a2")),
    "ث": ("D", lambda s, f: beh(s, f, "a3")),
    "پ": ("D", lambda s, f: beh(s, f, "b3")),
    "ن": ("D", lambda s, f: noon(s, f, "a1")),
    "ي": ("D", lambda s, f: yeh(s, f, "b2", "b2")),
    "ی": ("D", lambda s, f: yeh(s, f, "b2", None)),
    "ى": ("D", lambda s, f: yeh(s, f, None, None)),
    "ئ": ("D", lambda s, f: yeh(s, f, "ha", "ha")),
    "ج": ("D", lambda s, f: jeem(s, f, "b1")),
    "ح": ("D", lambda s, f: jeem(s, f, None)),
    "خ": ("D", lambda s, f: jeem(s, f, "a1")),
    "چ": ("D", lambda s, f: jeem(s, f, "b3")),
    "د": ("R", lambda s, f: dal(s, f, None)),
    "ذ": ("R", lambda s, f: dal(s, f, "a1")),
    "ر": ("R", lambda s, f: reh(s, f, None)),
    "ز": ("R", lambda s, f: reh(s, f, "a1")),
    "ژ": ("R", lambda s, f: reh(s, f, "a3")),
    "س": ("D", lambda s, f: seen(s, f, None)),
    "ش": ("D", lambda s, f: seen(s, f, "a3")),
    "ص": ("D", lambda s, f: sad(s, f, None)),
    "ض": ("D", lambda s, f: sad(s, f, "a1")),
    "ط": ("D", lambda s, f: tah(s, f, None)),
    "ظ": ("D", lambda s, f: tah(s, f, "a1")),
    "ع": ("D", lambda s, f: ain(s, f, None)),
    "غ": ("D", lambda s, f: ain(s, f, "a1")),
    "ف": ("D", lambda s, f: feh(s, f, "a1")),
    "ڤ": ("D", lambda s, f: feh(s, f, "a3")),
    "ق": ("D", lambda s, f: feh(s, f, "a2", qaf=True)),
    "ك": ("D", lambda s, f: kaf(s, f, None, "arabic")),
    "ک": ("D", lambda s, f: kaf(s, f, None, "keheh")),
    "گ": ("D", lambda s, f: kaf(s, f, None, "gaf")),
    "ل": ("D", lambda s, f: lam(s, f, None)),
    "م": ("D", lambda s, f: meem(s, f, None)),
    "ه": ("D", lambda s, f: heh(s, f, None)),
    "ة": ("R", lambda s, f: heh(s, f, "a2")),
    "و": ("R", lambda s, f: waw(s, f, None)),
    "ؤ": ("R", lambda s, f: waw(s, f, "ha")),
    "ء": ("U", lambda s, f: hamza(s, f, None)),
    "ـ": ("D", lambda s, f: tatweel(s, f, None)),
}
