"""Letter forms built from six parts: stem, baseline join, head (ring), tail/bowl, dot, dash.

Each builder draws one letter in local coordinates: the right edge of the letter body is x = 0,
the letter grows to the left (negative x), baseline is y = 0.
`form` is one of: "i" isolated, "I" initial, "M" medial, "F" final.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import geometry as g
from .style import Style


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
    return form in ("I", "M")


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
    if kind == "1":
        out.append(g.circle(cx, cy, d / 2, st.tan))
    elif kind == "2":
        if st.dash_marks:
            out.append(g.rect(cx - 0.75 * d, cx + 0.75 * d, cy - h / 2, cy + h / 2))
        else:
            out += [g.circle(cx - 0.6 * d, cy, d / 2, st.tan), g.circle(cx + 0.6 * d, cy, d / 2, st.tan)]
    elif kind == "3":
        step = 0.95 * d if above else -0.95 * d
        if st.dash_marks:
            out.append(g.rect(cx - 0.75 * d, cx + 0.75 * d, cy - h / 2, cy + h / 2))
            out.append(g.circle(cx, cy + step * 0.75 + (d / 2 - h / 2) * (1 if above else -1) * 0.2, d / 2, st.tan))
        else:
            out += [g.circle(cx - 0.6 * d, cy, d / 2, st.tan), g.circle(cx + 0.6 * d, cy, d / 2, st.tan), g.circle(cx, cy + step, d / 2, st.tan)]
    elif spec in ("ha", "hb"):
        out.append(g.open_arc(st, cx, cy, 0.42 * d, 0.5 * d, 40, 320, scale=0.55))
    elif spec == "md":
        out.append(g.rect(cx - 0.9 * d, cx + 0.9 * d, cy - h / 2, cy + h / 2))
    return out


# ---------------------------------------------------------------- shared parts
def teeth(st: Style, n: int, entry: bool, exit: bool, top: Optional[float] = None,
          x0: float = 0.0) -> Tuple[List[g.Shape], List[float]]:
    """n teeth joined at the baseline, right to left, starting at right edge x0."""
    u, P = st.stem / 2, st.pitch
    top = st.x_height if top is None else top
    xs = [x0 - u - i * P for i in range(n)]
    shapes: List[g.Shape] = []
    for i, xc in enumerate(xs):
        ent = entry if i == 0 else True
        ex = exit if i == n - 1 else True
        shapes.append(g.stem(st, xc, top, entry=ent, exit=ex,
                             round_br=(i == 0 and not entry), round_bl=(i == n - 1 and not exit)))
        if i > 0:
            shapes.append(g.connector(st, xc + u, xs[i - 1] - u))
    return shapes, xs


def bowl_down(st: Style, r: float, top: float, width: float, left_top: float) -> g.Shape:
    """Arm at x = r goes down below the baseline, across, and up on the left (ن ل ق س ص ي)."""
    D, h, X = st.descender, st.hair, st.x_height
    Lx, yb, k = r - width, -D + h / 2, 0.08 * X
    return g.stroke(st, g.path(
        g.line((r, top), (r, k)),
        g.cubic((r, k), (r, -0.72 * D), (r - 0.2 * width, yb), (r - 0.5 * width, yb)),
        g.cubic((r - 0.5 * width, yb), (Lx + 0.2 * width, yb), (Lx, -0.72 * D), (Lx, k)),
        g.line((Lx, k), (Lx, left_top))))


def tail(st: Style, r: float, top: float) -> g.Shape:
    """Stem that drops below the baseline and sweeps left as a hairline (ر و)."""
    U, D, h = st.stem, st.descender, st.hair
    yb = -D + h / 2
    return g.stroke(st, g.path(
        g.line((r, top), (r, 0)),
        g.cubic((r, 0), (r, -0.6 * D), (r - 0.52 * U, yb), (r - 1.48 * U, yb)),
        g.cubic((r - 1.48 * U, yb), (r - 2.17 * U, yb), (r - 2.6 * U, -0.9 * D), (r - 3.05 * U, -0.8 * D))))


def sweep_right(st: Style, x: float, top: float, reach: float) -> g.Shape:
    """Down on the left, then a bowl that swings back to the right (ح ع isolated/final)."""
    D, h, X = st.descender, st.hair, st.x_height
    yb = -D + h / 2
    return g.stroke(st, g.path(
        g.line((x, top), (x, 0.05 * X)),
        g.cubic((x, 0.05 * X), (x, -0.75 * D), (x + 0.3 * reach, yb), (x + 0.6 * reach, yb)),
        g.cubic((x + 0.6 * reach, yb), (x + 0.85 * reach, yb), (x + 0.95 * reach, -0.85 * D), (x + reach, -0.6 * D))))


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
        shapes += marks(st, final_mark, r - w / 2, at=-0.42 * D)
    return Glyph(shapes, left=r - w - u, stems=[(r, 0.45 * X, 0)])


def jeem(st: Style, form: str, mark: Optional[str]) -> Glyph:
    u, P, X, h, D = st.stem / 2, st.pitch, st.x_height, st.hair, st.descender
    d0 = -u if not _e(form) else -0.6 * P - u      # apex of the head
    shapes: List[g.Shape] = [g.rect(d0 - 1.3 * P, d0, X - h, X)]
    if _e(form):   # the join from the right climbs to the apex
        shapes.append(g.stroke(st, g.cubic((0, h / 2), (d0 * 0.45, h / 2), (d0, 0.3 * X), (d0, X - h / 2))))
    if form in ("I", "M"):
        foot = d0 - 1.25 * P
        shapes.append(g.stroke(st, g.cubic((d0, X - h / 2), (d0, 0.5 * X), (d0 - 0.8 * P, 0.2 * X), (foot, h / 2))))
        shapes += marks(st, mark, d0 - 0.7 * P)
        return Glyph(shapes, left=d0 - 1.3 * P, exit_x=foot + 1, exit_edge=foot)
    x = d0 - 1.15 * P
    shapes.append(g.stroke(st, g.cubic((d0, X - h / 2), (d0, 0.5 * X), (x, 0.45 * X), (x, 0.05 * X))))
    shapes.append(sweep_right(st, x, 0.05 * X, 2.0 * P))
    if mark and mark[0] == "b":
        shapes += marks(st, mark, x + 0.75 * P, at=-0.45 * D)
    else:
        shapes += marks(st, mark, d0 - 0.65 * P)
    return Glyph(shapes, left=d0 - 1.3 * P)


def dal(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, h = st.stem, st.stem / 2, st.x_height, st.hair
    shapes = [g.stem(st, -u, X, entry=_e(form), exit=True),
              g.rect(-U - 0.57 * U, -U, X - h, X),
              g.connector(st, -U - 1.75 * U, -U)]
    shapes += marks(st, mark, -u)
    return Glyph(shapes, left=-U - 1.75 * U, stems=[(-u, X, 0)])


def reh(st: Style, form: str, mark: Optional[str]) -> Glyph:
    u, X = st.stem / 2, st.x_height
    top = 0.62 * X
    shapes = [tail(st, -u, top)]
    if _e(form):
        shapes.append(g.stem(st, -u, top, entry=True))
    shapes += marks(st, mark, -u, above_ref=top)
    return Glyph(shapes, left=-st.stem, stems=[(-u, top, 0)], tail=-u - 3.05 * st.stem - st.hair)


def seen(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, P, X = st.stem, st.stem / 2, st.pitch, st.x_height
    if form in ("I", "M"):
        shapes, xs = teeth(st, 3, _e(form), _x(form))
        shapes += marks(st, mark, xs[1])
        return Glyph(shapes, left=xs[-1] - u, exit_x=xs[-1] - u, exit_edge=xs[-1] - u, stems=[(x, X, 0) for x in xs])
    shapes, xs = teeth(st, 2, _e(form), True)
    r = xs[-1] - P
    w = 3.4 * U
    shapes += [g.connector(st, r + u, xs[-1] - u), g.stem(st, r, X, entry=True), bowl_down(st, r, 0.4 * X, w, 0.45 * X)]
    shapes += marks(st, mark, xs[1])
    return Glyph(shapes, left=r - w - u, stems=[(x, X, 0) for x in xs] + [(r, X, 0)])


def _loop(st: Style) -> Tuple[float, float]:
    """The long flat loop of ص ط."""
    w = 2 * st.stem + 1.6 * st.white
    return -w, 0.72 * st.x_height


def sad(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X = st.stem, st.stem / 2, st.x_height
    xl, ht = _loop(st)
    shapes = [g.ring(st, xl, 0, 0, ht)]
    xt = xl + u
    if form in ("I", "M"):
        shapes.append(g.stem(st, xt, X, exit=_x(form), round_bl=not _x(form)))
        shapes += marks(st, mark, xl / 2, above_ref=ht)
        return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0, exit_x=xl, exit_edge=xl, stems=[(xt, X, 0)])
    w = 3.4 * U
    shapes.append(bowl_down(st, xt, 0.4 * ht, w, 0.45 * X))
    shapes += marks(st, mark, xl / 2, above_ref=ht)
    return Glyph(shapes, left=xt - w - u, entry_x=-U, entry_edge=0)


def tah(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, A = st.stem, st.stem / 2, st.x_height, st.ascender
    xl, ht = _loop(st)
    xt = xl + u
    shapes = [g.ring(st, xl, 0, 0, ht), g.stem(st, xt, X + A, exit=_x(form), round_bl=not _x(form))]
    shapes += marks(st, mark, xl * 0.38, above_ref=ht)
    return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0, exit_x=xl if _x(form) else None,
                 exit_edge=xl if _x(form) else None, stems=[(xt, X + A, 0)])


def ain(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, P, X = st.stem, st.stem / 2, st.pitch, st.x_height
    if form in ("M", "F"):
        xl = -(2 * U + 0.6 * st.white)
        top = 0.8 * X
        shapes = [g.ring(st, xl, 0, 0, top)]
        shapes += marks(st, mark, xl / 2, above_ref=top)
        if form == "M":
            return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0, exit_x=xl + U, exit_edge=xl)
        shapes.append(sweep_right(st, xl + u, 0.4 * top, 1.8 * P))
        return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0)
    a, b = 0.55 * P, 0.3 * X
    cx, cy = -a * 0.95, X - b
    shapes = [g.open_arc(st, cx, cy, a, b, 35, 325)]
    xs = cx - a
    shapes += marks(st, mark, cx)
    if form == "I":
        shapes.append(g.stem(st, xs, cy, exit=True))
        return Glyph(shapes, left=xs - u, exit_x=xs - u, exit_edge=xs - u, stems=[(xs, cy, 0)])
    shapes.append(sweep_right(st, xs, cy, 2.0 * P))
    return Glyph(shapes, left=xs - u)


def feh(st: Style, form: str, mark: Optional[str], qaf: bool = False) -> Glyph:
    U, u, P, X = st.stem, st.stem / 2, st.pitch, st.x_height
    xl, w = head_box(st)
    shapes = [g.ring(st, xl, 0, 0, X)]
    shapes += marks(st, mark, xl / 2)
    common = dict(entry_x=-U, entry_edge=0)
    if form in ("I", "M"):
        return Glyph(shapes, left=xl, exit_x=xl + U, exit_edge=xl, **common)
    if qaf:
        bw = 3.5 * U
        shapes.append(bowl_down(st, -u, X / 2, bw, 0.6 * X))
        return Glyph(shapes, left=xl, **common)
    post = xl - 2.0 * P
    shapes += [g.connector(st, post + u, xl + U), g.stem(st, post, 0.5 * X, entry=True, round_bl=True)]
    return Glyph(shapes, left=post - u, stems=[(post, 0.5 * X, 0)], **common)


def kaf(st: Style, form: str, mark: Optional[str], variant: str) -> Glyph:
    U, u, P, X, A, h = st.stem / 2 * 2, st.stem / 2, st.pitch, st.x_height, st.ascender, st.hair
    top = X + A
    if form in ("I", "M"):
        # stem to the x-height, then a thin stroke rising to the upper left (كـ)
        x_end = -U - 1.25 * P
        bar = [[(-U, X - h), (-U, X), (x_end, top), (x_end, top - h * 1.4)]]
        shapes = [g.stem(st, -u, X, entry=_e(form), exit=True, round_br=not _e(form)), bar]
        if variant == "gaf":
            off = 3.0 * h
            shapes.append([[(-U - 0.25 * P, X - h + off + 0.2 * A), (-U - 0.25 * P, X + off + 0.2 * A),
                            (x_end + 0.1 * P, top + off), (x_end + 0.1 * P, top + off - h * 1.4)]])
        return Glyph(shapes, left=min(x_end, -U), exit_x=-U, exit_edge=-U, stems=[(-u, X, 0)])
    bar_left = -U - 1.3 * P
    shapes = [g.rect(bar_left, -U, top - h, top)]
    if variant == "gaf":
        shapes.append(g.rect(bar_left + 0.15 * P, -U - 0.1 * P, top + 1.4 * h, top + 2.4 * h))
    post = -U - 2.2 * P
    shapes += [g.stem(st, -u, top, entry=_e(form), exit=True, round_br=not _e(form)),
               g.connector(st, post + u, -U), g.stem(st, post, 0.45 * X, entry=True, round_bl=True)]
    if variant == "arabic":
        shapes += marks(st, "ha", (post - u) / 2, at=0.42 * X)
    return Glyph(shapes, left=min(bar_left, post - u), stems=[(-u, top, 0), (post, 0.45 * X, 0)])


def lam(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, A = st.stem, st.stem / 2, st.x_height, st.ascender
    if form in ("I", "M"):
        shapes = [g.stem(st, -u, X + A, entry=_e(form), exit=True, round_br=not _e(form))]
        return Glyph(shapes, left=-U, exit_x=-U, exit_edge=-U, stems=[(-u, X + A, 0)])
    w = 3.4 * U
    shapes = [g.stem(st, -u, X + A, entry=_e(form)), bowl_down(st, -u, 0.4 * X, w, 0.45 * X)]
    return Glyph(shapes, left=-u - w - u, stems=[(-u, X + A, 0)])


def meem(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X, D = st.stem, st.stem / 2, st.x_height, st.descender
    xl = -(2 * U + 0.55 * st.white)
    top = 0.6 * X
    shapes = [g.ring(st, xl, 0, 0, top)]
    common = dict(entry_x=-U, entry_edge=0)
    if form in ("I", "M"):
        return Glyph(shapes, left=xl, exit_x=xl + U, exit_edge=xl, **common)
    shapes.append(g.stem(st, xl + u, top / 2, bottom=-D))
    return Glyph(shapes, left=xl, stems=[(xl + u, top / 2, -D)], **common)


def heh(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, X, A, h = st.stem, st.x_height, st.ascender, st.hair
    xl, w = head_box(st)
    top = X + 0.5 * A if form == "I" else X
    shapes = [g.ring(st, xl, 0, 0, top)]
    if form == "M":   # ـهـ : a hairline splits the counter into two eyes
        shapes.append(g.rect(xl + U * 0.9, -U * 0.9, X / 2 - h / 2, X / 2 + h / 2))
    shapes += marks(st, mark, xl / 2)
    return Glyph(shapes, left=xl, entry_x=-U, entry_edge=0,
                 exit_x=(xl + U) if _x(form) else None, exit_edge=xl if _x(form) else None)


def waw(st: Style, form: str, mark: Optional[str]) -> Glyph:
    U, u, X = st.stem, st.stem / 2, st.x_height
    xl, w = head_box(st)
    shapes = [g.ring(st, xl, 0, 0, X), tail(st, -u, X / 2)]
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
