"""The 18 skeletons of a logo assembled from its own parts (the twelve kinds), the way Bidonaq was made
from Sirang: each part cut from the logo's outline where it is, the letters put together by where each
part's centre line ends and which way it runs there. Nothing is a fixed shape: a part the logo has is the
logo's; a part it lacks is drawn with the logo's own pen (qalam.Nib measured on its ink) along a centre
line written as a share of the logo's x-height, and marked as drawn, not taken (the designer may replace
it: the slot of a missing part).

The kinds (the owner's reference: twelve parts with an entry and an exit):
  stem  tooth  bowl (down a side, across, up)  flat (a long base)  tail  flag  head (closed)
  open_head  loop  dot  bar (the dash for two dots)  hamza
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

KINDS = ["stem", "tooth", "bowl", "flat", "tail", "flag", "head", "open_head", "loop", "dot", "bar", "hamza"]


@dataclass
class Piece:
    kind: str
    geom: object                       # shapely, y up
    spine: Optional[np.ndarray] = None # its centre line, in order (start = entry, end = exit)
    source: str = "logo"               # "logo": cut from it; "pen": drawn with its pen; "letter": a whole letter of it

    def end(self, last=True) -> Tuple[np.ndarray, np.ndarray]:
        sp = self.spine
        if sp is None or len(sp) < 2:
            c = self.geom.centroid; return np.array([c.x, c.y]), np.array([0.0, -1.0])
        p, q = (sp[-1], sp[max(-6, -len(sp))]) if last else (sp[0], sp[min(5, len(sp) - 1)])
        d = p - q
        return p, d / (np.hypot(*d) or 1.0)

    def moved(self, dx, dy) -> "Piece":
        return Piece(self.kind, affinity.translate(self.geom, dx, dy),
                     None if self.spine is None else self.spine + [dx, dy], self.source)

    def scaled(self, sx, sy, origin) -> "Piece":
        g = affinity.scale(self.geom, sx, sy, origin=tuple(origin))
        sp = None if self.spine is None else (self.spine - origin) * [sx, sy] + origin
        return Piece(self.kind, g, sp, self.source)

    @property
    def box(self):
        return self.geom.bounds


def weld(*gs, r=1.0):
    g = unary_union([x.geom if isinstance(x, Piece) else x for x in gs])
    return g.buffer(r, join_style=1).buffer(-r, join_style=1)


# ------------------------------------------------------------------ parts drawn with the logo's pen
# centre lines in shares of the x-height X, the entry at the right (Arabic runs right to left), y up,
# the baseline at 0. Only the course of the line is written here: its thickness everywhere is the
# logo's pen; its size the logo's x-height.
def _bez(p0, c1, c2, p1, n=24):
    t = np.linspace(0, 1, n)[:, None]
    p0, c1, c2, p1 = map(np.array, (p0, c1, c2, p1))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t * t * c2 + t ** 3 * p1

PEN_LINES = {
    "stem": lambda: np.array([[0, 1.45], [0, 0.0]]),
    "tooth": lambda: np.array([[0, 0.75], [0, 0.0]]),
    "flag": lambda: np.array([[0.0, 0.0], [-0.32, 0.18]]),
    "flat": lambda: np.vstack([_bez((0.15, 0.35), (0.05, -0.05), (-0.6, -0.05), (-1.6, 0.0)),
                               _bez((-1.6, 0.0), (-1.75, 0.0), (-1.85, 0.1), (-1.85, 0.3))[1:]]),
    "bowl": lambda: np.vstack([_bez((0.0, 0.5), (0.0, -0.5), (-1.3, -0.6), (-1.35, 0.25))]),
    "tail": lambda: _bez((0.0, 0.5), (0.0, -0.2), (-0.4, -0.55), (-1.1, -0.6)),
    "head": lambda: np.vstack([_bez((0.0, 0.25), (0.05, 0.95), (-0.7, 0.95), (-0.7, 0.35)),
                               _bez((-0.7, 0.35), (-0.7, -0.05), (-0.1, -0.05), (0.0, 0.25))[1:]]),
    "open_head": lambda: _bez((-0.05, 0.95), (-0.55, 1.15), (-0.95, 0.75), (-0.55, 0.45)),
    "loop": lambda: np.vstack([_bez((0.0, 0.05), (0.1, 0.75), (-1.0, 0.75), (-1.1, 0.25)),
                               _bez((-1.1, 0.25), (-1.15, -0.05), (-0.4, -0.05), (0.0, 0.05))[1:]]),
    "hah_head": lambda: _bez((0.05, 0.9), (-0.25, 1.0), (-0.7, 0.95), (-0.95, 0.72)),
    "hamza": lambda: np.vstack([_bez((0.0, 0.3), (-0.25, 0.4), (-0.3, 0.15), (-0.05, 0.12)),
                                _bez((-0.05, 0.12), (-0.25, 0.05), (-0.25, -0.05), (0.05, 0.0))[1:]]),
}


def pen_piece(kind: str, X: float, nib, at=(0.0, 0.0), slant: float = 0.0) -> Piece:
    """A part the logo lacks, drawn with its pen along the centre line written for that kind, leaned as the
    logo leans (slant, degrees: the centre line is sheared, the nib keeps its own angle)."""
    from .qalam import stroke, dot
    t = math.tan(math.radians(slant))
    if kind == "dot":
        return Piece(kind, dot(at[0], at[1], nib), None, "pen")
    if kind == "bar":
        sp = np.array([[0.0, 0.0], [-0.45 * X, 0.08 * X]]) + at
        return Piece(kind, stroke(sp, nib, taper=(0.15, 0.3)), sp, "pen")
    sp = PEN_LINES[kind]() * X + np.asarray(at, float)
    sp = sp + np.c_[sp[:, 1] * t, np.zeros(len(sp))]
    return Piece(kind, stroke(sp, nib, taper=(0.1, 0.25) if kind not in ("head", "loop") else (0.0, 0.0)), sp, "pen")


# ------------------------------------------------------------------ the logo's parts
@dataclass
class Kit:
    """What the logo gives: its x-height, its pen, its parts by kind, and its whole letters."""
    X: float
    nib: object
    parts: Dict[str, Piece] = field(default_factory=dict)
    letters: Dict[str, Piece] = field(default_factory=dict)
    slant: float = 0.0                  # degrees, the logo's lean (its parts lean already; drawn ones are leaned)

    def get(self, kind: str) -> Piece:
        p = self.parts.get(kind)
        return p if p is not None else pen_piece(kind, self.X, self.nib, slant=self.slant)

    def has(self, kind: str) -> bool:
        return kind in self.parts


# ------------------------------------------------------------------ putting parts together
def foot(p: Piece) -> np.ndarray:
    pts = np.array(p.geom.exterior.coords) if p.geom.geom_type == "Polygon" else np.vstack([np.array(q.exterior.coords) for q in p.geom.geoms])
    return pts[pts[:, 1].argmin()]


def top_point(p: Piece) -> np.ndarray:
    pts = np.array(p.geom.exterior.coords) if p.geom.geom_type == "Polygon" else np.vstack([np.array(q.exterior.coords) for q in p.geom.geoms])
    return pts[pts[:, 1].argmax()]


def base(p: Piece, y=0.0) -> Piece:
    """stood on the baseline by its lowest point"""
    return p.moved(0.0, y - p.box[1])


def stand_on(stem: Piece, at: np.ndarray, sink: float) -> Piece:
    f = foot(stem)
    return stem.moved(at[0] - f[0], at[1] - f[1] - sink)


def hang_under(lower: Piece, upper: Piece, overlap: float, end_last=True) -> Piece:
    """the lower part's entry (its spine's start, or end) set under the upper part's lowest point"""
    p, _ = lower.end(last=not end_last) if lower.spine is not None else (top_point(lower), None)
    f = foot(upper)
    return lower.moved(f[0] - p[0], f[1] - p[1] + overlap)


def mark(body, m: Piece, above=True, gap=0.12, X=1.0, at=0.5) -> Piece:
    x0, y0, x1, y1 = body.bounds if not isinstance(body, Piece) else body.box
    mx0, my0, mx1, my1 = m.box
    cx = x0 + at * (x1 - x0)
    ty = (y1 + gap * X) if above else (y0 - gap * X - (my1 - my0))
    return m.moved(cx - (mx0 + mx1) / 2, ty - my0)


def height_to(p: Piece, h: float) -> Piece:
    """a stem made as tall as h, as a designer drags its top (vparts.stretch_letter), else scaled"""
    from .vparts import stretch_letter
    x0, y0, x1, y1 = p.box
    if y1 - y0 >= h - 1e-6:
        return p
    g, _ = stretch_letter(p.geom, "y", h - (y1 - y0), max(1.0, (y1 - y0)))
    if g is None:
        return p.scaled(1.0, h / (y1 - y0), (x0, y0))
    sp = None
    if p.spine is not None:
        sp = p.spine.copy(); top = sp[:, 1] > (y0 + y1) / 2; sp[top, 1] += h - (y1 - y0)
    return Piece(p.kind, g, sp, p.source)


def skeletons(kit: Kit) -> Dict[str, Tuple[object, List[str]]]:
    """The 18 skeletons, isolated, each with the kinds of part it was made of that the logo lacked."""
    X = kit.X
    out: Dict[str, Tuple[object, List[str]]] = {}
    used: List[str] = []

    def take(kind):
        if not kit.has(kind):
            used.append(kind)
        return base(kit.get(kind)) if kind not in ("dot", "bar", "hamza") else kit.get(kind)

    def done(c, *gs):
        out[c] = (weld(*gs), sorted(set(used))); used.clear()

    whole = kit.letters
    # ا: the logo's own, else a stem as tall as the ascender
    a = base(whole["ا"]) if "ا" in whole else height_to(take("stem"), 1.45 * X)
    done("ا", a)
    # ب: the long base, a dot under it
    f = take("flat"); done("ب", f, mark(f, take("dot"), above=False, X=X))
    # د: a short upright with a flag, on a short base: the logo's own, else a tooth on the base, a flag
    if "د" in whole:
        done("د", base(whole["د"]))
    else:
        t = take("tooth"); fl = take("flag")
        b2 = pen_piece("flat", 0.55 * X, kit.nib, slant=kit.slant); used.append("flat")
        b2 = base(b2); t2 = stand_on(t, b2.end(last=False)[0], 0.1 * X)
        fl2 = fl.moved(t2.box[0] - fl.box[2] + 0.05 * X, t2.box[3] - fl.box[3])
        done("د", b2, t2, fl2)
    # ر و: the logo's own
    for c in ("ر", "و"):
        if c in whole:
            done(c, base(whole[c]))
        elif c == "ر":
            done(c, take("tail"))
        else:
            h = take("head"); tl = hang_under(take("tail"), h, 0.12 * X); done(c, h, tl)
    # س: the head of the logo's ش if it draws its teeth as one (else teeth), into a bowl
    b = take("bowl")
    if kit.has("teeth"):
        te = kit.parts["teeth"]
    else:
        te = take("tooth")
    te = te.moved(b.end(last=False)[0][0] - te.box[0] - 0.08 * X, 0.0)
    te = base(te, b.end(last=False)[0][1] - 0.15 * X)
    done("س", b, te)
    # ص: a loop on a tooth, into a bowl
    lp = take("loop"); b = take("bowl")
    lp = lp.moved(b.box[2] - lp.box[0] - 0.05 * X, b.end(last=False)[0][1] - lp.box[1] - 0.25 * X)
    done("ص", b, lp)
    # ط: a loop and a stem rising from its left part
    lp = base(take("loop")); st = height_to(take("stem"), 1.45 * X)
    st = stand_on(st, np.array([lp.box[0] + 0.35 * (lp.box[2] - lp.box[0]), lp.box[1]]), 0.0)
    done("ط", lp, st)
    # ع ح: the open head over a bowl turned to the right
    for c in ("ع", "ح"):
        oh = take("open_head") if c == "ع" else take("hah_head")
        lower = take("bowl").scaled(-1.0, 1.0, (0.0, 0.0))     # mirrored: the bowl of ح ع opens right
        lower = Piece(lower.kind, lower.geom, None if lower.spine is None else lower.spine, "pen" if not kit.has("bowl") else "logo-mirrored")
        oh = oh.moved(lower.box[0] - oh.box[0], lower.box[3] - oh.box[1] - 0.1 * X)
        if c == "ح":
            oh = oh.moved(0.0, 0.0)
        done(c, lower, oh)
    # ف ق: the head, on a base (ف) or a bowl (ق) going left from under it
    for c, low, dots in (("ف", "flat", ("dot",)), ("ق", "bowl", ("bar",))):
        h = take("head"); lo = take(low)
        p0 = lo.end(last=False)[0]; f0 = foot(h)
        h = h.moved(p0[0] - f0[0] - 0.05 * X, p0[1] - f0[1] - 0.12 * X)
        ms = [mark(h, take(dots[0]), X=X)]
        done(c, lo, h, *ms)
    # ك: a base, an upright at its right end, the arm (the logo's dash) to the left of the upright's top
    f = take("flat"); st = height_to(take("stem"), 1.3 * X)
    st = stand_on(st, f.end(last=False)[0], 0.12 * X)
    arm = take("bar"); arm = arm.moved(st.box[0] - arm.box[2] + 0.1 * X, st.box[3] - arm.box[3] - 0.1 * X)
    done("ك", f, st, arm)
    # ل: a bowl, an upright on its entry
    b = take("bowl"); st = height_to(take("stem"), 1.45 * X)
    done("ل", b, stand_on(st, b.end(last=False)[0], 0.12 * X))
    # م: the head, a tail straight down from under it
    h = base(take("head")); tl = take("tail")
    p = foot(h); q = top_point(tl)
    done("م", h, tl.moved(p[0] - q[0] + 0.1 * X, p[1] - q[1] + 0.15 * X))
    # ن: a bowl, a dot over it
    b = take("bowl"); done("ن", b, mark(b, take("dot"), X=X))
    # ه: the head alone
    done("ه", base(take("head")))
    # ى: the bowl alone (the logo's own final ی if it has one)
    done("ى", base(whole["ى"]) if "ى" in whole else take("bowl"))
    return out
