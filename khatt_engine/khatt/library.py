"""Stage 4: the part library — real pieces cut out of the logo, traced as Bezier curves.

Parts (all in original-image pixels, upright, baseline at y = 0):
  shaft        upper part of a stem with the logo's own top cut; stretches to any height
  half_r_join  bottom-right of a stem that flows into a join (half of a counter)
  half_l_join  bottom-left of a stem that flows into a join
  half_r_free  bottom-right of a stem with nothing on its right
  half_l_free  bottom-left of a stem with nothing on its left
  tail         the part of a descending stem below the join line, with its hairline tail
  dot          a dot
  dash         a hairline dash (two dots, or the lines of گ)
  loop         a closed head (when the logo has one)
"""
from __future__ import annotations

import contextvars
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours

from .bezier import fit_closed, sample
from .measure import Line, Stem
from .understand import Reading

Poly = List[Tuple[float, float]]
ACTIVE: "contextvars.ContextVar[Optional[Library]]" = contextvars.ContextVar("khatt_library", default=None)


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


class Tagged(list):
    """A shape (list of polygons, even-odd) that remembers where it came from."""
    def __init__(self, polys, source: str = "engine", name: str = ""):
        super().__init__(polys)
        self.source = source
        self.name = name


class Compound(list):
    """Several shapes that must be unioned (never even-odd'ed together)."""


PART_NAMES = {
    "bottom_jj": "أسفل سن متصل من الجهتين", "bottom_fj": "أسفل سن حر يمينًا متصل يسارًا",
    "bottom_jf": "أسفل سن متصل يمينًا حر يسارًا", "bottom_ff": "أسفل سن حر من الجهتين",
    "shaft": "جذع (الطرف العلوي)", "half_r_join": "أسفل الجذع نحو وصلة يمين", "half_l_join": "أسفل الجذع نحو وصلة يسار",
    "half_r_free": "أسفل الجذع الحر يمينًا", "half_l_free": "أسفل الجذع الحر يسارًا", "tail": "ذيل نازل",
    "dot": "نقطة", "dash": "شرطة شعرية", "loop": "رأس مغلق",
}


@dataclass
class Part:
    key: str
    source: str                         # which letter of the logo it was cut from
    curves: List[List[tuple]]           # Bezier segments per contour (normalised)
    polys: List[Poly]                   # the same, sampled
    info: Dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return PART_NAMES.get(self.key, self.key)


@dataclass
class Library:
    stem: float
    hair: float
    x_height: float
    white: float
    y_cut: float
    parts: Dict[str, Part] = field(default_factory=dict)
    used: Dict[str, int] = field(default_factory=dict)

    def _log(self, key: str) -> None:
        self.used[key] = self.used.get(key, 0) + 1

    def has(self, key: str) -> bool:
        return key in self.parts

    # -------------------------------------------------------------- builders used by the letters
    def shaft(self, xc: float, top: float, bottom: Optional[float] = None) -> Optional[Tagged]:
        p = self.parts.get("shaft")
        if p is None:
            return None
        bottom = (self.y_cut - 0.3 * self.stem) if bottom is None else bottom
        cap = p.info["cap"]
        ymin = p.info["ymin"]                # normalised (top = 0, negative downwards)
        if top - bottom < cap + 1:
            return None
        s = (top - cap - bottom) / max(1e-6, (-cap - ymin))
        polys = [[(x + xc, top + y if y >= -cap else top - cap + (y + cap) * s) for x, y in poly] for poly in p.polys]
        self._log("shaft")
        return Tagged(polys, "logo", p.name)

    def half(self, side: str, joined: bool, xc: float, white: Optional[float] = None) -> Optional[Tagged]:
        key = f"half_{side}_{'join' if joined else 'free'}"
        p = self.parts.get(key)
        if p is None:
            return None
        w2 = self.stem / 2
        k = 1.0
        if joined and white:
            k = (white / 2) / max(1e-6, p.info["white"] / 2)

        def mx(x):
            ax = abs(x)
            return x if ax <= w2 else math.copysign(w2 + (ax - w2) * k, x)
        self._log(key)
        return Tagged([[(mx(x) + xc, y) for x, y in poly] for poly in p.polys], "logo", p.name)

    def tail(self, xc: float) -> Optional[Tagged]:
        p = self.parts.get("tail")
        if p is None:
            return None
        self._log("tail")
        return Tagged([[(x + xc, y) for x, y in poly] for poly in p.polys], "logo", p.name)

    def dot(self, cx: float, cy: float) -> Optional[Tagged]:
        p = self.parts.get("dot")
        if p is None:
            return None
        self._log("dot")
        return Tagged([[(x + cx, y + cy) for x, y in poly] for poly in p.polys], "logo", p.name)

    def dash(self, cx: float, cy: float, length: float) -> Optional[Tagged]:
        p = self.parts.get("dash")
        if p is None:
            return None
        k = length / max(1e-6, p.info["length"])
        self._log("dash")
        return Tagged([[(x * k + cx, y + cy) for x, y in poly] for poly in p.polys], "logo", p.name)

    def loop(self, xl: float, xr: float, yb: float, yt: float) -> Optional[Tagged]:
        p = self.parts.get("loop")
        if p is None:
            return None
        kx, ky = (xr - xl) / p.info["w"], (yt - yb) / p.info["h"]
        if not (0.6 <= kx <= 1.6 and 0.6 <= ky <= 1.6):
            return None            # too much distortion: let the engine draw it with the logo's pen
        self._log("loop")
        return Tagged([[(xl + x * kx, yb + y * ky) for x, y in poly] for poly in p.polys], "logo", p.name)

    def bottom(self, xc: float, joined_r: bool, joined_l: bool, white: Optional[float]) -> Optional[Tagged]:
        """The foot of a stem, from the logo. Exact type if the logo has it; otherwise the right
        and left sides are taken from two real feet with the same kind of side."""
        key = "bottom_" + ("j" if joined_r else "f") + ("j" if joined_l else "f")
        w2 = self.stem / 2

        def place(p: Part, side: Optional[str]) -> List[Poly]:
            kr = (white / 2) / max(1e-6, p.info["white_r"] / 2) if joined_r and white and p.info.get("white_r") else 1.0
            kl = (white / 2) / max(1e-6, p.info["white_l"] / 2) if joined_l and white and p.info.get("white_l") else 1.0

            def mx(x):
                if x > w2:
                    return w2 + (x - w2) * kr
                if x < -w2:
                    return -w2 + (x + w2) * kl
                return x
            polys = p.polys
            if side == "r":
                polys = [clip_x(poly, -0.1 * self.stem, True) for poly in polys]
            elif side == "l":
                polys = [clip_x(poly, 0.1 * self.stem, False) for poly in polys]
            return [[(mx(x) + xc, y) for x, y in poly] for poly in polys if len(poly) >= 3]

        if key in self.parts:
            self._log(key)
            return Tagged(place(self.parts[key], None), "logo", self.parts[key].name)
        right = [k for k in self.parts if k.startswith("bottom_" + ("j" if joined_r else "f"))]
        left = [k for k in self.parts if k.startswith("bottom_") and k.endswith("j" if joined_l else "f")]
        if right and left:
            self._log(key)
            polys = place(self.parts[right[0]], "r") + place(self.parts[left[0]], "l")
            return Compound([Tagged(polys[: len(place(self.parts[right[0]], "r"))], "logo", self.parts[right[0]].name),
                             Tagged(polys[len(place(self.parts[right[0]], "r")):], "logo", self.parts[left[0]].name)])
        return None

    def build_stem(self, xc: float, top: float, entry: bool, exit: bool, fallback) -> Optional[Compound]:
        """A whole stem from logo parts: the shaft and the foot. `fallback(top)` draws an engine
        stem for the foot when the logo has nothing usable."""
        sh = self.shaft(xc, top)
        if sh is None:
            return None
        out = Compound([sh])
        foot = self.bottom(xc, entry, exit, self.white)
        if foot is None:
            out.append(Tagged(fallback(self.y_cut + 0.3 * self.stem), "engine", "أسفل جذع مستنتج"))
        elif isinstance(foot, Compound):
            out += foot
        else:
            out.append(foot)
        return out

    def summary(self) -> List[Dict]:
        return [{"key": p.key, "name": p.name, "source": p.source,
                 **{k: round(v, 2) for k, v in p.info.items()}} for p in self.parts.values()]


# ------------------------------------------------------------------ extraction
def _trace(L: Line, comp: int, c0: float, c1: float, r0: float, r1: float, k: float,
           ox: float, oy: float, err: float) -> Tuple[List[List[tuple]], List[Poly]]:
    """Trace the ink of one component inside a window (raster cols c0..c1, rows r0..r1);
    the window edges become straight cuts. Returns normalised Bezier curves and polygons."""
    H, W = L.G.shape
    a0, a1 = max(0, int(math.floor(r0))), min(H, int(math.ceil(r1)) + 1)
    b0, b1 = max(0, int(math.floor(c0))), min(W, int(math.ceil(c1)) + 1)
    own = ndi.binary_dilation(L.lab[a0:a1, b0:b1] == comp, iterations=2)
    g = np.where(own, L.G[a0:a1, b0:b1], 0.0)
    g = np.pad(g, 1)
    curves_all, polys = [], []
    for c in find_contours(g, L.level):
        if len(c) < 8:
            continue
        pts = [((L.x_of_col(q[1] - 1 + b0) - ox) * k, (L.y_of_row(q[0] - 1 + a0) - oy) * k) for q in c]
        cv = fit_closed(pts, err)
        if not cv:
            continue
        curves_all.append(cv)
        polys.append(sample(cv, 8))
    return curves_all, polys


def build_library(L: Line, R: Reading, k: float) -> Library:
    """k converts working pixels to original-image pixels."""
    X = L.x_height
    lib = Library(stem=L.stem * k, hair=L.hair * k, x_height=X * k, white=(L.white if L.white == L.white else L.stem) * k,
                  y_cut=0.45 * X * k)
    err = max(0.3, 0.02 * L.stem)
    names = lambda s: (f"{s.letter}" if s.letter else "") + (" — " if s.letter else "") + {"tooth": "سن", "lowered": "سن منخفض", "tall": "جذع صاعد", "short": "جذع قصير", "descending": "جذع نازل"}.get(s.role, "جذع")
    joins = {(j.right, j.left): j for j in R.joins if j.kind == "join"}
    y_cut = 0.45 * X
    # ---- shaft: the most typical full-height tooth
    teeth = [i for i, s in enumerate(L.stems) if s.role == "tooth"] or [i for i, s in enumerate(L.stems) if s.role in ("lowered", "tall")]
    if teeth:
        i = min(teeth, key=lambda j: abs(L.stems[j].w - L.stem))
        s = L.stems[i]
        cu, po = _trace(L, s.comp, L.col_of_x(s.x - s.w / 2 - 3), L.col_of_x(s.x + s.w / 2 + 3),
                        L.row_of_y(s.top + 3), L.row_of_y(y_cut - 0.3 * L.stem), k, s.x, s.top, err)
        if po:
            ys = [y for poly in po for _, y in poly]
            lib.parts["shaft"] = Part("shaft", f"السطر {L.idx + 1}: {names(s)}", cu, po,
                                      {"width": s.w * k, "cap": 0.6 * s.w * k, "ymin": min(ys), "top_above_x": (s.top - X) * k})
    # ---- feet: the whole bottom of a stem, classified by what is attached on each side
    join_of = {(j.right, j.left): j for j in R.joins}
    feet: Dict[str, List] = {}
    for i, s_ in enumerate(L.stems):
        if s_.role == "descending":
            continue
        jr, jl = join_of.get((i - 1, i)), join_of.get((i, i + 1))
        if (jr and jr.kind != "join") or (jl and jl.kind != "join"):
            continue                          # sides that open into a wide bowl are not teeth feet
        if not jr and not jl and abs(s_.bottom) > 0.25 * L.stem:
            continue
        nb_r = i > 0 and L.stems[i - 1].comp == s_.comp and L.stems[i - 1].x - s_.x < 6 * L.stem
        nb_l = i + 1 < len(L.stems) and L.stems[i + 1].comp == s_.comp and s_.x - L.stems[i + 1].x < 6 * L.stem
        if (nb_r and not jr) or (nb_l and not jl):
            continue                          # attached to something that is not a plain join
        if (jr and L.stems[i - 1].role == "descending") or (jl and L.stems[i + 1].role == "descending"):
            continue                          # the counter next to a descending stem is not a typical one
        if jr:
            a = L.stems[i - 1]
            right = ((a.x - a.w / 2) + (s_.x + s_.w / 2)) / 2 + 0.12 * jr.white
        else:
            right = s_.x + s_.w / 2 + 0.7 * s_.w
        if jl:
            b = L.stems[i + 1]
            left = ((s_.x - s_.w / 2) + (b.x + b.w / 2)) / 2 - 0.12 * jl.white
        else:
            left = s_.x - s_.w / 2 - 0.7 * s_.w
        key = "bottom_" + ("j" if jr else "f") + ("j" if jl else "f")
        feet.setdefault(key, []).append((i, left, right, jr.white if jr else 0.0, jl.white if jl else 0.0))
    typical = L.white if L.white == L.white else L.stem
    for key, ex in feet.items():
        # the foot whose counters are closest to the typical white: it needs the least stretching
        i, left, right, wr, wl = min(ex, key=lambda e: (abs(e[3] - typical) if e[3] else 0) + (abs(e[4] - typical) if e[4] else 0))
        s_ = L.stems[i]
        cu, po = _trace(L, s_.comp, L.col_of_x(left), L.col_of_x(right), L.row_of_y(y_cut + 0.3 * L.stem),
                        L.row_of_y(-0.6 * L.stem), k, s_.x, 0.0, err)
        if po:
            lib.parts[key] = Part(key, f"السطر {L.idx + 1}: {names(s_)}", cu, po,
                                  {"white_r": wr * k, "white_l": wl * k, "width": s_.w * k})
    # ---- tail of a descending stem
    for t in R.tails:
        if t.stem is None or t.direction != "left":
            continue
        s = L.stems[t.stem]
        cu, po = _trace(L, s.comp, L.col_of_x(t.x0 - 2), L.col_of_x(s.x + s.w / 2 + 1), L.row_of_y(y_cut + 0.3 * L.stem),
                        L.row_of_y(t.y0 - 2), k, s.x, 0.0, err)
        if po:
            lib.parts["tail"] = Part("tail", f"السطر {L.idx + 1}: {names(s)}", cu, po,
                                     {"depth": -t.y0 * k, "reach": (s.x - t.x0) * k, "top": s.top * k})
            break
    # ---- dot and dash
    if L.dots:
        med = float(np.median([p.size for p in L.dots]))
        p = min(L.dots, key=lambda d: abs(d.size - med))
        po = [[((x - p.cx) * k, (y - p.cy) * k) for x, y in poly] for poly in p.contours]
        cu = [fit_closed(poly, err * k) for poly in po]
        lib.parts["dot"] = Part("dot", f"السطر {L.idx + 1}: نقطة", cu, [sample(c, 8) for c in cu if c],
                                {"diameter": p.size * k, "roundness": p.roundness})
    dashes = sorted(L.dashes, key=lambda d: d.size)
    if dashes:
        p = dashes[0]                        # the shorter dash stands for two dots
        po = [[((x - (p.x0 + p.x1) / 2) * k, (y - p.cy) * k) for x, y in poly] for poly in p.contours]
        cu = [fit_closed(poly, err * k) for poly in po]
        lib.parts["dash"] = Part("dash", f"السطر {L.idx + 1}: شرطة", cu, [sample(c, 8) for c in cu if c],
                                 {"length": p.size * k, "thick": p.thick * k})
    # ---- a closed head
    heads = [c for c in L.counters if (c.x1 - c.x0) < 2.5 * L.stem]
    if heads:
        c = max(heads, key=lambda v: v.area)
        x0, x1 = c.x0 - 1.2 * L.stem, c.x1 + 1.2 * L.stem
        y0, y1 = c.y0 - 1.6 * L.hair, c.y1 + 1.6 * L.hair
        cu, po = _trace(L, c.comp, L.col_of_x(x0), L.col_of_x(x1), L.row_of_y(y1), L.row_of_y(y0), k, x0, y0, err)
        if po:
            xs = [x for poly in po for x, _ in poly]; ys = [y for poly in po for _, y in poly]
            sx, sy = min(xs), min(ys)
            po = [[(x - sx, y - sy) for x, y in poly] for poly in po]
            cu = [[tuple((q[0] - sx, q[1] - sy) for q in seg) for seg in c] for c in cu]
            lib.parts["loop"] = Part("loop", f"السطر {L.idx + 1}: رأس مغلق", cu, po,
                                     {"w": max(xs) - sx, "h": max(ys) - sy})
    return lib
