"""Measuring a logo built of blocks, as سیرنگ was measured before بيدوناق was written from it.

The model looks at the logo and understands how its designer built it; this code measures it:

1. `outline`: the logo's ink as clean outlines. Every edge of a block logo is a straight line, so the
   contour is traced to a fraction of a pixel, its corners found, a line fitted to each edge between
   corners, and each corner put where two edge lines meet (as one would draw it with the pen tool).
   On كيوانو: 82 corners, the outline matches the logo's pixels 97.65%; no edge strays from its line by
   more than 0.4 px.
2. `cut_parts`: the outline cut into the logo's parts where its letters touch, along the lines the
   model gives (from one inner corner to the next, or through the white between two letters): each
   letter of the logo's word, its dots apart. They are measured (widths at every height, leans, the
   cut of every top, the head and its slot, the tail, the joins, the dot and the bar), and
   `rebuild` puts them back with the two measured spacing rules (a slit between joined letters,
   touching otherwise): on كيوانو 95.67%, as سیرنگ rebuilt from its measured parts matched 95.3%. That
   is the check of the measures; the parts themselves are never written into a word (the logo is its
   designer's work: what is taken from it is how it is built).
3. `fit`: the rules of the construction (`Style.block`), one value each, fitted to the logo's word
   drawn by the rules alone (blocks.py). Its designer cut every letter by hand (its two و, laid over
   each other after turning one 14 degrees, cover each other only 82.6%; its ن and ي teeth 69%): the
   rules carry how the logo is built, not each cut of its designer's knife.

`match` lays a word set by the engine over the logo (the share of ink in common, IoU).
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .style import Style

FRACTIONS = ("flare_h", "kaf_end_n", "kaf_corner_n")
FIXED = ("space", "touch", "slit", "into", "rock")


def raster(lay, size: Tuple[int, int], origin: Tuple[float, float], ss: int = 4) -> np.ndarray:
    """The layout as a mask of the logo's pixels: the word's right edge at origin[0], baseline at origin[1]."""
    from PIL import Image, ImageDraw
    W, H = size
    ox, base = origin
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for shape in lay.shapes:
        d.polygon([((ox + x) * ss, (base - y) * ss) for x, y in shape[0]], fill=255)
        for hole in shape[1:]:
            d.polygon([((ox + x) * ss, (base - y) * ss) for x, y in hole], fill=0)
    return np.array(im.resize((W, H), Image.BOX)) >= 128


def match(word: str, st: Style, ink: np.ndarray, origin: Tuple[float, float]) -> float:
    """How much of its ink the engine's drawing of `word` shares with the logo's (IoU, 0..1)."""
    from .typeset import typeset
    got = raster(typeset(word, st), (ink.shape[1], ink.shape[0]), origin)
    union = (got | ink).sum()
    return float((got & ink).sum() / union) if union else 0.0


# ---------------------------------------------------------------- 1. the outline
def _edges(gray: np.ndarray, thresh: float, zoom: int, tol: float, ink_dark: bool):
    """The contours of the ink, each as edges between corners, a line fitted to each edge."""
    from PIL import Image
    from skimage import measure
    g = np.asarray(gray, float)
    if not ink_dark:
        g = 255.0 - g
    big = np.array(Image.fromarray(g.astype(np.uint8)).resize((g.shape[1] * zoom, g.shape[0] * zoom), Image.BICUBIC), float)
    out = []
    for c in measure.find_contours(np.pad(big, 4, constant_values=255), thresh):
        c = (c - 4) / zoom
        pts = np.c_[c[:, 1], c[:, 0]]                                  # x, y (image)
        corners = measure.approximate_polygon(c, tol)
        corners = np.c_[corners[:, 1], corners[:, 0]]
        area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
        idx = [int(np.argmin(np.hypot(*(pts - q).T))) for q in corners]
        segs = []
        for k in range(len(idx) - 1):
            a, b = idx[k], idx[k + 1]
            seg = pts[a:b + 1] if b > a else np.r_[pts[a:], pts[:b + 1]]
            L = math.hypot(*(seg[-1] - seg[0]))
            s = dict(p0=seg[0], p1=seg[-1], len=L)
            if L >= 5.0 and len(seg) >= 6:
                inner = seg[len(seg) // 6: len(seg) - len(seg) // 6]   # the rounded ends left out
                m = inner.mean(0)
                d = np.linalg.svd(inner - m)[2][0]
                if np.dot(d, seg[-1] - seg[0]) < 0:
                    d = -d
                s.update(mid=m, dir=d, angle=math.degrees(math.atan2(-d[1], d[0])),
                         off=float(np.abs((inner - m) @ np.array([-d[1], d[0]])).max()))
            segs.append(s)
        out.append((area, segs))
    return out


def outline(gray: np.ndarray, thresh: float = 128.0, zoom: int = 4, tol: float = 1.2, ink_dark: bool = True,
            corner_reach: float = 3.0):
    """The ink of a block logo (a grey image, ink dark unless `ink_dark` is False) as clean outlines in
    image coordinates: each corner where two fitted edge lines meet (the short rounded corners kept).
    Returns a shapely geometry and the largest distance of an edge from its line (how straight the
    logo is: a block logo stays under half a pixel)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    def meet(a, b):
        M = np.array([[a["dir"][0], -b["dir"][0]], [a["dir"][1], -b["dir"][1]]])
        if abs(np.linalg.det(M)) < 1e-6:
            return None
        t = np.linalg.solve(M, b["mid"] - a["mid"])[0]
        return a["mid"] + t * a["dir"]

    inks, holes, worst = [], [], 0.0
    for area, segs in _edges(gray, thresh, zoom, tol, ink_dark):
        n = len(segs)
        if n < 3:
            continue
        pts = []
        for k in range(n):
            a, b = segs[k], segs[(k + 1) % n]
            corner = a["p1"]
            if "dir" in a and "dir" in b:
                worst = max(worst, a["off"])
                ang = abs((a["angle"] - b["angle"] + 180) % 360 - 180)
                x = meet(a, b)
                if ang > 12 and x is not None and np.hypot(*(x - corner)) < corner_reach:
                    pts.append(tuple(x))
                    continue
            pts.append(tuple(corner))
        poly = Polygon(pts).buffer(0)
        (inks if area < 0 else holes).append(poly)
    ink = unary_union(inks)
    if holes:
        ink = ink.difference(unary_union(holes))
    return ink, worst


# ---------------------------------------------------------------- 2. the parts
def cut_parts(ink, regions: Sequence[Tuple[str, Sequence[Tuple[float, float]]]], base: float,
              marks: Optional[Dict[str, Sequence[str]]] = None, body_above: float = 20.0) -> Dict:
    """Cut the logo's outline (`outline`, image coordinates) into its parts. `regions`: (name, polygon)
    in the order they are taken (each takes what is left of the outline inside its polygon): the marks
    first, then the letters. A letter's name is "letter.form" (ن.I); "نقطة", "شرطة" and "رأس" are the
    logo's dot, two-dot bar and head. `marks`: for a letter, the names of the marks that are its own
    (they stay with it when the letter stands for itself, and are left off when it lends its body to
    a sibling). Returns Style.block_parts: each part in the letter's coordinates, y up from the
    baseline `base`, x = 0 at the right edge of the letter's body (its ink above `body_above`)."""
    from shapely.affinity import scale, translate
    from shapely.geometry import Polygon, box

    def up(g):
        return scale(translate(g, 0, -base), 1, -1, origin=(0, 0))

    def rings(g):
        """[shape, ...], each shape [outer ring, holes...]"""
        out = []
        for p in getattr(g, "geoms", [g]):
            if p.geom_type == "Polygon" and not p.is_empty and p.area > 1.0:
                out.append([[[round(x, 2), round(y, 2)] for x, y in list(p.exterior.coords)[:-1]]] +
                           [[[round(x, 2), round(y, 2)] for x, y in list(h.coords)[:-1]] for h in p.interiors])
        return out

    taken = None
    raw = {}
    for name, poly in regions:
        g = ink.intersection(Polygon(poly))
        if taken is not None:
            g = g.difference(taken)
        taken = g if taken is None else taken.union(g)
        raw[name] = up(g)
    out: Dict = {}
    for name, g in raw.items():
        if g.is_empty:
            continue
        body = g.intersection(box(-1e5, body_above, 1e5, 1e5))
        right = (body if not body.is_empty else g).bounds[2]
        part = {"polys": rings(translate(g, -right, 0))}
        own = [raw[m] for m in (marks or {}).get(name, []) if m in raw]
        if own:
            part["marks"] = [r for m in own for r in rings(translate(m, -right, 0))]
        out[name] = part
    return out


def rebuild(parts: Dict, keys: Sequence[str], st: Style):
    """The check of the measures: the logo's word put back together from its measured parts (`keys`,
    "letter.form" in the word's order, a part used twice taking its "/2" drawing the second time),
    with the measured spacing rules alone (a slit between joined letters, touching otherwise). Returns
    the shapes (y up, the first letter's right edge at x = 0)."""
    from shapely.affinity import translate
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .blocks import above, gap_shift, rules
    R = rules(st)
    jy = R["bar"] + 2.0

    def geom(shapes):
        gs = []
        for sh in shapes or []:
            g = Polygon(sh[0]).buffer(0)
            for h in sh[1:]:
                g = g.difference(Polygon(h).buffer(0))
            gs.append(g)
        return unary_union(gs) if gs else Polygon()

    seen: Dict[str, int] = {}
    out, prev, prev_joins = [], None, False
    for key in keys:
        k = seen.get(key, 0)
        seen[key] = k + 1
        part = parts.get(f"{key}/{k + 1}") if k else None
        part = part or parts[key]
        body, marks = geom(part.get("polys")), geom(part.get("marks"))
        form = key.split(".")[1]
        joined = form in ("M", "F") and prev_joins
        x = 0.0 if prev is None else gap_shift(above(prev, jy if joined else 0.0), above(body, jy if joined else 0.0),
                                               R["slit"] if joined else R["touch"])
        prev = translate(body, x, 0)
        out += [prev, translate(marks, x, 0)]
        prev_joins = form in ("I", "M")
    return unary_union([g for g in out if not g.is_empty])


# ---------------------------------------------------------------- 3. the rules
def fit(word: str, st: Style, ink: np.ndarray, origin: Tuple[float, float], cycles: int = 2,
        log: Callable[[str], None] = print) -> Style:
    """Fit the rules of the block construction (`Style.block`) to the logo's ink (a boolean mask), the
    logo's `word` drawn by the rules alone (no part of the logo), its right edge at origin[0] and its
    baseline at origin[1]. Each rule is searched alone, in turn."""
    from scipy.optimize import minimize
    from .blocks import rules

    block = dict(rules(st))
    cur = replace(st, construction="block", block=block, block_parts={})

    def score() -> float:
        return match(word, cur, ink, origin)

    def search(put, x0, steps, iters):
        def f(x):
            put(x)
            return -score()
        f0 = f(x0)
        simplex = [x0] + [x0 + np.eye(len(x0))[j] * steps[j] for j in range(len(x0))]
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=np.array(simplex), maxiter=iters, xatol=0.05, fatol=1e-6))
        put(res.x if res.fun <= f0 else x0)

    log(f"start {100 * score():.2f}%")
    for cycle in range(cycles):
        # the numbers only: the choices of taste (words) are the designer's, not the logo's measures
        for k in [k for k in block if k not in FIXED and isinstance(block[k], (int, float))]:
            def put(x, k=k):
                block[k] = float(x[0])
            search(put, np.array([block[k]]), np.array([0.03 if k in FRACTIONS else 0.5 if k == "round" else 1.5]), 40)
        log(f"cycle {cycle}: rules {100 * score():.2f}%")
    return cur
