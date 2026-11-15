"""Letters drawn on the structure of a reference typeface, with the logo's own pen.

This is how the hand-made words were drawn when the logo lacked a letter: the typeface (shaped with
HarfBuzz, so every letter takes its form) says how the letter is built: where its nose is, where its
counter, where its tooth. The logo says everything else: the x-height, the pen (thick and thin
by the direction of the stroke), the rhythm, the dots. Only the structure is taken from the typeface;
its weight and its outline are not.

  1. the body of the letter in its form, without its dots and without the joins to its neighbours;
  2. its centre line (skeleton), each stroke followed from end to end;
  3. scaled to the logo: its tooth to the logo's x-height, its width to the logo's rhythm;
  4. drawn again with the logo's pen (style.width_at): strokes that rest on the baseline or reach
     the x-height keep their edge there, and every stroke ends where the typeface ends it;
  5. the dots and marks of the engine's letter (which are the logo's) put over it.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .letters import LETTERS, Glyph
from .library import Tagged
from .style import Style

PX = 120.0                     # the typeface's tooth drawn this many pixels high to read its structure
FORMS = {"i": "{}", "I": "{}ـ", "M": "ـ{}ـ", "F": "ـ{}"}


def faces() -> List[str]:
    """The reference typefaces at hand (the open typefaces of the examination, if fetched)."""
    from .bench.fonts import font_files
    return [str(p) for p in font_files()]


@lru_cache(maxsize=4096)
def _structure(path: str, char: str, form: str):
    """The centre line of the letter's body in the typeface, in units of its tooth (x-height = 1):
    a list of strokes, each a list of (x, y, r) with r the typeface's half-width there; x = 0 at the
    right end of the body. None when the typeface has no such letter."""
    from skimage.morphology import skeletonize
    from .bench.fonts import load, shape
    from .bench.truth import drop_marks, raster, x_height
    from .skeleton import _trace
    face = load(path)
    text = FORMS[form].format(char)
    idx = text.index(char)
    try:
        cs = [c for cl, cont in shape(face, text) if cl == idx for c in cont]
    except Exception:
        return None
    if not cs:
        return None
    X = x_height(face)
    s = PX / X
    xs = [p[0] for c in cs for p in c]; ys = [p[1] for c in cs for p in c]
    pad = 4
    W, H = int((max(xs) - min(xs)) * s) + 2 * pad, int((max(ys) - min(ys)) * s) + 2 * pad
    if W < 4 or H < 4 or W * H > 4_000_000:
        return None
    top = max(ys) + pad / s
    m = raster(cs, min(xs) - pad / s, top, s, W, H, ss=2) > 0.5
    m = drop_marks(m, PX)
    if m.sum() < 20:
        return None
    sk = skeletonize(m).astype(bool)
    rad = ndi.distance_transform_edt(m)
    # a thick stroke's skeleton grows short branches towards its corners and flared ends: a branch
    # from a free end to a junction shorter than the stroke is wide there is pruned, until none is left
    for _ in range(6):
        nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
        ends = sk & (nb == 1)
        cut = False
        for path in _trace(sk):
            a, b = path[0], path[-1]
            free_a, free_b = ends[a], ends[b]
            if free_a == free_b:
                continue
            junction = b if free_a else a
            if len(path) < 1.6 * rad[junction] + 2:
                for q in (path[:-1] if free_a else path[1:]):
                    sk[q] = False
                cut = True
        if not cut:
            break
    holes = ndi.binary_fill_holes(m) & ~m
    lab, nh = ndi.label(holes)
    counters = [float(np.sum(lab == i)) / (PX * PX) for i in range(1, nh + 1) if np.sum(lab == i) >= 0.004 * PX * PX]
    strokes = []
    for path in _trace(sk):
        if len(path) < 3:
            continue
        # smoothed along the stroke (a pixel staircase is not the typeface's curve), every 3 pixels
        arr = np.array(path, float)
        if len(arr) >= 5:
            k = np.ones(5) / 5
            sm = np.vstack([np.convolve(np.pad(arr[:, j], 2, mode="edge"), k, mode="valid") for j in range(2)]).T
            sm[0], sm[-1] = arr[0], arr[-1]
            arr = sm
        keep = [0] + [i for i in range(1, len(arr) - 1) if i % 3 == 0] + [len(arr) - 1]
        pts = []
        for r, c in arr[keep]:
            ri, ci = int(round(min(max(r, 0), H - 1))), int(round(min(max(c, 0), W - 1)))
            x = (c - pad) / s + min(xs)
            y = top - r / s
            pts.append((x / X, y / X, float(rad[ri, ci]) / PX))
        if len(pts) >= 2:
            strokes.append(pts)
    if not strokes:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    right = ((cols[-1] - pad) / s + min(xs)) / X
    return [[(x - right, y, r) for x, y, r in stroke] for stroke in strokes], ((cols[0] - pad) / s + min(xs)) / X - right, counters


WIDER = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.4)


def ref_glyph(path: str, char: str, form: str, st: Style, wide: float = 1.0, engine: Optional[Glyph] = None) -> Optional[Glyph]:
    """`char` in `form`, built on the structure of the typeface at `path` and drawn with the logo's pen.
    `wide`: how much wider than the typeface the logo's rhythm draws a letter. A letter with a counter
    (a loop, a head) is drawn wider still if the logo's pen would close it, as a hand lengthens a loop
    to fit a heavy pen. The marks are the engine's letter's (`engine`, built if not given)."""
    got = _structure(path, char, form)
    if got is None:
        return None
    need = len(got[2])
    best = None
    for k in WIDER:
        g = _draw(got, char, form, st, wide * k, engine)
        if g is None:
            return None
        open_ = _counters(g, st)
        if open_ >= need:
            return g
        if best is None or open_ > best[0]:
            best = (open_, g)
    return best[1]


def _counters(g: Glyph, st: Style) -> int:
    """How many open counters the body has, big enough to read (a quarter of the white between teeth, squared)."""
    from shapely.geometry import Polygon
    ring = g.shapes[0]
    n = 0
    for poly in ring[1:]:
        if len(poly) >= 3 and abs(Polygon(poly).area) >= 0.25 * st.white * st.white:
            n += 1
    return n


def _draw(got, char: str, form: str, st: Style, wide: float, engine: Optional[Glyph]) -> Optional[Glyph]:
    """One drawing of the structure with the logo's pen, at one width."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    strokes, left, _ = got
    X = st.x_height
    sx, sy = X * wide, X
    edge = 0.06 * X                                    # how near a line an edge must come to rest on it
    shapes = []
    for stroke in strokes:
        pts = []
        n = len(stroke)
        for i, (x, y, r) in enumerate(stroke):
            a, b = stroke[max(0, i - 1)], stroke[min(n - 1, i + 1)]
            dx, dy = (b[0] - a[0]) * sx, (b[1] - a[1]) * sy
            d = math.hypot(dx, dy) or 1.0
            tx, ty = dx / d, dy / d
            w = st.width_at(ty)
            X0, Y0, R = x * sx, y * sy, r * sy
            # an edge the typeface rests on the baseline or the x-height stays there with the logo's pen
            if abs(Y0 - R) < edge and abs(ty) < 0.5:
                Y0 = w / 2
            elif abs(Y0 + R - X) < edge and abs(ty) < 0.5:
                Y0 = X - w / 2
            pts.append((X0, Y0, tx, ty, w, R))
        # every stroke ends where the typeface ends it: the centre line stops short by its half-width
        for end, j in ((0, 1), (len(pts) - 1, len(pts) - 2)):
            if len(pts) < 2:
                break
            X0, Y0, tx, ty, w, R = pts[end]
            xa, ya = pts[j][0], pts[j][1]
            ux, uy = X0 - xa, Y0 - ya
            d = math.hypot(ux, uy) or 1.0
            ext = max(0.0, R - w / 2)
            pts[end] = (X0 + ux / d * ext, Y0 + uy / d * ext, tx, ty, w, R)
        quads = []
        for p, q in zip(pts, pts[1:]):
            (x0, y0, _, _, w0, _), (x1, y1, _, _, w1, _) = p, q
            dx, dy = x1 - x0, y1 - y0
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            nx, ny = -dy / d, dx / d
            quads.append(Polygon([(x0 + nx * w0 / 2, y0 + ny * w0 / 2), (x1 + nx * w1 / 2, y1 + ny * w1 / 2),
                                  (x1 - nx * w1 / 2, y1 - ny * w1 / 2), (x0 - nx * w0 / 2, y0 - ny * w0 / 2)]).buffer(0))
        # the joints between pieces, so that a turn does not leave a notch
        quads += [Point(x, y).buffer(w / 2, 12) for x, y, _, _, w, _ in pts[1:-1:3]]
        if quads:
            shapes.append(unary_union(quads))
    if not shapes:
        return None
    body = unary_union(shapes)
    if body.is_empty:
        return None
    rings = []
    for poly in getattr(body, "geoms", [body]):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        rings.append(list(poly.exterior.coords)[:-1])
        rings += [list(h.coords)[:-1] for h in poly.interiors]
    xs = [x for ring in rings for x, _ in ring]
    right, lft = max(xs), min(xs)
    rings = [[(x - right, y) for x, y in ring] for ring in rings]
    lft -= right
    if engine is None:
        engine = LETTERS[char][1](st, form)
    from .typeset import _is_mark, flat
    marks = [sh for sh in flat(engine.shapes) if _is_mark(sh, st)]
    if marks:
        # the marks keep their place over the body: centred on it as on the engine's letter
        exs = [x for sh in flat(engine.shapes) if not _is_mark(sh, st) for poly in sh for x, _ in poly]
        if exs:
            shift = 0.5 * (lft + 0.0) - 0.5 * (min(exs) + max(exs))
            marks = [Tagged([[(x + shift, y) for x, y in poly] for poly in sh], getattr(sh, "source", "engine"),
                            getattr(sh, "name", "")) for sh in marks]
    joins_right, joins_left = form in ("M", "F"), form in ("I", "M")
    return Glyph([Tagged(rings, "reference", "بنية الخط المرجعي")] + marks, left=lft,
                 entry_x=0.0 if joins_right else 0.0, entry_edge=0.0,
                 exit_x=lft if joins_left else None, exit_edge=lft if joins_left else None)


WIDTHS = (0.85, 1.0, 1.2)


def nearest(L, text: str, st: Style, k: float, widths=WIDTHS) -> Optional[Dict]:
    """The reference typeface whose structure, drawn with the logo's pen, redraws the logo's own word
    best: {"path", "wide", "match", "engine"} (the engine's match for comparison), or None when no
    typeface is at hand. Every letter of the word is built on the typeface for this judgement."""
    from .selfcheck import self_check
    paths = [f for f in faces() if f.endswith("-400.woff2")] or faces()
    if not paths or not text:
        return None
    base = self_check(L, text, st, k)
    if not base:
        return None
    best = None
    for path in paths:
        for w in widths:
            try:
                r = self_check(L, text, Style(**{**st.__dict__, "reference": path, "ref_wide": w, "ref_all": True}), k)
            except Exception:                       # a typeface that cannot shape the word is passed over
                continue
            if r and (best is None or r["overall"] > best["match"]):
                best = {"path": path, "wide": w, "match": r["overall"]}
    if best:
        best["engine"] = base["overall"]
    return best
