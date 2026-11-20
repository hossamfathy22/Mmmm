"""A vector logo taken apart: its letters, and each letter's strokes with their own thickness.

The thickness is not a pen's: it is read point by point from the designer's shapes (how far the centre
line is from the edges), so a stroke drawn from this is the designer's stroke, whether it was drawn with
a pen, built from rectangles and curves, or traced. A stroke is a centre line and its width at every
point; it can be lengthened, shortened, bent, and drawn again with the same widths, which is what the
parts of the six (twelve) families are.

  letters(shapes, text)   each letter's ink: exact where the shapes know their letter (a typeface), else
                          found by matching the typed word (match.match) on a picture of the shapes
  strokes(mask, frame)    the strokes of a letter: [[(x, y, width)]] in the logo's units, y up
  draw(strokes)           the strokes drawn again (a shapely geometry): their widths at every point, the
                          ends cut square where the stroke ends
  rebuild(shapes, text)   the whole logo drawn again from its strokes, and how well it covers itself
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .vector import VShape, dna, raster

Stroke = List[Tuple[float, float, float]]


def letters(shapes: Sequence[VShape], text: str, px_x: float = 160.0, faces=None):
    """{index of the letter in text: its mask}, the mask's frame (x0, y1, scale), and the whole mask. The
    picture is drawn so that the x-height is about `px_x` pixels."""
    d = dna(shapes, text)
    scale = px_x / max(1e-6, d.get("x_height") or 0.3 * d["height"])
    whole, frame = raster(shapes, scale, pad=12)
    out: Dict[int, np.ndarray] = {}
    if all(s.index is not None for s in shapes):
        for i in sorted({s.index for s in shapes}):
            m, _ = raster([s for s in shapes if s.index == i], scale, pad=12, box=_box(shapes))
            out[i] = m[:whole.shape[0], :whole.shape[1]]
        return out, frame, whole
    from .match import match
    if faces is None:
        from .own import _faces
        faces = _faces()
    got = match(whole, text, faces)
    if got is not None:
        for i, c in enumerate(text):
            if not c.isspace():
                m = whole & (got["owner"] == i + 1)
                if m.any():
                    out[i] = m
    return out, frame, whole


def _box(shapes):
    pts = np.vstack([s.points() for s in shapes if len(s.points())])
    return (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())


def strokes(mask: np.ndarray, frame) -> List[Stroke]:
    """The strokes of a mask as centre lines with the full width at every point, in the logo's units."""
    from .reference import _mask_strokes
    x0, y1, sc = frame
    out = []
    for st in _mask_strokes(mask):
        out.append([(x0 + c / sc, y1 - r / sc, 2.0 * rad / sc) for r, c, rad in st])
    return out


def draw(sts: Sequence[Stroke], cap: bool = True):
    """Strokes drawn with their own widths: a quadrilateral between every two points, a disc at every
    inner point (the turns), and the ends carried out to where the stroke's edge ends (the centre line of
    a shape stops half a width short of its end), cut square."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    parts = []
    for st in sts:
        if len(st) < 2:
            continue
        pts = [list(p) for p in st]
        if cap:
            for end, nxt in ((0, 1), (len(pts) - 1, len(pts) - 2)):
                x, y, w = pts[end]
                dx, dy = x - pts[nxt][0], y - pts[nxt][1]
                d = math.hypot(dx, dy) or 1.0
                pts[end] = [x + dx / d * w / 2, y + dy / d * w / 2, w]
        for (xa, ya, wa), (xb, yb, wb) in zip(pts, pts[1:]):
            dx, dy = xb - xa, yb - ya
            d = math.hypot(dx, dy)
            if d < 1e-9:
                continue
            nx, ny = -dy / d, dx / d
            parts.append(Polygon([(xa + nx * wa / 2, ya + ny * wa / 2), (xb + nx * wb / 2, yb + ny * wb / 2),
                                  (xb - nx * wb / 2, yb - ny * wb / 2), (xa - nx * wa / 2, ya - ny * wa / 2)]).buffer(0))
        parts += [Point(x, y).buffer(w / 2, 16) for x, y, w in pts[1:-1]]
    return unary_union(parts) if parts else None


def rebuild(shapes: Sequence[VShape], text: str, px_x: float = 160.0) -> Dict:
    """The logo taken apart into letters and strokes and drawn again from them: the strokes, and how well
    the drawing covers the logo (IoU), letter by letter and whole."""
    from .own import rasterise as _ras
    lm, frame, whole = letters(shapes, text, px_x)
    x0, y1, sc = frame
    res = {"letters": {}, "strokes": {}}
    drawn = np.zeros_like(whole)
    for i, m in lm.items():
        sts = strokes(m, frame)
        g = draw(sts)
        if g is None:
            continue
        # back to the picture's frame
        from shapely import affinity
        gp = affinity.affine_transform(g, [sc, 0, 0, -sc, -x0 * sc, y1 * sc])
        r = _mask_from(gp, m.shape)
        res["letters"][i] = float((r & m).sum() / max(1, (r | m).sum()))
        res["strokes"][i] = sts
        drawn |= r
    res["whole"] = float((drawn & whole).sum() / max(1, (drawn | whole).sum()))
    res["frame"] = frame
    return res


def _mask_from(g, shape) -> np.ndarray:
    """A shapely geometry in pixel coordinates (x right, y down) as a mask."""
    from PIL import Image, ImageDraw
    H, W = shape
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for poly in getattr(g, "geoms", [g]):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        d.polygon(list(poly.exterior.coords), fill=1)
        for h in poly.interiors:
            d.polygon(list(h.coords), fill=0)
    return np.asarray(im) > 0
