"""Outline geometry for drawn letters: exact Bezier paths (skia-pathops), cut along the design's slant.

Coordinates: font units, y up, the row's baseline at y = 0. A design's slant is `tan` (x = x0 + y*tan is a
"stem line"); cutting along stem lines keeps a slanted design's joins clean.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import pathops
from fontTools.pens.areaPen import AreaPen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen

BIG = 1e6
Path = pathops.Path


def from_contours(contours, fn) -> Path:
    """contours: [[("M", [(x, y)]), ("L", [(x, y)]), ("C", [c1, c2, p]), ...], ...]; fn maps a source point."""
    p = Path()
    pen = p.getPen()
    for c in contours:
        if not c:
            continue
        for op, pts in c:
            if op == "M":
                pen.moveTo(fn(*pts[0]))
            elif op == "L":
                pen.lineTo(fn(*pts[0]))
            else:
                pen.curveTo(*(fn(*q) for q in pts))
        pen.closePath()
    try:
        p.simplify(fix_winding=True)
    except pathops.PathOpsError:
        pass
    return p


def union(paths: Iterable[Path]) -> Path:
    acc = Path()
    for q in paths:
        acc = pathops.op(acc, q, pathops.PathOp.UNION, fix_winding=True)
    return acc


def intersect(a: Path, b: Path) -> Path:
    return pathops.op(a, b, pathops.PathOp.INTERSECTION, fix_winding=True)


def minus(a: Path, b: Path) -> Path:
    return pathops.op(a, b, pathops.PathOp.DIFFERENCE, fix_winding=True)


def poly(points) -> Path:
    p = Path()
    pen = p.getPen()
    pen.moveTo(points[0])
    for q in points[1:]:
        pen.lineTo(q)
    pen.closePath()
    return p


def rect(x0, y0, x1, y1) -> Path:
    return poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def slab(tan: float, xl: Optional[float] = None, xr: Optional[float] = None) -> Path:
    """Everything between two stem lines through (xl, 0) and (xr, 0)."""
    xl = -BIG if xl is None else xl
    xr = BIG if xr is None else xr
    h = 1e5
    return poly([(xl - h * tan, -h), (xr - h * tan, -h), (xr + h * tan, h), (xl + h * tan, h)])


def clip(p: Path, tan: float, xl=None, xr=None) -> Path:
    return intersect(p, slab(tan, xl, xr))


def band(x0, x1, t, tan) -> Path:
    """A connector between x0 and x1 on the baseline, t thick, its ends along the stem lines."""
    return poly([(x0, 0), (x1, 0), (x1 + t * tan, t), (x0 + t * tan, t)])


def transformed(p: Path, m) -> Path:
    out = Path()
    p.draw(TransformPen(out.getPen(), m))
    return out


def shift(p: Path, dx, dy=0.0) -> Path:
    return transformed(p, (1, 0, 0, 1, dx, dy))


def deslant(p: Path, tan: float) -> Path:
    """x' = x - y*tan: stems become upright."""
    return transformed(p, (1, 0, -tan, 1, 0, 0))


def bounds(p: Path) -> Optional[Tuple[float, float, float, float]]:
    b = p.bounds
    if b is None or (b[2] <= b[0] and b[3] <= b[1]):
        return None
    return b


def contours(p: Path) -> List[list]:
    rec = RecordingPen()
    p.draw(rec)
    out, cur = [], []
    for op, args in rec.value:
        cur.append((op, args))
        if op in ("closePath", "endPath"):
            out.append(cur)
            cur = []
    return out


def contour_info(c):
    ap, bp = AreaPen(), BoundsPen(None)
    for op, args in c:
        getattr(ap, op)(*args)
        getattr(bp, op)(*args)
    return ap.value, bp.bounds


def from_recorded(cs) -> Path:
    p = Path()
    pen = p.getPen()
    for c in cs:
        for op, args in c:
            getattr(pen, op)(*args)
    return p


def runs_at(p: Path, y: float) -> List[Tuple[float, float]]:
    """The separate ink intervals in a thin strip at height y."""
    r = intersect(p, rect(-BIG, y - 1, BIG, y + 1))
    out = []
    for c in contours(r):
        _, b = contour_info(c)
        if b:
            out.append((b[0], b[2]))
    return sorted(out)


def run_at(p: Path, y: float):
    rs = runs_at(p, y)
    return (rs[0][0], rs[-1][1]) if rs else None


def close_gaps(p: Path, w: float = 3.0) -> Path:
    """Morphological closing sideways (dilate by w, erode back): seals the hairline gaps and notches
    where two drawn pieces only touch, without moving the outline elsewhere."""
    try:
        d = union([p, shift(p, w), shift(p, -w)])
        out = intersect(d, shift(d, w))
        return intersect(out, shift(d, -w))
    except pathops.PathOpsError:
        try:
            return union([p, shift(p, 0.6), shift(p, -0.6)])
        except pathops.PathOpsError:
            return p


def clean(p: Path, min_area: float = 150.0, min_side: float = 4.0) -> Tuple[Path, int]:
    """Drop slivers and specks (contours too small or too thin to be drawn on purpose)."""
    keep, dropped = [], 0
    for c in contours(p):
        a, b = contour_info(c)
        if b is None or abs(a) < min_area or min(b[2] - b[0], b[3] - b[1]) < min_side:
            dropped += 1
            continue
        keep.append(c)
    out = from_recorded(keep)
    try:
        out.simplify(fix_winding=True, clockwise=False)
    except pathops.PathOpsError:
        pass
    return out, dropped


def touches(a: Path, b: Path, tol: float = 0.5) -> bool:
    ba, bb = bounds(a), bounds(b)
    if ba is None or bb is None:
        return False
    if ba[0] - tol > bb[2] or bb[0] - tol > ba[2] or ba[1] - tol > bb[3] or bb[1] - tol > ba[3]:
        return False
    if bounds(intersect(a, b)) is not None:
        return True
    return bounds(intersect(union([a, shift(a, tol), shift(a, -tol), shift(a, 0, tol), shift(a, 0, -tol)]), b)) is not None
