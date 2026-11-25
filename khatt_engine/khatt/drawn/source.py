"""Read a designer's drawing as vectors: every filled path on a PDF page (as exported from Illustrator),
with its exact curves. Nothing is traced: a curve the designer drew stays that curve."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Piece:
    i: int
    fill: Optional[str]
    contours: list                     # [[("M", [(x, y)]), ("L", ...), ("C", [c1, c2, p])], ...] in points, y down
    rect: Tuple[float, float, float, float]


def _hex(c) -> Optional[str]:
    if c is None:
        return None
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3])


def read_pdf(path: str, page: int = 0) -> List[Piece]:
    import pymupdf

    doc = pymupdf.open(path)
    pg = doc[page]
    out: List[Piece] = []
    for i, d in enumerate(pg.get_drawings()):
        if d.get("fill") is None:                       # outlines with no fill are guides, not letters
            continue
        contours, cur, last = [], [], None

        def start(p):
            nonlocal cur
            if cur:
                contours.append(cur)
            cur = [("M", [(p.x, p.y)])]

        for it in d["items"]:
            kind = it[0]
            if kind == "l":
                p0, p1 = it[1], it[2]
                if last is None or abs(p0.x - last[0]) > 1e-3 or abs(p0.y - last[1]) > 1e-3:
                    start(p0)
                cur.append(("L", [(p1.x, p1.y)]))
                last = (p1.x, p1.y)
            elif kind == "c":
                p0, c1, c2, p3 = it[1], it[2], it[3], it[4]
                if last is None or abs(p0.x - last[0]) > 1e-3 or abs(p0.y - last[1]) > 1e-3:
                    start(p0)
                cur.append(("C", [(c1.x, c1.y), (c2.x, c2.y), (p3.x, p3.y)]))
                last = (p3.x, p3.y)
            elif kind == "re":
                r = it[1]
                if cur:
                    contours.append(cur)
                    cur = []
                contours.append([("M", [(r.x0, r.y0)]), ("L", [(r.x1, r.y0)]), ("L", [(r.x1, r.y1)]), ("L", [(r.x0, r.y1)])])
                last = None
            elif kind == "qu":
                q = it[1]
                if cur:
                    contours.append(cur)
                    cur = []
                contours.append([("M", [(q.ul.x, q.ul.y)]), ("L", [(q.ur.x, q.ur.y)]), ("L", [(q.lr.x, q.lr.y)]),
                                 ("L", [(q.ll.x, q.ll.y)])])
                last = None
        if cur:
            contours.append(cur)
        r = d["rect"]
        out.append(Piece(i, _hex(d.get("fill")), contours, (r.x0, r.y0, r.x1, r.y1)))
    return out


def straight_edges(pieces: List[Piece], min_len: float) -> List[Tuple[float, float]]:
    """(angle from vertical in degrees, length) of every straight edge at least min_len long."""
    import math

    out = []
    for pc in pieces:
        for c in pc.contours:
            prev = None
            for op, pts in c:
                p = pts[-1]
                if op == "L" and prev is not None:
                    dx, dy = p[0] - prev[0], p[1] - prev[1]
                    ln = math.hypot(dx, dy)
                    if ln >= min_len and abs(dy) > abs(dx):
                        # y grows downward: a stem leaning right at the top has dx > 0 when dy < 0
                        out.append((math.degrees(math.atan2(-dx if dy > 0 else dx, abs(dy))), ln))
                prev = p
    return out
