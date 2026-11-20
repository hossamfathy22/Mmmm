"""A vector logo taken apart into letters and parts, and parts changed as a designer changes them.

Two ways, the second the better:

1. Strokes (`strokes`, `draw`, `rebuild`): each letter as centre lines with the width of the designer's
   shape at every point, drawn again from them. No pen is assumed, but the drawing is a new outline: it
   covers the word 0.85-0.93, its sharp corners and cut ends come out round, and it is not the designer's
   curves.

2. The designer's own outline (`parts_of`, `cut_parts`, `stretch_letter`, `to_shapes`): the letter's
   outline is cut at its joints (square to the centre line, just outside where strokes meet) into parts
   that tile it exactly; a letter is made wider or taller as in Illustrator: a line across it where it
   crosses only strokes running across it (never along a stem), one side dragged, every crossing closed by
   a straight bridge. The ends, the corners and the curves stay exactly as drawn; a closed loop is pulled
   from both its sides; the marks (dots, dashes) are never cut and go with the side they stand over. The
   result goes back to a path of lines and Bezier curves (`ring_to_segments`: even steps, corners by how
   sharply the outline turns, Beziers between them, straight ones made lines), with about as many
   segments as the typeface's own, for an SVG the designer opens in Illustrator.
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


# ------------------------------------------------------------------ parts on a spine (the better way)
# A part is a piece of the designer's own outline, cut at the joints, with its centre line as a spine.
# The spine only guides changes: the outline is the designer's. To lengthen a part, the outline is cut
# where the spine runs straightest and the far side moved along it, as a designer drags a selection of
# anchor points in Illustrator (Direct Selection): the ends, the corners and the curves stay as drawn,
# only the straight middle grows. To make it thicker, every point moves away from the spine.
from dataclasses import dataclass, field as _field


@dataclass
class Part:
    index: int                      # the letter (index in the typed text)
    kind: str                       # "stroke", "joint" (where strokes meet) or "dot"
    geom: object                    # a shapely polygon (or several), in the logo's units
    spine: Optional[np.ndarray] = None   # (m, 2) centre line, in order
    width: Optional[np.ndarray] = None   # the width of the shape along the spine
    ends: Tuple[str, str] = ("free", "free")


def letter_geometry(shapes: Sequence[VShape], index: int, tol: float):
    from shapely.ops import unary_union
    from .vector import geometry
    gs = [geometry(s, tol) for s in shapes if s.index == index]
    gs = [g for g in gs if g is not None and not g.is_empty]
    return unary_union(gs) if gs else None


def _spine_coords(spine: np.ndarray, pts: np.ndarray):
    """For every point: its distance along the spine (the ends extended straight) and its signed distance
    from it (left positive)."""
    A, B = spine[:-1], spine[1:]
    d = B - A
    L = np.hypot(d[:, 0], d[:, 1]); L[L == 0] = 1e-9
    u = d / L[:, None]
    cum = np.r_[0.0, np.cumsum(L)]
    s_out = np.zeros(len(pts)); n_out = np.zeros(len(pts))
    for k, P in enumerate(pts):
        w = P - A
        t = np.einsum("ij,ij->i", w, u)
        tt = np.clip(t, 0, L)
        tt[0] = min(t[0], L[0])                       # before the first point: straight on
        tt[-1] = max(t[-1], 0.0)                      # after the last: straight on
        Q = A + u * tt[:, None]
        dist = np.hypot(*(P - Q).T)
        j = int(np.argmin(dist))
        s_out[k] = cum[j] + tt[j]
        n_out[k] = u[j, 0] * w[j, 1] - u[j, 1] * w[j, 0]
    return s_out, n_out, cum


def _at(spine: np.ndarray, s: float):
    """The point at distance s along the spine (straight on past its ends) and the direction there."""
    d = np.diff(spine, axis=0)
    L = np.hypot(d[:, 0], d[:, 1]); L[L == 0] = 1e-9
    cum = np.r_[0.0, np.cumsum(L)]
    j = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(L) - 1))
    u = d[j] / L[j]
    return spine[j] + u * (s - cum[j]), u


def cut_parts(geom, sts: List[Stroke], index: int, cut_margin: float = 1.05) -> List[Part]:
    """The letter's outline cut into parts: where strokes meet, each stroke is cut square to its spine
    just outside the largest circle that fits in the meeting (so the meeting is a part of its own, a
    joint); the cut runs across that stroke only. The pieces tile the letter exactly."""
    from shapely.geometry import LineString, Point
    from shapely.ops import split, unary_union
    if geom is None or geom.is_empty:
        return []
    polys = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon"]
    # dots: pieces of the letter with no stroke of length through them
    ends = []
    for k, st in enumerate(sts):
        ends.append((k, 0, np.array(st[0][:2])))
        ends.append((k, len(st) - 1, np.array(st[-1][:2])))
    cuts = []
    for k, i, P in ends:
        others = [(k2, i2) for k2, i2, Q in ends if (k2, i2) != (k, i) and np.hypot(*(P - Q)) < 0.6 * max(1e-6, sts[k][i][2])]
        if not others:
            continue                                  # a free end: no cut
        rJ = 0.5 * max(sts[k][i][2], *[sts[k2][i2][2] for k2, i2 in others])
        st = np.array([p[:2] for p in sts[k]])
        if i != 0:
            st = st[::-1]
        seg = np.hypot(*np.diff(st, axis=0).T)
        cum = np.r_[0.0, np.cumsum(seg)]
        want = cut_margin * rJ
        if cum[-1] <= want * 1.2:
            continue                                  # a stroke all inside the meeting
        j = int(np.searchsorted(cum, want))
        j = min(max(j, 1), len(st) - 1)
        P0 = st[j]
        u = st[min(j + 1, len(st) - 1)] - st[j - 1]
        u = u / (np.hypot(*u) or 1.0)
        nrm = np.array([-u[1], u[0]])
        half = 1.6 * max(p[2] for p in sts[k]) + 1.0
        line = LineString([tuple(P0 - nrm * half), tuple(P0 + nrm * half)])
        piece = geom.intersection(line)
        segs = [q for q in getattr(piece, "geoms", [piece]) if q.geom_type == "LineString" and not q.is_empty]
        if not segs:
            continue
        best = min(segs, key=lambda q: q.distance(Point(*P0)))
        (ax, ay), (bx, by) = best.coords[0], best.coords[-1]
        e = np.array([bx - ax, by - ay]); e = e / (np.hypot(*e) or 1.0) * (0.002 * half + 1e-3)
        cuts.append(LineString([(ax - e[0], ay - e[1]), (bx + e[0], by + e[1])]))
    pieces = polys
    for c in cuts:
        nxt = []
        for p in pieces:
            try:
                got = split(p, c)
                nxt += [g for g in got.geoms if g.area > 0]
            except Exception:
                nxt.append(p)
        pieces = nxt
    spines = [LineString([p[:2] for p in st]) for st in sts if len(st) >= 2]
    out = []
    for p in pieces:
        inside = [(p.intersection(sp).length, k) for k, sp in enumerate(spines)]
        L, k = max(inside) if inside else (0.0, -1)
        if k >= 0 and L > 0.25 * max(1e-6, math.sqrt(p.area)):
            st = np.array(sts[k])
            # the spine of this piece: the stroke's centre line inside it, in order
            keep = [i for i, q in enumerate(st) if p.buffer(1e-6).contains(Point(q[0], q[1]))]
            if len(keep) >= 2:
                sp = st[keep[0]:keep[-1] + 1]
                out.append(Part(index, "stroke", p, sp[:, :2], sp[:, 2]))
                continue
        out.append(Part(index, "joint", p))
    # a piece that touches no other piece and no stroke runs through is a mark (a dot, a dash)
    for q in out:
        if q.kind == "joint" and not any(sp.intersects(q.geom) for sp in spines) and \
                not any(r is not q and r.geom.distance(q.geom) < 1e-6 for r in out):
            q.kind = "dot"
    return out


def _map_coords(g, fn):
    from shapely.geometry import Polygon, MultiPolygon
    def one(poly):
        ext = fn(np.array(poly.exterior.coords))
        ints = [fn(np.array(r.coords)) for r in poly.interiors]
        return Polygon(ext, ints).buffer(0)
    if g.geom_type == "Polygon":
        return one(g)
    return MultiPolygon([one(p) for p in g.geoms]).buffer(0)


def straightest(part: Part, keep_ends: float = 0.2, direction: Optional[Tuple[float, float]] = None,
                within: float = 35.0) -> Optional[float]:
    """Where along the spine it runs straightest, away from its ends (where a part is lengthened); with
    `direction`, only where the spine runs within `within` degrees of it (to widen a bowl, where it runs
    across; to raise a stem, where it runs up). None where it never does."""
    sp = part.spine
    d = np.diff(sp, axis=0)
    seg = np.hypot(*d.T)
    cum = np.r_[0.0, np.cumsum(seg)]
    L = cum[-1]
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    bend = np.abs(np.gradient(ang)) if len(ang) > 2 else np.zeros(len(ang))
    ok = []
    w = part.width if part.width is not None and len(part.width) == len(sp) else None
    wm = float(np.median(w)) if w is not None else 0.0
    margin = max(keep_ends * L, wm)
    for i in range(len(bend)):
        if not (margin <= cum[i] <= L - margin) or seg[i] == 0:
            continue
        if direction is not None:
            u = d[i] / seg[i]
            c = abs(u[0] * direction[0] + u[1] * direction[1]) / (math.hypot(*direction) or 1.0)
            if c < math.cos(math.radians(within)):
                continue
        ok.append((bend[i], i))
    # the cut there must cross the stroke and nothing more: as long as the stroke is wide (a cut longer
    # than that runs down a stem or through a corner, and the bridge would be a block)
    for _, i in sorted(ok):
        at = float(0.5 * (cum[i] + cum[i + 1]))
        cut, _, _ = _cut_at(part, at)
        if cut is None:
            continue
        here = 0.5 * (w[i] + w[i + 1]) if w is not None else wm
        widest = float(np.max(w)) if w is not None else wm
        # a cut running along another stroke (down a stem from a corner) is much longer than this stroke is
        # wide anywhere; a wedge-shaped stroke's cut differs from the skeleton's width, but not by that much
        if here <= 0 or cut.length <= max(1.5 * here, 1.2 * widest):
            return at
    return None


def _cut_at(part: Part, at: float):
    """The cut square across the part at `at` along its spine: (segment, point, direction)."""
    from shapely.geometry import LineString, Point
    P0, u = _at(part.spine, at)
    nrm = np.array([-u[1], u[0]])
    half = 2.0 * (float(np.max(part.width)) if part.width is not None else math.sqrt(part.geom.area)) + 1.0
    line = LineString([tuple(P0 - nrm * half), tuple(P0 + nrm * half)])
    got = part.geom.intersection(line)
    segs = [q for q in getattr(got, "geoms", [got]) if q.geom_type == "LineString" and not q.is_empty]
    if not segs:
        return None, P0, u
    best = min(segs, key=lambda q: q.distance(Point(*P0)))
    (ax, ay), (bx, by) = best.coords[0], best.coords[-1]
    e = np.array([bx - ax, by - ay]); e = e / (np.hypot(*e) or 1.0) * 1e-3 * half
    return LineString([(ax - e[0], ay - e[1]), (bx + e[0], by + e[1])]), P0, u


def lengthen(part: Part, delta: float, at: Optional[float] = None, move_far: bool = True) -> Part:
    """The part `delta` longer, as in Illustrator: cut square across it where it runs straightest, the far
    side dragged along the spine, the gap closed by the straight bridge between the two edges of the cut.
    The ends, the corners and the curves stay exactly as drawn."""
    from shapely.geometry import Polygon
    from shapely.ops import split, unary_union
    at = straightest(part) if at is None else at
    if at is None:
        return part
    cut, P0, u = _cut_at(part, at)
    if cut is None:
        return part
    D = u * delta if move_far else -u * delta
    pieces = list(split(part.geom, cut).geoms)
    moved = []
    for q in pieces:
        c = np.array(q.representative_point().coords[0])
        s, _, _ = _spine_coords(part.spine, c[None, :])
        moved.append(_translate(q, D) if (s[0] > at) == move_far else q)
    (ax, ay), (bx, by) = cut.coords[0], cut.coords[-1]
    bridge = Polygon([(ax, ay), (bx, by), (bx + D[0], by + D[1]), (ax + D[0], ay + D[1])]).buffer(0)
    s_, _, _ = _spine_coords(part.spine, part.spine)
    sp = part.spine + ((s_ > at) == move_far)[:, None] * D[None, :]
    # the seams of the cut closed (a hair's breadth in and out again)
    eps = 1e-4 * math.sqrt(max(part.geom.area, 1e-9))
    g = unary_union(moved + [bridge]).buffer(eps, join_style=2).buffer(-eps, join_style=2)
    return Part(part.index, part.kind, g, sp, part.width, part.ends)


def thicken(geom, d: float):
    """Thicker by d on every side, its corners kept sharp: Illustrator's Offset Path with mitre joins."""
    return geom.buffer(d, join_style=2, mitre_limit=4.0)


def parts_of(shapes: Sequence[VShape], text: str, px_x: float = 240.0) -> Tuple[List[Part], Dict]:
    """Every letter of the logo cut into parts on spines. Returns the parts and, per letter, how well
    the parts put back together cover it (IoU: 1.0 when they tile it, as they must)."""
    from shapely.ops import unary_union
    lm, frame, whole = letters(shapes, text, px_x)
    H = dna(shapes, text)["height"]
    tol = 0.0005 * H
    out, cover = [], {}
    for i, m in lm.items():
        g = letter_geometry(shapes, i, tol)
        if g is None:
            continue
        ps = cut_parts(g, strokes(m, frame), i)
        out += ps
        u = unary_union([p.geom for p in ps]) if ps else None
        cover[i] = float(u.intersection(g).area / max(1e-9, u.union(g).area)) if u is not None else 0.0
    return out, cover


def _translate(g, D):
    from shapely import affinity
    return affinity.translate(g, float(D[0]), float(D[1]))


def lengthen_in_letter(parts: List[Part], k: int, delta: float, at: Optional[float] = None,
                       direction: Optional[Tuple[float, float]] = None) -> Optional[List[Part]]:
    """A letter's parts with part k made `delta` longer: the part grows in its straight middle, and every
    part attached beyond its far side moves with it, so the letter stays whole (None where the part is
    in a closed loop of the letter, which cannot grow without breaking it)."""
    p = parts[k]
    if p.kind != "stroke" or p.spine is None or len(p.spine) < 3:
        return None
    at = straightest(p, direction=direction) if at is None else at
    if at is None:
        return None
    _, u = _at(p.spine, at)
    D = u * delta                                     # the far side moves on along the spine: the part grows
    eps = 1e-6 * max(1.0, math.sqrt(p.geom.area))
    touch = {i: [j for j in range(len(parts)) if j != i and parts[i].geom.distance(parts[j].geom) < eps]
             for i in range(len(parts))}
    # the side of each neighbour of p: beyond `at` along its spine, or before it
    far, near = set(), set()
    for j in touch[k]:
        c = np.array(parts[j].geom.representative_point().coords[0])
        s, _, _ = _spine_coords(p.spine, c[None, :])
        (far if s[0] > at else near).add(j)
    # everything reachable from the far neighbours without passing through p
    seen, stack = set(far), list(far)
    while stack:
        i = stack.pop()
        for j in touch[i]:
            if j != k and j not in seen:
                seen.add(j); stack.append(j)
    if seen & near:
        return None
    rest = set(range(len(parts))) - seen - {k}
    # the smaller side moves: a stem grows up from where it stands, a bowl widens at its free tip
    from shapely.ops import split
    cut, _, _ = _cut_at(p, at)
    if cut is None:
        return None
    halves = list(split(p.geom, cut).geoms)
    def own_side(far_side):
        return sum(q.area for q in halves
                   if (_spine_coords(p.spine, np.array(q.representative_point().coords[0])[None, :])[0][0] > at) == far_side)
    a_far = own_side(True) + sum(parts[i].geom.area for i in seen)
    a_near = own_side(False) + sum(parts[i].geom.area for i in rest)
    move_far = a_far <= a_near
    D = D if move_far else -D
    moving = seen if move_far else rest
    out = []
    for i, q in enumerate(parts):
        if i == k:
            out.append(lengthen(q, delta, at, move_far))
        elif i in moving:
            out.append(Part(q.index, q.kind, _translate(q.geom, D), None if q.spine is None else q.spine + D, q.width, q.ends))
        else:
            out.append(q)
    return out


def best_stroke(parts: List[Part], direction: Tuple[float, float], within: float = 35.0) -> List[int]:
    """The strokes in order of how far they run the way asked (the one to grow first)."""
    score = []
    for i, p in enumerate(parts):
        if p.kind != "stroke" or p.spine is None or len(p.spine) < 3:
            continue
        d = np.diff(p.spine, axis=0)
        seg = np.hypot(*d.T); seg[seg == 0] = 1e-9
        c = np.abs(d @ np.array(direction)) / (seg * (math.hypot(*direction) or 1.0))
        run = float(seg[c >= math.cos(math.radians(within))].sum())
        if run > 0:
            score.append((run, i))
    return [i for _, i in sorted(score, reverse=True)]


def ring_to_segments(pts, err: float):
    """A closed outline as a designer's path, as vectorising programs make one: the outline resampled at
    even steps, its corners found by how sharply it turns over a few steps, cubic Beziers fitted between
    the corners (as few as keep within `err`), and every cubic whose handles lie on its chord made a
    straight line."""
    from .bezier import _fit, _unit
    p = np.asarray(pts, float)
    if len(p) > 1 and np.hypot(*(p[0] - p[-1])) < 1e-9:
        p = p[:-1]
    if len(p) < 3:
        return []
    # even steps along the outline
    ring = np.vstack([p, p[:1]])
    d = np.hypot(*np.diff(ring, axis=0).T)
    cum = np.r_[0.0, np.cumsum(d)]
    step = max(err, cum[-1] / 2000.0)
    t = np.arange(0.0, cum[-1], step)
    q = np.c_[np.interp(t, cum, ring[:, 0]), np.interp(t, cum, ring[:, 1])]
    n = len(q)
    if n < 8:
        return [("L", tuple(a), tuple(b)) for a, b in zip(p, np.roll(p, -1, axis=0))]
    # the turn over k steps either side; a corner is the sharpest point of a sharp turn
    k = max(2, int(round(2.5 * err / step)))
    a = q - np.roll(q, k, axis=0); b = np.roll(q, -k, axis=0) - q
    cosv = (a * b).sum(1) / (np.hypot(*a.T) * np.hypot(*b.T) + 1e-12)
    turn = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    cs = [i for i in range(n) if turn[i] >= 35 and turn[i] == max(turn[(i + j) % n] for j in range(-k, k + 1))]
    cs = [c for i, c in enumerate(cs) if i == 0 or c - cs[i - 1] > k]
    if len(cs) < 2:
        cs = sorted({int(np.argmin(q[:, 0])), int(np.argmax(q[:, 0]))})
    segs = []
    for j, c0 in enumerate(cs):
        c1 = cs[(j + 1) % len(cs)]
        run = q[c0:c1 + 1] if c1 > c0 else np.vstack([q[c0:], q[:c1 + 1]])
        if len(run) < 2:
            continue
        ch = run[-1] - run[0]; cl = float(np.hypot(*ch))
        if cl > 0:
            dev = np.abs((run[:, 0] - run[0, 0]) * ch[1] - (run[:, 1] - run[0, 1]) * ch[0]) / cl
            if dev.max() < err:
                segs.append(("L", tuple(run[0]), tuple(run[-1]))); continue
        if len(run) == 2:
            segs.append(("L", tuple(run[0]), tuple(run[1]))); continue
        t1 = _unit(run[min(2, len(run) - 1)] - run[0]); t2 = _unit(run[max(-3, -len(run))] - run[-1])
        for bz in _fit(run, t1, t2, err):
            p0, c1_, c2_, p1 = bz
            ch = p1 - p0; cl = float(np.hypot(*ch))
            if cl > 0 and max(abs((c1_ - p0)[0] * ch[1] - (c1_ - p0)[1] * ch[0]),
                              abs((c2_ - p0)[0] * ch[1] - (c2_ - p0)[1] * ch[0])) / cl < 0.5 * err:
                segs.append(("L", tuple(p0), tuple(p1)))
            else:
                segs.append(("C",) + tuple(tuple(map(float, v)) for v in bz))
    return segs


def to_shapes(parts: Sequence[Part], err: Optional[float] = None) -> List[VShape]:
    """Parts back to shapes of lines and Bezier curves, one per part, for an SVG a designer opens in
    Illustrator: straight edges stay lines, corners stay corners."""
    out = []
    for p in parts:
        g = p.geom
        e = err if err is not None else max(0.05, 0.004 * math.sqrt(max(g.area, 1e-9)))
        contours = []
        for poly in getattr(g, "geoms", [g]):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            for ring in [poly.exterior] + list(poly.interiors):
                segs = ring_to_segments(list(ring.coords), e)
                if segs:
                    contours.append(segs)
        if contours:
            out.append(VShape(contours, f"{p.kind}-{p.index}", p.index))
    return out


# ------------------------------------------------------------------ stretching a letter, the Illustrator way
def body_and_marks(geom):
    """A letter's body and its marks (dots, dashes, the hamza): every piece but the largest that is small
    and compact (a dot: up to a third of the body, filling most of its box), or small (a dash, a stroke
    of a mark: under an eighth of it). A thin naskh body can be only three times its dot."""
    from shapely.ops import unary_union
    polys = [p for p in getattr(geom, "geoms", [geom]) if p.geom_type == "Polygon" and not p.is_empty]
    if not polys:
        return geom, []
    main = max(polys, key=lambda p: p.area)
    body, marks = [main], []
    for p in polys:
        if p is main:
            continue
        x0, y0, x1, y1 = p.bounds
        w, h = x1 - x0, y1 - y0
        compact = 0.4 <= (w / h if h else 9) <= 2.5 and p.area >= 0.45 * w * h
        (marks if (compact and p.area < 0.35 * main.area) or p.area < 0.125 * main.area else body).append(p)
    return unary_union(body), marks


def stretch_letter(geom, axis: str, delta: float, x_height: float, at: Optional[float] = None, keep: str = "auto"):
    """A letter made wider (axis "x") or taller (axis "y") by `delta`, as a designer does it: a line drawn
    across the whole letter where it crosses only strokes running across it (never along a stem or a bar),
    everything on one side of it dragged, every crossing bridged straight. A closed loop is pulled from
    both of its sides at once. Returns (geometry, the line's position) or (None, None) where no such line
    exists. `keep`: which side stays put ("low" keeps the left or the bottom, "high" the other, "auto" the
    larger, so a free tip moves and the body stays)."""
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import split, unary_union
    geom, marks = body_and_marks(geom)
    x0, y0, x1, y1 = geom.bounds
    lo, hi = (x0, x1) if axis == "x" else (y0, y1)
    span = hi - lo
    limit = 0.45 * x_height                     # a crossing longer than this runs along a stroke

    def line_at(v):
        far = 10 * max(x1 - x0, y1 - y0) + 1
        return LineString([(v, y0 - far), (v, y1 + far)]) if axis == "x" else LineString([(x0 - far, v), (x1 + far, v)])

    def crossings(v):
        got = geom.intersection(line_at(v))
        return [q for q in getattr(got, "geoms", [got]) if q.geom_type == "LineString" and q.length > 0]

    cands = [at] if at is not None else list(np.linspace(lo + 0.15 * span, hi - 0.15 * span, 61))
    good = []
    for v in cands:
        cs = crossings(v)
        if cs and all(c.length <= limit for c in cs):
            good.append((sum(c.length for c in cs), abs(v - 0.5 * (lo + hi)), v))
    if not good:
        return None, None
    # the thinnest crossing, nearest the middle among equals: where the letter is easiest to pull
    best = min(good, key=lambda g: (round(g[0] / (0.05 * x_height)), g[1]))
    v = best[2]
    cs = crossings(v)
    pieces = []
    for poly in getattr(geom, "geoms", [geom]):
        try:
            pieces += list(split(poly, line_at(v)).geoms)
        except Exception:
            pieces.append(poly)
    low = [p for p in pieces if (p.centroid.x if axis == "x" else p.centroid.y) < v]
    high = [p for p in pieces if (p.centroid.x if axis == "x" else p.centroid.y) >= v]
    if keep == "auto":
        keep = "low" if sum(p.area for p in low) >= sum(p.area for p in high) else "high"
    D = (delta, 0.0) if axis == "x" else (0.0, delta)
    if keep == "high":
        D = (-D[0], -D[1])
    moving, staying = (high, low) if keep == "low" else (low, high)
    if axis == "y":
        # a letter is made taller by its tallest stem alone (a designer selects the top of the stem, not
        # the free tip of a bowl the same line happens to cross)
        top = max(moving, key=lambda p: p.bounds[3] if keep == "low" else -p.bounds[1]) if moving else None
        if top is not None:
            cs = [c for c in cs if c.distance(top) < 1e-6 * max(1.0, x_height)]
            staying = staying + [p for p in moving if p is not top]
            moving = [top]
    moved = [_translate(p, np.array(D)) for p in moving]
    # the marks go with the side they stand over
    side = lambda q: (q.centroid.x if axis == "x" else q.centroid.y) >= v
    for q in marks:
        if (side(q) and keep == "low") or (not side(q) and keep == "high"):
            if axis == "x" or (moving and q.distance(moving[0]) < 0.6 * x_height):
                moved.append(_translate(q, np.array(D))); continue
        staying.append(q)
    bridges = []
    for c in cs:
        (ax, ay), (bx, by) = c.coords[0], c.coords[-1]
        bridges.append(Polygon([(ax, ay), (bx, by), (bx + D[0], by + D[1]), (ax + D[0], ay + D[1])]).buffer(0))
    eps = 1e-4 * math.sqrt(max(geom.area, 1e-9))
    g = unary_union(staying + moved + bridges).buffer(eps, join_style=2).buffer(-eps, join_style=2)
    return g, v
