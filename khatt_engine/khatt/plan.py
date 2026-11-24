"""Step one of the hybrid system: a plan, written by the one who looked at the logo, carried out by
the engine.

The model (a person, or a model that sees) reads the logo and writes a plan for it:
  frame    where the logo's line stands in the picture: the crop, the baseline row, the x origin, and
           `lift`, how far the bottom of the logo's join sits below the baseline (so that the engine's
           join, from 0 to the pen's height, meets the logo's joins without a step)
  style    the logo's measures, in the frame's units (the logo's pixels)
  cuts     the pieces of the logo: each from the logo's clean drawing (or from another cut), by boxes
           to keep, boxes and polygons and curves to leave out, a half plane, a mirror, a move
  glyphs   for each letter form the plan gives ("و.F", "ب.M" ...): the cuts and bars it is put
           together from, the corners to round, and where its marks go
The engine draws the logo by numbers (vector.trace), carries the plan out exactly, and gives every
glyph its join points, so that the typesetter joins the logo's letters, and the letters the engine
draws by its rules for the forms the plan does not give, with one join: the logo's.

A letter form the plan gives serves the letters with the same body (ب for ت ث ن ي in the joined
forms, ف for ق), each with its own marks.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Box = List[float]


# ---------------------------------------------------------------- geometry helpers
def _boxes(bs):
    from shapely.geometry import box
    from shapely.ops import unary_union
    return unary_union([box(*b) for b in bs])


def _bezier(p0, p1, p2, p3, n: int = 48):
    t = np.linspace(0, 1, n)[:, None]
    P = [np.array(q, float) for q in (p0, p1, p2, p3)]
    return ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t * t * P[2] + t ** 3 * P[3]).tolist()


def _rings(geom) -> List[List[List[float]]]:
    out = []
    for p in getattr(geom, "geoms", [geom]):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        out.append([[round(x, 3), round(y, 3)] for x, y in list(p.exterior.coords)[:-1]])
        out += [[[round(x, 3), round(y, 3)] for x, y in list(h.coords)[:-1]] for h in p.interiors]
    return out


def _geom(rings):
    from shapely.geometry import Polygon
    acc = None
    for r in rings:
        if len(r) >= 3:
            q = Polygon(r).buffer(0)
            acc = q if acc is None else acc.symmetric_difference(q)
    return acc


def round_corner(geom, x: float, y: float, r: float, sx: float, sy: float):
    """Round the corner of `geom` at (x, y) with radius r; (sx, sy) point from the corner into the ink."""
    from shapely.geometry import Point, box
    cx, cy = x + sx * r, y + sy * r
    sq = box(min(x, cx), min(y, cy), max(x, cx), max(y, cy))
    return geom.difference(sq.difference(Point(cx, cy).buffer(r, 64)))


# ---------------------------------------------------------------- the logo drawn by numbers
def logo_geometry(grey: np.ndarray, frame: Dict[str, Any], err: float = 0.42):
    """The logo's line drawn by numbers in the plan's frame: x from the frame's origin, y up from the
    baseline, in the picture's pixels."""
    from shapely import affinity
    from .vector import trace
    x0, y0, x1, y1 = [int(round(v)) for v in frame["crop"]]
    geom, _ = trace(np.asarray(grey, float)[y0:y1, x0:x1], err=err)
    geom = affinity.scale(geom, 1, -1, origin=(0, 0))                     # y up
    return affinity.translate(geom, x0 - frame.get("origin_x", 0.0), frame["baseline"] - y0)


def _cut(src, spec: Dict[str, Any], cuts: Dict[str, Any]):
    """One piece: `src` (the logo, or the cut named in spec['from']) by the spec's operations, in
    this order: keep (boxes), keep_poly, drop (boxes), drop_poly, drop_bezier, mirror_x, dx/dy."""
    from shapely import affinity
    from shapely.geometry import Polygon
    g = cuts[spec["from"]] if spec.get("from") and spec["from"] != "logo" else src
    if spec.get("keep"):
        g = g.intersection(_boxes(spec["keep"]))
    if spec.get("keep_poly"):
        g = g.intersection(Polygon(spec["keep_poly"]).buffer(0))
    if spec.get("drop"):
        g = g.difference(_boxes(spec["drop"]))
    if spec.get("drop_poly"):
        g = g.difference(Polygon(spec["drop_poly"]).buffer(0))
    if spec.get("drop_bezier"):
        b = spec["drop_bezier"]
        g = g.difference(Polygon(_bezier(*b["curve"]) + [tuple(q) for q in b["close"]]).buffer(0))
    if spec.get("mirror_x") is not None:
        g = affinity.scale(g, -1, 1, origin=(float(spec["mirror_x"]), 0))
    if spec.get("parts"):                                    # a cut put together from others
        from shapely.ops import unary_union
        g = unary_union([g] + [_part(src, q, cuts, spec.get("pen")) for q in spec["parts"]]) if spec.get("from") else \
            unary_union([_part(src, q, cuts, spec.get("pen")) for q in spec["parts"]])
    if spec.get("dx") or spec.get("dy"):
        g = affinity.translate(g, float(spec.get("dx", 0.0)), float(spec.get("dy", 0.0)))
    return g.buffer(0)


PEN = {"thick": 10.0, "thin": 5.6, "p": 4.0}     # overridden by the plan's style["pen"]: the logo's measured pen


def _path_points(path, n: int = 48):
    """A centre line ["M", x, y], ["L", x, y], ["C", x1, y1, x2, y2, x, y] as points with unit tangents."""
    pts, cur = [], None
    for cmd in path:
        if cmd[0] == "M":
            cur = (float(cmd[1]), float(cmd[2]))
            continue
        if cmd[0] == "L":
            a, b = np.array(cur), np.array(cmd[1:3], float)
            d = b - a
            L = float(np.hypot(*d)) or 1.0
            seg = [(*(a + d * t), *(d / L)) for t in np.linspace(0, 1, max(4, int(L / 1.5)))]
            cur = tuple(b)
        else:
            P = [np.array(cur)] + [np.array(cmd[i:i + 2], float) for i in (1, 3, 5)]
            seg = []
            for t in np.linspace(0, 1, n):
                m = 1 - t
                p = m ** 3 * P[0] + 3 * m * m * t * P[1] + 3 * m * t * t * P[2] + t ** 3 * P[3]
                d = 3 * m * m * (P[1] - P[0]) + 6 * m * t * (P[2] - P[1]) + 3 * t * t * (P[3] - P[2])
                L = float(np.hypot(*d)) or 1.0
                seg.append((*p, *(d / L)))
            cur = tuple(P[3])
        pts += seg if not pts else seg[1:]
    return pts


def pen_stroke(path, pen: Dict[str, float]):
    """The logo's pen along a centre line: as thick as the pen is in each direction (thick level, thin
    upright: thin + (thick - thin) |cos|^p), built of small quadrilaterals so that a tight inside of a
    curve folds in on itself without a hole."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    th, tn, pw = float(pen["thick"]), float(pen["thin"]), float(pen["p"])
    pts = _path_points(path)
    W = lambda tx: tn + (th - tn) * abs(tx) ** pw
    quads = []
    for (x0, y0, tx0, ty0), (x1, y1, tx1, ty1) in zip(pts, pts[1:]):
        w0, w1 = W(tx0) / 2, W(tx1) / 2
        quads.append(Polygon([(x0 - ty0 * w0, y0 + tx0 * w0), (x1 - ty1 * w1, y1 + tx1 * w1),
                              (x1 + ty1 * w1, y1 - tx1 * w1), (x0 + ty0 * w0, y0 - tx0 * w0)]).buffer(0.02))
    return unary_union(quads).buffer(0)


def _superellipse(cx, cy, a, b, n, N: int = 240):
    from shapely.geometry import Polygon
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return Polygon(np.c_[cx + a * np.sign(c) * np.abs(c) ** (2 / n), cy + b * np.sign(s) * np.abs(s) ** (2 / n)])


def _part(src, q: Dict[str, Any], cuts: Dict[str, Any], pen: Optional[Dict[str, float]] = None):
    """One part of a glyph or a cut: a rectangle, a polygon, a stroke of the logo's pen, a ring (an
    outer and an inner superellipse), or a cut of the logo; any of them may be clipped, moved, mirrored."""
    from shapely import affinity
    from shapely.geometry import Polygon, box
    if "rect" in q:
        g = box(*q["rect"])
    elif "poly" in q:
        g = Polygon(q["poly"]).buffer(0)
    elif "stroke" in q:
        g = pen_stroke(q["stroke"], {**PEN, **(pen or {}), **q.get("pen", {})})
    elif "ring" in q:
        g = _superellipse(*q["ring"]["outer"]).difference(_superellipse(*q["ring"]["inner"])).buffer(0)
    else:
        return _cut(src, q, cuts)
    if q.get("keep"):
        g = g.intersection(_boxes(q["keep"]))
    if q.get("drop"):
        g = g.difference(_boxes(q["drop"]))
    if q.get("mirror_x") is not None:
        g = affinity.scale(g, -1, 1, origin=(float(q["mirror_x"]), 0))
    if q.get("dx") or q.get("dy"):
        g = affinity.translate(g, float(q.get("dx", 0.0)), float(q.get("dy", 0.0)))
    return g.buffer(0)


# ---------------------------------------------------------------- the glyphs
def _band_min_x(g, y0: float, y1: float) -> Optional[float]:
    from shapely.geometry import box
    s = g.intersection(box(-1e6, y0, 1e6, y1))
    return None if s.is_empty else s.bounds[0]


def level_joins(g, hair: float, reach: float = 0.8, step: float = 0.5,
                notch_depth: float = 0.3, notch_width: float = 1.0):
    """The joins that sit on the baseline set exactly on it: where the glyph's lowest ink is within
    `reach` of the baseline (a join, not a tail or a cup), its bottom made level at 0. The logo's join is
    one bar; drawn by hand, and pieced together from two places of the logo, its bottom wavers by a
    fraction of a pixel, and the eye sees each step where two pieces meet.

    A notch in a join's bottom where two pieces meet (white under the bar, ink on the baseline on either
    side of it, no deeper than `notch_depth` and no wider than `notch_width` of the pen's height) is
    filled too. A deeper or wider arch is the designer's (the arch under the belly of a ه) and is kept."""
    from shapely.geometry import LineString, box
    from shapely.ops import unary_union
    x0, _, x1, _ = g.bounds
    xs = np.arange(x0, x1 + step, step)
    on, lows = [], []
    for x in xs:
        s = g.intersection(LineString([(x, -50.0), (x, hair)]))
        ys = [c[1] for gm in getattr(s, "geoms", [s]) if not gm.is_empty for c in gm.coords]
        lows.append(min(ys) if ys else None)
        on.append(bool(ys) and -reach < min(ys) < reach)
    fills, i, n = [], 0, len(xs)
    while i < n:
        lo = lows[i]
        if lo is not None and reach <= lo < notch_depth * hair:
            j = i
            while j < n and lows[j] is not None and reach <= lows[j] < notch_depth * hair:
                j += 1
            bounded = i > 0 and j < n and lows[i - 1] is not None and lows[j] is not None and \
                lows[i - 1] < reach and lows[j] < reach
            if bounded and xs[j] - xs[i - 1] <= notch_width * hair:
                fills.append(box(xs[i - 1], 0.0, xs[j], max(lows[i:j]) + 0.3))
                on[i:j] = [True] * (j - i)
            i = j
        else:
            i += 1
    if fills:
        g = g.union(unary_union(fills)).buffer(0)
    runs, start = [], None
    for x, o in zip(list(xs) + [xs[-1] + step], on + [False]):      # a run reaching the end is closed too
        if o and start is None:
            start = x
        if not o and start is not None:
            if x - start >= 3 * step:
                runs.append((start, x - step))
            start = None
    if not runs:
        return g
    add = unary_union([box(a, 0.0, b, 0.5 * hair) for a, b in runs])
    cut = unary_union([box(a, -reach - 0.01, b, 0.0) for a, b in runs])
    return g.difference(cut).union(add.intersection(g.buffer(reach + 0.05))).buffer(0)


def fill_join(g, hair: float, edge: float, side: str, stub: float = 3.0):
    """The join at one end of a glyph made one solid bar: white left in the join's band between the
    glyph's edge (x = `edge`) and its bar, under a fillet or a head that reaches past the end of the bar
    with nothing under it, is filled from the edge to the bar. `side` is "right" (the join arriving) or
    "left" (the join leaving)."""
    from shapely.geometry import LineString, box
    reach = 4.0 * hair
    gap = None
    for f in (0.15, 0.5, 0.85):
        y = f * hair
        if side == "right":
            s = g.intersection(LineString([(edge - reach, y), (edge, y)]))
            ends = [q.bounds[2] for q in getattr(s, "geoms", [s]) if not q.is_empty]
            if ends and max(ends) < edge - stub:
                gap = max(ends) if gap is None else min(gap, max(ends))
        else:
            s = g.intersection(LineString([(edge, y), (edge + reach, y)]))
            starts = [q.bounds[0] for q in getattr(s, "geoms", [s]) if not q.is_empty]
            if starts and min(starts) > edge + stub:
                gap = min(starts) if gap is None else max(gap, min(starts))
    if gap is None:
        return g
    fill = box(gap - 0.3, 0.0, edge, hair) if side == "right" else box(edge, 0.0, gap + 0.3, hair)
    return g.union(fill).buffer(0)


def build_glyph(src, key: str, spec: Dict[str, Any], cuts: Dict[str, Any], frame: Dict[str, Any],
                hair: float, x_height: float, pen: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
    """One letter form: its parts put together, its corners rounded, the seams closed; lifted onto
    the engine's baseline, its right edge at x = 0, and its join points and marks' places worked out
    for its form (the right end of a form joined from the right is where the join arrives; the left
    end of the join a joining form leaves at the baseline is where the next letter's join begins)."""
    from shapely import affinity
    from shapely.ops import unary_union
    form = key.split(".")[1]
    g = unary_union([_part(src, q, cuts, pen) for q in spec["parts"]]).buffer(0)
    if spec.get("drop"):
        g = g.difference(_boxes(spec["drop"]))
    if spec.get("holes"):                                    # counters cut out of the union (a loop's white)
        from shapely.geometry import Polygon as _Pg
        g = g.difference(unary_union([_Pg(h).buffer(0) for h in spec["holes"]]))
    for x, y, r, sx, sy in spec.get("round", []):
        g = round_corner(g, x, y, r, sx, sy)
    r = float(spec.get("close", 1.5))
    if r > 0:
        g = g.buffer(r, join_style=1).buffer(-r, join_style=1)
    if g.is_empty:
        return None
    lift = float(frame.get("lift", 0.0))
    # the glyph's right edge: where a join from the right arrives, or where the letter begins; a part may
    # reach further right below the baseline (the bowl of ى and ح runs on under the letter before it)
    right = float(spec["right"]) if spec.get("right") is not None else g.bounds[2]
    g = affinity.translate(g, -right, lift)
    if spec.get("level", True):
        g = level_joins(g, hair)
    mk = {k: [float(v[0]) - right, float(v[1]) + lift] for k, v in spec.get("marks", {}).items()}
    entry = form in ("M", "F")
    exits = form in ("I", "M")
    overlap = 0.5 * hair
    # the ends of its joins squared to the engine's join (0 to the pen's height), so that the two meet
    # without a step: the logo's join is a bar of that height, drawn by hand a fraction of a pixel off
    from shapely.geometry import box as _box
    stub = float(spec.get("stub", 3.0))
    if entry:
        g = fill_join(g, hair, 0.0, "right", stub)
        g = g.difference(_box(-stub, -1.0, 0.01, hair + 1.0)).union(_box(-stub, 0.0, 0.0, hair)).buffer(0)
    if exits:
        ex0 = _band_min_x(g, 0.3 * hair, 0.7 * hair)
        if ex0 is not None:
            g = fill_join(g, hair, ex0, "left", stub)
            g = g.difference(_box(ex0 - 0.01, -1.0, ex0 + stub, hair + 1.0)).union(_box(ex0, 0.0, ex0 + stub, hair)).buffer(0)
    body_low = -0.15 * x_height
    above = g.intersection(_boxes([[-1e6, body_low, 1e6, 1e6]]))
    left = above.bounds[0] if not above.is_empty else g.bounds[0]
    tail = g.bounds[0] if g.bounds[0] < left - 0.5 else None
    out = dict(polys=_rings(g), left=left, tail=tail, entry_edge=0.0,
               entry_x=-overlap if entry else 0.0, marks=mk, round=bool(spec.get("round_right")))
    if exits:
        ex = _band_min_x(g, 0.3 * hair, 0.7 * hair)
        if ex is None:
            return None
        out["exit_edge"] = ex
        out["exit_x"] = ex + overlap
    return out


def execute(plan: Dict[str, Any], grey: np.ndarray) -> Dict[str, Any]:
    """The plan carried out on the logo's picture (grey, ink dark): the glyphs in the engine's form,
    the measures, and what was done."""
    frame = plan["frame"]
    style = dict(plan.get("style", {}))
    W = logo_geometry(grey, frame)
    cuts: Dict[str, Any] = {}
    for name, spec in plan.get("cuts", {}).items():           # in order: a cut may use those before it
        cuts[name] = _cut(W, spec, cuts)
    hair = float(style.get("hair", 10.0))
    xh = float(style.get("x_height", 40.0))
    glyphs, failed = {}, []
    for key, spec in plan.get("glyphs", {}).items():
        got = build_glyph(W, key, spec, cuts, frame, hair, xh, style.get("pen"))
        if got is None:
            failed.append(key)
        else:
            glyphs[key] = got
    return dict(version=1, glyphs=glyphs, style=style, failed=failed,
                cuts={k: round(v.area, 1) for k, v in cuts.items()}, logo_area=round(W.area, 1))


# ---------------------------------------------------------------- the engine's side: glyphs for the typesetter
MARK_SPEC = {"DOT ABOVE": "a1", "DOT BELOW": "b1", "2 DOTS ABOVE": "a2", "2 DOTS BELOW": "b2",
             "3 DOTS ABOVE": "a3", "3 DOTS BELOW": "b3", "HAMZA ABOVE": "ha", "HAMZA BELOW": "hb",
             "MADDA ABOVE": "md"}


def _spec(c: str, form: str) -> Optional[str]:
    from .anatomy import marks
    got = [MARK_SPEC[m] for m in marks(c, form) if m in MARK_SPEC]
    return got[0] if got else None


def find(st, c: str, form: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """The plan's glyph for this letter form: its own, or a letter's with the same body."""
    G = (st.plan or {}).get("glyphs") if getattr(st, "plan", None) else None
    if not G:
        return None
    key = f"{c}.{form}"
    if key in G:
        return key, G[key]
    from .anatomy import same_body
    for k, v in G.items():
        s, f = k.split(".")
        if f == form and s != c and same_body(s, c, form):
            return k, v
    return None


def glyph_for(st, c: str, form: str):
    """The plan's letter as a glyph of the engine: its body as the logo draws it (from the plan's
    cuts), and this letter's own marks, drawn the logo's way, where the plan says marks go."""
    from .letters import Glyph, marks
    from .library import Tagged
    got = find(st, c, form)
    if got is None:
        return None
    key, v = got
    shapes = [Tagged(v["polys"], "logo", "من الشعار")]
    spec = _spec(c, form)
    # the marks the plan's own letter had are not part of its body (they are the engine's, by spec)
    m = v.get("marks", {})
    if spec:
        if spec[0] in "ah" and spec != "hb":
            if "inside" in m and spec in ("a1", "a2") and form in ("i", "F"):
                shapes += marks(st, spec, m["inside"][0], at=m["inside"][1])
            elif "above" in m:
                shapes += marks(st, spec, m["above"][0], above_ref=m["above"][1])
            else:
                return None                                   # the plan gives no place for this letter's marks
        else:
            if "inside_below" in m:
                shapes += marks(st, spec, m["inside_below"][0], at=m["inside_below"][1])
            elif "below" in m:
                shapes += marks(st, spec, m["below"][0], below_ref=m["below"][1])
            else:
                return None
    ex = v.get("exit_x")
    return Glyph(shapes, left=float(v["left"]), entry_x=float(v["entry_x"]), entry_edge=float(v["entry_edge"]),
                 exit_x=None if ex is None else float(ex),
                 exit_edge=None if v.get("exit_edge") is None else float(v["exit_edge"]),
                 tail=None if v.get("tail") is None else float(v["tail"]))


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
