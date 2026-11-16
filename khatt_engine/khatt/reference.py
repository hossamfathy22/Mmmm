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


def _mask_strokes(m: np.ndarray) -> List[List[Tuple[float, float, float]]]:
    """The centre line of an inked shape, stroke by stroke: [(row, col, half-width in pixels)], spurs
    pruned, smoothed along each stroke and kept every 3 pixels."""
    from skimage.morphology import skeletonize
    from .skeleton import _trace
    H, W = m.shape
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
    out = []
    for path in _trace(sk):
        if len(path) < 3:
            continue
        # smoothed along the stroke (a pixel staircase is not the drawn curve), every 3 pixels
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
            pts.append((float(r), float(c), float(rad[ri, ci])))
        out.append(pts)
    return out


@lru_cache(maxsize=4096)
def _structure(path: str, char: str, form: str):
    """The centre line of the letter's body in the typeface, in units of its tooth (x-height = 1):
    a list of strokes, each a list of (x, y, r) with r the typeface's half-width there; x = 0 at the
    right end of the body. None when the typeface has no such letter."""
    from .bench.fonts import load, shape
    from .bench.truth import drop_marks, raster, x_height
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
    holes = ndi.binary_fill_holes(m) & ~m
    lab, nh = ndi.label(holes)
    counters = [float(np.sum(lab == i)) / (PX * PX) for i in range(1, nh + 1) if np.sum(lab == i) >= 0.004 * PX * PX]
    strokes = []
    for path in _mask_strokes(m):
        pts = [(((c - pad) / s + min(xs)) / X, (top - r / s) / X, rad / PX) for r, c, rad in path]
        if len(pts) >= 2:
            strokes.append(pts)
    if not strokes:
        return None
    cols = np.flatnonzero(m.any(axis=0))
    right = ((cols[-1] - pad) / s + min(xs)) / X
    return [[(x - right, y, r) for x, y, r in stroke] for stroke in strokes], ((cols[0] - pad) / s + min(xs)) / X - right, counters


WIDER = (1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.4)


def ref_glyph(path: str, char: str, form: str, st: Style, wide: float = 1.0, engine: Optional[Glyph] = None,
              fit: bool = False) -> Optional[Glyph]:
    """`char` in `form`, built on the structure of the typeface at `path` and drawn with the logo's pen
    (st.ref_how "pen" or "strokes"), or taken from its outline at the logo's weight ("outline").
    `wide`: how much wider than the typeface the logo's rhythm draws a letter. A letter with a counter
    (a loop, a head) is drawn wider still if the logo's pen would close it, as a hand lengthens a loop
    to fit a heavy pen. The marks are the engine's letter's (`engine`, built if not given)."""
    if st.ref_how == "outline":
        return ref_outline(path, char, form, st, engine, fit=fit, wide=wide)
    # "pen" (and "strokes", without the check below): the structure, drawn with the logo's pen
    got = _structure(path, char, form)
    if got is None:
        return None
    if fit and engine is not None:
        # the reference's structure, the engine's proportions: the letter as wide as the engine draws
        # it for this logo (its width carries the logo's rhythm and the hand's judgement), the pen's
        # thickness kept (the structure is drawn wider or narrower, not stretched)
        from .typeset import _is_mark, flat
        exs = [x for sh in flat(engine.shapes) if not _is_mark(sh, st) for poly in sh for x, _ in poly]
        span = -got[1]
        if exs and span > 0.05:
            wide = max(0.3, (max(exs) - min(exs) - st.stem) / (span * st.x_height))
    need = len(got[2])
    best = None
    for k in WIDER:
        g = _draw(got, char, form, st, wide * k, engine)
        if g is None:
            return None
        open_ = _counters(g, st)
        if open_ >= need:
            best = (open_, g)
            break
        if best is None or open_ > best[0]:
            best = (open_, g)
    g = best[1]
    if st.ref_how == "pen" and not _whole(g, path, char, form, st, need):
        # the centre line of a heavy typeface breaks where its strokes meet (Lalezar's ع ه ف came out in
        # pieces): its outline at the logo's weight then, or (refused by the critic too) the engine's
        return ref_outline(path, char, form, st, engine, fit=fit, wide=wide)
    return g


def _whole(g: Glyph, path: str, char: str, form: str, st: Style, need: int) -> bool:
    """A letter drawn on the reference's centre line holds together as the typeface draws it (no more
    pieces than its outline has) and passes the critic's measures."""
    from shapely.geometry import Polygon
    acc = None
    for ring in g.shapes[0]:
        if len(ring) >= 3:
            q = Polygon(ring).buffer(0)
            acc = q if acc is None else acc.symmetric_difference(q)
    if acc is None or acc.is_empty:
        return False
    rings = _outline(path, char, form)
    if rings is None:
        return False
    out = None
    for ring in rings:
        q = Polygon(ring).buffer(0)
        out = q if out is None else out.symmetric_difference(q)
    pieces = lambda shape: len([p for p in getattr(shape, "geoms", [shape]) if p.geom_type == "Polygon" and p.area > 1e-4 * shape.area])
    if pieces(acc) > pieces(out):
        return False
    # and it is still the typeface's letter: against its outline at the logo's weight, as wide, the
    # common ink over all (a good typeface's letters, drawn with the logo's pen, 0.56 to 0.67; Lalezar's,
    # broken into stubs and hooks, 0.33 to 0.51)
    from shapely import affinity
    X = st.x_height
    stem = _stem(path)
    ref = affinity.scale(out, xfact=X, yfact=X, origin=(0, 0))
    if stem is not None:
        ref = ref.buffer(0.5 * (st.stem - stem * X), 8)
    if ref.is_empty:
        return False
    a = affinity.translate(acc, xoff=-acc.bounds[2])
    ref = affinity.translate(ref, xoff=-ref.bounds[2])
    w = ref.bounds[2] - ref.bounds[0]
    if w > 1e-6:
        ref = affinity.scale(ref, xfact=(a.bounds[2] - a.bounds[0]) / w, yfact=1.0, origin=(0, 0))
    if a.intersection(ref).area < SAME_SHAPE * a.union(ref).area:
        return False
    return _passes(acc, st, need)


SAME_SHAPE = 0.52          # how much of a letter drawn with the logo's pen must lie on the typeface's own


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
        # the pen's width follows the line's direction, and a skeleton's direction wavers from point
        # to point: the widths are smoothed along the stroke, or the edge comes out notched
        if len(pts) >= 3:
            ws = [p[4] for p in pts]
            ws = [sum(ws[max(0, i - 2):i + 3]) / len(ws[max(0, i - 2):i + 3]) for i in range(len(ws))]
            pts = [(X0, Y0, tx, ty, w, R) for (X0, Y0, tx, ty, _, R), w in zip(pts, ws)]
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
        quads += [Point(x, y).buffer(w / 2, 12) for x, y, _, _, w, _ in pts[1:-1]]
        if quads:
            # a hair's breadth in and out again: the steps where two pieces meet at an angle go
            shapes.append(unary_union(quads).buffer(0.02 * X, 8).buffer(-0.02 * X, 8))
    if not shapes:
        return None
    body = unary_union(shapes)
    if body.is_empty:
        return None
    return _finish(body, char, form, st, engine)


def _finish(body, char: str, form: str, st: Style, engine: Optional[Glyph]) -> Optional[Glyph]:
    """A letter's body (a shapely shape in the logo's units) as a glyph: its right end at 0, the
    engine's marks over it, joined where the form joins."""
    rings = []
    for poly in getattr(body, "geoms", [body]):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        rings.append(list(poly.exterior.coords)[:-1])
        rings += [list(h.coords)[:-1] for h in poly.interiors]
    if not rings:
        return None
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


# ------------------------------------------------------------------ a part of the logo fitted to another letter
PART_PX = 6.0               # pixels per unit when a part's centre line is read
FIT_RANGE = (0.6, 1.6)      # how far a part may be shortened or lengthened, made lower or taller


def part_strokes(polys) -> List[List[Tuple[float, float, float]]]:
    """The centre line of a part cut from the logo, in its own units: [(x, y, half-width)]."""
    from PIL import Image, ImageDraw
    rings = [r for r in polys if len(r) >= 3]
    if not rings:
        return []
    xs = [x for r in rings for x, _ in r]; ys = [y for r in rings for _, y in r]
    s = PART_PX
    pad = 3
    W, H = int((max(xs) - min(xs)) * s) + 2 * pad, int((max(ys) - min(ys)) * s) + 2 * pad
    if W < 4 or H < 4 or W * H > 4_000_000:
        return []
    acc = np.zeros((H, W), bool)
    for r in rings:                                   # even-odd: a ring inside another is a counter
        layer = Image.new("L", (W, H), 0)
        ImageDraw.Draw(layer).polygon([((x - min(xs)) * s + pad, (max(ys) - y) * s + pad) for x, y in r], fill=255)
        acc ^= np.array(layer) > 0
    if acc.sum() < 10:
        return []
    return [[((c - pad) / s + min(xs), max(ys) - (r - pad) / s, rad / s) for r, c, rad in path] for path in _mask_strokes(acc)]


def draw_strokes(strokes, transform) -> List[List[Tuple[float, float]]]:
    """Strokes (x, y, half-width) moved by `transform` (a point -> point function) and drawn again with
    their own widths: the part changes its length and height, not its pen."""
    from shapely.geometry import Point, Polygon
    from shapely.ops import unary_union
    shapes = []
    for stroke in strokes:
        pts = [(*transform(x, y), r) for x, y, r in stroke]
        # the centre line stops short of the stroke's end by its half-width: put the end back
        for end, j in ((0, 1), (len(pts) - 1, len(pts) - 2)):
            if len(pts) < 2:
                break
            x0, y0, r0 = pts[end]
            ux, uy = x0 - pts[j][0], y0 - pts[j][1]
            d = math.hypot(ux, uy) or 1.0
            pts[end] = (x0 + ux / d * r0, y0 + uy / d * r0, r0)
        pieces = []
        for (x0, y0, r0), (x1, y1, r1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            nx, ny = -dy / d, dx / d
            pieces.append(Polygon([(x0 + nx * r0, y0 + ny * r0), (x1 + nx * r1, y1 + ny * r1),
                                   (x1 - nx * r1, y1 - ny * r1), (x0 - nx * r0, y0 - ny * r0)]).buffer(0))
        pieces += [Point(x, y).buffer(r, 12) for x, y, r in pts[1:-1]]
        if pieces:
            shapes.append(unary_union(pieces))
    if not shapes:
        return []
    body = unary_union(shapes)
    rings = []
    for poly in getattr(body, "geoms", [body]):
        if poly.geom_type == "Polygon" and not poly.is_empty:
            rings.append(list(poly.exterior.coords)[:-1])
            rings += [list(h.coords)[:-1] for h in poly.interiors]
    return rings


def box(rings):
    xs = [x for r in rings for x, _ in r]; ys = [y for r in rings for _, y in r]
    return min(xs), min(ys), max(xs), max(ys)


def fit_part(part: Dict, target) -> Optional[List[List[Tuple[float, float]]]]:
    """The logo's part fitted to the engine's part of another letter: the box of the engine's part in
    the letter it was cut from is mapped onto the box of the engine's part in this letter (right ends
    and bottoms kept together, lengths and heights scaled within FIT_RANGE), and the logo's part,
    read as a centre line, is drawn again on that map with its own widths."""
    if not part.get("engine"):
        return None
    if "strokes" not in part:
        part["strokes"] = part_strokes(part["polys"])
    if not part["strokes"]:
        return None
    sx0, sy0, sx1, sy1 = box(part["engine"])
    tx0, ty0, tx1, ty1 = box(target)
    if sx1 - sx0 < 1e-6 or sy1 - sy0 < 1e-6:
        return None
    kx = float(np.clip((tx1 - tx0) / (sx1 - sx0), *FIT_RANGE))
    ky = float(np.clip((ty1 - ty0) / (sy1 - sy0), *FIT_RANGE))
    return draw_strokes(part["strokes"], lambda x, y: (tx1 + (x - sx1) * kx, ty0 + (y - sy0) * ky))


# ------------------------------------------------------------------ parts as real typefaces size them
# Words set in each reference typeface and cut into parts the way a logo is: they tell, from real
# designs, how the same part differs from letter to letter (how a ق's head stands to a م's).
REF_WORDS = ("مفقود", "قلم ونور", "فن ولين", "سفر وزن", "صوم فطر", "يعمل قعد", "نبع وكد", "طمع وفم",
             "همم قمر", "فول قمح", "ممر عين", "نقد وفن", "هوى وفم", "ضرب وطن")
_LIBRARY: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
_SHAPES: Dict[str, Dict[str, list]] = {}        # typeface -> "letter.form|part" -> outlines in x-heights (right end at 0)


def family(path_or_stem: str) -> str:
    from pathlib import Path
    stem = Path(path_or_stem).stem
    return stem.rsplit("-", 1)[0] if stem.rsplit("-", 1)[-1].isdigit() else stem


def part_library(rebuild: bool = False, log=None) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """{typeface: {"letter.form|part": (width, height) in x-heights}}, read once and kept on disk."""
    global _LIBRARY
    import json
    import os
    from pathlib import Path
    if _LIBRARY is not None and not rebuild:
        return _LIBRARY
    cache = Path(os.environ.get("KHATT_CACHE", Path.home() / ".cache" / "khatt")) / "ref_parts.json"
    paths = [f for f in faces() if f.endswith("-400.woff2")]
    if cache.exists() and not rebuild:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if (data.get("fonts") == sorted(Path(p).stem for p in paths) and data.get("words") == list(REF_WORDS)
                    and "shapes" in data):
                _LIBRARY = {f: {k: tuple(v) for k, v in d.items()} for f, d in data["parts"].items()}
                _SHAPES.clear(); _SHAPES.update(data["shapes"])
                return _LIBRARY
        except (ValueError, KeyError):
            pass
    from .bench.fonts import load
    from .bench.truth import png, word_image
    from .pipeline import TAKE_MIN, analyse
    from .selfcheck import take_parts
    lib: Dict[str, Dict[str, Tuple[float, float]]] = {}
    shapes: Dict[str, Dict[str, list]] = {}
    for p in paths:
        face = load(p)
        got: Dict[str, Tuple[float, float, float]] = {}
        outl: Dict[str, list] = {}
        for w in REF_WORDS:
            try:
                an = analyse(png(word_image(face, w, x_px=90)[0]), texts=[w], read=False)
            except Exception:
                continue
            if an["health"]["confidence"] < TAKE_MIN:
                continue
            L, k, st = an["lines"][an["src"]], an["k"], an["style"]
            for e in take_parts(L, w, st, k, every=True):
                x0, y0, x1, y1 = box(e["polys"])
                key = f"{e['from']}|{e['name']}"
                if key not in got or e["match"] > got[key][2]:
                    got[key] = ((x1 - x0) / st.x_height, (y1 - y0) / st.x_height, e["match"])
                    X = st.x_height
                    outl[key] = [[[round(x / X, 4), round(y / X, 4)] for x, y in ring] for ring in e["polys"]]
        lib[Path(p).stem] = {k: (round(v[0], 4), round(v[1], 4)) for k, v in got.items()}
        shapes[Path(p).stem] = outl
        if log:
            log(f"{Path(p).stem}: {len(got)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"fonts": sorted(lib), "words": list(REF_WORDS), "parts": lib, "shapes": shapes},
                                ensure_ascii=False), encoding="utf-8")
    _LIBRARY = lib
    _SHAPES.clear(); _SHAPES.update(shapes)
    return lib


def part_ratio(source: str, target: str, name: str, skip: str = "", least: int = 3) -> Optional[Tuple[float, float]]:
    """How the part `name` of letter form `target` stands to the same part of `source` in real typefaces:
    the median ratio of their widths and of their heights over the typefaces that have both (`skip`:
    a typeface family left out, the one standing in for the logo in the examination)."""
    lib = part_library()
    kx, ky = [], []
    for font, parts in lib.items():
        if skip and family(font) == skip:
            continue
        a, b = parts.get(f"{source}|{name}"), parts.get(f"{target}|{name}")
        if a and b and a[0] > 0 and a[1] > 0:
            kx.append(b[0] / a[0]); ky.append(b[1] / a[1])
    if len(kx) < least:
        return None
    return float(np.median(kx)), float(np.median(ky))



# ------------------------------------------------------------------ which part can stand for which, by real designs
_SUBS: Dict[str, Dict[Tuple[str, str, str, str], float]] = {}


def substitutions(skip: str = "", least: int = 3) -> Dict[Tuple[str, str, str, str], float]:
    """{(source "letter.form|part", target "letter.form|part", transform, ""): median IoU over real
    typefaces}: how alike a part of one letter, mirrored or turned, is to another part of another
    letter in real designs (right ends together, and bottoms together when turned). Only pairs found
    in `least` typefaces or more; `skip`: a typeface family left out."""
    if skip in _SUBS:
        return _SUBS[skip]
    from .typeset import TRANSFORMS, _placement, _same_part
    part_library()
    per: Dict[Tuple[str, str, str, str], List[float]] = {}
    for font, parts in _SHAPES.items():
        if skip and family(font) == skip:
            continue
        keys = list(parts)
        for a in keys:
            for b in keys:
                if a == b or a.split("|")[1] == b.split("|")[1]:
                    continue                 # the same kind of part passes as it is, not by substitution
                for t in TRANSFORMS:
                    got = _placement(parts[a], parts[b], t)
                    if got is None:
                        continue
                    per.setdefault((a, b, t, ""), []).append(_same_part(got[2], parts[b]))
    out = {k: float(np.median(v)) for k, v in per.items() if len(v) >= least}
    # and by kinds of part alone, for letters the words did not show: every pair of letters pooled
    kinds: Dict[Tuple[str, str, str, str], List[float]] = {}
    for (a, b, t, _), v in per.items():
        kinds.setdefault((a.split("|")[1], b.split("|")[1], t, "kind"), []).extend(v)
    out.update({k: float(np.median(v)) for k, v in kinds.items() if len(v) >= least})
    _SUBS[skip] = out
    return out


# ------------------------------------------------------------------ the typeface's own outline
# The centre line, redrawn with the logo's pen, breaks where a heavy typeface's strokes meet (Lalezar's
# ع, ه, ف came out in pieces, with stubs). The outline as the typeface draws it is whole: it is taken
# as it is, its weight brought to the logo's stem by moving the outline in or out all round.
@lru_cache(maxsize=4096)
def _outline(path: str, char: str, form: str):
    """The body of the letter in the typeface, its marks left out: rings in units of its tooth
    (x-height = 1), the right end at 0, the baseline at 0. None when the typeface has no such letter."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from .bench.fonts import load, shape
    from .bench.truth import _area, x_height
    face = load(path)
    text = FORMS[form].format(char)
    idx = text.index(char)
    try:
        cs = [c for cl, cont in shape(face, text) if cl == idx for c in cont]
    except Exception:
        return None
    cs = [c for c in cs if len(c) >= 3]
    if not cs:
        return None
    X = x_height(face)
    big = max(cs, key=lambda c: abs(_area(c)))
    outer = 1 if _area(big) > 0 else -1
    fill = [Polygon([(x / X, y / X) for x, y in c]).buffer(0) for c in cs if (1 if _area(c) > 0 else -1) == outer]
    holes = [Polygon([(x / X, y / X) for x, y in c]).buffer(0) for c in cs if (1 if _area(c) > 0 else -1) != outer]
    body = unary_union(fill)
    if holes:
        body = body.difference(unary_union(holes))
    pieces = [g for g in getattr(body, "geoms", [body]) if g.geom_type == "Polygon" and not g.is_empty]
    if not pieces:
        return None
    top = max(g.area for g in pieces)
    keep = [g for g in pieces if not (g.area < 0.2 * top and (g.bounds[1] >= 0.75 or g.bounds[3] <= 0.1))]
    body = unary_union(keep)
    right = body.bounds[2]
    out = []
    for poly in getattr(body, "geoms", [body]):
        if poly.geom_type == "Polygon" and not poly.is_empty:
            out.append(tuple((round(x - right, 4), round(y, 4)) for x, y in list(poly.exterior.coords)[:-1]))
            out += [tuple((round(x - right, 4), round(y, 4)) for x, y in list(h.coords)[:-1]) for h in poly.interiors]
    return tuple(out) if out else None


@lru_cache(maxsize=64)
def _stem(path: str) -> Optional[float]:
    """The typeface's stem in units of its tooth: twice the median half-width along the centre line of
    its medial ب (and its ا)."""
    rs = []
    for c, f in (("ب", "M"), ("ا", "i")):
        got = _structure(path, c, f)
        if got is not None:
            rs += [r for stroke in got[0] for _, _, r in stroke]
    return 2.0 * float(np.median(rs)) if rs else None


def _without_joins(acc, X: float, right: bool, left: bool):
    """The shape without the thin strokes along the baseline at its joining sides: cut where, going
    in from the edge, the shape first rises above the lower part of the tooth."""
    from shapely.geometry import box
    x0, _, x1, _ = acc.bounds

    def low(x: float) -> bool:
        cut = acc.intersection(box(x - 0.01 * X, -1e5, x + 0.01 * X, 1e5))
        if cut.is_empty:
            return False
        geoms = [g for g in getattr(cut, "geoms", [cut]) if not g.is_empty and g.area > 0]
        if len(geoms) != 1:
            return False
        _, ya, _, yb = geoms[0].bounds
        return ya >= -0.15 * X and yb <= 0.4 * X
    step = 0.02 * X
    for side in (("right",) if right else ()) + (("left",) if left else ()):
        x = x1 if side == "right" else x0
        edge = x
        while abs(x - edge) < 0.6 * X and low(x - 0.5 * step if side == "right" else x + 0.5 * step):
            x = x - step if side == "right" else x + step
        if abs(x - edge) >= 0.6 * X or abs(x - edge) < step:
            continue                                     # all join (a lone stroke), or none
        acc = acc.intersection(box(-1e6, -1e6, x, 1e6) if side == "right" else box(x, -1e6, 1e6, 1e6))
        x0, _, x1, _ = acc.bounds
    return acc


def ref_outline(path: str, char: str, form: str, st: Style, engine: Optional[Glyph] = None,
                fit: bool = False, wide: float = 1.0) -> Optional[Glyph]:
    """`char` in `form` as the typeface at `path` draws it, at the logo's x-height, its weight the logo's
    stem, the engine's marks over it. `fit`: as wide as the engine draws the letter for this logo."""
    from shapely.geometry import Polygon
    from shapely import affinity
    rings = _outline(path, char, form)
    stem = _stem(path)
    if rings is None or stem is None:
        return None
    X = st.x_height
    acc = None
    for ring in rings:
        q = Polygon([(x * X, y * X) for x, y in ring]).buffer(0)
        acc = q if acc is None else acc.symmetric_difference(q)
    if acc is None or acc.is_empty:
        return None
    # the typeface's own join, on a side the letter joins, is cut off at the letter's body: the
    # engine's join meets the body, as every join of the word does (two joins of two thicknesses
    # met in a lump, and the critic saw it)
    acc = _without_joins(acc, X, right=form in ("M", "F"), left=form in ("I", "M"))
    if acc is None or acc.is_empty:
        return None
    if fit and engine is not None:
        from .typeset import _is_mark, flat
        exs = [x for sh in flat(engine.shapes) if not _is_mark(sh, st) for poly in sh for x, _ in poly]
        w = acc.bounds[2] - acc.bounds[0]
        if exs and w > 1e-6:
            acc = affinity.scale(acc, xfact=max(0.5, min(2.0, (max(exs) - min(exs)) / w)), yfact=1.0, origin=(0, 0))
    # the typeface's weight brought to the logo's: the outline moved in or out by half the difference;
    # a counter the heavier pen would close is kept open as the hand keeps it, the letter drawn wider
    grow = 0.5 * (st.stem - stem * X)
    need = sum(1 for poly in getattr(acc, "geoms", [acc]) if poly.geom_type == "Polygon"
               for h in poly.interiors if abs(Polygon(h).area) >= 0.02 * X * X)
    # The letter must pass what the engine's own letters pass (critic.critique): its counters open
    # (as wide as half the white between the logo's teeth), no ink piled thicker than the pen. Wider
    # first, as the hand lengthens a loop; then lighter, as the hand draws a loop lighter than its
    # stem; a letter that passes neither way is left to the engine.
    for frac in (1.0, 0.75, 0.5, 0.25, 0.0, -0.25):
        for k in (1.0, 1.15, 1.3, 1.5):
            a = affinity.scale(acc, xfact=k * wide, yfact=1.0, origin=(0, 0)) if k * wide != 1.0 else acc
            g = grow * frac if grow > 0 else grow + (1.0 - frac) * 0.15 * st.stem * (-1)
            body = a.buffer(g, 8, join_style=1) if abs(g) > 1e-6 else a
            if body.is_empty or body.area < 0.3 * a.area:
                continue
            if _passes(body, st, need):
                return _finish(body, char, form, st, engine)
    return None


def _passes(body, st: Style, need: int) -> bool:
    """The critic's measures on one letter: `need` counters open (0.55 white wide at their widest
    row), and nowhere thicker than the pen allows."""
    from PIL import Image, ImageDraw
    s = 20.0 / max(st.stem, 1e-6)                  # the stem 20 pixels thick
    x0, y0, x1, y1 = body.bounds
    W, H = int((x1 - x0) * s) + 6, int((y1 - y0) * s) + 6
    if W * H > 4_000_000:
        return True
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for poly in getattr(body, "geoms", [body]):
        if poly.geom_type != "Polygon":
            continue
        d.polygon([((x - x0) * s + 3, (y1 - y) * s + 3) for x, y in poly.exterior.coords], fill=255)
        for h in poly.interiors:
            d.polygon([((x - x0) * s + 3, (y1 - y) * s + 3) for x, y in h.coords], fill=0)
    m = np.array(im) > 0
    mono = float(np.clip((st.hair / st.stem - 0.5) / 0.5, 0.0, 1.0)) if st.stem else 0.0
    dist = ndi.distance_transform_edt(m)
    if 2 * float(dist.max()) > (1.32 + 0.43 * mono) * 20.0:
        return False
    # nor thinner than the logo's hairline (its centre line, the ends pruned, as the critic measures)
    from skimage.morphology import skeletonize
    sk = skeletonize(m)
    for _ in range(12):
        nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
        ends = sk & (nb <= 2)
        if not ends.any():
            break
        sk = sk & ~ends
    w = 2 * dist[sk]
    if len(w) > 20 and float(np.percentile(w, 5)) < (0.6 - 0.1 * mono) * st.hair * s:
        return False
    holes = ndi.binary_fill_holes(m) & ~m
    lab, n = ndi.label(holes)
    wide_open = 0
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        hm = lab[sl] == i
        if hm.sum() >= 36 and float(hm.sum(axis=1).max()) >= 0.55 * st.white * s:
            wide_open += 1
    return wide_open >= need


# ------------------------------------------------------------------ the logo's parts on the reference's structure
def with_logo_head(g: Glyph, st: Style) -> Optional[Glyph]:
    """The reference's letter with its loop (the head of ف ق ه, around its counter) taken out and the
    logo's own head (st.parts["رأس"]) put in its place: as tall as the loop it stands for, centred on
    it, its shape and its counter the logo's. None when the letter has no loop or the logo no head."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from shapely import affinity
    head = st.parts.get("رأس")
    if not head or not head.get("polys") or not g.shapes:
        return None
    acc = None
    for ring in g.shapes[0]:
        if len(ring) >= 3:
            q = Polygon(ring).buffer(0)
            acc = q if acc is None else acc.symmetric_difference(q)
    if acc is None or acc.is_empty:
        return None
    holes = [Polygon(h) for poly in getattr(acc, "geoms", [acc]) if poly.geom_type == "Polygon" for h in poly.interiors]
    if not holes:
        return None
    hole = max(holes, key=lambda h: h.area)
    loop = acc.intersection(hole.buffer(1.3 * st.stem, 8))      # the stroke around the counter
    if loop.is_empty:
        return None
    lx0, ly0, lx1, ly1 = loop.bounds
    logo = None
    for ring in head["polys"]:
        if len(ring) >= 3:
            q = Polygon(ring).buffer(0)
            logo = q if logo is None else logo.symmetric_difference(q)
    if logo is None or logo.is_empty:
        return None
    hx0, hy0, hx1, hy1 = logo.bounds
    k = (ly1 - ly0) / max(hy1 - hy0, 1e-6)
    logo = affinity.scale(logo, xfact=k, yfact=k, origin=(hx1, hy0))
    hx0, hy0, hx1, hy1 = logo.bounds
    logo = affinity.translate(logo, xoff=0.5 * (lx0 + lx1) - 0.5 * (hx0 + hx1), yoff=ly0 - hy0)
    rest = acc.difference(hole.buffer(1.3 * st.stem, 8).union(logo.envelope.buffer(-0.2 * st.stem)))
    body = unary_union([rest, logo]).buffer(0)
    rings = []
    for poly in getattr(body, "geoms", [body]):
        if poly.geom_type == "Polygon" and not poly.is_empty and poly.area > 0.02 * st.stem * st.stem:
            rings.append(list(poly.exterior.coords)[:-1])
            rings += [list(h.coords)[:-1] for h in poly.interiors]
    if not rings:
        return None
    xs = [x for r in rings for x, _ in r]
    return Glyph([Tagged(rings, "reference", "بنية الخط المرجعي ورأس الشعار")] + list(g.shapes[1:]),
                 left=min(min(xs), g.left), entry_x=g.entry_x, entry_edge=g.entry_edge,
                 exit_x=g.exit_x if g.exit_x is None else min(g.exit_x, min(xs)),
                 exit_edge=g.exit_edge if g.exit_edge is None else min(g.exit_edge, min(xs)))

