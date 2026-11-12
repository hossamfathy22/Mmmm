"""Learning letterforms from open-licensed typefaces.

1. Fingerprint   every font is measured with the same engine as the logo (stem/x-height, hair/stem,
                 pen power, stem tops, dots, heads, joins) and the result is cached.
2. Adapt         a font is brought to the logo's rules: scaled to the logo's x-height, its outlines
                 offset until its stems are as thick as the logo's, slanted like the logo, and its
                 dots replaced by the logo's dot (two dots by the logo's dash when the logo does that).
3. Choose        each candidate redraws the logo's own text over the image (same alignment and score
                 as fit.py). The source that redraws the logo best builds the new word, and each
                 letter of the logo reports which source draws it best.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import fonts as F
from .fit import _iou, _shifted_iou, rasterize
from .geometry import circle, rect
from .library import Tagged
from .letters import Glyph
from .style import Style
from .typeset import Layout, Placed, shape_forms

PROBE = "سسس بنت لمن قلم فيه"
FP_FILE = "fingerprints.json"


# ------------------------------------------------------------------ outlines -> shapely, per letter
def _letters(face: F.Face, text: str) -> Dict[int, object]:
    """Shapely geometry of each source character (non-zero fill: holes are contours of opposite
    orientation lying inside an outer contour)."""
    per: Dict[int, List[List[Tuple[float, float]]]] = {}
    for cl, contours in F.shape(face, text):
        per.setdefault(cl, []).extend(contours)
    out = {}
    for cl, contours in per.items():
        polys = []
        for c in contours:
            if len(c) < 3:
                continue
            p = Polygon(c)
            if not p.is_valid:
                p = p.buffer(0)
            a = sum((c[i][0] * c[(i + 1) % len(c)][1] - c[(i + 1) % len(c)][0] * c[i][1]) for i in range(len(c))) / 2
            polys.append((p, a))
        if not polys:
            continue
        big = max(polys, key=lambda v: abs(v[1]))
        outer_sign = 1 if big[1] > 0 else -1
        outers = [p for p, a in polys if (a > 0) == (outer_sign > 0)]
        holes = [p for p, a in polys if (a > 0) != (outer_sign > 0)]
        g = unary_union(outers)
        if holes:
            g = g.difference(unary_union(holes))
        out[cl] = g
    return out


def _parts(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]


def _shape_of(poly: Polygon, src: str, name: str) -> Tagged:
    rings = [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
    return Tagged(rings, src, name)


# ------------------------------------------------------------------ fingerprint
def _fp_path() -> Path:
    return F.CACHE / FP_FILE


def fingerprint(face: F.Face) -> Optional[Dict[str, float]]:
    """Measure a font the way a logo is measured. Cached by file name and size."""
    cache = {}
    try:
        cache = json.loads(_fp_path().read_text())
    except Exception:
        pass
    key = f"{face.path.name}:{face.path.stat().st_size}"
    if key in cache:
        return cache[key]
    from .verify import measure_word
    # render at about 320 units for the em: large enough to see hairlines, small enough to be quick;
    # lengths are converted back to font units afterwards
    s = 320.0 / face.upm
    st = Style(slant=0.0, stem=32.0, x_height=160.0)
    lay = _raw_layout(face, PROBE, st, s)
    got = measure_word(lay)
    fp = None
    if got and got.get("stem") == got.get("stem") and got.get("x_height"):
        u = lambda v: (v / s) if isinstance(v, (int, float)) and v == v else None
        fp = {"x_height_u": u(got["x_height"]), "stem_u": u(got["stem"]), "hair_u": u(got["hair"]),
              "white_u": u(got.get("white")), "power": got.get("power", 2.0), "angle": got.get("angle", 0.0),
              **{k[2:]: v for k, v in got.items() if k.startswith("f_")}}
        for k_ in ("dot_size", "overshoot", "seen_drop", "group_gap"):
            if fp.get(k_) is not None:
                fp[k_] = fp[k_] / s
    cache[key] = fp
    F.CACHE.mkdir(parents=True, exist_ok=True)
    _fp_path().write_text(json.dumps(cache, ensure_ascii=False))
    return fp


def distance(logo: Style, fp: Dict[str, float]) -> float:
    """How far a font's style is from the logo's, in units of what each measure can be trusted to."""
    terms = [
        ((logo.stem / logo.x_height) - fp["stem_u"] / fp["x_height_u"], 0.04),
        ((logo.hair / logo.stem) - fp["hair_u"] / fp["stem_u"], 0.08),
        # the pen angle only means something when both have contrast
        (((logo.pen_angle - fp.get("angle", 0.0)) / 20.0) if (logo.hair / logo.stem < 0.7 and fp["hair_u"] / fp["stem_u"] < 0.7) else 0.0, 1.0),
        (logo.contrast_power - fp.get("power", 2.0), 0.8),
        (logo.top_round - fp.get("top_round", 0.0), 0.3),
        (logo.dot_n - fp.get("dot_n", 2.0), 1.5),
    ]
    if "ring_n" in fp:
        terms.append((logo.ring_n - fp["ring_n"], 1.0))
    if "join_in" in fp:
        terms.append((logo.join_in - fp["join_in"], 0.12))
    return float(math.sqrt(sum((d / s) ** 2 for d, s in terms) / len(terms)))


# ------------------------------------------------------------------ adapting a font to the logo
def _raw_layout(face: F.Face, text: str, st: Style, s: float) -> Layout:
    lay = Layout(text=text, style=st)
    forms = [f for _, f in shape_forms(text)]
    chars = [c for c, _ in shape_forms(text)]
    letters = _letters(face, text)
    for cl in sorted(letters, key=lambda c: c):
        geom = letters[cl]
        shapes = [_shape_of(p, f"font:{face.label}", "حرف من خط") for p in _parts(geom)]
        shapes = [Tagged([[(x * s, y * s) for x, y in ring] for ring in sh], sh.source, sh.name) for sh in shapes]
        if not shapes:
            continue
        xs = [x for sh in shapes for ring in sh for x, _ in ring]
        g = Glyph(shapes, left=min(xs))
        ch = chars[cl] if cl < len(chars) else "?"
        form = forms[cl] if cl < len(forms) else "i"
        lay.placed.append(Placed(ch, form, 0.0, g))
        lay.shapes += shapes
    # right to left reading order, like the engine's layouts
    lay.placed.sort(key=lambda p: -max(x for sh in p.glyph.shapes for ring in sh for x, _ in ring))
    return lay


def text_height(face: F.Face, text: str) -> float:
    ys = [y for _, contours in F.shape(face, text) for c in contours for _, y in c]
    return (max(ys) - min(ys)) if ys else 0.0


def adapt(face: F.Face, text: str, logo: Style, fp: Dict[str, float], size_k: float = 1.0,
          weight_k: float = 1.0, base_scale: Optional[float] = None) -> Layout:
    """The font, set to the logo's rules: size, stem weight, slant, dots.
    base_scale (font units -> logo px) comes from matching the height of the same text in the logo
    and in the font; without it the x-heights are matched. size_k and weight_k fine-tune both."""
    s = (base_scale if base_scale else logo.x_height / fp["x_height_u"]) * size_k
    delta = (logo.stem - fp["stem_u"] * s) / 2 * weight_k
    letters = _letters(face, text)
    chars = [c for c, _ in shape_forms(text)]
    forms = [f for _, f in shape_forms(text)]
    lay = Layout(text=text, style=logo)
    src = f"font:{face.label}"
    for cl in sorted(letters):
        geom = letters[cl]
        from shapely import affinity
        geom = affinity.scale(geom, s, s, origin=(0, 0))
        dots, body = [], []
        for p in _parts(geom):
            minx, miny, maxx, maxy = p.bounds
            w, h = maxx - minx, maxy - miny
            compact = p.area / max(1e-6, w * h)
            if (max(w, h) < 1.9 * logo.dot and 0.5 < w / max(h, 1e-6) < 2.2 and compact > 0.45
                    and not list(p.interiors) and (miny > 0.45 * logo.x_height or maxy < 0.15 * logo.x_height)):
                dots.append(p)
            else:
                body.append(p)
        shapes: List[Tagged] = []
        if body:
            b = unary_union(body)
            if abs(delta) > 0.05:
                b = b.buffer(delta, join_style=2, mitre_limit=2.5)
            shapes += [_shape_of(p, src, "حرف من خط") for p in _parts(b) if p.area > 1]
        # the logo's own marks instead of the font's
        cents = sorted(((d.centroid.x, d.centroid.y) for d in dots), key=lambda c: c[0])
        used = set()
        for i, (x, y) in enumerate(cents):
            if i in used:
                continue
            pair = next((j for j in range(i + 1, len(cents)) if j not in used and abs(cents[j][1] - y) < 0.6 * logo.dot
                         and abs(cents[j][0] - x) < 2.4 * logo.dot), None)
            if pair is not None and logo.dash_marks:
                used.add(pair)
                mx = (x + cents[pair][0]) / 2
                shapes.append(Tagged(rect(mx - 0.75 * logo.dot, mx + 0.75 * logo.dot, y - logo.hair / 2, y + logo.hair / 2), "logo", "شرطة الشعار"))
            else:
                shapes.append(Tagged(circle(x, y, logo.dot / 2, logo.tan, shape_n=logo.dot_n, rot=logo.dot_rot), "logo", "نقطة الشعار"))
        if not shapes:
            continue
        xs = [x for sh in shapes for ring in sh for x, _ in ring]
        ch = chars[cl] if cl < len(chars) else "?"
        lay.placed.append(Placed(ch, forms[cl] if cl < len(forms) else "i", 0.0, Glyph(shapes, left=min(xs))))
        lay.shapes += shapes
    lay.placed.sort(key=lambda p: -max(x for sh in p.glyph.shapes for ring in sh for x, _ in ring))
    return lay


# ------------------------------------------------------------------ choosing the source
@dataclass
class Candidate:
    label: str
    source: str                 # "engine" or "font:<label>"
    distance: float
    iou: float
    letters: List[Dict] = field(default_factory=list)
    tune: Dict[str, float] = field(default_factory=dict)
    pick: Optional[tuple] = field(default=None, repr=False)     # (face, fingerprint, tuning) for fonts


def overlay(lay: Layout, L, ink: np.ndarray, scale: float) -> bytes:
    """Difference map of a source's redrawing over the logo (dark both, red logo only, blue redrawing only)."""
    import io
    from PIL import Image
    pad = int(0.6 * L.stem)
    r0, r1 = max(0, L.top - pad), min(ink.shape[0], L.bot + pad + 1)
    target = ink[r0:r1].astype(bool)
    m, _, _ = rasterize(lay, scale, target.shape, float(L.x1) + 1.0, (L.base_row - r0) - L.baseline_shift,
                        target=target, reach=3.0 * L.stem, v_reach=0.25 * L.x_height)
    rgb = np.full(target.shape + (3,), 255, np.uint8)
    rgb[target & m] = (40, 48, 64)
    rgb[target & ~m] = (220, 60, 70)
    rgb[~target & m] = (40, 130, 230)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def evaluate(lay: Layout, L, ink: np.ndarray, scale: float) -> Tuple[float, List[Dict]]:
    pad = int(0.6 * L.stem)
    r0, r1 = max(0, L.top - pad), min(ink.shape[0], L.bot + pad + 1)
    target = ink[r0:r1].astype(bool)
    base_row = (L.base_row - r0) - L.baseline_shift
    m, spans, _ = rasterize(lay, scale, target.shape, float(L.x1) + 1.0, base_row, target=target, reach=3.0 * L.stem,
                            v_reach=0.25 * L.x_height)
    letters = []
    for ch, a, b in spans:
        c0, c1 = max(0, int(a)), min(target.shape[1], int(math.ceil(b)) + 1)
        if c1 > c0:
            letters.append({"letter": ch, "iou": round(_shifted_iou(m, target, c0, c1, int(0.6 * L.stem)), 3)})
    return _iou(m, target), letters


def choose(L, ink: np.ndarray, scale: float, logo_text: str, logo: Style, engine_iou: float,
           engine_letters: List[Dict], top: int = 6) -> Tuple[List[Candidate], Optional[Tuple[F.Face, Dict]]]:
    """Rank the fonts by style, redraw the logo with the closest ones, and return all candidates
    (the engine included) sorted by how well they redraw the logo, plus the winning font if any."""
    faces = []
    for p in F.font_files():
        try:
            face = F.load(str(p))
            fp = fingerprint(face)
        except Exception:
            continue
        if fp:
            faces.append((distance(logo, fp), face, fp))
    # every font redraws the logo once, untuned; the style fingerprint only breaks ties. The best
    # few are then tuned. (A fingerprint alone is not a reliable judge of which letters will fit.)
    quick = []
    logo_h = (L.height - 2) / scale                     # the line's full ink height, original px
    for d, face, fp in faces:
        try:
            th = text_height(face, logo_text)
            fp = dict(fp, base_scale=(logo_h / th) if th else None)
            q = evaluate(adapt(face, logo_text, logo, fp, base_scale=fp["base_scale"]), L, ink, scale)[0]
        except Exception:
            continue
        quick.append((q - 0.002 * d, d, face, fp))
    quick.sort(key=lambda v: -v[0])
    cands = [Candidate("المحرك (القطع الست)", "engine", 0.0, engine_iou, engine_letters)]
    best_font = None
    best_iou = engine_iou
    for _, d, face, fp in quick[:top]:
        try:
            # fit size and weight of this font to the logo (small pattern search on the redrawing)
            ks = {"size_k": 1.0, "weight_k": 1.0}
            bs = fp.get("base_scale")
            cur = evaluate(adapt(face, logo_text, logo, fp, base_scale=bs, **ks), L, ink, scale)[0]
            steps = {"size_k": 0.08, "weight_k": 0.5}
            for _ in range(10):
                moved = False
                for key in ks:
                    for sgn in (1, -1):
                        trial = dict(ks); trial[key] = ks[key] + sgn * steps[key]
                        if not (0.7 <= trial["size_k"] <= 1.4 and -1.0 <= trial["weight_k"] <= 3.0):
                            continue
                        v = evaluate(adapt(face, logo_text, logo, fp, base_scale=bs, **trial), L, ink, scale)[0]
                        if v > cur + 1e-4:
                            cur, ks, moved = v, trial, True
                            break
                if not moved:
                    steps = {k: v / 2 for k, v in steps.items()}
                    if steps["size_k"] < 0.01:
                        break
            iou, letters = evaluate(adapt(face, logo_text, logo, fp, base_scale=bs, **ks), L, ink, scale)
        except Exception:
            continue
        cands.append(Candidate(face.label, f"font:{face.label}", round(d, 2), round(iou, 4), letters, {**ks, "base_scale": bs},
                               (face, fp, {**ks, "base_scale": bs})))
        if iou > best_iou + 0.01:
            best_iou, best_font = iou, (face, fp, ks)
    cands.sort(key=lambda c: -c.iou)
    return cands, best_font
