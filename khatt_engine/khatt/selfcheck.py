"""The self-check: write the logo's own word with the rules read from it, lay it over the logo, and
say how well each letter matches.

Everything the engine draws rests on what it measured and on how it believes each letter is built.
The one word whose right drawing is known is the word in the logo. If the engine cannot redraw that,
a letter it never saw will not come out in the logo's manner either. The check answers, per letter:

  matches   the engine's letter covers the logo's (the rules were read right, the form is the logo's)
  differs   the logo draws this letter its own way: it should be taken from the logo, not built

Letters are matched one by one: each may slide sideways a little, because spacing is a separate
question from shape.
"""
from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from . import geometry as gm
from .letters import LETTERS
from .measure import Line
from .style import Style
from .typeset import flat, typeset

GOOD, FAIR = 0.75, 0.55          # share of common ink over all ink (IoU)
# a letter that does not match fully may be taken from the logo: what the logo draws of a letter it
# contains is the designer's own drawing. Whether the cut is kept is judged by check_and_take.
CUT_BELOW = GOOD
FORM_AR = {"i": "منفصل", "I": "أول", "M": "وسط", "F": "آخر"}


def _take(logo: np.ndarray, glyphs, letters, s: float, x_of: float, base_row: float, pitch: float,
          grey: Optional[np.ndarray] = None, level: float = 0.5, joins: Optional[np.ndarray] = None) -> Dict[str, Dict]:
    """Cut the letters the logo draws its own way out of the logo. The letters that matched say where
    they end: what is left of the logo's ink, near a letter that did not match, is that letter as the
    designer drew it. It is returned in the engine's units, relative to where the engine places that
    letter, so that it can stand in for the engine's letter in any word."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from skimage.measure import find_contours
    good = np.zeros_like(logo) if joins is None else joins.copy()      # a join's ink is no letter's to take
    for (p, m, dx), x in zip(glyphs, letters):
        if x["match"] >= CUT_BELOW:
            good |= m
    rest = logo & ~ndi.binary_dilation(good, iterations=max(1, int(round(0.06 * pitch))))
    bad = [(i, p, m, dx) for i, ((p, m, dx), x) in enumerate(zip(glyphs, letters)) if x["match"] < CUT_BELOW]
    if not rest.any() or not bad:
        return {}
    # every leftover piece goes to the nearest letter that did not match. A piece of a run of ink that
    # the matched letters mostly cover belongs to them (the end of a tail that reaches under the next
    # letter, say), and is left out.
    lab, n = ndi.label(rest, structure=np.ones((3, 3)))
    runs, _ = ndi.label(logo, structure=np.ones((3, 3)))
    run_area = np.bincount(runs.ravel())
    run_good = np.bincount(runs.ravel(), weights=(good & logo).ravel().astype(float), minlength=len(run_area))
    dist = np.stack([ndi.distance_transform_edt(~m) for _, _, m, _ in bad])
    # each letter that did not match owns the columns between the letters on either side of it that
    # did (a letter drawn far too wide must not reach into its neighbour's ink)
    def edge(idx: int, side: str) -> Optional[float]:
        cols = np.flatnonzero(glyphs[idx][1].any(axis=0))
        return None if not len(cols) else float(cols[0] if side == "left" else cols[-1])
    matched = [x["match"] >= CUT_BELOW for x in letters]
    centre = np.array([0.0] + [float(np.argwhere(lab == q)[:, 1].mean()) for q in range(1, n + 1)])
    bounds = []
    for i, p, m, dx in bad:
        right = next((edge(q, "left") for q in range(i - 1, -1, -1) if matched[q]), None)
        left = next((edge(q, "right") for q in range(i + 1, len(glyphs)) if matched[q]), None)
        bounds.append((left, right))

    def inside(b, c):
        left, right = b
        return (right is None or c <= right) and (left is None or c >= left)
    out: Dict[str, Dict] = {}
    for j, (i, p, m, dx) in enumerate(bad):
        mine = np.zeros_like(rest)
        for piece in range(1, n + 1):
            sel = lab == piece
            if sel.sum() < 4 or not inside(bounds[j], centre[piece]):
                continue
            r = np.bincount(runs[sel]).argmax()
            if run_area[r] and run_good[r] / run_area[r] >= 0.5:
                continue
            # a piece in this letter's territory alone is this letter's; where territories overlap
            # (two letters in a row that did not match), the nearest one takes it
            claim = [q for q in range(len(bad)) if inside(bounds[q], centre[piece])]
            d = dist[:, sel].mean(axis=1)
            if len(claim) == 1 or min(claim, key=lambda q: d[q]) == j:
                mine |= sel
        if mine.sum() < 0.05 * m.sum():
            continue
        # The letter that did not match was placed by a poor fit; its neighbour on the right matched,
        # so the cut is kept where the logo has it relative to that neighbour.
        if i > 0 and letters[i - 1]["match"] >= CUT_BELOW:
            dx = glyphs[i - 1][2]
        # outlines of the cut, in engine units relative to the placed letter (x = 0 at its right edge);
        # traced on the grey picture where there is one, for edges finer than a pixel
        if grey is not None:
            field, lvl = np.where(ndi.binary_dilation(mine, iterations=2), grey, 0.0), level
        else:
            field, lvl = mine.astype(float), 0.5
        polys = []
        for c in find_contours(np.pad(field, 1), lvl):
            pts = [(((col - 1) - x_of - dx) / s - p.x, (base_row - (row - 1)) / s) for row, col in c]
            if len(pts) >= 4:
                q = Polygon(pts).buffer(0)
                if not q.is_empty:
                    polys.append(q)
        if not polys:
            continue
        shape = None
        for q in polys:                                      # even-odd: an outline inside another is a counter
            shape = q if shape is None else shape.symmetric_difference(q)
        shape = shape.simplify(max(0.25 / s, 0.012 * pitch / s), preserve_topology=True)   # a quarter of a pixel, or a hundredth of a pitch
        rings = []
        for poly in getattr(shape, "geoms", [shape]):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            rings.append([[round(x, 2), round(y, 2)] for x, y in list(poly.exterior.coords)[:-1]])
            rings += [[[round(x, 2), round(y, 2)] for x, y in list(h.coords)[:-1]] for h in poly.interiors]
        if rings:
            out[f"{p.char}.{p.form}"] = {"polys": rings, "match_before": letters[i]["match"]}
    return out


def _mask(shapes, s: float, W: int, H: int, x_of: float, base_row: float) -> np.ndarray:
    """Engine shapes (upright units) on the logo's upright canvas. s: canvas pixels per unit."""
    out = np.zeros((H, W), bool)
    for sh in flat(shapes):
        part = None
        for poly in sh:
            if len(poly) < 3:
                continue
            layer = Image.new("L", (W, H), 0)
            ImageDraw.Draw(layer).polygon([(x * s + x_of, base_row - y * s) for x, y in poly], fill=255)
            a = np.array(layer) > 0
            part = a if part is None else part ^ a
        if part is not None:
            out |= part
    return out


def _shift(m: np.ndarray, dx: int) -> np.ndarray:
    if dx == 0:
        return m
    out = np.zeros_like(m)
    if dx > 0:
        out[:, dx:] = m[:, :-dx]
    else:
        out[:, :dx] = m[:, -dx:]
    return out


def _slide(lay, upright: Style, logo: np.ndarray, s: float, x_of: float, base_row: float, pitch: float):
    """Each letter (with the join it leaves) drawn on the logo's canvas and slid sideways to where it
    covers the logo best: [(placed letter, body mask, shift in pixels, mask of the join it leaves)]."""
    H, W = logo.shape
    reach = int(max(4, round(0.7 * pitch)))
    glyphs = []
    drift = 0
    for p in lay.placed:
        shapes = [[[(x + p.x, y) for x, y in poly] for poly in sh] for sh in flat(p.glyph.shapes)]
        body = _mask(shapes, s, W, H, x_of, base_row)
        if not body.any():
            continue
        # the join it leaves to the next letter is its own ink: it moves with the letter and counts in the
        # word's match, but the letter slides, and is judged, by its body (a join drawn where the logo
        # has none must not pull a letter off its own ink)
        join = _mask([gm.connector(upright, *p.join_out)], s, W, H, x_of, base_row) if p.join_out is not None else None
        g = body
        # a letter slides to where it covers the logo best; the one before it sets where to start looking.
        # If it finds little there (the letter before it was drawn too wide or too narrow and pushed it
        # away), it looks further, so one wrong letter is not charged to all that follow it.
        area = float(g.sum())
        rows = np.flatnonzero(g.any(axis=1))
        cols = np.flatnonzero(g.any(axis=0))
        gr = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
        band = logo[rows[0]:rows[-1] + 1]

        def search(lo: int, hi: int, step: int = 1):
            # score = ink covered - half the ink drawn outside the logo = 1.5 x covered - 0.5 x area
            top, at = -1e18, drift
            for dx in range(lo, hi + 1, step):
                a, b = cols[0] + dx, cols[-1] + 1 + dx
                if b <= 0 or a >= W:
                    score = -0.5 * area
                else:
                    ca, cb = max(a, 0), min(b, W)
                    score = 1.5 * float(np.count_nonzero(gr[:, ca - a:cb - a] & band[:, ca:cb])) - 0.5 * area
                if score > top:
                    top, at = score, dx
            return top, at
        best, best_dx = search(drift - reach, drift + reach)
        if best < 0.45 * area:
            wide = int(4 * pitch)
            b2, d2 = search(drift - wide, drift + wide, 2)
            if b2 > best:
                best, best_dx = search(d2 - 2, d2 + 2)
        drift = best_dx
        glyphs.append((p, _shift(body, best_dx), best_dx, _shift(join, best_dx) if join is not None else None))
    return glyphs


def self_check(L: Line, text: str, st: Style, k: float) -> Optional[Dict[str, Any]]:
    """Redraw `text` (the word in the logo) in the measured style and match it on the logo's line.
    Returns None when the text cannot be drawn. Scores are IoU: common ink over all ink."""
    text = "".join(c for c in text if c in LETTERS or c.isspace()).strip()
    if not text or not L.stems:
        return None
    upright = Style(**{**st.__dict__, "slant": 0.0})
    lay = typeset(text, upright)
    if not lay.placed:
        return None
    logo = np.asarray(L.B, bool)
    H, W = logo.shape
    s = 1.0 / k                                              # working pixels per unit
    base_row = L.row_of_y(0.0)
    # first placement: the right ends of the ink agree
    xs = [p[0] for sh in lay.shapes for poly in sh for p in poly]
    cols = np.flatnonzero(logo.any(axis=0))
    x_of = float(cols[-1]) - max(xs) * s
    pitch = st.pitch * s
    slid = _slide(lay, upright, logo, s, x_of, base_row, pitch)
    glyphs = [(p, m, dx) for p, m, dx, _ in slid]
    joins = np.zeros_like(logo)
    for *_, j in slid:
        if j is not None:
            joins |= j
    if not glyphs:
        return None
    union = joins.copy()
    for _, m, _ in glyphs:
        union |= m
    # every pixel of the logo belongs to the nearest letter (so a letter is not blamed for its neighbour)
    dist = np.stack([ndi.distance_transform_edt(~m) for _, m, _ in glyphs])
    owner = dist.argmin(axis=0)
    near = dist.min(axis=0) <= 0.6 * pitch
    letters = []
    for i, (p, m, dx) in enumerate(glyphs):
        mine = logo & (owner == i) & near
        inter, uni = float((m & mine).sum()), float((m | mine).sum())
        iou = inter / uni if uni else 0.0
        letters.append({"letter": p.char, "form": p.form, "match": round(iou, 3), "shift": round(dx * k, 1),
                        "verdict": "يطابق" if iou >= GOOD else ("قريب" if iou >= FAIR else "يختلف")})
    overall = float((union & logo).sum()) / max(1.0, float((union | logo).sum()))
    differ = [x for x in letters if x["verdict"] == "يختلف"]
    taken = _take(logo, glyphs, letters, s, x_of, base_row, pitch, np.asarray(L.G, float), float(L.level), joins) if any(x["match"] < CUT_BELOW for x in letters) else {}
    notes = []
    if differ:
        notes.append("الشعار يرسم " + "، ".join(f"«{x['letter']}»" for x in differ)
                     + " بطريقته الخاصة لا بصيغة المحرك: الأصحّ أخذ الحرف من الشعار نفسه.")
    spread = [x["shift"] for x in letters]
    if len(spread) > 1 and (max(spread) - min(spread)) > 0.5 * st.pitch:
        notes.append(f"المسافات بين حروف الشعار تختلف عن إيقاع المحرك بما يصل إلى {max(spread) - min(spread):.0f}: "
                     "الشعار يباعد بين الحروف أو يقارب بغير قاعدة البياض الواحد.")
    rgb = np.full((H, W, 3), 255, np.uint8)
    rgb[logo & ~union] = (70, 110, 220); rgb[union & ~logo] = (220, 70, 60); rgb[union & logo] = (40, 40, 60)
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG")
    if taken:
        notes.append("أُخذ من الشعار نفسه: " + "، ".join(f"«{k.split('.')[0]}» ({FORM_AR[k.split('.')[1]]})" for k in taken)
                     + ". يُستعمل في كل كلمة جديدة يقع فيها الحرف في الموضع نفسه.")
    return {"text": text, "overall": round(overall, 3), "letters": letters, "notes": notes, "taken": taken,
            "verdict": "يطابق" if overall >= GOOD else ("قريب" if overall >= FAIR else "يختلف"), "overlay_png": buf.getvalue()}


STRETCH_GRID = (0.0, 0.08, 0.16, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
GAP_GRID = (-0.3, -0.2, -0.1)          # a letter that only leaves white after it may leave less
SLIDE_COST = 0.02


class _Trial:
    """The logo's own word, redrawn under trial rules and laid over the logo, scored as one number:
    how well the whole word covers the logo, less how far its letters must slide to their places.
    Shared by the fits below. `hide`: the index of a letter whose ink is left out of the score (the
    held-out examination: rules measured on the other letters, judged on this one)."""

    def __init__(self, L: Line, text: str, st: Style, k: float, hide: Optional[int] = None):
        self.text = "".join(c for c in text if c in LETTERS or c.isspace()).strip()
        self.ok = bool(self.text) and bool(L.stems)
        if not self.ok:
            return
        self.up = Style(**{**st.__dict__, "slant": 0.0})
        self.cache: Dict = {}
        first = typeset(self.text, self.up, cache=self.cache)
        self.ok = bool(first.placed)
        if not self.ok:
            return
        self.placed = [(p.char, p.form) for p in first.placed]
        logo = np.asarray(L.B, bool)
        f = 2 if logo.shape[1] > 600 else 1                     # half the resolution is plenty to judge rules
        if f > 1:
            H2, W2 = logo.shape[0] // f, logo.shape[1] // f
            logo = logo[:H2 * f, :W2 * f].reshape(H2, f, W2, f).mean(axis=(1, 3)) >= 0.5
        self.logo, self.s = logo, 1.0 / (k * f)
        self.base_row = L.row_of_y(0.0) / f
        xs = [p[0] for sh in first.shapes for poly in sh for p in poly]
        self.x_of = float(np.flatnonzero(logo.any(axis=0))[-1]) - max(xs) * self.s
        self.pitch = st.pitch * self.s
        self.X = st.x_height
        self.hide = hide
        self.count = np.ones_like(logo)                          # the pixels that count in the score
        if hide is not None:
            slid = _slide(first, self.up, logo, self.s, self.x_of, self.base_row, self.pitch)
            dist = np.stack([ndi.distance_transform_edt(~m) for _, m, _, _ in slid])
            idx = [i for i, p in enumerate(first.placed) if any(p is q for q, _, _, _ in slid)]
            owner = np.array(idx)[dist.argmin(axis=0)]
            self.count = owner != hide
        self.logo_sum = float((logo & self.count).sum())

    CACHED = {"fitted", "chosen", "traits", "stretch", "params", "variants", "measured_on"}

    def layout(self, **changes):
        # the cache knows what letters read of the parameters and variants; any other change of the
        # style (the construction family) builds the letters afresh
        cache = self.cache if set(changes) <= self.CACHED else None
        return typeset(self.text, Style(**{**self.up.__dict__, **changes}), cache=cache)

    def score(self, **changes) -> float:
        return self.scores(**changes)[0]

    def scores(self, **changes) -> Tuple[float, List[float]]:
        """The word's score, and each letter's own match (its body against the logo's ink in its columns)."""
        lay = self.layout(**changes)
        union = np.zeros_like(self.logo)
        slid = _slide(lay, self.up, self.logo, self.s, self.x_of, self.base_row, self.pitch)
        moved, n = 0.0, 0
        each = []
        for i, (p, m, dx, j) in enumerate(slid):
            union |= m if j is None else (m | j)
            if i != self.hide:
                moved, n = moved + abs(dx), n + 1
            cols = np.flatnonzero(m.any(axis=0))
            if len(cols) and i != self.hide:
                band = self.logo[:, cols[0]:cols[-1] + 1] & self.count[:, cols[0]:cols[-1] + 1]
                mm = m[:, cols[0]:cols[-1] + 1]
                inter = float((mm & band).sum())
                each.append(inter / max(1.0, float(mm.sum()) + float(band.sum()) - inter))
            else:
                each.append(float("nan"))
        union &= self.count
        inter = float((union & self.logo).sum())
        iou = inter / max(1.0, float(union.sum()) + self.logo_sum - inter)
        # the slides say how far each letter stands from where the rules put it: the right rules leave
        # little to slide (a letter slid one x-height costs SLIDE_COST of match)
        return iou - SLIDE_COST * moved / max(1, n) / (self.X * self.s), each

    def reads(self, skip: Optional[int] = None) -> List[str]:
        """What the letters of the word read while they were built (parameters, and "v:" variants);
        with `skip`, only what some other letter than that one reads."""
        out: List[str] = []
        for i, (c, f) in enumerate(self.placed):
            if i == skip:
                continue
            for (cc, ff, _), (_, deps) in self.cache.items():
                if (cc, ff) == (c, f):
                    out += [d for d in deps if d not in out]
        return out


def fit_stretch(L: Line, text: str, st: Style, k: float, trial: Optional[_Trial] = None) -> Dict[str, float]:
    """How much longer the logo draws each letter than the engine builds it (its kashida).

    The logo's own word is redrawn with each letter form drawn longer by a trial length, keeping, one
    change at a time, the length under which the whole word covers the logo best. Every
    letter after a longer one moves along with it, so a letter that is lengthened into its neighbour's
    ink pushes the neighbour off its own ink, and loses. Lengths are in fractions of the x-height,
    up to four. The letters still slide to where they cover the logo best, and how far they must slide
    counts against the lengths: the right lengths put every letter where the logo has it.
    Returns {} when no lengthening helps by at least a point."""
    t = trial or _Trial(L, text, Style(**{**st.__dict__, "stretch": {}}), k)
    if not t.ok:
        return {}
    X = t.X
    keys = list(dict.fromkeys(f"{c}.{f}" for i, (c, f) in enumerate(t.placed) if i != t.hide))
    cur: Dict[str, float] = {}
    base = now = t.score(stretch=cur)
    # best first: every round tries every letter at every length and keeps the one change that helps
    # most, so a length goes to the letter that puts the most letters in place, not to the first tried
    for _ in range(2 * len(keys)):
        best = None
        for key in keys:
            grid = STRETCH_GRID + (GAP_GRID if key.split(".")[1] in "iF" else ())
            for v in grid:
                if abs(v * X - cur.get(key, 0.0)) < 1e-9:
                    continue
                sc = t.score(stretch={**cur, key: v * X})
                if sc > now + 1e-4 and (best is None or sc > best[0]):
                    best = (sc, key, v * X)
        if best is None:
            break
        now, cur = best[0], {**cur, best[1]: best[2]}
    if now < base + 0.01:
        return {}
    return {q: round(v, 2) for q, v in cur.items() if abs(v) >= 0.04 * X}


FIT_GRID = (0.5, 0.65, 0.8, 0.9, 1.1, 1.25, 1.5, 1.75, 2.0)
FIT_MIN = 0.003           # a measured rule must earn this much of the score: smaller gains are noise


def fit_letters(L: Line, text: str, st: Style, k: float, trial: Optional[_Trial] = None, own: bool = True
                ) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, float]]:
    """The rules of the letters, measured on the logo's own word: for every letter form in it, the
    construction (variant) that draws it best, then every parameter its letters read, each tried over
    its handle's range and kept where the word covers the logo best. What the designer has set is left
    alone. A parameter shared by several letters is measured on all of them at once, and serves every
    other letter that shares it (the cup measured on ن draws ل and ق).

    Each letter's construction, then (with `own`) each letter's own parameters: these serve the
    letters of the logo. Then, apart, the designer's general manner (recipes.TRAITS: how wide, how
    high, how open, how round) measured on all the letters at once: it serves letters the logo does
    not contain.

    Returns (parameters, constructions, manner) for `style.fitted`, `style.chosen`, `style.traits`."""
    from .recipes import DECLARED, PARAMS, SKELETON, TRAIT_GRID, TRAITS, VARIANTS, general_rule
    t = trial or _Trial(L, text, Style(**{**st.__dict__, "fitted": {}, "chosen": {}, "traits": {}}), k)
    if not t.ok:
        return {}, {}, {}
    reads = t.reads(skip=t.hide)
    hidden = t.placed[t.hide] if t.hide is not None else None
    # constructions: a variant key a letter reads, with more than one option, the designer has not set
    options: Dict[str, List[str]] = {}
    for c, f in dict.fromkeys(t.placed):
        if (c, f) == hidden:
            continue
        for letter in dict.fromkeys((c, SKELETON.get(c, c))):
            key = f"{letter}.{f}"
            names = [v.name for v in VARIANTS.get(key, [])] or [n for n, _ in DECLARED.get(key, ([], None))[0]]
            if len(names) > 1 and key not in st.variants and letter not in st.variants and ("v:" + key) in reads:
                options[key] = names
    params = [d for d in reads if d in PARAMS and d not in st.params]
    fitted: Dict[str, float] = {}
    chosen: Dict[str, str] = {}
    now = t.score()
    for key, names in options.items():
        best = None
        for name in names:
            sc = t.score(chosen={**chosen, key: name})
            if best is None or sc > best[0]:
                best = (sc, name)
        if best[0] > now + FIT_MIN:
            now, chosen = best[0], {**chosen, key: best[1]}
    # then each letter's own rules
    for _ in range(2 if own else 0):
        grew = False
        for name in params:
            p = PARAMS[name]
            rule = general_rule(t.up, name)
            span = (p.lo * rule, p.hi * rule) if rule > 0 else (0.0, 0.5 * t.X)
            grid = [rule * m for m in FIT_GRID if span[0] - 1e-9 <= rule * m <= span[1] + 1e-9] if rule > 0 \
                else [span[1] * m for m in (0.1, 0.25, 0.5, 0.75, 1.0)]
            best = None
            for v in grid:
                if abs(v - fitted.get(name, rule)) < 1e-9:
                    continue
                sc = t.score(fitted={**fitted, name: v}, chosen=chosen)
                if best is None or sc > best[0]:
                    best = (sc, v)
            if best is not None and best[0] > now + FIT_MIN:
                now, fitted, grew = best[0], {**fitted, name: round(best[1], 3)}, True
        if not grew:
            break
    # then, apart, the general manner on all the letters at once (with their constructions, not their
    # own measures): it is what reaches letters the logo does not contain
    t.up = Style(**{**t.up.__dict__, "chosen": chosen, "fitted": {}})
    now, _ = t.scores()
    # A manner is what the designer repeats: a trait is kept only where it draws most of the letters
    # better and none worse (one letter drawn long is a kashida, not a manner).
    traits: Dict[str, float] = {}
    kinds = [q for q, (_, names) in TRAITS.items() if any(n in params for n in names)]
    _, each_now = t.scores()
    for _ in range(2):
        grew = False
        for q in kinds:
            best = None
            for f in TRAIT_GRID + (1.0,):
                if abs(f - traits.get(q, 1.0)) < 1e-9:
                    continue
                sc, each = t.scores(traits={**traits, q: f})
                up = sum(1 for a, b in zip(each_now, each) if b > a + 0.02)
                down = sum(1 for a, b in zip(each_now, each) if b < a - 0.02)
                # a manner shows in most of the letters, not in two of them
                if up >= max(2, (len(each) + 1) // 2) and down == 0 and (best is None or sc > best[0]):
                    best = (sc, f, each)
            if best is not None and best[0] > now + FIT_MIN:
                now, traits, grew, each_now = best[0], {**traits, q: best[1]}, True, best[2]
        if not grew:
            break
    traits = {q: f for q, f in traits.items() if abs(f - 1.0) > 1e-9}
    return fitted, chosen, traits


def check_and_take(L: Line, text: str, st: Style, k: float, take: bool = True) -> Optional[Dict[str, Any]]:
    """The self-check, and when the logo draws some letters its own way, the same check again with
    those letters cut out of the logo: `st.taken` is filled, and the result says how far it helped.
    With take=False (the logo's word is only a guess) nothing is cut: a wrong reading would cut a
    wrong letter."""
    if take:
        from .typeset import shape_forms
        st.seen = " ".join(dict.fromkeys(f"{c}.{f}" for c, f in shape_forms(text) if f in "iIMF"))
    first = self_check(L, text, st, k)
    if first and take:
        first = _measure_rules(L, text, st, k, first)
    if first and not take and first.get("taken"):
        first["taken"] = {}
        first["notes"] = [n for n in first["notes"] if not n.startswith("أُخذ من الشعار")]
        first["notes"].append("لم يُؤخذ شيء من الشعار: الكلمة مقروءة بالرسم لا مكتوبة، أو قياس الصورة غير موثوق.")
    if not first or not first.get("taken"):
        return _lengthen(L, text, st, k, first, None) if first and take else first
    # A cut is kept only if it earns its place: the logo's word matches better with it, and no letter
    # that matched before matches worse (a cut that swallowed its neighbours fails this).
    # Each cut is judged against the word as it stands with the cuts kept so far, and kept when the word
    # matches clearly better, or when its own letter now matches far better and the word no worse.
    # Twice over: a cut can earn its place only once another has put its neighbours right.
    cur = [(x["letter"], x["form"], x["match"], x["shift"]) for x in first["letters"]]
    now = first["overall"]
    kept = {}
    for _ in range(2):
        grew = False
        for key, cut in first["taken"].items():
            if key in kept:
                continue
            trial = Style(**{**st.__dict__, "taken": {**kept, key: cut}})
            r = self_check(L, text, trial, k)
            if not r or len(r["letters"]) != len(cur):
                continue
            # A letter the cut displaced is harmed if it matches worse. A letter left where it was (the
            # same drawing in the same place) only changes score because the real letter now stands
            # beside it and takes back its own ink: that is not harm.
            hurt = any(m0 >= FAIR and x["match"] < m0 - 0.08 and abs(x["shift"] - s0) > 0.15 * st.pitch
                       for (c0, f0, m0, s0), x in zip(cur, r["letters"]) if f"{c0}.{f0}" != key)
            own = [x["match"] - m0 for (c0, f0, m0, _), x in zip(cur, r["letters"]) if f"{c0}.{f0}" == key]
            gain = sum(own) / len(own) if own else 0.0
            if not hurt and (r["overall"] > now + 0.02 or (gain >= 0.2 and r["overall"] >= now - 0.005)):
                kept[key], now, grew = cut, r["overall"], True
                cur = [(x["letter"], x["form"], x["match"], x["shift"]) for x in r["letters"]]
        if not grew:
            break
    if not kept:
        first["taken"] = {}
        first["notes"] = [n for n in first["notes"] if not n.startswith("أُخذ من الشعار")]
        first["notes"].append("حاول المحرك أخذ ما يختلف من الشعار نفسه فلم يتحسن التطابق: الحروف هنا بعيدة عن صيغه، فبقيت كما يبنيها.")
        return _lengthen(L, text, st, k, first, None)
    if len(kept) < len(first["taken"]):
        dropped = [q.split(".")[0] for q in first["taken"] if q not in kept]
        first["notes"].append("لم يُؤخذ من الشعار: " + "، ".join(f"«{q}»" for q in dropped) + " (القصّ لم يحسّن التطابق).")
        first["notes"] = [n if not n.startswith("أُخذ من الشعار") else
                          "أُخذ من الشعار نفسه: " + "، ".join(f"«{q.split('.')[0]}» ({FORM_AR[q.split('.')[1]]})" for q in kept)
                          + ". يُستعمل في كل كلمة جديدة يقع فيها الحرف في الموضع نفسه." for n in first["notes"]]
    first["taken"] = kept
    st.taken = dict(kept)
    again = self_check(L, text, st, k)
    if again:
        first["overall_after"] = again["overall"]
        first["letters_after"] = [{"letter": x["letter"], "form": x["form"], "match": x["match"]} for x in again["letters"]]
        first["overlay_after_png"] = again["overlay_png"]
    return _lengthen(L, text, st, k, first, again) if take else first


def _measure_rules(L: Line, text: str, st: Style, k: float, first: Dict[str, Any]) -> Dict[str, Any]:
    """The rules of the letters read off the logo's own word (fit_letters), kept when the word then
    matches the logo better. Letters are cut from the logo only where no rule draws them."""
    if st.fitted or st.chosen or st.traits:
        return first
    fitted, chosen, traits = fit_letters(L, text, st, k)
    if not fitted and not chosen and not traits:
        return first
    from .recipes import SKELETON
    on = "".join(sorted({SKELETON.get(c, c) for c in text if c in LETTERS}))
    trial = Style(**{**st.__dict__, "fitted": fitted, "chosen": chosen, "traits": traits, "measured_on": on})
    r = self_check(L, text, trial, k)
    if not r or r["overall"] < first["overall"] + 0.01:
        return first
    st.fitted, st.chosen, st.traits, st.measured_on = fitted, chosen, traits, on
    r["overall_rules"] = first["overall"]           # before the rules were measured on the logo
    r["fitted"], r["chosen"], r["traits"] = dict(fitted), dict(chosen), dict(traits)
    from .recipes import general_rule
    changed = sorted(fitted, key=lambda n: -abs(fitted[n] / max(1e-9, general_rule(trial, n)) - 1))[:6]
    if traits:
        r["notes"].insert(0, "أسلوب المصمم العام كما قيس من حروف الشعار كلها، ويُطبَّق على كل حرف ولو لم يكن في الشعار: "
                          + "، ".join(f"{q} × {f:.2f}" for q, f in traits.items()) + ".")
    r["notes"].insert(0, f"قيس من حروف الشعار {len(fitted)} معاملًا" + (f" و{len(chosen)} صيغة" if chosen else "")
                      + f" بدل قواعد المحرك العامة، فارتفع التطابق من {first['overall'] * 100:.0f}% إلى {r['overall'] * 100:.0f}%. أبرزها: "
                      + "، ".join(f"{n} × {fitted[n] / max(1e-9, general_rule(trial, n)):.2f}" for n in changed)
                      + (" — " + "، ".join(f"«{q}» {v}" for q, v in chosen.items()) if chosen else "") + ".")
    return r


def _lengthen(L: Line, text: str, st: Style, k: float, first: Dict[str, Any], last: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """After the letters the logo draws its own way are taken from it, how long it draws the others:
    a letter drawn too short is not a letter drawn wrong. The lengths are kept only when the word
    then matches the logo better."""
    if st.stretch:
        return first
    fitted = fit_stretch(L, text, st, k)
    if not fitted:
        return first
    before = (last or first)["overall"]
    trial = Style(**{**st.__dict__, "stretch": fitted})
    r = self_check(L, text, trial, k)
    slid = lambda c: sum(abs(x["shift"]) for x in c["letters"]) / max(1, len(c["letters"]))
    # kept when the word matches clearly better, or as well while the letters stand where the logo has
    # them instead of being slid there
    if not r or not (r["overall"] >= before + 0.02 or
                     (r["overall"] >= before - 0.002 and slid(r) <= 0.7 * slid(last or first) - 0.1 * st.x_height)):
        return first
    st.stretch = fitted
    first["overall_after"] = r["overall"]
    first["letters_after"] = [{"letter": x["letter"], "form": x["form"], "match": x["match"], "shift": x["shift"]} for x in r["letters"]]
    first["overlay_after_png"] = r["overlay_png"]
    first["stretch"] = dict(fitted)
    longer = [f"«{q.split('.')[0]}» ({FORM_AR[q.split('.')[1]]}) {v / st.x_height:.2f}" for q, v in fitted.items() if v > 0]
    if longer:
        first["notes"].append("الشعار يمدّ " + "، ".join(longer) + " من ارتفاع الحروف أطول مما يبنيه المحرك: مدّها مثله فارتفع التطابق من "
                              f"{before * 100:.0f}% إلى {r['overall'] * 100:.0f}%. يُستعمل المدّ نفسه في كل كلمة جديدة.")
    return first


def prune_parts(lib, L: Line, text: str, st: Style, k: float) -> List[str]:
    """The pieces cut from the logo (the parts box) must earn their place like everything else: each
    is left out in turn, and dropped if the logo's own word then matches better. Returns what was
    dropped (and removes it from `lib`)."""
    from .library import ACTIVE

    def score() -> float:
        tok = ACTIVE.set(lib)
        try:
            r = self_check(L, text, st, k)
        finally:
            ACTIVE.reset(tok)
        return r["overall"] if r else 0.0
    if not text:
        return []
    now, dropped = score(), []
    for key in list(lib.parts):
        part = lib.parts.pop(key)
        sc = score()
        if sc > now + 0.01:
            now = sc
            dropped.append(key)
        else:
            lib.parts[key] = part
    return dropped
