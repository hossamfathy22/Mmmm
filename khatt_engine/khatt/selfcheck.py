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
        if x["verdict"] != "يختلف":
            good |= m
    rest = logo & ~ndi.binary_dilation(good, iterations=max(1, int(round(0.06 * pitch))))
    bad = [(i, p, m, dx) for i, ((p, m, dx), x) in enumerate(zip(glyphs, letters)) if x["verdict"] == "يختلف"]
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
    matched = [x["verdict"] != "يختلف" for x in letters]
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
        if i > 0 and letters[i - 1]["verdict"] != "يختلف":
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
    taken = _take(logo, glyphs, letters, s, x_of, base_row, pitch, np.asarray(L.G, float), float(L.level), joins) if differ else {}
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


def fit_stretch(L: Line, text: str, st: Style, k: float) -> Dict[str, float]:
    """How much longer the logo draws each letter than the engine builds it (its kashida).

    The logo's own word is redrawn with each letter form drawn longer by a trial length, keeping, one
    change at a time, the length under which the whole word covers the logo best. Every
    letter after a longer one moves along with it, so a letter that is lengthened into its neighbour's
    ink pushes the neighbour off its own ink, and loses. Lengths are in fractions of the x-height,
    up to four. The letters still slide to where they cover the logo best, and how far they must slide
    counts against the lengths: the right lengths put every letter where the logo has it.
    Returns {} when no lengthening helps by at least a point."""
    text = "".join(c for c in text if c in LETTERS or c.isspace()).strip()
    if not text or not L.stems:
        return {}
    upright = Style(**{**st.__dict__, "slant": 0.0, "stretch": {}})
    first = typeset(text, upright)
    if not first.placed:
        return {}
    logo = np.asarray(L.B, bool)
    f = 2 if logo.shape[1] > 600 else 1                     # half the resolution is plenty to judge lengths
    if f > 1:
        H2, W2 = logo.shape[0] // f, logo.shape[1] // f
        logo = logo[:H2 * f, :W2 * f].reshape(H2, f, W2, f).mean(axis=(1, 3)) >= 0.5
    H, W = logo.shape
    s = 1.0 / (k * f)
    base_row = L.row_of_y(0.0) / f
    xs = [p[0] for sh in first.shapes for poly in sh for p in poly]
    right = float(np.flatnonzero(logo.any(axis=0))[-1])
    x_of = right - max(xs) * s
    logo_sum = float(logo.sum())

    pitch = st.pitch * s
    X = st.x_height

    built: Dict = {}                                       # the letters themselves do not change, only their lengths

    def score(stretch: Dict[str, float]) -> float:
        lay = typeset(text, Style(**{**upright.__dict__, "stretch": stretch}), cache=built)
        union = np.zeros_like(logo)
        slid = _slide(lay, upright, logo, s, x_of, base_row, pitch)
        for _, m, _, j in slid:
            union |= m if j is None else (m | j)
        inter = float((union & logo).sum())
        iou = inter / max(1.0, float(union.sum()) + logo_sum - inter)
        # the slides say how far each letter stands from where the lengths put it: the right lengths
        # leave little to slide (a letter slid one x-height costs SLIDE_COST of match)
        moved = sum(abs(dx) for _, _, dx, _ in slid) / max(1, len(slid)) / (X * s)
        return iou - SLIDE_COST * moved
    keys = list(dict.fromkeys(f"{p.char}.{p.form}" for p in first.placed))
    cur: Dict[str, float] = {}
    base = now = score(cur)
    # best first: every round tries every letter at every length and keeps the one change that helps
    # most, so a length goes to the letter that puts the most letters in place, not to the first tried
    for _ in range(2 * len(keys)):
        best = None
        for key in keys:
            grid = STRETCH_GRID + (GAP_GRID if key.split(".")[1] in "iF" else ())
            for v in grid:
                if abs(v * X - cur.get(key, 0.0)) < 1e-9:
                    continue
                sc = score({**cur, key: v * X})
                if sc > now + 1e-4 and (best is None or sc > best[0]):
                    best = (sc, key, v * X)
        if best is None:
            break
        now, cur = best[0], {**cur, best[1]: best[2]}
    if now < base + 0.01:
        return {}
    return {q: round(v, 2) for q, v in cur.items() if abs(v) >= 0.04 * X}


def check_and_take(L: Line, text: str, st: Style, k: float, take: bool = True) -> Optional[Dict[str, Any]]:
    """The self-check, and when the logo draws some letters its own way, the same check again with
    those letters cut out of the logo: `st.taken` is filled, and the result says how far it helped.
    With take=False (the logo's word is only a guess) nothing is cut: a wrong reading would cut a
    wrong letter."""
    first = self_check(L, text, st, k)
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
