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

from .letters import LETTERS
from .measure import Line
from .style import Style
from .typeset import flat, typeset

GOOD, FAIR = 0.75, 0.55          # share of common ink over all ink (IoU)
FORM_AR = {"i": "منفصل", "I": "أول", "M": "وسط", "F": "آخر"}


def _take(logo: np.ndarray, glyphs, letters, s: float, x_of: float, base_row: float, pitch: float,
          grey: Optional[np.ndarray] = None, level: float = 0.5) -> Dict[str, Dict]:
    """Cut the letters the logo draws its own way out of the logo. The letters that matched say where
    they end: what is left of the logo's ink, near a letter that did not match, is that letter as the
    designer drew it. It is returned in the engine's units, relative to where the engine places that
    letter, so that it can stand in for the engine's letter in any word."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from skimage.measure import find_contours
    good = np.zeros_like(logo)
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
    reach = int(max(4, round(0.7 * pitch)))
    glyphs = []
    drift = 0
    for p in lay.placed:
        shapes = [[[(x + p.x, y) for x, y in poly] for poly in sh] for sh in flat(p.glyph.shapes)]
        g = _mask(shapes, s, W, H, x_of, base_row)
        if not g.any():
            continue
        # a letter slides to where it covers the logo best; the one before it sets where to start looking.
        # If it finds little there (the letter before it was drawn too wide or too narrow and pushed it
        # away), it looks further, so one wrong letter is not charged to all that follow it.
        area = float(g.sum())

        def search(lo: int, hi: int, step: int = 1):
            top, at = -1e18, drift
            for dx in range(lo, hi + 1, step):
                m = _shift(g, dx)
                score = float((m & logo).sum()) - 0.5 * float((m & ~logo).sum())
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
        glyphs.append((p, _shift(g, best_dx), best_dx))
    if not glyphs:
        return None
    union = np.zeros_like(logo)
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
    taken = _take(logo, glyphs, letters, s, x_of, base_row, pitch, np.asarray(L.G, float), float(L.level)) if differ else {}
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


def check_and_take(L: Line, text: str, st: Style, k: float, take: bool = True) -> Optional[Dict[str, Any]]:
    """The self-check, and when the logo draws some letters its own way, the same check again with
    those letters cut out of the logo: `st.taken` is filled, and the result says how far it helped.
    With take=False (the logo's word is only a guess) nothing is cut: a wrong reading would cut a
    wrong letter."""
    first = self_check(L, text, st, k)
    if first and not take and first.get("taken"):
        first["taken"] = {}
        first["notes"] = [n for n in first["notes"] if not n.startswith("أُخذ من الشعار")]
        first["notes"].append("الكلمة مقروءة بالرسم لا مكتوبة: لم يُؤخذ شيء من الشعار. اكتب نص الشعار ليُؤخذ منه ما يرسمه بطريقته.")
    if not first or not first.get("taken"):
        return first
    # A cut is kept only if it earns its place: the logo's word matches better with it, and no letter
    # that matched before matches worse (a cut that swallowed its neighbours fails this).
    before = {(x["letter"], x["form"]): x["match"] for x in first["letters"]}
    kept = {}
    for key, cut in first["taken"].items():
        trial = Style(**{**st.__dict__, "taken": {**kept, key: cut}})
        r = self_check(L, text, trial, k)
        if not r:
            continue
        after = {(x["letter"], x["form"]): x["match"] for x in r["letters"]}
        hurt = any(before[q] >= FAIR and after.get(q, 0.0) < before[q] - 0.08 for q in before if q != tuple(key.split(".")))
        if r["overall"] > first["overall"] + 0.02 and not hurt:
            kept[key] = cut
    if not kept:
        first["taken"] = {}
        first["notes"] = [n for n in first["notes"] if not n.startswith("أُخذ من الشعار")]
        first["notes"].append("حاول المحرك أخذ ما يختلف من الشعار نفسه فلم يتحسن التطابق: الحروف هنا بعيدة عن صيغه، فبقيت كما يبنيها.")
        return first
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
    return first
