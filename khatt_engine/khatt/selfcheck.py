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
    return {"text": text, "overall": round(overall, 3), "letters": letters, "notes": notes,
            "verdict": "يطابق" if overall >= GOOD else ("قريب" if overall >= FAIR else "يختلف"), "overlay_png": buf.getvalue()}
