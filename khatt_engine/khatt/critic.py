"""The critic: the defects a designer sees when looking at a drawn letter, turned into measurements.

When the hand-made letters were drawn, each one was rendered, looked at and corrected. The
corrections were all of a few kinds, and each kind can be measured on the drawing:

  crack      a thin white seam where two parts of a letter fail to meet
  blob       a pool of ink thicker than the pen, where parts pile up
  thin       a stroke thinner than the hairline pen
  grid       a part above the ascender line or below the descender line
  counter    an enclosed white space narrower than the white between teeth
  confusable two different letters drawn so alike that one reads as the other
  touch      a dot, dash or hamza touching the letter it belongs to

`critique(layout)` returns the defects of one drawing; `specimen_defects(style)` looks at every
letter in every form, and at every pair of letter skeletons, for one style.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw
from scipy import ndimage as ndi

from .style import Style
from .typeset import Layout, typeset

MARK_NAMES = {"نقطة", "شرطة", "همزة"}
NAMES = {"crack": "شق أبيض بين جزأين", "blob": "تكدّس حبر أثقل من القلم", "thin": "خط أرفع من القلم الشعري",
         "grid": "جزء خارج خطوط الشبكة", "counter": "فراغ داخلي أضيق من الأبيض", "confusable": "يُقرأ كحرف آخر",
         "touch": "النقطة أو العلامة تلمس الحرف"}


@dataclass
class Defect:
    kind: str
    where: str
    value: float
    limit: float

    def text(self) -> str:
        return f"{self.where}: {NAMES[self.kind]} ({self.value:.2f} مقابل الحد {self.limit:.2f})"


def raster(lay: Layout, px_per_stem: float = 24.0, marks: bool = False, frame_marks: bool = True) -> Tuple[np.ndarray, float, float, float]:
    """Upright (unslanted) drawing of the letter bodies (and marks if asked) on a frame that holds
    the whole drawing, marks included, so two drawings of one layout line up; returns
    (mask, scale, x0, y_top)."""
    st = lay.style
    s = px_per_stem / st.stem
    every = [p for sh in lay.shapes for poly in sh for p in poly] if frame_marks else None
    from .render import merged
    body_only = Layout(text=lay.text, style=lay.style,
                       shapes=[sh for sh in lay.shapes if marks or getattr(sh, "name", "") not in MARK_NAMES])
    shapes = merged(body_only)            # judged as it is printed: parts unioned
    pts = every or [p for sh in shapes for poly in sh for p in poly]
    if not pts or not shapes:
        return np.zeros((1, 1), bool), s, 0.0, 0.0
    pad = 2 * st.stem
    x0, x1 = min(p[0] for p in pts) - pad, max(p[0] for p in pts) + pad
    y0, y1 = min(p[1] for p in pts) - pad, max(p[1] for p in pts) + pad
    W, H = int((x1 - x0) * s) + 1, int((y1 - y0) * s) + 1
    mask = Image.new("1", (W, H), 0)
    for sh in shapes:
        polys = [[((x - x0) * s, (y1 - y) * s) for x, y in poly] for poly in sh if len(poly) >= 3]
        part = None
        for p in polys:
            layer = Image.new("1", (W, H), 0)
            ImageDraw.Draw(layer).polygon(p, fill=1)
            part = layer if part is None else ImageChops.logical_xor(part, layer)
        if part is not None:
            mask = ImageChops.logical_or(mask, part)
    return np.ascontiguousarray(np.array(mask, dtype=np.uint8) > 0), s, x0, y1


def critique(lay: Layout, where: str = "") -> List[Defect]:
    st = lay.style
    body, s, x0, ytop = raster(lay)
    if not body.any():
        return []
    out: List[Defect] = []
    U, h = st.stem * s, st.hair * s
    # crack: a seam the pen would never leave, long but thinner than half the hairline. In a pen of
    # one weight the "hairline" is as thick as the stem, and a white that wide is an ordinary inner
    # corner, not a seam: there the seam is judged against a third of the stem.
    r = max(1, int(round(0.35 * min(h, 0.3 * U))))
    closed = ndi.binary_closing(body, structure=np.ones((2 * r + 1, 2 * r + 1)), iterations=1)
    seams = closed & ~body
    lab, n = ndi.label(seams, structure=np.ones((3, 3)))
    outside = ~closed
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        length = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if length < 0.45 * U:
            continue
        m = lab == i
        # a crack has ink on both sides and opens only at its ends; a shallow dent in an outer edge
        # opens to the outside along its whole length, and is not a seam between two parts
        open_side = float((ndi.binary_dilation(m) & outside).sum()) / max(1.0, float(m.sum()))
        if open_side < 0.3:
            out.append(Defect("crack", where, length / U, 0.45))
    # blob: ink thicker than the pen, where pieces are piled on each other
    # In a contrast pen a stem meets hairlines, and anything much thicker than the stem is a pile-up.
    # In a pen of one weight two full strokes meet at every join, and a join carries up to 1.75 pens.
    mono = float(np.clip((st.hair / st.stem - 0.5) / 0.5, 0.0, 1.0)) if st.stem else 0.0
    blob_limit = 1.32 + 0.43 * mono
    dist = ndi.distance_transform_edt(body)
    thick = 2 * float(dist.max())
    if thick > blob_limit * U:
        out.append(Defect("blob", where, thick / U, blob_limit))
    # thin: strokes thinner than the hairline (5th percentile of the centre-line widths)
    from skimage.morphology import skeletonize
    sk = skeletonize(body)
    # the last stretch of each branch runs into a corner of a cut end, where the ink is thin by
    # geometry, not by pen: prune it before measuring
    for _ in range(max(1, int(0.6 * U))):
        nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant")
        ends = sk & (nb <= 2)
        if not ends.any():
            break
        sk = sk & ~ends
    w = 2 * dist[sk]
    if len(w) > 20:
        w5 = float(np.percentile(w, 5))
        thin_limit = 0.6 - 0.1 * mono        # an angled cut at the top of a one-weight stem ends in a tip
        if w5 < thin_limit * h:
            out.append(Defect("thin", where, w5 / max(h, 1e-6), thin_limit))
    # grid: nothing above the ascender line or below the descender line (dots excluded)
    rows = np.flatnonzero(body.any(axis=1))
    top_y = ytop - rows[0] / s
    bot_y = ytop - rows[-1] / s
    tol = 0.12 * st.x_height
    if top_y > st.x_height + st.ascender + tol:
        out.append(Defect("grid", where, top_y / st.x_height, (st.x_height + st.ascender + tol) / st.x_height))
    if bot_y < -st.descender - tol:
        out.append(Defect("grid", where, -bot_y / st.x_height, (st.descender + tol) / st.x_height))
    # counters: every enclosed white space is at least 0.55 white wide at its widest row
    holes = ndi.binary_fill_holes(body) & ~body
    lab, n = ndi.label(holes)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        if m.sum() < (0.3 * U) ** 2:
            continue                                   # a pinhole: reported as a crack if it matters
        width = float(m.sum(axis=1).max())
        if width < 0.55 * st.white * s:
            out.append(Defect("counter", where, width / (st.white * s), 0.55))
    # touch: marks keep their own white (a fifth of the stem) from the letter
    allm, *_ = raster(lay, marks=True)
    if allm.shape == body.shape:
        only = allm & ~body
        if only.sum() > 0:
            near = ndi.binary_dilation(body, iterations=max(1, int(round(0.2 * U))))
            lab, n = ndi.label(only, structure=np.ones((3, 3)))
            for i in range(1, n + 1):
                m = lab == i
                if m.sum() > 0.05 * U * U and (m & near).any():
                    out.append(Defect("touch", where, float((m & near).sum() / m.sum()), 0.0))
                    break
    return out


# ---------------------------------------------------------------- the whole alphabet
FORMS = {"i": "{}", "I": "{}ـ", "M": "ـ{}ـ", "F": "ـ{}"}
FORM_NAMES = {"i": "منفصل", "I": "أول", "M": "وسط", "F": "آخر"}
# one letter per skeleton: letters that differ only by their dots share a skeleton
SKELETONS = "ا ب ح د ر س ص ط ع ف ق ك ل م ن ه و ی".split()
SAME = [set("بتثپن"), set("حجخچ"), set("دذ"), set("رزژ"), set("سش"), set("صض"), set("طظ"), set("عغ"),
        set("فڤق"), set("كکگل"), set("یيى")]
JOINING = "بحسصطعفقكلمنهی"
SAME_JOINED = [set("بنی")]        # initial and medial ب ن ی are one tooth; only their dots differ


def specimen_defects(st: Style, letters: Optional[str] = None) -> List[Defect]:
    from .letters import LETTERS
    out: List[Defect] = []
    for c in (letters or "".join(k for k in LETTERS if k not in "ـء")):
        forms = "iIMF" if LETTERS[c][0] == "D" else "iF"
        for f in forms:
            lay = typeset(FORMS[f].format(c), st)
            out += critique(lay, f"{c} {FORM_NAMES[f]}")
    out += confusable(st)
    return out


def _norm(lay: Layout) -> np.ndarray:
    """The letter body on a fixed frame: baseline and right edge aligned, true proportions kept
    (a tall ل and a short ب must stay different)."""
    st = lay.style
    m, s, x0, ytop = raster(lay, px_per_stem=10, frame_marks=False)
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    H = int((st.x_height + st.ascender + st.descender) * s) + 2
    W = int(6 * (st.stem + st.white) * s)
    out = np.zeros((H, W), bool)
    if not len(rows):
        return out
    base_row = int(round(ytop * s))                 # row of y = 0 in the raster
    top_row = base_row - int((st.x_height + st.ascender) * s)
    right = cols[-1]
    for r in rows:
        rr = r - top_row
        if 0 <= rr < H:
            seg = m[r, max(0, right - W + 1):right + 1]
            out[rr, W - len(seg):] = seg
    return out


def confusable(st: Style, limit: float = 0.86) -> List[Defect]:
    """Two letters whose skeletons should differ must not overlap more than `limit` (IoU)."""
    out: List[Defect] = []
    for f in ("i", "M"):
        shapes = {}
        for c in SKELETONS:
            if f == "M" and c not in JOINING:
                continue
            lay = typeset(FORMS[f].format(c), st)
            if lay.shapes:
                shapes[c] = _norm(lay)
        for a, b in itertools.combinations(shapes, 2):
            if any(a in g and b in g for g in SAME + (SAME_JOINED if f == "M" else [])):
                continue
            A, B = shapes[a], shapes[b]
            iou = float((A & B).sum() / max(1, (A | B).sum()))
            # medial forms are read inside a word, beside their neighbours: a little more alike is fine
            if iou > (limit + 0.02 if f == "M" else limit):
                out.append(Defect("confusable", f"{a} و{b} {FORM_NAMES[f]}", iou, limit))
    return out
