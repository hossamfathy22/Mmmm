"""The logo's own letters, drawn with its own pen.

The engine's letters are constructions of its own (written on بيدوناق and سیرنگ), given the logo's
numbers: every logo gets the same ح, only thicker or thinner. Here the letters come from the logo:

  1. the typed word is found in the logo letter by letter (match.locate: the word set in real
     typefaces and warped onto the logo; 97% of the ink to its own letter on typeface logos);
  2. the logo's pen is measured on the word's strokes (qalam.fit_nib: width, thickness, angle);
  3. every letter is read as the centre lines of its strokes and the places of its dots, and drawn
     again: the nib moved along each centre line (qalam.stroke), a dot as one print of the nib turned.

What comes out is not the logo's pixels pasted: it is how the logo's hand wrote each letter (where the
pen went) and the pen it wrote with, which can be redrawn at any size, joined, and taken apart into the
parts of the six families (the stem, the bowl, the tooth, the loop, the arc, the tail) for the letters
the logo does not have.

All coordinates are in the working picture's pixels (guard.prepare), x to the right, y up (y = -row).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

Point = Tuple[float, float]

DOT_AREA = 0.6      # a piece of a letter smaller than this many nib widths squared is a dot (or a mark)


@dataclass
class OwnLetter:
    char: str
    form: str                                   # i I M F: isolated, initial, medial, final
    index: int                                  # its place in the typed text
    strokes: List[List[Point]] = field(default_factory=list)     # centre lines, y up
    dots: List[Tuple[float, float, float]] = field(default_factory=list)   # (x, y, size as a share of a nib print)
    mask: Optional[np.ndarray] = None           # its ink in the logo (rows down)
    match: float = 0.0                          # how well the redrawing covers its ink (IoU)

    def box(self) -> Tuple[float, float, float, float]:
        pts = [p for s in self.strokes for p in s] + [(x, y) for x, y, _ in self.dots]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys)) if pts else (0.0, 0.0, 0.0, 0.0)


def _faces():
    from .bench import fonts
    from .pipeline import MATCH_FACES
    files = {f.stem: f for f in fonts.font_files()}
    return [fonts.load(str(files[n])) for n in MATCH_FACES if n in files]


def body_of(mask: np.ndarray) -> np.ndarray:
    """The strokes of a line, its dots and dashes left out (pieces under 5% of the largest)."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if not n:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes > 0.05 * sizes.max()))


def read_letter(m: np.ndarray, nib) -> Tuple[List[List[Point]], List[Tuple[float, float, float]]]:
    """A letter's ink as the centre lines of its strokes and its dots."""
    from .qalam import dot
    from .reference import _mask_strokes
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    sizes = ndi.sum(m, lab, range(1, n + 1)) if n else []
    one = dot(0.0, 0.0, nib).area or 1.0
    strokes, dots = [], []
    for j in range(n):
        comp = lab == j + 1
        if sizes[j] < DOT_AREA * nib.w * nib.w:
            ys, xs = np.nonzero(comp)
            dots.append((float(xs.mean()), -float(ys.mean()), float(sizes[j] / one)))
            continue
        for st in _mask_strokes(comp):
            pts = [(c, -r) for r, c, _ in st]
            if len(pts) >= 2:
                strokes.append(pts)
    return strokes, dots


def draw(strokes: Sequence[Sequence[Point]], dots: Sequence[Tuple[float, float, float]], nib):
    """The letter drawn again with the nib: a shapely shape, y up."""
    from shapely.ops import unary_union
    from .qalam import dot, stroke
    parts = [stroke(s, nib) for s in strokes if len(s) >= 2]
    parts += [dot(x, y, nib, k=max(0.3, min(2.5, k))) for x, y, k in dots]
    return unary_union(parts) if parts else None


def rasterise(geom, shape: Tuple[int, int]) -> np.ndarray:
    """A shape (y up) back to a mask of the picture (rows down)."""
    from PIL import Image, ImageDraw
    H, W = shape
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    if geom is not None:
        for poly in getattr(geom, "geoms", [geom]):
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
            d.polygon([(x, -y) for x, y in poly.exterior.coords], fill=1)
            for h in poly.interiors:
                d.polygon([(x, -y) for x, y in h.coords], fill=0)
    return np.asarray(im) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = float((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def own_word(image: bytes, text: str, faces=None, line: Optional[int] = None) -> Optional[Dict]:
    """The typed word's letters as the logo writes them, and its pen: {"nib", "nib_fit", "letters":
    [OwnLetter], "line": the word's ink, "shape": the picture's size, "match": the whole word redrawn
    against its ink}. None where the word is not found (no typeface at hand to find it with)."""
    from .match import locate
    from .qalam import Nib, fit_nib
    from .typeset import shape_forms
    faces = faces if faces is not None else _faces()
    if not faces:
        return None
    got = locate(image, text, faces, line=line)
    if got is None:
        return None
    word, owner = np.asarray(got["line"], bool), got["owner"]
    nb = fit_nib(body_of(word), 1)
    nib = Nib(nb["width"], max(0.5, nb["thick"]), nb["angle"], nb["power"])
    forms = shape_forms(text)
    letters: List[OwnLetter] = []
    for i, c in enumerate(text):
        if c.isspace() or i >= len(forms):
            continue
        m = word & (owner == i + 1)
        if m.sum() < 10:
            continue
        strokes, dots = read_letter(m, nib)
        o = OwnLetter(c, forms[i][1], i, strokes, dots, m)
        o.match = iou(rasterise(draw(strokes, dots, nib), m.shape), m)
        letters.append(o)
    whole = np.zeros_like(word)
    for o in letters:
        whole |= rasterise(draw(o.strokes, o.dots, nib), word.shape)
    return {"nib": nib, "nib_fit": nb, "letters": letters, "line": word, "shape": word.shape,
            "match": iou(whole, word), "face": got["face"].name if hasattr(got["face"], "name") else ""}
