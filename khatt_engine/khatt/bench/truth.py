"""Ground truth from a typeface: words rendered as pictures, and single letters in their forms."""
from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from .fonts import Contour, Face, shape

FORMS = {"i": "{}", "I": "{}ـ", "M": "ـ{}ـ", "F": "ـ{}"}


def _area(c: Contour) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(c, c[1:] + c[:1]))


def raster(contours: List[Contour], x0: float, y_top: float, s: float, W: int, H: int, ss: int = 3) -> np.ndarray:
    """Coverage (0..1) of outlines filled by the non-zero winding rule, as typefaces are."""
    wind = np.zeros((H * ss, W * ss), np.int16)
    for c in contours:
        if len(c) < 3:
            continue
        layer = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(layer).polygon([((x - x0) * s * ss, (y_top - y) * s * ss) for x, y in c], fill=1)
        wind += np.array(layer, dtype=np.int16) * (1 if _area(c) > 0 else -1)
    m = (wind != 0).astype(np.float32)
    return m.reshape(H, ss, W, ss).mean(axis=(1, 3))


def x_height(face: Face) -> float:
    """The height of a tooth in font units: the top of the body of a medial ب."""
    cs = [c for cl, cont in shape(face, "ـبـ") if cl == 1 for c in cont]
    if not cs:
        return 0.5 * face.upm
    body = max(cs, key=lambda c: abs(_area(c)))
    return max(y for _, y in body)


def word_image(face: Face, text: str, x_px: float = 130.0, pad: float = 0.9) -> Tuple[Image.Image, float]:
    """`text` set in the typeface as a clean black-on-white picture whose teeth are `x_px` high.
    Returns the picture and the scale (pixels per font unit)."""
    cs = [c for _, cont in shape(face, text) for c in cont]
    X = x_height(face)
    s = x_px / X
    xs = [p[0] for c in cs for p in c]; ys = [p[1] for c in cs for p in c]
    m = pad * x_px
    W, H = int((max(xs) - min(xs)) * s + 2 * m), int((max(ys) - min(ys)) * s + 2 * m)
    cov = raster(cs, min(xs) - m / s, max(ys) + m / s, s, W, H)
    return Image.fromarray((255 - cov * 255).astype(np.uint8)), s


def png(im: Image.Image) -> bytes:
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def letter_mask(face: Face, char: str, form: str, px: float, canvas: Tuple[int, int], base_row: int, right_col: int,
                X: Optional[float] = None) -> Optional[np.ndarray]:
    """The body of one letter in one form (its dots left out), drawn so that the typeface's tooth
    is `px` pixels high, the baseline on `base_row` and the right end of the body on `right_col`."""
    text = FORMS[form].format(char)
    idx = text.index(char)
    cs = [c for cl, cont in shape(face, text) if cl == idx for c in cont]
    if not cs:
        return None
    X = X or x_height(face)
    s = px / X
    H, W = canvas
    xs = [p[0] for c in cs for p in c]
    big = raster(cs, min(xs) - 2 / s, (base_row + 0.0) / s, s, int((max(xs) - min(xs)) * s) + 5, H, ss=2) > 0.5
    return place(drop_marks(big, px), base_row, right_col, px, canvas)


def drop_marks(m: np.ndarray, px: float) -> np.ndarray:
    """Keep the letter body: the largest piece of ink and every piece of comparable size."""
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n <= 1:
        return m
    area = np.bincount(lab.ravel()); area[0] = 0
    main = int(area.argmax())
    keep = [i for i in range(1, n + 1) if i == main or area[i] >= 0.45 * area[main]]
    return np.isin(lab, keep)


def core_span(m: np.ndarray, base_row: int, px: float) -> Tuple[int, int]:
    """The columns of the letter itself: of the ink outside the baseline band, where the joins to
    the neighbours run. A letter that lies wholly in that band keeps its whole width."""
    band = np.zeros(m.shape[0], bool)
    band[max(0, int(base_row - 0.28 * px)):min(m.shape[0], int(base_row + 0.12 * px) + 1)] = True
    cols = np.flatnonzero(m[~band].any(axis=0))
    if len(cols) < 2:
        cols = np.flatnonzero(m.any(axis=0))
    return (int(cols[0]), int(cols[-1])) if len(cols) else (0, 0)


def place(m: np.ndarray, base_row: int, right_col: int, px: float, canvas: Tuple[int, int]) -> np.ndarray:
    """Copy the letter itself (a mask whose row `base_row` is the baseline) onto the canvas, its
    right end on `right_col`. The joins that run on from it to its neighbours are left behind,
    with a margin of a seventh of the x-height, so a letter drawn with its joins and the same
    letter drawn without them compare as the same."""
    H, W = canvas
    out = np.zeros((H, W), bool)
    if not m.any():
        return out
    c0, c1 = core_span(m, base_row, px)
    margin = int(round(0.15 * px))
    shift = right_col - c1
    for c in range(max(0, c0 - margin), min(m.shape[1], c1 + margin + 1)):
        d = c + shift
        if 0 <= d < W:
            out[:min(H, m.shape[0]), d] = m[:min(H, m.shape[0]), c]
    return out
