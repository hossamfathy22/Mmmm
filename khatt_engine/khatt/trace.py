"""A logo's picture made a vector drawing, so that it is measured as a drawing (vector.dna), not as
pixels (measure.py). The edge of the ink is followed to a fraction of a pixel (skimage's marching
squares on the grey picture), and each outline is made a designer's path: corners kept, curves fitted as
cubic Beziers, straight runs as lines (vparts.ring_to_segments). Where the typed word was found in the
picture (match.locate: which letter owns each pixel), every letter is traced on its own, as a typeface
gives its glyphs, so that the word says which shape is a tooth and which a tall letter."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .vector import VShape


def _rings(grey: np.ndarray, level: float = 0.5) -> List[np.ndarray]:
    from skimage.measure import find_contours
    pad = np.pad(grey, 2)
    out = []
    for c in find_contours(pad, level):
        if len(c) < 6:
            continue
        rows, cols = c[:, 0] - 2, c[:, 1] - 2
        out.append(np.c_[cols, -rows])                   # x right, y up
    return out


def trace(grey: np.ndarray, owner: Optional[np.ndarray] = None, err: float = 0.35, smooth: float = 0.6) -> List[VShape]:
    """grey: ink intensity 0..1 (1 = ink), rows down. owner: the index in the typed text of the letter
    each pixel belongs to (+1; 0 none), as match.locate gives it. Returns shapes in picture pixels, y up."""
    from scipy import ndimage as ndi
    from .vparts import ring_to_segments
    g = ndi.gaussian_filter(np.asarray(grey, float), smooth) if smooth else np.asarray(grey, float)
    groups = [(None, g)]
    if owner is not None:
        ink = g >= 0.5
        groups = []
        for i in sorted(set(np.unique(owner[ink]).tolist()) - {0}):
            m = ndi.binary_dilation(owner == i, iterations=1)
            groups.append((int(i) - 1, np.where(m, g, 0.0)))
    shapes = []
    for idx, gg in groups:
        contours = []
        for r in _rings(gg):
            segs = ring_to_segments(r, err)
            if segs:
                contours.append(segs)
        if contours:
            shapes.append(VShape(contours, f"traced-{idx}", idx))
    return shapes


def measure(grey: np.ndarray, owner: Optional[np.ndarray], text: str) -> dict:
    """The style of a picture read as a drawing: the x-height and the ascender from its letters traced one
    by one (the word says which is a tooth, which a tall letter); the stem, the slant and the hairline from
    the word traced whole (a letter cut from its neighbour has a cut where the join was, not a join)."""
    from .vector import dna
    whole = dna(trace(grey), text)
    if owner is None:
        return whole
    each = dna(trace(grey, owner), text)
    out = dict(whole)
    for k in ("x_height", "x_from", "ascender"):
        if k in each:
            out[k] = each[k]
    return out
