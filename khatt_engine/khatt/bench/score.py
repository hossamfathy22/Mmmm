"""Comparing two drawings of a letter: the distance between their skeletons, and who reads as whom."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

PX = 32                      # pixels per x-height on the comparison canvas
CANVAS = (int(4.6 * PX), int(7.0 * PX))          # rows, cols
BASE_ROW = int(3.2 * PX)     # room for ascenders above, descenders below
RIGHT_COL = CANVAS[1] - int(0.5 * PX)
SHIFTS = range(-int(0.45 * PX), int(0.45 * PX) + 1, 3)
SCALES = (0.8, 1.0, 1.25)    # typefaces do not agree on how tall a tooth is beside a loop: sizes may differ this much


class Shape:
    """A letter body on the comparison canvas, with what is needed to compare it quickly."""

    def __init__(self, mask: np.ndarray):
        self.mask = np.array(mask, dtype=np.uint8) > 0          # a fresh, plain array whatever came in
        self.skel = skeletonize(self.mask) if self.mask.any() else self.mask
        self.pts = np.argwhere(self.skel)
        # distance from every pixel to the skeleton, capped: a missing limb costs, but not without end
        self.dist = np.minimum(ndi.distance_transform_edt(~self.skel), 1.2 * PX) if self.skel.any() else np.full(mask.shape, 1.2 * PX)
        self._sized: Optional[List["Shape"]] = None

    @property
    def empty(self) -> bool:
        return len(self.pts) == 0

    def sized(self) -> List["Shape"]:
        """The same letter a little smaller and a little larger, kept on the baseline and on its
        right end."""
        if self._sized is None:
            out = []
            H, W = self.mask.shape
            for k in SCALES:
                if k == 1.0:
                    out.append(self); continue
                yy, xx = np.mgrid[0:H, 0:W]
                src_y = BASE_ROW + (yy - BASE_ROW) / k
                src_x = RIGHT_COL + (xx - RIGHT_COL) / k
                m = ndi.map_coordinates(self.mask.astype(np.float32), [src_y, src_x], order=1, cval=0.0) > 0.5
                out.append(Shape(m))
            self._sized = out
        return self._sized


def _once(a: Shape, b: Shape) -> float:
    if a.empty or b.empty:
        return 1.2
    H, W = a.mask.shape
    best = 1e9
    for dx in SHIFTS:
        ca = a.pts[:, 1] + dx
        ok = (ca >= 0) & (ca < W)
        cb = b.pts[:, 1] - dx
        ok2 = (cb >= 0) & (cb < W)
        if ok.sum() < 0.7 * len(a.pts) or ok2.sum() < 0.7 * len(b.pts):
            continue
        d = 0.5 * (b.dist[a.pts[ok, 0], ca[ok]].mean() + a.dist[b.pts[ok2, 0], cb[ok2]].mean())
        best = min(best, d)
    return float(best / PX) if best < 1e9 else 1.2


def distance(a: Shape, b: Shape, sizes: bool = True) -> float:
    """How far apart two letters are, in x-heights: the mean distance from the centre line of each to
    the centre line of the other, at the sideways shift and the size (within a quarter) that suit
    them best; a change of size costs a little. Weight does not count (a bold and a light drawing
    of one skeleton are at distance 0); a missing or extra limb does."""
    if not sizes:
        return _once(a, b)
    return min(_once(v, b) * (1.0 + 0.3 * abs(np.log(k))) + 0.02 * abs(np.log(k)) for k, v in zip(SCALES, a.sized()))


def read(x: Shape, refs: Dict[str, Shape]) -> Tuple[str, float, float]:
    """Read a letter by the nearest reference. Returns (the letter it reads as, its distance, the
    margin to the second nearest)."""
    ds = sorted(((distance(x, r), c) for c, r in refs.items()))
    return ds[0][1], ds[0][0], (ds[1][0] - ds[0][0]) if len(ds) > 1 else 0.0
