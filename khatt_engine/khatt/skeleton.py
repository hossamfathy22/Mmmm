"""Reading the skeleton of the letters: the centre line the pen travelled.

The skeleton of the letter bodies is traced into a graph (ends, junctions and the strokes between
them). Every stroke is followed along its length and each change of direction is judged by its
radius, the way the eye judges it:

- a turn tighter than half the pen (centre-line radius < 0.5 stem) is a CORNER: the designer joined two straight
  strokes, as in a geometric kufi logo;
- a wider turn is a CURVE: the pen swept round, as in a calligraphic tail or bowl;
- what does not turn is a STRAIGHT stroke, whose angle is recorded (horizontal, vertical, diagonal).

From these the logo's vocabulary is measured: how much of its turning happens at corners
(`angular`), how round those corners are (`corner_r`), and which stroke angles it uses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

from .measure import Line

Pt = Tuple[float, float]


@dataclass
class Segment:
    kind: str                     # "straight" | "curve" | "corner"
    pts: List[Pt]                 # in line coordinates (x right, y up from the baseline)
    angle: float = 0.0            # straight: direction in degrees (0 horizontal, 90 vertical)
    turn: float = 0.0             # curve/corner: signed turning in degrees
    radius: float = 0.0           # curve/corner: radius of the centre line


@dataclass
class Skeleton:
    strokes: List[List[Pt]] = field(default_factory=list)
    segments: List[Segment] = field(default_factory=list)
    angular: float = float("nan")         # share of the turning that happens at corners (0..1)
    corner_r: float = float("nan")        # median corner radius of the outer edge, px
    angles: List[Tuple[float, float]] = field(default_factory=list)   # (angle, share of straight length)
    total_length: float = 0.0

    def summary(self, k: float = 1.0) -> Dict:
        return {"angular": round(self.angular, 3) if self.angular == self.angular else None,
                "corner_r": round(self.corner_r * k, 2) if self.corner_r == self.corner_r else None,
                "angles": [{"angle": round(a, 1), "share": round(s, 3)} for a, s in self.angles],
                "segments": {kind: sum(1 for s in self.segments if s.kind == kind) for kind in ("straight", "curve", "corner")}}


# ------------------------------------------------------------------ tracing the graph
_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _trace(sk: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Split a one-pixel skeleton into strokes between ends and junctions."""
    H, W = sk.shape
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    nb[~sk] = 0
    node = sk & (nb != 2)
    seen = np.zeros_like(sk, bool)
    strokes = []

    def neighbours(r, c):
        for dr, dc in _NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and sk[rr, cc]:
                yield rr, cc

    starts = list(zip(*np.nonzero(node)))
    for r0, c0 in starts:
        for r1, c1 in neighbours(r0, c0):
            if seen[r1, c1] and not node[r1, c1]:
                continue
            path = [(r0, c0), (r1, c1)]
            prev, cur = (r0, c0), (r1, c1)
            while not node[cur]:
                seen[cur] = True
                nxt = [p for p in neighbours(*cur) if p != prev and p not in path[-3:]]
                if not nxt:
                    break
                # prefer 4-connected steps, they keep the path on the centre
                nxt.sort(key=lambda p: abs(p[0] - cur[0]) + abs(p[1] - cur[1]))
                prev, cur = cur, nxt[0]
                path.append(cur)
            key = (min(path[0], path[-1]), max(path[0], path[-1]), len(path))
            strokes.append((key, path))
    out, keys = [], set()
    for key, path in strokes:
        if key in keys:
            continue
        keys.add(key)
        out.append(path)
    # closed loops without any node (a round counter drawn in one line)
    rest = sk & ~seen & ~node
    lab, n = ndi.label(rest, structure=np.ones((3, 3)))
    for i in range(1, n + 1):
        rr, cc = np.nonzero(lab == i)
        if len(rr) < 8:
            continue
        r, c = rr[0], cc[0]
        path, prev = [(r, c)], None
        while True:
            nxt = [p for p in neighbours(*path[-1]) if lab[p] == i and p != prev and p not in path[-3:]]
            if not nxt or (len(path) > 3 and nxt[0] == path[0]):
                break
            prev = path[-1]
            path.append(nxt[0])
            if len(path) > len(rr) + 2:
                break
        out.append(path)
    return out


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    if d[-1] < step:
        return pts
    s = np.arange(0, d[-1] + 1e-9, step)
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


# ------------------------------------------------------------------ reading one stroke
def _split(pts: np.ndarray, U: float) -> List[Segment]:
    """Follow a stroke and cut it into straight runs, curves and corners."""
    step = max(0.5, 0.12 * U)
    p = _resample(pts, step)
    if len(p) < 4:
        return []
    sig = max(1.0, 0.25 * U / step)
    sm = np.c_[ndi.gaussian_filter1d(p[:, 0], sig, mode="nearest"), ndi.gaussian_filter1d(p[:, 1], sig, mode="nearest")]
    d = np.gradient(sm, axis=0)
    th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    kappa = np.gradient(th) / step                       # curvature, 1/px
    turning = np.abs(kappa) > 1.0 / (10.0 * U)           # radius under 10 stems counts as turning
    spread = 2.56 * sig * step                           # length a sharp corner occupies after smoothing
    # pixel steps make tiny wobbles; a wobble that turns less than 12° is part of the straight stroke
    i = 0
    while i < len(turning):
        j = i
        while j + 1 < len(turning) and turning[j + 1] == turning[i]:
            j += 1
        if turning[i] and abs(np.sum(kappa[i:j + 1]) * step) < math.radians(12):
            turning[i:j + 1] = False
        i = j + 1
    segs: List[Segment] = []
    i, n = 0, len(p)
    while i < n:
        j = i
        while j + 1 < n and turning[j + 1] == turning[i]:
            j += 1
        run = sm[i:j + 1]
        length = step * (j - i + 1)
        if turning[i]:
            dk = np.abs(kappa[i:j + 1]) * step
            turn = float(np.sum(kappa[i:j + 1]) * step)
            if abs(turn) > math.radians(12):
                # radius from the central 80 % of the turning, with the smoothing taken out
                c = np.cumsum(dk) / dk.sum()
                a, b = np.searchsorted(c, 0.1), np.searchsorted(c, 0.9)
                l80 = step * max(1, b - a)
                r = math.sqrt(max(0.0, l80 ** 2 - spread ** 2)) / (0.8 * abs(turn))
                kind = "corner" if r < 0.5 * U else "curve"
                segs.append(Segment(kind, [tuple(q) for q in run], turn=math.degrees(turn), radius=r))
        elif length >= 0.6 * U:
            a = math.degrees(math.atan2(run[-1, 1] - run[0, 1], run[-1, 0] - run[0, 0])) % 180
            segs.append(Segment("straight", [tuple(q) for q in run], angle=a))
        i = j + 1
    return segs


def read_skeleton(L: Line) -> Skeleton:
    """Skeleton of the letter bodies of one measured line (dots and dashes are left out)."""
    from skimage.morphology import skeletonize
    out = Skeleton()
    body_ids = [p.comp for p in L.pieces if p.kind == "body"]
    if not body_ids or not L.stem == L.stem:
        return out
    U = L.stem
    mask = np.isin(L.lab, body_ids) & L.B.astype(bool)
    mask = ndi.binary_closing(mask, iterations=1)
    sk = skeletonize(mask)
    dist = ndi.distance_transform_edt(mask)
    paths = _trace(sk)
    # prune spurs: short strokes that end in the open (corner artefacts of the skeleton)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    keep = []
    for path in paths:
        ends = sum(1 for q in (path[0], path[-1]) if nb[q] == 1)
        length = len(path)
        if ends >= 1 and length < 0.9 * U and len(paths) > 1:
            continue
        keep.append(path)
    for path in keep:
        rc = np.array(path, float)
        xy = np.c_[[L.x_of_col(c) for c in rc[:, 1]], [L.y_of_row(r) for r in rc[:, 0]]]
        out.strokes.append([tuple(q) for q in xy])
        out.segments += _split(xy, U)
    out.total_length = float(sum(np.hypot(*np.diff(np.array(s), axis=0).T).sum() for s in out.strokes if len(s) > 1))
    turn_c = sum(abs(s.turn) for s in out.segments if s.kind == "corner")
    turn_v = sum(abs(s.turn) for s in out.segments if s.kind == "curve")
    if turn_c + turn_v > 0:
        out.angular = turn_c / (turn_c + turn_v)
    corners = [s for s in out.segments if s.kind == "corner" and abs(s.turn) > 45]
    if corners:
        # the outer edge of a corner is the centre-line radius plus half the pen
        out.corner_r = float(np.median([s.radius for s in corners])) + U / 2
    lens: Dict[float, float] = {}
    total = 0.0
    for s in out.segments:
        if s.kind != "straight":
            continue
        a = s.angle
        a = 0.0 if (a < 8 or a > 172) else (90.0 if abs(a - 90) < 8 else round(a / 5) * 5.0)
        ln = float(np.hypot(*np.diff(np.array(s.pts)[[0, -1]], axis=0).T)[0])
        lens[a] = lens.get(a, 0.0) + ln
        total += ln
    if total > 0:
        out.angles = sorted(((a, v / total) for a, v in lens.items() if v / total >= 0.03), key=lambda t: -t[1])
    return out
