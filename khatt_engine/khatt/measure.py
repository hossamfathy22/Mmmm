"""Stage 1-2: precise measurement of one text line.

Everything is measured on a *deskewed* copy of the line (the slant is removed by shearing),
in "upright" coordinates: x to the right, y up, baseline at y = 0, units = working pixels.
Edges are located with sub-pixel accuracy on the anti-aliased grey image.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.morphology import medial_axis

Poly = List[Tuple[float, float]]


# ------------------------------------------------------------------ data
@dataclass
class Stem:
    x: float                 # upright centre
    w: float                 # sub-pixel width
    top: float               # upright y of the top edge
    bottom: float            # upright y of the bottom edge (0 = baseline)
    comp: int                # body it belongs to
    role: str = ""           # tooth / lowered / tall / short / descending
    letter: str = ""         # filled by understanding
    joined_right: bool = False
    joined_left: bool = False


@dataclass
class Piece:
    """A separate piece of ink: a dot, a dash or a letter body."""
    kind: str                # dot / dash / body
    comp: int
    x0: float; y0: float; x1: float; y1: float     # upright box
    area: float
    cx: float; cy: float
    size: float = 0.0        # dot diameter or dash length
    thick: float = 0.0       # dash thickness
    roundness: float = 0.0
    contours: List[Poly] = field(default_factory=list)
    label: str = ""


@dataclass
class Counter:
    comp: int
    x0: float; y0: float; x1: float; y1: float
    area: float


@dataclass
class Level:
    y: float
    n: int
    name: str = ""


@dataclass
class Line:
    idx: int
    top: int; bot: int; x0: int; x1: int        # image rows/cols (working px)
    base_row: int
    slant: float; t: float; off: int
    G: np.ndarray                                # deskewed ink intensity 0..1, rows top..bot
    B: np.ndarray                                # deskewed binary
    lab: np.ndarray                              # deskewed component labels
    level: float                                 # ink threshold
    x_height: float = 0.0
    baseline_shift: float = 0.0                  # sub-pixel correction of the baseline
    ascender: float = 0.0
    descender: float = 0.0
    stem: float = float("nan")
    stem_runs: float = float("nan")
    hair: float = float("nan")
    power: float = 2.0
    pen_fit: Dict[str, float] = field(default_factory=dict)
    pen_points: Tuple[np.ndarray, np.ndarray] = (np.zeros(0), np.zeros(0))
    monoline: bool = False
    stems: List[Stem] = field(default_factory=list)
    levels: List[Level] = field(default_factory=list)
    white: float = float("nan")
    pitch: float = float("nan")
    pieces: List[Piece] = field(default_factory=list)
    counters: List[Counter] = field(default_factory=list)
    coverage: float = 0.0
    width: float = 0.0
    height: float = 0.0

    # raster row r <-> upright y
    def y_of_row(self, r: float) -> float:
        return (self.base_row - self.top) - r + self.baseline_shift

    def row_of_y(self, y: float) -> float:
        return (self.base_row - self.top) - y + self.baseline_shift

    def x_of_col(self, c: float) -> float:
        return c - self.off

    def col_of_x(self, x: float) -> float:
        return x + self.off

    def to_image(self, x: float, y: float) -> Tuple[float, float]:
        """Upright (x, y) -> working-image pixel (column, row)."""
        row = self.top + self.row_of_y(y)
        return x + (self.base_row - row) * self.t, row

    @property
    def dots(self) -> List[Piece]:
        return [p for p in self.pieces if p.kind == "dot"]

    @property
    def dashes(self) -> List[Piece]:
        return [p for p in self.pieces if p.kind == "dash"]


# ------------------------------------------------------------------ helpers
def _runs(row: np.ndarray) -> List[Tuple[int, int]]:
    if not row.any():
        return []
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return list(zip(s.tolist(), (e - s).tolist()))


def _mode(vals: List[float], lo: float, hi: float) -> Tuple[float, int]:
    v = [int(round(x)) for x in vals if lo <= x <= hi]
    if len(v) < 3:
        return float("nan"), len(v)
    hist = np.bincount(v, minlength=int(hi) + 3).astype(float)
    sm = np.convolve(hist, [1, 2, 1], mode="same")
    peak = int(np.argmax(sm[1:]) + 1)
    near = [x for x in vals if lo <= x <= hi and abs(x - peak) <= max(1.5, 0.2 * peak)]
    return float(np.median(near)), len(v)


def _cross(profile: np.ndarray, lvl: float, i: int, step: int) -> Optional[float]:
    """From index i (inside ink) walk by `step` until the profile drops below lvl; return the
    sub-pixel position of the crossing."""
    n = len(profile)
    j = i
    while 0 <= j + step < n and profile[j + step] >= lvl:
        j += step
    k = j + step
    if not (0 <= k < n):
        return None
    a, b = profile[j], profile[k]
    f = (a - lvl) / (a - b) if a != b else 0.5
    return j + step * f


def _otsu01(v: np.ndarray) -> float:
    hist, edges = np.histogram(v, bins=256, range=(0, 1))
    hist = hist.astype(float)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c) / np.maximum(w0, 1e-9)
    m1 = (np.sum(hist * c) - np.cumsum(hist * c)) / np.maximum(w1, 1e-9)
    var = w0 * w1 * (m0 - m1) ** 2
    return float(c[int(np.argmax(var))])


# ------------------------------------------------------------------ main
def measure_line(ink01: np.ndarray, binary: np.ndarray, top: int, bot: int, idx: int = 0) -> Optional[Line]:
    """ink01: working image as ink intensity 0..1 (1 = ink). binary: thresholded ink."""
    H, W = binary.shape
    region = binary[top:bot + 1]
    h = bot - top + 1
    if region.sum() < 50:
        return None
    cols = np.flatnonzero(region.any(axis=0))
    x0, x1 = int(cols[0]), int(cols[-1])
    rows = region.sum(axis=1)
    thr = float(np.percentile(rows[rows > 0], 90)) * 0.45
    dense = rows >= thr
    best, i = (0, h - 1, -1), 0
    while i < h:
        if dense[i]:
            j, miss, end = i, 0, i
            while j < h and miss <= 2:
                if dense[j]:
                    end, miss = j, 0
                else:
                    miss += 1
                j += 1
            if end - i > best[2]:
                best = (i, end, end - i)
            i = j
        else:
            i += 1
    x_top_row, base_row = top + best[0], top + best[1]
    # slant: the shear that makes the strokes inside the band sharpest
    ys, xs = np.nonzero(binary[x_top_row:base_row + 1, x0:x1 + 1])
    ys, xs = ys + x_top_row, xs + x0
    if len(xs) > 300_000:
        k = len(xs) // 300_000 + 1
        ys, xs = ys[::k], xs[::k]
    pad = int(math.tan(math.radians(36)) * h) + 4

    def score(a: float) -> float:
        tt = math.tan(math.radians(a))
        hist = np.bincount(np.round(xs - (base_row - ys) * tt).astype(int) + pad, minlength=W + 2 * pad)
        return float(np.dot(hist, hist))

    angles = np.arange(-35, 36, 1.0)
    a = float(angles[int(np.argmax([score(v) for v in angles]))])
    fine = np.arange(a - 1, a + 1.001, 0.05)
    a = float(fine[int(np.argmax([score(v) for v in fine]))])
    slant = 0.0 if abs(a) < 0.3 else round(a, 2)
    t = math.tan(math.radians(slant))
    # deskew the grey image row by row with linear interpolation
    off = int(abs(t) * h) + 3
    DW = W + 2 * off
    rr = np.arange(top, bot + 1, dtype=float)[:, None]
    cc = np.arange(DW, dtype=float)[None, :]
    src_x = cc - off + (base_row - rr) * t
    src_y = np.broadcast_to(rr, src_x.shape)
    G = ndi.map_coordinates(ink01, [src_y, src_x], order=1, mode="constant", cval=0.0)
    lvl = _otsu01(ink01[top:bot + 1, x0:x1 + 1])
    lvl = min(max(lvl, 0.25), 0.75)
    B = G >= lvl
    lab, _ = ndi.label(B, structure=np.ones((3, 3)))
    L = Line(idx=idx, top=top, bot=bot, x0=x0, x1=x1, base_row=base_row, slant=slant, t=t, off=off,
             G=G, B=B, lab=lab, level=lvl)
    L.width = x1 - x0 + 1
    L.height = h
    L.coverage = float(region.sum()) / ((x1 - x0 + 1) * h)
    xh_rows = base_row - x_top_row + 1
    # ---- first stem estimate: most common horizontal run in the middle of the band
    lens: List[int] = []
    for yy in range(x_top_row + round(xh_rows * 0.3), x_top_row + round(xh_rows * 0.7) + 1):
        lens += [n for _, n in _runs(B[yy - top]) if n >= 2]
    U0, _ = _mode(lens, 2, xh_rows)
    L.stem_runs = U0
    if not U0 == U0:
        L.x_height = float(xh_rows)
        return L
    _pieces(L, U0)
    _stems(L, U0, x_top_row, base_row)
    _rhythm(L)
    _levels(L, x_top_row, base_row)
    _pen(L, U0)
    _counters(L)
    return L


# ------------------------------------------------------------------ pieces: dots, dashes, bodies
def _contours(mask: np.ndarray, oy: int, ox: int, L: Line) -> List[Poly]:
    m = np.pad(mask.astype(float), 2)
    m = ndi.gaussian_filter(m, 0.6)
    out = []
    for c in find_contours(m, 0.5):
        out.append([(L.x_of_col(p[1] - 2 + ox), L.y_of_row(p[0] - 2 + oy)) for p in c])
    return out


def _pieces(L: Line, U0: float) -> None:
    objs = ndi.find_objects(L.lab)
    hair_guess = max(1.5, 0.22 * U0)
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        m = L.lab[sl] == i
        area = float(m.sum())
        if area < max(6, 0.02 * U0 * U0):
            continue
        hh, ww = m.shape
        r0, c0 = sl[0].start, sl[1].start
        yy, xx = np.nonzero(m)
        cy, cx = L.y_of_row(yy.mean() + r0), L.x_of_col(xx.mean() + c0)
        fill = area / (hh * ww)
        # a sheared circle keeps its area; measure its height instead of its (slanted) width
        diam = math.sqrt(4 * area / math.pi)
        kind = "body"
        if 0.5 * U0 <= hh <= 2.4 * U0 and 0.5 * U0 <= diam <= 2.2 * U0 and 0.6 < fill < 0.9 and 0.7 < (hh / diam) < 1.3:
            kind = "dot"
        elif ww >= 3 * hh and hh <= max(3.5 * hair_guess, 0.5 * U0) and ww >= 0.8 * U0:
            kind = "dash"
        p = Piece(kind=kind, comp=i, x0=L.x_of_col(c0), x1=L.x_of_col(c0 + ww), y0=L.y_of_row(r0 + hh), y1=L.y_of_row(r0),
                  area=area, cx=cx, cy=cy)
        if kind == "dot":
            p.size = diam
            p.roundness = min(1.0, (area / (math.pi * (hh / 2) ** 2)))
        elif kind == "dash":
            p.size = float(ww)
            p.thick = area / ww
        if kind != "body" or True:
            p.contours = _contours(m, r0, c0, L)
        L.pieces.append(p)


# ------------------------------------------------------------------ stems
def _stems(L: Line, U0: float, x_top_row: int, base_row: int) -> None:
    body_ids = {p.comp for p in L.pieces if p.kind == "body"}
    tracks: List[Dict] = []
    for r in range(x_top_row - L.top, base_row - L.top + 1):
        for s, n in _runs(L.B[r]):
            if not (0.6 * U0 <= n <= 1.6 * U0):
                continue
            c = s + n / 2
            comp = int(L.lab[r, int(c)])
            if comp not in body_ids:
                continue
            for tr in tracks:
                if tr["last"] == r - 1 and abs(tr["c"][-1] - c) < 0.5 * U0:
                    tr["c"].append(c); tr["last"] = r; tr["rows"].append(r)
                    break
            else:
                tracks.append({"c": [c], "rows": [r], "last": r, "comp": comp})
    xh_rows = base_row - x_top_row + 1
    lvl = L.level
    for tr in tracks:
        if len(tr["rows"]) < 0.4 * xh_rows:
            continue
        c = float(np.median(tr["c"]))
        rows = tr["rows"]
        mid = rows[len(rows) // 2]
        widths = []
        for r in rows[len(rows) // 4: 3 * len(rows) // 4 + 1]:
            prof = L.G[r]
            ci = int(round(c))
            if prof[ci] < lvl:
                continue
            a, b = _cross(prof, lvl, ci, -1), _cross(prof, lvl, ci, +1)
            if a is not None and b is not None:
                widths.append(b - a)
        if not widths:
            continue
        w = float(np.median(widths))
        # top and bottom: average the central 40 % of the stem width
        band = slice(max(0, int(c - 0.2 * w)), int(c + 0.2 * w) + 1)
        col = L.G[:, band].mean(axis=1)
        if col[mid] < lvl:
            continue
        tr_top = _cross(col, lvl, mid, -1)
        tr_bot = _cross(col, lvl, mid, +1)
        if tr_top is None or tr_bot is None:
            continue
        L.stems.append(Stem(x=L.x_of_col(c), w=w, top=L.y_of_row(tr_top), bottom=L.y_of_row(tr_bot), comp=tr["comp"]))
    L.stems.sort(key=lambda s: -s.x)   # right to left, the reading order of Arabic
    if L.stems:
        L.stem = float(np.median([s.w for s in L.stems]))


# ------------------------------------------------------------------ pen rule
def _pen(L: Line, U0: float) -> None:
    """Fit width = hair + (stem - hair) * |sin(angle)|^power to the skeleton of the letters."""
    body = np.isin(L.lab, [p.comp for p in L.pieces if p.kind == "body"])
    if body.sum() < 50:
        return
    skel, dist = medial_axis(body, return_distance=True)
    ys, xs = np.nonzero(skel)
    if len(xs) < 30:
        return
    pts = np.c_[xs, ys].astype(float)
    nb = ndi.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    deg = nb[ys, xs]
    tree = cKDTree(pts)
    special = pts[(deg == 1) | (deg >= 3)]
    keep = np.ones(len(pts), bool)
    if len(special):
        near = cKDTree(special).query(pts, k=1)[0]
        keep &= near > 0.6 * U0
    rad = max(3.0, 0.35 * U0)
    theta = np.full(len(pts), np.nan)
    for k in np.flatnonzero(keep):
        idx = tree.query_ball_point(pts[k], rad)
        if len(idx) < 5:
            continue
        q = pts[idx] - pts[idx].mean(axis=0)
        cov = q.T @ q
        ev, evec = np.linalg.eigh(cov)
        if ev[1] <= 0 or ev[0] / ev[1] > 0.08:
            continue
        v = evec[:, 1]
        theta[k] = abs(v[1]) / math.hypot(v[0], v[1])          # |sin| of the stroke direction
    ok = ~np.isnan(theta)
    s = theta[ok]
    w = 2 * dist[ys[ok], xs[ok]] - 0.5
    if len(s) < 30:
        return
    L.pen_points = (s, w)
    hair_runs = _hair_runs(L, U0)
    h0 = hair_runs if hair_runs == hair_runs else 0.2 * U0

    def resid(p):
        U, hh, pw = p
        return hh + (U - hh) * s ** pw - w

    fit = least_squares(resid, [U0, h0, 2.0], bounds=([0.5 * U0, 1.0, 0.4], [2.0 * U0, U0, 8.0]),
                        loss="soft_l1", f_scale=max(1.0, 0.08 * U0))
    U, hh, pw = fit.x
    rmse = float(np.sqrt(np.median(fit.fun ** 2)))
    L.pen_fit = {"stem": float(U), "hair": float(hh), "power": float(pw), "median_error": rmse, "samples": int(len(s))}
    L.power = float(pw)
    L.hair = hair_runs if hair_runs == hair_runs else float(hh)
    if not L.stem == L.stem:
        L.stem = float(U)
    n_thin = int((w < 0.45 * L.stem).sum())
    L.monoline = n_thin < max(8, 0.03 * len(w)) or L.hair > 0.75 * L.stem
    if L.monoline:
        L.hair = L.stem


def _hair_runs(L: Line, U0: float) -> float:
    """Most common short vertical run, then refined to sub-pixel on the grey profile."""
    lens, where = [], []
    for c in range(L.B.shape[1]):
        col = L.B[:, c]
        if not col.any():
            continue
        for s, n in _runs(col):
            if 2 <= n <= 0.6 * U0:
                lens.append(n); where.append((c, s, n))
    m, cnt = _mode(lens, 2, max(3, 0.6 * U0))
    if not m == m or cnt < 8:
        return float("nan")
    sub = []
    for c, s, n in where:
        if abs(n - m) <= max(1.0, 0.25 * m):
            prof = L.G[:, c]
            mid = s + n // 2
            a, b = _cross(prof, L.level, mid, -1), _cross(prof, L.level, mid, +1)
            if a is not None and b is not None:
                sub.append(b - a)
    return float(np.median(sub)) if sub else m


# ------------------------------------------------------------------ grid levels
def _cluster(vals: List[float], tol: float) -> List[List[float]]:
    out: List[List[float]] = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return out


def _levels(L: Line, x_top_row: int, base_row: int) -> None:
    # sub-pixel baseline: the bottom edge of the joins between teeth (the lowest point of each
    # counter); without joins, the bottom edge of the dense band
    approx_row = base_row - L.top
    bottoms = []
    for a, b in zip(L.stems, L.stems[1:]):
        if not (a.joined_left and b.joined_right):
            continue
        c = int(round(L.col_of_x(((a.x - a.w / 2) + (b.x + b.w / 2)) / 2)))
        col = L.G[:, max(0, c - 1):c + 2].mean(axis=1)
        best = None
        for r0, n in _runs(col >= L.level):
            r1 = r0 + n - 1
            if abs(r1 - approx_row) <= 0.4 * L.stem and (best is None or abs(r1 - approx_row) < abs(best - approx_row)):
                best = r1
        if best is not None:
            e = _cross(col, L.level, best, +1)
            if e is not None:
                bottoms.append(e)
    if not bottoms:
        prof = L.G[:, L.col_of_x(L.x0 - L.base_row * 0) if False else slice(None)].mean(axis=1)
        e = _cross(prof, max(prof[approx_row] * 0.5, 1e-6), approx_row, +1) if prof[approx_row] > 0 else None
        bottoms = [e if e is not None else approx_row + 0.5]
    shift = L.y_of_row(float(np.median(bottoms)))
    if abs(shift) > 0:
        L.baseline_shift = -shift
        for s in L.stems:
            s.top -= shift; s.bottom -= shift
        for p in L.pieces:
            p.y0 -= shift; p.y1 -= shift; p.cy -= shift
            p.contours = [[(x, y - shift) for x, y in poly] for poly in p.contours]
    tol = max(1.0, 0.12 * L.stem)
    groups = _cluster([s.top for s in L.stems], tol)
    groups.sort(key=len, reverse=True)
    if not groups:
        L.x_height = float(base_row - x_top_row + 1)
        return
    band_top = L.y_of_row(x_top_row - L.top)
    main = min(groups[:3], key=lambda g: abs(float(np.mean(g)) - band_top)) if len(groups) > 1 else groups[0]
    X = float(np.median(main))
    L.x_height = X
    levels = []
    for g in groups:
        y = float(np.median(g))
        if abs(y - X) <= tol:
            name = "x_height"
        elif y > X:
            name = "tall"
        elif y > 0.75 * X:
            name = "lowered"
        else:
            name = "short"
        levels.append(Level(y=y, n=len(g), name=name))
    L.levels = sorted(levels, key=lambda v: -v.y)
    for s in L.stems:
        if s.bottom < -0.3 * L.stem:
            s.role = "descending"
        elif abs(s.top - X) <= tol:
            s.role = "tooth"
        elif s.top > X:
            s.role = "tall"
        elif s.top > 0.75 * X:
            s.role = "lowered"
        else:
            s.role = "short"
    ink_rows = np.flatnonzero(L.B.any(axis=1))
    L.ascender = max(0.0, L.y_of_row(ink_rows[0]) - X)
    L.descender = max(0.0, -L.y_of_row(ink_rows[-1] + 1))


# ------------------------------------------------------------------ rhythm
def _rhythm(L: Line) -> None:
    pitches, whites = [], []
    for a, b in zip(L.stems, L.stems[1:]):          # a is to the right of b
        p = a.x - b.x
        if p < 3.2 * L.stem and a.comp == b.comp:
            pitches.append(p)
            whites.append((a.x - a.w / 2) - (b.x + b.w / 2))
            a.joined_left = b.joined_right = True
    if pitches:
        L.pitch = float(np.median(pitches))
        L.white = float(np.median(whites))


# ------------------------------------------------------------------ counters
def _counters(L: Line) -> None:
    bg = ~L.B
    blab, _ = ndi.label(bg)
    edge = set(np.unique(np.concatenate([blab[0], blab[-1], blab[:, 0], blab[:, -1]])).tolist())
    for i, sl in enumerate(ndi.find_objects(blab), start=1):
        if sl is None or i in edge:
            continue
        area = float((blab[sl] == i).sum())
        if area < 0.15 * L.stem * L.stem:
            continue
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        ring = L.lab[max(0, r0 - 1):r1 + 1, max(0, c0 - 1):c1 + 1]
        comps = np.unique(ring[ring > 0])
        L.counters.append(Counter(comp=int(comps[0]) if len(comps) else 0,
                                  x0=L.x_of_col(c0), x1=L.x_of_col(c1), y0=L.y_of_row(r1),
                                  y1=L.y_of_row(r0), area=area))
