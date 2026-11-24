"""A logo drawn by numbers, cleanly.

The outline is traced under the pixel on the grey picture (a zoom of eight and marching squares), the
noise of the picture is smoothed away along it (each run between two corners on its own, the corners
held), and each run is written as a straight line when it is straight within the error, else as the
fewest Bezier curves within it. This is how a type designer digitises a logo before drawing a
typeface from its letters: the designer's shapes kept, the JPEG's left out.

Coordinates are those of the given array: x to the right, y down, one unit a pixel.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .bezier import _fit, _unit, sample

Seg = Tuple[str, List[Tuple[float, float]]]      # ("L", [a, b]) or ("C", [p0, c1, c2, p1])


def _resample(p: np.ndarray, step: float) -> np.ndarray:
    p = np.vstack([p, p[:1]])
    d = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0, np.cumsum(d)]
    t = np.arange(0, s[-1], step)
    return np.c_[np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])]


def _turn(p: np.ndarray, k: int) -> np.ndarray:
    a, b = np.roll(p, k, 0), np.roll(p, -k, 0)
    v1, v2 = p - a, b - p
    c = (v1 * v2).sum(1) / (np.hypot(*v1.T) * np.hypot(*v2.T) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def corners(p: np.ndarray, k: int = 6, angle: float = 38.0) -> List[int]:
    """Indexes where the outline turns by `angle` degrees or more over `k` points on either side."""
    t = _turn(p, k)
    n, out = len(p), []
    for i in range(n):
        if t[i] >= angle and t[i] == max(t[(i + j) % n] for j in range(-k, k + 1)):
            if not out or i - out[-1] > k:
                out.append(i)
    if len(out) > 1 and out[0] + n - out[-1] <= k:
        out.pop()
    return out


def _smooth(p: np.ndarray, cs: List[int], sigma: float) -> np.ndarray:
    """Gaussian smoothing of each run between two corners, the corners themselves held."""
    from scipy.ndimage import gaussian_filter1d
    if not cs:
        return np.c_[gaussian_filter1d(p[:, 0], sigma, mode="wrap"), gaussian_filter1d(p[:, 1], sigma, mode="wrap")]
    q, n = p.copy(), len(p)
    for i, c in enumerate(cs):
        d = cs[(i + 1) % len(cs)]
        idx = np.arange(c, d + 1) if d > c else np.r_[np.arange(c, n), np.arange(0, d + 1)]
        seg = p[idx]
        if len(seg) > 5:
            pad = np.vstack([seg[:1].repeat(6, 0), seg, seg[-1:].repeat(6, 0)])
            sm = np.c_[gaussian_filter1d(pad[:, 0], sigma, mode="nearest"),
                       gaussian_filter1d(pad[:, 1], sigma, mode="nearest")][6:-6]
            q[idx[1:-1]] = sm[1:-1]
    return q


def _runs(p: np.ndarray, cs: List[int]) -> List[np.ndarray]:
    if len(cs) < 2:
        cs = sorted({int(np.argmin(p[:, 0])), int(np.argmax(p[:, 1])), int(np.argmax(p[:, 0])), int(np.argmin(p[:, 1]))})
    out = []
    for i, c in enumerate(cs):
        d = cs[(i + 1) % len(cs)]
        seg = p[c:d + 1] if d > c else np.r_[p[c:], p[:d + 1]]
        if len(seg) >= 2:
            out.append(seg)
    return out


def fit_ring(points: Sequence[Tuple[float, float]], err: float = 0.18, step: float = 0.3,
             smooth: float = 0.0, k: int = 5, angle: float = 32.0) -> List[Seg]:
    """A closed outline as straight lines (runs straight within `err`) and few Beziers elsewhere,
    its corners kept. `smooth` (in samples) first takes the noise off each run between corners."""
    p = _resample(np.asarray(points, float), step)
    cs = corners(p, k, angle)
    if smooth > 0:
        p = _smooth(p, cs, smooth)
    out: List[Seg] = []
    for seg in _runs(p, cs):
        a, b = seg[0], seg[-1]
        L = float(np.hypot(*(b - a)))
        if L > 2.5:
            dv = (b - a) / L
            nr = np.array([-dv[1], dv[0]])
            if np.abs((seg - a) @ nr).max() <= err:
                out.append(("L", [tuple(map(float, a)), tuple(map(float, b))]))
                continue
        t1 = _unit(seg[min(3, len(seg) - 1)] - seg[0])
        t2 = _unit(seg[max(-4, -len(seg))] - seg[-1])
        out += [("C", [tuple(map(float, q)) for q in bz]) for bz in _fit(seg, t1, t2, err)]
    return out


def ring_points(segs: List[Seg], n: int = 40) -> List[Tuple[float, float]]:
    """The fitted ring as a polygon: lines by their ends, curves by `n` points."""
    pts: List[Tuple[float, float]] = []
    for kind, q in segs:
        if kind == "L":
            pts.append(q[0])
        else:
            pts += sample([tuple(q)], n)
    return pts


def trace(grey: np.ndarray, level: float = 128.0, zoom: int = 8, err: float = 0.42, min_points: int = 30):
    """The ink of a grey picture (ink dark) as clean outlines: (shapely geometry, list of rings as
    segments). Outer rings and their counters are told apart by nesting."""
    from PIL import Image
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    from skimage import measure
    img = Image.fromarray(np.clip(grey, 0, 255).astype(np.uint8))
    big = np.array(img.resize((img.width * zoom, img.height * zoom), Image.LANCZOS)).astype(float)
    rings = []
    for c in measure.find_contours(big, level):
        raw = [((x + 0.5) / zoom - 0.5, (y + 0.5) / zoom - 0.5) for y, x in c]
        if len(raw) < min_points:
            continue
        segs = fit_ring(raw, err=err, step=0.35, smooth=1.8, k=6, angle=38.0)
        if not segs:
            continue
        poly = Polygon(ring_points(segs)).buffer(0)
        if poly.is_empty:
            continue
        rings.append((poly.area, poly, segs))
    rings.sort(key=lambda t: -t[0])
    outers: List[dict] = []
    for _, poly, segs in rings:
        host = next((o for o in outers if o["poly"].buffer(-0.01).contains(poly)), None)
        if host is not None:
            host["holes"].append(poly)
            host["hole_segs"].append(segs)
        else:
            outers.append(dict(poly=poly, segs=segs, holes=[], hole_segs=[]))
    geoms = [o["poly"].difference(unary_union(o["holes"])) if o["holes"] else o["poly"] for o in outers]
    return unary_union(geoms), [o["segs"] for o in outers] + [s for o in outers for s in o["hole_segs"]]


def path_d(geom, T=lambda x, y: (x, y), err: float = 0.18) -> Tuple[str, int]:
    """An SVG path of a shapely geometry, written as real curves (fitted again on the clean outline).
    Returns the path and its number of segments."""
    d, n = "", 0
    for p in getattr(geom, "geoms", [geom]):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        for r in [p.exterior] + list(p.interiors):
            segs = fit_ring(np.array(r.coords)[:-1], err=err)
            if not segs:
                continue
            n += len(segs)
            d += "M%.2f,%.2f" % T(*segs[0][1][0])
            for kind, q in segs:
                if kind == "L":
                    d += " L%.2f,%.2f" % T(*q[-1])
                else:
                    d += " C" + " ".join("%.2f,%.2f" % T(*c) for c in q[1:])
            d += "Z"
    return d, n
