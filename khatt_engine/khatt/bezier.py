"""Fit cubic Bezier curves to traced outlines (Philip J. Schneider, "An Algorithm for
Automatically Fitting Digitized Curves", Graphics Gems, 1990), with corner detection."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

Pt = Tuple[float, float]
Cubic = Tuple[Pt, Pt, Pt, Pt]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(*v))
    return v / n if n > 1e-12 else v


def _bez(b: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]; m = 1 - t
    return m**3 * b[0] + 3 * m * m * t * b[1] + 3 * m * t * t * b[2] + t**3 * b[3]


def _bez1(b: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]; m = 1 - t
    return 3 * m * m * (b[1] - b[0]) + 6 * m * t * (b[2] - b[1]) + 3 * t * t * (b[3] - b[2])


def _bez2(b: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    return 6 * (1 - t) * (b[2] - 2 * b[1] + b[0]) + 6 * t * (b[3] - 2 * b[2] + b[1])


def _chord(p: np.ndarray) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(p))


def _generate(p: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    a1 = np.outer(3 * (1 - u) ** 2 * u, t1)
    a2 = np.outer(3 * (1 - u) * u ** 2, t2)
    c00, c01, c11 = (a1 * a1).sum(), (a1 * a2).sum(), (a2 * a2).sum()
    first, last = p[0], p[-1]
    base = _bez(np.array([first, first, last, last]), u)
    tmp = p - base
    x0, x1 = (a1 * tmp).sum(), (a2 * tmp).sum()
    det = c00 * c11 - c01 * c01
    seg = float(np.hypot(*(last - first)))
    if abs(det) > 1e-12:
        al, ar = (x0 * c11 - x1 * c01) / det, (c00 * x1 - c01 * x0) / det
    else:
        al = ar = seg / 3
    eps = 1e-6 * seg
    if al < eps or ar < eps:
        al = ar = seg / 3
    return np.array([first, first + t1 * al, last + t2 * ar, last])


def _reparam(b: np.ndarray, p: np.ndarray, u: np.ndarray) -> np.ndarray:
    d = _bez(b, u) - p
    d1, d2 = _bez1(b, u), _bez2(b, u)
    num = (d * d1).sum(axis=1)
    den = (d1 * d1).sum(axis=1) + (d * d2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(np.abs(den) > 1e-12, u - num / den, u)
    return np.clip(nu, 0, 1)


def _fit(p: np.ndarray, t1: np.ndarray, t2: np.ndarray, err: float, depth: int = 0) -> List[np.ndarray]:
    if len(p) == 2 or depth > 30:
        d = float(np.hypot(*(p[-1] - p[0]))) / 3
        return [np.array([p[0], p[0] + t1 * d, p[-1] + t2 * d, p[-1]])]
    u = _chord(p)
    b = _generate(p, u, t1, t2)
    e = np.hypot(*(_bez(b, u) - p).T)
    if e.max() < err:
        return [b]
    if e.max() < 4 * err:
        for _ in range(20):
            u = _reparam(b, p, u)
            b = _generate(p, u, t1, t2)
            e = np.hypot(*(_bez(b, u) - p).T)
            if e.max() < err:
                return [b]
    split = int(np.clip(np.argmax(e), 1, len(p) - 2))
    tc = _unit(p[split - 1] - p[split + 1])
    return _fit(p[:split + 1], t1, tc, err, depth + 1) + _fit(p[split:], -tc, t2, err, depth + 1)


def corners(p: np.ndarray, angle: float = 40.0, k: int = 3) -> List[int]:
    """Indexes where the closed outline turns sharply (cut ends of stems, joins)."""
    n = len(p)
    if n < 2 * k + 3:
        return []
    a = np.roll(p, k, axis=0); b = np.roll(p, -k, axis=0)
    v1, v2 = p - a, b - p
    cos = (v1 * v2).sum(1) / (np.hypot(*v1.T) * np.hypot(*v2.T) + 1e-12)
    turn = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    out = []
    for i in range(n):
        if turn[i] >= angle and turn[i] == max(turn[(i + j) % n] for j in range(-k, k + 1)):
            if not out or i - out[-1] > k:
                out.append(i)
    return out


def fit_closed(points: Sequence[Pt], err: float = 0.5) -> List[Cubic]:
    """Fit a closed outline with cubic Beziers; corners stay sharp."""
    p = np.asarray(points, float)
    if len(p) > 2 and np.hypot(*(p[0] - p[-1])) < 1e-9:
        p = p[:-1]
    if len(p) < 4:
        return []
    cs = corners(p)
    if len(cs) < 2:
        # no corners: split at the extremes so every piece is a gentle arc
        cs = sorted({int(np.argmin(p[:, 0])), int(np.argmax(p[:, 1])), int(np.argmax(p[:, 0])), int(np.argmin(p[:, 1]))})
    out: List[Cubic] = []
    for i, c in enumerate(cs):
        d = cs[(i + 1) % len(cs)]
        seg = p[c:d + 1] if d > c else np.r_[p[c:], p[:d + 1]]
        if len(seg) < 2:
            continue
        t1 = _unit(seg[min(2, len(seg) - 1)] - seg[0])
        t2 = _unit(seg[max(-3, -len(seg))] - seg[-1])
        for b in _fit(seg, t1, t2, err):
            out.append(tuple(tuple(map(float, q)) for q in b))
    return out


def sample(curves: List[Cubic], n: int = 10) -> List[Pt]:
    """Turn Bezier segments back into a polygon."""
    pts: List[Pt] = []
    t = np.linspace(0, 1, n, endpoint=False)
    for c in curves:
        pts += [tuple(map(float, q)) for q in _bez(np.array(c), t)]
    return pts


def to_path(curves: List[Cubic], fmt=lambda x, y: f"{x:.2f},{y:.2f}") -> str:
    if not curves:
        return ""
    s = "M" + fmt(*curves[0][0])
    for c in curves:
        s += " C" + " ".join(fmt(*q) for q in c[1:])
    return s + "Z"
