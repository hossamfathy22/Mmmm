"""The judgements a designer makes by eye, turned into measurements.

For every logo the engine measures *how* its parts are shaped, not only how big they are:
  top_angle, top_round   how stems are cut at the top (flat, angled, rounded)
  join_in, join_out      how deep the curves of a join between two teeth go (fraction of x-height)
  ring_n, ring_ratio     how square the closed counters are (superellipse exponent), and their proportion
  dot_n, dot_rot         dot shape: circle (n=2), square (n large), diamond (rotated 45°)
  tail_reach             how far a tail sweeps, in stems
  overshoot              how far round bottoms dip below the flat ones
  seen_drop              how much lower the inner teeth of a run are
  group_gap              white between letters that do not join
Each value comes with its evidence (how many samples) so the report can say where it came from.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .measure import Line, _cross, _runs
from .understand import Reading


def _median(v: List[float]) -> Optional[float]:
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else None


def superellipse_area_ratio(n: float) -> float:
    """Area of |x|^n + |y|^n = 1 divided by its bounding box."""
    return math.gamma(1 + 1 / n) ** 2 / math.gamma(1 + 2 / n)


def superellipse_n(ratio: float) -> float:
    lo, hi = 1.05, 12.0
    ratio = min(max(ratio, superellipse_area_ratio(lo) + 1e-6), superellipse_area_ratio(hi) - 1e-6)
    for _ in range(60):
        mid = (lo + hi) / 2
        if superellipse_area_ratio(mid) < ratio:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _poly_area(p: List[Tuple[float, float]]) -> float:
    a = np.asarray(p)
    x, y = a[:, 0], a[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# ------------------------------------------------------------------ stem terminals
def _terminal(L: Line, s) -> Optional[Tuple[float, float]]:
    """Fit the top edge of one stem: a straight line (its angle) plus how much the corners drop."""
    r_mid = int(round(L.row_of_y((s.top + max(s.bottom, 0)) / 2)))
    xs, ys = [], []
    for c in np.arange(L.col_of_x(s.x - 0.49 * s.w), L.col_of_x(s.x + 0.49 * s.w) + 1e-6, 0.5):
        ci = int(round(c))
        col = L.G[:, ci]
        if not (0 <= r_mid < len(col)) or col[r_mid] < L.level:
            continue
        e = _cross(col, L.level, r_mid, -1)
        if e is None:
            continue
        xs.append((L.x_of_col(ci) - s.x) / (s.w / 2))
        ys.append(L.y_of_row(e))
    if len(xs) < 6:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    mid = np.abs(xs) < 0.5
    if mid.sum() < 3:
        return None
    b, a = np.polyfit(xs[mid] * s.w / 2, ys[mid], 1)
    drop = (a + b * xs * s.w / 2) - ys                     # how far below the straight cut each column ends
    # fit the radius of rounded corners: a corner of radius rho (fraction of the half width)
    # drops by R(1 - sqrt(1 - ((|u| - (1 - rho)) / rho)^2)) with R = rho * w / 2
    best, best_err = 0.0, float("inf")
    hw = s.w / 2
    for rho in np.linspace(0, 1, 51):
        if rho == 0:
            pred = np.zeros_like(xs)
        else:
            t = np.clip((np.abs(xs) - (1 - rho)) / rho, 0, 1)
            pred = rho * hw * (1 - np.sqrt(1 - t ** 2))
        err = float(np.sum((pred - drop) ** 2))
        if err < best_err:
            best, best_err = float(rho), err
    return math.degrees(math.atan(b)), best


# ------------------------------------------------------------------ joins
def _join_profile(L: Line, R: Reading) -> Tuple[List[float], List[float]]:
    """For each join: where the inner curve of the counter starts (height above the counter bottom),
    and where the outer edge of a free stem starts to round off."""
    ins, outs = [], []
    X = L.x_height
    for j in R.joins:
        if j.kind != "join":
            continue
        a, b = L.stems[j.right], L.stems[j.left]
        c0, c1 = L.col_of_x(b.x + b.w / 2), L.col_of_x(a.x - a.w / 2)
        if c1 - c0 < 3:
            continue
        full = (c1 - c0)
        mc = int(round((c0 + c1) / 2))
        prof = L.G[:, mc - 1:mc + 2].mean(axis=1)
        r0 = min(len(prof) - 1, max(0, int(round(L.row_of_y(0.0)))))
        # the bottom of the counter: walk up the middle column from the baseline until the ink ends
        r = r0
        while r > 0 and prof[r] < L.level and r > r0 - 0.3 * L.stem:
            r -= 1                                    # skip white just under a slightly raised join
        while r > 0 and prof[r] >= L.level:
            r -= 1
        y_bottom = L.y_of_row(r)
        for rr in range(min(r, L.G.shape[0] - 1), max(0, int(L.row_of_y(0.9 * X))), -1):
            w = float((L.G[rr, int(c0) - 1:int(c1) + 2] < L.level).sum())
            if w >= 0.92 * full:
                ins.append(L.y_of_row(rr) / X)
                break
    for i, s in enumerate(L.stems):
        free_r = (i == 0 or L.stems[i - 1].comp != s.comp) and not s.joined_right
        free_l = (i == len(L.stems) - 1 or L.stems[i + 1].comp != s.comp) and not s.joined_left
        for side, free in ((+1, free_r), (-1, free_l)):
            if not free or s.role not in ("tooth", "lowered", "tall") or abs(s.bottom) > 0.5 * L.stem:
                continue
            edge = s.x + side * s.w / 2
            start = None
            for y in np.linspace(0.7 * X, 0.0, 60):
                r = min(L.G.shape[0] - 1, max(0, int(round(L.row_of_y(y)))))
                row = L.G[r]
                ci = int(round(L.col_of_x(s.x)))
                e = _cross(row, L.level, ci, side)
                if e is None:
                    break
                if abs(L.x_of_col(e) - edge) > 0.06 * s.w:
                    start = y
                    break
            outs.append((start if start is not None else 0.0) / X)
    return ins, outs


# ------------------------------------------------------------------ rings and dots
def _ring_shape(lines: List[Line]) -> Tuple[Optional[float], Optional[float], int]:
    ns, ratios = [], []
    for L in lines:
        for c in L.counters:
            w, h = c.x1 - c.x0, c.y1 - c.y0
            if not (0.4 * L.stem <= w <= 3.5 * L.stem and h >= 0.3 * L.x_height):
                continue
            # the counter is measured upright (deskewed), so its fill ratio is not disturbed by the slant
            ns.append(superellipse_n(c.area / max(1.0, w * h)))
            ratios.append(h / max(1e-6, L.x_height))
    return _median(ns), _median(ratios), len(ns)


def _dot_shape(lines: List[Line]) -> Tuple[Optional[float], float, int, Optional[float]]:
    n_vals, rots, sizes = [], [], []
    for L in lines:
        for p in L.dots:
            for poly in p.contours[:1]:
                # back to the image's own orientation (re-apply the slant)
                q = [(x + y * L.t, y) for x, y in poly]
                A = _poly_area(q)
                xs, ys = [a for a, _ in q], [b for _, b in q]
                r0 = A / max(1e-6, (max(xs) - min(xs)) * (max(ys) - min(ys)))
                rq = [((x - y) / math.sqrt(2), (x + y) / math.sqrt(2)) for x, y in q]
                xs2, ys2 = [a for a, _ in rq], [b for _, b in rq]
                r45 = A / max(1e-6, (max(xs2) - min(xs2)) * (max(ys2) - min(ys2)))
                ratio = max(r0, r45)
                n_vals.append(superellipse_n(ratio)); rots.append(45.0 if r45 > r0 + 0.04 else 0.0)
                sizes.append(math.sqrt(A / superellipse_area_ratio(n_vals[-1])))   # side of its box
    if not n_vals:
        return None, 0.0, 0, None
    return _median(n_vals), (45.0 if rots.count(45.0) > len(rots) / 2 else 0.0), len(n_vals), _median(sizes)


# ------------------------------------------------------------------ overshoot, gaps
def _overshoot(lines: List[Line]) -> Tuple[Optional[float], int]:
    vals = []
    for L in lines:
        flat = [s.bottom for s in L.stems if s.role in ("tooth", "tall") and not s.joined_left and not s.joined_right
                and abs(s.bottom) < 0.08 * L.stem]
        rounds = []
        for c in L.counters:
            col = int(round(L.col_of_x((c.x0 + c.x1) / 2)))
            r = int(round(L.row_of_y(c.y0))) + 1
            prof = L.G[:, col]
            if 0 <= r < len(prof) and prof[r] >= L.level:
                e = _cross(prof, L.level, r, +1)
                if e is not None:
                    rounds.append(L.y_of_row(e))
        if len(flat) >= 2 and rounds:          # two flat feet on the baseline at least, or no evidence
            vals.append(float(np.median(flat)) - float(np.median(rounds)))
    v = _median(vals)
    return (max(0.0, v) if v is not None else None), len(vals)


def _group_gap(L: Line, R: Reading) -> Optional[float]:
    """White between neighbouring letter groups that do not touch, at mid x-height."""
    X = L.x_height
    gaps = []
    comps = sorted({s.comp for s in L.stems}, key=lambda c: -max(s.x for s in L.stems if s.comp == c))
    r = int(round(L.row_of_y(0.5 * X)))
    for ca, cb in zip(comps, comps[1:]):
        left_edge_a = min((s.x - s.w / 2) for s in L.stems if s.comp == ca)
        right_edge_b = max((s.x + s.w / 2) for s in L.stems if s.comp == cb)
        g = left_edge_a - right_edge_b
        if 0 < g < 4 * L.stem:
            gaps.append(g)
    return _median(gaps)


# ------------------------------------------------------------------ all together
def measure_features(lines: List[Line], readings: List[Reading], src: int, k: float) -> Dict[str, Dict]:
    """Returns {name: {"value": v, "samples": n, "unit": ..., "note": Arabic text}} (lengths in original px)."""
    L, R = lines[src], readings[src]
    out: Dict[str, Dict] = {}

    def put(name, value, samples, note, unit=""):
        if value is not None:
            out[name] = {"value": round(float(value), 3), "samples": int(samples), "note": note, "unit": unit}

    terms = [t for t in (_terminal(L, s) for s in L.stems if s.role in ("tooth", "tall", "lowered")) if t]
    if terms:
        ang = _median([t[0] for t in terms]); rnd = _median([t[1] for t in terms])
        put("top_angle", ang if abs(ang) >= 2.5 else 0.0, len(terms), "زاوية قطع طرف الجذع العلوي", "°")
        put("top_round", rnd if rnd >= 0.12 else 0.0, len(terms), "استدارة زوايا طرف الجذع (0 مقطوع، 1 نصف دائرة)")
    ins, outs = _join_profile(L, R)
    put("join_in", _median(ins), len(ins), "ارتفاع بداية منحنى الفراغ داخل الوصل ÷ ارتفاع الحروف")
    put("join_out", _median(outs), len(outs), "ارتفاع بداية استدارة الحافة الخارجية الحرة ÷ ارتفاع الحروف")
    n, ratio, cnt = _ring_shape(lines)
    put("ring_n", n, cnt, "تربيع الفراغات المغلقة (2 بيضاوي، أكبر = أقرب للمربع)")
    put("ring_ratio", ratio, cnt, "ارتفاع الفراغ المغلق ÷ ارتفاع الحروف")
    dn, drot, dc, dsize = _dot_shape(lines)
    put("dot_n", dn, dc, "شكل النقطة (2 دائرة، أكبر = أقرب للمربع)")
    put("dot_size", dsize * k if dsize is not None else None, dc, "حجم النقطة (ضلع صندوقها)", "px")
    if dc:
        put("dot_rot", drot, dc, "دوران النقطة (45° = معيّن)", "°")
    reaches = [L.stems[t.stem].x - t.x0 for t in R.tails if t.stem is not None and t.direction == "left"]
    put("tail_reach", (_median(reaches) / L.stem) if reaches else None, len(reaches), "امتداد الذيل إلى اليسار بوحدة الجذع")
    ov, oc = _overshoot(lines)
    put("overshoot", ov * k if ov is not None else None, oc, "نزول القيعان المستديرة تحت القيعان المسطحة", "px")
    lows = [v for v in L.levels if v.name == "lowered" and v.n >= 2]
    put("seen_drop", ((L.x_height - lows[0].y) * k) if lows else None, lows[0].n if lows else 0,
        "انخفاض الأسنان الداخلية عن ارتفاع الحروف", "px")
    gg = _group_gap(L, R)
    put("group_gap", gg * k if gg is not None else None, 1, "الأبيض بين حرفين غير متصلين", "px")
    return out
