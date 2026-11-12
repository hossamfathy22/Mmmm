"""Analysis by synthesis: redraw the logo's own text with the engine, compare it with the image,
and adjust all the rules together until the redrawing matches the logo as closely as the part
families allow.

The measured rules are the starting point and also a prior: moving a rule away from what was
measured costs something, so the fit cannot bend the rules to imitate a letter the engine draws
differently. The result is a maximum-a-posteriori estimate:

    score = IoU(redrawing, logo)  -  lam * sum(((p - p_measured) / sigma)^2)
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from .measure import Line
from .style import Style
from .typeset import Layout, flat, typeset

# parameter, sigma (absolute, or relative when the value is a fraction of a size), lower bound, upper bound
PARAMS: List[Tuple[str, str, float]] = [
    # rules measured to sub-pixel accuracy get a tight prior; shapes and spacing are free to move
    ("stem", "rel", 0.015), ("hair", "rel", 0.08), ("x_height", "rel", 0.01), ("white", "rel", 0.15),
    ("slant", "abs", 0.3), ("contrast_power", "abs", 0.4), ("join_in", "abs", 0.08), ("join_out", "abs", 0.08),
    ("top_round", "abs", 0.2), ("top_angle", "abs", 4.0), ("dot", "rel", 0.04), ("group_gap", "rel", 0.30),
    ("tail_reach", "abs", 0.6), ("ascender", "rel", 0.10), ("descender", "rel", 0.15), ("seen_drop", "abs", 3.0),
    ("ring_n", "abs", 0.6),
]
NAMES_AR = {"stem": "سمك الجذع", "hair": "الخط الشعري", "x_height": "ارتفاع الحروف", "white": "الأبيض بين الجذوع",
            "slant": "الميل", "contrast_power": "أس قاعدة القلم", "join_in": "عمق الوصل الداخلي", "join_out": "عمق الوصل الخارجي",
            "top_round": "استدارة طرف الجذع", "top_angle": "زاوية طرف الجذع", "dot": "حجم النقطة", "group_gap": "الأبيض بين الحروف",
            "tail_reach": "امتداد الذيل", "ascender": "الصعود", "descender": "النزول", "seen_drop": "انخفاض الأسنان", "ring_n": "تربيع الرأس"}
LAM = 0.0015


def _shifted_iou(m: np.ndarray, t: np.ndarray, c0: int, c1: int, r: int) -> float:
    """IoU of one letter's columns, after sliding the redrawing by the best horizontal offset:
    judges the letter's shape, not the spacing errors that accumulate before it."""
    best = 0.0
    W = m.shape[1]
    for dxs in range(-r, r + 1, max(1, r // 8)):
        a0, a1 = max(0, c0 + dxs), min(W, c1 + dxs)
        if a1 - a0 < 2:
            continue
        mm = m[:, a0 - dxs:a1 - dxs] if 0 <= a0 - dxs and a1 - dxs <= W else None
        if mm is None or mm.shape[1] != a1 - a0:
            continue
        best = max(best, _iou(mm, t[:, a0:a1]))
    return best


@dataclass
class Fit:
    text: str
    iou_before: float
    iou_after: float
    style: Style
    changes: List[Dict] = field(default_factory=list)
    letters: List[Dict] = field(default_factory=list)
    outliers: List[str] = field(default_factory=list)
    overlay_png: bytes = b""
    evaluations: int = 0


def _draw(shapes, tf, size: Tuple[int, int], ss: int) -> np.ndarray:
    H, W = size
    mask = Image.new("L", (W * ss, H * ss), 0)
    for sh in shapes:
        polys = [[tf(x, y) for x, y in poly] for poly in sh if len(poly) >= 3]
        if not polys:
            continue
        if len(polys) == 1:
            ImageDraw.Draw(mask).polygon(polys[0], fill=255)
            continue
        part = None
        for p in polys:
            layer = Image.new("1", mask.size, 0)
            ImageDraw.Draw(layer).polygon(p, fill=1)
            part = layer if part is None else ImageChops.logical_xor(part, layer)
        mask = ImageChops.lighter(mask, part.convert("L"))
    return np.asarray(mask.resize((W, H), Image.BILINEAR)) >= 128


def groups_of(lay: Layout) -> List[List[int]]:
    """Indexes of placed letters, split into groups that are joined at the baseline."""
    out: List[List[int]] = []
    for i, pl in enumerate(lay.placed):
        if out and pl.form in ("M", "F"):
            out[-1].append(i)
        else:
            out.append([i])
    return out


def rasterize(lay: Layout, scale: float, shape: Tuple[int, int], right: float, base_row: float,
              ss: int = 2, target: Optional[np.ndarray] = None, reach: float = 0.0, v_reach: float = 0.0):
    """Draw a layout into an image of `shape` (rows, cols), right edge at column `right`, baseline at
    `base_row`. With a target, every joined group is slid to its best horizontal position (within
    +-reach px), so the score judges shapes; the slides themselves measure the spacing.
    Returns (mask, letter spans, group offsets)."""
    H, W = shape
    t = lay.style.tan
    pts_x = [(x + y * t) * scale for s in lay.shapes for poly in s for x, y in poly]
    if not pts_x:
        return np.zeros(shape, bool), [], []
    dx = right - max(pts_x)
    if target is not None and v_reach > 0:
        # the baseline found in the image may sit a little off for scripts without a flat band (Naskh):
        # find the vertical offset that makes the whole redrawing overlap the logo best
        tf0 = lambda x, y: ((x * scale + y * scale * t + dx) * ss, (base_row - y * scale) * ss)
        whole = _draw([sh for sh in lay.shapes], tf0, shape, ss)
        best_v, best_ov = 0, -1
        for dv in range(-int(v_reach), int(v_reach) + 1):
            mv = np.roll(whole, dv, axis=0)
            if dv > 0:
                mv[:dv] = False
            elif dv < 0:
                mv[dv:] = False
            # horizontal tolerance: compare column-summed profiles as well as pixels
            ov = int(np.logical_and(mv, target).sum())
            if ov > best_ov:
                best_ov, best_v = ov, dv
        base_row += best_v
    total = np.zeros(shape, bool)
    spans, offsets = [], []
    for g in groups_of(lay):
        shapes = []
        for i in g:
            pl = lay.placed[i]
            shapes += [[[(x + pl.x, y) for x, y in poly] for poly in sh] for sh in flat(pl.glyph.shapes)]
        lo = min(lay.placed[i].x + lay.placed[i].glyph.left for i in g)
        hi = max(lay.placed[i].x for i in g)
        for a, b in lay.joins:                       # the joins inside this group
            if lo - 1 <= min(a, b) and max(a, b) <= hi + 1:
                shapes.append([[(a, 0), (b, 0), (b, lay.style.hair), (a, lay.style.hair)]])
        tf = lambda x, y: ((x * scale + y * scale * t + dx) * ss, (base_row - y * scale) * ss)
        m = _draw(shapes, tf, shape, ss)
        shift = 0
        if target is not None and reach > 0 and m.any():
            cols = np.flatnonzero(m.any(axis=0))
            c0, c1 = cols[0], cols[-1] + 1
            best = -1
            for d in range(-int(reach), int(reach) + 1):
                a0, a1 = c0 + d, c1 + d
                if a0 < 0 or a1 > W:
                    continue
                ov = int(np.logical_and(m[:, c0:c1], target[:, a0:a1]).sum())
                if ov > best:
                    best, shift = ov, d
            if shift:
                m = np.roll(m, shift, axis=1)
        total |= m
        offsets.append(shift)
        for i in g:
            pl = lay.placed[i]
            xs = [(x + pl.x + y * t) * scale + dx + shift for sh in flat(pl.glyph.shapes) for poly in sh for x, y in poly if y >= 0]
            if xs:
                spans.append((pl.char, min(xs), max(xs)))
    return total, spans, offsets


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def fit_line(L: Line, ink: np.ndarray, text: str, st0: Style, scale: float, max_evals: int = 420) -> Optional[Fit]:
    """L: measured line, ink: working-image binary, st0: rules in original pixels, scale: working / original."""
    pad = int(0.6 * L.stem)
    r0, r1 = max(0, L.top - pad), min(ink.shape[0], L.bot + pad + 1)
    target = ink[r0:r1].astype(bool)
    base_row = (L.base_row - r0) - L.baseline_shift            # the measured baseline inside the crop
    right = float(L.x1) + 1.0

    reach = 3.0 * L.stem

    def render(st: Style, with_offsets: bool = False):
        lay = typeset(text, st)
        if not lay.shapes:
            return (None, None, [], []) if with_offsets else (None, None, [])
        m, spans, offs = rasterize(lay, scale, target.shape, right, base_row, target=target, reach=reach)
        return (lay, m, spans, offs) if with_offsets else (lay, m, spans)

    def sig(name, kind, s, p0):
        return s * max(1e-6, abs(p0)) if kind == "rel" else s

    p0 = {n: float(getattr(st0, n)) for n, _, _ in PARAMS}
    if p0.get("group_gap", 0) <= 0:
        p0["group_gap"] = 1.15 * st0.white
    sigmas = {n: sig(n, k, s, p0[n]) for n, k, s in PARAMS}

    def make(p):
        st = Style(**{**st0.__dict__})
        for n, v in p.items():
            setattr(st, n, v)
        st.top_round = min(max(st.top_round, 0.0), 1.0)
        st.join_in = min(max(st.join_in, 0.08), 0.85)
        st.join_out = min(max(st.join_out, 0.08), 0.85)
        st.seen_drop = max(0.0, st.seen_drop)
        st.ring_n = max(1.5, st.ring_n)
        st.contrast_power = min(max(st.contrast_power, 0.6), 8.0)
        return st

    evals = 0
    # robust fitting: letters the part families draw very differently from the logo are left out of
    # the score, so they cannot pull the rules (they are reported instead)
    _, m_init, spans0 = render(make(p0))
    keep = np.ones(target.shape[1], bool)
    outliers: List[str] = []
    if m_init is not None and spans0:
        r_shift = int(0.6 * L.stem)
        scores0 = []
        for ch, a, b in spans0:
            c0, c1 = max(0, int(a)), min(target.shape[1], int(math.ceil(b)) + 1)
            scores0.append((ch, c0, c1, _shifted_iou(m_init, target, c0, c1, r_shift) if c1 > c0 else 0.0))
        med = float(np.median([v for *_, v in scores0])) if scores0 else 0.0
        for ch, c0, c1, v in scores0:
            if v < max(0.45, med - 0.25):
                keep[max(0, c0 - int(0.3 * L.stem)):c1] = False
                outliers.append(ch)

    def score(p) -> Tuple[float, float]:
        nonlocal evals
        evals += 1
        _, m, _ = render(make(p))
        if m is None:
            return -1.0, 0.0
        iou = _iou(m[:, keep], target[:, keep])
        prior = sum(((p[n] - p0[n]) / sigmas[n]) ** 2 for n in p)
        return iou - LAM * prior, iou

    cur = dict(p0)
    best, _ = score(cur)
    iou0 = _iou(m_init, target) if m_init is not None else 0.0
    steps = {n: sigmas[n] for n in cur}
    # pattern search: try each rule up and down, keep what helps, then refine the steps
    while evals < max_evals:
        improved = False
        for n in list(cur):
            for d in (+1, -1):
                trial = dict(cur)
                trial[n] = cur[n] + d * steps[n]
                s, _ = score(trial)
                if s > best + 1e-5:
                    best, cur, improved = s, trial, True
                    break
            if evals >= max_evals:
                break
        if not improved:
            steps = {n: v / 2 for n, v in steps.items()}
            if all(steps[n] < sigmas[n] / 8 for n in steps):
                break
    # spacing: the slides needed between neighbouring groups measure how wrong the gap is
    for _ in range(3):
        lay_s, _, _, offs = render(make(cur), with_offsets=True)
        if lay_s is None or len(offs) < 2:
            break
        grp = groups_of(lay_s)
        clean = [not any(lay_s.placed[i].char in outliers for i in g) for g in grp]
        diffs = [offs[i + 1] - offs[i] for i in range(len(offs) - 1) if clean[i] and clean[i + 1]]
        if not diffs:
            break                                          # no trustworthy pair of groups to measure the gap
        d = float(np.median(diffs)) / scale               # +: the next group sits too far left
        if abs(d) < 0.3:
            break
        cur["group_gap"] = max(0.3 * st0.white, cur["group_gap"] - d)
    st = make(cur)
    lay, m, spans = render(st)
    iou1 = _iou(m, target)
    _, m0, _ = render(make(p0))
    changes = []
    for n, _, _ in PARAMS:
        a, b = p0[n], getattr(st, n)
        if abs(b - a) > 0.02 * sigmas[n] * 8:
            changes.append({"key": n, "name": NAMES_AR.get(n, n), "measured": round(a, 3), "fitted": round(b, 3),
                            "shift_sigma": round((b - a) / sigmas[n], 2)})
    letters = []
    for ch, a, b in spans:
        c0, c1 = max(0, int(a)), min(target.shape[1], int(math.ceil(b)) + 1)
        if c1 > c0:
            letters.append({"letter": ch, "iou": round(_shifted_iou(m, target, c0, c1, int(0.6 * L.stem)), 3)})
    # a letter is reported as drawn differently only if it still differs after the fit
    final_scores = [x["iou"] for x in letters]
    med1 = float(np.median(final_scores)) if final_scores else 0.0
    outliers = sorted({x["letter"] for x in letters if x["iou"] < max(0.45, med1 - 0.25)})
    for x in letters:
        x["outlier"] = x["letter"] in outliers
    # overlay: both = dark, only in the logo = red (not reproduced), only in the redrawing = blue (extra)
    rgb = np.full(target.shape + (3,), 255, np.uint8)
    rgb[target & m] = (40, 48, 64)
    rgb[target & ~m] = (220, 60, 70)
    rgb[~target & m] = (40, 130, 230)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", optimize=True)
    return Fit(text=text, iou_before=round(iou0, 4), iou_after=round(iou1, 4), style=st, changes=changes,
               letters=letters, overlay_png=buf.getvalue(), evaluations=evals, outliers=outliers)
