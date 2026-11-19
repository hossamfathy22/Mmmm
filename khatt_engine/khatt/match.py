"""Where each letter of a logo is, without reading it: the logo's word is known (the designer types it),
so it is set in a real typeface, where HarfBuzz says which ink is which letter, and laid over the logo.

The two do not have the same proportions (a diwani logo stretches one letter and squeezes the next), so
the typeface's word is not placed but warped onto the logo along the line: its columns of ink are
matched to the logo's columns of ink by dynamic time warping, each column described by where its ink
lies between the top and the bottom of the line. Every ink pixel of the logo then goes to the letter of
the warped typeface nearest it.

Nothing here needs the stems the engine reads a line by: it works on a stroke in any style. Several
typefaces are tried and the one that warps onto the logo at the least cost is kept.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

BANDS = 12              # a column's ink, in this many bands from the top of the line to its bottom
JOIN_TELLS = 0.3        # lines that differ this much in how joined their writing is are told apart by it
BAND = 0.25             # the warp may not stretch one side more than this share of the word's length ahead


def ref_labels(face, text: str, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """The word set in `face`, `height` pixels from the top of its ink to the bottom: a label image
    (0 no ink, 1 + the index of the character in `text`) and its ink."""
    from PIL import Image, ImageDraw
    from .bench.fonts import shape
    glyphs = shape(face, text)
    pts = [p for _, cs in glyphs for c in cs for p in c]
    if not pts:
        return np.zeros((1, 1), int), np.zeros((1, 1), bool)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    s = height / max(1e-6, max(ys) - min(ys))
    W, H = int((max(xs) - min(xs)) * s) + 2, height + 2
    lab = np.zeros((H, W), int)
    for cl, cs in glyphs:
        im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
        acc = np.zeros((H, W), int)
        for c in cs:                                   # even-odd: counters are holes
            one = Image.new("L", (W, H), 0)
            ImageDraw.Draw(one).polygon([((x - min(xs)) * s + 1, (max(ys) - y) * s + 1) for x, y in c], fill=1)
            acc ^= np.asarray(one, int)
        lab[acc > 0] = cl + 1
    return lab, lab > 0


def _profile(ink: np.ndarray) -> np.ndarray:
    """Each column of a line as where its ink lies, in BANDS bands from the top to the bottom."""
    H = ink.shape[0]
    edges = np.linspace(0, H, BANDS + 1).astype(int)
    f = np.stack([ink[edges[i]:max(edges[i] + 1, edges[i + 1])].mean(axis=0) for i in range(BANDS)], axis=1)
    return f


def _dtw(a: np.ndarray, b: np.ndarray, band: float = BAND) -> Tuple[float, np.ndarray]:
    """Dynamic time warping of the columns of `a` (n x d) onto those of `b` (m x d), within a band round
    the diagonal: the cost per column and, for every column of `a`, the column of `b` it is matched to."""
    n, m = len(a), len(b)
    w = max(int(band * max(n, m)), abs(n - m) + 2)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF); D[0, 0] = 0.0
    P = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):
        c = i * m / n
        j0, j1 = max(1, int(c - w)), min(m, int(c + w))
        cost = np.abs(b[j0 - 1:j1] - a[i - 1]).sum(axis=1)
        for jj, j in enumerate(range(j0, j1 + 1)):
            best, k = D[i - 1, j - 1], 0
            if D[i - 1, j] < best: best, k = D[i - 1, j], 1
            if D[i, j - 1] < best: best, k = D[i, j - 1], 2
            D[i, j] = cost[jj] + best; P[i, j] = k
    i, j = n, m
    col = np.zeros(n, int)
    while i > 0 and j > 0:
        col[i - 1] = j - 1
        k = P[i, j]
        if k == 0: i, j = i - 1, j - 1
        elif k == 1: i -= 1
        else: j -= 1
    return float(D[n, m] / (n + m)), col


def _rows(ink: np.ndarray) -> Tuple[int, int]:
    rows = np.flatnonzero(ink.any(axis=1))
    return (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, ink.shape[0])


def warp(logo: np.ndarray, face, text: str, step: int = 2) -> Optional[Dict]:
    """The typeface's word warped onto the logo's line `logo` (binary, rows down): a label image the
    size of the logo (0 no ink, 1 + the character's index) and the cost of the warp."""
    top, bot = _rows(logo)
    if bot - top < 4:
        return None
    line = logo[top:bot]
    lab, ink = ref_labels(face, text, bot - top)
    if lab.shape[1] < 4:
        return None
    cols = np.flatnonzero(line.any(axis=0))
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    A = _profile(line[:, c0:c1])[::step]
    rt, rb = _rows(ink)
    B = _profile(ink[rt:rb])[::step]
    if not len(A) or not len(B):
        return None
    cost, col = _dtw(A, B)
    # every column of the logo takes the typeface's column it was matched to, its rows scaled to the line
    full = np.zeros_like(logo, dtype=int)
    rr = np.clip(((np.arange(bot - top) / max(1, bot - top - 1)) * (rb - rt - 1)).astype(int) + rt, 0, lab.shape[0] - 1)
    for xi in range(c1 - c0):
        cb = min(lab.shape[1] - 1, col[min(len(col) - 1, xi // step)] * step)
        full[top:bot, c0 + xi] = lab[rr, cb]
    return {"labels": full, "cost": cost}


def assign(logo: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """Every ink pixel of the logo to the nearest letter of the warped typeface (0 where none is near)."""
    out = np.zeros_like(labels)
    if not labels.any():
        return out
    d = np.stack([ndi.distance_transform_edt(labels != k) for k in range(1, n + 1)])
    near = d.argmin(axis=0) + 1
    out[logo] = near[logo]
    return out


def match(logo: np.ndarray, text: str, faces: Sequence) -> Optional[Dict]:
    """The letters of `text` found in the logo's line: {"owner": label image (1 + index of the character
    in text, 0 no ink), "face": the typeface that warped best, "cost": its cost}."""
    best = None
    for face in faces:
        try:
            got = warp(logo, face, text)
        except Exception:                            # a typeface that cannot set the word
            continue
        if got and (best is None or got["cost"] < best["cost"]):
            best = {**got, "face": face}
    if best is None:
        return None
    best["owner"] = assign(logo, best["labels"], len(text))
    return best


def word_lines(ink: np.ndarray) -> List[np.ndarray]:
    """The lines of writing in a picture, as masks, by its pieces of ink rather than by empty rows (a
    tail or a dot of one line reaches into the rows of the next; a date or a signature in a corner sits
    in the rows of the word): the large pieces that overlap in height make a line; a small piece (a dot,
    a mark) goes to the line it sits on or next to; one far from every line (a corner's text) to none."""
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if not n:
        return []
    objs = ndi.find_objects(lab)
    area = ndi.sum(ink, lab, range(1, n + 1))
    big = [i for i in range(n) if area[i] >= 0.02 * area.max()]
    lines: List[Dict] = []
    for i in sorted(big, key=lambda i: (objs[i][0].start + objs[i][0].stop) / 2):
        t, b = objs[i][0].start, objs[i][0].stop
        for L in lines:
            ov = min(b, L["b"]) - max(t, L["t"])
            if ov >= 0.3 * min(b - t, L["b"] - L["t"]):
                L["t"], L["b"] = min(L["t"], t), max(L["b"], b); L["ids"].append(i)
                L["x0"] = min(L["x0"], objs[i][1].start); L["x1"] = max(L["x1"], objs[i][1].stop)
                break
        else:
            lines.append({"t": t, "b": b, "ids": [i], "x0": objs[i][1].start, "x1": objs[i][1].stop})
    # pieces of one word stacked over each other (a flourish over its letters, سكينة) overlap in height a
    # little, lie over the same stretch of the line, and hold much less ink than it; a Latin name under
    # the word leaves a gap
    merged = True
    while merged:
        merged = False
        for A in lines:
            for B in lines:
                if A is B:
                    continue
                ov_y = min(A["b"], B["b"]) - max(A["t"], B["t"])
                ov_x = min(A["x1"], B["x1"]) - max(A["x0"], B["x0"])
                ink_a, ink_b = sum(area[i] for i in A["ids"]), sum(area[i] for i in B["ids"])
                # two lines of writing of a size (روزيتا's two words) stay two, however their tails reach
                if ov_y > 0 and ov_x >= 0.5 * min(A["x1"] - A["x0"], B["x1"] - B["x0"]) and min(ink_a, ink_b) < 0.6 * max(ink_a, ink_b):
                    A["t"], A["b"] = min(A["t"], B["t"]), max(A["b"], B["b"])
                    A["x0"], A["x1"] = min(A["x0"], B["x0"]), max(A["x1"], B["x1"]); A["ids"] += B["ids"]
                    lines.remove(B); merged = True
                    break
            if merged:
                break
    for i in range(n):
        if i in big or area[i] < 2:
            continue
        cy, cx = (objs[i][0].start + objs[i][0].stop) / 2, (objs[i][1].start + objs[i][1].stop) / 2
        best, d0 = None, 1e18
        for L in lines:
            h = L["b"] - L["t"]
            dy = max(0, L["t"] - cy, cy - L["b"])
            dx = max(0, L["x0"] - cx, cx - L["x1"])
            if dy <= 0.6 * h and dx <= 0.3 * h and dy + dx < d0:
                best, d0 = L, dy + dx
        if best is not None:
            best["ids"].append(i)
    out = []
    for L in lines:
        m = np.isin(lab, [i + 1 for i in L["ids"]])
        if m.sum() >= 0.05 * ink.sum():
            out.append(m)
    return out


def joined_share(line: np.ndarray) -> float:
    """The share of a line's ink in pieces wider than the line is high: Arabic letters join into long
    pieces, Latin ones mostly stand apart (a script hand joins them too: then this does not tell)."""
    rows = np.flatnonzero(line.any(axis=1))
    if not len(rows):
        return 0.0
    h = rows[-1] - rows[0] + 1
    lab, n = ndi.label(line, structure=np.ones((3, 3)))
    objs = ndi.find_objects(lab)
    area = ndi.sum(line, lab, range(1, n + 1))
    wide = sum(a for o, a in zip(objs, area) if o[1].stop - o[1].start > 1.2 * h)
    return float(wide / max(1.0, line.sum()))


def locate(image: bytes, text: str, faces: Sequence, line: Optional[int] = None) -> Optional[Dict]:
    """The letters of the word `text` found in a logo's picture, on whichever of its lines the word is
    written: every line is matched against the word, and the one it warps onto at the least cost,
    weighed by how joined its writing is, is its line (or `line`, the index top first, where the
    designer says). Returns {"line": its mask, "owner": label image of the working picture, "cost",
    "face", "ink", "lines": [(top, cost, joined)]}."""
    from .guard import prepare
    work, ink, scale, inverted, notes = prepare(image)
    ink = np.asarray(ink, bool)
    masks = sorted(word_lines(ink), key=lambda m: int(np.flatnonzero(m.any(axis=1))[0]))
    if line is not None and 0 <= line < len(masks):
        masks = [masks[line]]
    found = []
    for m in masks:
        got = match(m, text, faces)
        if got is not None:
            found.append((m, got, joined_share(m)))
    if not found:
        return None
    # how joined each line's writing is tells Arabic from Latin only where the lines differ in it (a
    # Latin name in block letters, ALEXANDRIA 0.0 against 0.74); where both are joined (Kivano's touching
    # capitals 1.0, its Arabic 0.96) it would decide on a difference of nothing: the warp alone decides
    js = [j for _, _, j in found]
    tell = max(js) - min(js) > JOIN_TELLS
    score = lambda got, j: got["cost"] * ((1.5 - 0.5 * j) if tell else 1.0)
    m, got, j = min(found, key=lambda f: score(f[1], f[2]))
    best = {**got, "line": m}
    tried = [(int(np.flatnonzero(mm.any(axis=1))[0]), round(g["cost"], 4), round(jj, 2)) for mm, g, jj in found]
    best["ink"] = ink; best["lines"] = tried; best["scale"] = scale
    return best
