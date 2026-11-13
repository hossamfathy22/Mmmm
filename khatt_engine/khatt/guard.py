"""The input guard: make the picture measurable, say what was done to it, and say how far the
reading can be trusted.

The measuring engine assumes flat ink on a flat ground, a level baseline, solid letters and
nothing but writing in the line. A logo file often breaks one of these, and the numbers then come
out wrong without any sign. Before anything is measured this module

  - evens out a ground that is not flat (a gradient, a vignette),
  - levels a rotated picture (rotation of the baseline, which is not the slant of the letters),
  - fills letters drawn as outlines only,
  - takes a symbol standing beside the writing out of the line,

and every correction is reported. After measuring, `assess` lists what still looks unreliable.
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .analyze import MAX_SIDE, MIN_SIDE, _otsu, find_lines

Note = Dict[str, Any]


def _note(kind: str, text: str, **ev) -> Note:
    return {"kind": kind, "text": text, **{k: (round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v)
                                         for k, v in ev.items()}}


# ------------------------------------------------------------------ loading
def _load_gray(data: bytes) -> Tuple[Image.Image, float]:
    img = Image.open(io.BytesIO(data))
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
    img = img.convert("L")
    w, h = img.size
    scale = min(1.0, MAX_SIDE / max(w, h))
    if max(w, h) * scale < MIN_SIDE:
        scale = min(3.0, MIN_SIDE / max(w, h))
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), scale


def _polarity(gray: np.ndarray) -> Tuple[int, bool]:
    th = _otsu(gray)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return th, bool((border <= th).mean() > 0.5)


# ------------------------------------------------------------------ 1. uneven ground
def _smooth_surface(v: np.ndarray) -> np.ndarray:
    """The best smooth (quadratic) surface through `v`, ignoring what does not belong to it: a solid
    shape left on the ground is an outlier of the fit, a gradient or a vignette is the fit itself."""
    H, W = v.shape
    yy, xx = np.mgrid[0:H, 0:W]
    x = (xx / max(W - 1, 1) - 0.5).ravel(); y = (yy / max(H - 1, 1) - 0.5).ravel()
    A = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y]
    z = v.ravel().astype(float)
    keep = np.ones(len(z), bool)
    coef = np.zeros(6)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A[keep], z[keep], rcond=None)
        res = z - A @ coef
        mad = float(np.median(np.abs(res[keep] - np.median(res[keep])))) + 1.0
        new = np.abs(res) <= 3.5 * mad
        if new.sum() < 0.3 * len(z) or (new == keep).all():
            break
        keep = new
    return (A @ coef).reshape(H, W)


def flatten_ground(gray: np.ndarray, inverted: bool) -> Tuple[np.ndarray, float]:
    """The ground as a smooth surface, and the picture with that surface taken out. The writing is
    wiped away first (everything narrower than a third of the picture), then a smooth surface is
    laid through what is left. Returns the corrected picture and how uneven the ground was, in
    grey levels; a ground uneven by less than 16 levels is left alone."""
    H, W = gray.shape
    size = max(9, int(0.33 * min(H, W)) | 1)
    small = max(1, int(round(max(H, W) / 240)))            # work on a reduced copy: the ground is smooth
    g = gray[::small, ::small].astype(np.float32)
    k = max(3, size // small) | 1
    if inverted:
        wiped = ndi.maximum_filter(ndi.minimum_filter(g, k), k)        # opening: wipe light writing
    else:
        wiped = ndi.minimum_filter(ndi.maximum_filter(g, k), k)        # closing: wipe dark writing
    ground = _smooth_surface(wiped)
    uneven = float(ground.max() - ground.min())
    if uneven < 16:
        return gray, uneven
    ground = ndi.zoom(ground, (H / ground.shape[0], W / ground.shape[1]), order=1)[:H, :W]
    if ground.shape != gray.shape:
        ground = np.pad(ground, ((0, H - ground.shape[0]), (0, W - ground.shape[1])), mode="edge")
    ref = float(np.median(ground))
    if inverted:
        out = gray.astype(np.float32) - ground + ref
    else:
        out = gray.astype(np.float32) * (ref / np.maximum(ground, 1.0))
    return np.clip(out, 0, 255).astype(np.uint8), uneven


# ------------------------------------------------------------------ 2. rotation
def baseline_rotation(ink: np.ndarray, limit: float = 12.0) -> Tuple[float, float]:
    """The angle (degrees, counter-clockwise positive) by which the picture must be turned to make
    its rows of writing level: the turn that makes the ink pile up in the fewest rows. The slant of
    the letters does not move ink between rows, so it does not disturb this reading.
    Returns (angle, gain): gain is how much sharper the rows become, 1.0 = no better than as it is."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 200:
        return 0.0, 1.0
    if len(xs) > 150_000:
        step = len(xs) // 150_000 + 1
        ys, xs = ys[::step], xs[::step]
    ys = ys - ys.mean(); xs = xs - xs.mean()
    H = int(np.hypot(*ink.shape)) + 4

    def score(a: float) -> float:
        r = math.radians(a)
        v = np.round(ys * math.cos(r) + xs * math.sin(r)).astype(int) + H
        h = np.bincount(v, minlength=2 * H)
        return float(np.dot(h, h))

    base = score(0.0)
    coarse = np.arange(-limit, limit + 0.01, 0.5)
    a = float(coarse[int(np.argmax([score(v) for v in coarse]))])
    fine = np.arange(a - 0.5, a + 0.501, 0.05)
    a = float(fine[int(np.argmax([score(v) for v in fine]))])
    return a, score(a) / max(base, 1e-9)


def _rotate(gray: np.ndarray, angle: float, fill: int) -> np.ndarray:
    return np.asarray(Image.fromarray(gray).rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=int(fill)))


# ------------------------------------------------------------------ 3. letters drawn as outlines
def hollow_ratio(ink: np.ndarray) -> float:
    """White enclosed by ink, as a multiple of the ink itself: a few tenths for solid letters with
    counters, several times the ink for letters drawn as outlines."""
    filled = ndi.binary_fill_holes(ink)
    return float((filled & ~ink).sum()) / max(1.0, float(ink.sum()))


def fill_outlines(ink: np.ndarray) -> np.ndarray:
    """Turn outlined letters into solid ones. Regions are numbered by how deeply they are nested:
    the ground outside is 0, the outer outline 1, the inside of the letter 2, the outline of a
    counter 3, the counter 4. Ink is every outline plus every region at depth 2 (6, 10 ...)."""
    ink = ink.astype(bool)
    bg_lab, nb = ndi.label(~ink)
    ink_lab, ni = ndi.label(ink, structure=np.ones((3, 3)))
    bg_level = np.full(nb + 1, -1); ink_level = np.full(ni + 1, -1)
    edge = np.unique(np.concatenate([bg_lab[0], bg_lab[-1], bg_lab[:, 0], bg_lab[:, -1]]))
    bg_level[edge[edge > 0]] = 0
    pairs = set()
    for dy, dx in ((0, 1), (1, 0)):
        a_i, a_b = ink_lab[:ink.shape[0] - dy, :ink.shape[1] - dx], bg_lab[:ink.shape[0] - dy, :ink.shape[1] - dx]
        b_i, b_b = ink_lab[dy:, dx:], bg_lab[dy:, dx:]
        for i, b in ((a_i, b_b), (b_i, a_b)):
            m = (i > 0) & (b > 0)
            if m.any():
                pairs.update(map(tuple, np.unique(np.c_[i[m], b[m]], axis=0)))
    by_ink: Dict[int, List[int]] = {}; by_bg: Dict[int, List[int]] = {}
    for i, b in pairs:
        by_ink.setdefault(int(i), []).append(int(b)); by_bg.setdefault(int(b), []).append(int(i))
    level = 0
    front_bg = [int(b) for b in np.flatnonzero(bg_level == 0)]
    while front_bg:
        front_ink = []
        for b in front_bg:
            for i in by_bg.get(b, []):
                if ink_level[i] < 0:
                    ink_level[i] = level + 1; front_ink.append(i)
        front_bg = []
        for i in front_ink:
            for b in by_ink.get(i, []):
                if bg_level[b] < 0:
                    bg_level[b] = level + 2; front_bg.append(b)
        level += 2
    inside = (bg_level % 4 == 2) & (bg_level > 0)
    return ink | inside[bg_lab]


# ------------------------------------------------------------------ 4. a symbol beside the writing
def _stroke(mask: np.ndarray) -> float:
    """The pen of a piece of ink: the thick strokes (90th percentile along the centre line)."""
    from skimage.morphology import skeletonize
    if not mask.any():
        return 0.0
    d = ndi.distance_transform_edt(mask)
    v = d[skeletonize(mask)]
    return float(2 * np.percentile(v, 90)) if len(v) else float(2 * d.max())


def find_symbols(ink: np.ndarray) -> List[Tuple[slice, slice, np.ndarray, str]]:
    """Pieces of ink that stand in a line of writing but are not writing:
    - a solid shape far thicker than the pen (a disc, a badge, a filled mark);
    - a piece well apart at one end of the line, whose strokes or height do not belong to the line.
    Returns (rows, cols, mask, reason) for each."""
    ink = ink.astype(bool)
    out = []
    pen = _stroke(ink)
    if pen <= 0:
        return out
    for a, b in find_lines(ink.astype(np.uint8)):
        strip = ink[a:b + 1]
        lab, n = ndi.label(strip, structure=np.ones((3, 3)))
        if n < 2:
            continue
        objs = ndi.find_objects(lab)
        info = []
        dist = ndi.distance_transform_edt(strip)
        for i, sl in enumerate(objs, start=1):
            m = lab[sl] == i
            info.append({"i": i, "sl": sl, "area": float(m.sum()), "thick": float(2 * dist[sl][m].max()),
                         "x0": sl[1].start, "x1": sl[1].stop, "h": sl[0].stop - sl[0].start})
        big = [c for c in info if c["area"] >= 0.5 * pen * pen]
        if len(big) < 2:
            continue
        typical_h = float(np.median([c["h"] for c in big]))
        total = sum(c["area"] for c in big)
        sym = set()
        for c in big:
            if c["thick"] > 2.4 * pen and c["area"] > 6 * pen * pen and c["area"] < 0.6 * total:
                sym.add(c["i"])                                         # a solid shape, not a stroke
        # a piece apart at an end of the line
        order = sorted(big, key=lambda c: c["x0"])
        H = b - a + 1
        for end in (order[0], order[-1]):
            others = [c for c in order if c is not end and c["i"] not in sym]
            if not others or end["i"] in sym:
                continue
            gap = (min(c["x0"] for c in others) - end["x1"]) if end is order[0] else (end["x0"] - max(c["x1"] for c in others))
            if gap < 0.9 * H:
                continue
            m = lab[end["sl"]] == end["i"]
            own = _stroke(m)
            rest = _stroke(np.isin(lab, [c["i"] for c in others]))
            if abs(own / max(rest, 1e-6) - 1) > 0.4 or end["h"] > 1.5 * typical_h:
                sym.add(end["i"])
        for i in sym:
            sl = objs[i - 1]
            reason = "solid" if next(c for c in info if c["i"] == i)["thick"] > 2.4 * pen else "apart"
            out.append((slice(a + sl[0].start, a + sl[0].stop), sl[1], lab[sl] == i, reason))
    return out


# ------------------------------------------------------------------ the guard
def prepare(data: bytes) -> Tuple[Image.Image, np.ndarray, float, bool, List[Note]]:
    """Load a picture and make it measurable. Returns the working grey image, the ink mask, the
    working scale, whether the picture is light on dark, and the notes about every correction."""
    work, scale = _load_gray(data)
    gray = np.asarray(work, dtype=np.uint8)
    notes: List[Note] = []
    th, inverted = _polarity(gray)
    flat, uneven = flatten_ground(gray, inverted)
    if flat is not gray:
        gray = flat
        th, inverted = _polarity(gray)
        notes.append(_note("ground", f"الخلفية غير مستوية (فرق {uneven:.0f} درجة رمادية): سُوّيت قبل القياس.", uneven=uneven))
    ink = (gray > th) if inverted else (gray <= th)
    angle, gain = baseline_rotation(ink)
    if abs(angle) >= 0.4 and gain > 1.04:
        ground = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
        gray = _rotate(gray, -angle, ground)
        ink = (gray > th) if inverted else (gray <= th)
        notes.append(_note("rotation", f"السطر مائل {abs(angle):.1f}° عن الأفق (دوران، لا ميل حروف): عُدّلت الصورة.", angle=angle))
        if abs(angle) > 11.5:
            notes.append(_note("warning", "الدوران عند حدّ ما يصحّحه المحرك (12°): قد يبقى ميل في السطر."))
    ratio = hollow_ratio(ink)
    if ratio > 1.2:
        solid = fill_outlines(ink)
        dark = 0 if not inverted else 255
        gray = np.where(solid & ~ink, dark, gray).astype(np.uint8)
        ink = solid
        notes.append(_note("outline", "الحروف مرسومة بخط خارجي فقط: مُلئت لتُقاس كحروف مصمتة، وسمك الخط الخارجي ليس سمك الجذع.",
                           hollow=ratio))
    # an outline that was filled may have closed a counter by mistake: such a piece would look like
    # a solid symbol, so nothing is taken out of a filled picture
    symbols = find_symbols(ink) if ratio <= 1.2 else []
    if symbols:
        ground = 255 if not inverted else 0
        gray = gray.copy(); ink = ink.copy()
        for rows, cols, m, _ in symbols:
            ink[rows, cols][m] = False
            gray[rows, cols][m] = ground
            # anti-aliased rim of the removed piece
            rim = ndi.binary_dilation(m, iterations=2)
            sub = gray[rows, cols]; sub[rim & ~ink[rows, cols]] = ground
        ink = (gray > th) if inverted else (gray <= th)
        notes.append(_note("symbol", f"في السطر {len(symbols)} شكل ليس كتابة (رمز أو أيقونة): استُبعد من القياس.", count=len(symbols)))
    # full contrast: the measuring reads grey as ink intensity, so the ground must be 0 and the ink 1
    # whatever their colours were (a grey ground or a coloured ink would otherwise count as part ink)
    if ink.any() and not ink.all():
        a, b = float(np.median(gray[ink])), float(np.median(gray[~ink]))
        lo, hi = min(a, b), max(a, b)
        if hi - lo >= 30 and (lo > 2 or hi < 253):
            gray = np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            if lo > 40 or hi < 215:
                notes.append(_note("contrast", "الحبر أو الخلفية ليسا أسود وأبيض صافيين: مُدّ التباين قبل القياس.", ink=a, ground=b))
    return Image.fromarray(gray), ink.astype(np.uint8), scale, inverted, notes


# ------------------------------------------------------------------ how far to trust the reading
def assess(lines, readings, src: int, notes: List[Note]) -> Dict[str, Any]:
    """What still looks unreliable after measuring the chosen line, and one figure for it."""
    L, R = lines[src], readings[src]
    warn: List[Note] = []
    penalty = 0.0

    def add(text: str, cost: float, **ev):
        nonlocal penalty
        warn.append(_note("warning", text, **ev)); penalty += cost

    if L.stem == L.stem and L.stem < 5:
        add(f"الصورة صغيرة: الجذع {L.stem:.1f} بكسل فقط. القياسات تقريبية.", 0.25, stem_px=L.stem)
    if len(L.stems) < 3:
        add(f"وُجد {len(L.stems)} جذع فقط: لا يكفي لقراءة الإيقاع والارتفاعات.", 0.35, stems=len(L.stems))
    elif L.stems:
        w = np.array([s.w for s in L.stems])
        spread = float(np.median(np.abs(w - np.median(w))) / max(np.median(w), 1e-6))
        if spread > 0.12:
            add(f"سمك الجذوع غير ثابت (تفاوت {spread * 100:.0f}%): قد لا يكون الخط بقلم واحد.", 0.2, spread=spread)
    main = [v for v in L.levels if v.name == "x_height"]
    if not main or main[0].n < 2:
        add("ارتفاع الحروف مأخوذ من جذع واحد: غير مؤكد.", 0.3)
    else:
        # a level a little lower with as many stems is the round letters or the inner teeth; one far
        # away with as many stems is another candidate for the x-height
        rival = [v for v in L.levels if v.name != "x_height" and v.n >= main[0].n and abs(v.y / main[0].y - 1) > 0.15]
        if rival:
            add(f"ارتفاعان يتنازعان على ارتفاع الحروف ({main[0].y:.0f} و{rival[0].y:.0f} بعدد الجذوع نفسه): قد يكون الآخر هو الصحيح.",
                0.25, chosen=main[0].y, other=rival[0].y)
        elif main[0].n < 3:
            add("ارتفاع الحروف مأخوذ من جذعين فقط.", 0.1)
    if R.script != "arabic":
        add("لم يُتعرَّف على السطر ككتابة عربية.", 0.3, script=R.script)
    elif not R.text_given:
        unsure = sum(1 for u in R.units if u.letter in "ٮ?" or not u.letter)
        if R.units and unsure / len(R.units) > 0.34:
            add(f"لم يُقرأ {unsure} من {len(R.units)} حرف: اكتب نص الشعار ليُفهم أدق.", 0.15, unread=unsure)
    if L.pen_fit and L.stem == L.stem and L.pen_fit.get("median_error", 0) > 0.12 * L.stem:
        add("قاعدة القلم لا تطابق الشعار جيدًا: السمك لا يتبع الاتجاه وحده.", 0.15, error=L.pen_fit["median_error"])
    for n in notes:
        if n["kind"] == "outline":
            penalty += 0.25                                  # filled outlines are read roughly
        elif n["kind"] in ("rotation", "ground", "symbol", "contrast"):
            penalty += 0.05                                  # corrected, but a corrected picture is less sure
    return {"corrections": [n for n in notes if n["kind"] != "warning"],
            "warnings": [n for n in notes if n["kind"] == "warning"] + warn,
            "confidence": round(float(np.clip(1.0 - penalty, 0.0, 1.0)), 2)}
