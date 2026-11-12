"""Stage 6: measure the generated word with the same engine and compare it with the logo's rules."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .render import to_png
from .style import Style
from .typeset import Layout

PX_PER_UNIT = 2.0

CHECKS = [
    # key, Arabic name, tolerance (relative), absolute tolerance
    ("stem", "سمك الجذع", 0.05, 0.6),
    ("hair", "الخط الشعري", 0.15, 0.6),
    ("x_height", "ارتفاع الحروف", 0.04, 1.0),
    ("white", "الأبيض بين الجذوع", 0.10, 1.0),
    ("dot", "قطر النقطة", 0.08, 1.0),
    ("slant", "الميل °", 0.0, 0.7),
]


def measure_word(lay: Layout) -> Optional[Dict[str, float]]:
    from .analyze import load_binary, find_lines
    from .measure import measure_line
    st = lay.style
    x0, y0, x1, y1 = lay.bbox()
    height = int(round((y1 - y0 + 3 * st.stem) * PX_PER_UNIT))
    png = to_png(lay, height=height)
    work, ink, scale, inv = load_binary(png)
    g = np.asarray(work, float) / 255.0
    ink01 = g if inv else 1 - g
    lines = find_lines(ink)
    if not lines:
        return None
    a, b = max(lines, key=lambda v: v[1] - v[0])
    L = measure_line(ink01, ink.astype(bool), a, b, 0)
    if L is None or not L.stems:
        return None
    k = 1.0 / (scale * PX_PER_UNIT)
    dots = [p.size for p in L.dots]
    return {"stem": L.stem * k, "hair": L.hair * k, "x_height": L.x_height * k,
            "white": L.white * k if L.white == L.white else float("nan"),
            "dot": float(np.median(dots)) * k if dots else float("nan"), "slant": L.slant,
            "power": L.power, "stems": len(L.stems)}


def compare(st: Style, got: Optional[Dict[str, float]]) -> List[Dict]:
    rows = []
    for key, name, rel, ab in CHECKS:
        want = getattr(st, key)
        have = (got or {}).get(key, float("nan"))
        if not (have == have):
            rows.append({"key": key, "name": name, "rule": round(want, 2), "word": None, "diff": None, "verdict": "غير مقيس"})
            continue
        d = have - want
        ok = abs(d) <= max(ab, rel * abs(want))
        near = abs(d) <= 2 * max(ab, rel * abs(want))
        rows.append({"key": key, "name": name, "rule": round(want, 2), "word": round(have, 2),
                     "diff": round(100 * d / want, 1) if want else None,
                     "verdict": "مطابق" if ok else ("قريب" if near else "مختلف")})
    return rows


def verify_and_correct(build, st: Style, rounds: int = 2):
    """build(style) -> Layout. Measure, and if the white between teeth drifted, correct it."""
    lay = build(st)
    got = measure_word(lay)
    history = [compare(st, got)]
    for _ in range(rounds):
        if not got or not (got.get("white") == got.get("white")) or not st.white:
            break
        ratio = st.white / got["white"]
        if abs(ratio - 1) <= 0.06:
            break
        target = st.white
        trial = Style(**{**st.__dict__, "white": st.white * ratio})
        lay2 = build(trial)
        got2 = measure_word(lay2)
        if got2 and got2.get("white") == got2.get("white") and abs(got2["white"] - target) < abs(got["white"] - target):
            lay, got = lay2, got2
            history.append(compare(st, got))
        break
    return lay, got, history[-1], len(history) - 1
