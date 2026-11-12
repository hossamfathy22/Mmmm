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
    from .features import measure_features
    from .understand import read_line
    feats = measure_features([L], [read_line(L)], 0, k)
    out = {"stem": L.stem * k, "hair": L.hair * k, "x_height": L.x_height * k,
           "white": L.white * k if L.white == L.white else float("nan"),
           "dot": float(np.median(dots)) * k if dots else float("nan"), "slant": L.slant,
           "power": L.power, "angle": getattr(L, "pen_angle", 0.0), "stems": len(L.stems)}
    out.update({"f_" + key: v["value"] for key, v in feats.items()})
    if "dot_size" in feats:
        out["dot"] = feats["dot_size"]["value"]          # measured the way the logo's dot was measured
    return out


SHAPE_CHECKS = [
    # style field, measured feature, Arabic name, absolute tolerance
    ("top_angle", "f_top_angle", "زاوية طرف الجذع °", 3.0),
    ("top_round", "f_top_round", "استدارة طرف الجذع", 0.15),
    ("join_in", "f_join_in", "عمق منحنى الوصل الداخلي", 0.06),
    ("dot_n", "f_dot_n", "شكل النقطة", 0.6),
    ("ring_n", "f_ring_n", "تربيع الرأس", 0.6),
    ("seen_drop", "f_seen_drop", "انخفاض الأسنان الداخلية", 2.0),
]


def compare(st: Style, got: Optional[Dict[str, float]]) -> List[Dict]:
    rows = []
    for key, fkey, name, tol in SHAPE_CHECKS:
        if not got or fkey not in got:
            continue
        want, have = getattr(st, key), got[fkey]
        if key == "seen_drop" and want == 0:
            continue
        d = have - want
        rows.append({"key": key, "name": name, "rule": round(want, 2), "word": round(have, 2),
                     "diff": None, "verdict": "مطابق" if abs(d) <= tol else ("قريب" if abs(d) <= 2 * tol else "مختلف")})
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
    shape_rows = [r for r in rows if r["key"] in {c[0] for c in SHAPE_CHECKS}]
    return [r for r in rows if r not in shape_rows] + shape_rows


def _error(st: Style, got: Optional[Dict[str, float]]) -> float:
    if not got:
        return 1e9
    e = 0.0
    if got.get("white") == got.get("white") and st.white:
        e += abs(got["white"] / st.white - 1) / 0.04
    if "f_join_in" in got:
        e += abs(got["f_join_in"] - st.join_in) / 0.03
    return e


def verify_and_correct(build, st: Style, rounds: int = 4):
    """build(style) -> Layout. Measure the word; where a drawing parameter does not produce the
    measured result the logo has (the white between teeth, the depth of the join curve), adjust
    that parameter and build again. A correction is kept only if the word gets closer to the logo."""
    cur = Style(**st.__dict__)
    lay = build(cur)
    got = measure_word(lay)
    steps = 0
    for _ in range(rounds):
        if not got:
            break
        nxt = Style(**cur.__dict__)
        changed = False
        if got.get("white") == got.get("white") and got.get("white") and abs(st.white / got["white"] - 1) > 0.04:
            nxt.white = cur.white * st.white / got["white"]
            changed = True
        if "f_join_in" in got and got["f_join_in"] > 0.02 and abs(got["f_join_in"] - st.join_in) > 0.03:
            nxt.join_in = float(np.clip(cur.join_in + (st.join_in - got["f_join_in"]), 0.1, 0.8))
            nxt.join_out = max(nxt.join_out, nxt.join_in - 0.15)
            changed = True
        if not changed:
            break
        lay2 = build(nxt)
        got2 = measure_word(lay2)
        if _error(st, got2) < _error(st, got):
            cur, lay, got = nxt, lay2, got2
            steps += 1
        else:
            break
    return lay, got, compare(st, got), steps
