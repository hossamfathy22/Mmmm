"""Which construction a logo is built in, decided by the logo itself.

The engine has two ways of building letters: the pen (letters.py: stems and hairlines, round heads and
cups, as بيدوناق was written) and the blocks (blocks.py: straight-edged blocks cut at angles). A new
logo is not assumed to be either. Its own word is drawn both ways from the same measures of the logo
(its pen, x-height, ascender, descender, hairline, white), the block rules are fitted a little to it,
and the construction that rebuilds the logo's ink better is the logo's (the blocks by a clear margin
and rebuilding it well; otherwise the pen, the engine's general construction, and the evidence is
reported as unsure when neither rebuilds it well). Nothing of another logo enters: the block rules start
from this logo's measures (blocks.start_rules) and are fitted on this logo's word only.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .style import Style

MARGIN = 0.03          # the blocks must rebuild the logo this much better than the pen to be chosen
BLOCK_MIN = 0.6        # ... and rebuild it well: where neither does, the evidence decides nothing
# what is searched on the logo's word: the rules no direct measure gives (the pen, the x-height, the
# dot, the cut of the tops and whether letters touch are measured, see measured_rules)
QUICK = ("A", "D", "J", "slit", "into", "head_w", "tooth_foot", "alef_foot", "join_line")


def measured_rules(an: Dict[str, Any], word: str, m: np.ndarray, base: float) -> Dict[str, float]:
    """The block rules the logo shows directly: the dot (its measured size), the cut of the tops (the
    measured angle of the uprights' tops), and whether letters that do not join touch: the word's
    letters stand in groups that do not join, and when the logo has fewer pieces of ink than the word
    has groups, its letters touch."""
    from scipy import ndimage as ndi
    from .typeset import shape_forms
    feats = {k: v["value"] for k, v in (an.get("feats") or {}).items()}
    out: Dict[str, float] = {}
    if isinstance(feats.get("dot_size"), float):
        out["dot"] = float(feats["dot_size"])
    if isinstance(feats.get("top_angle"), float) and abs(feats["top_angle"]) >= 2.0:
        out["top_cut"] = float(abs(feats["top_angle"]))
    groups, cur = 0, False
    for c, f in shape_forms(word):
        if c == " ":
            cur = False
            continue
        if f in ("i", "I") or not cur:
            groups += 1
        cur = f in ("I", "M")
    band = m[int(max(0, base - 0.9 * an["style"].x_height)):int(max(1, base - 0.15 * an["style"].x_height))]
    pieces = ndi.label(band.any(axis=0))[1] if band.size else 0
    if groups and pieces and pieces < groups:
        out["touch"] = 0.0
    return out


def logo_mask(an: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """The chosen line's ink, upright (its slant taken out), in the logo's pixels, and its baseline row."""
    from PIL import Image
    L, k = an["lines"][an["src"]], an["k"]
    m = np.asarray(L.G) > 0.5
    base = float(L.base_row - L.top)
    if abs(k - 1.0) > 1e-3:
        im = Image.fromarray((m * 255).astype(np.uint8)).resize((max(1, round(m.shape[1] * k)), max(1, round(m.shape[0] * k))), Image.BOX)
        m, base = np.array(im) >= 128, base * k
    return m, base


def rebuild(word: str, st: Style, m: np.ndarray, base: float) -> Tuple[float, Tuple[float, float]]:
    """How much ink the word drawn in `st` shares with the logo (IoU), its right edge slid a little
    along the line to sit where the logo's does; and the origin that gave it."""
    from .blockfit import raster
    from .typeset import typeset
    cols = np.nonzero(m.any(axis=0))[0]
    if not len(cols):
        return 0.0, (0.0, base)
    lay = typeset(word, st)
    xs = [x for sh in lay.shapes for x, _ in sh[0]]
    if not xs:
        return 0.0, (0.0, base)
    right = cols[-1] + 1.0
    best = (0.0, (right - max(xs), base))
    for dx in np.linspace(-0.6, 0.6, 7) * st.stem:
        o = (right - max(xs) + dx, base)
        got = raster(lay, (m.shape[1], m.shape[0]), o)
        u = (got | m).sum()
        v = float((got & m).sum() / u) if u else 0.0
        if v > best[0]:
            best = (v, o)
    return best


def choose(an: Dict[str, Any], word: str, fit_iters: int = 12) -> Dict[str, Any]:
    """The logo's construction, with the evidence: each construction's rebuild of the logo's word."""
    from .blockfit import fit
    st: Style = an["style"]
    m, base = logo_mask(an)
    up = replace(st, slant=0.0)
    pen, _ = rebuild(word, up, m, base)
    blk_st = replace(up, construction="block", block=measured_rules(an, word, m, base), block_parts={})
    start, origin = rebuild(word, blk_st, m, base)
    fitted = blk_st
    blk = start
    if start > pen - 0.15:          # worth fitting: the blocks are near the pen already
        fitted = fit(word, blk_st, m, origin, cycles=1, log=lambda s: None, keys=QUICK, iters=fit_iters)
        blk, _ = rebuild(word, fitted, m, base)
    use = "block" if blk > pen + MARGIN and blk >= BLOCK_MIN else "pen"
    out = {"construction": use, "pen": round(pen, 3), "block_start": round(start, 3), "block": round(blk, 3),
           "word": word, "sure": max(pen, blk) >= BLOCK_MIN}
    if use == "block":
        from .blocks import rules
        out["style"] = replace(st, construction="block", slant=0.0, block=dict(rules(fitted)), block_parts={})
    return out
