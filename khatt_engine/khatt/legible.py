"""Does a letter still read as itself?

The engine draws a letter differently from its own construction when it takes the logo's parts for it.
A drawing can then match the logo's manner and stop reading as the letter: a head with a stroke
along the baseline turns فقه into ححه. The pixels cannot tell; reading can. A letter is read by its
skeleton against the letters of the same style drawn in the engine's construction (the alphabet the
drawing has to stay distinct from), as the examination on the typefaces reads letters.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .letters import LETTERS, Glyph
from .style import Style

_REFS: Dict[Tuple, Dict[str, object]] = {}


def _shape(glyph: Glyph, st: Style):
    """The letter's body (its marks left out) on the comparison canvas of the examination."""
    from .bench.score import BASE_ROW, CANVAS, PX, RIGHT_COL, Shape
    from .bench.truth import place
    from .typeset import _is_mark, flat
    s = PX / st.x_height
    H, W = CANVAS
    m = np.zeros((H, 3 * W), bool)
    ox = 2.5 * W
    for sh in flat(glyph.shapes):
        if _is_mark(sh, st):
            continue
        part = None
        for poly in sh:
            if len(poly) < 3:
                continue
            layer = Image.new("L", (3 * W, H), 0)
            ImageDraw.Draw(layer).polygon([(ox + x * s, BASE_ROW - y * s) for x, y in poly], fill=255)
            a = np.array(layer) > 0
            part = a if part is None else part ^ a
        if part is not None:
            m |= part
    if not m.any():
        return None
    return Shape(place(m, BASE_ROW, RIGHT_COL, PX, CANVAS))


def _key(st: Style, form: str) -> Tuple:
    return (form, round(st.stem, 2), round(st.hair, 2), round(st.x_height, 2), round(st.white, 2), round(st.angular, 2),
            round(st.tail_drop, 2), round(st.top_join, 2), tuple(sorted(st.fitted.items())), tuple(sorted(st.chosen.items())),
            tuple(sorted(st.traits.items())), st.measured_on)


def _refs(st: Style, form: str) -> Dict[str, object]:
    """The alphabet of this style in its engine construction, one reference per skeleton."""
    from .bench.__main__ import SKELETONS
    key = _key(st, form)
    if key not in _REFS:
        plain = Style(**{**st.__dict__, "parts": {}, "taken": {}, "variants": {}, "reference": ""})
        refs = {}
        for c in SKELETONS:
            if form in ("I", "M") and LETTERS[c][0] != "D":
                continue
            try:
                sh = _shape(LETTERS[c][1](plain, form), plain)
            except Exception:
                continue
            if sh is not None and not sh.empty:
                refs[c] = sh
        if len(_REFS) > 64:
            _REFS.clear()
        _REFS[key] = refs
    return _REFS[key]


def reads_as(glyph: Glyph, char: str, form: str, st: Style) -> Optional[str]:
    """The letter `glyph` reads as among the engine's alphabet of this style (by its skeleton)."""
    from .bench.score import read
    x = _shape(glyph, st)
    if x is None or x.empty:
        return None
    refs = _refs(st, form)
    if not refs:
        return None
    return read(x, refs)[0]


def legible(glyph: Glyph, char: str, form: str, st: Style) -> bool:
    """True when the drawing reads as `char` (or a letter of the same body: the reading leaves the dots out)."""
    from .bench.__main__ import _same
    from .recipes import SKELETON
    got = reads_as(glyph, char, form, st)
    if got is None:
        return True
    return _same(got, SKELETON.get(char, char), form) or _same(got, char, form)


_KEPT: Dict[Tuple, bool] = {}


def keeps_reading(engine: Glyph, drawn: Glyph, char: str, form: str, st: Style) -> bool:
    """True when `drawn` (the letter with the logo's parts) reads as the engine's own drawing of it
    does, or as the letter itself: the parts changed its look, not what it is."""
    from .bench.__main__ import _same
    from .recipes import SKELETON
    # the parts by what they are, not by where they sit in memory (a dict freed and made again can
    # land at the same address with other parts in it)
    parts = tuple((q, v.get("from"), tuple(tuple(p) for p in v["polys"][0][:4])) for q, v in sorted(st.parts.items()) if v.get("polys"))
    key = (_key(st, form), char, parts, st.ref_skip, st.part_ref_fit)
    if key in _KEPT:
        return _KEPT[key]
    def fine(now, was):
        return now is None or now == was or _same(now, SKELETON.get(char, char), form) or _same(now, char, form)
    # two judges: the engine's alphabet of this style, and real typefaces' letters
    ok = fine(reads_as(drawn, char, form, st), reads_as(engine, char, form, st))
    if ok:
        ok = fine(reads_as_real(drawn, form, st), reads_as_real(engine, form, st))
    if len(_KEPT) > 4096:
        _KEPT.clear()
    _KEPT[key] = ok
    return ok


# ------------------------------------------------------------------ read against real letters too
# The engine's own alphabet is one judge; real typefaces are another, that does not share the
# engine's weaknesses (its joined ف and ق, say). A few typefaces of different kinds.
READERS = ("vazirmatn", "amiri", "noto-naskh-arabic", "cairo", "reem-kufi", "tajawal", "el-messiri", "changa")
_REAL: Dict[Tuple[str, str], Dict[str, object]] = {}


def _real_refs(form: str, skip: str = "") -> Dict[str, object]:
    """Letters of the reading typefaces in `form` (their bodies, on the comparison canvas), labelled
    "letter|typeface"; `skip` leaves a family out."""
    key = (form, skip)
    if key not in _REAL:
        from .bench.__main__ import SKELETONS
        from .bench.fonts import load
        from .bench.score import BASE_ROW, CANVAS, PX, RIGHT_COL, Shape
        from .bench.truth import letter_mask, x_height
        from .reference import faces, family
        refs = {}
        for path in faces():
            fam = family(path)
            if not path.endswith("-400.woff2") or fam not in READERS or fam == skip:
                continue
            face = load(path)
            cmap = face.tt.getBestCmap()
            X = x_height(face)
            for c in SKELETONS:
                if ord(c) not in cmap or (form in ("I", "M") and LETTERS[c][0] != "D"):
                    continue
                m = letter_mask(face, c, form, PX, CANVAS, BASE_ROW, RIGHT_COL, X)
                if m is not None and m.any():
                    refs[f"{c}|{fam}"] = Shape(m)
        _REAL[key] = refs
    return _REAL[key]


def reads_as_real(glyph: Glyph, form: str, st: Style) -> Optional[str]:
    """The letter `glyph` reads as among real typefaces' letters."""
    from .bench.score import read
    x = _shape(glyph, st)
    refs = _real_refs(form, st.ref_skip)
    if x is None or x.empty or not refs:
        return None
    return read(x, refs)[0].split("|")[0]
