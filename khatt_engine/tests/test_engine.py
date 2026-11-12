import io
import math
from pathlib import Path

import numpy as np
import pytest

from khatt import Style, run, typeset
from khatt.analyze import find_lines, load_binary
from khatt.bezier import fit_closed, sample
from khatt.letters import LETTERS
from khatt.library import build_library
from khatt.measure import measure_line
from khatt.render import to_png
from khatt.typeset import flat, shape_forms
from khatt.understand import read_line

EXAMPLE = Path(__file__).parent.parent / "examples" / "bidonaq.png"


def _lines(png: bytes):
    work, ink, scale, inv = load_binary(png)
    g = np.asarray(work, float) / 255
    ink01 = g if inv else 1 - g
    return [measure_line(ink01, ink.astype(bool), a, b, i) for i, (a, b) in enumerate(find_lines(ink))], 1 / scale


# ---------------------------------------------------------------- shaping and letters
def test_forms_follow_arabic_joining():
    assert shape_forms("بيدوناق") == [("ب", "I"), ("ي", "M"), ("د", "F"), ("و", "i"), ("ن", "I"), ("ا", "F"), ("ق", "i")]
    assert [f for _, f in shape_forms("محمد")] == ["I", "M", "M", "F"]


@pytest.mark.parametrize("letter", sorted(LETTERS))
@pytest.mark.parametrize("form", ["i", "I", "M", "F"])
def test_every_letter_draws_in_every_form(letter, form):
    glyph = LETTERS[letter][1](Style(), form)
    pts = [p for s in flat(glyph.shapes) for poly in s for p in poly]
    assert pts and all(math.isfinite(x) and math.isfinite(y) for x, y in pts)


def test_unknown_characters_are_reported_not_drawn():
    lay = typeset("abc نور", Style())
    assert lay.skipped == ["a", "b", "c"]


# ---------------------------------------------------------------- stage 1-2: precise measurement
def test_round_trip_measures_the_rules_it_was_drawn_with():
    st = Style(stem=30, hair=6, x_height=90, ascender=45, descender=45, slant=12, white=28, dot=33)
    lay = typeset("بيبيبي سنن", st)
    x0, y0, x1, y1 = lay.bbox()
    px = 2.0
    png = to_png(lay, height=int(round((y1 - y0 + 3 * st.stem) * px)))
    (L,), k = _lines(png)
    k = k / px
    assert L.slant == pytest.approx(12, abs=0.6)
    assert L.stem * k == pytest.approx(30, rel=0.04)
    assert L.hair * k == pytest.approx(6, rel=0.2)
    assert L.x_height * k == pytest.approx(90, rel=0.03)
    assert L.white * k == pytest.approx(28, rel=0.12)
    assert L.power == pytest.approx(2.0, abs=0.8)
    assert [p.size * k for p in L.dots] and all(abs(p.size * k - 33) < 3 for p in L.dots)


# ---------------------------------------------------------------- stage 3: understanding
def test_reading_aligns_a_typed_text_with_the_stems():
    (L,), _ = _lines(EXAMPLE.read_bytes())
    R = read_line(L, "بيدوناق")
    assert R.script == "arabic"
    letters = [s.letter for s in L.stems]
    assert letters[:2] == ["ب", "ي"] and "ا" in letters and "ن" in letters


def test_observations_find_the_two_pens():
    from khatt.understand import observations
    lines, k = _lines(EXAMPLE.read_bytes())
    R = [read_line(L) for L in lines]
    texts = " ".join(o["text"] for o in observations(lines, R, k))
    assert "قلمان" in texts


# ---------------------------------------------------------------- stage 4: parts and Bezier
def test_bezier_fit_is_accurate():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    c = fit_closed(np.c_[50 + 25 * np.cos(t), 50 + 25 * np.sin(t)], 0.3)
    s = np.array(sample(c, 30))
    assert np.abs(np.hypot(s[:, 0] - 50, s[:, 1] - 50) - 25).max() < 0.3


def test_library_cuts_real_parts():
    lines, k = _lines(EXAMPLE.read_bytes())
    L = lines[0]
    lib = build_library(L, read_line(L), k)
    assert {"shaft", "dot", "dash"} <= set(lib.parts)
    assert all(p.curves for p in lib.parts.values())


# ---------------------------------------------------------------- stages 5-7: build, verify, draw
def test_default_method_measures_then_redraws_clean_parts(tmp_path):
    r = run(EXAMPLE.read_bytes(), "بيدوناق", texts=["بيدوناق"], out_dir=str(tmp_path))
    assert r["mode"] == "ideal"
    assert not any(x["from_logo"] for x in r["recipes"])
    assert all(c["verdict"] == "مطابق" for c in r["verification"])


def test_cut_parts_mode_uses_the_logo_pixels_and_verifies(tmp_path):
    r = run(EXAMPLE.read_bytes(), "شركة", texts=["بيدوناق"], use_parts=True, out_dir=str(tmp_path))
    assert any(x["from_logo"] for x in r["recipes"])
    verdicts = {c["key"]: c["verdict"] for c in r["verification"]}
    assert verdicts["stem"] == "مطابق" and verdicts["slant"] == "مطابق"
    for name in ["word.svg", "word.png", "construction.svg", "logo_sheet.svg", "parts.svg", "pen_rule.svg",
                 "parts.json", "measurements.json", "report.md"]:
        assert (tmp_path / name).stat().st_size > 0


def test_style_keeps_room_under_the_baseline():
    r = run(EXAMPLE.read_bytes(), "بنت")
    assert r["style"]["descender"] >= 0.45 * r["style"]["x_height"] - 0.01


# ---------------------------------------------------------------- shape judgement on a logo of a different style
ROUND = Path(__file__).parent.parent / "examples" / "round_square.png"


def test_shapes_are_read_from_a_different_logo():
    """The test logo was drawn upright, with round stem tops, square dots and squarish heads."""
    r = run(ROUND.read_bytes(), "بيدوناق", texts=["سفينة نجم"])
    st = r["style"]
    assert st["slant"] == pytest.approx(0, abs=0.5)
    assert st["top_round"] > 0.75
    assert st["dot_n"] > 5
    assert st["ring_n"] == pytest.approx(3.5, abs=0.5)
    assert st["overshoot"] < 1.0
    assert all(c["verdict"] in ("مطابق", "قريب") for c in r["verification"])


def test_shapes_of_the_first_style():
    r = run(EXAMPLE.read_bytes(), "بيدوناق", texts=["بيدوناق"])
    st = r["style"]
    assert st["slant"] == pytest.approx(14, abs=1)
    assert st["top_round"] < 0.2 and abs(st["top_angle"]) < 3
    assert st["dot_n"] < 3


# ---------------------------------------------------------------- thin monoline logos with long horizontals
def test_band_survives_long_thin_horizontals():
    """A geometric logo whose long horizontal strokes make a few rows look dense: the x-height band
    must still span the letters, not collapse onto one stroke."""
    from khatt.measure import measure_line
    h, w = 140, 520
    ink = np.zeros((h, w), float)
    base, xh, pen = 100, 44, 8
    ink[base - pen:base, 40:480] = 1                       # a long baseline stroke (kashida)
    ink[base - xh:base - xh + pen, 300:480] = 1             # a long top stroke
    for x in range(60, 300, 40):                           # stems standing on the baseline
        ink[base - xh:base, x:x + 10] = 1
    ink[base - xh:base, 470:480] = 1
    L = measure_line(ink, ink > 0.5, 20, 120, 0)
    assert abs(L.x_height - xh) <= 3
    assert abs(L.stem - 10) <= 1.5


def test_a_row_of_big_square_dots_belongs_to_its_line():
    """Large square dots above the letters form their own ink strip; it must join the line below."""
    from khatt.analyze import find_lines
    ink = np.zeros((300, 600), np.uint8)
    ink[40:90, 100:150] = 1; ink[40:90, 200:250] = 1      # dots, 50 px tall
    ink[110:260, 80:520] = 1                              # the writing, 150 px tall
    assert find_lines(ink) == [(40, 259)]


# ---------------------------------------------------------------- the skeleton: corners or curves
def _line_of(ink):
    from khatt.analyze import find_lines
    from khatt.measure import measure_line
    a, b = find_lines(ink.astype(np.uint8))[0]
    return measure_line(ink.astype(float), ink.astype(bool), a, b, 0)


def test_skeleton_tells_corners_from_curves():
    """Square hooks read as corners; round hooks of the same size read as curves."""
    from khatt.skeleton import read_skeleton
    from skimage.draw import disk
    sq = np.zeros((200, 700), bool)
    rd = np.zeros((200, 700), bool)
    for x in range(80, 640, 140):
        sq[50:150, x:x + 14] = True; sq[136:150, x - 60:x + 14] = True          # down, then a corner, then left
        sq[50:62, x:x + 60] = True                                              # and a top bar
        yy, xx = disk((100, x - 20), 48, shape=rd.shape); rd[yy, xx] = True     # a ring with the same reach
        yy, xx = disk((100, x - 20), 34, shape=rd.shape); rd[yy, xx] = False
    a, b = read_skeleton(_line_of(sq)), read_skeleton(_line_of(rd))
    assert a.angular > 0.6
    assert b.angular < 0.3


def test_angular_tail_rests_on_the_baseline_with_a_round_corner():
    from khatt.letters import tail
    from khatt.typeset import flat
    st = Style(stem=11, hair=8, x_height=44, slant=0, angular=0.8, corner_r=8, tail_drop=0.0, tail_reach=1.9)
    pts = [p for sh in flat([tail(st, 0, 26)]) for poly in sh for p in poly]
    ys = [p[1] for p in pts]; xs = [p[0] for p in pts]
    assert min(ys) == pytest.approx(0, abs=0.5)                 # the foot sits on the baseline
    assert min(xs) == pytest.approx(-1.9 * 11, abs=0.5)         # as far left as the logo's
    # the outer corner (bottom right) is rounded: no point at the sharp corner itself
    assert not any(abs(x - 5.5) < 0.3 and abs(y) < 0.3 for x, y in pts)
