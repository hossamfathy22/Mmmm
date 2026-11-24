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
# where the engine stands now: quality floors and the critic's known findings ("no worse than this")
import json as _json
BASE = _json.loads((Path(__file__).parent / "baseline.json").read_text(encoding="utf-8"))
FLOOR = BASE["self_check"]["bidonaq.png"] - BASE["tolerance"]


def _no_new_defects(name, defects):
    """The critic finds nothing beyond what it found when the baseline was set (a defect fixed is
    welcome; a new one fails)."""
    new = sorted(f"{d.kind}|{d.where}" for d in defects if f"{d.kind}|{d.where}" not in BASE["critic_known"][name])
    assert new == [], [d.text() for d in defects if f"{d.kind}|{d.where}" in new]


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
@pytest.mark.slow
def test_default_method_measures_then_redraws_clean_parts(tmp_path):
    r = run(EXAMPLE.read_bytes(), "بيدوناق", texts=["بيدوناق"], out_dir=str(tmp_path))
    assert r["mode"] == "ideal"
    assert not any(x["from_logo"] for x in r["recipes"])
    assert all(c["verdict"] == "مطابق" for c in r["verification"])


@pytest.mark.slow
def test_cut_parts_mode_uses_the_logo_pixels_and_verifies(tmp_path):
    r = run(EXAMPLE.read_bytes(), "شركة", texts=["بيدوناق"], use_parts=True, out_dir=str(tmp_path))
    assert any(x["from_logo"] for x in r["recipes"])
    verdicts = {c["key"]: c["verdict"] for c in r["verification"]}
    assert verdicts["stem"] == "مطابق" and verdicts["slant"] == "مطابق"
    for name in ["word.svg", "word.png", "construction.svg", "logo_sheet.svg", "parts.svg", "pen_rule.svg",
                 "parts.json", "measurements.json", "report.md"]:
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.slow
def test_style_keeps_room_under_the_baseline():
    r = run(EXAMPLE.read_bytes(), "بنت")
    assert r["style"]["descender"] >= 0.45 * r["style"]["x_height"] - 0.01


# ---------------------------------------------------------------- shape judgement on a logo of a different style
ROUND = Path(__file__).parent.parent / "examples" / "round_square.png"


@pytest.mark.slow
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


@pytest.mark.slow
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


# ---------------------------------------------------------------- the critic: a designer's eye on every letter
def test_critic_sees_a_seam_and_a_touching_dot():
    from khatt.critic import critique
    from khatt.library import Tagged
    from khatt.typeset import Layout
    st = Style(stem=20, hair=4, x_height=60, slant=0, white=20)
    gap = Layout("", st, shapes=[[[(0, 0), (40, 0), (40, 4), (0, 4)]], [[(40.4, 0), (80, 0), (80, 4), (40.4, 4)]],
                                 [[(0, 0), (6, 0), (6, 60), (0, 60)]]])
    # two pieces that do not quite meet leave a hairline seam only if the gap is real; here it is
    # closed by the union (0.4 < a tenth of the hairline), so no seam
    assert not [d for d in critique(gap) if d.kind == "crack"]
    # two stems standing side by side with a hairline gap between them, joined at the top
    wide = Layout("", st, shapes=[[[(0, 0), (20, 0), (20, 60), (0, 60)]], [[(22.5, 0), (42.5, 0), (42.5, 60), (22.5, 60)]],
                                  [[(0, 56), (42.5, 56), (42.5, 60), (0, 60)]]])
    assert [d for d in critique(wide) if d.kind == "crack"]
    dot = Tagged([[(22, 1), (37, 1), (37, 16), (22, 16)]], "engine", "نقطة")
    touch = Layout("", st, shapes=[[[(0, 0), (20, 0), (20, 60), (0, 60)]], dot])
    assert [d for d in critique(touch) if d.kind == "touch"]


@pytest.mark.slow
@pytest.mark.parametrize("example", ["bidonaq.png", "round_square.png"])
def test_every_letter_passes_the_critic(example):
    """Every letter, in every form, drawn in the style measured from the example logo."""
    from khatt.critic import specimen_defects
    r = run((EXAMPLE.parent / example).read_bytes(), "بيدوناق")
    st = Style(**{k: v for k, v in r["style"].items() if k in Style.__dataclass_fields__})
    _no_new_defects(example, specimen_defects(st))
    assert r["critic"]["word"] == []


@pytest.mark.slow
def test_every_letter_passes_the_critic_in_an_angular_style():
    from khatt.critic import specimen_defects
    st = Style(stem=11, hair=7.7, x_height=43.5, ascender=18.2, descender=18.2, slant=0, white=18.8, dot=9.2,
               top_round=0, ring_n=8, dot_n=5.5, join_in=0.2, join_out=0.23, angular=0.72, corner_r=8.3,
               tail_drop=0.0, tail_reach=1.9, contrast_power=2.3)
    _no_new_defects("angular", specimen_defects(st))


# ---------------------------------------------------------------- skeletons taken from the reference typefaces
def _body(word: str, st: Style):
    """The drawn word without its marks, as one upright geometry, and the layout."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    lay = typeset(word, st)
    geoms = []
    for sh in flat(lay.shapes):
        if getattr(sh, "name", "") in ("نقطة", "شرطة", "همزة"):
            continue
        acc = None
        for poly in sh:
            if len(poly) >= 3:
                q = Polygon(poly).buffer(0)
                acc = q if acc is None else acc.symmetric_difference(q)
        if acc is not None:
            geoms.append(acc)
    return unary_union(geoms), lay


def test_loop_of_sad_is_a_teardrop_with_the_tooth_at_its_point():
    from shapely.geometry import Point
    st = Style()
    body, lay = _body("ص", st)
    X, U, h = st.x_height, st.stem, st.hair
    L = 0.8 * X + 1.35 * U
    assert len(body.interiors) == 1                                  # one counter
    # the counter is tall at the round end and comes to a point at the tooth
    assert body.contains(Point(-L + 0.2 * U, 0.2 * X))                # the slope arrives at the foot of the tooth
    assert not body.contains(Point(-1.6 * U, 0.5 * X))                # white inside the round end
    assert body.contains(Point(-L - U / 2, 0.9 * X))                  # the tooth stands left of the loop, to the x-height
    assert not body.contains(Point(-L + 0.3 * U, 0.95 * X))           # and the loop's top has come down beside it


def test_tah_carries_its_stem_at_the_point_of_the_loop():
    from shapely.geometry import Point
    st = Style()
    body, _ = _body("ط", st)
    X, U, A = st.x_height, st.stem, st.ascender
    L = 0.8 * X + 1.35 * U
    assert body.contains(Point(-L - U / 2, X + 0.9 * A))              # the tall stem is the leftmost part
    assert not body.contains(Point(-L / 2, X + 0.3 * A))              # nothing rises over the loop


def test_hah_has_a_nose_at_the_right_above_the_middle():
    st = Style()
    body, _ = _body("حب", st)
    X = st.x_height
    xs, ys = zip(*[p for p in body.exterior.coords])
    right = max(xs)
    nose = [y for x, y in zip(xs, ys) if x > right - 0.02 * st.stem]
    assert 0.4 * X < sum(nose) / len(nose) < 0.85 * X                 # the rightmost point is the nose, not the top
    assert max(ys) <= X + 1e-6 and min(ys) >= -1e-6                   # joined ح stays between the lines


def test_medial_heh_has_an_eye_under_the_baseline():
    st = Style()
    body, _ = _body("بهب", st)
    assert len(body.interiors) == 2
    lows = sorted(min(y for _, y in r.coords) for r in body.interiors)
    assert lows[0] < -0.3 * st.descender and lows[1] > 0              # one counter under the line, one above it


def test_reh_hangs_from_the_top_of_the_tooth_when_the_logo_says_so():
    from shapely.geometry import Point
    X, U, h, W = 66.0, 23.0, 4.5, 22.0
    joined, lay = _body("بر", Style(top_join=1.0))
    plain, _ = _body("بر", Style())
    reh = [p for p in lay.placed if p.char == "ر"][0].x              # right edge of the ر stem
    gap = reh + W / 2                                                 # middle of the white between tooth and ر
    assert joined.contains(Point(gap, X - h / 2))                     # the hairline bar at the top
    assert not joined.contains(Point(gap, h / 2))                     # and no join on the baseline
    assert plain.contains(Point(reh + 0.2 * W, h / 2)) and not plain.contains(Point(gap, X - h / 2))
    assert lay.placed[1].glyph.stems[0][1] == X                       # the ر stem is a full tooth
    # after a round head there is no tooth to hang from: the baseline join stays
    round_, lay2 = _body("فر", Style(top_join=1.0))
    assert not round_.contains(Point([p for p in lay2.placed if p.char == "ر"][0].x + W / 2, X - h / 2))


# ---------------------------------------------------------------- the input guard
def _png(im) -> bytes:
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


def _measured(im):
    from khatt.pipeline import analyse
    an = analyse(_png(im), read=False)                 # the guard alone: reading by drawing is tested apart
    L, k = an["lines"][an["src"]], an["k"]
    return L.x_height * k, L.stem * k, L.slant, an["health"]


@pytest.fixture(scope="module")
def clean_logo():
    from PIL import Image
    im = Image.open(EXAMPLE).convert("L")
    return im, _measured(im)


def test_guard_leaves_a_clean_picture_alone(clean_logo):
    im, (X, U, slant, health) = clean_logo
    assert health["corrections"] == [] and health["warnings"] == [] and health["confidence"] == 1.0


@pytest.mark.parametrize("angle", [4, -7, 10])
def test_guard_levels_a_rotated_picture(clean_logo, angle):
    from PIL import Image
    im, (X, U, slant, _) = clean_logo
    x, u, s, health = _measured(im.rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC))
    assert [n["kind"] for n in health["corrections"]] == ["rotation"]
    assert abs(health["corrections"][0]["angle"] - angle) < 0.3          # the turn is found
    assert abs(x / X - 1) < 0.01 and abs(u / U - 1) < 0.02 and abs(s - slant) < 0.3   # and the letters' slant is untouched


def test_guard_takes_a_symbol_out_of_the_line(clean_logo):
    from PIL import Image, ImageDraw
    im, (X, U, slant, _) = clean_logo
    W, H = im.size
    pic = Image.new("L", (W + H, H), 255); pic.paste(im, (0, 0))
    ImageDraw.Draw(pic).ellipse([W + 30, 0.2 * H, W + H - 30, 0.8 * H], fill=0)
    x, u, s, health = _measured(pic)
    assert [n["kind"] for n in health["corrections"]] == ["symbol"]
    assert abs(x / X - 1) < 0.01 and abs(u / U - 1) < 0.01


@pytest.mark.parametrize("ground", ["gradient", "vignette", "grey"])
def test_guard_evens_out_the_ground(clean_logo, ground):
    from PIL import Image
    im, (X, U, slant, _) = clean_logo
    a = np.asarray(im).astype(float)
    H, W = a.shape
    if ground == "gradient":
        a = a * np.linspace(1.0, 0.47, W)[None, :]
    elif ground == "vignette":
        yy, xx = np.mgrid[0:H, 0:W]
        a = a * (1 - 0.25 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2))
    else:
        a = a * 0.6 + 40                                                  # flat, but neither black nor white
    x, u, s, health = _measured(Image.fromarray(a.astype(np.uint8)))
    kinds = [n["kind"] for n in health["corrections"]]
    assert ("ground" in kinds) == (ground != "grey")
    assert abs(x / X - 1) < 0.01 and abs(u / U - 1) < 0.02


def test_guard_fills_letters_drawn_as_outlines(clean_logo):
    from PIL import Image
    from scipy import ndimage as ndi
    im, (X, U, slant, _) = clean_logo
    a = np.asarray(im) < 128
    x, u, s, health = _measured(Image.fromarray(np.where(a & ~ndi.binary_erosion(a, iterations=4), 0, 255).astype(np.uint8)))
    assert "outline" in [n["kind"] for n in health["corrections"]]
    assert abs(x / X - 1) < 0.01 and abs(u / U - 1) < 0.03                # not the 4 px of the outline
    assert health["confidence"] <= 0.75                                   # and it says the reading is rough


def test_guard_says_when_the_reading_is_not_sure():
    """One stem is not a rhythm: the figure drops and the reason is given."""
    from PIL import Image, ImageDraw
    im = Image.new("L", (900, 400), 255)
    ImageDraw.Draw(im).rectangle([400, 100, 440, 300], fill=0)
    from khatt.pipeline import analyse
    try:
        h = analyse(_png(im))["health"]
    except ValueError:
        return                                                            # refusing to measure is also an honest answer
    assert h["confidence"] < 0.7 and h["warnings"]


# ---------------------------------------------------------------- the benchmark's yardstick
def test_reading_metric_knows_the_same_skeleton_from_another():
    """The distance used by the benchmark: weight does not count, shape does."""
    from PIL import Image, ImageDraw
    from khatt.bench.score import BASE_ROW, CANVAS, PX, RIGHT_COL, Shape, distance, read

    def draw(fn):
        im = Image.new("L", (CANVAS[1], CANVAS[0]), 0)
        fn(ImageDraw.Draw(im)); return Shape(np.array(im) > 0)
    x, b = RIGHT_COL, BASE_ROW
    tooth_light = draw(lambda d: d.rectangle([x - 6, b - PX, x, b], fill=1))
    tooth_bold = draw(lambda d: d.rectangle([x - 14, b - PX, x, b], fill=1))
    tall = draw(lambda d: d.rectangle([x - 6, b - 2 * PX, x, b], fill=1))
    ring = draw(lambda d: d.ellipse([x - 2 * PX, b - PX, x, b], outline=1, width=6))
    assert distance(tooth_light, tooth_bold) < 0.06                       # the same skeleton, two weights
    same = distance(tooth_light, tooth_bold)
    assert distance(tooth_light, tall) > 2 * same and distance(tooth_light, ring) > 0.15   # another height, another shape
    assert read(tooth_bold, {"ب": tooth_light, "ل": tall, "و": ring})[0] == "ب"
    assert read(draw(lambda d: d.ellipse([x - 2 * PX - 4, b - PX, x - 4, b], outline=1, width=10)), {"ب": tooth_light, "ل": tall, "و": ring})[0] == "و"


# ---------------------------------------------------------------- recipes: named parameters and variants
def test_every_parameter_has_a_rule_and_a_source():
    from khatt.recipes import BY_EYE, DERIVED, FITTED, MEASURED, PARAMS, handles
    assert PARAMS, "no named parameters registered"
    hs = handles(Style())
    assert {h["name"] for h in hs} == set(PARAMS)
    for h in hs:
        assert h["rule"] and h["source"] in (BY_EYE, DERIVED, FITTED, MEASURED)
        assert h["value"] == h["rule_value"] and not h["set_by_designer"]      # nothing moved yet


def test_a_designer_moves_a_parameter_without_touching_code():
    from khatt.recipes import handles, rule_value
    rule = rule_value(Style(), "meem.height")               # whatever the rule is now
    plain, _ = _body("م", Style())
    moved, _ = _body("م", Style(params={"meem.height": 0.6 * rule}))
    assert abs(plain.bounds[3] - rule) < 0.5                   # drawn by the rule
    assert abs(moved.bounds[3] - 0.6 * rule) < 0.5             # the designer's value, no code touched
    h = {x["name"]: x for x in handles(Style(params={"meem.height": 0.6 * rule}))}["meem.height"]
    assert h["set_by_designer"] and h["value"] == round(0.6 * rule, 3) and h["rule_value"] == round(rule, 3)


def test_a_designer_picks_another_variant_of_a_letter():
    from khatt.recipes import choices
    funnel, _ = _body("بعب", Style())
    ring, _ = _body("بعب", Style(variants={"ع.M": "حلقة بشقّ"}))
    assert funnel.symmetric_difference(ring).area > 50        # two different drawings
    from khatt.recipes import choose
    cs = {(c["letter"], c["form"]): c for c in choices(Style())}
    default = choose(Style(), "ع", "M").name                  # the engine's own choice, whichever it is now
    assert cs[("ع", "M")]["chosen"] == default and {"قمع", "حلقة بشقّ"} <= {o["name"] for o in cs[("ع", "M")]["options"]}
    assert {(c["letter"], c["form"]): c for c in choices(Style(variants={"ع.M": "حلقة بشقّ"}))}[("ع", "M")]["set_by_designer"]


def test_joined_feh_stands_on_a_neck_only_where_there_is_room():
    light, _ = _body("بفب", Style())                          # hairline 4.5: white under the head
    heavy, _ = _body("بفب", Style(hair=16.0))                 # a heavy hairline would leave a crack
    from shapely.geometry import Point
    lay = typeset("بفب", Style())
    fx = [p for p in lay.placed if p.char == "ف"][0].x
    assert not light.contains(Point(fx - 12.0, 7.0))          # white between baseline and head, beside the neck
    assert max(y for _, y in light.exterior.coords) <= 66.0 + 1e-6   # and the head stays under the x-height


@pytest.mark.slow
def test_every_letter_passes_the_critic_in_a_one_weight_curved_style():
    """A pen of one weight that draws curves (most sans typefaces): the hairline is as thick as the
    stem, so seams, pile-ups and thin strokes are judged for such a pen, and ح takes its leaning arm."""
    from khatt.critic import specimen_defects
    mono = Style(stem=24.0, hair=24.0, x_height=130.0, white=65.0, ascender=90.0, descender=60.0, slant=0.0,
                 join_in=0.25, join_out=0.12, contrast_power=6.0, dot=28.0)
    _no_new_defects("one_weight", specimen_defects(mono))
    body, _ = _body("حب", mono)
    xs, ys = zip(*body.exterior.coords)
    top_left = max(y for x, y in zip(xs, ys) if x < min(xs) + 0.2 * (max(xs) - min(xs)) or True)
    assert abs(top_left - 130.0) < 1.0                       # the arm starts at the x-height on the left
    nose = [y for x, y in zip(xs, ys) if x > max(xs) - 1.0]
    assert 0.5 * 130 < sum(nose) / len(nose) < 0.85 * 130    # and comes down to a nose at the right


def test_critic_still_sees_a_real_seam_in_a_one_weight_pen():
    from khatt.critic import critique
    from khatt.library import Tagged
    mono = Style(stem=24.0, hair=24.0, x_height=130.0, white=65.0, ascender=90.0, descender=60.0, slant=0.0)
    lay = typeset("ب", mono)
    # two blocks that fail to meet by a sliver
    lay.shapes = [Tagged([[(0, 0), (60, 0), (60, 130), (0, 130)]], "engine", "a"),
                  Tagged([[(63, 0), (120, 0), (120, 130), (63, 130)]], "engine", "b")]
    assert any(d.kind == "crack" for d in critique(lay, "x"))


# ---------------------------------------------------------------- the self-check
def test_self_check_redraws_the_logo_word_and_says_what_differs():
    from khatt.pipeline import analyse
    from khatt.selfcheck import self_check
    an = analyse(EXAMPLE.read_bytes(), texts=["بيدوناق"])
    L, st, k = an["lines"][an["src"]], an["style"], an["k"]
    good = self_check(L, "بيدوناق", st, k)
    assert good["overall"] >= FLOOR and all(x["verdict"] == "يطابق" for x in good["letters"])
    assert [x["letter"] for x in good["letters"]] == list("بيدوناق") and not good["notes"]
    # the same logo read with a rule that is wrong for one letter: that letter is the one named
    wrong = Style(**{**st.__dict__, "params": {"dal.base": 5 * st.stem, "dal.flag": 3 * st.stem}})
    bad = self_check(L, "بيدوناق", wrong, k)
    by = {x["letter"]: x["match"] for x in bad["letters"]}
    assert by["د"] < 0.75 and by["د"] == min(by.values()) and bad["overall"] < good["overall"]


@pytest.mark.slow
def test_run_reports_the_self_check(tmp_path):
    r = run(EXAMPLE.read_bytes(), "ناب", texts=["بيدوناق"], out_dir=str(tmp_path))
    assert r["self_check"]["text"] == "بيدوناق" and r["self_check"]["overall"] >= FLOOR
    assert (tmp_path / "selfcheck.png").exists() and "الفحص الذاتي" in r["report_md"]


# ---------------------------------------------------------------- the editor (the judge's seat)
@pytest.fixture(scope="module")
def read_example():
    from khatt.editor import read_logo
    return read_logo(EXAMPLE.read_bytes(), "بيدوناق")


@pytest.mark.slow
def test_editor_reads_a_logo_once(read_example):
    import json
    r = read_example
    json.dumps(r)                                              # everything goes over the wire as JSON
    assert r["health"]["confidence"] == 1.0 and r["read"] == "بيدوناق" and r["read_sure"]
    assert r["self_check"]["overall"] >= FLOOR and r["self_check_png"]
    assert r["style"]["stem"] > 0 and r["measures"]["الجذع"] == pytest.approx(r["style"]["stem"], abs=0.01)


def test_the_logos_part_library_travels_with_its_style_and_builds_the_letters():
    """The parts box cut from the logo (its stem, the feet of its teeth, its tail...) is kept in the
    style the editor gets, and the letters drawn from that style are built of it."""
    from khatt.library import Library
    from khatt.pipeline import analyse
    an = analyse(EXAMPLE.read_bytes(), texts=["بيدوناق"], read=False)
    lib = an["lib"]
    assert lib.parts
    back = Library.from_dict(lib.to_dict())
    assert set(back.parts) == set(lib.parts)
    st = Style(**{**an["style"].__dict__, "library": lib.to_dict()})
    sources = {getattr(sh, "source", "") for p in typeset("بيت", st).placed for sh in flat(p.glyph.shapes)}
    assert "logo" in sources
    plain = {getattr(sh, "source", "") for p in typeset("بيت", an["style"]).placed for sh in flat(p.glyph.shapes)}
    assert "logo" not in plain


@pytest.mark.slow
def test_the_editor_keeps_the_pieces_that_earn_their_place(read_example):
    lib = read_example["style"].get("library") or {}
    kept = set((lib.get("parts") or {}))
    assert set(read_example["self_check"].get("library") or []) == kept

@pytest.mark.slow
def test_editor_draws_with_the_designers_changes(read_example):
    from khatt.editor import draw
    read_example = {**read_example, "style": {**read_example["style"], "ref_auto": False}}
    plain = draw(read_example["style"], "معهد")
    assert plain["svg"].startswith("<svg") and plain["critic"] == []
    used = {h["name"] for h in plain["handles"] if h["used"]}
    assert {"meem.height", "ain.eye_w", "dal.base"} <= used and "kaf.arm" not in used      # the word's own letters
    own = [c for c in plain["choices"] if any(o["name"] not in ("المحرك", "من المرجع") for o in c["options"])]
    from khatt.recipes import choose
    from khatt.editor import style_of
    assert [(c["letter"], c["form"], c["chosen"]) for c in own] == [("ع", "M", choose(style_of(read_example["style"], {}, {}), "ع", "M").name)]
    rule = {h["name"]: h for h in plain["handles"]}["meem.height"]
    assert rule["range"][0] < rule["value"] < rule["range"][1] and not rule["set_by_designer"]
    moved = draw(read_example["style"], "معهد", {"meem.height": rule["range"][0]}, {"ع.M": "حلقة بشقّ"})
    assert moved["svg"] != plain["svg"]
    h = {x["name"]: x for x in moved["handles"]}["meem.height"]
    assert h["set_by_designer"] and h["value"] == rule["range"][0] and h["rule_value"] == rule["rule_value"]
    assert moved["choices"][0]["chosen"] == "حلقة بشقّ" and moved["choices"][0]["set_by_designer"]
    with pytest.raises(ValueError):
        draw(read_example["style"], "abc")


@pytest.mark.slow
def test_editor_logs_what_was_changed(read_example, tmp_path):
    from khatt.editor import draw, load_edits, save_edit, summarise
    style = {**read_example["style"], "ref_auto": False}
    rule = {h["name"]: h for h in draw(style, "معهد", check=False)["handles"]}["meem.height"]["rule_value"]
    log = tmp_path / "edits.jsonl"
    out = save_edit(style, "معهد", {"meem.height": 0.8 * rule, "dal.base": None}, {"ع.M": "حلقة بشقّ"},
                    {"meem.height": "always"}, "logo-a.png", log)
    assert out["params"] == 1 and out["variants"] == 1
    # a value equal to the rule, or a variant that is already the default, is not an edit
    assert save_edit(style, "معهد", {"meem.height": rule}, {"ع.M": "قمع"}, {}, "logo-a.png", log)["saved"] == 0
    save_edit(style, "ميم", {"meem.height": 0.7 * rule}, {}, {}, "logo-b.png", log)
    edits = load_edits(log)
    assert len(edits) == 3 and edits[0]["measures"]["stem"] == style["stem"]
    (m,) = edits[0]["params"]
    assert m["name"] == "meem.height" and m["scope"] == "always" and abs(m["rule"] - rule) < 1e-2
    assert abs(m["chosen"] - 0.8 * rule) < 1e-3 and abs(m["ratio"] - 0.8) < 1e-3
    assert edits[0]["variants"] == [{"form": "ع.M", "chosen": "حلقة بشقّ", "instead_of": "قمع", "scope": "logo"}]
    s = summarise(edits)[0]
    assert s["name"] == "meem.height" and s["times"] == 2 and s["logos"] == 2 and s["same_way"] and s["always"] == 1
    assert abs(s["median_ratio"] - 0.75) < 1e-3


# ---------------------------------------------------------------- learning from the edits
def test_a_correction_repeated_on_several_logos_becomes_a_proposal(tmp_path):
    from khatt.learning import propose
    def edit(logo, ratio, name="meem.height"):
        return {"logo": logo, "params": [{"name": name, "rule": 50.0, "chosen": 50.0 * ratio, "ratio": ratio, "scope": "logo"}]}
    once = [edit("a", 0.8)]
    twice_opposite = [edit("a", 0.8), edit("b", 1.2), edit("c", 0.9)]
    three = [edit("a", 0.8), edit("b", 0.9), edit("c", 0.85), edit("c", 0.7)]
    assert propose(once) == [] and propose(twice_opposite) == []           # one logo, or two ways: not a rule
    (p,) = propose(three)
    assert p["name"] == "meem.height" and p["logos"] == 3 and abs(p["factor"] - 0.825) < 1e-6


def test_a_learned_factor_sits_on_top_of_the_rule_and_can_be_removed(tmp_path):
    import json
    from khatt import recipes
    from khatt.learning import apply
    st = Style()
    plain = recipes.P(st, "meem.height")
    path = tmp_path / "learned.json"
    try:
        apply([{"name": "meem.height", "factor": 0.8, "logos": 3, "times": 4}], path)
        assert abs(recipes.P(st, "meem.height") - 0.8 * plain) < 1e-6
        assert abs(recipes.P(Style(params={"meem.height": 33.0}), "meem.height") - 33.0) < 1e-9   # the designer still wins
        h = {x["name"]: x for x in recipes.handles(st)}["meem.height"]
        assert "تعلّمه" in h["rule"] and abs(h["rule_value"] - round(0.8 * plain, 3)) < 1e-6
        data = json.loads(path.read_text(encoding="utf-8")); data.pop("meem.height")
        path.write_text(json.dumps(data), encoding="utf-8")
        recipes.load_learned(path)
        assert abs(recipes.P(st, "meem.height") - plain) < 1e-9
    finally:
        recipes.LEARNED.clear()


@pytest.mark.slow
def test_a_letter_the_logo_draws_its_own_way_is_taken_from_it(monkeypatch):
    """The engine builds ق wrongly (as ا) on the example: the self-check names it, cuts the logo's ق
    out of the logo, the logo's word matches again, and a new word with an isolated ق uses the cut."""
    from khatt import letters
    from khatt.pipeline import analyse
    from khatt.recipes import choices
    from khatt.selfcheck import check_and_take
    monkeypatch.setitem(letters.LETTERS, "ق", ("D", lambda s, f: letters.alef(s, f, None)))
    an = analyse(EXAMPLE.read_bytes(), texts=["بيدوناق"])
    L, k, st = an["lines"][an["src"]], an["k"], an["style"]
    r = check_and_take(L, "بيدوناق", st, k)
    assert {x["letter"]: x["verdict"] for x in r["letters"]}["ق"] == "يختلف"
    assert list(r["taken"]) == ["ق.i"] and set(st.taken) == {"ق.i"}
    after = {x["letter"]: x["match"] for x in r["letters_after"]}
    assert after["ق"] > 0.85 and r["overall_after"] > r["overall"] + 0.05
    q = [p for p in typeset("بق ق", st).placed if p.char == "ق"]
    assert {p.form for p in q} == {"i", "F"}
    assert all([s.name for s in p.glyph.shapes] == ["من الشعار"] for p in q)      # the final form borrows the isolated cut
    # the designer can still ask for the engine's letter
    st2 = Style(**{**st.__dict__, "variants": {"ق.i": "المحرك"}})
    q2 = [p for p in typeset("ق", st2).placed][0]
    assert "من الشعار" not in [getattr(s, "name", "") for s in q2.glyph.shapes]
    c = {(x["letter"], x["form"]): x for x in choices(st)}[("ق", "i")]
    assert c["chosen"] == "من الشعار" and {o["name"] for o in c["options"]} == {"من الشعار", "المحرك"}


# ---------------------------------------------------------------- reading by drawing
@pytest.mark.slow
def test_the_example_is_read_by_drawing_when_the_stems_are_not_enough():
    from khatt.pipeline import analyse, own_text
    an = analyse(EXAMPLE.read_bytes())
    R = an["readings"][an["src"]]
    assert R.text_drawn and R.text_guess == "بيدوناق" and R.drawn_match >= 0.75
    assert own_text(R) == "بيدوناق"
    assert any("بالرسم" in w["text"] for w in an["health"]["warnings"])


@pytest.mark.slow
def test_a_poor_reading_by_drawing_is_not_trusted():
    from khatt.pipeline import analyse, own_text
    an = analyse((EXAMPLE.parent / "round_square.png").read_bytes())       # "سفينة نجم"
    R = an["readings"][an["src"]]
    assert not R.text_drawn and own_text(R) == "" and R.drawn_match < 0.75
    an2 = analyse((EXAMPLE.parent / "round_square.png").read_bytes(), read=False)
    assert an2["readings"][an2["src"]].drawn_match == 0.0                    # and reading can be skipped


def test_nothing_is_taken_from_the_logo_on_a_guessed_reading(monkeypatch):
    from khatt import letters
    from khatt.pipeline import analyse
    from khatt.selfcheck import check_and_take
    monkeypatch.setitem(letters.LETTERS, "ق", ("D", lambda s, f: letters.alef(s, f, None)))
    an = analyse(EXAMPLE.read_bytes(), texts=["بيدوناق"])
    L, k, st = an["lines"][an["src"]], an["k"], an["style"]
    r = check_and_take(L, "بيدوناق", st, k, take=False)
    assert r["taken"] == {} and st.taken == {} and any("بالرسم" in n for n in r["notes"])


def test_a_typed_arabic_text_makes_the_line_arabic():
    """A thin line the stems take for Latin: the designer's Arabic word decides."""
    from khatt.understand import read_line
    from khatt.pipeline import analyse
    an = analyse(EXAMPLE.read_bytes(), read=False)
    L = an["lines"][an["src"]]
    assert read_line(L, "بيدوناق").script == "arabic"
    R = read_line(L, "Sirang")
    assert R.script == an["readings"][an["src"]].script          # a Latin text changes nothing


# ---------------------------------------------------------------- the length of a letter (kashida)
def test_a_letter_with_a_floor_can_be_drawn_longer_and_one_without_cannot():
    from khatt.typeset import stretch_glyph, stretch_run
    st = Style()
    beh = LETTERS["ب"][1](st, "i")
    alef = LETTERS["ا"][1](st, "i")
    assert stretch_run(beh, st) is not None and stretch_run(alef, st) is None
    e = 2.0 * st.stem
    longer = stretch_glyph(beh, e, st)
    w = lambda gl: (lambda xs: max(xs) - min(xs))([x for sh in flat(gl.shapes) for poly in sh for x, _ in poly])
    assert w(longer) == pytest.approx(w(beh) + e, abs=0.05 * st.stem)
    assert longer.left == pytest.approx(beh.left - e) and longer.entry_x == beh.entry_x
    assert stretch_glyph(alef, e, st) is None


def test_a_joining_letter_lengthens_its_join_and_a_missing_form_borrows_its_partner():
    st = Style()
    plain = typeset("كا", st)
    e = 0.8 * st.x_height
    long = typeset("كا", Style(**{**st.__dict__, "stretch": {"ك.I": e}}))
    assert (plain.bbox()[2] - plain.bbox()[0]) + e == pytest.approx(long.bbox()[2] - long.bbox()[0], abs=0.05 * st.stem)
    (a0, b0), (a1, b1) = plain.joins[0], long.joins[0]
    assert (b1 - a1) - (b0 - a0) == pytest.approx(e, abs=1e-6)
    assert long.placed[0].join_out is not None
    # the medial ك was not seen: it takes the initial's length; the designer's handle wins over both
    mid = typeset("بكا", Style(**{**st.__dict__, "stretch": {"ك.I": e}}))
    base = typeset("بكا", st)
    assert (mid.bbox()[2] - mid.bbox()[0]) - (base.bbox()[2] - base.bbox()[0]) == pytest.approx(e, abs=0.05 * st.stem)
    set_ = typeset("كا", Style(**{**st.__dict__, "stretch": {"ك.I": e}, "params": {"مد.ك.I": 0.0}}))
    assert set_.bbox()[2] - set_.bbox()[0] == pytest.approx(plain.bbox()[2] - plain.bbox()[0], abs=1e-6)


@pytest.mark.slow
def test_the_length_a_logo_gives_a_letter_is_measured_back():
    """A logo drawn by the engine with its medial ي one x-height longer: the self-check finds the
    length again, the word matches better with it, and it is offered as a handle."""
    from PIL import Image
    from khatt.pipeline import analyse
    from khatt.recipes import handles
    from khatt.selfcheck import check_and_take
    an = analyse(EXAMPLE.read_bytes(), texts=["بيدوناق"], read=False)
    st = an["style"]
    long = Style(**{**st.__dict__, "stretch": {"ي.M": 1.0 * st.x_height}})
    png = to_png(typeset("بيدوناق", long), height=300)
    an2 = analyse(png, texts=["بيدوناق"], read=False)
    L, k, st2 = an2["lines"][an2["src"]], an2["k"], an2["style"]
    r = check_and_take(L, "بيدوناق", st2, k)
    # the letters after ي no longer have to be slid to their places, and the word matches no worse
    slid = lambda xs: sum(abs(x["shift"]) for x in xs) / len(xs)
    assert r["overall_after"] >= r["overall"] and slid(r["letters_after"]) < 0.3 * slid(r["letters"])
    got = st2.stretch.get("ي.M", 0.0) / st2.x_height
    assert 0.75 <= got <= 1.25 and r["stretch"]
    assert sum(abs(v) for q, v in st2.stretch.items() if q != "ي.M") < 0.3 * st2.x_height
    h = {x["name"]: x for x in handles(st2)}["مد.ي.M"]
    assert h["shape"] == "المدّ" and h["value"] == pytest.approx(st2.stretch["ي.M"], abs=1e-3)


def test_the_editor_offers_the_length_and_saves_its_change(tmp_path):
    from khatt.editor import draw, save_edit, style_of
    st = Style(stretch={"ك.I": 30.0})
    d = st.to_dict()
    assert style_of(d, {"مد.ك.I": 10.0}).params == {"مد.ك.I": 10.0}
    out = draw(d, "كتاب")
    h = {x["name"]: x for x in out["handles"]}["مد.ك.I"]
    assert h["used"] and h["value"] == pytest.approx(30.0)
    assert not {x["name"]: x for x in draw(d, "باب")["handles"]}["مد.ك.I"]["used"]
    rec = save_edit(d, "كتاب", {"مد.ك.I": 10.0}, {}, path=tmp_path / "e.jsonl")
    assert rec["params"] == 1


# ---------------------------------------------------------------- the rules of the letters, measured on the logo
def test_a_wrong_letter_rule_is_measured_back_from_the_logo():
    """The general rule for the base of د made twice too long: measured on the logo's own word, the
    base comes back to its length, and the word matches the logo again."""
    from khatt import recipes
    from khatt.pipeline import analyse
    from khatt.selfcheck import fit_letters, self_check
    an = analyse(EXAMPLE.read_bytes(), texts=["بيدوناق"], read=False)
    L, k, st = an["lines"][an["src"]], an["k"], an["style"]
    right = recipes.general_rule(st, "dal.base")
    recipes.LEARNED["dal.base"] = 2.0
    try:
        bad = self_check(L, "بيدوناق", st, k)
        fitted, chosen, traits = fit_letters(L, "بيدوناق", st, k)
        assert "dal.base" in fitted and abs(fitted["dal.base"] / right - 1) <= 0.2
        good = self_check(L, "بيدوناق", Style(**{**st.__dict__, "fitted": fitted, "chosen": chosen}), k)
        d = lambda r: {x["letter"]: x["match"] for x in r["letters"]}["د"]
        assert d(good) > d(bad) + 0.03 and good["overall"] > bad["overall"]
    finally:
        recipes.LEARNED.clear()


def test_measured_rules_serve_the_logos_letters_and_the_manner_serves_the_others():
    st = Style()
    plain_d, plain_b = typeset("د", st), typeset("ب", st)
    width = lambda lay: lay.bbox()[2] - lay.bbox()[0]
    from khatt.recipes import general_rule
    m = Style(**{**st.__dict__, "fitted": {"dal.base": 2 * general_rule(st, "dal.base"), "beh.width": 0.5 * general_rule(st, "beh.width")},
                 "measured_on": "د.i", "traits": {"عرض": 1.5}})
    assert width(typeset("د", m)) > width(plain_d) + 0.5 * general_rule(st, "dal.base")   # د is in the logo: its own measure
    assert width(typeset("ذ", m)) > width(typeset("ذ", st)) + 0.5 * general_rule(st, "dal.base")   # ذ shares its skeleton
    assert width(typeset("ب", m)) > width(plain_b)              # ب is not: the logo's manner (wider), not ب's own value
    on_b = Style(**{**m.__dict__, "measured_on": "ب.i د.i"})
    assert width(typeset("ب", on_b)) < width(plain_b)           # once ب is a letter of the logo, its measure


def test_a_measure_serves_the_letter_form_it_was_measured_on():
    """م.I's measures are not م.F's: a logo that shows one form of a letter says nothing of its other forms."""
    from khatt.recipes import general_rule
    st = Style()
    width = lambda lay: lay.bbox()[2] - lay.bbox()[0]
    m = Style(**{**st.__dict__, "fitted": {"dal.base": 2 * general_rule(st, "dal.base")}, "measured_on": "د.F"})
    assert width(typeset("بد", m)) > width(typeset("بد", st)) + 0.5 * general_rule(st, "dal.base")   # د.F: measured
    assert width(typeset("د", m)) == pytest.approx(width(typeset("د", st)), abs=1e-6)              # د.i: not


def test_the_logos_pieces_say_which_letters_they_hold():
    """A letter that joins nothing after it ends a piece of ink: the word is read piece by piece, not
    by where the engine's letters slide to."""
    from khatt.pipeline import analyse
    from khatt.selfcheck import _align, _slide
    st = Style()
    an = analyse(to_png(typeset("بيدوناق", st), height=300), texts=["بيدوناق"], read=False)
    L, k, st2 = an["lines"][an["src"]], an["k"], an["style"]
    up = Style(**{**st2.__dict__, "slant": 0.0})
    lay = typeset("بيدوناق", up)
    logo = np.asarray(L.B, bool)
    s = 1.0 / k
    xs = [p[0] for sh in lay.shapes for poly in sh for p in poly]
    x_of = float(np.flatnonzero(logo.any(axis=0))[-1]) - max(xs) * s
    slid = _slide(lay, up, logo, s, x_of, L.row_of_y(0.0), st2.pitch * s)
    got = _align(logo, [(p, m, dx) for p, m, dx, _ in slid], L.row_of_y(0.0), st2.x_height * s, st2.pitch * s)
    assert got is not None
    runs = [(a, b) for _, a, b in got[1]]
    assert runs == [(0, 3), (3, 4), (4, 6), (6, 7)]            # بيد | و | نا | ق


def test_two_cut_cups_share_their_arm_and_a_cup_rises_into_alef():
    """Cut from the logo, a cup that ends in an upright arm and a letter that begins with an upright
    stand on one stroke, as the logo's cups do; a bowl's side is no arm."""
    from khatt.typeset import _upright
    st = Style()
    X, w = st.x_height, st.stem
    cup = {"polys": [[(0, X), (0, 0), (-3 * w, 0), (-3 * w, X), (-2 * w, X), (-2 * w, w), (-w, w), (-w, X)]], "piece": True}
    lay = typeset("بب", Style(**{**st.__dict__, "taken": {"ب.I": cup, "ب.F": cup}}))
    a, b = lay.placed
    assert _upright(a.glyph, st, "left") and _upright(b.glyph, st, "right")
    assert lay.bbox()[2] - lay.bbox()[0] == pytest.approx(5 * w, abs=0.1 * w)   # two cups of three stems, one arm between
    assert not lay.joins                                                     # and no join of the engine's
    pixels = {q: {"polys": v["polys"]} for q, v in (("ب.I", cup), ("ب.F", cup))}
    assert typeset("بب", Style(**{**st.__dict__, "taken": pixels})).joins     # cut where the engine's letters stood: its joins


def test_alef_and_lam_on_the_logos_reh_are_the_designers_choice():
    from khatt.recipes import ON_REH, choices
    from khatt.typeset import find_cut, stem_from_reh
    st = Style()
    X, w = st.x_height, st.stem
    # an upright ر: a stem from the x-height down past the baseline, its foot turned to the left
    reh = {"polys": [[(0, X), (0, -0.3 * X), (-2 * w, -0.4 * X), (-2 * w, -0.4 * X + w), (-w, -0.3 * X + w), (-w, X)]]}
    on = Style(**{**st.__dict__, "taken": {"ر.F": reh}, "seen": "ر.F", "stem_kin": True})
    alef = stem_from_reh(on, "ا", "i")
    ys = [y for ring in alef["polys"] for _, y in ring]
    top = max(y for sh in flat(LETTERS["ا"][1](on, "i").shapes) for poly in sh for _, y in poly)
    assert min(ys) == pytest.approx(0.0, abs=1e-6) and max(ys) == pytest.approx(top, abs=1e-6)
    lam = stem_from_reh(on, "ل", "F")
    assert min(y for ring in lam["polys"] for _, y in ring) < -0.2 * X               # ل alone keeps the ر's fall
    off = Style(**{**on.__dict__, "stem_kin": False})
    assert find_cut(off, "ا", "i")[0] is None                                      # the engine's ا, unless chosen
    assert find_cut(Style(**{**off.__dict__, "variants": {"ا.i": ON_REH}}), "ا", "i")[0] is not None
    assert any(c["letter"] == "ا" and any(o["name"] == ON_REH for o in c["options"]) for c in choices(off))
    lean = {"polys": [[(0, 0.6 * X), (-3 * w, -0.3 * X), (-4 * w, -0.3 * X), (-w, 0.6 * X)]]}
    assert stem_from_reh(Style(**{**on.__dict__, "taken": {"ر.F": lean}}), "ا", "i") is None   # a ر that leans is no stem


def test_seen_is_built_of_the_logos_tooth_where_its_teeth_are_cups():
    """A logo that hangs its teeth as cups sharing an arm (Mootanroo) builds س of three such cups;
    a tooth cut with its joins on either side builds it only when the designer asks."""
    from khatt.recipes import TEETH_CHOICE
    from khatt.typeset import find_cut, teeth_from_logo
    st = Style()
    X, w = st.x_height, st.stem
    cup = [[(0, X), (0, 0), (-3 * w, 0), (-3 * w, X), (-2 * w, X), (-2 * w, w), (-w, w), (-w, X)]]
    st_cup = Style(**{**st.__dict__, "taken": {"ت.I": {"polys": cup, "piece": True}}, "seen": "ت.I"})
    seen, _, kin = find_cut(st_cup, "س", "M")
    assert seen is not None and kin
    xs = [x for ring in seen["polys"] for x, _ in ring]
    assert max(xs) - min(xs) == pytest.approx(7 * w, abs=0.1 * w)          # three cups, two arms shared
    lay = typeset("شمس", st_cup)
    assert lay.placed[0].glyph.shapes[0].source == "logo" and lay.placed[2].glyph.shapes[0].source == "logo"
    plain = Style(**{**st.__dict__, "taken": {"ت.I": {"polys": cup}}, "seen": "ت.I"})     # cut where the engine's letter stood
    assert find_cut(plain, "س", "M")[0] is None
    assert find_cut(Style(**{**plain.__dict__, "variants": {"س.M": TEETH_CHOICE}}), "س", "M")[0] is not None
    assert teeth_from_logo(Style(), "س", "M") is None                       # no tooth in the logo


def test_kaf_on_the_logos_reh_carries_its_arm_or_its_mark():
    from khatt.recipes import ON_REH
    from khatt.typeset import find_cut
    st = Style()
    X, w = st.x_height, st.stem
    reh = {"polys": [[(0, X), (0, -0.3 * X), (-2 * w, -0.4 * X), (-2 * w, -0.4 * X + w), (-w, -0.3 * X + w), (-w, X)]]}
    on = Style(**{**st.__dict__, "taken": {"ر.F": reh}, "seen": "ر.F", "variants": {"ك.I": ON_REH, "ك.F": ON_REH}})
    initial = find_cut(on, "ك", "I")[0]
    xs = [x for ring in initial["polys"] for x, _ in ring]
    assert min(xs) < -2.5 * w                                               # the arm reaches out to the left
    final = find_cut(on, "ك", "F")[0]
    assert final.get("marks_dx") is not None
    lay = typeset("مك", on)
    mark = [sh for sh in lay.placed[-1].glyph.shapes if getattr(sh, "name", "") == "شرطة"]
    stem_left = -w + lay.placed[-1].x
    assert mark and max(x for sh in mark for poly in sh for x, _ in poly) + lay.placed[-1].x < stem_left


def test_a_siblings_body_serves_the_letters_the_logo_lacks():
    """موتنرو draws its ن as a cup hanging from the top line: ب and ي of new words take that body."""
    from khatt.typeset import find_cut
    cut = {"polys": [[[0, 0], [-4, 0], [-4, 30], [0, 30]]]}
    st = Style(taken={"ن.M": cut}, seen="م.I و.F ت.I ن.M")
    got, _, kin = find_cut(st, "ب", "M")
    assert got is cut and kin
    assert find_cut(st, "ن", "M")[0] is cut and not find_cut(st, "ن", "M")[2]   # ن keeps its own
    assert find_cut(st, "ت", "I")[0] is None                 # ت is in the logo: its own drawing, not ن's

def test_a_cut_outline_is_cleaned_of_pixel_notches_and_facets_but_keeps_its_corners():
    from shapely.geometry import Polygon, Point
    from khatt.selfcheck import clean_outline
    # a square with a notch half a pixel wide cut into its top
    sq = Polygon([(0, 0), (10, 0), (10, 10), (5.25, 10), (5.25, 8), (4.75, 8), (4.75, 10), (0, 10)])
    got = clean_outline(sq, 1.0)
    assert got.area == pytest.approx(100, rel=0.02)          # the notch is filled
    ring = list(got.exterior.coords)
    for corner in ((0, 0), (10, 0), (10, 10), (0, 10)):      # the square's corners stay sharp
        assert min((x - corner[0]) ** 2 + (y - corner[1]) ** 2 for x, y in ring) < 1e-6
    # a bowl 20 pixels across traced in facets 3 pixels long comes out rounder (gentler turns), as
    # thick as it was (the outline moves in by a fraction of a pixel)
    import math
    def sharpest(poly):
        pts = list(poly.exterior.coords)[:-1]
        worst = 0.0
        for i in range(len(pts)):
            a, v, b = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            t = math.atan2(b[1] - v[1], b[0] - v[0]) - math.atan2(v[1] - a[1], v[0] - a[0])
            worst = max(worst, abs((t + math.pi) % (2 * math.pi) - math.pi))
        return worst
    facets = Point(0, 0).buffer(10, 5)                       # 20 facets of about 3 pixels
    round_ = clean_outline(facets, 1.0)
    assert sharpest(round_) < 0.5 * sharpest(facets)
    assert round_.area == pytest.approx(facets.area, rel=0.03)


def test_the_layout_cache_rebuilds_only_what_a_change_touches():
    from khatt.recipes import general_rule
    st = Style()
    cache = {}
    typeset("بيد", st, cache=cache)
    before = {key: g for key, (g, _) in cache.items()}
    st2 = Style(**{**st.__dict__, "fitted": {"dal.base": 1.5 * general_rule(st, "dal.base")}})
    lay = typeset("بيد", st2, cache=cache)
    after = {key: g for key, (g, _) in cache.items()}
    changed = {key[0] for key in before if before[key] is not after[key]}
    assert changed == {"د"}
    fresh = typeset("بيد", st2)
    assert lay.bbox() == pytest.approx(fresh.bbox())


def test_a_letter_cut_from_the_logo_draws_its_siblings_with_their_own_marks():
    """ب cut from the logo draws ن (same body when joined) with ن's dot above, not ب's below; a letter
    form the logo has keeps its own drawing."""
    st = Style()
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    beh = LETTERS["ب"][1](st, "M")
    ink = unary_union([Polygon(poly).buffer(0) for sh in flat(beh.shapes) for poly in sh])   # as a logo's outline
    rings = [list(q.exterior.coords)[:-1] for q in getattr(ink, "geoms", [ink])]
    cut = Style(**{**st.__dict__, "taken": {"ب.M": {"polys": rings}}, "seen": "ب.M"})
    noon = [p for p in typeset("بنب", cut).placed if p.char == "ن"][0]
    names = [n for _, n in noon.parts]
    assert "من الشعار" in names
    ys = [y for sh in flat(noon.glyph.shapes) if getattr(sh, "name", "") == "من الشعار" for poly in sh for _, y in poly]
    assert min(ys) > -0.05 * st.x_height                 # ب's dot under the baseline was taken off the body
    marks = [sh for sh in flat(noon.glyph.shapes) if getattr(sh, "name", "") == "نقطة"]
    assert marks and min(y for sh in marks for poly in sh for _, y in poly) > 0.9 * st.x_height     # ن's own dot, above
    own = Style(**{**cut.__dict__, "seen": "ب.M ن.M"})
    noon2 = [p for p in typeset("بنب", own).placed if p.char == "ن"][0]
    assert "من الشعار" not in [n for _, n in noon2.parts]
    assert [p for p in typeset("بنب", own).placed if p.char == "ب"][0].parts[0][1] == "من الشعار"


def test_compound_parts_inside_compound_parts_are_laid_out():
    from khatt.library import Compound, Tagged
    from khatt.typeset import flat
    sq = [[(0, 0), (1, 0), (1, 1), (0, 1)]]
    nested = [Compound([Tagged(sq, "logo", "a"), Compound([Tagged(sq, "engine", "b")])])]
    assert [s.name for s in flat(nested)] == ["a", "b"]


# ---------------------------------------------------------------- the structure of a reference typeface
def _a_reference():
    from khatt.reference import faces
    fs = [f for f in faces() if "vazirmatn-400" in f]
    if not fs:
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    return fs[0]


def test_a_letter_is_built_on_a_reference_structure_with_the_logos_pen():
    from khatt.reference import ref_glyph
    path = _a_reference()
    st = Style(slant=0.0, ref_how="strokes")
    sad = ref_glyph(path, "ص", "M", st)
    body = sad.shapes[0]
    assert len(body) >= 2                                   # its loop keeps an open counter under the logo's pen
    ys = [y for poly in body for _, y in poly]
    assert max(ys) == pytest.approx(st.x_height, abs=0.25 * st.x_height) and min(ys) > -0.15 * st.x_height
    hah = ref_glyph(path, "ح", "I", st)
    assert hah.exit_x is not None and hah.left < -st.stem
    beh = ref_glyph(path, "ت", "M", st)                     # the marks are the engine's (the logo's): two dots above
    dots = [sh for sh in beh.shapes[1:] if getattr(sh, "name", "") in ("نقطة", "شرطة")]
    assert dots and all(min(y for p in sh for _, y in p) > 0.9 * st.x_height for sh in dots)


def test_a_letter_is_taken_whole_from_the_reference_outline_at_the_logos_weight():
    """The typeface's own outline, not its centre line redrawn (which broke where a heavy typeface's
    strokes meet): whole, its counters open, as thick as the logo's stem."""
    from shapely.geometry import Polygon
    from skimage.morphology import skeletonize
    from scipy import ndimage as ndi_
    from khatt.reference import ref_outline
    path = _a_reference()
    for st in (Style(slant=0.0), Style(slant=0.0, stem=1.6 * Style().stem)):
        for c, f in (("ع", "I"), ("ح", "I"), ("ص", "M"), ("ه", "i")):
            g = ref_outline(path, c, f, st)
            body = g.shapes[0]
            acc = None
            for ring in body:
                q = Polygon(ring).buffer(0)
                acc = q if acc is None else acc.symmetric_difference(q)
            assert len(getattr(acc, "geoms", [acc])) == 1, (c, f)                 # one piece
            if c in "صه":
                assert any(len(p.interiors) for p in getattr(acc, "geoms", [acc])), (c, f)   # the counter open
            # its weight: twice the half-width along its centre line, at 4 pixels a unit
            x0, y0, x1, y1 = acc.bounds
            from PIL import Image, ImageDraw
            k = 4.0
            im = Image.new("L", (int((x1 - x0) * k) + 4, int((y1 - y0) * k) + 4), 0)
            for poly in getattr(acc, "geoms", [acc]):
                ImageDraw.Draw(im).polygon([((x - x0) * k + 2, (y1 - y) * k + 2) for x, y in poly.exterior.coords], fill=255)
                for h in poly.interiors:
                    ImageDraw.Draw(im).polygon([((x - x0) * k + 2, (y1 - y) * k + 2) for x, y in h.coords], fill=0)
            m = np.array(im) > 0
            w = 2 * float(np.median(ndi_.distance_transform_edt(m)[skeletonize(m)])) / k
            assert w == pytest.approx(st.stem, rel=0.3), (c, f, w, st.stem)


def test_a_reference_letter_is_the_designers_choice_for_letters_the_logo_lacks():
    from khatt.recipes import REFERENCE, choices
    path = _a_reference()
    st = Style(reference=path, seen="ب.I", ref_auto=False)
    plain = typeset("حب", st)
    asked = typeset("حب", Style(**{**st.__dict__, "variants": {"ح.I": REFERENCE}}))
    assert "بنية الخط المرجعي" not in [n for p in plain.placed for _, n in p.parts]
    assert "بنية الخط المرجعي" in [n for _, n in asked.placed[0].parts]
    keys = {(c["letter"], c["form"]) for c in choices(st) if any(o["name"] == REFERENCE for o in c["options"])}
    assert ("ح", "I") in keys and ("ب", "I") not in keys          # a letter the logo has keeps its own drawing


def test_the_letters_few_logos_show_come_from_the_reference_only_when_asked():
    """By default every letter is built on its recipe of the logo's parts (the typefaces are a reference
    for structure, not what is drawn); asked for (ref_auto, or the choice letter by letter), ح ص ط ع ف ق ه
    the logo lacks come from the reference; a letter the logo has, or one set to the engine's, do not."""
    from khatt.recipes import REFERENCE, choices
    path = _a_reference()
    plain = typeset("حبع", Style(reference=path, seen="ب.I"))
    assert all("بنية الخط المرجعي" not in [n for _, n in p.parts] for p in plain.placed)
    st = Style(reference=path, seen="ب.I", ref_auto=True)
    lay = typeset("حبع", st)
    names = [[n for _, n in p.parts] for p in lay.placed]
    assert "بنية الخط المرجعي" in names[0] and "بنية الخط المرجعي" in names[2]
    assert "بنية الخط المرجعي" not in names[1]                     # ب: not a letter of few logos
    chosen = {(c["letter"], c["form"]): c["chosen"] for c in choices(st)}
    assert chosen[("ح", "I")] == REFERENCE
    engine = typeset("حب", Style(**{**st.__dict__, "variants": {"ح.I": "المحرك"}}))
    assert "بنية الخط المرجعي" not in [n for p in engine.placed for _, n in p.parts]
    # the gate is the critic's: ink piled thicker than the pen, a counter closed, or a stroke thinner than
    # the hairline is refused (and the engine's letter drawn)
    from shapely.geometry import Polygon, box
    from khatt.reference import _passes
    w = st.stem
    assert not _passes(box(0, 0, 3 * w, 3 * w), st, 0)                       # a lump
    ring = box(0, 0, 4 * w, st.x_height).difference(box(w, w, 1.2 * w, st.x_height - w))
    assert not _passes(ring, st, 1)                                       # a counter shut to a slit
    assert _passes(box(0, 0, 3 * w, st.x_height).difference(box(w, w, 2 * w, st.x_height - w)), st, 1)


# ---------------------------------------------------------------- what the logo measures and the drawing now uses
def test_the_white_between_joined_letters_is_measured_apart_from_the_teeth():
    """A word drawn with joined letters set wider apart than the teeth of one letter: the engine
    measures that white apart, and lays new words out with it."""
    from khatt.pipeline import analyse
    st = Style(slant=0.0, join_white=1.6 * Style().white)
    png = to_png(typeset("سبس", st), height=300)
    an = analyse(png, texts=["سبس"], read=False)
    got = an["style"]
    assert got.join_white == pytest.approx(1.6 * got.white, rel=0.2)
    a, b = typeset("سب", Style(**{**got.__dict__, "join_white": 0.0})), typeset("سب", got)
    assert (b.bbox()[2] - b.bbox()[0]) - (a.bbox()[2] - a.bbox()[0]) == pytest.approx(got.join_white - got.white, abs=1.0)


def test_a_reh_hung_from_the_tooth_is_seen_with_a_short_tail():
    """ر hung from the top of the tooth before it, with a tail barely under the baseline (as in
    سیرنگ): the logo is read as joining from the top, and the corner of the ر is measured."""
    from khatt.pipeline import analyse
    base = Style(slant=0.0, top_join=1.0, tail_drop=0.3)
    png = to_png(typeset("بير", base), height=300)
    an = analyse(png, texts=["بير"], read=False)
    got = an["style"]
    assert got.top_join >= 0.5 and got.reh_corner > 0


def test_a_picture_cropped_through_the_letters_is_named():
    """Ink running into the edge of the picture: part of the letters may be cut off (a tail, a dot);
    the guard says so and trusts the measuring less."""
    from PIL import Image
    from khatt.pipeline import analyse
    png = to_png(typeset("بير", Style(slant=0.0)), height=300)
    im = Image.open(io.BytesIO(png)).convert("L")
    w, h = im.size
    tight = im.crop((0, 0, w, int(0.62 * h)))           # the bottom cut off through the letters
    b = io.BytesIO(); tight.save(b, "PNG")
    full = analyse(png, texts=["بير"], read=False)["health"]
    cut = analyse(b.getvalue(), texts=["بير"], read=False)["health"]
    assert not any("حافة الصورة" in w["text"] for w in full["warnings"])
    assert any("حافة الصورة" in w["text"] for w in cut["warnings"]) and cut["confidence"] < full["confidence"]


def test_the_tail_is_drawn_as_deep_as_the_logo_draws_it():
    from khatt.pipeline import analyse
    st = Style(slant=0.0, tail_drop=0.55)
    png = to_png(typeset("بر", st), height=300)
    got = analyse(png, texts=["بر"], read=False)["style"]
    depth = got.descender * got.tail_drop
    assert depth == pytest.approx(st.descender * st.tail_drop, rel=0.25)


@pytest.mark.slow
def test_measuring_the_letters_gives_the_same_rules_on_every_run():
    """The fits try parameters in order; that order must not hang on Python's hashing of strings,
    which changes from run to run (a logo once measured 53% or 80% depending on it)."""
    import os
    import subprocess
    import sys
    code = ("import warnings; warnings.filterwarnings('ignore');"
            "from pathlib import Path; from khatt.pipeline import analyse; from khatt.selfcheck import fit_letters;"
            f"an = analyse(Path({str(EXAMPLE)!r}).read_bytes(), texts=['بيدوناق'], read=False);"
            "L, k, st = an['lines'][an['src']], an['k'], an['style'];"
            "from khatt import recipes; recipes.LEARNED['dal.base'] = 1.6;"
            "print(sorted(fit_letters(L, 'بيدوناق', st, k)[0].items()))")
    root = str(Path(__file__).parent.parent)
    out = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                          env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": root}).stdout.strip().splitlines()[-1]
           for seed in ("1", "2")]
    assert out[0] == out[1] and "dal.base" in out[0]


# ---------------------------------------------------------------- letters built from the logo's parts
def test_a_part_cut_from_the_logo_builds_the_letters_the_logo_lacks():
    """A head as the logo draws it (here: a plain square) goes into every letter with a head that the
    logo lacks, in place of the engine's; a letter the logo has keeps its own; a part of another
    shape under the same name is left alone."""
    from khatt.typeset import _same_part
    st = Style(slant=0.0)
    X = st.x_height
    meem = LETTERS["م"][1](st, "I")
    head = [sh for sh in flat(meem.shapes) if getattr(sh, "name", "") == "رأس"][0]
    xr = max(x for p in head for x, _ in p)
    engine_head = [[(x - xr, y) for x, y in p] for p in head]
    square = [[(-0.9 * X, 0.0), (0.0, 0.0), (0.0, 0.8 * X), (-0.9 * X, 0.8 * X)]]
    logo = Style(**{**st.__dict__, "parts": {"رأس": {"polys": square, "engine": engine_head, "from": "م.I"}}, "seen": "م.I",
                    "part_legible": False, "part_fit": False})
    fa = typeset("فب", logo).placed[0]
    assert ("logo", "رأس") in fa.parts                                 # ف takes the logo's head
    assert ("logo", "رأس") not in typeset("مب", logo).placed[0].parts  # م is in the logo: its own drawing
    assert _same_part(engine_head, engine_head) == pytest.approx(1.0)
    odd = Style(**{**logo.__dict__, "parts": {"رأس": {"polys": square, "engine": [[(-5 * X, 3 * X), (-4 * X, 3 * X), (-4 * X, 4 * X)]], "from": "?"}}})
    assert ("logo", "رأس") not in typeset("فب", odd).placed[0].parts    # not the same part: left alone


def test_parts_are_cut_from_a_logo_drawn_with_them():
    """A logo drawn by the engine: its letters' parts are cut back out of it, with the letter they came from."""
    from khatt.pipeline import analyse
    from khatt.selfcheck import take_parts
    png = to_png(typeset("مور", Style(slant=0.0)), height=300)
    an = analyse(png, texts=["مور"], read=False)
    parts = take_parts(an["lines"][an["src"]], "مور", an["style"], an["k"])
    assert "رأس" in parts and parts["رأس"]["from"] in ("م.I", "و.F")
    assert "ذيل" in parts and parts["ذيل"]["polys"]


def test_a_part_the_logo_lacks_is_made_from_another_mirrored():
    """The logo gives only a head; the loop of ص ط, which the engine draws as a head mirrored, is made
    from it when making parts is asked for; it is off by default (the examination found it worse)."""
    st = Style(slant=0.0)
    head = [sh for sh in flat(LETTERS["ف"][1](st, "i").shapes) if getattr(sh, "name", "") == "رأس"][0]
    xr = max(x for p in head for x, _ in p)
    eng = [[(x - xr, y) for x, y in p] for p in head]
    logo = Style(**{**st.__dict__, "parts": {"رأس": {"polys": eng, "engine": eng, "from": "ف.i"}}, "seen": "ف.i",
                    "derive_parts": True, "part_legible": False})
    assert ("logo", "عروة") in typeset("صب", logo).placed[0].parts
    off = Style(**{**logo.__dict__, "derive_parts": False})
    assert ("engine", "عروة") in typeset("صب", off).placed[0].parts
    assert Style().derive_parts is False


def test_a_part_that_makes_a_letter_read_as_another_is_refused():
    """The logo's parts may change how a letter looks, not what it reads as: a part that turns a
    letter into another (judged by its skeleton among the style's own letters) is not used."""
    from khatt.legible import keeps_reading, reads_as
    from khatt.typeset import with_parts
    st = Style(slant=0.0)
    X = st.x_height
    for c, f in (("ف", "I"), ("و", "i"), ("م", "I")):
        assert reads_as(LETTERS[c][1](st, f), c, f, st) == c          # the engine's own letters read as themselves
    head = [sh for sh in flat(LETTERS["م"][1](st, "I").shapes) if getattr(sh, "name", "") == "رأس"][0]
    xr = max(x for p in head for x, _ in p)
    eng = [[(x - xr, y) for x, y in p] for p in head]
    bar = [[(-2.4 * X, 0.0), (0.0, 0.0), (0.0, 0.25 * X), (-2.4 * X, 0.25 * X)]]   # a "head" that is a flat bar
    logo = Style(**{**st.__dict__, "parts": {"رأس": {"polys": bar, "engine": eng, "from": "م.I"}}, "seen": "م.I", "part_fit": False})
    built = with_parts(LETTERS["و"][1](logo, "i"), logo)
    assert not keeps_reading(LETTERS["و"][1](logo, "i"), built, "و", "i", logo)
    assert ("logo", "رأس") not in typeset("و", logo).placed[0].parts      # refused: و keeps the engine's head


def test_the_designer_can_lengthen_raise_and_thin_a_part_of_the_logo():
    """Each part taken from the logo has three handles: length, height, thickness. Moving them redraws
    the part from its centre line; the editor offers them where the word uses the part."""
    from khatt.editor import draw
    from khatt.reference import box
    from khatt.recipes import handles
    st = Style(slant=0.0)
    head = [sh for sh in flat(LETTERS["م"][1](st, "I").shapes) if getattr(sh, "name", "") == "رأس"][0]
    xr = max(x for p in head for x, _ in p)
    eng = [[(x - xr, y) for x, y in p] for p in head]
    logo = Style(**{**st.__dict__, "parts": {"رأس": {"polys": eng, "engine": eng, "from": "م.I"}}, "seen": "م.I"})
    names = {h["name"] for h in handles(logo)}
    assert {"قطعة.رأس.طول", "قطعة.رأس.ارتفاع", "قطعة.رأس.سمك"} <= names

    def head_of(st_):
        sh = [s for s in flat(typeset("فب", st_).placed[0].glyph.shapes) if getattr(s, "name", "") == "رأس"][0]
        return box(list(sh))
    x0, y0, x1, y1 = head_of(Style(**{**logo.__dict__, "params": {"قطعة.رأس.طول": 1.0}}))   # set by hand: the designer's eye judges
    a0, b0, a1, b1 = head_of(Style(**{**logo.__dict__, "params": {"قطعة.رأس.طول": 1.4}}))
    # the centre line is 1.4 times longer; the pen is not, so the outline grows a little less
    assert 1.15 * (x1 - x0) < (a1 - a0) < 1.4 * (x1 - x0) and abs((b1 - b0) - (y1 - y0)) < 0.15 * (y1 - y0)
    out = draw(logo.to_dict(), "فب", params={"قطعة.رأس.سمك": 0.8})
    h = {x["name"]: x for x in out["handles"]}["قطعة.رأس.طول"]
    assert h["used"] and h["value"] == 1.0


def test_real_typefaces_say_how_a_part_differs_from_letter_to_letter():
    """The parts of words set in the open typefaces, cut the way a logo's are: the ratio between the
    same part in two letters comes from real designs, and a typeface can be left out of it."""
    from khatt.reference import faces, part_library, part_ratio
    if not [f for f in faces() if f.endswith("-400.woff2")]:
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    lib = part_library()
    assert len(lib) >= 10 and sum(len(v) for v in lib.values()) >= 100
    got = part_ratio("و.F", "م.I", "رأس")
    assert got is not None and 0.6 < got[0] < 1.6 and 0.6 < got[1] < 1.6
    fam = next(iter(lib)).rsplit("-", 1)[0]
    assert part_ratio("و.F", "م.I", "رأس", skip=fam) is not None


def test_letters_are_read_against_real_typefaces_too():
    from khatt.legible import reads_as_real
    from khatt.reference import faces
    if not [f for f in faces() if "vazirmatn-400" in f]:
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    st = Style(slant=0.0)
    for c, f in (("و", "i"), ("ر", "i"), ("ح", "I"), ("ك", "i")):
        assert reads_as_real(LETTERS[c][1](st, f), f, st) == c


@pytest.mark.slow
def test_real_typefaces_say_which_part_can_stand_for_which():
    """From real designs: the loop of ط ص is drawn like the head of some letters (so it may be made
    from the logo's head of those), and no part is drawn like a cup of ن ي ل (so no cup is made
    from another part)."""
    from khatt.reference import faces, substitutions
    if not [f for f in faces() if f.endswith("-400.woff2")]:
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    subs = substitutions()
    # between particular letters (the head of ه and the loop of ط), not between every head and loop
    assert max(v for (a, b, t, kind), v in subs.items() if not kind and a.endswith("|رأس") and b.endswith("|عروة")) >= 0.55
    assert all(v < 0.55 for (a, b, t, kind), v in subs.items() if b.endswith("كأس سفلي"))


# ---------------------------------------------------------------- the parts kit, and the designer's hand in it
def test_the_kit_says_where_each_part_comes_from_and_which_letters_it_builds():
    from khatt.kit import KIT, status
    st = Style(slant=0.0)
    got = {x["name"]: x for x in status(st)}
    assert set(got) == {n for n, _, _ in KIT} and all(x["source"] == "engine" for x in got.values())
    assert {"ف.I", "ق.M", "م.I", "و.i"} <= set(got["رأس"]["letters"])
    head = {"polys": [[(0, 0), (-10, 0), (-10, 10)]], "engine": [[(0, 0), (-10, 0), (-10, 10)]], "from": "و.F"}
    got = {x["name"]: x for x in status(Style(**{**st.__dict__, "parts": {"رأس": head}}))}
    assert got["رأس"]["source"] == "logo" and got["رأس"]["from"] == "و.F"


def test_a_line_is_drawn_with_the_logos_pen():
    """Thick across a vertical, thin across a horizontal, as the pen rule of a contrast logo says."""
    from khatt.kit import pen_line
    from khatt.reference import box
    st = Style(slant=0.0)
    v = box(pen_line([(0, 0), (0, 60)], st, smooth=False, ends="flat"))
    h = box(pen_line([(0, 0), (60, 0)], st, smooth=False, ends="flat"))
    assert (v[2] - v[0]) == pytest.approx(st.stem, rel=0.1) and (h[3] - h[1]) == pytest.approx(st.hair, rel=0.15)
    assert len(pen_line([(0, 0), (40, 0), (40, 40), (0, 40)], st, closed=True)) == 2      # a ring has a counter


def test_a_part_the_designer_draws_goes_into_every_letter_that_needs_it():
    from khatt.kit import designer_part, engine_part
    st = Style(slant=0.0)
    eng = engine_part(st, "عروة")
    pts = [(x + eng["xr"], y) for x, y in [(-10, 10), (-20, 40), (-50, 45), (-60, 15)]]
    part = designer_part(st, "عروة", pts, closed=True, thickness=0.8)
    assert part["source"] == "designer" and part["polys"] and part["line"]
    logo = Style(**{**st.__dict__, "parts": {"عروة": part}})
    for word in ("صب", "طب", "ضب"):
        assert ("logo", "عروة") in typeset(word, logo).placed[0].parts        # the reading gate leaves the designer's part alone


@pytest.mark.slow
def test_the_editor_offers_the_kit_suggestions_and_a_board_to_draw_on(tmp_path):
    from khatt.editor import draw, kit, kit_candidates, kit_stroke, save_edit
    from khatt.reference import faces
    st = Style()
    k = kit(st.to_dict())
    assert k["total"] >= 12 and k["complete"] == 0
    loop = {p["name"]: p for p in k["parts"]}["عروة"]
    assert loop["preview"].startswith("<svg") and loop["board"]["lines"]["x_height"] == pytest.approx(st.x_height)
    assert len(loop["board"]["line"]) >= 3 and loop["board"]["engine_part"]
    s = kit_stroke(st.to_dict(), "عروة", loop["board"]["line"], closed=True)
    d0, d1 = draw(st.to_dict(), "صبر"), draw(st.to_dict(), "صبر", parts={"عروة": s["part"]})
    assert d0["svg"] != d1["svg"]
    assert kit(st.to_dict(), parts={"عروة": s["part"]})["complete"] == 1
    rec = save_edit(st.to_dict(), "صبر", {}, {}, path=tmp_path / "e.jsonl", parts={"عروة": s["part"]})
    assert rec["parts"] == 1
    if [f for f in faces() if f.endswith("-400.woff2")]:
        c = kit_candidates(st.to_dict(), "عروة")["candidates"]
        assert c and c[0]["part"]["polys"] and c[0]["preview"].startswith("<svg")


# ---------------------------------------------------------------- the block construction (blocks.py)
def _block(parts=None, **rules):
    """A block logo of the tests' own: its basic measures given, every other rule started from them
    (blocks.start_rules) unless the test sets it. No real logo's numbers."""
    return Style(construction="block", slant=0.0, stem=26.0, hair=17.0, x_height=70.0, ascender=22.0,
                 descender=27.0, white=12.0, block=rules, block_parts=parts or {})


def _area(lay):
    from shapely.geometry import Polygon
    return sum(Polygon(sh[0], sh[1:]).area for sh in lay.shapes)


@pytest.mark.parametrize("letter", sorted(__import__("khatt.blocks", fromlist=["BUILD"]).BUILD))
def test_every_letter_builds_of_blocks_in_every_form(letter):
    from shapely.geometry import Polygon
    for word in (letter, "ب" + letter + "ب", letter + "ب", "ب" + letter):
        lay = typeset(word, _block())
        assert lay.shapes and not lay.skipped
        for shape in lay.shapes:
            poly = Polygon(shape[0], shape[1:])
            assert poly.is_valid and poly.area > 0


def test_blocks_leave_nothing_to_chance():
    """The same word is drawn the same way; a letter with a free upright leans by the logo's measured
    lean, and with no lean it stands upright."""
    a, b = typeset("بيت سلام", _block()), typeset("بيت سلام", _block())
    assert [s[0] for s in a.shapes] == [s[0] for s in b.shapes]
    upright = typeset("ا", _block(lean_alef=0.0, lean_mode="turn"))
    leaning = typeset("ا", _block(lean_alef=6.0, lean_mode="turn"))
    assert [s[0] for s in upright.shapes] != [s[0] for s in leaning.shapes]
    assert max(y for sh in upright.shapes for _, y in sh[0]) > _block().x_height      # ا rises above the teeth


def _square(x0, y0, x1, y1):
    return [[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]


def test_words_are_drawn_by_the_rules_not_cut_from_the_logo():
    """The logo's measured parts check the measures; they are never written into a word (the logo is
    its designer's work: what is taken from it is how it is built)."""
    parts = {"ا.i": {"polys": [[[[0, 0], [0, 120], [-10, 120], [-40, 0]]]]}, "ن.I": {"polys": [_square(-26, 0, 0, 70)]}}
    for w in ("ا", "نا", "بيدوناق"):
        assert [s[0] for s in typeset(w, _block(parts)).shapes] == [s[0] for s in typeset(w, _block()).shapes]


def test_the_measures_are_checked_by_rebuilding_the_logo_word():
    """blockfit.rebuild puts the measured parts back with the two spacing rules: a slit between
    joined letters (above the join), touching otherwise."""
    from khatt import blockfit
    from shapely.geometry import box
    tooth = {"polys": [_square(-26, -1, 0, 70)]}
    alef = {"polys": [_square(-24, 0, 0, 92)]}
    g = blockfit.rebuild({"ن.I": tooth, "ن.M": tooth, "ا.i": alef}, ["ن.I", "ن.M"], _block(slit=6.0))
    upper = g.intersection(box(-1e4, 30, 1e4, 1e4))
    parts = sorted(getattr(upper, "geoms", [upper]), key=lambda q: -q.bounds[2])
    assert len(parts) == 2 and abs(parts[0].bounds[0] - parts[1].bounds[2] - 6.0) < 0.5
    g = blockfit.rebuild({"ا.i": alef}, ["ا.i", "ا.i"], _block(touch=0.0))
    assert abs(g.area - 2 * 24 * 92) < 2.0 and len(getattr(g, "geoms", [g])) == 1      # they touch


def test_joined_letters_keep_the_slit_and_the_others_touch():
    from shapely.geometry import Polygon, box
    lay = typeset("ننن", _block(slit=6.0))
    bodies = [Polygon(sh[0], sh[1:]) for sh in lay.shapes]
    from shapely.ops import unary_union
    from khatt.blocks import rules
    line = rules(_block(slit=6.0))["join_line"]
    upper = unary_union(bodies).intersection(box(-1e4, line + 0.5, 1e4, 60))   # above the join: the slit
    teeth = sorted(getattr(upper, "geoms", [upper]), key=lambda q: -q.bounds[2])
    assert len(teeth) >= 2 and abs(teeth[0].distance(teeth[1]) - 6.0) < 0.8
    lay = typeset("اد", _block(touch=0.0))
    assert len(lay.shapes) == 1                                           # ا and د touch: one piece


def test_lam_alef_is_one_letter():
    """لا is the ligature every style writes (W3C alreq 4.2.6): one letter, two uprights on one base,
    isolated or joined from the right as the alef is; لأ carries its hamza."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from khatt.blocks import rules
    lay = typeset("لا", _block())
    assert [(q.char, q.form) for q in lay.placed] == [("لا", "i")]
    g = unary_union([Polygon(sh[0], sh[1:]) for sh in lay.shapes])
    assert len(getattr(g, "geoms", [g])) == 1                            # one piece: the base joins them
    top = g.intersection(box(-1e4, 60, 1e4, 1e4))
    assert len(getattr(top, "geoms", [top])) == 2                        # two uprights
    assert [(q.char, q.form) for q in typeset("سلام", _block()).placed] == [("س", "I"), ("لا", "F"), ("م", "i")]
    A = rules(_block())["A"]
    assert any(min(y for _, y in sh[0]) > A for sh in typeset("لأ", _block()).shapes)   # the hamza above


def test_kashida_lengthens_the_join_and_keeps_the_word():
    """A kashida is the join drawn longer (W3C alreq 7.2.5): the word grows by its length and every
    letter after it stays in its place (they used to fall back to the word's start)."""
    from khatt.blocks import rules
    plain, long = typeset("محمد", _block()), typeset("مـحـمـد", _block())
    xs = [q.x for q in long.placed]
    assert all(a > b for a, b in zip(xs, xs[1:]))                        # every letter left of the one before
    width = lambda lay: max(x for sh in lay.shapes for x, _ in sh[0]) - min(x for sh in lay.shapes for x, _ in sh[0])
    R = rules(_block())
    assert width(long) - width(plain) > 3 * (1.5 * R["U"] - R["into"] - R["slit"])   # each at least its length less overlap and slit


def test_joins_are_one_band_below_the_joining_line_and_a_slit_above():
    """As كيوانو joins (ي→و, ن→و): two joined letters merge in one band up to the joining line, and
    above it they stand the slit apart. The band runs from foot to foot wherever spacing put them (a
    join of one length used to break on the line, and the slit used to begin only at 26.8)."""
    from shapely.affinity import translate
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from khatt.blocks import rules
    st = _block(); R = rules(st); line = R["join_line"]
    for w in ("حلم", "نعم", "تعلم", "حديقة", "مسجد", "صبح", "سنبلة", "بستان"):
        lay = typeset(w, st)
        ink = unary_union([Polygon(sh[0], sh[1:]) for sh in lay.shapes])
        for a, b in zip(lay.placed, lay.placed[1:]):
            if a.form not in ("I", "M") or b.form not in ("M", "F") or a.char in "كک":
                continue
            ga = translate(unary_union([Polygon(sh[0], sh[1:]) for sh in a.glyph.shapes]), a.x, 0)
            gb = translate(unary_union([Polygon(sh[0], sh[1:]) for sh in b.glyph.shapes]), b.x, 0)
            lo, hi = sorted((gb.bounds[2] - 2, a.x + a.glyph.left + 2))
            band = box(lo, 3, hi, line - 3)
            assert ink.intersection(band).area > 0.97 * band.area, (w, a.char, b.char)      # no break
            up = box(-1e4, line + 1, 1e4, 1e4)
            assert ga.intersection(up).distance(gb.intersection(up)) > R["slit"] - 1.0, (w, a.char, b.char)


def test_kashida_once_per_word_at_a_chosen_join():
    """One kashida a word (alreq 7.2.5), between two joined letters: after the floor of س ص when there
    is one, otherwise before the last joined letter; a word that holds a ـ keeps only that."""
    from khatt.blocks import kashida_join, ligatures
    from khatt.typeset import shape_forms
    at = lambda w: [ligatures(shape_forms(w))[i - 1][0] + ligatures(shape_forms(w))[i][0] for i in sorted(kashida_join(ligatures(shape_forms(w))))]
    assert at("محمد") == ["مد"] and at("بستان") == ["ست"] and at("سلام") == ["سلا"]
    assert at("حمد لله") == ["مد", "له"] and at("مـحمد") == [] and at("وردة") == []
    plain, long = typeset("محمد", _block()), typeset("محمد", _block(kashida=40.0))
    width = lambda lay: max(x for sh in lay.shapes for x, _ in sh[0]) - min(x for sh in lay.shapes for x, _ in sh[0])
    assert abs(width(long) - width(plain) - 40.0) < 2.0


def test_one_rule_for_the_head():
    """The two و of كيوانو are not one head turned (one transform from the second to the first leaves
    corners 9 px and the tail 22 px apart): the first is its designer's own cut. The rule draws every
    head one way."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    lay = typeset("وو", _block())
    ps = sorted(lay.placed, key=lambda q: -q.x)
    a, b = (unary_union([Polygon(sh[0], sh[1:]) for sh in q.glyph.shapes]) for q in ps)
    assert abs(a.area - b.area) < 1.0 and a.symmetric_difference(b).area < 1.0


def test_dots_of_neighbours_do_not_run_together():
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    lay = typeset("بيت", _block())
    pieces = [Polygon(sh[0], sh[1:]) for sh in lay.shapes]
    below = [p for p in pieces if p.bounds[3] < 0]
    assert len(below) == 2                                               # the ب's dot, the ي's bar: apart
    assert below[0].distance(below[1]) > 0


def test_block_outline_is_measured_to_a_fraction_of_a_pixel():
    """Two touching blocks and a square dot, drawn on an image: the outline has straight edges, and
    the parts cut along the line between them come back in the letter's coordinates."""
    from PIL import Image, ImageDraw
    from khatt import blockfit
    im = Image.new("L", (200, 140), 255)
    d = ImageDraw.Draw(im)
    d.polygon([(120, 30), (150, 30), (150, 110), (120, 110)], fill=0)       # an upright: x 120..150, baseline 110
    d.polygon([(60, 90), (150, 90), (150, 110), (60, 110)], fill=0)         # a bar along the baseline, touching it
    d.polygon([(70, 40), (90, 40), (90, 60), (70, 60)], fill=0)             # a dot
    ink, worst = blockfit.outline(np.array(im))
    drawn = int((np.array(im) < 128).sum())
    assert worst < 0.6 and abs(ink.area - drawn) < 0.03 * drawn
    parts = blockfit.cut_parts(ink, [("نقطة", [(60, 30), (100, 30), (100, 70), (60, 70)]),
                                     ("ا.i", [(120, 0), (200, 0), (200, 140), (120, 140)]),   # along the upright's edge
                                     ("ب.F", [(0, 0), (120, 0), (120, 140), (0, 140)])], base=110.0)
    xs = [x for sh in parts["ا.i"]["polys"] for x, _ in sh[0]]
    ys = [y for sh in parts["ا.i"]["polys"] for _, y in sh[0]]
    # the upright, its body's right edge at 0, standing on the baseline, 31 wide and 81 high as drawn
    assert abs(max(xs)) < 0.6 and abs((max(xs) - min(xs)) - 31) < 1.5 and abs(max(ys) - 80.5) < 1.5 and abs(min(ys)) < 1.0
    assert set(parts) == {"نقطة", "ا.i", "ب.F"}


def test_construction_is_a_choice_and_parts_stay_the_default():
    assert Style().construction == "parts"
    assert typeset("بيت", Style()).shapes and all(getattr(s, "source", "") != "block" for s in typeset("بيت", Style()).shapes)
    assert all(s.source == "block" for s in typeset("بيت", _block()).shapes)


@pytest.mark.slow
def test_block_measuring_finds_a_moved_rule_again():
    """The measuring tool, on a logo drawn by the rules with a taller x-height, finds it."""
    from khatt import blockfit
    truth = _block(T=80.0, lean_tooth=0.0)
    lay = typeset("نون", truth)
    x0, y0, x1, y1 = lay.bbox()
    size = (int(x1 - x0) + 40, int(y1 - y0) + 40)
    origin = (20 - x0, 20 + y1)
    ink = blockfit.raster(lay, size, origin)
    start = _block(T=70.0, lean_tooth=0.0)
    before = blockfit.match("نون", start, ink, origin)
    got = blockfit.fit("نون", start, ink, origin, cycles=1, log=lambda s: None)
    assert blockfit.match("نون", got, ink, origin) > max(before, 0.9)
    assert abs(got.block["T"] - 80.0) < abs(70.0 - 80.0)


def test_choices_of_taste_change_only_their_letter():
    """What the logo has no witness for is the designer's choice, kept in the rules: each choice
    changes its own letter and nothing else."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    def g(word, **opts):
        lay = typeset(word, _block(**opts))
        return unary_union([Polygon(sh[0], sh[1:]) for sh in lay.shapes])
    assert g("هلال").symmetric_difference(g("هلال", heh_initial="stem")).area > 100
    assert g("بلال").symmetric_difference(g("بلال", heh_initial="stem")).area < 1.0
    assert g("بعد").symmetric_difference(g("بعد", ain_medial="closed")).area > 100
    assert g("عمر").symmetric_difference(g("عمر", ain_medial="closed")).area < 1.0
    # no lean: ا stands upright; as on the logo: every ا leans to the left
    from shapely.geometry import box
    up = g("ا", lean_mode="none", lean_alef=6.0)
    foot, head = up.intersection(box(-1e3, 0, 1e3, 10)), up.intersection(box(-1e3, 75, 1e3, 85))
    assert abs(foot.centroid.x - head.centroid.x) < 0.6                  # its top stands over its foot
    first, second = (g("اا", lean_mode=m, lean_alef=6.0) for m in ("logo", "turn"))
    assert first.symmetric_difference(second).area > 50


# ---------------------------------------------------------------- the anatomy: one sourced basis
def test_the_anatomy_reads_every_letter_from_unicode():
    """Skeleton and marks come from Unicode's joining groups and schematic names, not from a list."""
    from khatt import anatomy as A
    assert len(A.UCD) >= 150
    assert (A.skeleton("ت", "i"), A.marks("ت")) == ("BEH", ("2 DOTS ABOVE",))
    assert (A.skeleton("ج", "M"), A.marks("ج")) == ("HAH", ("DOT BELOW",))
    assert A.marks("أ") == ("HAMZA ABOVE",) and A.skeleton("أ", "i") == "ALEF"
    assert A.marks("ة") == ("2 DOTS ABOVE",) and A.skeleton("ة", "F") == "HEH"
    # ی has its dots only when joined to the next letter; Jawi ڽ moves its dots below when joined
    assert A.marks("ی", "i") == () and A.marks("ی", "I") == ("2 DOTS BELOW",)
    assert A.marks("ڽ", "F") == ("3 DOTS ABOVE",) and A.marks("ڽ", "M") == ("3 DOTS BELOW",)
    assert A.marks("گ") == ("SARKASH ABOVE",)                       # the gaf's second stroke is a mark
    assert (A.forms("ا"), A.forms("ء"), A.forms("ب"), A.forms("ـ")) == ("iF", "i", "iIMF", "iIMF")


def test_the_anatomy_says_which_letters_share_a_body_in_which_form():
    from khatt import anatomy as A
    assert A.same_body("ب", "ن", "I") and not A.same_body("ب", "ن", "i")
    assert A.same_body("ف", "ق", "M") and not A.same_body("ف", "ق", "F")
    assert A.same_body("ك", "گ", "I") and A.same_body("ک", "گ", "i") and not A.same_body("ك", "ک", "i")
    assert not any(A.same_body("ك", "ل", f) for f in "iIMF")
    assert A.same_body("ي", "ى", "F") and A.same_body("ي", "ب", "M") and A.same_body("ه", "ة", "i")
    assert A.representative("ق", "I") == "ف" and A.representative("ق", "i") == "ق"
    assert A.representative("ئ", "M") == "ب" and A.representative("إ", "i") == "ا"
    # every basic shape of the core set has its parts named in the established terms
    shapes = A.basic_shapes(A.ARABIC_CORE)
    assert len(shapes) == 59 and all(form in A.PARTS[sk] for sk, form in shapes)
    assert set(A.TERMS) >= {"head", "body", "eye", "tooth", "neck", "cup", "ascender", "descender"}
    assert [f for f, _ in A.FEATURES][:6] == ["ccmp", "isol", "fina", "medi", "init", "rlig"]


def test_the_anatomy_agrees_with_real_typefaces():
    """For every two letters of the core set in a form the anatomy says one body or two; in eight open
    typefaces their bodies (dots left out) coincide or not. Where several disagree it is an alternate:
    the joined tooth of Naskh typefaces raised for its neighbours, ة alone drawn apart from ه, and the
    bowl of ج خ alone or last drawn apart from ح's in three Kufi typefaces, to make room for the dot."""
    import itertools
    from khatt import anatomy as A
    from khatt.bench.fonts import load
    from khatt.bench.score import BASE_ROW, CANVAS, PX, RIGHT_COL
    from khatt.bench.truth import letter_mask, x_height
    from khatt.reference import faces, family
    fams = ['noto-naskh-arabic', 'amiri', 'vazirmatn', 'noto-kufi-arabic', 'reem-kufi', 'cairo', 'tajawal', 'kufam']
    paths = {family(f): f for f in faces() if f.endswith('-400.woff2')}
    if not all(f in paths for f in fams):
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    letters = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويىةپچژکگی'
    agree = total = 0
    off = {}
    for fam in fams:
        face = load(paths[fam]); cmap = face.tt.getBestCmap(); X = x_height(face)
        M = {}
        for c in letters:
            for f in A.forms(c):
                m = letter_mask(face, c, f, PX, CANVAS, BASE_ROW, RIGHT_COL, X) if ord(c) in cmap else None
                if m is not None and m.any():
                    M[(c, f)] = m
        for (a, fa), (b, fb) in itertools.combinations(M, 2):
            if fa == fb:
                ma, mb = M[(a, fa)], M[(b, fb)]
                same = (ma & mb).sum() / max(1, (ma | mb).sum()) > 0.92
                total += 1
                if same == A.same_body(a, b, fa):
                    agree += 1
                else:
                    off.setdefault((A.skeleton(a, fa), A.skeleton(b, fb), fa), set()).add(fam)
    assert agree / total > 0.985
    # what three or more typefaces do otherwise are the alternates the anatomy names (see ALTERNATES)
    common = {k for k, v in off.items() if len(v) >= 3}
    assert common <= {("BEH", "BEH", "I"), ("BEH", "BEH", "M"), ("HEH", "HEH", "i"), ("HAH", "HAH", "F"), ("HAH", "HAH", "i")}, off


def test_the_old_tables_are_the_anatomys_now():
    """The skeleton tables once written in five modules are one: the critic's letters are one per
    Unicode joining group, the reading counts ن ق ل alone as themselves, siblings come from Unicode."""
    from khatt import anatomy as A
    from khatt.bench.__main__ import _same
    from khatt.critic import SKELETONS
    from khatt.recipes import base, siblings
    groups = [A.skeleton(c, "i") for c in SKELETONS]
    assert len(set(groups)) == len(groups) == 18
    assert set(groups) | {"HAMZA"} == {A.skeleton(c, "i") for c in A.ARABIC_CORE}
    assert not _same("ب", "ن", "i") and not _same("ف", "ق", "i") and not _same("ك", "ل", "M")
    assert _same("ب", "ن", "I") and _same("ف", "ق", "M") and _same("ک", "ك", "I")
    assert "ا" in siblings("أ", "i") and "و" in siblings("ؤ", "F") and "ي" in siblings("ئ", "F")
    assert "ف" not in siblings("ق", "i") and "ف" in siblings("ق", "I") and set("کگ") <= set(siblings("ك", "M"))
    assert base("ن", "I") == "ب" and base("ن", "i") == "ن" and base("ة", "F") == "ه"


def test_joined_kaf_is_the_keheh_and_not_a_lam():
    """Joined ك ک گ are the keheh, as every open typeface measured draws them: an arm rising to the
    top right from a vertex on the left, and the bottom closed on the right. The stem with an arm at
    its top stays as the designer's choice (square kufi), and for a ك built on the logo's ر."""
    from khatt.critic import _norm, typeset as _t
    from khatt.letters import KEHEH, STEM_ARM
    st = Style(slant=0.0)
    for f in "IM":
        k, l = _norm(_t(("ـ{}ـ" if f == "M" else "{}ـ").format("ك"), st)), _norm(_t(("ـ{}ـ" if f == "M" else "{}ـ").format("ل"), st))
        assert (k & l).sum() / (k | l).sum() < 0.6                            # the stem-and-arm ك was 0.95
    g = LETTERS["ك"][1](st, "I")
    shape = [sh for sh in flat(g.shapes) if getattr(sh, "name", "") == "زاوية الكاف"]
    assert shape
    pts = [p for sh in shape for poly in sh for p in poly]
    top = max(y for _, y in pts)
    assert abs(top - (st.x_height + st.ascender)) < 1.0                       # on the ascender line
    left = min(x for x, _ in pts); top_x = [x for x, y in pts if y > top - 2.0]
    assert min(top_x) > left + 0.6 * st.x_height                              # the top far right of the vertex
    gaf = LETTERS["گ"][1](st, "M")
    bar = [p for sh in flat(gaf.shapes) if getattr(sh, "name", "") == "شرطة الگاف" for poly in sh for p in poly]
    assert bar and max(y for _, y in bar) <= st.x_height + st.ascender + 1e-6
    old = LETTERS["ك"][1](Style(slant=0.0, variants={"ك.I": STEM_ARM}), "I")
    assert any(getattr(sh, "name", "") == "ذراع الكاف" for sh in flat(old.shapes)) and KEHEH != STEM_ARM


def test_joined_kaf_reads_as_kaf_among_real_typefaces():
    from khatt.legible import reads_as_real
    from khatt.reference import faces
    if not [f for f in faces() if "vazirmatn-400" in f]:
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    st = Style(slant=0.0)
    got = [reads_as_real(LETTERS["ك"][1](st, f), f, st) for f in "IM"]
    assert "ل" not in got


@pytest.mark.slow
def test_the_professional_exam_measures_every_rule_and_names_what_is_missing():
    """Every rule of khatt.anatomy.RULES is measured on a style; what the engine cannot draw yet
    (digits, punctuation, harakat) is reported, not hidden."""
    from khatt import anatomy as A
    from khatt.professional import exam, report
    res = exam(Style(), fonts=False)
    assert [r["rule"] for r in res] == [k for k, _, _ in A.RULES]
    by = {r["rule"]: r for r in res}
    core = len(A.ARABIC_CORE + "ٱ")
    assert f"{core}/{core}" in by["coverage"]["measure"] and not by["coverage"]["ok"]   # digits, punctuation, harakat
    for k in ("stroke", "kashida", "lam-alef", "marks", "counters"):
        assert by[k]["ok"], by[k]
    import re
    n = re.search(r"(\d+)/(\d+) وصلة على السطر", by["joins"]["measure"])
    assert n and n.group(1) == n.group(2)                                            # every join on the line
    assert report(res).startswith("قواعد الخط الاحترافي")


# ---------------------------------------------------------------- the engine keeps no logo
@pytest.mark.slow
def test_the_engine_keeps_no_logo_and_each_logo_starts_from_itself():
    """Every logo is read afresh: the block construction starts from the logo's own measures (twice
    the logo, twice every length), no table of any logo's numbers lives in the engine, and a logo's
    result is the same whether another logo was read before it or not."""
    import subprocess, sys
    from khatt import blocks
    assert not hasattr(blocks, "KIVANO")
    small = Style(stem=20.0, hair=10.0, x_height=60.0, ascender=30.0, descender=20.0, white=10.0)
    big = Style(stem=40.0, hair=20.0, x_height=120.0, ascender=60.0, descender=40.0, white=20.0)
    a, b = blocks.start_rules(small), blocks.start_rules(big)
    for k, v in a.items():
        if isinstance(v, float) and not k.endswith(("_a", "_n", "angle", "flare_h")) and k not in ("lean_alef", "lean_tooth", "top_cut"):
            assert abs(b[k] - 2 * v) < 1e-6, k
    code = ("import sys, json; sys.path.insert(0, %r)\n"
            "from pathlib import Path\nfrom khatt.pipeline import run\nout = None\n"
            "for n in sys.argv[1:]:\n"
            "    out = run((Path(%r) / n).read_bytes(), 'بيدوناق')['style']\n"
            "print(json.dumps(out, sort_keys=True, default=str))") % (str(Path(__file__).parent.parent), str(EXAMPLE.parent))
    alone = subprocess.run([sys.executable, "-c", code, "round_square.png"], capture_output=True, text=True).stdout
    after = subprocess.run([sys.executable, "-c", code, "bidonaq.png", "round_square.png"], capture_output=True, text=True).stdout
    assert alone and alone == after


@pytest.mark.slow
def test_the_logo_chooses_its_construction():
    """A new logo is not assumed pen or blocks: its own word is drawn both ways from its own measures
    and the one that rebuilds it better is kept. A logo made of blocks (drawn here by the engine from
    rules the test knows) is found to be blocks, and its rules come back from its ink: its pen, its
    touching letters, the cut of its tops. A pen logo stays a pen."""
    truth = Style(construction="block", slant=0.0, stem=26.0, hair=17.0, x_height=70.0, ascender=22.0, descender=28.0,
                  white=12.0, block={"top_cut": 7.0, "slit": 5.0, "touch": 0.0}, block_parts={})
    logo = to_png(typeset("بيدوناق", truth), 220)
    r = run(logo, "سلام", texts=["بيدوناق"])
    c = r["construction"]
    assert c["construction"] == "block" and c["block"] >= 0.85 and c["block"] > c["pen"] + 0.2
    st = Style(**{k: v for k, v in r["style"].items() if k in Style.__dataclass_fields__})
    R = st.block
    scale = R["T"] / 70.0
    assert abs(R["U"] / scale - 26.0) < 3.0 and R["touch"] == 0.0 and abs(R["top_cut"] - 7.0) < 2.0
    pen = run(EXAMPLE.read_bytes(), "سلام", texts=["بيدوناق"])
    assert pen["construction"]["construction"] == "pen" and pen["style"]["construction"] != "block"


@pytest.mark.slow
def test_the_gallery_keeps_each_logo_with_its_own_record(tmp_path, monkeypatch):
    """A logo given to the gallery is read afresh and kept in a folder of its own: the picture, the
    record (construction and its evidence, every measure with its source), the sheet of its letters;
    anything typed is drawn in its style; and it can be removed. The gallery lives with the user's
    data, not in the code."""
    from khatt import gallery
    monkeypatch.setenv("KHATT_DATA", str(tmp_path))
    r = gallery.add(EXAMPLE.read_bytes(), "بيدوناق", "بيدوناق", exam=False)
    folder = tmp_path / "gallery" / r["id"]
    assert (folder / "logo.png").read_bytes() == EXAMPLE.read_bytes() and (folder / "specimen.png").exists()
    assert r["construction"]["construction"] == "pen" and r["measures"]["الجذع"] > 0
    assert len(r["sources"]) > 40 and all(s["source"] for s in r["sources"])
    assert [x["id"] for x in gallery.items()] == [r["id"]] and gallery.get(r["id"])["name"] == "بيدوناق"
    assert gallery.write(r["id"], "محمد كتب")["svg"].startswith("<svg")
    with pytest.raises(ValueError):
        gallery.get("../etc")
    assert gallery.remove(r["id"]) and gallery.items() == []


@pytest.mark.slow
def test_a_real_reader_judges_whole_words():
    """Tesseract's Arabic model reads the proof's words: real typefaces' words it reads right (it is
    worth something as a judge), and it reads the engine's words of a style. Skipped without it."""
    from khatt import ocr_judge as J
    from khatt.bench.fonts import load
    from khatt.bench.truth import word_image
    from khatt.reference import faces
    if not J.available():
        pytest.skip("tesseract with its Arabic model is not installed")
    fs = [f for f in faces() if "vazirmatn-400" in f]
    if not fs:
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    face = load(fs[0])
    right = sum(J.read(np.array(word_image(face, w, J.X_PX)[0].convert("L")) < 128) == w for w in ("خط", "ظل", "هلال", "فلك"))
    assert right >= 3
    r = J.judge(Style(slant=0.0), ["خط", "فلك"])
    assert set(r) >= {"words", "letters", "read", "real"} and len(r["read"]) == 2


@pytest.mark.slow
def test_the_engines_words_read_no_worse_than_now():
    """The proof's fifteen words drawn by the engine in the bench's 53 typeface styles and read by the
    real reader: no fewer read exactly than when the baseline was set (an improvement passes, a step
    back fails)."""
    from dataclasses import replace
    from khatt import ocr_judge as J
    from khatt.bench.cache import Bench
    from khatt.reference import faces
    if not J.available():
        pytest.skip("tesseract with its Arabic model is not installed")
    if not [f for f in faces() if f.endswith("-400.woff2")]:
        pytest.skip("the open typefaces are not fetched (python -m khatt.bench)")
    B = Bench()
    got = [J.read(J.word_mask(w, replace(B.styles[n], slant=0.0))) == w for n in B.names for w in J.WORDS]
    assert np.mean(got) >= BASE["ocr_words"] - BASE["tolerance"]
