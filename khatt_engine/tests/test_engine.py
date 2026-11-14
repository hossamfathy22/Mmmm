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


@pytest.mark.parametrize("example", ["bidonaq.png", "round_square.png"])
def test_every_letter_passes_the_critic(example):
    """Every letter, in every form, drawn in the style measured from the example logo."""
    from khatt.critic import specimen_defects
    r = run((EXAMPLE.parent / example).read_bytes(), "بيدوناق")
    st = Style(**{k: v for k, v in r["style"].items() if k in Style.__dataclass_fields__})
    assert [d.text() for d in specimen_defects(st)] == []
    assert r["critic"]["alphabet"] == [] and r["critic"]["word"] == []


def test_every_letter_passes_the_critic_in_an_angular_style():
    from khatt.critic import specimen_defects
    st = Style(stem=11, hair=7.7, x_height=43.5, ascender=18.2, descender=18.2, slant=0, white=18.8, dot=9.2,
               top_round=0, ring_n=8, dot_n=5.5, join_in=0.2, join_out=0.23, angular=0.72, corner_r=8.3,
               tail_drop=0.0, tail_reach=1.9, contrast_power=2.3)
    assert [d.text() for d in specimen_defects(st)] == []


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
    assert abs(body.bounds[1] + st.descender) < 1.0                   # and it reaches the descender line


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
    an = analyse(_png(im))
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
    from khatt.recipes import handles
    plain, _ = _body("م", Style())
    moved, _ = _body("م", Style(params={"meem.height": 40.0}))
    assert abs(plain.bounds[3] - 0.88 * 66.0) < 0.5           # the rule: 0.88 of the x-height
    assert abs(moved.bounds[3] - 40.0) < 0.5                  # the designer's value
    h = {x["name"]: x for x in handles(Style(params={"meem.height": 40.0}))}["meem.height"]
    assert h["set_by_designer"] and h["value"] == 40.0 and h["rule_value"] == round(0.88 * 66.0, 3)


def test_a_designer_picks_another_variant_of_a_letter():
    from khatt.recipes import choices
    funnel, _ = _body("بعب", Style())
    ring, _ = _body("بعب", Style(variants={"ع.M": "حلقة بشقّ"}))
    assert funnel.symmetric_difference(ring).area > 50        # two different drawings
    cs = {(c["letter"], c["form"]): c for c in choices(Style())}
    assert cs[("ع", "M")]["chosen"] == "قمع" and {"قمع", "حلقة بشقّ"} <= {o["name"] for o in cs[("ع", "M")]["options"]}
    assert {(c["letter"], c["form"]): c for c in choices(Style(variants={"ع.M": "حلقة بشقّ"}))}[("ع", "M")]["set_by_designer"]


def test_joined_feh_stands_on_a_neck_only_where_there_is_room():
    light, _ = _body("بفب", Style())                          # hairline 4.5: white under the head
    heavy, _ = _body("بفب", Style(hair=16.0))                 # a heavy hairline would leave a crack
    from shapely.geometry import Point
    lay = typeset("بفب", Style())
    fx = [p for p in lay.placed if p.char == "ف"][0].x
    assert not light.contains(Point(fx - 12.0, 7.0))          # white between baseline and head, beside the neck
    assert max(y for _, y in light.exterior.coords) <= 66.0 + 1e-6   # and the head stays under the x-height
