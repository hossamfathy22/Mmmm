import io
import math

import pytest
from PIL import Image

from khatt import Style, analyze_image, generate, run, typeset
from khatt.letters import LETTERS
from khatt.typeset import shape_forms


def test_forms_follow_arabic_joining():
    assert shape_forms("بيدوناق") == [("ب", "I"), ("ي", "M"), ("د", "F"), ("و", "i"), ("ن", "I"), ("ا", "F"), ("ق", "i")]
    assert [f for _, f in shape_forms("محمد")] == ["I", "M", "M", "F"]


@pytest.mark.parametrize("letter", sorted(LETTERS))
@pytest.mark.parametrize("form", ["i", "I", "M", "F"])
def test_every_letter_draws_in_every_form(letter, form):
    glyph = LETTERS[letter][1](Style(), form)
    pts = [p for s in glyph.shapes for poly in s for p in poly]
    assert pts and all(math.isfinite(x) and math.isfinite(y) for x, y in pts)


def test_unknown_characters_are_reported_not_drawn():
    lay = typeset("abc نور", Style())
    assert lay.skipped == ["a", "b", "c"]
    assert [p.char for p in lay.placed] == ["ن", "و", "ر"]


def test_round_trip_measures_the_rules_it_was_drawn_with():
    st = Style(stem=30, hair=6, x_height=90, ascender=45, descender=45, slant=12, white=28, dot=33)
    png = generate("بيبيبي سنن", st)["png"]
    a = analyze_image(png)
    line = a["lines"][0]
    k = Image.open(io.BytesIO(png)).height / 1.0  # rendered height in pixels
    lay = typeset("بيبيبي سنن", st)
    x0, y0, x1, y1 = lay.bbox()
    scale = k / (y1 - y0 + 3 * st.stem)        # pixels per design unit
    assert line["slant_deg"] == pytest.approx(12, abs=1.5)
    assert line["stem"] / scale == pytest.approx(30, rel=0.12)
    assert line["x_height"] / scale == pytest.approx(90, rel=0.08)
    assert line["white_between_stems"] / scale == pytest.approx(28, rel=0.2)


def test_pipeline_produces_every_output(tmp_path):
    data = (__import__("pathlib").Path(__file__).parent.parent / "examples" / "bidonaq.png").read_bytes()
    r = run(data, "شركة", out_dir=str(tmp_path))
    assert r["result"]["svg"].startswith("<svg")
    assert r["style"]["slant"] == pytest.approx(14, abs=1.5)
    for name in ["analysis_overlay.png", "word.svg", "word.png", "construction.svg", "style.json", "measurements.json", "report.md"]:
        assert (tmp_path / name).stat().st_size > 0


def test_style_from_line_keeps_room_under_the_baseline():
    st = Style.from_line({"stem": 23, "hairline": 4.5, "x_height": 66, "ascender": 36, "descender": 10,
                          "slant_deg": 14, "white_between_stems": 22, "dots": [], "dashes": []})
    assert st.descender >= 0.45 * 66
