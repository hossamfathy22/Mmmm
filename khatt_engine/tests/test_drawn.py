"""The drawing-first path (khatt/drawn): Arabic joining read from the text, connectors found on the
baseline, and (when a drawing is given) a whole font built from a designer's vectors."""
import os

import pytest

from khatt.drawn import arabic as A
from khatt.drawn import geom as G
from khatt.drawn.layout import connector_zones


def test_forms_follow_unicode_joining():
    assert A.forms_of("فاطمة") == [("ف", "init"), ("ا", "fina"), ("ط", "init"), ("م", "medi"), ("ة", "fina")]
    assert A.forms_of("حـ") == [("ح", "init")]                       # a tatweel only gives the letter its join
    assert A.forms_of("ـه") == [("ه", "fina")]


def test_runs_are_the_connected_pieces_of_a_word():
    runs = A.runs_of("بيدوناق")
    assert ["".join(u for u, _ in r) for r in runs] == ["بيد", "و", "نا", "ق"]
    assert ["".join(u for u, _ in r) for r in A.runs_of("سلام")] == ["سلا", "م"]
    assert A.units("سلام")[1] == "لا"                                # lam-alef is one unit


def test_a_thin_stroke_on_the_baseline_between_two_stems_is_one_connector():
    xh, t = 450.0, 33.0
    stems = G.union([G.rect(0, 0, 120, 450), G.rect(300, 0, 420, 450), G.rect(100, 0, 320, t)])
    zones = connector_zones(stems, xh, t, 2.0)
    assert len(zones) == 1
    xs = [c[0] for c in zones[0]]
    assert 120 <= min(xs) and max(xs) <= 300


def test_a_tail_passing_below_the_baseline_is_not_a_connector():
    xh, t = 450.0, 33.0
    tail = G.union([G.rect(0, 0, 120, 450), G.rect(300, 0, 420, 450), G.rect(100, -120, 320, -120 + t)])
    assert connector_zones(tail, xh, t, 2.0) == []


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("KHATT_DRAWN_PDF"), reason="set KHATT_DRAWN_PDF to a designer's drawing")
def test_a_font_from_a_drawing(tmp_path):
    """KHATT_DRAWN_PDF: a drawing (page 1: an alphabet line and words), KHATT_DRAWN_LINES: its lines joined by |."""
    uharfbuzz = pytest.importorskip("uharfbuzz")
    from fontTools.ttLib import TTFont
    from khatt.drawn.__main__ import run

    lines = os.environ["KHATT_DRAWN_LINES"].split("|")
    r = run(os.environ["KHATT_DRAWN_PDF"], lines, str(tmp_path), 1, "Test")
    assert r["missing"] == []
    f = TTFont(r["font"])
    order = f.getGlyphOrder()
    font = uharfbuzz.Font(uharfbuzz.Face(uharfbuzz.Blob.from_file_path(r["font"])))
    buf = uharfbuzz.Buffer()
    buf.add_str("محمد")
    buf.guess_segment_properties()
    uharfbuzz.shape(font, buf, {})
    assert [order[i.codepoint] for i in buf.glyph_infos][::-1] == ["meem.init", "hah.medi", "meem.medi", "dal.fina"]


def _blob(w=50.0, h=50.0, marks=()):
    from khatt.drawn.layout import Blob
    return Blob([0], G.rect(0, 0, w, h), (0.0, 0.0, w, h), list(marks))


def test_a_text_that_does_not_match_the_drawing_is_flagged_by_its_dots():
    from khatt.drawn.layout import _dot_doubts
    dot = _blob(10, 10)
    sign = _blob(40, 20)                                  # a kaf's sign: not round
    ok = [([((0, "ب"), [("ب", "isol")])], _blob(marks=[dot])),
          ([((1, "ك"), [("ك", "isol")])], _blob(marks=[sign])),
          ([((2, "ا"), [("ا", "isol")])], _blob())]
    assert _dot_doubts(ok) == []
    shifted = [([((0, "ب"), [("ب", "isol")])], _blob()),             # beh's text on alef's shape
               ([((1, "ا"), [("ا", "isol")])], _blob(marks=[dot]))]  # alef's text on beh's shape
    assert len(_dot_doubts(shifted)) == 2


def test_persian_yeh_joined_carries_its_dots():
    from khatt.drawn.layout import _dot_doubts
    dot = _blob(10, 10)
    run = [((0, "سیر"), [("س", "init"), ("ی", "medi"), ("ر", "fina")])]
    assert _dot_doubts([(run, _blob(marks=[dot, dot]))]) == []
    assert _dot_doubts([(run, _blob())]) != []
