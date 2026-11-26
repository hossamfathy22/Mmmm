"""The examination of the engine against a designer's drawing (khatt/bench/designer.py): its two sides must
be compared upright and at one size, or every number it gives is about the comparison, not the letters."""
import pytest

pathops = pytest.importorskip("pathops")

from khatt.bench.designer import outline_mask, token_of  # noqa: E402
from khatt.drawn import geom as G  # noqa: E402


def test_the_drawing_is_compared_upright():
    tan = 0.25
    upright = G.rect(0, 0, 20, 100)
    slanted = G.poly([(0, 0), (20, 0), (20 + 100 * tan, 100), (100 * tan, 100)])
    a = outline_mask([upright], 1.0, 0.0)
    b = outline_mask([slanted], 1.0, tan)
    assert abs(a.shape[1] - b.shape[1]) <= 2           # the slant taken off, the stem is as wide as it was drawn
    assert abs(a.shape[0] - b.shape[0]) <= 1


def test_the_word_a_letter_was_drawn_in():
    assert token_of("line 2, «فاطمة»", "ف") == "فاطمة"
    assert token_of("line 1", "ب") == "ب"
