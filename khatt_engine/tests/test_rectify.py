"""The designer's letters brought to the logos' measures (khatt/drawn/rectify.py): heights move to the
logos' levels; widths, the slant and what already stands on a level do not."""
import math

import pytest

pathops = pytest.importorskip("pathops")

from khatt.drawn import geom as G  # noqa: E402
from khatt.drawn.layout import Drawn, Reading  # noqa: E402
from khatt.drawn.rectify import height_map, rectify  # noqa: E402
from khatt.drawn.spec import Spec  # noqa: E402

TAN = math.tan(math.radians(14))
X = 450.0


def slanted_box(x0, x1, y0, y1):
    return G.poly([(x0 + y0 * TAN, y0), (x1 + y0 * TAN, y0), (x1 + y1 * TAN, y1), (x0 + y1 * TAN, y1)])


def letter(unit, path):
    return Drawn(unit, "isol", path, [], None, None, f"line 1, «{unit}»", False)


def reading(letters):
    return Reading(14.0, TAN, 1.0, X, 30.0, letters, G.rect(-80, -80, 80, 80), [], None, {})


def spec():
    return Spec(x_height=66.0, ascender=36.0, bowl=36.0, tail=23.0, hair=4.5, dot=25.0, dash=36.0, white=21.0,
                gap=27.0, slant=14.0, below_dot=8.5, below_dash=15.0)


def width_at(p, y):
    runs = G.runs_at(G.deslant(p, TAN), y)
    return runs[0][1] - runs[0][0]


def test_a_head_under_the_x_height_rises_to_it_keeping_its_width_and_slant():
    teeth = [letter(u, slanted_box(0, 150, 0, X)) for u in "بتث"]       # the line: three letters reach it
    head = letter("و", slanted_box(0, 300, 0, 0.873 * X))
    R2, report, L = rectify(reading(teeth + [head]), spec())
    moved = R2.letters[-1].body
    assert G.bounds(moved)[3] == pytest.approx(X, abs=1.0)
    assert width_at(moved, 0.5 * X) == pytest.approx(300, abs=2.0)       # widths do not change
    up = G.bounds(G.deslant(moved, TAN))
    assert up[2] - up[0] == pytest.approx(300, abs=3.0)                  # upright, it is the same box: slant kept
    assert all(R2.letters[i] is teeth[i] for i in range(3))              # on the line already: untouched
    assert any("x-height" in n for n in report[next(iter(report))])


def test_a_bowl_goes_as_deep_as_the_spec():
    teeth = [letter(u, slanted_box(0, 150, 0, X)) for u in "بتث"]
    bowl = letter("ق", slanted_box(0, 300, -0.4 * X, X))
    R2, _, L = rectify(reading(teeth + [bowl]), spec())
    assert G.bounds(R2.letters[-1].body)[1] == pytest.approx(-36.0 * L.k, abs=1.0)


def test_the_height_map_is_monotone_and_keeps_its_anchors():
    f = height_map([(0, 0), (30, 30), (300, 330), (393, 450)], X)
    ys = [f(y) for y in range(-100, 600, 7)]
    assert all(b >= a for a, b in zip(ys, ys[1:]))
    assert f(393) == pytest.approx(450) and f(0) == pytest.approx(0) and f(30) == pytest.approx(30)
