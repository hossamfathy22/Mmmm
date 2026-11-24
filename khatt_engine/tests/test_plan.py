"""The plan executor (hybrid step one): a plan written for a logo is carried out exactly, and its
letters are joined by the engine's join without a step. A made-up logo stands in for a real one (no
logo is kept in the engine or its tests)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import LineString, box

from khatt.plan import execute, glyph_for, level_joins, _geom
from khatt.style import Style
from khatt.typeset import typeset
from khatt.vector import fit_ring, trace

BASE = 90          # the made-up logo's baseline row


def _logo():
    """A tooth (an upright 9 wide, 42 high) standing on a join 10 high, and a ring (a head)."""
    im = Image.new("L", (320, 130), 255)
    d = ImageDraw.Draw(im)
    d.rectangle([40, BASE - 10, 140, BASE - 1], fill=0)          # the join
    d.rectangle([131, BASE - 42, 140, BASE - 1], fill=0)         # the tooth at its right end
    d.ellipse([200, BASE - 44, 244, BASE - 1], fill=0)           # the head
    d.ellipse([212, BASE - 32, 232, BASE - 12], fill=255)        # its counter
    return np.array(im).astype(float)


PLAN = {
    "frame": {"crop": [0, 0, 320, 130], "origin_x": 0, "baseline": BASE, "lift": 0.0},
    "style": {"hair": 10.0, "x_height": 42.0},
    "cuts": {"tooth": {"keep": [[60, -3, 150, 60]]}, "head": {"keep": [[195, -3, 250, 60]]}},
    "glyphs": {"ب.I": {"parts": [{"from": "tooth"}], "marks": {"above": [135.5, 42.0], "below": [120.0, 0.0]}},
               "و.i": {"parts": [{"from": "head"}]}},
}


def test_the_logo_is_drawn_by_numbers_cleanly():
    """The made-up logo traced: its straight sides come out as straight lines, its rings as curves,
    and the drawing covers the logo's ink."""
    grey = _logo()
    geom, rings = trace(grey)
    ink = grey < 128
    assert abs(geom.area - ink.sum()) / ink.sum() < 0.03
    kinds = [k for r in rings for k, _ in r]
    assert "L" in kinds and "C" in kinds


def test_a_square_stays_square():
    """Straight runs are drawn as single lines and the corners stay where they are."""
    pts = [(x, 0.0) for x in np.linspace(0, 20, 60)] + [(20.0, y) for y in np.linspace(0, 20, 60)] + \
          [(x, 20.0) for x in np.linspace(20, 0, 60)] + [(0.0, y) for y in np.linspace(20, 0, 60)]
    segs = fit_ring(pts)
    assert [k for k, _ in segs] == ["L"] * 4


def test_the_plan_is_carried_out():
    done = execute(PLAN, _logo())
    assert not done["failed"] and set(done["glyphs"]) == {"ب.I", "و.i"}
    g = done["glyphs"]["ب.I"]
    shape = _geom(g["polys"])
    assert abs(shape.bounds[2]) < 1e-6                     # its right edge at 0
    assert g["exit_edge"] < -60 and g["exit_x"] > g["exit_edge"]
    # the end of its join is the engine's join exactly: from 0 to the pen's height
    x = g["exit_edge"] + 1.0
    s = shape.intersection(LineString([(x, -5), (x, 30)]))
    assert s.bounds[1] == pytest.approx(0.0, abs=1e-6) and s.bounds[3] == pytest.approx(10.0, abs=1e-6)


def test_the_plans_letters_go_into_words_and_serve_their_siblings():
    """A letter form the plan gives is drawn so in every word, and a letter of the same body (ت for
    ب) takes it with its own marks; the engine joins it with its own join."""
    done = execute(PLAN, _logo())
    st = Style(stem=8.0, hair=10.0, x_height=42.0, ascender=26.0, descender=20.0, dot=9.0, plan={"glyphs": done["glyphs"]})
    for c in "بت":
        lay = typeset(c + "ب", st)
        first = lay.placed[0]
        assert any(getattr(sh, "source", "") == "logo" for sh in first.glyph.shapes), c
        marks = [sh for sh in first.glyph.shapes if getattr(sh, "name", "") in ("نقطة", "شرطة")]
        ys = [y for sh in marks for poly in sh for _, y in poly]
        assert ys and ((max(ys) < 0) if c == "ب" else (min(ys) > 42.0))
    assert len(typeset("بب", st).joins) == 1
    # without a plan nothing changes
    plain = Style(stem=8.0, hair=10.0, x_height=42.0, ascender=26.0, descender=20.0, dot=9.0)
    assert not any(getattr(sh, "source", "") == "logo" for p in typeset("بب", plain).placed for sh in p.glyph.shapes)


def test_a_letter_the_plan_has_no_place_for_marks_falls_back():
    """A body the plan gives without a place for this letter's marks is not used for it (the engine
    draws that letter instead of putting its dots anywhere)."""
    plan = {**PLAN, "glyphs": {"ب.I": {"parts": [{"from": "tooth"}], "marks": {"below": [120.0, 0.0]}}}}
    done = execute(plan, _logo())
    st = Style(stem=8.0, hair=10.0, x_height=42.0, ascender=26.0, descender=20.0, dot=9.0, plan={"glyphs": done["glyphs"]})
    assert glyph_for(st, "ب", "I") is not None
    assert glyph_for(st, "ت", "I") is None


def test_joins_on_the_baseline_are_levelled():
    """A join pieced together from two places of a logo, its bottom a fraction of a unit apart, is set
    level on the baseline; a tail going below it is left alone."""
    from shapely.ops import unary_union
    g = unary_union([box(-40, -0.4, -20, 10), box(-20, 0.3, 0, 10), box(-45, -20, -40, 10)])
    out = level_joins(g, 10.0)
    for x in (-30, -10):
        s = out.intersection(LineString([(x, -5), (x, 30)]))
        assert s.bounds[1] == pytest.approx(0.0, abs=1e-6)
    assert out.bounds[1] == pytest.approx(-20.0, abs=1e-6)


def test_a_join_under_an_overhang_is_filled():
    """A fillet reaching past the end of the bar, white under it, leaves no slit where the join arrives."""
    from khatt.plan import fill_join
    from shapely.ops import unary_union
    g = unary_union([box(-30, 0, -5.2, 10), box(-12, 9, 0, 16)])       # bar ends 5.2 short of the edge
    out = fill_join(g, 10.0, 0.0, "right")
    for x in (-4.0, -1.0):
        s = out.intersection(LineString([(x, -5), (x, 30)]))
        assert s.bounds[1] == pytest.approx(0.0, abs=1e-6)


def test_shallow_notches_are_filled_and_the_designers_arch_kept():
    from shapely.ops import unary_union
    notch = unary_union([box(-40, 0, -22, 10), box(-22, 2, -18, 10), box(-18, 0, 0, 10)])
    out = level_joins(notch, 10.0)
    s = out.intersection(LineString([(-20, -5), (-20, 30)]))
    assert s.bounds[1] == pytest.approx(0.0, abs=1e-6)
    arch = unary_union([box(-60, 0, -40, 12), box(-40, 5, -18, 12), box(-18, 0, 0, 12)])
    out = level_joins(arch, 10.0)
    s = out.intersection(LineString([(-30, -5), (-30, 30)]))
    assert s.bounds[1] == pytest.approx(5.0, abs=1e-6)
