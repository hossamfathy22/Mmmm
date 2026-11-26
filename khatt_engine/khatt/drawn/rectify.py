"""The designer's letters brought to the measures of the logos they were drawn from (khatt.drawn.spec).

The designer gives each letter its shape; the logos give the measures every letter keeps: the x-height
heads and teeth reach, the top line tall letters reach, how deep a bowl or a tail goes, letters sitting on
the line, the white between letters that do not join. The shapes are kept: only heights move.

How a letter is moved: upright (the slant taken off, the stems vertical), each of its levels is found
on its own outline (the top of its heads and teeth, the top of its tall strokes, its bottom), and a smooth,
monotone map of heights takes each level to the spec's. The stroke at each level moves whole (a band as
thick as the stroke there is kept rigid), and the joins on the line do not move: what lies between the
levels is stretched. Widths do not change, so stems keep the logo's stem, joins the logo's hairline, and
the slant is put back as it was. Curves are cut where they span much height, so the map bends them
smoothly. Marks are not stretched: they follow their letter, and the font builder sets them by the spec.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

from fontTools.misc.bezierTools import splitCubicAtT, splitQuadraticAtT
from fontTools.pens.recordingPen import RecordingPen
from scipy.interpolate import PchipInterpolator

from . import geom as G
from .layout import Drawn, Reading
from .spec import STEM, Spec

TAILS = set("رزژ")
SITS = 0.2             # a bottom less deep than this many x-heights sits on the line (and is brought to it)


@dataclasses.dataclass
class Levels:
    """The spec in this drawing's units."""
    x: float           # x-height line
    top: float         # the top line
    bowl: float        # depth of bowls (positive)
    tail: float        # depth of ر ز ژ
    k: float           # drawing units per unit of the spec (a stem of 23)


def drawing_x_height(R: Reading) -> float:
    """The drawing's x-height line: the commonest top of its letters near the reader's x-height (teeth and
    heads; the lower level some heads keep is less common)."""
    tops = []
    for d in R.letters:
        b = G.bounds(d.body)
        if b and 0.75 * R.x_height < b[3] < 1.15 * R.x_height:
            tops.append(b[3])
    if not tops:
        return R.x_height
    tol = 0.012 * R.x_height
    return max(tops, key=lambda t: (sum(abs(t - u) < tol for u in tops), t))


def levels(R: Reading, spec: Spec) -> Levels:
    X = drawing_x_height(R)
    k = X / spec.x_height
    return Levels(x=X, top=X + spec.ascender * k, bowl=spec.bowl * k, tail=spec.tail * k, k=k)


# ------------------------------------------------------------------ a letter's own levels
def _column(p: G.Path, x: float, w: float) -> List[Tuple[float, float]]:
    r = G.intersect(p, G.rect(x - w, -G.BIG, x + w, G.BIG))
    out = []
    for c in G.contours(r):
        _, b = G.contour_info(c)
        if b:
            out.append((b[1], b[3]))
    return sorted(out)


def _envelope(up: G.Path, X: float):
    """(x, top, thickness of the stroke at the top, bottom, thickness at the bottom) along the letter."""
    b = G.bounds(up)
    if b is None:
        return []
    step = max(2.0, (b[2] - b[0]) / 60)
    out = []
    x = b[0] + step / 2
    while x < b[2]:
        runs = _column(up, x, step / 4)
        if runs:
            out.append((x, runs[-1][1], runs[-1][1] - runs[-1][0], runs[0][0], runs[0][1] - runs[0][0]))
        x += step
    return out


def _plateaus(env, X: float) -> List[float]:
    """The heights where the letter's top is flat over some width (a head, a tooth, a stem's cut top): a
    level of the letter. The flank of a stroke, where the top falls column by column, is none."""
    out, run = [], []
    for e in env + [None]:
        if e is not None and run and abs(e[1] - run[-1][1]) <= 0.015 * X:
            run.append(e)
            continue
        if run and run[-1][0] - run[0][0] >= 0.08 * X:
            out.append(max(r[1] for r in run))
        run = [e] if e is not None else []
    return out


def anchors(d: Drawn, L: Levels, band: float, tan: float) -> Tuple[List[Tuple[float, float]], List[str]]:
    """Pairs (height drawn, height wanted), each level with the stroke there kept whole, and what moved."""
    up = G.deslant(d.body, tan)
    env = _envelope(up, L.x)
    if not env:
        return [], []
    X = L.x
    pairs: List[Tuple[float, float]] = [(0.0, 0.0), (band, band)]
    notes: List[str] = []
    # tops, from the lowest level up: the heads and teeth, then the tall strokes. The stroke at a level is
    # kept whole (as thick as it is there, but never more than a quarter x-height: a stem's column is the
    # whole stem, and never into the level below)
    below = band
    tall = L.x + 0.75 * (L.top - L.x)        # a tall stroke nears the top line; the gaf's leaning body does not
    flats = _plateaus(env, X)
    for lo, hi, want, name in ((0.75 * X, 1.15 * X, L.x, "x-height"), (tall, 3 * X, L.top, "top line")):
        tops = [t for t in flats if lo < t <= hi]
        if not tops:
            continue
        top = max(tops)
        cols = [e for e in env if abs(e[1] - top) <= 0.015 * X]
        thick = max(e[2] for e in cols if e[1] > top - 0.01 * X)
        thick = min(thick, 0.25 * X, 0.45 * (top - below), 0.45 * (want - below))
        pairs += [(top - thick, want - thick), (top, want)]
        below = top
        if abs(want - top) > 0.01 * X:
            notes.append(f"{name} {top / X:.3f} → {want / X:.3f}")
    # the bottom: on the line, or as deep as the spec
    bottom = min(e[3] for e in env)
    joined = d.jl is not None or d.jr is not None
    if bottom < -SITS * X:
        want = -(L.tail if d.unit in TAILS else L.bowl)
        thick = max(e[4] for e in env if e[3] < bottom + 0.01 * X)
        thick = min(thick, 0.25 * X, 0.45 * (-bottom - band), 0.45 * (-want - band))
        pairs += [(bottom, want), (bottom + thick, want + thick)]
        if abs(want - bottom) > 0.01 * X:
            notes.append(f"depth {-bottom / X:.3f} → {-want / X:.3f}")
    elif bottom < -0.01 * X and not joined:
        # sits under the line: the bottom comes up to it, its stroke whole; the line is no fixed point then
        thick = max(e[4] for e in env if e[3] < bottom + 0.01 * X)
        thick = min(thick, 0.25 * X)
        pairs = [p for p in pairs if p[0] > bottom + thick]
        pairs += [(bottom, 0.0), (bottom + thick, thick)]
        notes.append(f"bottom {-bottom / X:.3f} under the line → on it")
    return pairs, notes


def height_map(pairs: List[Tuple[float, float]], X: float):
    """A smooth monotone map of heights through the pairs; outside them, a shift."""
    pairs = sorted(set(pairs))
    clean = []
    for s, t in pairs:
        if clean and (s - clean[-1][0] < 1e-6 or t - clean[-1][1] < 1e-6):
            continue
        clean.append((s, t))
    if len(clean) < 2:
        return lambda y: y
    lo, hi = clean[0], clean[-1]
    clean = [(lo[0] - 3 * X, lo[1] - 3 * X)] + clean + [(hi[0] + 3 * X, hi[1] + 3 * X)]
    f = PchipInterpolator([s for s, _ in clean], [t for _, t in clean])
    return lambda y: float(f(y))


# ------------------------------------------------------------------ moving outlines
def _subdivided(rec, span: float):
    """The recording with every curve that rises or falls more than `span` cut into pieces."""
    out = []
    for c in rec:
        cur = None
        nc = []
        for op, args in c:
            if op == "moveTo":
                cur = args[0]
                nc.append((op, args))
            elif op == "lineTo":
                cur = args[0]
                nc.append((op, args))
            elif op == "curveTo" and cur is not None:
                pts = (cur,) + tuple(args)
                ys = [p[1] for p in pts]
                n = max(1, min(8, math.ceil((max(ys) - min(ys)) / span)))
                segs = splitCubicAtT(*pts, *[i / n for i in range(1, n)]) if n > 1 else [pts]
                for s in segs:
                    nc.append(("curveTo", tuple(s[1:])))
                cur = args[-1]
            elif op == "qCurveTo" and cur is not None and len(args) == 2:
                pts = (cur,) + tuple(args)
                ys = [p[1] for p in pts]
                n = max(1, min(8, math.ceil((max(ys) - min(ys)) / span)))
                segs = splitQuadraticAtT(*pts, *[i / n for i in range(1, n)]) if n > 1 else [pts]
                for s in segs:
                    nc.append(("qCurveTo", tuple(s[1:])))
                cur = args[-1]
            else:
                if args:
                    cur = args[-1]
                nc.append((op, args))
        out.append(nc)
    return out


def moved(p: G.Path, f, tan: float, X: float) -> G.Path:
    """Every point taken upright, its height mapped, and slanted back."""
    rec = _subdivided(G.contours(p), 0.12 * X)

    def mv(pt):
        x, y = pt
        y2 = f(y)
        return (x + (y2 - y) * tan, y2)
    out = [[(op, tuple(mv(a) for a in args)) for op, args in c] for c in rec]
    return G.from_recorded(out)


def shifted_with(m: G.Path, f) -> G.Path:
    """A mark moved whole, as far as its letter moved at the mark's height."""
    b = G.bounds(m)
    if b is None:
        return m
    yc = (b[1] + b[3]) / 2
    return G.shift(m, 0.0, f(yc) - yc)


def _inside(m: G.Path, body: G.Path) -> bool:
    """A piece within the letter's own box (a construction piece of a colour drawing, as inside لا) is
    the letter's and moves with it point by point; a dot, a hamza or a bar above moves whole."""
    mb, bb = G.bounds(m), G.bounds(body)
    if mb is None or bb is None:
        return False
    cx, cy = (mb[0] + mb[2]) / 2, (mb[1] + mb[3]) / 2
    return bb[0] < cx < bb[2] and bb[1] < cy < bb[3] and mb[3] <= bb[3] + 1 and mb[1] >= bb[1] - 1


def rectify(R: Reading, spec: Spec) -> Tuple[Reading, Dict[str, List[str]], Levels]:
    """A new reading whose letters keep the spec's measures, and what moved in each."""
    L = levels(R, spec)
    out, report = [], {}
    for d in R.letters:
        pairs, notes = anchors(d, L, R.band, R.tan)
        if not notes:
            out.append(d)
            continue
        f = height_map(pairs, L.x)
        # the tan in `moved` slants back by the drawing's own slant: the move is made upright
        body = moved(d.body, f, R.tan, L.x)
        marks = [moved(m, f, R.tan, L.x) if _inside(m, d.body) else shifted_with(m, f) for m in d.marks]
        out.append(dataclasses.replace(d, body=body, marks=marks))
        report[f"{d.unit} {d.form} ({d.where})"] = notes
    R2 = dataclasses.replace(R, letters=out)
    # the white between letters that do not join: the spec's; each pair keeps the designer's difference
    if R.spacing:
        gap = spec.gap * L.k
        R2 = dataclasses.replace(R2, spacing=gap, kerns={k: v + gap - R.spacing for k, v in R.kerns.items()})
    R2 = dataclasses.replace(R2, spec=spec, levels=L)      # the font builder sets the marks by them
    return R2, report, L
