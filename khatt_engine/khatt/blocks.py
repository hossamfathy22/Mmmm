"""Letters built of blocks, by rules: the construction of logos drawn with a broad marker or cut from paper.

The engine's other construction (letters.py) is the one of بيدوناق: stems and hairlines of a
contrast pen, round heads and cups. A logo like كيوانو is built another way, and is written here the
way بيدوناق was written from سیرنگ: its rules are measured on the logo, and every letter is drawn
from them, cleanly, from scratch. No letter and no piece of the logo is copied into a word: the logo
is its designer's work; what is taken from it is how it is built.

The rules (`Style.block`; KIVANO below is كيوانو's, measured): one pen; uprights that widen towards
their top, their top cut at an angle rising to the side they lean to; a tooth joined on both sides
that stands upright and flares at its foot into its joins; the head, a block with its corners cut
and a narrow slot with a pointed foot; a tail thrown down to the left; a join lighter than the pen;
a dot that is a square; two dots that are one bar; a white between joined letters (the slit), none
between the others (they touch); and the letters with a free upright (ا, ل, a tooth that begins a
word) lean, in turn to one side and the other, as the logo's do.

How كيوانو was measured, and the checks: README, "طريقة البناء بالكتل".

Coordinates as in letters.py: a letter's right edge at x = 0, it grows to the left, baseline y = 0,
y up. All lengths in pixels of the logo.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .letters import Glyph
from .library import Tagged
from .style import Style

Pt = Tuple[float, float]

# كيوانو, measured on its clean outline (the corners where its measured edge lines meet), each value then
# measured again by code on the logo's word drawn by these rules alone (blockfit.fit) and kept where
# the two agree; where they did not (the tooth's foot cut, the top cut, the head's top left corner,
# the left end of the kaf's base) the direct measure is kept: the fit was making up for something else
KIVANO: Dict[str, float] = dict(
    U=26.0,            # the pen (uprights 24-30.5; the kaf's riser 30)
    T=69.9,            # x-height: the tops of teeth and heads
    A=91.5,            # the top of ا (at its high corner)
    D=26.9,            # the depth of tails, cups and the bar of dots under the baseline
    J=17.5,            # a join along the baseline (measured 15-18.5), lighter than the pen
    join_rise=4.0,     # its bottom rises towards the left by up to this much (7-14 degrees over a short join)
    # a tooth joined on one side (ن): 24 wide at 22 above the baseline, 26.4 at its top; under 22 its foot
    # is cut back to 20 on the side it leans to (where the slit between it and its neighbour opens)
    tooth_foot=23.3, tooth_top=26.3, tooth_chamfer=4.9, tooth_knee=20.9, tooth_foot_cut=4.0,
    # a tooth joined on both sides (ي): upright, its foot flared into both joins, 0.93 of the x-height
    flare_foot=40.7, flare_top=23.0, flare_h=0.93,
    # ا: 24 at its foot, 30.5 at its top; its high top corner cut 6
    alef_foot=24.0, alef_top=30.0, alef_chamfer=6.0,
    top_cut=7.0,       # every top is cut this many degrees, rising to the side the letter leans to (6-9)
    lean_alef=5.4,     # ا leans this many degrees ...
    lean_tooth=3.0,    # ... a tooth that begins a word this many; to one side and the other in turn
    # the head (the و that stands upright): 68.5 x 70 standing on its join, top right corner cut 23 x 22,
    # bottom left 25 x 9, top left 6.5 x 4; its walls 24 (left) and 30 (right)
    head_w=67.0, head_bottom=5.0, head_tr_x=24.5, head_tr_y=22.7, head_bl_x=24.8, head_bl_y=7.8,
    head_tl_x=6.1, head_tl_y=4.3,
    # its slot: 12.6 wide, its centre 36.5 from the right, its top 57.5 falling 3.3 to the right, its
    # right side down to 26.5, a point at 18 (4 from its right side), its left side down to 33
    slot_w=12.6, slot_x=36.2, slot_top=56.0, slot_fall=3.3, slot_right_low=25.6, slot_point=17.8,
    slot_point_dx=5.5, slot_left_low=31.5,
    # the tail: from the head's foot (38.4 to 4.8 from its right edge) down at 63 degrees to the depth;
    # its tip's bottom rises 10 degrees to the left, its outer tip corner cut 7
    tail_root_l=38.8, tail_root_r=4.8, tail_angle=63.4, tail_tip_rise=7.4, tail_tip_cut=5.1,
    bar=24.8,          # a bar along the baseline (the kaf's base, the bottom of a cup)
    arm=21.0,          # an arm (the kaf's bar, a flag)
    r_low=24.0, r_high=21.0,   # the outer corners of the kaf's right end, rounded (bottom, top)
    dot=26.3, dot_chamfer=6.6, dot_turn=2.6,   # a dot: a square, its bottom left corner cut, turned
    # two dots: one bar 58.6 long, 13.4 high at its left end, 21.4 at its right, its top right corner
    # cut 5.3, its bottom flat; under a tooth its right end is 8.9 past the tooth's right edge
    dash_l=58.6, dash_h0=13.4, dash_h1=21.4, dash_chamfer=6.0, dash_dx=8.9, dash_top=6.0,
    # the kaf, by its measured angles: its base 167 long, its top flat at 24.8, its bottom falling from 3.9
    # at its left end to -3 under its right end; its right end a long curve like a D (from the base's
    # bottom 20.5 left of its right edge, rising to the right edge at 49.2: a superellipse of exponent
    # 1.6; then a nearly straight corner to the top at 68.2, 26.7 left of the right edge: exponent 1.3);
    # its riser 33.5 wide; its tip's high corner 165.7 left of the right edge at 54.2; the slash's upper
    # edge from there at 41.6 degrees, 63.1 long, its end cut (17.4 right and 11.9 down, then 5.4 down);
    # its lower edge back at 37.1 degrees to where it meets the top of the bar; the bar's top rising 6.7
    # degrees to the right end, its bottom 4.2 degrees from 17.2 right and 17.3 below the tip; the tip cut
    # 10 right and 20.8 down
    kaf_l=167.0, kaf_riser=32.2, kaf_top=68.2, kaf_base_top=24.8, kaf_base_left=3.9, kaf_base_right=-2.2,
    kaf_end_x=20.5, kaf_side_y=48.3, kaf_end_n=1.6, kaf_corner_x=26.4, kaf_corner_n=1.3,
    kaf_tip_x=165.7, kaf_tip_high=54.2, kaf_slash_a=41.1, kaf_slash_l=63.7, kaf_slash_low_a=36.4,
    kaf_bar_rise_top=6.9, kaf_bar_rise_low=4.2, kaf_bar_dx=17.6, kaf_bar_dy=17.3, kaf_tip_cut_dx=10.8, kaf_tip_cut_dy=20.8,
    # choices of taste where the logo has no witness (the designer's, as the size of و's head was in
    # بيدوناق): ه starting a word ("slots": two slots side by side; "stem": a head with an upright on
    # its left); ع in the middle ("v": a V open at the top; "closed": a small closed head); how free
    # uprights lean ("turn": to one side and the other in turn; "logo": always as on the logo, ا to
    # the left and a tooth to the right; "none")
    heh_initial="slots", ain_medial="v", lean_mode="turn", hah_end="bowl",
    slit=6.0,          # white between joined letters, above their join (5-8)
    join_line=16.5,    # joined letters merge in one band up to here, the top of the join (ي→و 16, ن→و 17)
    kashida=0.0,       # how much one join of the word is drawn longer (0: none)
    touch=0.0,         # white between letters that do not join: they touch
    space=45.0,        # between words
    into=4.0,          # a join runs this far into the letter it reaches
)


def rules(st: Style) -> Dict[str, float]:
    R = dict(KIVANO)
    R.update(st.block or {})
    return R


# ---------------------------------------------------------------- geometry
def _poly(pts):
    from shapely.geometry import Polygon
    return Polygon(pts).buffer(0)


def _union(gs):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    gs = [g for g in gs if g is not None and not g.is_empty]
    return unary_union(gs) if gs else Polygon()


def turn(pts, deg: float, c: Optional[Pt] = None):
    if not deg:
        return list(pts)
    cx = sum(p[0] for p in pts) / len(pts) if c is None else c[0]
    cy = sum(p[1] for p in pts) / len(pts) if c is None else c[1]
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts]


def round_corner(g, x: float, y: float, r: float, sx: int, sy: int):
    """Round the corner of `g` at (x, y); (sx, sy) point from the corner into the shape."""
    from shapely.geometry import Point, box
    sq = box(min(x, x + sx * r), min(y, y + sy * r), max(x, x + sx * r), max(y, y + sy * r))
    return g.difference(sq.difference(Point(x + sx * r, y + sy * r).buffer(r, 48)))


class Pen:
    """Draws one letter by the rules. `sign` is the side the letter leans to, if it leans (+1: its
    top to the left)."""

    def __init__(self, st: Style, R: Dict[str, float], sign: float = 1.0):
        self.st, self.R, self.sign = st, R, sign       # 0: upright
        self.body: list = []            # the letter's ink
        self.holes: list = []           # slots
        self.joins: list = []           # its join to the next letter
        self.marks: List[Tuple[object, Optional[Pt]]] = []   # (a mark, its other place)
        self.join_top: Optional[float] = None    # how high it merges with the letter it joins (None: the joining line)

    # -- uprights
    def upright(self, xr: float, foot_w: float, top_w: float, h: float, lean: float, chamfer: float = 0.0,
                y0: float = 0.0, cut: Optional[float] = None):
        """A stroke standing from y0 to h: its foot right edge at xr; leaning `lean` degrees (+: its top
        to the left); its top cut `cut` degrees (rising to the side it leans to), its high corner cut."""
        R = self.R
        cut = R["top_cut"] if cut is None else cut
        rise = 1.0 if lean >= 0 else -1.0                 # +1: the top rises to the left
        fc = xr - foot_w / 2
        tc = fc - math.tan(math.radians(lean)) * (h - y0)
        k = math.tan(math.radians(cut)) * top_w / 2
        tl, tr = (tc - top_w / 2, h + rise * k), (tc + top_w / 2, h - rise * k)
        pts = [(xr - foot_w, y0), (xr, y0)]
        if chamfer > 0 and rise < 0:                      # the high corner is the right one
            pts += [(tr[0] + 0.15 * chamfer, tr[1] - chamfer), (tr[0] - chamfer, tr[1] + 0.3 * chamfer), tl]
        elif chamfer > 0:
            pts += [tr, (tl[0] + chamfer, tl[1] + 0.3 * chamfer), (tl[0] - 0.15 * chamfer, tl[1] - chamfer)]
        else:
            pts += [tr, tl]
        g = _poly(pts)
        self.body.append(g)
        return g

    def tooth(self, xr: float = 0.0, h: Optional[float] = None) -> float:
        """A tooth joined on one side (as ن begins كيوانو): it leans, its top cut, its high corner cut, its
        foot cut back on the side it leans to. Its top left corner is kept (self.top_left) for its dot."""
        R = self.R
        h = R["T"] if h is None else h
        lean = R["lean_tooth"] * self.sign
        g = self.upright(xr, R["tooth_foot"], R["tooth_top"], h, lean, R["tooth_chamfer"])
        cut, k = R["tooth_foot_cut"], R["tooth_knee"]
        if cut > 0:
            from shapely.geometry import box
            pts = list(g.exterior.coords)
            if lean < 0:      # leaning right: the right foot is cut back
                wedge = _poly([(xr - cut, -1.0), (xr + 2.0, -1.0), (xr + 2.0, k), (xr + 0.1, k)])
            else:             # leaning left: the left foot
                x0 = xr - R["tooth_foot"]
                wedge = _poly([(x0 + cut, -1.0), (x0 - 2.0, -1.0), (x0 - 2.0, k), (x0 - 0.1, k)])
            cut_g = g.difference(wedge)
            self.body[-1] = cut_g
            g = cut_g
        top = [c for c in g.exterior.coords if c[1] > h - 7.0]
        self.top_left = min(top, key=lambda c: c[0]) if top else (xr - R["tooth_foot"], h)
        return xr - R["tooth_foot"]

    def medial_tooth(self, xr: float = 0.0, h: Optional[float] = None) -> float:
        """A tooth joined on both sides (as ي in كيوانو): upright, flaring at its foot into its joins."""
        R = self.R
        h = R["T"] * R["flare_h"] if h is None else h
        w0, w1 = R["flare_foot"], R["flare_top"]
        c = xr - w0 / 2 + 3.5
        k = math.tan(math.radians(R["top_cut"] - 1.0)) * w1 / 2
        self.body.append(_poly([(c - w0 / 2, 0.0), (c + w0 / 2, 0.0), (c + w1 / 2, h - k), (c - w1 / 2, h + k)]))
        return c - w1 / 2 - 0.27 * (w0 - w1)

    def alef(self, xr: float = 0.0, h: Optional[float] = None, y0: float = 0.0) -> float:
        R = self.R
        h = R["A"] if h is None else h
        self.upright(xr, R["alef_foot"], R["alef_top"], h, R["lean_alef"] * self.sign, R["alef_chamfer"], y0=y0)
        return xr - R["alef_foot"]

    # -- the head
    def head(self, xr: float = 0.0, bottom: Optional[float] = None, width: Optional[float] = None,
             top: Optional[float] = None, slots: int = 1) -> float:
        """The head: a block with its corners cut and a narrow slot with a pointed foot. Its walls keep
        the pen's measure whatever its width; its slot takes what is left."""
        R = self.R
        b = R["head_bottom"] if bottom is None else bottom
        w = R["head_w"] if width is None else width
        H = R["T"] if top is None else top
        s = (H - b) / (R["T"] - R["head_bottom"])         # heights scale with the head
        x0 = xr - w
        outer = [(x0 + R["head_tl_x"], H), (xr - R["head_tr_x"], H), (xr - 1.2, H - R["head_tr_y"] * s),
                 (xr - 0.5, b + 28.0 * s), (xr - 1.2, b + 13.5 * s), (xr - R["tail_root_r"], b),
                 (xr - R["tail_root_l"] * w / R["head_w"], b + 1.0), (x0 + 3.8, b + R["head_bl_y"] * s),
                 (x0, b + 12.0 * s), (x0, H - R["head_tl_y"])]
        self.body.append(_poly(outer))
        n = max(1, slots)
        span = w - 24.0 - 30.0                             # between the walls (left 24, right 30)
        for i in range(n):
            if n == 1:
                cx = xr - R["slot_x"] * min(1.0, w / R["head_w"]) if w <= 1.2 * R["head_w"] else xr - 30.0 - span / 2
                sw = R["slot_w"] if w <= 1.2 * R["head_w"] else span
            else:
                part = (span - (n - 1) * 24.0) / n
                cx = xr - 30.0 - part / 2 - i * (part + 24.0)
                sw = part
            top_l = b + (R["slot_top"] - R["head_bottom"]) * s
            top_r = b + (R["slot_top"] - R["slot_fall"] - R["head_bottom"]) * s
            hole = [(cx - sw / 2, top_l), (cx + sw / 2, top_r), (cx + sw / 2, b + (R["slot_right_low"] - R["head_bottom"]) * s),
                    (cx + sw / 2 - R["slot_point_dx"], b + (R["slot_point"] - R["head_bottom"]) * s),
                    (cx - sw / 2, b + (R["slot_left_low"] - R["head_bottom"]) * s)]
            self.holes.append(_poly(hole))
        return x0

    def tail(self, xr: float = 0.0, b: Optional[float] = None, angle: Optional[float] = None,
             root: Optional[Tuple[float, float]] = None):
        """The tail, thrown from a foot (root: its left and right ends, measured from xr) down to the depth."""
        R = self.R
        b = R["head_bottom"] + 0.8 if b is None else b
        a = math.radians(R["tail_angle"] if angle is None else angle)
        rl, rr = root if root is not None else (R["tail_root_l"], R["tail_root_r"])
        dy = b + R["D"]
        run = dy / math.tan(a)
        tip_r = (xr - rr - run, -R["D"])
        tip_l_x = xr - rl - run + R["tail_tip_cut"]
        rise = math.tan(math.radians(R["tail_tip_rise"])) * (tip_r[0] - tip_l_x)
        pts = [(xr - rl, b), (xr - rl - (dy - rise - R["tail_tip_cut"]) / math.tan(a), -R["D"] + rise + R["tail_tip_cut"]),
               (tip_l_x, -R["D"] + rise), tip_r, (xr - rr, b - 0.8)]
        g = _poly(pts)
        self.body.append(g)
        return g

    # -- bars
    def join(self, x0: float, x1: float):
        """A join along the baseline from x0 (left) to x1: lighter than the pen, its bottom rising to the left."""
        R = self.R
        rise = min(R["join_rise"], 0.14 * (x1 - x0))
        g = _poly([(x0, -1.0 + rise), (x1, -1.0), (x1, R["J"] - 1.0), (x0, R["J"] - 1.0)])
        self.joins.append(g)
        return g

    def bar(self, x0: float, x1: float, y0: float, thick: float, cut_l: float = 7.0, cut_r: float = 0.0,
            fall: float = 0.0, joins: bool = False):
        """A bar from x0 to x1 standing on y0 (its bottom falling `fall` to the right), its ends cut."""
        t0, t1 = math.tan(math.radians(cut_l)) * thick, math.tan(math.radians(cut_r)) * thick
        g = _poly([(x0, y0), (x1, y0 - fall), (x1 - t1, y0 + thick), (x0 + t0, y0 + thick)])
        (self.joins if joins else self.body).append(g)
        return g

    def kaf_end(self, xr: float = 0.0, flag_to: Optional[float] = None, base_to: Optional[float] = None,
                top: Optional[float] = None) -> float:
        """The right end of ک: its riser between its bar and its base, its outer side the long curve of a D
        (as measured); the base reaches left to base_to, a short flag of the bar to flag_to (د is this
        end alone, a short flag on a base)."""
        R = self.R
        top = R["kaf_top"] if top is None else top
        rw = R["kaf_riser"]
        flag_to = xr - rw - 13.0 if flag_to is None else flag_to
        base_to = xr - rw - 30.0 if base_to is None else base_to
        bl, br = R["kaf_base_left"], R["kaf_base_right"]
        ex, sy, cxr = R["kaf_end_x"], R["kaf_side_y"], R["kaf_corner_x"]
        span = max(1.0, R["kaf_l"] - ex)

        def bottom_at(x):            # the base's bottom: 3.9 at the kaf's left end, -3 under its right end
            return br + (bl - br) * min(1.0, max(0.0, (xr - ex - x) / span))
        n1, n2 = max(0.5, R["kaf_end_n"]), max(0.5, R["kaf_corner_n"])
        low = [(xr - ex + ex * abs(math.cos(t)) ** (2 / n1), sy - (sy - br) * abs(math.sin(t)) ** (2 / n1))
               for t in np.linspace(math.pi / 2, 0, 24)]
        up = [(xr - cxr + cxr * abs(math.cos(t)) ** (2 / n2), sy + (top - sy) * abs(math.sin(t)) ** (2 / n2))
              for t in np.linspace(0, math.pi / 2, 16)]
        end = _poly([(xr - rw, bottom_at(xr - rw)), *low, *up[1:], (xr - rw, top)])
        base = _poly([(base_to, bottom_at(base_to)), (xr - ex, br), (xr - rw + 2.0, R["kaf_base_top"]),
                      (base_to + math.tan(math.radians(7.0)) * (R["kaf_base_top"] - bottom_at(base_to)), R["kaf_base_top"])])
        # a short flag of the bar, its top rising 6.7 and its bottom 4.2 degrees towards the right end
        t1, t2 = math.tan(math.radians(R["kaf_bar_rise_top"])), math.tan(math.radians(R["kaf_bar_rise_low"]))
        x_in = xr - rw + 2.0
        low_in = top - 22.9
        flag = _poly([(flag_to, low_in - (x_in - flag_to) * t2), (x_in, low_in), (x_in, top),
                      (flag_to + 3.0, top - (x_in - flag_to) * t1)])
        self.body += [end, base, flag]
        return min(base_to, flag_to)

    def cup(self, xr: float, xl: float, up_top: float, right_side: bool = True):
        """A cup under the baseline: its right side a tail (or none), its bottom a bar of the pen rising a
        little to the left, its outer left corner rounded as the kaf rounds its corners, an upright end."""
        R = self.R
        d = -R["D"]
        x_in = xr
        if right_side:
            b = R["head_bottom"] + 1.0                    # the tail starts at the head's foot, no white between
            self.tail(xr, b=b)
            x_in = xr - R["tail_root_r"] - (b + R["D"]) / math.tan(math.radians(R["tail_angle"]))
        bottom = _poly([(xl, d), (x_in, d), (x_in - 6.0, d + R["bar"]), (xl, d + R["bar"])])
        bottom = round_corner(bottom, xl, d, 0.7 * R["r_low"], 1, 1)
        self.body.append(bottom)
        # the upturn rises flush from the bottom's left end
        self.upright(xl + R["U"], R["U"], R["U"], up_top, 0.0, 0.0, y0=d + R["bar"] - 4.0)

    # -- marks
    def dot(self, x_right: float, y_bottom: float, alt: Optional[Pt] = None):
        """A dot: a square, its bottom left corner cut, turned a little; bottom right at (x_right, y_bottom)."""
        R = self.R
        d, c = R["dot"], R["dot_chamfer"]
        pts = [(x_right, y_bottom), (x_right, y_bottom + d), (x_right - d, y_bottom + d), (x_right - d, y_bottom + c),
               (x_right - d + c, y_bottom)]
        self.marks.append((_poly(turn(pts, R["dot_turn"])), alt))

    def dots_bar(self, x_right: float, y_bottom: float, alt: Optional[Pt] = None):
        """Two dots as one bar: low at its left, high at its right, its top right corner cut."""
        R = self.R
        L, h0, h1, c = R["dash_l"], R["dash_h0"], R["dash_h1"], R["dash_chamfer"]
        pts = [(x_right - L, y_bottom), (x_right - 1.2, y_bottom), (x_right, y_bottom + h1 - c),
               (x_right - c, y_bottom + h1), (x_right - L, y_bottom + h0)]
        self.marks.append((_poly(pts), alt))

    def hamza(self, cx: float, y: float):
        """ء of blocks: a small square head, a stroke down to the left, a bar back to the right."""
        s = 0.8 * self.R["dot"]
        g = _union([_poly([(cx + s / 2, y + 0.55 * s), (cx + s / 2, y + 1.15 * s), (cx - 0.1 * s, y + 1.15 * s), (cx - 0.1 * s, y + 0.55 * s)]),
                    _poly([(cx + 0.05 * s, y + 0.75 * s), (cx + 0.3 * s, y + 0.55 * s), (cx - 0.35 * s, y + 0.1 * s), (cx - 0.6 * s, y + 0.3 * s)]),
                    _poly([(cx - 0.6 * s, y), (cx + 0.55 * s, y), (cx + 0.55 * s, y + 0.38 * s), (cx - 0.55 * s, y + 0.38 * s)])])
        self.marks.append((g, None))

    def mark(self, spec: Optional[str], cx: float, top: float, bottom: float = 0.0, alt: Optional[Pt] = None):
        """a1/a2/a3 over a letter whose top is `top`; b1/b2/b3 under one whose bottom is `bottom`; ha/hb a
        hamza; md madda. They keep the slit from their letter."""
        if not spec:
            return
        R = self.R
        g = R["slit"]
        low = min(bottom, 0.0) - R["dash_top"]
        if spec == "a1":
            self.dot(cx + R["dot"] / 2, top + g, alt)
        elif spec == "b1":
            self.dot(cx + R["dot"] / 2, low - R["dot"])
        elif spec in ("a2", "md"):
            self.dots_bar(cx + R["dash_l"] / 2, top + g, alt)
        elif spec == "b2":
            self.dots_bar(cx + R["dash_l"] / 2, low - R["dash_h1"])
        elif spec == "a3":
            self.dots_bar(cx + R["dash_l"] / 2, top + g)
            self.dot(cx + R["dot"] / 2, top + g + R["dash_h1"] + g)
        elif spec == "b3":
            self.dots_bar(cx + R["dash_l"] / 2, low - R["dash_h1"])
            self.dot(cx + R["dot"] / 2, low - R["dash_h1"] - g - R["dot"])
        elif spec == "ha":
            self.hamza(cx, top + g)
        elif spec == "hb":
            self.hamza(cx, low - 0.92 * R["dot"])


# ---------------------------------------------------------------- the letters
def _exit(p: Pen, form: str, left: float, foot: float) -> None:
    """A letter joined to the next draws the join, from its foot to `into` past the slit."""
    if form in ("I", "M"):
        R = p.R
        p.join(left - R["slit"] - R["into"], foot)


LEANS = set("اأإآٱل")          # letters with a free upright: they lean (and teeth that begin a word)


def _alef(p: Pen, form, mark):
    R = p.R
    left = p.alef(0.0)
    if mark in ("ha", "md"):
        p.mark(mark, -R["alef_foot"] / 2 - math.tan(math.radians(R["lean_alef"] * p.sign)) * R["A"], R["A"] + 3.0)
    elif mark == "hb":
        p.mark(mark, -R["alef_foot"] / 2, R["A"], 0.0)
    return left


def _tooth_letter(p: Pen, form, mark, letter):
    """ب ت ث ن ي and their kin joined on the left: a tooth (upright and flared when joined on both sides)."""
    R = p.R
    h = R["T"] * (R["flare_h"] if letter in "يیئ" else 1.0)
    if form == "M":
        left = p.medial_tooth(0.0, h)
        foot = -4.0
    else:
        left = p.tooth(0.0, h)
        foot = -2.0
    top = h + 2.0
    cx = left / 2
    if mark == "a1":
        # the dot stands on the tooth's top left corner, 2.8 below it (as on كيوانو's ن), or, with no
        # room there, over the tooth
        tl = getattr(p, "top_left", (left, h))
        x_r, y_b = tl[0], tl[1] - 2.8
        p.dot(x_r, y_b, alt=(cx + R["dot"] / 2 - x_r, top + R["slit"] - y_b))
    elif mark == "b2":
        p.dots_bar(R["dash_dx"], -R["dash_top"] - R["dash_h1"])     # under the tooth, as under كيوانو's ي
    elif mark:
        p.mark(mark, cx, top)
    _exit(p, form, left, foot)
    return left


def _beh_end(p: Pen, form, mark, letter):
    """ب ت ث alone or last: a boat, a short upright at each end and a bar on the baseline between."""
    R = p.R
    w = 3.4 * R["U"]
    p.upright(0.0, R["tooth_foot"], R["tooth_top"], 0.55 * R["T"], 0.0, R["tooth_chamfer"])
    p.bar(-w, -4.0, -1.0, R["J"] + 2.0, cut_l=0.0)
    p.upright(-w + R["U"] - 2.0, R["U"] - 4.0, R["U"] - 2.0, 0.42 * R["T"], 0.0, 0.0)
    p.mark(mark, -w / 2, 0.55 * R["T"] + 2.0, -1.0)
    return -w


def _noon_end(p: Pen, form, mark):
    R = p.R
    w = 2.8 * R["U"]
    p.upright(0.0, R["tooth_foot"], R["tooth_top"], 0.8 * R["T"], 0.0, R["tooth_chamfer"], y0=-R["D"] + 8.0)
    p.cup(-R["tooth_foot"] + 6.0, -w, 0.38 * R["T"], right_side=False)
    if mark:
        p.dot(-w / 2 + R["dot"] / 2, 0.8 * R["T"] + R["slit"])
    return -w


def _yeh_end(p: Pen, form, mark):
    """ي ی ى alone or last: a short upright, then a wide cup back under it; its two dots under the cup."""
    R = p.R
    w = 3.2 * R["U"]
    p.upright(0.0, R["tooth_foot"], R["tooth_top"], 0.5 * R["T"], 0.0, R["tooth_chamfer"], y0=-R["D"] + 8.0)
    p.cup(-R["tooth_foot"] + 6.0, -w, 0.3 * R["T"], right_side=False)
    if mark == "b2":
        p.dots_bar(-w / 2 + R["dash_l"] / 2, -R["D"] - R["slit"] - R["dash_h1"])
    elif mark == "ha":
        p.mark("ha", -w / 2, 0.5 * R["T"] + 2.0)
    return -w


def _reh(p: Pen, form, mark):
    """ر ز ژ: the و without its loop: the right wall of the head, half as high, and its tail."""
    R = p.R
    wall = 30.0
    p.upright(0.0, wall, wall - 2.0, 0.55 * R["T"], 0.0, R["tooth_chamfer"])
    p.tail(0.0, b=4.0, root=(wall + 4.0, R["tail_root_r"]))
    p.mark(mark, -wall / 2, 0.55 * R["T"] + 2.0)
    return -wall


def _dal(p: Pen, form, mark):
    """د ذ: the right end of ک alone: its riser with its rounded corners, a short flag, a longer base."""
    R = p.R
    left = p.kaf_end(0.0)
    p.mark(mark, -R["kaf_riser"] / 2, R["kaf_top"] + 2.0)
    return left


def _waw(p: Pen, form, mark):
    R = p.R
    left = p.head(0.0)
    p.tail(0.0)
    if mark == "ha":
        p.mark("ha", left / 2, R["T"] + 2.0)
    return left


def _head_letter(p: Pen, form, mark, letter):
    """ف ق: the head; joined, it goes on; alone or last, ف lies along the baseline, ق drops into a cup."""
    R = p.R
    left = p.head(0.0)
    body_left = left
    if form in ("i", "F"):
        if letter == "ق":
            body_left = left - 1.2 * R["U"]
            p.cup(0.0, body_left, 0.42 * R["T"])
        else:
            body_left = left - 1.6 * R["U"]
            p.bar(body_left, left + 20.0, -1.0, R["J"] + 2.0, cut_l=0.0)
            p.upright(body_left + R["U"] - 2.0, R["U"] - 4.0, R["U"] - 2.0, 0.4 * R["T"], 0.0, 0.0)
    if mark:
        p.mark(mark, left / 2, R["T"] + 2.0)
    _exit(p, form, left, -6.0)
    return body_left


def _meem(p: Pen, form, mark):
    R = p.R
    left = p.head(0.0)
    if form in ("i", "F"):
        p.upright(left + R["U"], R["U"] - 2.0, R["U"], R["head_bottom"] + 12.0, 0.0, 0.0, y0=-R["D"])
    _exit(p, form, left, -6.0)
    return left


def _heh(p: Pen, form, mark):
    """ه ة. Starting a word: a wider head with two slots (as square kufic draws it); in the middle: two
    heads one over the other, the join between them; last: a head with a short arm up to the left;
    alone (and ة): the head."""
    R = p.R
    if form == "I" and R["heh_initial"] == "stem":
        left = p.head(0.0)
        p.upright(left + R["U"] - 2.0, R["U"] - 4.0, R["U"] - 2.0, 1.2 * R["T"], 0.0, R["alef_chamfer"], y0=R["T"] - 14.0)
    elif form == "I":
        left = p.head(0.0, width=1.4 * R["head_w"], slots=2)
    elif form == "M":
        left = p.head(0.0, bottom=R["T"] * 0.12, width=0.8 * R["head_w"], top=R["T"])
        p.head(0.0, bottom=-R["T"] * 0.55, width=0.8 * R["head_w"], top=R["T"] * 0.1)
    elif form == "F" and mark is None:
        left = p.head(0.0)
        p.upright(left + R["U"] - 4.0, R["U"] - 6.0, R["U"] - 4.0, R["T"] + 18.0, R["lean_alef"], R["alef_chamfer"], y0=R["T"] - 12.0)
    else:
        left = p.head(0.0)
    if mark:
        p.mark(mark, left / 2, R["T"] + 2.0)
    _exit(p, form, left, -6.0)
    return left


def _sad(p: Pen, form, mark, letter):
    """ص ض ط ظ: a long head lying on the baseline; ص ends in a small tooth, ط carries the upright of ا."""
    R = p.R
    w = 1.7 * R["head_w"]
    if letter in "صض":
        p.upright(0.0, R["tooth_foot"], R["tooth_top"], 0.55 * R["T"], 0.0, R["tooth_chamfer"])
        left = p.head(-R["tooth_foot"] + 4.0, width=w, top=0.85 * R["T"])
    else:
        left = p.head(0.0, width=w, top=0.85 * R["T"])
        p.alef(left + 1.3 * R["U"], y0=R["head_bottom"])
    body_left = left
    if form in ("i", "F") and letter in "صض":
        body_left = left - 1.4 * R["U"]
        p.cup(left + 20.0, body_left, 0.4 * R["T"], right_side=False)
    if mark:
        p.mark(mark, left / 2 if letter in "صض" else left * 0.6, 0.85 * R["T"] + 2.0)
    _exit(p, form, left, -6.0)
    return body_left


def _seen(p: Pen, form, mark):
    """س ش: three teeth joined on both sides, on one join; alone or last, a cup."""
    R = p.R
    x = 0.0
    lefts = []
    for i in range(3):
        left = p.medial_tooth(x, R["T"] * (0.8 if i else 0.88))
        lefts.append(left)
        x = left - R["slit"] + 4.0
    left = lefts[-1]
    p.bar(left - 2.0, -4.0, -1.0, R["J"], cut_l=0.0, joins=True)
    body_left = left
    if form in ("i", "F"):
        body_left = left - 1.6 * R["U"]
        p.cup(left + 12.0, body_left, 0.4 * R["T"], right_side=False)
    if mark:
        p.mark(mark, (lefts[0] + lefts[1]) / 2 + 6.0, R["T"] * 0.88 + 2.0)
    _exit(p, form, left, -6.0)
    return body_left


def _stroke(top: Pt, foot: Pt, w: float):
    """A stroke of the pen falling from `top` to `foot`, `w` wide across."""
    (x0, y0), (x1, y1) = top, foot
    return _poly([(x0 - w / 2, y0), (x0 + w / 2, y0), (x1 + w / 2, y1), (x1 - w / 2, y1)])


def _leg(p: Pen, x_left: float):
    """The tail of the logo's و falling from the line at a letter's left end: in a heavy face the bowl
    under the line becomes a leg (as the block kufic Blaka draws ح and ع), there is no room for a
    counter under so shallow a line."""
    p.tail(x_left + p.R["tail_root_l"], b=2.0)


def _jeem(p: Pen, form, mark):
    """ج ح خ as the reference (Kufam) builds them: one stroke falling from the top left to the line at
    the right, and the base running on to the left. Alone or last it keeps the bowl thrown back to the
    left (hah_end="bowl"): with the و's tail falling from the base instead (hah_end="leg", Blaka's way)
    the stroke and the leg read as ر (صبح as صبر)."""
    R = p.R
    if form in ("i", "F") and R.get("hah_end", "bowl") == "bowl":
        return _jeem_bowl(p, form, mark)
    U, T = R["U"], R["T"]
    w = 2.6 * U
    top, foot = (-w + 0.75 * U, T), (-0.6 * U, 0.0)
    p.body.append(_stroke(top, foot, 1.15 * U))
    p.body.append(_poly([(-w, -1.0), (foot[0] + 0.55 * U, -1.0), (foot[0] + 0.55 * U, R["bar"] - 4.0), (-w, R["bar"] - 4.0)]))
    if form in ("i", "F"):
        _leg(p, -w)
    if mark == "b1":
        p.dot(-0.55 * w + R["dot"] / 2, (-R["dash_top"] if form in ("I", "M") else -R["slit"]) - R["dot"])
    elif mark == "a1":
        p.mark("a1", -w + 0.6 * U, T)
    elif mark == "b3":
        p.mark("b3", -0.55 * w, 0.0, 0.0)
    _exit(p, form, -w, -4.0)
    return -w


def _jeem_bowl(p: Pen, form, mark):
    """ج ح خ (the choice hah_end="bowl"): a hat on the left at the top, and from its right end a stroke falling to the right, to the
    join (joined) or, alone or last, on into a bowl thrown back under the baseline to the left."""
    R = p.R
    U, T = R["U"], R["T"]
    w = 2.6 * U
    p.bar(-w, -1.0 * U, T - R["arm"], R["arm"], cut_l=7.0, cut_r=0.0)
    top_x = -1.55 * U
    if form in ("I", "M"):
        foot = -0.55 * U
        p.body.append(_poly([(top_x - 0.55 * U, T - 2.0), (top_x + 0.6 * U, T - 2.0), (foot + 0.6 * U, 0.0), (foot - 0.55 * U, 0.0)]))
        below = 0.0
    else:
        knee = (-0.5 * U, 0.3 * T)
        p.body.append(_poly([(top_x - 0.55 * U, T - 2.0), (top_x + 0.6 * U, T - 2.0), (knee[0] + 0.6 * U, knee[1]), (knee[0] - 0.55 * U, knee[1] - 0.2 * U)]))
        p.cup(knee[0] + 0.6 * U, -w, 0.12 * T, right_side=False)
        # the back, down to the left: its outer edge, the bottom, its inner edge (in order round it)
        p.body.append(_poly([(knee[0] + 0.6 * U, knee[1]), (-w + 1.4 * U, -R["D"] + 6.0),
                             (-w + 0.4 * U, -R["D"] + R["bar"]), (knee[0] - 0.55 * U, knee[1] - 0.2 * U)]))
        below = -R["D"]
    if mark == "b1":
        if form in ("I", "M"):
            p.dot(-1.15 * U + R["dot"] / 2, -R["dash_top"] - R["dot"])
        else:
            p.dot(-1.1 * U + R["dot"] / 2, below - R["slit"] - R["dot"])
    elif mark == "a1":
        p.mark("a1", -w + 1.0 * U, T)
    elif mark == "b3":
        p.mark("b3", -1.1 * U, 0.0, below)
    _exit(p, form, -w, -4.0)
    return -w


def _ain(p: Pen, form, mark):
    """ع غ. Starting a word: the reference's open head ('c', Kufam) drawn in blocks as a chevron open to
    the right, standing on the line (the sister of the medial V; a slotted block read as a head, م);
    alone or last, the same with the و's tail falling from its left end. In the middle: the V or the
    closed head (the choice ain_medial)."""
    R = p.R
    if form != "M":
        U, T = R["U"], R["T"]
        w, top = 2.2 * U, 0.95 * T
        px, py = -w + 0.2 * U, 0.45 * top
        p.body.append(_poly([(-0.15 * U, top), (-1.25 * U, top), (px - 0.1 * U, py), (-w + 0.9 * U, -1.0), (-0.1 * U, -1.0),
                             (-0.1 * U, R["bar"] - 4.0), (px + 1.1 * U, py)]))
        if form in ("i", "F"):
            _leg(p, -w)
        if mark:
            p.mark(mark, -0.6 * U, top + 2.0)
        _exit(p, form, -w, -4.0)
        return -w
    return _ain_middle(p, form, mark)


def _ain_middle(p: Pen, form, mark):
    """ع غ. Starting a word: a ⊏ open to the right, as square kufic draws it: an arm at the top, a back
    on the left, the join along the baseline; alone, the same ⊏ falling into the bowl of ح. In the
    middle: a V open at the top, standing on the join; last: the V falling into the bowl of ح."""
    R = p.R
    U, T = R["U"], R["T"]
    if form in ("i", "I"):
        w = 2.0 * U
        top = 0.95 * T
        back_x = -w + U                                   # the back's right edge
        p.bar(back_x - 2.0, -0.12 * U, top - R["arm"], R["arm"], cut_l=0.0, cut_r=7.0)
        if form == "I":
            p.upright(back_x, U, U, top, 0.0, 0.0)
            p.join(-w - R["slit"] - R["into"], -0.2 * U)   # the bottom of the ⊏ and the join, one bar
            left = -w
        else:
            knee = (back_x - U / 2, 0.25 * T)
            p.upright(back_x, U, U, top, 0.0, 0.0, y0=knee[1])
            d = R["D"]
            p.body.append(_poly([(back_x, knee[1] + 2.0), (-w + 1.2 * U - U, -d + 6.0), (-w + 0.2 * U - U, -d + R["bar"]),
                                 (back_x - U, knee[1] + 2.0)]))
            p.cup(-0.3 * U, -w - 0.8 * U, 0.1 * T, right_side=False)
            left = -w - 0.8 * U
    elif form == "M" and R["ain_medial"] == "closed":
        # a small closed head, wider at its top, its slot pointed at its foot (the logo's slot)
        w = 2.1 * U
        top = 0.9 * T
        foot_x = -w / 2
        p.body.append(_poly([(-w, top), (0.0, top), (-0.55 * U, 0.0), (-w + 0.55 * U, 0.0)]))
        p.holes.append(_poly([(foot_x - R["slot_w"] / 2, top - 0.22 * T), (foot_x + R["slot_w"] / 2, top - 0.22 * T - 2.0),
                              (foot_x + 1.0, 0.32 * T), (foot_x - R["slot_w"] / 2, 0.45 * T)]))
        left = -w
    else:
        w = 2.4 * U
        top = 0.95 * T
        foot_x = -w / 2
        p.body.append(_poly([(foot_x - 0.5 * U, 0.0), (foot_x + 0.45 * U, 0.0), (-0.1 * U, top), (-0.95 * U, top - 2.0)]))
        p.body.append(_poly([(foot_x - 0.45 * U, 0.0), (foot_x + 0.5 * U, 0.0), (-w + 0.95 * U, top - 2.0), (-w + 0.1 * U, top)]))
        left = -w
        if form == "F":
            left = -w - 0.4 * U
            p.cup(foot_x + 0.6 * U, left, 0.1 * T, right_side=False)
            p.body.append(_poly([(foot_x - 0.5 * U, 2.0), (foot_x + 0.5 * U, 2.0), (left + 1.5 * U, -R["D"] + 8.0),
                                 (left + 0.6 * U, -R["D"] + R["bar"])]))
    if mark:
        p.mark(mark, -w / 2, top + 2.0)
    if form == "M":
        _exit(p, form, -w, -4.0)
    return left


def _kaf(p: Pen, form, mark, letter):
    """ک as كيوانو draws it, by its measured angles: a long base, its right end with its riser and its D
    curve, the bar back to a tip, and from the tip the slash up to the right, its lower edge less steep
    than its upper, meeting the bar's top. Alone or last, a short upright stands at the base's left end."""
    R = p.R
    L = R["kaf_l"]
    base_left = -L - R["into"] if form in ("I", "M") else -L + 0.4 * R["U"]
    p.kaf_end(0.0, flag_to=-R["kaf_riser"] - 2.0, base_to=base_left)
    if form in ("I", "M"):
        p.join_top = R["kaf_base_top"] + 2.0      # its long base is the join the next letter stands on
    tip = (-R["kaf_tip_x"], R["kaf_tip_high"])
    a, a2 = math.radians(R["kaf_slash_a"]), math.radians(R["kaf_slash_low_a"])
    up_end = (tip[0] + R["kaf_slash_l"] * math.cos(a), tip[1] + R["kaf_slash_l"] * math.sin(a))
    e1 = (up_end[0] + 17.4, up_end[1] - 11.9)
    e2 = (e1[0] + 0.5, e1[1] - 5.4)
    t1, t2 = math.tan(math.radians(R["kaf_bar_rise_top"])), math.tan(math.radians(R["kaf_bar_rise_low"]))
    top_at_end = R["kaf_top"]
    xr_bar = -R["kaf_corner_x"]
    # where the slash's lower edge meets the bar's top: the bar's top runs back from the right end at 6.7
    # degrees, the slash's lower edge down from its end at 37.1 degrees
    # top: y = top_at_end - (xr_bar - x) t1 ; slash: y = e2.y - (e2.x - x) tan(a2)
    ta2 = math.tan(a2)
    xj = (top_at_end - xr_bar * t1 - e2[1] + e2[0] * ta2) / (ta2 - t1)
    yj = top_at_end - (xr_bar - xj) * t1
    b0 = (tip[0] + R["kaf_bar_dx"], tip[1] - R["kaf_bar_dy"])
    x_in = -R["kaf_riser"] + 6.0
    arm = [tip, up_end, e1, e2, (xj, yj), (x_in, top_at_end - (xr_bar - x_in) * t1),
           (x_in, b0[1] + (x_in - b0[0]) * t2), b0, (tip[0] + R["kaf_tip_cut_dx"], tip[1] - R["kaf_tip_cut_dy"])]
    p.body.append(_poly(arm))
    if letter == "گ":
        off = 0.9 * R["U"]
        p.body.append(_poly([(tip[0] + 22.0, tip[1] + off), (up_end[0] + 22.0, up_end[1] + off), (e2[0] + 22.0, e2[1] + off),
                             (xj + 22.0, yj + off)]))
    if form in ("i", "F"):
        p.upright(base_left + R["U"], R["U"] - 4.0, R["U"] - 2.0, 0.4 * R["T"], 0.0, 0.0)
    return -L if form in ("I", "M") else base_left


def _lam(p: Pen, form, mark):
    R = p.R
    if form in ("i", "F"):
        left = p.alef(0.0, y0=-R["D"] + 6.0)
        body_left = left - 1.6 * R["U"]
        p.cup(left + 6.0, body_left, 0.38 * R["T"], right_side=False)
        return body_left
    left = p.alef(0.0)
    _exit(p, form, left, -4.0)
    return left


def _hamza(p: Pen, form, mark):
    R = p.R
    p.hamza(-0.5 * R["dot"], 0.12 * R["T"])
    return -R["dot"]


def _tatweel(p: Pen, form, mark):
    """The kashida: the join itself drawn longer, the same bar at the same height and thickness as
    every join (the connecting stroke of the script, W3C alreq 7.2.5), reaching into the letter before.
    It is the letter's body too, so the word keeps its letters apart by its length."""
    R = p.R
    g = p.join(-1.5 * R["U"], R["into"])
    p.body.append(g)
    return -1.5 * R["U"]


LAM_ALEF = {"ا": None, "أ": "ha", "إ": "hb", "آ": "md"}


def _lam_alef(p: Pen, form, mark):
    """لا, the ligature every style of the script writes for lam followed by alef (W3C alreq 4.2.6:
    "ignoring this ligature ... leads to incorrectly rendered text"). The lam and the alef stand on one
    base as one letter: the lam on the right, the alef on the left, leaning away from each other as
    the logo leans its uprights (upright when it does not lean), their feet joined by a base as thick
    as a join. Only its right side joins (to the letter before), as the alef does."""
    R = p.R
    sign = abs(p.sign)
    p.sign = -sign                                     # the lam leans to the right
    left_l = p.alef(0.0)
    xr = left_l - R["slit"] - 0.35 * R["U"]
    p.sign = sign                                      # the alef to the left
    left_a = p.alef(xr)
    p.bar(left_a + 2.0, -2.0, -1.0, R["J"] + 2.0, cut_l=0.0)
    if mark in ("ha", "md"):
        p.mark(mark, xr - R["alef_foot"] / 2 - math.tan(math.radians(R["lean_alef"] * sign)) * R["A"], R["A"] + 3.0)
    elif mark == "hb":
        p.mark(mark, xr - R["alef_foot"] / 2, R["A"], 0.0)
    return left_a


FLOORS = set("سشصض")


def kashida_join(pairs) -> set:
    """Where a word takes its kashida: once in the word (W3C alreq 7.2.5: typographers keep it to
    chosen joins and avoid several in one word), only between two joined letters (never inside لا,
    which is one letter). By
    default after the floor of س ش ص ض when the word has one (their long base invites it), otherwise
    before the word's last joined letter. Returns the index of the letter the longer join reaches."""
    out, start = set(), 0
    for end in [i for i, (c, f) in enumerate(pairs) if f == " "] + [len(pairs)]:
        word = range(start, end)
        joins = [i for i in word if i > start and pairs[i][1] in ("M", "F") and pairs[i - 1][1] in ("I", "M")
                 and "ـ" not in (pairs[i][0], pairs[i - 1][0])]
        if joins and not any(pairs[i][0] == "ـ" for i in word):    # a word that already holds one keeps it
            after_floor = [i for i in joins if pairs[i - 1][0] in FLOORS]
            out.add(after_floor[-1] if after_floor else joins[-1])
        start = end + 1
    return out


def ligatures(pairs):
    """Lam followed by an alef that joins it becomes the one letter لا (لأ لإ لآ): isolated when the lam
    began its group, final when a letter before joined it."""
    out, i = [], 0
    while i < len(pairs):
        c, form = pairs[i]
        if c == "ل" and form in ("I", "M") and i + 1 < len(pairs) and pairs[i + 1][0] in LAM_ALEF and pairs[i + 1][1] == "F":
            out.append(("ل" + pairs[i + 1][0], "i" if form == "I" else "F"))
            i += 2
            continue
        out.append((c, form))
        i += 1
    return out


BUILD = {
    "ا": (_alef, None), "أ": (_alef, "ha"), "إ": (_alef, "hb"), "آ": (_alef, "md"), "ٱ": (_alef, None),
    "ب": ("beh", "b1"), "ت": ("beh", "a2"), "ث": ("beh", "a3"), "پ": ("beh", "b3"),
    "ن": ("noon", "a1"), "ي": ("yeh", "b2"), "ی": ("yeh", "b2*"), "ى": ("yeh", None), "ئ": ("yeh", "ha"),
    "ج": (_jeem, "b1"), "ح": (_jeem, None), "خ": (_jeem, "a1"), "چ": (_jeem, "b3"),
    "د": (_dal, None), "ذ": (_dal, "a1"), "ر": (_reh, None), "ز": (_reh, "a1"), "ژ": (_reh, "a3"),
    "س": (_seen, None), "ش": (_seen, "a3"), "ص": ("sad", None), "ض": ("sad", "a1"),
    "ط": ("sad", None), "ظ": ("sad", "a1"), "ع": (_ain, None), "غ": (_ain, "a1"),
    "ف": ("head", "a1"), "ڤ": ("head", "a3"), "ق": ("head", "a2"),
    "ك": ("kaf", None), "ک": ("kaf", None), "گ": ("kaf", None),
    "ل": (_lam, None), "م": (_meem, None), "ه": (_heh, None), "ة": (_heh, "a2"),
    "و": (_waw, None), "ؤ": (_waw, "ha"), "ء": (_hamza, None), "ـ": (_tatweel, None),
}


def is_lam_alef(c: str) -> bool:
    return len(c) == 2 and c[0] == "ل" and c[1] in LAM_ALEF


def leans(c: str, form: str) -> bool:
    """A letter with a free upright leans: ا ل (and لا), and a tooth that begins a word."""
    if is_lam_alef(c):
        return True
    how = BUILD.get(c, (None, None))[0]
    return c in LEANS or (how in ("beh", "noon", "yeh") and form == "I")


def build(c: str, form: str, p: Pen) -> float:
    """Draw letter `c` in `form` with pen `p`; returns the body's left edge."""
    if is_lam_alef(c):
        return _lam_alef(p, form, LAM_ALEF[c[1]])
    how, mark = BUILD[c]
    if how == "beh":
        return _tooth_letter(p, form, mark, c) if form in ("I", "M") else _beh_end(p, form, mark, c)
    if how == "noon":
        return _tooth_letter(p, form, mark, c) if form in ("I", "M") else _noon_end(p, form, mark)
    if how == "yeh":
        m = "b2" if mark == "b2*" and form in ("I", "M") else (None if mark == "b2*" else mark)
        return _tooth_letter(p, form, m, c) if form in ("I", "M") else _yeh_end(p, form, m)
    if how == "sad":
        return _sad(p, form, mark, c)
    if how == "head":
        return _head_letter(p, form, mark, c)
    if how == "kaf":
        return _kaf(p, form, mark, c)
    return how(p, form, mark)


# ---------------------------------------------------------------- the word
def _shapes_of(geom) -> List[List[List[Pt]]]:
    out = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        out.append([list(poly.exterior.coords)[:-1]] + [list(h.coords)[:-1] for h in poly.interiors])
    return out


def gap_shift(prev, body, white: float) -> float:
    """How far left of 0 `body` (its right edge at 0) must go to stand `white` clear of `prev` (placed)."""
    from shapely.affinity import translate
    if prev.is_empty or body.is_empty:
        return prev.bounds[0] if not prev.is_empty else 0.0
    hi = prev.bounds[2] - body.bounds[0] + 1.0
    lo = prev.bounds[0] - body.bounds[2] - 4 * white - 50.0

    def apart(dx):
        b = translate(body, dx, 0)
        return prev.distance(b) if not prev.intersects(b) else -prev.intersection(b).area
    for _ in range(50):             # the rightmost place at which they are `white` apart
        mid = (lo + hi) / 2
        if apart(mid) >= white - 1e-3:
            lo = mid
        else:
            hi = mid
    return lo


def above(geom, y: float):
    from shapely.geometry import box
    return geom.intersection(box(-1e5, y, 1e5, 1e5))


def clear_marks(bodies, groups, R: Dict[str, float]):
    """Set each mark clear of the other letters and of the marks already set, as a calligrapher sets
    the dots of neighbours. `groups`: (owner, mark, other place). A mark may touch its own letter (the dot
    of كيوانو's ن stands on the tooth's shoulder) and keeps the logo's slit from the others (كيوانو: 6.3,
    7.0, 10.2). One that would come nearer goes, in this order: to its other place (the dot on a
    shoulder where the neighbour leaves no room goes over its tooth); a little sideways (up to half its
    width); away from the baseline."""
    from shapely.affinity import translate
    from shapely.ops import unary_union
    clear = max(1.0, R["slit"] - 1.0)
    shapes = [b.buffer(clear) for b in bodies]
    placed = None
    out = []
    for i, g0, alt in groups:
        if g0.is_empty:
            continue
        others = [sh for j, sh in enumerate(shapes) if j != i and not sh.is_empty]
        if placed is not None:
            others.append(placed)
        taken = unary_union(others) if others else None

        def free(dx, dy):
            return taken is None or not translate(g0, dx, dy).intersects(taken)
        w = g0.bounds[2] - g0.bounds[0]
        tries = [(0.0, 0.0)] + ([alt] if alt is not None else [])
        tries += [(sg * k, 0.0) for k in np.arange(2.0, 0.5 * w + 0.01, 2.0) for sg in (1, -1)]
        step = 0.3 * R["U"] * (1 if g0.centroid.y > 0 else -1)
        base = alt if alt is not None else (0.0, 0.0)
        tries += [(base[0], base[1] + step * k) for k in range(1, 7)]
        dx, dy = next((t for t in tries if free(*t)), tries[-1])
        g = translate(g0, dx, dy)
        placed = g.buffer(clear) if placed is None else placed.union(g.buffer(clear))
        out.append(g)
    return out


def typeset_blocks(word: str, st: Style):
    """Lay out `word` right to left in the block construction, every letter drawn by the rules."""
    from shapely.affinity import translate
    from shapely.geometry import box
    from .typeset import Layout, Placed, shape_forms
    R = rules(st)
    lay = Layout(text=word, style=st)
    # Joins as كيوانو draws them (measured on its ي→و and ن→و): the two letters merge in one band on the
    # line up to the joining line (16–17, the top of the join, J − 1), and above it they stand a vertical
    # slit apart. The base of ک is the join of the letter after it, up to its own top.
    line = R["join_line"]
    prev_body, prev_left, prev_joins, prev_top = None, 0.0, False, line + 0.5
    n, leaning = 0, 0
    ink, holes, bodies, groups = [], [], [], []
    pairs = ligatures(shape_forms(word))
    stretch_at = kashida_join(pairs) if R.get("kashida", 0.0) > 0 else set()
    for i, (c, form) in enumerate(pairs):
        if form == " ":
            if prev_body is not None:
                prev_left = prev_left - R["space"]
                prev_body = None
            continue
        if form == "?" or (c not in BUILD and not is_lam_alef(c)):
            lay.skipped.append(c)
            continue
        sign = 1.0
        if leans(c, form):
            mode = R.get("lean_mode", "turn")
            if mode == "logo":                           # as on the logo: ا to the left, a tooth to the right
                sign = 1.0 if c in LEANS else -1.0
            else:                                        # in turn: the first to the left, the next to the right
                sign = 1.0 if leaning % 2 == 0 else -1.0
            leaning += 1
        p = Pen(st, R, 0.0 if R.get("lean_mode") == "none" and leans(c, form) else sign)
        left = build(c, form, p)
        body = _union(p.body)
        hole = _union(p.holes)
        solid = body.difference(hole) if not hole.is_empty else body
        joined = form in ("M", "F") and prev_joins
        jy = max(prev_top, p.join_top if p.join_top is not None else line + 0.5)
        prev_above = above(prev_body, jy if joined else 0.0) if prev_body is not None else None
        this_above = above(solid, jy if joined else 0.0)
        if prev_body is None:
            x = 0.0 if n == 0 else prev_left - (solid.bounds[2] if not solid.is_empty else 0.0)
        elif joined and (prev_above.is_empty or this_above.is_empty):
            # a kashida has nothing above the joining line: the letters meet on the join itself
            # (spacing them by what stands above it put every letter after a kashida at the word's start)
            x = prev_left - (solid.bounds[2] if not solid.is_empty else 0.0) + R["into"]
        else:
            x = gap_shift(prev_above, this_above, R["slit"] if joined else R["touch"])
        if joined and i in stretch_at:
            x -= R["kashida"]                     # the join between them is drawn this much longer
        placed = translate(solid, x, 0)
        ink += [translate(g, x, 0) for g in p.body + p.joins]
        if joined:
            # the join runs from this letter's foot to the one before, wherever spacing put them (a bar of
            # one length broke on the line when a letter had to stand further off), into both by `into`
            foot = solid.intersection(box(-1e5, -2.0, 1e5, line))
            x0 = x + (foot.bounds[2] if not foot.is_empty else solid.bounds[2]) - R["into"]
            x1 = prev_left + R["into"]
            if x1 > x0 + 0.5:
                rise = min(R["join_rise"], 0.14 * (x1 - x0))
                ink.append(_poly([(x0, -1.0 + rise), (x1, -1.0), (x1, R["J"] - 1.0), (x0, R["J"] - 1.0)]))
        holes += [translate(g, x, 0) for g in p.holes]
        bodies.append(placed)
        groups += [(len(bodies) - 1, translate(g, x, 0), alt) for g, alt in p.marks]
        lay.placed.append(Placed(c, form, x, Glyph(_shapes_of(solid), left=left, entry_x=0.0, entry_edge=0.0,
                                                    exit_x=left if form in ("I", "M") else None,
                                                    exit_edge=left if form in ("I", "M") else None),
                                 parts=[("block", "قطعة")]))
        prev_body, prev_left = placed, x + left
        prev_joins = form in ("I", "M")
        prev_top = p.join_top if p.join_top is not None else line + 0.5
        n += 1
    whole = _union(ink)
    if holes:
        whole = whole.difference(_union(holes))
    whole = _union([whole] + clear_marks(bodies, groups, R))
    # where two blocks meet edge to edge no hairline of white may be left, nor a sliver of ink
    whole = whole.buffer(0.6, join_style=2).buffer(-0.6, join_style=2)
    whole = _union([q for q in getattr(whole, "geoms", [whole]) if q.area > 4.0])
    for shape in _shapes_of(whole):
        lay.shapes.append(Tagged(shape, "block", "كتلة"))
    return lay
