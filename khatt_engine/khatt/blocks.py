"""Letters built of blocks: a construction for logos drawn with a broad marker or cut from paper.

The engine's other construction (letters.py) is the one of بيدوناق: stems and hairlines of a
contrast pen, round heads and cups. A logo like كيوانو is built another way: every edge of every
letter is a straight line (measured: none strays from its line by more than 0.1 to 0.4 px), so a
letter is one or a few blocks of a flat pen with their ends cut at an angle; a head is a block with
a narrow slot in it and its corners cut; a tail is one block thrown down to the left; a dot is a
square; two dots are one bar; letters touch, and neighbours lean towards or away from each other in
turn (a V or a Λ of white between them).

A word is written as بيدوناق was written from سیرنگ:
1. the logo's own letters, measured (`Style.block_parts`: each letter of the logo's word as a clean
   outline, cut where it touches its neighbours, with its dots apart; and its dot, its two-dot bar
   and its head as parts), for every letter the logo shows, as it shows them;
2. a tooth the logo lacks (ب ت ث ...) is the logo's tooth with its own dots, made of the logo's dot
   and bar;
3. any other letter is built of blocks by the rules (`Style.block`), measured on the logo: its pen,
   wedge, heights, head and slot, the cut of its ends; with the logo's own dot, bar and head where
   they fit; and leaning, as the logo's letters lean, in turn to one side and the other.
Nothing is left to chance: the same word is drawn the same way, every lean is the logo's.

How كيوانو was measured, and the proof: README, "طريقة البناء بالكتل".

Coordinates as in letters.py: a letter's right edge at x = 0, it grows to the left, baseline y = 0,
y up. All lengths in pixels of the logo.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .letters import Glyph, LETTERS
from .library import Tagged
from .style import Style

Pt = Tuple[float, float]
Poly = List[Pt]

# the rules of the construction, measured on كيوانو (see the README); a logo built this way gives
# its own in Style.block, and what it does not give is taken from here
KIVANO: Dict[str, float] = dict(
    W=30.0,        # pen: width of a stroke at its far end (uprights measured 24-30, the kaf's riser 30)
    k=0.84,        # wedge: a stroke at its root is this part of its far end (bars and tails)
    kv=0.87,       # the same for upright strokes (ا 24.3 at its foot, 30.5 at its top)
    h=0.84,        # bars (the kaf's arms and base, dots) are this part of the pen
    jh=0.633,      # joins along the baseline are this part of the pen (measured 15-17.5)
    end_cut=9.0,   # the far end of a stroke is cut this many degrees off square (tops cut 6-9 degrees)
    T=68.4,        # x-height: teeth, heads
    A=92.0,        # alef, lam
    teeth=1.0,     # a tooth is this part of the x-height
    teeth_yeh=0.92,  # the tooth of ي ی ئ (the medial ي of كيوانو is lower than its ن)
    head=66.4,     # head of و: width (68 measured on the outline)
    cut=22.5,      # the head's top right corner cut (23 x 22)
    cut_bl=18.4,   # the head's bottom left corner cut
    slot_w=14.4, slot_h=30.1, slot_x=35.1, slot_y=23.9,   # the slot in a head: width, height, centre from the right, bottom
    tail_x=14.7, tail_dx=-35.9, tail_dy=-24.5,            # a tail's root from the right edge, and its run
    dot=25.6,      # a dot is a square block
    dot_dx=1.2,    # a dot over a tooth sits on its left shoulder, this far from the tooth's left edge
    dots_l=54.3, dots_y=-18.0,                            # two dots are one bar this long, under the baseline here
    kaf_l=163.5, kaf_bar=0.69, kaf_riser=0.78, slash_l=62.7, slash_a=40.2, bar_rise=4.0,
    round=3.5,     # outer corners of the letters built by rule softened by this radius
    slit=6.0,      # white between two joined letters, above the join (measured 5-8)
    touch=0.0,     # white between letters that do not join: they touch (both lines of the logo are one piece)
    space=45.0,    # between words
    into=4.0,      # a join runs this far into the letter it reaches
    rock=4.0,      # a letter built by rule leans this many degrees, to one side and the other in turn (ا 5.4, ن -3)
)


def rules(st: Style) -> Dict[str, float]:
    R = dict(KIVANO)
    R.update(st.block or {})
    return R


# ---------------------------------------------------------------- one stroke of the pen
def quad(p0: Pt, p1: Pt, w0: float, w1: float, c0: float = 0.0, c1: float = 0.0) -> Poly:
    """Centre line p0 -> p1, width w0 at p0 and w1 at p1, end edges turned c0 / c1 degrees off square."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1e-6
    dx, dy = dx / L, dy / L
    nx, ny = -dy, dx

    def end(p, w, c):
        a = math.radians(c)
        ex, ey = nx * math.cos(a) - ny * math.sin(a), nx * math.sin(a) + ny * math.cos(a)
        k = w / 2 / max(math.cos(a), 0.2)
        return (p[0] + k * ex, p[1] + k * ey), (p[0] - k * ex, p[1] - k * ey)

    a, b = end(p0, w0, c0)
    c, d = end(p1, w1, c1)
    return [a, c, d, b]


def turn(pts: Poly, deg: float, c: Optional[Pt] = None) -> Poly:
    if not deg:
        return pts
    cx = sum(p[0] for p in pts) / len(pts) if c is None else c[0]
    cy = sum(p[1] for p in pts) / len(pts) if c is None else c[1]
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * ca - (y - cy) * sa, cy + (x - cx) * sa + (y - cy) * ca) for x, y in pts]


# ---------------------------------------------------------------- the logo's parts
def part_shapes(part) -> List[List[Poly]]:
    """A part of Style.block_parts as shapes: [outer ring, holes...] each (even-odd)."""
    return [[[(float(x), float(y)) for x, y in ring] for ring in shape] for shape in (part or {}).get("polys", [])]


def _bounds(shapes: List[List[Poly]]):
    xs = [x for sh in shapes for x, _ in sh[0]]
    ys = [y for sh in shapes for _, y in sh[0]]
    return min(xs), min(ys), max(xs), max(ys)


def _moved(shapes: List[List[Poly]], dx: float, dy: float) -> List[List[Poly]]:
    return [[[(x + dx, y + dy) for x, y in ring] for ring in sh] for sh in shapes]


class Pen:
    """Draws the strokes of one letter: by the rules, with the logo's own dot, bar and head."""

    def __init__(self, st: Style, R: Dict[str, float], parts: Optional[Dict] = None):
        self.st, self.R = st, R
        self.parts = parts or {}
        self.ink: List[Poly] = []           # the letter's body
        self.joins: List[Poly] = []         # its join to the next letter (along the baseline: it does not lean)
        self.holes: List[Poly] = []
        self.logo: List[List[Poly]] = []    # shapes taken whole from the logo's parts (they keep their own lean)
        self.mark_groups: List[List[List[Poly]]] = []   # dots, bars, hamzas: set clear of the neighbours' after layout
        self._marking: Optional[List[List[Poly]]] = None

    # -- strokes
    def _put(self, poly: Poly, hole: bool = False) -> None:
        if self._marking is not None and not hole:
            self._marking.append([poly])
        else:
            (self.holes if hole else self.ink).append(poly)

    def stroke(self, name: str, p0: Pt, p1: Pt, wide: float, k: Optional[float] = None, hole: bool = False) -> Poly:
        q = quad(p0, p1, wide * (self.R["k"] if k is None else k), wide, 0.0, self.R["end_cut"])
        self._put(q, hole)
        return q

    def up(self, name: str, x: float, y0: float, y1: float, wide: Optional[float] = None) -> Poly:
        """An upright stroke at x from y0 (its root) to y1."""
        return self.stroke(name, (x, y0), (x, y1), self.R["W"] if wide is None else wide, self.R["kv"])

    def bar(self, name: str, x0: float, x1: float, y: float, wide: float) -> Poly:
        """A bar from its root at x0 to x1 (bars are wide at the right)."""
        return self.stroke(name, (x0, y), (x1, y), wide)

    def join(self, name: str, x0: float, x1: float) -> Poly:
        hw = self.R["jh"] * self.R["W"]
        q = quad((x0, hw / 2), (x1, hw / 2), hw * self.R["k"], hw, 0.0, self.R["end_cut"])
        self.joins.append(q)
        return q

    def head(self, name: str, xr: float, width: float, height: float, slot: bool = True, bottom: float = 0.0) -> None:
        """A head: the logo's own head where it is of this size, else a block with its top right and
        bottom left corners cut and a narrow slot."""
        R = self.R
        own = part_shapes(self.parts.get("رأس"))
        if own:
            x0, y0, x1, y1 = _bounds(own)
            if abs((x1 - x0) - width) <= 0.12 * width and abs((y1 - y0) - height) <= 0.12 * height:
                self.logo += _moved(own, xr - x1, bottom - y0)
                return
        w = width
        c = min(R["cut"] * width / R["head"], 0.6 * min(w, height))
        b = min(R["cut_bl"] * width / R["head"], 0.5 * min(w, height))
        T, x0, y0 = bottom + height, xr - w, bottom
        self._put([(x0, T), (xr - c, T), (xr, T - c), (xr, y0), (x0 + b, y0), (x0, y0 + b)])
        if slot:
            sh = R["slot_h"] * height / R["T"]
            sx = xr - R["slot_x"] * width / R["head"]
            sy = bottom + R["slot_y"] * height / R["T"]
            self._put(quad((sx, sy), (sx, sy + sh), R["slot_w"], R["slot_w"], 0.0, R["end_cut"]), hole=True)

    def block(self, name: str, pts: Poly, slot: Optional[Tuple[Pt, Pt, float]] = None) -> None:
        """Any block (a head that is not the box of و): its outline, and a slot from p0 to p1 this wide."""
        self._put(pts)
        if slot:
            a, b, w = slot
            self._put(quad(a, b, w, w, 0.0, self.R["end_cut"]), hole=True)

    def square(self, name: str, x: float, y: float, size: Optional[float] = None) -> None:
        """A dot with its bottom right corner at (x, y): the logo's own dot, else a square."""
        own = part_shapes(self.parts.get("نقطة"))
        d = self.R["dot"] if size is None else size
        if own and size is None:
            x0, y0, x1, y1 = _bounds(own)
            for sh in _moved(own, x - x1, y - y0):
                if self._marking is not None:
                    self._marking.append(sh)
                else:
                    self.logo.append(sh)
            return
        self._put([(x - d, y), (x, y), (x, y + d), (x - d, y + d)])

    def dots_bar(self, cx: float, cy: float) -> None:
        """Two dots as one bar centred at (cx, cy): the logo's own, else by the rule."""
        own = part_shapes(self.parts.get("شرطة"))
        if own:
            x0, y0, x1, y1 = _bounds(own)
            for sh in _moved(own, cx - (x0 + x1) / 2, cy - (y0 + y1) / 2):
                if self._marking is not None:
                    self._marking.append(sh)
                else:
                    self.logo.append(sh)
            return
        R = self.R
        bh = R["h"] * R["W"] * 0.8
        self.bar("dots", cx - R["dots_l"] / 2, cx + R["dots_l"] / 2, cy, bh)

    def bar_height(self) -> float:
        own = part_shapes(self.parts.get("شرطة"))
        if own:
            x0, y0, x1, y1 = _bounds(own)
            return y1 - y0
        return self.R["h"] * self.R["W"] * 0.8

    def tail(self, name: str, x: float, y: float, scale: float = 1.0) -> None:
        R = self.R
        self.stroke(name, (x, y), (x + R["tail_dx"] * scale, R["tail_dy"] * scale), R["W"])

    def cup(self, name: str, xr: float, xl: float, top_r: float, top_l: float) -> None:
        """A cup under the baseline, of three strokes: down on the right, along the bottom, up on the left."""
        R = self.R
        W, d = R["W"], self.depth()
        self.up(name + ".right", xr - W / 2, -d - 0.4 * W, top_r)
        self.bar(name + ".bottom", xl, xr, -d, R["h"] * W)
        self.up(name + ".left", xl + W / 2, -d - 0.4 * W, top_l)

    def depth(self) -> float:
        """How far a cup goes under the baseline: as far as the logo's tails (centre of its bottom)."""
        R = self.R
        return -R["tail_dy"] + 0.1 * R["W"]

    # -- marks
    def mark_start(self) -> None:
        self._marking = []

    def mark_end(self, _=None) -> None:
        if self._marking:
            self.mark_groups.append(self._marking)
        self._marking = None

    def marks(self, spec: Optional[str], cx: float, top: float, bottom: float = 0.0) -> None:
        """a1/a2/a3 dots above (top = the top of the letter there), b1/b2/b3 under; ha hamza; md madda."""
        if not spec:
            return
        self.mark_start()
        self._marks(spec, cx, top, bottom)
        self.mark_end()

    def hamza(self, cx: float, y: float, s: float) -> None:
        """ء of blocks: a small head on the right, a stroke down to the left from it, a bar back to the right."""
        hw = 0.42 * s
        self.square("hamza", cx + s / 2, y + 0.55 * s, 0.6 * s)
        self.stroke("hamza.down", (cx + 0.1 * s, y + 0.6 * s), (cx - 0.45 * s, y + 0.1 * s), hw)
        self.bar("hamza.bar", cx - 0.5 * s, cx + 0.55 * s, y + hw / 2, hw)

    def _marks(self, spec: str, cx: float, top: float, bottom: float = 0.0) -> None:
        R = self.R
        d, g = R["dot"], 0.25 * R["W"]
        bh = self.bar_height()
        if spec in ("a1", "b1"):
            y = top + g if spec == "a1" else min(bottom, 0) + R["dots_y"] - d / 2
            self.square("dot", cx + d / 2, y)
        elif spec in ("a2", "b2"):
            y = top + g + bh / 2 if spec == "a2" else min(bottom, 0) + R["dots_y"]
            self.dots_bar(cx, y)
        elif spec in ("a3", "b3"):
            if spec == "a3":
                y = top + g + bh / 2
                self.dots_bar(cx, y)
                self.square("dot", cx + d / 2, y + bh / 2 + 0.2 * g)
            else:
                y = min(bottom, 0) + R["dots_y"]
                self.dots_bar(cx, y)
                self.square("dot", cx + d / 2, y - bh / 2 - 0.2 * g - d)
        elif spec in ("ha", "hb"):
            s = 1.3 * d
            y = top + g if spec == "ha" else min(bottom, 0) + R["dots_y"] - 0.6 * s
            self.hamza(cx, y, s)
        elif spec == "md":
            self.dots_bar(cx, top + g + bh / 2)


# ---------------------------------------------------------------- the letters
def _exit(p: Pen, form: str, left: float) -> None:
    if form in ("I", "M"):
        R = p.R
        p.join("join", left - R["slit"] - R["into"], 0.0)


def _alef(p: Pen, form, mark):
    R = p.R
    W = R["W"]
    p.up("stem", -W / 2, 0, R["A"])
    p.marks(mark, -W / 2, R["A"])
    return -W


def _tooth_letter(p: Pen, form, mark, letter):
    """ب ت ث ن ي and their kin joined on the left: a tooth."""
    R = p.R
    W = R["W"]
    top = R["T"] * (R["teeth"] if letter not in "يیئ" else R.get("teeth_yeh", 0.94))
    p.up("tooth", -W / 2, 0, top)
    if mark and mark.startswith("a"):
        # a dot sits on the tooth's left shoulder (the ن of كيوانو)
        if mark == "a1":
            p.mark_start()
            p.square("dot", -W + R["dot_dx"], top)
            p.mark_end()
        else:
            p.marks(mark, -W / 2, top)
    else:
        p.marks(mark, -W / 2 - 0.1 * W, 0)
    _exit(p, form, -W)
    return -W


def _beh_end(p: Pen, form, mark, letter):
    """ب ت ث alone or last: a boat, a short post at each end and the baseline between."""
    R = p.R
    W, T = R["W"], R["T"]
    w = 3.2 * W
    p.up("post.r", -W / 2, 0, 0.5 * T)
    p.join("base", -w, 0)
    p.up("post.l", -w + W / 2, 0, 0.4 * T, 0.8 * W)
    p.marks(mark, -w / 2, 0.5 * T)
    return -w


def _noon_end(p: Pen, form, mark):
    R = p.R
    W, T = R["W"], R["T"]
    w = 2.6 * W
    p.cup("cup", 0, -w, 0.75 * T, 0.35 * T)
    if mark:
        p.mark_start()
        p.square("dot", -w / 2 + R["dot"] / 2, 0.75 * T + 0.25 * W)
        p.mark_end()
    return -w


def _yeh_end(p: Pen, form, mark):
    """ي ی ى alone or last: a short post, then a wide cup back under it; its two dots under the cup."""
    R = p.R
    W, T = R["W"], R["T"]
    w = 3.0 * W
    p.cup("cup", 0, -w, 0.5 * T, 0.3 * T)
    if mark in ("b2",):
        d = p.depth() + 0.4 * W
        p.mark_start()
        p.dots_bar(-w / 2, -d - 0.6 * W - p.bar_height() / 2)
        p.mark_end()
    elif mark == "ha":
        p.marks("ha", -w / 2, 0.5 * T)
    return -w


def _reh(p: Pen, form, mark):
    """ر ز ژ: a tail, one block thrown down to the left from above the baseline."""
    R = p.R
    W, T = R["W"], R["T"]
    x = -W / 2
    p.stroke("tail", (x, 0.5 * T), (x + 1.2 * R["tail_dx"], 1.2 * R["tail_dy"]), W)
    p.marks(mark, x - 0.2 * W, 0.5 * T + 0.2 * W)
    return -1.1 * W


def _dal(p: Pen, form, mark):
    """د ذ: an upright on the right and the baseline back to the left."""
    R = p.R
    W, T = R["W"], R["T"]
    w = 1.9 * W
    p.up("post", -W / 2, 0, 0.62 * T)
    p.join("base", -w, 0)
    p.marks(mark, -w / 2, 0.62 * T)
    return -w


def _waw(p: Pen, form, mark):
    R = p.R
    p.head("waw", 0.0, R["head"], R["T"])
    p.tail("tail", -R["tail_x"], R["W"] * 0.3)
    if mark == "ha":
        p.marks("ha", -R["head"] / 2, R["T"])
    return -R["head"]


def _head_letter(p: Pen, form, mark, letter):
    """ف ق: a head on the baseline; joined, it goes on; alone or last, ف lies along the baseline and
    ق drops into a cup."""
    R = p.R
    W, T = R["W"], R["T"]
    hw, ht = 0.85 * R["head"], 0.85 * T
    p.head("head", 0.0, hw, ht)
    left = -hw
    if form in ("i", "F"):
        if letter == "ق":
            p.cup("cup", -hw + W, -hw - 1.6 * W, 0.0, 0.35 * T)
            left = -hw - 1.6 * W
        else:
            left = -hw - 1.6 * W
            p.join("base", left, -hw + W)
            p.up("post.l", left + W / 2, 0, 0.35 * T, 0.8 * W)
    p.marks(mark, -hw / 2, ht)
    _exit(p, form, -hw)
    return left


def _meem(p: Pen, form, mark):
    R = p.R
    W, T = R["W"], R["T"]
    hw, ht = 0.8 * R["head"], 0.62 * T
    p.head("head", 0.0, hw, ht)
    if form in ("i", "F"):
        p.stroke("tail", (-hw + W / 2, 0.3 * W), (-hw + W / 2 - 0.15 * W, -p.depth() - 0.3 * W), W, R["kv"])
    _exit(p, form, -hw)
    return -hw


def _heh(p: Pen, form, mark):
    """ه ة. Starting a word: a block with two slots side by side (as square kufic draws it: a head with
    one stroke above reads as ط); in the middle: two heads one over the other, the join between them;
    last: a head with a short flick thrown up to the left; alone (and ة): a head."""
    R = p.R
    W, T = R["W"], R["T"]
    if form == "I":
        hw, ht = 1.15 * R["head"], 0.95 * T
        c = R["cut"] * 0.8
        pts = [(-hw, ht), (-c, ht), (0.0, ht - c), (0.0, 0.0), (-hw, 0.0)]
        sh = 0.5 * ht
        p.block("head", pts, ((-0.3 * hw, 0.25 * ht), (-0.3 * hw, 0.25 * ht + sh), R["slot_w"]))
        p.holes.append(quad((-0.72 * hw, 0.35 * ht), (-0.72 * hw, 0.35 * ht + sh), R["slot_w"], R["slot_w"], 0.0, R["end_cut"]))
    elif form == "M":
        hw = 0.65 * R["head"]
        p.head("head", 0.0, hw, 0.6 * T, bottom=0.1 * T)
        p.head("low", 0.0, hw, 0.6 * T, bottom=-0.6 * T)
        ht = 0.7 * T
    elif form == "F" and mark is None:
        hw, ht = 0.8 * R["head"], 0.7 * T
        p.head("head", 0.0, hw, ht)
        p.stroke("flick", (-hw + 0.6 * W, ht - 0.3 * W), (-hw - 0.3 * W, ht + 0.45 * T), 0.8 * W)
    else:
        hw, ht = 0.8 * R["head"], 0.8 * T
        p.head("head", 0.0, hw, ht)
    p.marks(mark, -hw / 2, ht)
    _exit(p, form, -hw)
    return -hw


def _sad(p: Pen, form, mark, letter):
    """ص ض ط ظ: a long head lying on the baseline; ص ends in a small tooth, ط carries a tall stroke."""
    R = p.R
    W, T = R["W"], R["T"]
    hw, ht = 1.9 * R["head"] * 0.85, 0.62 * T
    left = -hw
    if letter in "صض":
        p.up("tooth", -W / 2, 0, 0.62 * T)
        p.head("head", -W, hw - W, ht)
    else:
        p.head("head", 0.0, hw, ht)
        p.up("stem", -hw + 1.2 * W, 0, R["A"])
    if form in ("i", "F") and letter in "صض":
        p.cup("cup", -hw + W, -hw - 1.8 * W, 0.0, 0.35 * T)
        left = -hw - 1.8 * W
    p.marks(mark, -hw / 2, ht)
    _exit(p, form, -hw)
    return left


def _seen(p: Pen, form, mark):
    R = p.R
    W, T = R["W"], R["T"]
    t = 0.55 * T
    jw = R["slot_w"]                      # the white inside a letter is the head's slot
    xs = [-W / 2 - i * (W + jw) for i in range(3)]
    for i, x in enumerate(xs):
        p.up(f"tooth{i}", x, 0, t, 0.9 * W)
    left = xs[-1] - W / 2
    p.join("base", left, 0)
    if form in ("i", "F"):
        p.cup("cup", left + W, left - 1.8 * W, 0.0, 0.35 * T)
        left = left - 1.8 * W
    p.marks(mark, xs[1], t)
    _exit(p, form, xs[-1] - W / 2)
    return left


def _jeem(p: Pen, form, mark):
    """ج ح خ: a hat on the left at the top and from its right end a stroke falling to the right, to the
    baseline (joined) or, alone or last, on into a bowl thrown back under the baseline to the left."""
    R = p.R
    W, T, h = R["W"], R["T"], R["h"]
    hb = h * W
    w = 2.5 * W
    p.bar("hat", -w, -1.15 * W, T - hb / 2, hb)
    if form in ("I", "M"):
        p.stroke("fall", (-1.45 * W, T - 0.3 * W), (-0.5 * W, 0.45 * W), W)
        below = 0.0
    else:
        d = p.depth() + 0.3 * W
        knee = (-0.45 * W, 0.3 * T)
        p.stroke("fall", (-1.45 * W, T - 0.3 * W), knee, W)
        p.stroke("back", knee, (-w + 0.3 * W, -d), W)
        p.bar("bottom", -w, -0.2 * W, -d, hb)
        below = -d - hb / 2
    if mark == "b1":
        p.marks("b1", -1.0 * W if form in ("I", "M") else -1.2 * W, 0, below)
    elif mark == "a1":
        p.marks("a1", -w + 0.9 * W, T)
    elif mark == "b3":
        p.marks("b3", -1.0 * W, 0, below)
    _exit(p, form, -w)
    return -w


def _ain(p: Pen, form, mark):
    """ع غ. Starting a word: a mouth open to the right (an arm on top, a back falling to the baseline);
    in the middle and at the end: a V open at the top, standing on the join; at the end and alone a
    bowl is thrown back under the baseline to the left."""
    R = p.R
    W, T, h = R["W"], R["T"], R["h"]
    hb = h * W
    d = p.depth() + 0.3 * W
    if form in ("i", "I"):
        w = 2.0 * W
        top = 1.05 * T
        p.bar("arm", -w + 0.3 * W, -0.15 * W, top - hb / 2, hb)
        p.stroke("back", (-1.2 * W, 0.35 * W), (-w + 0.75 * W, top - 0.2 * W), W, R["kv"])
        foot = (-1.2 * W, 0.2 * W)
    else:
        # open at the top, a V on the join (a closed head would read as م)
        w = 2.3 * W
        top = 0.95 * T
        foot = (-w / 2, 0.35 * W)
        p.stroke("right", foot, (-0.45 * W, top - 0.2 * W), 0.9 * W, R["kv"])
        p.stroke("left", foot, (-w + 0.45 * W, top - 0.2 * W), 0.9 * W, R["kv"])
    left = -w
    if form in ("i", "F"):
        p.stroke("drop", foot, (-w - 0.2 * W, -d), W)
        p.bar("bottom", -w - 0.5 * W, foot[0] + 0.6 * W, -d, hb)
        left = -w - 0.5 * W
    p.marks(mark, -w / 2, top)
    _exit(p, form, -w)
    return left


def _kaf(p: Pen, form, mark, letter):
    """ک as كيوانو draws it: a long base, a riser on the right, the bar back to a tip, and from the tip
    the slash up to the right. Alone or last, a short post stands at the left end of the base."""
    R = p.R
    W, T, h = R["W"], R["T"], R["h"]
    L = R["kaf_l"]
    hb = h * W
    if form in ("I", "M"):
        p.bar("base", -L - R["into"], 0, hb / 2, hb)
    else:
        p.bar("base", -L + 0.3 * W, 0, hb / 2, hb)
        p.up("post.l", -L + 0.3 * W + W / 2, 0, 0.35 * T, 0.8 * W)
    p.up("riser", -W / 2, 0, T * R["kaf_riser"])
    tip = (-L + 0.07 * L, T * R["kaf_bar"])
    a = math.radians(R["bar_rise"])
    p.stroke("bar", tip, (-W / 2, tip[1] + (L * 0.93 - W / 2) * math.tan(a)), h * W)
    s = math.radians(R["slash_a"])
    p.stroke("slash", tip, (tip[0] + R["slash_l"] * math.cos(s), tip[1] + R["slash_l"] * math.sin(s)), h * W)
    if letter == "گ":
        p.stroke("slash2", (tip[0] + 0.35 * W, tip[1] + 0.9 * W),
                 (tip[0] + 0.35 * W + R["slash_l"] * math.cos(s), tip[1] + 0.9 * W + R["slash_l"] * math.sin(s)), h * W)
    return -L if form in ("I", "M") else -L + 0.3 * W


def _lam(p: Pen, form, mark):
    R = p.R
    W, T = R["W"], R["T"]
    p.up("stem", -W / 2, 0, R["A"])
    left = -W
    if form in ("i", "F"):
        p.cup("cup", 0, -2.4 * W, 0.0, 0.35 * T)
        left = -2.4 * W
    _exit(p, form, -W)
    return left


def _hamza(p: Pen, form, mark):
    R = p.R
    p.marks("ha", -0.5 * R["dot"], 0.1 * R["T"])
    return -R["dot"]


def _tatweel(p: Pen, form, mark):
    R = p.R
    p.join("join", -1.5 * R["W"], 0)
    return -1.5 * R["W"]


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


def build(c: str, form: str, p: Pen) -> float:
    """Draw letter `c` in `form` with pen `p`; returns the body's left edge."""
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


# ---------------------------------------------------------------- the logo's letters
TEETH_KIN = "بتثنيیئپ"
# letters with one body that differ by their dots: the logo's drawing of one serves the others, with
# their own dots (in the forms where the bodies are one)
SAME_BODY = [(TEETH_KIN, "IM"), ("بتثپ", "iF"), ("جحخچ", "iIMF"), ("دذ", "iF"), ("رزژ", "iF"), ("سش", "iIMF"),
             ("صض", "iIMF"), ("طظ", "iIMF"), ("عغ", "iIMF"), ("فقڤ", "IM"), ("فڤ", "iF"), ("اأإآٱ", "iF"),
             ("وؤ", "iF"), ("هة", "iF")]
TWINS = {"ك": "ک", "ک": "ك", "ي": "ی", "ی": "ي"}     # one drawing (ي and ی differ only when alone or last)


def logo_letter(parts: Dict, c: str, form: str, seen: Optional[Dict[str, int]] = None):
    """The logo's own drawing of `c` in `form` (or in the form that differs only by the join from the
    right), and whether it is another letter's tooth standing in (then it comes without its dots).
    A letter the logo draws more than once in different ways ("و.F", "و.F/2") takes its drawings in
    turn, as the logo does (`seen` counts them)."""
    other = {"i": "F", "F": "i", "I": "M", "M": "I"}[form]
    names = [c] + ([TWINS[c]] if c in TWINS and (form in ("I", "M") or c in "كک") else [])

    def pick(key):
        alts = [parts[key]] + [parts[f"{key}/{i}"] for i in range(2, 9) if f"{key}/{i}" in parts]
        if seen is None:
            return alts[0]
        k = seen.get(key, 0)
        seen[key] = k + 1
        return alts[k % len(alts)]

    for name in names:
        for key in (f"{name}.{form}", f"{name}.{other}"):
            if key in parts:
                return pick(key), False
    for group, forms in SAME_BODY:
        if c in group and form in forms:
            for q in group:
                for f in (form, other):
                    if f in forms and f"{q}.{f}" in parts:
                        return parts[f"{q}.{f}"], True
    return None, False


def _geom(shapes: List[List[Poly]]):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    gs = []
    for sh in shapes:
        if len(sh[0]) >= 3:
            g = Polygon(sh[0]).buffer(0)
            for hole in sh[1:]:
                if len(hole) >= 3:
                    g = g.difference(Polygon(hole).buffer(0))
            gs.append(g)
    return unary_union(gs) if gs else Polygon()


def _shapes_of(geom) -> List[List[Poly]]:
    out = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        out.append([list(poly.exterior.coords)[:-1]] + [list(h.coords)[:-1] for h in poly.interiors])
    return out


def soften(ink: List[Poly], holes: List[Poly], r: float):
    """The union of the blocks, the slots cut out, the outer corners softened (as shapes)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    S = unary_union([Polygon(q).buffer(0) for q in ink if len(q) >= 3])
    if holes:
        S = S.difference(unary_union([Polygon(q).buffer(0) for q in holes if len(q) >= 3]))
    if r > 0 and not S.is_empty:
        S = S.buffer(-r, join_style=1).buffer(r, join_style=1)
    return _shapes_of(S)


def clear_marks(bodies, groups, R: Dict[str, float]):
    """Set each mark clear of the other letters and of the marks already set, as a calligrapher sets
    the dots of neighbours. `groups`: (owner, shapes, other place) in order. A mark may touch its own
    letter (the dot of كيوانو's ن stands on the tooth's shoulder). A mark that would touch another
    letter or mark goes, in this order: to its other place (the logo's dot on a shoulder where the
    neighbour leaves no room goes over its tooth); a little sideways (up to half its width); away from
    the baseline (down under it, up over it)."""
    from shapely.ops import unary_union
    # the white the logo keeps between a mark and another letter is its slit (كيوانو: 6.3, 7.0, 10.2)
    clear = max(1.0, R["slit"] - 1.0)
    shapes = [b.buffer(clear) for b in bodies]
    placed = None
    out = []
    for i, grp, alt in groups:
        g0 = _geom(grp)
        if g0.is_empty:
            continue
        others = [sh for j, sh in enumerate(shapes) if j != i and not sh.is_empty]
        if placed is not None:
            others.append(placed)
        taken = unary_union(others) if others else None

        def free(dx, dy):
            if taken is None:
                return True
            from shapely.affinity import translate
            return not translate(g0, dx, dy).intersects(taken)

        w = g0.bounds[2] - g0.bounds[0]
        tries = [(0.0, 0.0)]
        if alt is not None:
            tries.append(alt)
        tries += [(sg * k, 0.0) for k in np.arange(2.0, 0.5 * w + 0.01, 2.0) for sg in (1, -1)]
        up = g0.centroid.y > 0
        step = 0.3 * R["W"] * (1 if up else -1)
        base = alt if alt is not None else (0.0, 0.0)
        tries += [(base[0], base[1] + step * k) for k in range(1, 7)]
        dx, dy = next((t for t in tries if free(*t)), tries[-1])
        from shapely.affinity import translate
        g = translate(g0, dx, dy)
        placed = g.buffer(clear) if placed is None else placed.union(g.buffer(clear))
        out += _moved(grp, dx, dy)
    return out


def _gap_shift(prev, body, white: float) -> float:
    """How far left of 0 `body` (its right edge at 0) must go to stand `white` clear of `prev` (placed)."""
    from shapely.affinity import translate
    if prev.is_empty or body.is_empty:
        return prev.bounds[0] if not prev.is_empty else 0.0
    hi = prev.bounds[2] - body.bounds[0] + 1.0          # right of here they cannot meet... search leftwards
    lo = prev.bounds[0] - body.bounds[2] - 4 * white - 50.0
    def apart(dx):
        b = translate(body, dx, 0)
        return prev.distance(b) if not prev.intersects(b) else -prev.intersection(b).area
    if apart(hi) < white:           # (cannot happen for letters set right to left, kept for safety)
        hi = prev.bounds[2] + abs(body.bounds[0]) + 1.0
    for _ in range(50):             # the rightmost place at which they are `white` apart
        mid = (lo + hi) / 2
        if apart(mid) >= white - 1e-3:
            lo = mid
        else:
            hi = mid
    return lo


def _above(geom, y: float):
    from shapely.geometry import box
    return geom.intersection(box(-1e5, y, 1e5, 1e5))


# ---------------------------------------------------------------- the word
def typeset_blocks(word: str, st: Style):
    """Lay out `word` right to left in the block construction (see the module's docstring)."""
    from shapely.affinity import rotate, translate
    from shapely.ops import unary_union
    from .typeset import Layout, Placed, shape_forms
    R = rules(st)
    parts = st.block_parts or {}
    lay = Layout(text=word, style=st)
    # joined letters keep their slit above the bars on the baseline (the joins, and the base of ک that
    # runs on into the next letter as its join: 23-26 high in كيوانو)
    jy = 0.9 * R["W"]
    prev_body, prev_left, prev_joins = None, 0.0, False
    n = 0
    built_ink: List[Poly] = []
    built_holes: List[Poly] = []
    logo_shapes: List[List[Poly]] = []
    bodies, groups = [], []
    seen: Dict[str, int] = {}
    for c, form in shape_forms(word):
        if form == " ":
            if prev_body is not None:
                prev_left = prev_left - R["space"]
                prev_body = None
            continue
        if form == "?" or c not in BUILD:
            lay.skipped.append(c)
            continue
        own, kin = logo_letter(parts, c, form, seen)
        p = Pen(st, R, parts)
        if own is not None:
            body_shapes = part_shapes({"polys": own.get("polys", [])})
            mark_shapes = [] if kin else part_shapes({"polys": own.get("marks", [])})
            if kin or "marks" not in own:
                # the logo's body (its tooth, its head, a letter the model made of its parts), and this
                # letter's own dots, made of the logo's dot and bar: over the top of the body or under it
                bx0, by0, bx1, by1 = _bounds(body_shapes) if body_shapes else (-R["W"], 0.0, 0.0, R["T"])
                how, mark = BUILD[c]
                if mark == "b2*":
                    mark = "b2" if form in ("I", "M") else None
                top_part = _above(_geom(body_shapes), 0.5 * R["T"])
                if mark and mark.startswith(("a", "h", "m")) and not top_part.is_empty:
                    hi = _above(top_part, top_part.bounds[3] - 0.25 * R["W"])
                    cx, top = (hi.bounds[0] + hi.bounds[2]) / 2, hi.bounds[3] + R["slit"] - 0.25 * R["W"]
                    p.marks(mark, cx, top)
                elif mark:
                    p.marks(mark, (bx0 + bx1) / 2 if form in ("i", "F") else -R["W"] / 2 - 0.1 * R["W"], 0, min(0.0, by0))
            body = _geom(body_shapes)
            left = body.bounds[0] if not body.is_empty else -R["W"]
            angle = 0.0
        else:
            left = build(c, form, p)
            # a letter built by rule leans as the logo's letters lean, to one side and the other in turn
            angle = R["rock"] * (1.0 if n % 2 else -1.0)
            pivot = (left / 2.0, 0.0)
            p.ink = [turn(q, angle, pivot) for q in p.ink]
            p.holes = [turn(q, angle, pivot) for q in p.holes]
            p.logo = [[turn(r, angle, pivot) for r in sh] for sh in p.logo]
            body_shapes = [[q] for q in p.ink] + p.logo
            mark_shapes = []
            body = _geom(body_shapes)
            if p.holes:
                body = body.difference(_geom([[q] for q in p.holes]))
        joined = form in ("M", "F") and prev_joins
        if prev_body is None:
            x = 0.0 if n == 0 else prev_left - (body.bounds[2] if not body.is_empty else 0.0)
        else:
            ref_prev = _above(prev_body, jy if joined else 0.0)
            ref_this = _above(body, jy if joined else 0.0)
            x = _gap_shift(ref_prev, ref_this, R["slit"] if joined else R["touch"])
        placed_body = translate(body, x, 0)
        if own is not None:
            logo_shapes += _moved(body_shapes, x, 0.0)
            if mark_shapes:
                # the logo's own dots, where the logo has them; their other place: over the tooth
                m = _geom(mark_shapes)
                above_top = m.centroid.y > 0.5 * R["T"]
                top = _above(body, 0.5 * R["T"])
                if above_top and not top.is_empty:
                    hi = _above(body, top.bounds[3] - 0.25 * R["W"])
                    cx = (hi.bounds[0] + hi.bounds[2]) / 2
                    alt = (cx - (m.bounds[0] + m.bounds[2]) / 2, hi.bounds[3] + R["slit"] - m.bounds[1])
                else:
                    alt = None
                groups.append((len(bodies), _moved(mark_shapes, x, 0.0), alt))
        else:
            built_ink += [[(px + x, py) for px, py in q] for q in p.ink]
            built_holes += [[(px + x, py) for px, py in q] for q in p.holes]
            logo_shapes += _moved(p.logo, x, 0.0)
        built_ink += [[(px + x, py) for px, py in q] for q in p.joins]
        bodies.append(placed_body)
        groups += [(len(bodies) - 1, _moved(grp, x, 0.0), None) for grp in p.mark_groups]
        glyph = Glyph([[q] for q in p.ink] + [sh for sh in body_shapes if own is not None], left=left, entry_x=0.0,
                      entry_edge=0.0, exit_x=left if form in ("I", "M") else None,
                      exit_edge=left if form in ("I", "M") else None)
        lay.placed.append(Placed(c, form, x, glyph, parts=[("logo" if own is not None else "block", "قطعة")]))
        prev_body, prev_left = placed_body, x + left
        prev_joins = form in ("I", "M")
        n += 1
    marks = clear_marks(bodies, groups, R)
    built = _geom([[q] for q in built_ink])
    if built_holes:
        built = built.difference(_geom([[q] for q in built_holes]))
    if R["round"] > 0 and not built.is_empty:
        built = built.buffer(-R["round"], join_style=1).buffer(R["round"], join_style=1)
    whole = unary_union([built, _geom(logo_shapes), _geom(marks)])
    for shape in _shapes_of(whole):
        lay.shapes.append(Tagged(shape, "block", "كتلة"))
    return lay
