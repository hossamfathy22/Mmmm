"""Letters built of blocks: a construction for logos drawn with a broad marker or cut from paper.

The engine's other construction (letters.py) is the one of بيدوناق: stems and hairlines of a
contrast pen, round heads and cups. A logo like كيوانو is built another way: every part of every
letter is one straight stroke of a flat pen, a block with straight sides whose ends are cut at an
angle; a head is a block with a narrow slot in it and its corners cut; a tail is one block thrown
down to the left; a dot is a square; two dots are one bar. And no stroke is exactly where the rule
puts it: the designer's hand tilts each one a little, cuts its ends at its own angle, makes it a
little longer or shorter, heavier or lighter.

So a letter here is (1) the rule: which strokes it has and where they go, in the logo's own
measures (`Style.block`), and (2) the hand: for each stroke a tilt, two end cuts, a length, a
weight, a wedge and a small shift. For the logo's own word the hand is measured (`Style.block_hand`);
for any other word it is drawn from the spread measured on the logo, the same for the same word.

How the measures were found, and the proof that they are the logo's (كيوانو rebuilt from them,
without cutting anything from the logo, matches it 95%): README, "طريقة البناء بالكتل".

Coordinates as in letters.py: a letter's right edge at x = 0, it grows to the left, baseline y = 0,
y up. All lengths in pixels of the logo.
"""
from __future__ import annotations

import math
import zlib
from typing import Dict, List, Optional, Tuple

from .letters import Glyph, LETTERS
from .library import Tagged
from .style import Style

Pt = Tuple[float, float]
Poly = List[Pt]

# the rules of the construction, measured on كيوانو (see the README); a logo built this way gives
# its own in Style.block, and what it does not give is taken from here
KIVANO: Dict[str, float] = dict(
    W=30.0,        # pen: width of a stroke at its far end
    k=0.84,        # wedge: a stroke at its root is this part of its far end (bars and tails)
    kv=0.87,       # the same for upright strokes (they widen upwards)
    h=0.84,        # bars (the kaf's arms and base, dots) are this part of the pen
    jh=0.633,      # joins along the baseline are this part of the pen
    end_cut=9.0,   # the far end of a stroke is cut this many degrees off square (10 strokes in 13)
    T=68.4,        # x-height: teeth, heads
    A=92.0,        # alef, lam
    teeth=1.0,     # a tooth is this part of the x-height
    teeth_yeh=0.92,  # the tooth of ي ی ئ (the medial ي of كيوانو is lower than its ن)
    head=66.4,     # head of و: width (ف ق م ه scale it)
    cut=22.5,      # the head's top right corner cut
    cut_bl=18.4,   # the head's bottom left corner cut
    slot_w=14.4, slot_h=30.1, slot_x=35.1, slot_y=23.9,   # the slot in a head: width, height, centre from the right, bottom
    tail_x=14.7, tail_dx=-35.9, tail_dy=-24.5,            # a tail's root from the right edge, and its run
    dot=25.6,      # a dot is a square block
    dot_dx=1.2,    # a dot over a tooth sits on its left shoulder, this far from the tooth's left edge
    dots_l=54.3, dots_y=-18.0,                            # two dots are one bar this long, under the baseline here
    kaf_l=163.5, kaf_bar=0.69, kaf_riser=0.78, slash_l=62.7, slash_a=40.2, bar_rise=4.0,
    round=3.5,     # outer corners softened by this radius
    join_white=8.4,   # white between a tooth and the letter it joins to
    gap=-0.3,      # white between letters that do not join (they touch)
    space=45.0,    # between words
    into=4.0,      # a join runs this far into the letter it reaches
    # the hand for a word the logo does not show: each stroke's tilt (degrees), end cuts (degrees),
    # length and weight are drawn evenly from +- these, twice the median measured on كيوانو's strokes
    # (so the mean matches); `hand` scales them all (0 = the rule alone, clean blocks)
    hand=1.0, hand_tilt=7.0, hand_cut=9.9, hand_len=8.0, hand_weight=3.3,
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


class Pen:
    """Draws the strokes of one letter with the hand that belongs to it."""

    def __init__(self, st: Style, R: Dict[str, float], key: str, measured: Optional[Dict] = None):
        self.st, self.R, self.key = st, R, key
        self.measured = measured
        self.ink: List[Poly] = []
        self.mark_groups: List[List[Poly]] = []   # dots, dashes, hamzas: set clear of the neighbours' after layout
        self.holes: List[Poly] = []

    def hand(self, name: str, n: int = 8) -> List[float]:
        """The hand of one stroke: measured on the logo's word, else drawn from the logo's spread."""
        if self.measured is not None:
            v = self.measured.setdefault(f"{self.key}.{name}", [0.0] * n)
            while len(v) < n:
                v.append(0.0)
            return v
        R = self.R
        amp = R["hand"]
        seed = zlib.crc32(f"{self.key}.{name}".encode())
        def r(i):   # a number in [-1, 1], the same for the same word, letter and stroke
            return ((zlib.crc32(f"{seed}:{i}".encode()) % 20001) / 10000.0) - 1.0
        if n == 8:
            return [amp * R["hand_tilt"] * r(0), amp * R["hand_cut"] * r(1), amp * R["hand_cut"] * r(2),
                    amp * R["hand_len"] * r(3), amp * R["hand_weight"] * r(4), 0.0, 0.0, 0.0]
        return [amp * R["hand_tilt"] * r(0) * 0.5] + [0.0] * (n - 1)

    # -- strokes
    def stroke(self, name: str, p0: Pt, p1: Pt, wide: float, k: Optional[float] = None, hole: bool = False) -> Poly:
        t, c0, c1, dl, dw, dk, sx, sy = self.hand(name)
        p0 = (p0[0] + sx, p0[1] + sy)
        p1 = (p1[0] + sx, p1[1] + sy)
        a = math.radians(t)
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = math.hypot(dx, dy) or 1e-6
        f = max(0.05, (n + dl) / n)
        dx, dy = dx * f, dy * f
        p1 = (p0[0] + dx * math.cos(a) - dy * math.sin(a), p0[1] + dx * math.sin(a) + dy * math.cos(a))
        wide = max(2.0, wide + dw)
        q = quad(p0, p1, wide * ((self.R["k"] if k is None else k) + dk), wide, c0, c1 + self.R["end_cut"])
        (self.holes if hole else self.ink).append(q)
        return q

    def up(self, name: str, x: float, y0: float, y1: float, wide: Optional[float] = None) -> Poly:
        """An upright stroke at x from y0 (its root) to y1."""
        return self.stroke(name, (x, y0), (x, y1), self.R["W"] if wide is None else wide, self.R["kv"])

    def bar(self, name: str, x0: float, x1: float, y: float, wide: float) -> Poly:
        """A bar from its root at x0 to x1 (bars are wide at the right)."""
        return self.stroke(name, (x0, y), (x1, y), wide)

    def join(self, name: str, x0: float, x1: float) -> Poly:
        hw = self.R["jh"] * self.R["W"]
        return self.bar(name, x0, x1, hw / 2, hw)

    def head(self, name: str, xr: float, width: float, height: float, slot: bool = True, bottom: float = 0.0) -> None:
        """A head: a block with its top right and bottom left corners cut and a narrow slot."""
        R = self.R
        t, slope, cc, bc, hw = self.hand(name + ".head", 8)[:5] if self.measured is not None else (self.hand(name + ".head", 5)[0], 0, 0, 0, 0)
        w = width + hw
        c = min(R["cut"] * width / R["head"] + cc, 0.6 * min(w, height))
        b = min(R["cut_bl"] * width / R["head"] + bc, 0.5 * min(w, height))
        T, x0, y0 = bottom + height, xr - w, bottom
        pts = [(x0, T), (xr - c, T - slope), (xr, T - c - slope), (xr, y0), (x0 + b, y0), (x0, y0 + b)]
        self.ink.append(turn(pts, t, (x0 + w / 2, y0 + height / 2)))
        if slot:
            sh = R["slot_h"] * height / R["T"]
            sx = xr - R["slot_x"] * width / R["head"]
            sy = bottom + R["slot_y"] * height / R["T"]
            t2, c0, c1 = (self.hand(name + ".slot", 8)[:3] if self.measured is not None else (0.0, 0.0, 0.0))
            a = math.radians(t2)
            self.holes.append(quad((sx, sy), (sx - sh * math.sin(a), sy + sh * math.cos(a)), R["slot_w"], R["slot_w"], c0, c1 + R["end_cut"]))

    def block(self, name: str, pts: Poly, slot: Optional[Tuple[Pt, Pt, float]] = None) -> None:
        """Any block (a head that is not the box of و): its outline, and a slot from p0 to p1 this wide."""
        t = self.hand(name, 8)[0] * (1.0 if self.measured is not None else 0.5)
        c = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        self.ink.append(turn(pts, t, c))
        if slot:
            a, b, w = slot
            self.holes.append(turn(quad(a, b, w, w, 0.0, self.R["end_cut"]), t, c))

    def square(self, name: str, x: float, y: float, size: Optional[float] = None) -> None:
        """A dot: a square with its bottom right corner at (x, y)."""
        d = self.R["dot"] if size is None else size
        t = self.hand(name, 8)[0]
        self.ink.append(turn([(x - d, y), (x, y), (x, y + d), (x - d, y + d)], t))

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
    def mark_start(self) -> int:
        return len(self.ink)

    def mark_end(self, n0: int) -> None:
        """The strokes drawn since mark_start are a mark."""
        if len(self.ink) > n0:
            self.mark_groups.append(self.ink[n0:])
            del self.ink[n0:]

    def marks(self, spec: Optional[str], cx: float, top: float, bottom: float = 0.0) -> None:
        """a1/a2/a3 dots above (top = the top of the letter there), b1/b2/b3 under; ha hamza; md madda."""
        if not spec:
            return
        n0 = self.mark_start()
        self._marks(spec, cx, top, bottom)
        self.mark_end(n0)

    def hamza(self, cx: float, y: float, s: float) -> None:
        """ء of blocks: a small head on the right, a stroke down to the left from it, a bar back to the right."""
        hw = 0.42 * s
        self.square("hamza", cx + s / 2, y + 0.55 * s, 0.6 * s)
        self.stroke("hamza.down", (cx + 0.1 * s, y + 0.6 * s), (cx - 0.45 * s, y + 0.1 * s), hw)
        self.bar("hamza.bar", cx - 0.5 * s, cx + 0.55 * s, y + hw / 2, hw)

    def _marks(self, spec: str, cx: float, top: float, bottom: float = 0.0) -> None:
        R = self.R
        d, g = R["dot"], 0.25 * R["W"]
        bh = R["h"] * R["W"] * 0.8
        if spec in ("a1", "b1"):
            y = top + g if spec == "a1" else min(bottom, 0) + R["dots_y"] - d / 2
            self.square("dot", cx + d / 2, y)
        elif spec in ("a2", "b2"):
            y = top + g + bh / 2 if spec == "a2" else min(bottom, 0) + R["dots_y"]
            self.bar("dots", cx - R["dots_l"] / 2, cx + R["dots_l"] / 2, y, bh)
        elif spec in ("a3", "b3"):
            if spec == "a3":
                y = top + g + bh / 2
                self.bar("dots", cx - R["dots_l"] / 2, cx + R["dots_l"] / 2, y, bh)
                self.square("dot", cx + d / 2, y + bh / 2 + 0.2 * g)
            else:
                y = min(bottom, 0) + R["dots_y"]
                self.bar("dots", cx - R["dots_l"] / 2, cx + R["dots_l"] / 2, y, bh)
                self.square("dot", cx + d / 2, y - bh / 2 - 0.2 * g - d)
        elif spec in ("ha", "hb"):
            s = 1.3 * d
            y = top + g if spec == "ha" else min(bottom, 0) + R["dots_y"] - 0.6 * s
            self.hamza(cx, y, s)
        elif spec == "md":
            self.bar("madda", cx - 0.9 * R["dots_l"] / 2, cx + 0.9 * R["dots_l"] / 2, top + g + bh / 2, bh)


# ---------------------------------------------------------------- the letters
def _exit(p: Pen, form: str, left: float) -> None:
    if form in ("I", "M"):
        R = p.R
        p.join("join", left - R["join_white"] - R["into"], 0.0)


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
            n0 = p.mark_start()
            p.square("dot", -W + R["dot_dx"], top)
            p.mark_end(n0)
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
        n0 = p.mark_start()
        p.square("dot", -w / 2 + R["dot"] / 2, 0.75 * T + 0.25 * W)
        p.mark_end(n0)
    return -w


def _yeh_end(p: Pen, form, mark):
    """ي ی ى alone or last: a short post, then a wide cup back under it; its two dots under the cup."""
    R = p.R
    W, T = R["W"], R["T"]
    w = 3.0 * W
    p.cup("cup", 0, -w, 0.5 * T, 0.3 * T)
    if mark in ("b2",):
        d = p.depth() + 0.4 * W
        bh = R["h"] * W * 0.8
        n0 = p.mark_start()
        p.bar("dots", -w / 2 - R["dots_l"] / 2, -w / 2 + R["dots_l"] / 2, -d - 0.6 * W - bh / 2, bh)
        p.mark_end(n0)
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
    jw = R["join_white"]
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
    return -L + R["join_white"] if form in ("I", "M") else -L + 0.3 * W


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


# ---------------------------------------------------------------- the word
def soften(ink: List[Poly], holes: List[Poly], r: float):
    """The union of the blocks, the slots cut out, the outer corners softened."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    S = unary_union([Polygon(q).buffer(0) for q in ink if len(q) >= 3])
    if holes:
        S = S.difference(unary_union([Polygon(q).buffer(0) for q in holes if len(q) >= 3]))
    if r > 0:
        S = S.buffer(-r, join_style=1).buffer(r, join_style=1)
    out = []
    for poly in getattr(S, "geoms", [S]):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        out.append([list(poly.exterior.coords)[:-1]] + [list(h.coords)[:-1] for h in poly.interiors])
    return out


def clear_marks(bodies: List[List[Poly]], groups: List[Tuple[int, List[Poly]]], R: Dict[str, float]) -> List[Poly]:
    """Set each mark clear of the letters and of the marks already set: one that would touch them moves
    away from the baseline (down under it, up over it), as a calligrapher staggers the dots of
    neighbours. A mark may touch its own letter (the dot of كيوانو's ن sits on the tooth's shoulder)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    clear = 0.2 * R["W"]
    shapes = [unary_union([Polygon(q).buffer(0) for q in b if len(q) >= 3]).buffer(clear) for b in bodies]
    placed = None
    out: List[Poly] = []
    for i, grp in groups:
        others = [sh for j, sh in enumerate(shapes) if j != i and not sh.is_empty]
        if placed is not None:
            others.append(placed)
        taken = unary_union(others) if others else Polygon()
        g = unary_union([Polygon(q).buffer(0) for q in grp if len(q) >= 3])
        if g.is_empty:
            continue
        up = g.centroid.y > 0
        step = 0.3 * R["W"] * (1 if up else -1)
        dy = 0.0
        for _ in range(6):
            if not g.intersects(taken):
                break
            dy += step
            g = unary_union([Polygon([(x, y + dy) for x, y in q]).buffer(0) for q in grp if len(q) >= 3])
        placed = g.buffer(clear) if placed is None else placed.union(g.buffer(clear))
        out += [[(x, y + dy) for x, y in q] for q in grp]
    return out


def typeset_blocks(word: str, st: Style):
    """Lay out `word` right to left in the block construction."""
    from .typeset import Layout, Placed, shape_forms
    R = rules(st)
    lay = Layout(text=word, style=st)
    measured = None
    hand = st.block_hand or {}
    if hand.get("word") == word:
        measured = {k: list(v) for k, v in hand.get("hand", {}).items()}
    x, prev, n = 0.0, None, 0
    groups: List[Tuple[int, List[Poly]]] = []
    bodies: List[List[Poly]] = []
    ink: List[Poly] = []
    holes: List[Poly] = []
    for c, form in shape_forms(word):
        if form == " ":
            if prev is not None:
                x = prev.x + prev.glyph.left - R["space"]
            prev = None
            continue
        if form == "?" or c not in BUILD:
            lay.skipped.append(c)
            continue
        p = Pen(st, R, f"{n}" if measured is not None else f"{word}|{n}", measured)
        left = build(c, form, p)
        if prev is not None:
            joined = form in ("M", "F")
            x = prev.x + prev.glyph.left - (R["join_white"] if joined else R["gap"])
        glyph = Glyph([[q] for q in p.ink], left=left, entry_x=0.0, entry_edge=0.0,
                      exit_x=left if form in ("I", "M") else None, exit_edge=left if form in ("I", "M") else None)
        here = Placed(c, form, x, glyph, parts=[("block", "قطعة")] * len(p.ink))
        lay.placed.append(here)
        ink += [[(px + x, py) for px, py in q] for q in p.ink]
        holes += [[(px + x, py) for px, py in q] for q in p.holes]
        bodies.append([[(px + x, py) for px, py in q] for q in p.ink])
        groups += [(len(bodies) - 1, [[(px + x, py) for px, py in q] for q in grp]) for grp in p.mark_groups]
        prev = here
        n += 1
    if measured is None:
        ink += clear_marks(bodies, groups, R)
    else:                                   # the logo's own word: its marks where it has them
        ink += [q for _, grp in groups for q in grp]
    for shape in soften(ink, holes, R["round"]):
        lay.shapes.append(Tagged(shape, "block", "كتلة"))
    return lay

