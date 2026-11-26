"""From the letters a designer drew to every letter in every form, then a font.

Order of sources for each letter form, best first:
  1. drawn      the designer drew this letter in this form: taken as drawn (its own dots included)
  2. shared     the designer drew another letter with the same body in this form (ت from ب): that body, this
                letter's dots placed where the designer puts dots on that body
  3. derived    built from the designer's own drawing of this letter (or its body) by a rule:
                final = isolated + a join, medial = initial + a join, initial = medial without its join,
                initial = the isolated letter's head (what stands right of its first connector) + a join
  4. missing    reported; the designer is asked for it
Every glyph keeps the name of its source, for the report."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.t2CharStringPen import T2CharStringPen

from . import arabic as A
from . import geom as G
from .layout import Drawn, Reading, connector_zones


@dataclass
class Glyph:
    name: str
    path: G.Path
    jl: Optional[float]
    jr: Optional[float]
    source: str                          # drawn / drawn+join / shared / derived / missing
    how: str = ""
    unicode: Optional[int] = None


@dataclass
class Body:
    path: G.Path
    jl: Optional[float]
    jr: Optional[float]
    dots: List[G.Path]                   # its round marks
    signs: List[G.Path]                  # its other marks (the kaf's sign, the gaf's bars)
    source: str
    how: str
    origin: Optional[Drawn] = None       # the drawing it stands on (same coordinates)


def _round(m: G.Path) -> bool:
    b = G.bounds(m)
    return b is not None and 0.7 < (b[2] - b[0]) / max(b[3] - b[1], 1e-6) < 1.45


class Builder:
    def __init__(self, R: Reading):
        self.R = R
        self.tan, self.T, self.xh = R.tan, R.band, R.x_height
        self.L = 7.0 * R.scale                       # visible connector from a body to its join
        self.SB = (R.spacing / 2) if R.spacing else 7.0 * R.scale
        self.notes: List[str] = []
        self.drawn: Dict[Tuple[str, str], List[Drawn]] = {}
        for d in R.letters:
            self.drawn.setdefault((d.unit, d.form), []).append(d)

    # ------------------------------------------------------------ choosing among drawings
    def _score(self, d: Drawn) -> float:
        s = 0.0
        b = G.bounds(d.body)
        body = A.body(d.unit) if len(d.unit) == 1 else "lamalef"
        if body in A.DESCENDS and d.form in ("isol", "fina"):
            s += 2.0 if b and b[1] < -0.2 * self.xh else -2.0          # the reader expects it below the line
        s += 0.5 if not d.colour else 0.0
        s += 0.3 if "«" in d.where and len(d.where.split("«")[1]) > 3 else 0.0
        return s

    def best(self, unit: str, form: str) -> Optional[Drawn]:
        c = self.drawn.get((unit, form))
        if not c:
            return None
        return max(c, key=self._score)

    def _body_of(self, d: Drawn, source: str, how: str) -> Body:
        return Body(d.body, d.jl, d.jr, [m for m in d.marks if _round(m)], [m for m in d.marks if not _round(m)],
                    source, how, d)

    def _with(self, b: Body, path, jl, jr, how) -> Body:
        return Body(path, jl, jr, b.dots, b.signs, b.source, b.how + how, b.origin)

    # ------------------------------------------------------------ operations on bodies
    def join_right(self, b: Body) -> Body:
        run = G.run_at(b.path, self.T / 2)
        if run is None:
            bb = G.bounds(b.path)
            run = (bb[0], bb[2] - 0.15 * self.xh)
        xr = run[1]
        jr = xr + self.L
        p = G.union([b.path, G.band(xr - 0.12 * self.xh, jr, self.T, self.tan)])
        return self._with(b, p, b.jl, jr, " + a join on the right")

    def join_left(self, b: Body, at: Optional[float] = None) -> Body:
        run = G.run_at(b.path, self.T / 2)
        xl = at if at is not None else (run[0] if run else G.bounds(b.path)[0])
        jl = xl - self.L
        p = G.union([b.path, G.band(jl, xl + 0.12 * self.xh, self.T, self.tan)])
        # a sign standing left of the join would sit on the next letter: the connector runs on under it
        for m in b.signs:
            mb = G.bounds(G.deslant(m, self.tan))
            if mb[0] - 0.02 * self.xh < jl:
                reach = mb[0] - 0.03 * self.xh
                p = G.union([p, G.band(reach, jl + 0.05 * self.xh, self.T, self.tan)])
                jl = reach
        return self._with(b, p, jl, b.jr, " + a join on the left")

    def drop_right(self, b: Body) -> Body:
        """Cut away the connector that leaves on the right, where the body's own foot ends."""
        D = G.deslant(b.path, self.tan)
        zones = connector_zones(D, self.xh, self.T, 0.4 * self.R.scale)
        bb = G.bounds(D)
        right = [z for z in zones if bb[2] - z[-1][0] < 0.08 * self.xh]
        x = min(c[0] for c in right[-1]) if right else (b.jr - self.L if b.jr else bb[2])
        return self._with(b, G.clip(b.path, self.tan, None, x), b.jl, None, " without its join on the right")

    def head(self, b: Body) -> Optional[Body]:
        """What stands right of the first connector from the right (the stem of ل, the loop of ص)."""
        D = G.deslant(b.path, self.tan)
        zones = connector_zones(D, self.xh, self.T, 0.4 * self.R.scale)
        if not zones:
            return None
        z = max(zones, key=lambda z: z[0][0])
        x = z[len(z) // 2][0]
        h = self._with(b, G.clip(b.path, self.tan, x, None), x, b.jr, "'s head (right of its first connector)")
        h.dots = []
        return h

    # ------------------------------------------------------------ dots
    def dot_group(self, n: int, cx: float, cy: float, above: bool) -> List[G.Path]:
        dot = self.R.dot
        dx = 1.04 * (G.bounds(dot)[2] - G.bounds(dot)[0])
        if n == 1:
            return [G.shift(dot, cx, cy)]
        out = [G.shift(dot, cx - dx / 2, cy), G.shift(dot, cx + dx / 2, cy)]
        if n == 3:
            cy2 = cy + (0.9 * dx if above else -0.9 * dx)
            out.append(G.shift(dot, cx + (cy2 - cy) * self.tan, cy2))
        return out

    def dot_place(self, body: Body, above: bool, ref: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        if ref is not None:
            return ref
        p = body.path
        y = 0.85 * self.xh if above else 0.15 * self.xh
        run = G.run_at(p, y) or G.run_at(p, 0.5 * self.xh) or (G.bounds(p)[0], G.bounds(p)[2])
        cy = 1.28 * self.xh if above else -0.35 * self.xh
        return (run[0] + run[1]) / 2 + (cy - y) * self.tan, cy

    def spec_marks(self, n: int, above: bool, x_at: Tuple[float, float], body: Optional[G.Path] = None) -> List[G.Path]:
        """The marks by the logos' measures (khatt.drawn.spec): one dot; two dots as the logos draw them (a
        hairline dash, or two dots); three as the engine builds them (the dash where one mark stands, the dot
        beyond it). Above the letters a mark's top touches the top line; under the line a dot's top and a
        dash's centre stand where the logos put them. `x_at` is a point of the marks' stem line (x, y).
        A mark keeps the white the logos keep between a dot and its letter (0.44 of the dot, as بيدوناق's
        ن): where the letter itself stands there (the bowl of ج under the line), the marks move to the
        nearest place that keeps it, inside the bowl or past it."""
        S, L, tan = self.R.spec, self.R.levels, self.tan
        k = L.k
        dot = self.R.dot
        db = G.bounds(dot)
        d = db[3] - db[1]
        hair, length = S.hair * k, S.dash * k

        def at(y):                                          # x on the marks' stem line at height y
            return x_at[0] + (y - x_at[1]) * tan

        def a_dot(y):
            return G.shift(dot, at(y), y)

        def a_dash(y):
            x = at(y)
            return G.shift(G.band(x - length / 2 - (hair / 2) * tan, x + length / 2 - (hair / 2) * tan, hair, tan), 0.0, y - hair / 2)
        step = 0.7125 * d                                   # dash to dot, as the engine's three marks (0.95 x 0.75)
        if above:
            if n == 1:
                group = [a_dot(L.top - d / 2)]
            elif n == 2:
                group = ([a_dash(L.top - hair / 2)] if S.two_dots == "dash" else
                         [G.shift(a_dot(L.top - d / 2), -0.52 * d, 0), G.shift(a_dot(L.top - d / 2), 0.52 * d, 0)])
            else:
                yd = L.top - d / 2
                group = [a_dash(yd), a_dot(yd + step)]
        else:
            top = -S.below_dot * k
            yc = -S.below_dash * k
            if n == 1:
                group = [a_dot(top - d / 2)]
            elif n == 2:
                group = ([a_dash(yc)] if S.two_dots == "dash" else
                         [G.shift(a_dot(top - d / 2), -0.52 * d, 0), G.shift(a_dot(top - d / 2), 0.52 * d, 0)])
            else:
                group = [a_dash(yc), a_dot(yc - step)]
        return self._clear(group, body, 0.44 * d, above, d) if body is not None else group

    def _clear(self, group: List[G.Path], body: G.Path, white: float, above: bool, d: float) -> List[G.Path]:
        """The group moved (along the stem lines and up or down) to the nearest place where it keeps
        `white` from the letter."""
        def grown(m):
            b = G.bounds(m)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            sx = 1 + 2 * white / max(1.0, b[2] - b[0])
            sy = 1 + 2 * white / max(1.0, b[3] - b[1])
            return G.transformed(m, (sx, 0, 0, sy, cx * (1 - sx), cy * (1 - sy)))

        def free(ms):
            return all(G.bounds(G.intersect(grown(m), body)) is None for m in ms)
        if free(group):
            return group
        bb = G.bounds(body)
        sign = 1 if above else -1
        best = None
        for j in range(0, 41):
            # away from the letter first, and back towards it as far: the opening of a bowl (ج) is a place
            for dy in ((0.0,) if j == 0 else (sign * j * d / 4, -sign * j * d / 4)):
                if -sign * dy > 1.2 * self.xh:
                    continue
                for i in range(0, 13):
                    for dx in ((0.0,) if i == 0 else (i * d / 4, -i * d / 4)):
                        if best is not None and (dx * dx + dy * dy) >= best[0]:
                            continue
                        moved = [G.shift(m, dx + dy * self.tan, dy) for m in group]
                        if free(moved):
                            best = (dx * dx + dy * dy, moved)
            if best is not None and (j * d / 4) ** 2 >= best[0]:
                break
        if best is not None:
            return best[1]
        # nowhere near: past the letter, on the side the marks belong to
        gb = [G.bounds(m) for m in group]
        dy = (bb[3] + white - min(b[1] for b in gb)) if above else (bb[1] - white - max(b[3] for b in gb))
        return [G.shift(m, dy * self.tan, dy) for m in group]

    @staticmethod
    def centre(ms: List[G.Path]) -> Optional[Tuple[float, float]]:
        if not ms:
            return None
        bs = [G.bounds(m) for m in ms]
        return (sum((b[0] + b[2]) / 2 for b in bs) / len(bs), sum((b[1] + b[3]) / 2 for b in bs) / len(bs))


def build_glyphs(R: Reading) -> Tuple[List[Glyph], List[str]]:
    B = Builder(R)
    glyphs: List[Glyph] = []
    hamza_sign = next((m for d in R.letters if d.unit == "ك" for m in d.marks if not _round(m)), None)
    bar_sign = next((m for d in R.letters if d.unit == "گ" for m in d.marks if not _round(m)), None)
    if hamza_sign is None:
        B.notes.append("no hamza is drawn: ء أ إ ؤ ئ are left out until the designer draws one")
    else:
        B.notes.append("the hamza is the kaf's small sign (no ء is drawn): worth drawing one")

    def drawn_body(unit, form, how_prefix="drawn"):
        d = B.best(unit, form)
        return B._body_of(d, "drawn", f"{how_prefix} ({d.where})") if d else None

    def class_units(unit):
        b = A.body(unit)
        return [u for u in A.LETTERS if A.body(u) == b and u != unit]

    def tooth(form):                                # the tooth of ب ت ث ن ي ی in joined forms
        for u in A.LETTERS:
            if A.body(u) in A.TOOTH:
                d = B.best(u, form)
                if d:
                    t = B._body_of(d, "shared", f"the tooth of {d.unit} ({d.where})")
                    t.dots = []
                    return t
        return None

    def source(unit: str, form: str) -> Optional[Body]:
        body = A.body(unit)
        exact = drawn_body(unit, form)
        if exact:
            return exact
        if form in ("init", "medi") and body in A.TOOTH:
            t = tooth(form) or (tooth("medi") if form == "init" else None)
            if t and form == "init" and t.jr is not None:
                t = B.drop_right(t)
            if t:
                return t
        if form == "isol":                          # the letter's own final drawing, its join taken off,
            own = drawn_body(unit, "fina")          # before another letter's drawing (ة before ه)
            if own and own.jr is not None:
                return B.drop_right(own)
        if body == "keheh":                         # ک is the kaf without its small sign
            s = source("ك", form)
            if s:
                return Body(s.path, s.jl, s.jr, [], [], "shared", s.how + ", without the kaf's sign", s.origin)
        if form in ("init", "medi") and body in A.SHARES_JOINED:
            for u in A.LETTERS:
                if A.body(u) == A.SHARES_JOINED[body]:
                    s = source(u, form)
                    if s:
                        return Body(s.path, s.jl, s.jr, [], s.signs, "shared", f"{s.how}, as the joined {A.name(unit)}",
                                    s.origin)
        for u in class_units(unit):
            d = B.best(u, form)
            if d:
                return Body(d.body, d.jl, d.jr, [], [m for m in d.marks if not _round(m)] if body != "kaf" else [],
                            "shared", f"the body of {d.unit} ({d.where})", d)
        # derived from the same letter's other forms (or its body's)
        def any_form(f):
            return drawn_body(unit, f) or next((drawn_body(u, f) for u in class_units(unit) if B.best(u, f)), None)
        if form == "fina":
            iso = source(unit, "isol") if (unit, "isol") != (unit, form) else None
            if iso:
                return B.join_right(iso)
        if form == "medi":
            ini = source(unit, "init")
            if ini:
                return B.join_right(ini)
        if form == "init":
            med = any_form("medi")
            if med:
                return B.drop_right(med)
            iso = any_form("isol")
            if iso:
                h = B.head(iso)
                if h:
                    return B.join_left(h, at=h.jl)
        if form == "isol":
            fin = any_form("fina")
            if fin:
                return B.drop_right(fin)
        return None

    def add_dots(unit: str, form: str, b: Body) -> List[G.Path]:
        n, above = A.LETTERS[unit][2], A.LETTERS[unit][3]
        if unit in A.DOTS_JOINED_ONLY:
            n, above = A.DOTS_JOINED_ONLY[unit] if form in ("init", "medi") else (0, True)
        if n == 0:
            return []
        if B.R.spec is not None:
            # every letter's marks by the logos' measures: where the designer put the marks of this
            # drawing on this side, their stem line; else above or below the letter's middle
            ref = None
            if b.origin is not None:
                ds = [m for m in b.origin.marks if _round(m) and ((G.bounds(m)[1] > 0.45 * B.xh) == above)]
                ref = B.centre(ds)
            return B.spec_marks(n, above, ref if ref is not None else B.dot_place(b, above, None), b.path)
        if b.source == "drawn" and len(b.dots) == n:
            return b.dots                           # the designer's own dots, as drawn
        # the designer's dots on this very drawing, on the same side: the new group stands where they stood
        ref = None
        if b.origin is not None:
            ds = [m for m in b.origin.marks if _round(m) and ((G.bounds(m)[1] > 0.45 * B.xh) == above)]
            ref = B.centre(ds)
        cx, cy = B.dot_place(b, above, ref)
        return B.dot_group(n, cx, cy, above)

    forms_of = lambda u: ("isol", "init", "medi", "fina") if A.LETTERS[u][4] else ("isol", "fina")
    for unit in A.LETTERS:
        if unit == "ء":
            continue
        for form in forms_of(unit):
            b = source(unit, form)
            gname = A.name(unit) + ("" if form == "isol" else "." + form)
            if b is None:
                glyphs.append(Glyph(gname, G.Path(), None, None, "missing", "nothing drawn to build it from"))
                continue
            extra = add_dots(unit, form, b)
            signs = list(b.signs)
            if A.body(unit) == "kaf" and unit == "ک":
                signs = []                                  # keheh: the kaf without its sign
            if unit == "گ" and not any(not _round(m) for m in signs) and bar_sign is not None and form != "fina":
                signs = _bars_for(B, b, bar_sign)
            if unit in ("أ", "إ", "آ", "ؤ", "ئ"):
                if hamza_sign is None and unit != "آ":
                    glyphs.append(Glyph(gname, G.Path(), None, None, "missing", "no hamza drawn"))
                    continue
                signs = signs + _hamza_for(B, b, unit, hamza_sign, bar_sign)
            path = G.union([b.path] + extra + signs)
            how = b.how
            src = b.source
            if src == "drawn" and ("join" in how):
                src = "drawn+join"
            glyphs.append(Glyph(gname, path, b.jl, b.jr, src, how,
                                ord(unit) if form == "isol" else None))
    # lam-alef ligatures
    for la in ("لا", "لأ", "لإ", "لآ"):
        base = drawn_body("لا", "isol") or drawn_body("لا", "fina")
        if base is None:
            continue
        for form in ("isol", "fina"):
            b = base if form == "isol" else B.join_right(base)
            gname = A.unit_name(la) + ("" if form == "isol" else ".fina")
            signs = list(b.signs)
            if la[1] != "ا":
                if hamza_sign is None and la[1] != "آ":
                    continue
                signs += _hamza_for(B, b, la[1], hamza_sign, bar_sign, lamalef=True)
            glyphs.append(Glyph(gname, G.union([b.path] + signs), None if form == "isol" else None, b.jr,
                                "drawn" if form == "isol" else "drawn+join", b.how))
    if hamza_sign is not None:
        hb = G.bounds(hamza_sign)
        glyphs.append(Glyph("hamza", G.shift(hamza_sign, -hb[0], -hb[1] + 0.05 * B.xh), None, None, "derived",
                            "the kaf's sign", 0x0621))
    glyphs.append(Glyph("tatweel", G.band(0, 3.5 * B.L, B.T, B.tan), 0.0, 3.5 * B.L, "derived", "the connector",
                        0x0640))
    return glyphs, B.notes


def _bars_for(B: Builder, b: Body, bar: G.Path) -> List[G.Path]:
    bb = G.bounds(bar)
    w, cy0 = bb[2] - bb[0], (bb[1] + bb[3]) / 2
    out = []
    for k, h in enumerate((cy0, cy0 + 1.8 * (bb[3] - bb[1]) + 0.1 * B.xh)):
        runs = G.runs_at(b.path, h) or G.runs_at(b.path, 0.9 * B.xh)
        stem_left = runs[-1][0] if runs else G.bounds(b.path)[2]
        out.append(G.shift(bar, stem_left - 0.04 * B.xh - w - bb[0], h - cy0))
    return out


def _hamza_for(B: Builder, b: Body, unit: str, hamza: Optional[G.Path], bar: Optional[G.Path],
               lamalef: bool = False) -> List[G.Path]:
    p = b.path
    bb = G.bounds(p)
    top = bb[3]
    if lamalef:                                   # the alef of لا is its left stroke
        h = 0.8 * B.xh
        runs = G.runs_at(p, h)
        x = (runs[0][0] + runs[0][1]) / 2 if runs else bb[0]
        top = h
    else:
        runs = G.runs_at(p, top - 0.06 * B.xh)
        x = (runs[-1][0] + runs[-1][1]) / 2 if runs else (bb[0] + bb[2]) / 2
    if unit in ("ؤ", "ئ"):
        runs = G.runs_at(p, 0.85 * B.xh)
        x = (runs[-1][0] + runs[-1][1]) / 2 if runs else (bb[0] + bb[2]) / 2
        top = 0.95 * B.xh
    if unit == "آ" and bar is not None:
        mb = G.bounds(bar)
        y = top + 0.12 * B.xh
        return [G.shift(bar, x + (y - top) * B.tan - (mb[0] + mb[2]) / 2, y - mb[1])]
    if hamza is None:
        return []
    mb = G.bounds(hamza)
    if unit == "إ":
        foot = G.run_at(p, 0.05 * B.xh) or (bb[0], bb[2])
        y = -0.12 * B.xh - (mb[3] - mb[1])
        return [G.shift(hamza, (foot[0] + foot[1]) / 2 + y * B.tan - (mb[0] + mb[2]) / 2, y - mb[1])]
    y = top + 0.1 * B.xh
    return [G.shift(hamza, x + (y - top) * B.tan - (mb[0] + mb[2]) / 2, y - mb[1])]


# ------------------------------------------------------------------ the font
def place(g: Glyph, R: Reading, SB: float, overlap: float = 8.0):
    p = g.path
    above = G.intersect(p, G.rect(-G.BIG, 0, G.BIG, G.BIG))
    db = G.bounds(G.deslant(above if G.bounds(above) else p, R.tan))
    left = g.jl if g.jl is not None else db[0] - SB
    right = g.jr if g.jr is not None else db[2] + SB
    p = G.shift(p, -left)
    adv = right - left
    if g.jl is not None:
        p = G.union([p, G.band(-overlap, 0.1 * R.x_height, R.band, R.tan)])
    if g.jr is not None:
        p = G.union([p, G.band(adv - 0.1 * R.x_height, adv + overlap, R.band, R.tan)])
    p = G.close_gaps(p)
    p, dropped = G.clean(p)
    return p, max(1, round(adv)), dropped


def features(names, kern=None) -> str:
    dual = sorted(n for n in names if "." not in n and n + ".init" in names and n + ".medi" in names)
    right = sorted(n for n in names if "." not in n and n + ".fina" in names and n + ".init" not in names
                   and not n.startswith("lamAlef"))
    lines = ["languagesystem DFLT dflt;", "languagesystem arab dflt;", ""]
    for feat, sfx, group in (("init", ".init", dual), ("medi", ".medi", dual), ("fina", ".fina", dual + right)):
        lines.append(f"feature {feat} {{")
        lines += [f"  sub {n} by {n}{sfx};" for n in group]
        lines.append(f"}} {feat};")
    lines.append("feature rlig {")
    for alef, lig in (("alef", "lamAlef"), ("alefHamzaAbove", "lamAlefHamzaAbove"),
                      ("alefHamzaBelow", "lamAlefHamzaBelow"), ("alefMadda", "lamAlefMadda")):
        if lig in names and alef + ".fina" in names:
            lines.append(f"  sub lam.init {alef}.fina by {lig};")
            if lig + ".fina" in names:
                lines.append(f"  sub lam.medi {alef}.fina by {lig}.fina;")
    lines.append("} rlig;")
    if kern:
        lines.append("feature kern {")
        lines += [f"  pos {a} {b} {v};" for (a, b), v in sorted(kern.items()) if a in names and b in names]
        lines.append("} kern;")
    return "\n".join(lines) + "\n"


def write_font(glyphs: List[Glyph], R: Reading, path: str, family: str = "Khatt Drawn") -> Dict:
    SB = (R.spacing / 2) if R.spacing else 7.0 * R.scale
    real = [g for g in glyphs if g.source != "missing"]
    order = [".notdef", "space"] + [g.name for g in real]
    cmap = {0x20: "space"}
    charstrings, metrics, dropped_total = {}, {}, 0
    pen = T2CharStringPen(500, None)
    pen.moveTo((50, 0)); pen.lineTo((450, 0)); pen.lineTo((450, 700)); pen.lineTo((50, 700)); pen.closePath()
    charstrings[".notdef"], metrics[".notdef"] = pen.getCharString(), (500, 50)
    space = round(0.45 * R.x_height)
    charstrings["space"], metrics["space"] = T2CharStringPen(space, None).getCharString(), (space, 0)
    ymin, ymax = 0.0, 0.0
    for g in real:
        p, adv, dropped = place(g, R, SB)
        dropped_total += dropped
        pen = T2CharStringPen(adv, None)
        p.draw(pen)
        charstrings[g.name] = pen.getCharString()
        b = G.bounds(p)
        metrics[g.name] = (adv, int(math.floor(b[0])) if b else 0)
        if b:
            ymin, ymax = min(ymin, b[1]), max(ymax, b[3])
        if g.unicode:
            cmap[g.unicode] = g.name
    # room above and below for every glyph, and for a hamza or dots stacked on the tallest letters
    asc, desc = int(math.ceil(ymax + 0.25 * R.x_height)), int(math.ceil(-ymin + 0.1 * R.x_height))
    fb = FontBuilder(1000, isTTF=False)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    ps = "".join(ch for ch in family if ch.isalnum()) + "-Regular"
    fb.setupCFF(ps, {"FullName": family, "FamilyName": family}, charstrings, {})
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=asc, descent=-desc)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=asc, sTypoDescender=-desc, sTypoLineGap=0, usWinAscent=asc, usWinDescent=desc,
                sxHeight=int(R.x_height), achVendID="KHAT")
    fb.setupPost()
    # the designer's own white between a pair of letters, where the drawing shows that pair
    kern = {k: round(v - 2 * SB) for k, v in (R.kerns or {}).items() if abs(v - 2 * SB) > 0.01 * R.x_height}
    addOpenTypeFeaturesFromString(fb.font, features(set(order), kern))
    fb.save(path)
    return {"glyphs": len(order), "ascent": asc, "descent": desc, "slivers_removed": dropped_total, "kern_pairs": len(kern)}
