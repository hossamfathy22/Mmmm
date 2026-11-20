"""The shape numbers of the letters: each logo's own, not Sirang's.

The builders of letters.py and geometry.py are full of decimal numbers: where a curve's control point
sits (0.72 of the descender), how far a tail turns back (2.6 of its unit), how wide a bowl opens. They
were measured once, by hand, on the logo سیرنگ (the hand-made بيدوناق) and served every logo since.

Here every such number has a name, its builder and its place in it ("tail#3"), and every logo has its
own value for it: a factor on Sirang's number, `style.numbers[name]`. The factor is what the typefaces
nearest the logo needed (the median of the five nearest by the logo's measures: stem, hairline,
ascender and descender to the x-height, slant), each fitted letter by letter to its own outlines
(bench.free_numbers, gathered by bench.numbers_table into data/numbers.json). A number no near typeface
moved keeps Sirang's value.

Measured (bench.hidden_numbers, 55 typefaces each playing a logo, 4 words, 177 cases): the letters a
logo lacks matched their true shapes 0.4935 with Sirang's numbers, 0.5011 with the nearest typefaces'
(better in 97 cases, worse in 9); ر +0.03, و ك +0.02, ح +0.017. And بيدوناق drawn from سیرنگ with its
own numbers taken away matched the hand-made one 0.596 (with them 0.591, bench.sirang_blind).

The numbers are read through `_K(name, value)`: at import the builders are recompiled so that each
decimal literal in their bodies reads it (`install`). Nothing in a builder's text changes; a number
whose builder's text has changed since the table was made keeps Sirang's value (fingerprints)."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# the factors in force while a word is built (typeset sets them from style.numbers)
NUMBERS: ContextVar = ContextVar("khatt_numbers", default=None)
# factors set by an examination; they win over the logo's (bench.free_numbers)
OVERRIDE: Dict[str, float] = {}
# the names read since this was last cleared (an examination's way of knowing what a letter reads)
USED: set = set()
# how the centre lines drawn since this was last cleared turn: [total turning in radians, crossings]
PATHS = [0.0, 0]


def note_path(pts) -> None:
    """Keep count of how much a centre line turns and whether it crosses itself (geometry.path)."""
    if len(pts) < 3:
        return
    a = np.array([(p[0], p[1]) for p in pts], float)
    d = np.diff(a, axis=0)
    d = d[np.hypot(d[:, 0], d[:, 1]) > 1e-9]
    if len(d) < 2:
        return
    ang = np.arctan2(d[:, 1], d[:, 0])
    PATHS[0] += float(np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi).sum())
    from shapely.geometry import LineString
    if not LineString(a).is_simple:
        PATHS[1] += 1


def _K(key: str, v: float) -> float:
    USED.add(key)
    m = OVERRIDE.get(key)
    if m is None:
        cur = NUMBERS.get()
        m = cur.get(key) if cur else None
    return v * m if m is not None else v


# number crunching, not drawing: Bezier arithmetic, polygon cleanup, marks (dots are kept as drawn), names
SKIP = {"cubic", "line", "arc", "_pts", "soften", "fuse", "clear_marks", "_split_at_baseline", "family_names",
        "top_joins", "marks", "_rings", "_named", "_lib", "circle", "hamza", "tatweel", "dot", "dash", "rect",
        "path", "translate", "box", "flat", "_draw", "_e", "_x", "clip_x", "Point"}


class _Name(ast.NodeTransformer):
    def __init__(self, fname: str):
        self.f, self.i = fname, 0

    def visit_Compare(self, node):            # thresholds choose between variants: left as they are
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, float) and node.value != 0.0:
            key = f"{self.f}#{self.i}"; self.i += 1
            return ast.copy_location(ast.Call(ast.Name("_K", ast.Load()), [ast.Constant(key), ast.Constant(node.value)], []), node)
        return node


NAMED: Dict[str, int] = {}          # builder -> how many numbers it has
PRINTS: Dict[str, str] = {}         # builder -> fingerprint of its text


def _print(src: str) -> str:
    return hashlib.sha1(src.encode("utf-8")).hexdigest()[:12]


def install() -> Dict[str, int]:
    """Recompile every builder of geometry.py and letters.py so that its decimal numbers read _K."""
    if NAMED:
        return NAMED
    from . import geometry, letters
    for mod in (geometry, letters):
        mod._K = _K
        source = inspect.getsource(mod)
        tree = ast.parse(source)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name in SKIP:
                continue
            fn = getattr(mod, node.name, None)
            if not callable(fn) or not hasattr(fn, "__code__") or fn.__code__.co_freevars:
                continue
            text = ast.get_source_segment(source, node) or ""
            node.decorator_list = []
            t = _Name(node.name)
            node.body = [t.visit(b) for b in node.body]
            if not t.i:
                continue
            code = compile(ast.fix_missing_locations(ast.Module([node], [])), inspect.getsourcefile(mod), "exec")
            fn.__code__ = [c for c in code.co_consts if hasattr(c, "co_name") and c.co_name == node.name][0]
            NAMED[node.name] = t.i
            PRINTS[node.name] = _print(text)
    return NAMED


# ------------------------------------------------------------------ each logo's numbers
TABLE = Path(__file__).parent / "data" / "numbers.json"
NEAREST = 5


def measures(stem: float, hair: float, x_height: float, ascender: float, descender: float, slant: float) -> List[float]:
    """What a logo and a typeface are compared by."""
    X = max(1e-6, x_height)
    return [stem / X, hair / X, ascender / X, descender / X, slant]


_TABLE: Dict = {}


def table() -> Dict:
    if not _TABLE and TABLE.exists():
        _TABLE.update(json.loads(TABLE.read_text(encoding="utf-8")))
    return _TABLE


def nearest(me: List[float], skip_family: str = "") -> List[str]:
    t = table().get("fonts", {})
    names = [f for f in t if not skip_family or f.rsplit("-", 1)[0] != skip_family]
    if not names:
        return []
    V = np.array([t[f]["measures"] for f in names], float)
    sd = V.std(0) + 1e-6
    d = np.abs((V - np.array(me, float)) / sd).sum(1)
    return [names[i] for i in np.argsort(d)[:NEAREST]]


def for_logo(st, skip_family: str = "") -> Dict[str, float]:
    """The factors on Sirang's numbers for a logo of these measures: the median of what the nearest
    typefaces needed. Off with KHATT_NUMBERS=sirang (every number Sirang's, as before)."""
    if os.environ.get("KHATT_NUMBERS", "") == "sirang":
        return {}
    install()
    t = table()
    if not t:
        return {}
    near = nearest(measures(st.stem, st.hair, st.x_height, st.ascender, st.descender, st.slant), skip_family)
    fresh = {f for f, p in t.get("prints", {}).items() if PRINTS.get(f) == p}
    # the numbers of one builder are one curve: they are taken together from one typeface, the one of
    # the nearest whose numbers for that builder are nearest their median (taken number by number,
    # the medians drew و's tail of the end of one typeface's curve and the control point of another's,
    # and it turned back on itself)
    by: Dict[str, Dict[str, Dict[str, float]]] = {}
    for f in near:
        for key, v in t["fonts"][f]["numbers"].items():
            b = key.split("#")[0]
            if b in fresh:
                by.setdefault(b, {}).setdefault(f, {})[key] = v
    probes = _probes(st)
    base_style = _without_numbers(st)
    seen: Dict[str, tuple] = {}

    def base_of(p):
        if p not in seen:
            seen[p] = _topology(p, base_style)
        return seen[p]

    def whole(p, cand_style):
        # the critic judges all the builders together, once the logo's numbers are chosen (prune)
        return _same(_topology(p, cand_style), base_of(p))
    out = {}
    for b, fonts in by.items():
        keys = sorted({k for nums in fonts.values() for k in nums})
        M = np.log(np.array([[fonts[f].get(k, 1.0) for k in keys] for f in fonts], float))
        med = np.median(M, 0)
        order = [list(fonts)[i] for i in np.argsort(np.abs(M - med).sum(1))]
        # a builder's numbers are taken only where they draw every letter that uses them whole: no more
        # pieces and no more holes than Sirang's (a typeface's fit is judged by its likeness, and a thin
        # loop at the end of a tail costs it nothing; drawn for another logo, it is a knot)
        for f in order:
            cand = {k: v for k, v in fonts[f].items() if abs(v - 1.0) > 1e-3}
            cs = Style_with(base_style, cand)
            if all(whole(p, cs) for p in probes.get(b, [])):
                out.update({k: round(float(v), 4) for k, v in cand.items()})
                break
    return out


TATWEEL = "\u0640"


def _without_numbers(st):
    from .style import Style
    return Style(**{**st.__dict__, "numbers": {}})


def Style_with(st, numbers: Dict[str, float]):
    from .style import Style
    return Style(**{**st.__dict__, "numbers": dict(numbers)})


def _probes(st) -> Dict[str, List[str]]:
    """Every letter form the engine draws (alone, with a tatweel for its joining forms; one letter of each
    body), by the builders whose numbers it reads."""
    from .letters import LETTERS
    from .recipes import SKELETON
    from .typeset import typeset
    base = _without_numbers(st)
    out: Dict[str, List[str]] = {}
    for c, (joining, _) in LETTERS.items():
        # one letter of each body: its siblings differ by their marks, which read no shape number
        if c == TATWEEL or SKELETON.get(c, c) != c or c in "ء":
            continue
        for text in ([c] + ([c + TATWEEL, TATWEEL + c + TATWEEL, TATWEEL + c] if joining == "D" else [TATWEEL + c])):
            OVERRIDE.clear(); USED.clear()
            try:
                typeset(text, base)
            except Exception:
                continue
            for b in {k.split("#")[0] for k in USED}:
                out.setdefault(b, []).append(text)
    USED.clear()
    return out


TURN_MORE = math.radians(60)     # more turning than this over a letter's centre lines is a hook or a knot


def _same(cand: tuple, base: tuple) -> bool:
    """A letter drawn with other numbers is whole where it has as many pieces and holes as with Sirang's,
    its centre lines cross themselves no more, they turn no more than a little further (the tail of و
    ended past its own control point, and hooked back), and the critic finds no more defects in it
    (ج drawn with a typeface's numbers piled its ink up heavier than the pen)."""
    return cand[:2] == base[:2] and cand[3] <= base[3] and cand[2] <= base[2] + TURN_MORE


def _faults(text: str, st) -> int:
    """How many defects the critic finds in `text`."""
    from .critic import critique
    from .typeset import typeset
    try:
        return len(critique(typeset(text, st)))
    except Exception:
        return 99


def _topology(text: str, st) -> tuple:
    """(pieces of ink, holes in it, turning of its centre lines, their crossings) of `text` drawn small."""
    from PIL import Image, ImageDraw
    from scipy import ndimage as ndi
    from .typeset import flat, typeset
    PATHS[0], PATHS[1] = 0.0, 0
    try:
        lay = typeset(text, st)
    except Exception:
        return (-1, -1, 0.0, 0)
    turn, cross = PATHS[0], PATHS[1]
    rings = [[p for p in sh if len(p) >= 3] for sh in flat(lay.shapes)]
    pts = [q for sh in rings for p in sh for q in p]
    if not pts:
        return (0, 0, turn, cross)
    s = 40.0 / max(1e-6, st.x_height)
    x0 = min(p[0] for p in pts); y1 = max(p[1] for p in pts)
    W = int((max(p[0] for p in pts) - x0) * s) + 8; H = int((y1 - min(p[1] for p in pts)) * s) + 8
    img = np.zeros((H, W), bool)
    for sh in rings:
        acc = np.zeros((H, W), bool)
        for p in sh:
            im = Image.new("L", (W, H), 0)
            ImageDraw.Draw(im).polygon([((x - x0) * s + 4, (y1 - y) * s + 4) for x, y in p], fill=1)
            acc ^= np.asarray(im, bool)
        img |= acc
    _, pieces = ndi.label(img, structure=np.ones((3, 3)))
    bg, nb = ndi.label(~img)
    border = set(np.unique(np.r_[bg[0], bg[-1], bg[:, 0], bg[:, -1]]))
    holes = sum(1 for i in range(1, nb + 1) if i not in border and (bg == i).sum() >= 2)
    return (pieces, holes, turn, cross)


def prune(st) -> List[str]:
    """The logo's numbers once the rest of its rules are read (the self-check fits its rules after the
    numbers are chosen): any letter form they now draw worse than Sirang's do (more pieces, holes,
    knots, or defects the critic sees) gives back the numbers of every builder it reads. Returns the
    builders given back."""
    if not st.numbers:
        return []
    from .typeset import typeset
    dropped: List[str] = []
    base = _without_numbers(st)
    base_topo: Dict[str, tuple] = {}
    base_faults: Dict[str, int] = {}
    # the builders each form reads: a form that reads none of the logo's numbers is drawn as with Sirang's
    reads: Dict[str, set] = {}
    for text in _forms():
        OVERRIDE.clear(); USED.clear()
        try:
            typeset(text, base)
        except Exception:
            continue
        reads[text] = {k.split("#")[0] for k in USED}
    USED.clear()
    for _ in range(3):
        worse = set()
        moved = {k.split("#")[0] for k in st.numbers}
        for text in [t for t, b in reads.items() if b & moved]:
            if text not in base_topo:
                base_topo[text] = _topology(text, base)
            bad = not _same(_topology(text, st), base_topo[text])
            if not bad:
                mine = _faults(text, st)
                if mine:                   # Sirang's drawing is judged only where this one has a defect
                    if text not in base_faults:
                        base_faults[text] = _faults(text, base)
                    bad = mine > base_faults[text]
            if bad:
                worse |= reads[text] & moved
        if not worse:
            break
        st.numbers = {k: v for k, v in st.numbers.items() if k.split("#")[0] not in worse}
        dropped += sorted(worse)
    return dropped


def _forms() -> List[str]:
    """Every letter in each of its forms, as the critic's specimen writes them (its marks too: ج drew a
    pile of ink where ح drew none)."""
    from .letters import LETTERS
    out = []
    for c, (joining, _) in LETTERS.items():
        if c in TATWEEL + "ء":
            continue
        out += [c] + ([c + TATWEEL, TATWEEL + c + TATWEEL, TATWEEL + c] if joining == "D" else [TATWEEL + c])
    return out
