"""The measures a drawing is brought to: read by the engine from the logos the drawing was made from.

python -m khatt.drawn.spec --logo sirang.png "سیرنگ" --logo bidonaq.png "بيدوناق" -o spec.json

Every length is given at a stem of 23 (the measure of سیرنگ), so that logos of any size speak one
language, and every value keeps where it was measured. The engine reads each logo (khatt.editor.read_logo:
stem, hairline, x-height, ascender, descender, white, dot, dash, slant, the white between letters that do
not join); the marks are found on the picture itself: where a mark above stands against the x-height,
where a dot and a dash under the line stand.

What the logos say together is the spec (`combine`): a value two logos measure is their mean; a value only
one shows is that one's; a value two logos measure differently, because they measure different letters,
is kept apart (the tail of سیرنگ's ر against the bowls of بيدوناق's و ق).
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

STEM = 23.0


@dataclass
class Spec:
    x_height: float          # all lengths at a stem of 23
    ascender: float          # the top line, above the x-height: tall letters, and the top of every mark above
    bowl: float              # how far a bowl or a tail goes under the line (ق و ى ح ع ...)
    tail: float              # how far ر ز ژ go under the line
    hair: float              # the hairline pen: joins, and the dash two dots become
    dot: float
    dash: float              # length of the dash two dots become
    white: float
    gap: float               # white between letters that do not join
    slant: float
    below_dot: float         # a dot under the line: its top, under the baseline
    below_dash: float        # a dash under the line: its centre, under the baseline
    two_dots: str = "dash"   # "dash" (as the logos draw them) or "dots"
    heads: str = "x-height"  # round heads reach the x-height ("x-height"), or keep their drawn level ("drawn")
    sources: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str) -> "Spec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------ one logo
def _marks_on_picture(image: bytes, st: Dict) -> Dict[str, List[float]]:
    """The marks of a logo as they stand against its lines (pixels of the picture, as the engine's style)."""
    from io import BytesIO

    from PIL import Image
    from scipy import ndimage
    g = np.asarray(Image.open(BytesIO(image)).convert("L"), dtype=float)
    ink = g < (g.min() + g.max()) / 2
    if ink.mean() > 0.5:                       # light letters on a dark ground
        ink = ~ink
    lab, n = ndimage.label(ink)
    comps = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        ys, xs = np.nonzero(lab[sl] == i + 1)
        comps.append((sl[1].start + xs.min(), sl[1].start + xs.max(), sl[0].start + ys.min(), sl[0].start + ys.max(), len(xs)))
    X, U = st["x_height"], st["stem"]
    big = [c for c in comps if c[3] - c[2] > 0.6 * X]
    if not big:
        return {}
    tops = [c[2] for c in big]
    # the x-height line: the commonest top of the big pieces; a tall letter rises above it, so among tops
    # as common as each other the lowest is the line (two pieces: سیر at the line, نگ under the gaf's bars)
    x_top = max(tops, key=lambda t: (sum(abs(t - u) < 0.04 * X for u in tops), t))
    base = x_top + X
    out: Dict[str, List[float]] = {"above_top": [], "below_dot_top": [], "below_dash_centre": []}
    k = STEM / U
    for x0, x1, y0, y1, area in comps:
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if h > 0.5 * X or area < 0.02 * U * U:
            continue
        if y1 < x_top + 0.2 * X:                              # above the letters
            out["above_top"].append((x_top - y0) * k)
        elif y0 > base - 0.1 * X:                             # under the line
            if 0.7 <= w / h <= 1.4:
                out["below_dot_top"].append((y0 - base) * k)
            elif w / h > 2:
                out["below_dash_centre"].append(((y0 + y1) / 2 - base) * k)
    return out


def measure_logo(image: bytes, text: str) -> Dict:
    """The engine's measures of one logo at a stem of 23, and where its marks stand."""
    from ..editor import read_logo
    st = read_logo(image, text)["style"]
    k = STEM / st["stem"]
    m = {f: st[f] * k for f in ("hair", "x_height", "ascender", "descender", "white", "dot", "dash", "group_gap", "seen_drop")}
    m["slant"] = st["slant"]
    m["marks"] = _marks_on_picture(image, st)
    m["text"] = text
    return m


# ------------------------------------------------------------------ the logos together
TAIL_LETTERS = "رزژ"


def combine(logos: List[Dict]) -> Spec:
    """One spec from several logos' measures, with the source of each value."""
    src: Dict[str, str] = {}

    def mean_of(key, name, pick=None):
        vals = [(l[key], l["text"]) for l in logos if l.get(key)]
        if pick:
            vals = [v for v in vals if pick(v[1])] or vals
        v = statistics.mean(x for x, _ in vals)
        src[name] = " و".join(t for _, t in vals) + f" ({', '.join(f'{x:.1f}' for x, _ in vals)})"
        return v

    def marks(kind):
        vals = [(x, l["text"]) for l in logos for x in l["marks"].get(kind, [])]
        return vals

    spec_vals = {
        "x_height": mean_of("x_height", "x_height"),
        "ascender": mean_of("ascender", "ascender"),
        "hair": mean_of("hair", "hair"),
        "dot": mean_of("dot", "dot"),
        "white": mean_of("white", "white"),
        "slant": mean_of("slant", "slant"),
    }
    # a logo's descender is measured on its own letters: the deepest letters that go down tell the bowls
    by_depth = sorted(logos, key=lambda l: l["descender"])
    spec_vals["bowl"] = by_depth[-1]["descender"]
    src["bowl"] = f"{by_depth[-1]['text']} ({by_depth[-1]['descender']:.1f})"
    tails = [l for l in logos if any(c in l["text"] for c in TAIL_LETTERS)]
    t = min(tails, key=lambda l: l["descender"]) if tails else by_depth[0]
    spec_vals["tail"] = t["descender"]
    src["tail"] = f"{t['text']} ({t['descender']:.1f})"
    # dash and the white between unjoined letters: the logo that shows the most of them speaks
    dashes = [l for l in logos if l.get("dash")]
    d = min(dashes, key=lambda l: l["dash"]) if dashes else None
    spec_vals["dash"] = d["dash"] if d else 1.5 * spec_vals["dot"]
    src["dash"] = f"{d['text']} ({d['dash']:.1f})" if d else "1.5 × النقطة (لا شرطة في الشعارات)"
    gap_logo = max(logos, key=lambda l: len(l["text"]) - sum(c in "بتثنيیسشصضطظعغفقكلمهـ" for c in l["text"]))
    spec_vals["gap"] = gap_logo["group_gap"] or 1.15 * spec_vals["white"]
    src["gap"] = f"{gap_logo['text']} ({spec_vals['gap']:.1f})"
    # marks: above, their top on one line; under the line, a dot's top and a dash's centre
    above = marks("above_top")
    if above:
        spec_vals["ascender"] = statistics.median(x for x, _ in above)
        src["ascender"] = "قمة العلامات فوق الحروف: " + "، ".join(f"{t} {x:.1f}" for x, t in above)
    bd, bs = marks("below_dot_top"), marks("below_dash_centre")
    spec_vals["below_dot"] = statistics.median(x for x, _ in bd) if bd else 0.35 * spec_vals["dot"]
    src["below_dot"] = ("، ".join(f"{t} {x:.1f}" for x, t in bd)) if bd else "0.35 × النقطة (لا نقطة تحت السطر في الشعارات)"
    spec_vals["below_dash"] = statistics.median(x for x, _ in bs) if bs else spec_vals["below_dot"] + spec_vals["dot"] / 2
    src["below_dash"] = ("، ".join(f"{t} {x:.1f}" for x, t in bs)) if bs else "تحت النقطة"
    two = "dash" if any(l.get("dash") for l in logos) else "dots"
    src["two_dots"] = "شرطة كما في " + "، ".join(l["text"] for l in logos if l.get("dash")) if two == "dash" else "نقطتان"
    src["heads"] = "الرؤوس إلى ارتفاع الحروف كما في " + "، ".join(l["text"] for l in logos if any(c in l["text"] for c in "وفقم"))
    return Spec(**spec_vals, two_dots=two, heads="x-height", sources=src)


def main() -> int:
    p = argparse.ArgumentParser(prog="khatt.drawn.spec", description="the measures of the logos a drawing is made from")
    p.add_argument("--logo", nargs=2, action="append", metavar=("PICTURE", "TEXT"), required=True)
    p.add_argument("-o", "--out", default="spec.json")
    a = p.parse_args()
    logos = [measure_logo(Path(img).read_bytes(), text) for img, text in a.logo]
    for l in logos:
        print(l["text"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in l.items() if k not in ("text", "marks")},
              {k: [round(x, 1) for x in v] for k, v in l["marks"].items()})
    spec = combine(logos)
    Path(a.out).write_text(spec.to_json(), encoding="utf-8")
    print(spec.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
