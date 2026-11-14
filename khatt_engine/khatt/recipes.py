"""The table behind the letters: named parameters with their rules, and the variants of each letter form.

A letter used to be a function full of bare numbers (0.636 of the x-height, 1.78 pitches ...), each
tuned by eye on one logo. Here every such number has a name, a rule that derives it from what was
measured on the logo, and a note of where the rule came from:

  measured   read from the logo itself
  derived    follows from measured values by the stated rule
  fitted     the factor was chosen by the benchmark, over the open typefaces
  by eye     chosen by looking at one logo; a candidate for fitting

A designer (or anything sitting in the judge's seat) changes a letter without touching code:
`style.params[name] = value` replaces a rule, `style.variants["ه.I"] = "عينان"` picks another
construction of a letter form. `handles(style)` lists everything that can be moved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .style import Style

MEASURED, DERIVED, FITTED, BY_EYE = "مقيس", "مشتق", "مضبوط على الخطوط", "بالعين"


@dataclass
class Param:
    name: str
    rule: Callable[[Style], float]
    text: str                    # the rule, as a designer would say it
    source: str
    shape: str                   # the key shape or letter it belongs to
    lo: float = 0.5              # a handle may move the value between lo x rule and hi x rule
    hi: float = 2.0


PARAMS: Dict[str, Param] = {}


def param(name: str, rule: Callable[[Style], float], text: str, source: str, shape: str, lo: float = 0.5, hi: float = 2.0) -> str:
    PARAMS[name] = Param(name, rule, text, source, shape, lo, hi)
    return name


def P(st: Style, name: str) -> float:
    """The value of a named parameter for this logo: the designer's, if set, else the rule's."""
    v = st.params.get(name)
    return float(v) if v is not None else float(PARAMS[name].rule(st))


# ------------------------------------------------------------------ variants of a letter form
@dataclass
class Variant:
    name: str
    build: Callable                      # (style, form, mark) -> Glyph
    note: str
    when: Optional[Callable[[Style], bool]] = None     # the logos this variant is the default for


VARIANTS: Dict[str, List[Variant]] = {}


def variant(letter: str, forms: str, name: str, note: str = "", when: Optional[Callable[[Style], bool]] = None):
    """Register a construction of `letter` in each of `forms` (i, I, M, F). The first variant
    registered for a form whose `when` holds (or has none) is its default."""
    def deco(fn):
        for f in forms:
            VARIANTS.setdefault(f"{letter}.{f}", []).append(Variant(name, fn, note, when))
        return fn
    return deco


def choose(st: Style, letter: str, form: str) -> Optional[Variant]:
    """The variant in force for this logo: the designer's choice, else the first whose `when` holds."""
    options = VARIANTS.get(f"{letter}.{form}")
    if not options:
        return None
    want = st.variants.get(f"{letter}.{form}") or st.variants.get(letter)
    if want:
        for v in options:
            if v.name == want:
                return v
    for v in options:
        if v.when is None or v.when(st):
            return v
    return options[-1]


def curved(st: Style) -> bool:
    return st.angular < 0.5


def angular(st: Style) -> bool:
    return st.angular >= 0.5


# ------------------------------------------------------------------ what a designer can move
def handles(st: Style) -> List[Dict]:
    """Every named parameter with its value for this logo, its rule and where the rule came from,
    and every letter form that has more than one construction."""
    out = []
    for p in PARAMS.values():
        rule = float(p.rule(st))
        out.append({"name": p.name, "shape": p.shape, "value": round(P(st, p.name), 3), "rule_value": round(rule, 3),
                    "rule": p.text, "source": p.source, "range": [round(p.lo * rule, 3), round(p.hi * rule, 3)],
                    "set_by_designer": p.name in st.params})
    return out


def choices(st: Style) -> List[Dict]:
    out = []
    for key, options in VARIANTS.items():
        if len(options) < 2:
            continue
        letter, form = key.split(".")
        cur = choose(st, letter, form)
        out.append({"letter": letter, "form": form, "chosen": cur.name if cur else "",
                    "options": [{"name": v.name, "note": v.note} for v in options],
                    "set_by_designer": key in st.variants or letter in st.variants})
    return out
