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

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .style import Style

MEASURED, DERIVED, FITTED, BY_EYE = "مقيس", "مشتق", "مضبوط على الخطوط", "بالعين"
FROM_LOGO = "مقيس من حروف الشعار"


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


# Rules learned from the designer's edits (learning.py): a factor on a rule, applied to every logo.
# Kept outside the code, beside the log of edits, so that learning never rewrites a rule silently.
LEARNED: Dict[str, float] = {}


def load_learned(path=None) -> Dict[str, float]:
    import json
    import os
    from pathlib import Path
    p = Path(path) if path else Path(os.environ.get("KHATT_DATA", Path.home() / ".khatt")) / "learned.json"
    LEARNED.clear()
    if p.exists():
        try:
            LEARNED.update({k: float(v["factor"]) for k, v in json.loads(p.read_text(encoding="utf-8")).items() if k in PARAMS})
        except (ValueError, KeyError, TypeError):
            pass
    return LEARNED


# What a letter reads while it is built: the names of the parameters and variants it depends on.
# Set by typeset's cache, so that trying another value rebuilds only the letters that use it.
READS: ContextVar = ContextVar("khatt_reads", default=None)


# The letter being built (set by typeset): a parameter measured on the logo's letters serves only those
# letters; any other takes the rule turned by the logo's general manner.
BUILDING: ContextVar = ContextVar("khatt_letter", default=None)


def _read(key: str) -> None:
    r = READS.get()
    if r is not None:
        r.add(key)


def general_rule(st: Style, name: str) -> float:
    """The rule as written, with what has been learned from the designer's edits (no logo's measure)."""
    return float(PARAMS[name].rule(st)) * LEARNED.get(name, 1.0)


# The designer's general manner: kinds of rules that a designer turns together, whatever the letter.
# A logo that draws its cups wide draws its bowls and bases wide too. Measured on all the letters of
# the logo's word at once (selfcheck.fit_letters), so that they reach letters the logo does not have.
TRAITS: Dict[str, Tuple[str, List[str]]] = {
    "عرض": ("عرض أجسام الحروف: الكؤوس والقواعد والرؤوس والذيول",
            ["cup.width", "bowl.reach", "beh.width", "yeh.reach", "hah.span", "hah.reach", "dal.base", "kaf.arm",
             "kaf.width", "fa.tail", "ain.eye_w", "ain.head_w", "loop.length", "loop.ring_length"]),
    "ارتفاع": ("ارتفاع الأطراف والرؤوس نسبةً إلى ارتفاع الحروف",
               ["cup.arm", "cup.left", "beh.post", "noon.stem", "yeh.stem", "yeh.arm", "yeh.head", "reh.top", "ain.eye_h",
                "ain.head_h", "fa.end", "qaf.left", "kaf.post", "meem.height", "hah.nose_y", "dal.flag"]),
    "انفتاح": ("سعة الفراغ داخل الرؤوس والعيون", ["fa.counter", "meem.counter", "heh.counter", "heh.lower_w"]),
    "استدارة": ("استدارة الأركان والانعطافات", ["ain.round", "reh.corner", "hah.turn"]),
}
TRAIT_OF = {n: t for t, (_, names) in TRAITS.items() for n in names}
TRAIT_GRID = (0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.35, 1.5, 1.75)


def trait_rule(st: Style, name: str) -> float:
    """The rule as written, turned by the logo's general manner."""
    return general_rule(st, name) * st.traits.get(TRAIT_OF.get(name, ""), 1.0)


def rule_value(st: Style, name: str) -> float:
    """The rule's value for this logo. A letter the logo contains takes what was measured on it (or the
    rule, where nothing better was measured); a letter it does not contain takes the rule turned by the
    logo's general manner (TRAITS)."""
    v = st.fitted.get(name)
    cur = BUILDING.get()
    if st.measured_on and cur is not None:
        if SKELETON.get(cur, cur) in st.measured_on:          # a letter of the logo: its own measure
            return float(v) if v is not None else general_rule(st, name)
        return trait_rule(st, name)                           # any other letter: the logo's manner
    return float(v) if v is not None else trait_rule(st, name)


def value(st: Style, name: str) -> float:
    """P without noting the read (for comparing what a letter was built with)."""
    v = st.params.get(name)
    return float(v) if v is not None else rule_value(st, name)


def P(st: Style, name: str) -> float:
    """The value of a named parameter for this logo: the designer's, if set, else the rule's."""
    _read(name)
    v = st.params.get(name)
    return float(v) if v is not None else rule_value(st, name)


def pick(st: Style, *keys: str) -> Optional[str]:
    """The construction asked for a letter form (keys: "ه.I", "ه", ...): the designer's, else the one
    measured on the logo, else None (the builder's default)."""
    for key in keys:
        _read("v:" + key)
    for source in (st.variants, st.chosen):
        for key in keys:
            if source.get(key):
                return source[key]
    return None


# ------------------------------------------------------------------ the length of a letter (kashida)
STRETCH = "مد."                  # handle names for lengths: "مد.ك.I"
PARTNER = {"i": "F", "F": "i", "I": "M", "M": "I"}


def fitted_stretch(st: Style, key: str) -> float:
    """How much longer the logo draws a letter form than the engine builds it, as fitted on the logo's
    own word (selfcheck.fit_stretch). A form the logo did not show takes its partner's: isolated and
    final share a body, as do initial and medial, which share their join to the left."""
    if key in st.stretch:
        return float(st.stretch[key])
    letter, form = key.split(".")
    return float(st.stretch.get(f"{letter}.{PARTNER[form]}", 0.0))


def stretch_of(st: Style, key: str) -> float:
    v = st.params.get(STRETCH + key)
    return float(v) if v is not None else fitted_stretch(st, key)


def is_handle(name: str) -> bool:
    return name in PARAMS or (name.startswith(STRETCH) and name.count(".") == 2 and name[-1] in PARTNER)


def handle_rule(st: Style, name: str) -> float:
    """What a handle is worth before the designer moves it."""
    return fitted_stretch(st, name[len(STRETCH):]) if name.startswith(STRETCH) else rule_value(st, name)


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
    want = pick(st, f"{letter}.{form}", letter)
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


# Variants chosen inside a builder rather than by a registered function are declared here, so that
# they are listed and can be picked like the others.
DECLARED: Dict[str, Tuple[List[Tuple[str, str]], Callable[[Style], str]]] = {}


def declare(letter: str, forms: str, options: List[Tuple[str, str]], default: Callable[[Style], str]) -> None:
    for f in forms:
        DECLARED[f"{letter}.{f}"] = (options, default)


# which letters a parameter's shape belongs to: the editor shows a handle when its letter is in the word
SHAPE_LETTERS = {"الميم": "م", "العين": "عغ", "الفاء والقاف": "فڤق", "الهاء": "هة", "الياء": "يیىئ", "الحاء": "حجخچ",
                 "الكأس": "نلقسشصضيیىئ", "الكأس الراجع": "حجخچعغ", "العروة": "صضطظ", "الصاد": "صض", "الباء": "بتثپ", "النون": "ن",
                 "الدال": "دذ", "الراء": "رزژ", "الكاف": "كکگ", "النقاط": "بتثپنيیئجخچذزژشضظغفڤقة"}
SKELETON = {"غ": "ع", "ج": "ح", "خ": "ح", "چ": "ح", "ة": "ه", "ی": "ي", "ى": "ي", "ئ": "ي", "ڤ": "ف", "ق": "ف",
            "ض": "ص", "ظ": "ط", "ش": "س", "ذ": "د", "ز": "ر", "ژ": "ر", "ت": "ب", "ث": "ب", "پ": "ب"}


# ------------------------------------------------------------------ what a designer can move
def handles(st: Style) -> List[Dict]:
    """Every named parameter with its value for this logo, its rule and where the rule came from,
    and every letter form that has more than one construction."""
    out = []
    for p in PARAMS.values():
        rule = rule_value(st, p.name)
        span = [p.lo * rule, p.hi * rule] if rule > 0 else [0.0, 0.5 * st.x_height]     # a rule of 0 still has somewhere to go
        fitted = p.name in st.fitted
        out.append({"name": p.name, "shape": p.shape, "value": round(P(st, p.name), 3), "rule_value": round(rule, 3),
                    "rule": p.text + (f" (× {LEARNED[p.name]:.3g} مما تعلّمه من تعديلاتك)" if p.name in LEARNED else "")
                    + (f"؛ قيس من حروف الشعار: {rule:.3g} بدل {general_rule(st, p.name):.3g}" if fitted else "")
                    + (f"؛ × {st.traits[TRAIT_OF[p.name]]:.2f} من {TRAIT_OF[p.name]} الشعار" if not fitted and st.traits.get(TRAIT_OF.get(p.name, "")) else ""),
                    "source": FROM_LOGO if fitted else p.source, "range": [round(span[0], 3), round(span[1], 3)],
                    "set_by_designer": p.name in st.params})
    # the lengths the logo gives its letters, and any the designer set
    keys = list(st.stretch) + [n[len(STRETCH):] for n in st.params if n.startswith(STRETCH) and n[len(STRETCH):] not in st.stretch]
    for key in keys:
        letter, form = key.split(".")
        rule, name = fitted_stretch(st, key), STRETCH + key
        out.append({"name": name, "shape": "المدّ", "letters": letter, "form": form, "value": round(stretch_of(st, key), 3),
                    "rule_value": round(rule, 3), "source": MEASURED,
                    "rule": f"مدّ «{letter}» ({ {'i': 'منفصلًا', 'I': 'أولًا', 'M': 'وسطًا', 'F': 'آخرًا'}[form]}) كما يمدّه الشعار: "
                            + ("وصلته إلى ما بعده" if form in "IM" else "جسمه على السطر، أو البياض بعده"),
                    "range": [round(min(0.0, rule) if form in "IM" else -0.3 * st.x_height, 3), round(max(2 * rule, 1.5 * st.x_height), 3)],
                    "set_by_designer": name in st.params})
    return out


def choices(st: Style) -> List[Dict]:
    """Every letter form that has more than one construction, with the one in force for this logo."""
    out = []
    for key, options in VARIANTS.items():
        if len(options) < 2:
            continue
        letter, form = key.split(".")
        cur = choose(st, letter, form)
        out.append({"letter": letter, "form": form, "chosen": cur.name if cur else "",
                    "options": [{"name": v.name, "note": v.note} for v in options],
                    "set_by_designer": key in st.variants or letter in st.variants, "from_logo": key in st.chosen})
    partner = {"i": "F", "F": "i", "I": "M", "M": "I"}
    keys = list(st.taken) + [f"{k.split('.')[0]}.{partner[k.split('.')[1]]}" for k in st.taken
                             if f"{k.split('.')[0]}.{partner[k.split('.')[1]]}" not in st.taken]
    for key in keys:
        letter, form = key.split(".")
        out.append({"letter": letter, "form": form, "chosen": "المحرك" if st.variants.get(key) == "المحرك" else "من الشعار",
                    "options": [{"name": "من الشعار", "note": "الحرف كما رسمه الشعار، مقصوصًا منه"},
                                {"name": "المحرك", "note": "الحرف كما يبنيه المحرك بقواعد الشعار"}],
                    "set_by_designer": key in st.variants})
    for key, (options, default) in DECLARED.items():
        letter, form = key.split(".")
        want = pick(st, key, letter)
        names = [n for n, _ in options]
        out.append({"letter": letter, "form": form, "chosen": want if want in names else default(st),
                    "options": [{"name": n, "note": note} for n, note in options],
                    "set_by_designer": key in st.variants or letter in st.variants})
    return out
