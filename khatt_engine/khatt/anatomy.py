"""The anatomy of the Arabic script as type designers build a typeface: one source for the engine.

Every table here is taken from a source, read for it (see README, «الأساس المدروس»):
- Unicode 18.0, ArabicShaping.txt and the core specification 9.2: every letter's joining type and
  joining group, that is its skeleton, and its schematic name, that is its skeleton and its marks
  («DOTLESS BEH WITH 2 DOTS ABOVE» is ت); the four forms; the obligatory لا.
- W3C, Arabic & Persian Layout Requirements (alreq): the forms, لا and its variants, the kashida
  (the connection drawn longer, at chosen joins, once in a word), the tatweel.
- Microsoft, Developing OpenType Fonts for Arabic Script: the features and the order they apply in.
- Glyphs, Creating an Arabic font, and its handbook: the dotless basic shapes are drawn first and the
  letters composed of them and of mark components hung on anchors; every connection at one height,
  its top the joining line, as thick as the stroke; small overlaps where letters connect; exit and
  entry anchors for cursive attachment.
- TypeTogether, Arabic type anatomy (Azza Alameddine): the established terms head, body, eye, tooth,
  neck, cup, with ascender and descender (not x-height); classical books name alternates of a letter
  by width, shape and style (س compact, extended, halted; لا muhaqqaq or paper-like).
- 29LT, Arabic type anatomy (Pascal Zoghbi): Arabic has several levels instead of one x-height (tooth,
  loop, eye; two ascender and two or three descender levels); a Kufi needs four or five.
- iwantype, an Arabic foundry: what a professional Arabic typeface ships (weights, Arabic with Latin
  and sometimes Kurdish, OpenType alternates such as a ي with a shorter descender and round dots,
  extended letters, ligatures, the riyal sign) and how a font for a brand is made (key letters
  first, then the whole character set, tested in the brand's real uses).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------- 1. the letters (Unicode)
# char: (schematic name, joining type, joining group), every letter of the Arabic block U+0620..U+06FF,
# generated from Unicode's ArabicShaping-18.0.0.txt (© Unicode, Inc.; Unicode terms of use).
UCD = {
    "\u0620": ("KASHMIRI YEH", "D", "KASHMIRI YEH"),   # ؠ
    "\u0621": ("HAMZA", "U", "No_Joining_Group"),   # ء
    "\u0622": ("ALEF WITH MADDA ABOVE", "R", "ALEF"),   # آ
    "\u0623": ("ALEF WITH HAMZA ABOVE", "R", "ALEF"),   # أ
    "\u0624": ("WAW WITH HAMZA ABOVE", "R", "WAW"),   # ؤ
    "\u0625": ("ALEF WITH HAMZA BELOW", "R", "ALEF"),   # إ
    "\u0626": ("DOTLESS YEH WITH HAMZA ABOVE", "D", "YEH"),   # ئ
    "\u0627": ("ALEF", "R", "ALEF"),   # ا
    "\u0628": ("BEH", "D", "BEH"),   # ب
    "\u0629": ("TEH MARBUTA", "R", "TEH MARBUTA"),   # ة
    "\u062A": ("DOTLESS BEH WITH 2 DOTS ABOVE", "D", "BEH"),   # ت
    "\u062B": ("DOTLESS BEH WITH 3 DOTS ABOVE", "D", "BEH"),   # ث
    "\u062C": ("HAH WITH DOT BELOW", "D", "HAH"),   # ج
    "\u062D": ("HAH", "D", "HAH"),   # ح
    "\u062E": ("HAH WITH DOT ABOVE", "D", "HAH"),   # خ
    "\u062F": ("DAL", "R", "DAL"),   # د
    "\u0630": ("DAL WITH DOT ABOVE", "R", "DAL"),   # ذ
    "\u0631": ("REH", "R", "REH"),   # ر
    "\u0632": ("REH WITH DOT ABOVE", "R", "REH"),   # ز
    "\u0633": ("SEEN", "D", "SEEN"),   # س
    "\u0634": ("SEEN WITH 3 DOTS ABOVE", "D", "SEEN"),   # ش
    "\u0635": ("SAD", "D", "SAD"),   # ص
    "\u0636": ("SAD WITH DOT ABOVE", "D", "SAD"),   # ض
    "\u0637": ("TAH", "D", "TAH"),   # ط
    "\u0638": ("TAH WITH DOT ABOVE", "D", "TAH"),   # ظ
    "\u0639": ("AIN", "D", "AIN"),   # ع
    "\u063A": ("AIN WITH DOT ABOVE", "D", "AIN"),   # غ
    "\u063B": ("KEHEH WITH 2 DOTS ABOVE", "D", "GAF"),   # ػ
    "\u063C": ("KEHEH WITH 3 DOTS BELOW", "D", "GAF"),   # ؼ
    "\u063D": ("FARSI YEH WITH INVERTED V ABOVE", "D", "FARSI YEH"),   # ؽ
    "\u063E": ("FARSI YEH WITH 2 DOTS ABOVE", "D", "FARSI YEH"),   # ؾ
    "\u063F": ("FARSI YEH WITH 3 DOTS ABOVE", "D", "FARSI YEH"),   # ؿ
    "\u0640": ("TATWEEL", "C", "No_Joining_Group"),   # ـ
    "\u0641": ("FEH", "D", "FEH"),   # ف
    "\u0642": ("QAF", "D", "QAF"),   # ق
    "\u0643": ("KAF", "D", "KAF"),   # ك
    "\u0644": ("LAM", "D", "LAM"),   # ل
    "\u0645": ("MEEM", "D", "MEEM"),   # م
    "\u0646": ("NOON", "D", "NOON"),   # ن
    "\u0647": ("HEH", "D", "HEH"),   # ه
    "\u0648": ("WAW", "R", "WAW"),   # و
    "\u0649": ("DOTLESS YEH", "D", "YEH"),   # ى
    "\u064A": ("YEH", "D", "YEH"),   # ي
    "\u066E": ("DOTLESS BEH", "D", "BEH"),   # ٮ
    "\u066F": ("DOTLESS QAF", "D", "QAF"),   # ٯ
    "\u0671": ("ALEF WITH WASLA ABOVE", "R", "ALEF"),   # ٱ
    "\u0672": ("ALEF WITH WAVY HAMZA ABOVE", "R", "ALEF"),   # ٲ
    "\u0673": ("ALEF WITH WAVY HAMZA BELOW", "R", "ALEF"),   # ٳ
    "\u0674": ("HIGH HAMZA", "U", "No_Joining_Group"),   # ٴ
    "\u0675": ("HIGH HAMZA ALEF", "R", "ALEF"),   # ٵ
    "\u0676": ("HIGH HAMZA WAW", "R", "WAW"),   # ٶ
    "\u0677": ("HIGH HAMZA WAW WITH COMMA ABOVE", "R", "WAW"),   # ٷ
    "\u0678": ("HIGH HAMZA DOTLESS YEH", "D", "YEH"),   # ٸ
    "\u0679": ("DOTLESS BEH WITH TAH ABOVE", "D", "BEH"),   # ٹ
    "\u067A": ("DOTLESS BEH WITH VERTICAL 2 DOTS ABOVE", "D", "BEH"),   # ٺ
    "\u067B": ("DOTLESS BEH WITH VERTICAL 2 DOTS BELOW", "D", "BEH"),   # ٻ
    "\u067C": ("DOTLESS BEH WITH ATTACHED RING BELOW AND 2 DOTS ABOVE", "D", "BEH"),   # ټ
    "\u067D": ("DOTLESS BEH WITH INVERTED 3 DOTS ABOVE", "D", "BEH"),   # ٽ
    "\u067E": ("DOTLESS BEH WITH 3 DOTS BELOW", "D", "BEH"),   # پ
    "\u067F": ("DOTLESS BEH WITH 4 DOTS ABOVE", "D", "BEH"),   # ٿ
    "\u0680": ("DOTLESS BEH WITH 4 DOTS BELOW", "D", "BEH"),   # ڀ
    "\u0681": ("HAH WITH HAMZA ABOVE", "D", "HAH"),   # ځ
    "\u0682": ("HAH WITH VERTICAL 2 DOTS ABOVE", "D", "HAH"),   # ڂ
    "\u0683": ("HAH WITH 2 DOTS BELOW", "D", "HAH"),   # ڃ
    "\u0684": ("HAH WITH VERTICAL 2 DOTS BELOW", "D", "HAH"),   # ڄ
    "\u0685": ("HAH WITH 3 DOTS ABOVE", "D", "HAH"),   # څ
    "\u0686": ("HAH WITH 3 DOTS BELOW", "D", "HAH"),   # چ
    "\u0687": ("HAH WITH 4 DOTS BELOW", "D", "HAH"),   # ڇ
    "\u0688": ("DAL WITH TAH ABOVE", "R", "DAL"),   # ڈ
    "\u0689": ("DAL WITH ATTACHED RING BELOW", "R", "DAL"),   # ډ
    "\u068A": ("DAL WITH DOT BELOW", "R", "DAL"),   # ڊ
    "\u068B": ("DAL WITH DOT BELOW AND TAH ABOVE", "R", "DAL"),   # ڋ
    "\u068C": ("DAL WITH 2 DOTS ABOVE", "R", "DAL"),   # ڌ
    "\u068D": ("DAL WITH 2 DOTS BELOW", "R", "DAL"),   # ڍ
    "\u068E": ("DAL WITH 3 DOTS ABOVE", "R", "DAL"),   # ڎ
    "\u068F": ("DAL WITH INVERTED 3 DOTS ABOVE", "R", "DAL"),   # ڏ
    "\u0690": ("DAL WITH 4 DOTS ABOVE", "R", "DAL"),   # ڐ
    "\u0691": ("REH WITH TAH ABOVE", "R", "REH"),   # ڑ
    "\u0692": ("REH WITH V ABOVE", "R", "REH"),   # ڒ
    "\u0693": ("REH WITH ATTACHED RING BELOW", "R", "REH"),   # ړ
    "\u0694": ("REH WITH DOT BELOW", "R", "REH"),   # ڔ
    "\u0695": ("REH WITH V BELOW", "R", "REH"),   # ڕ
    "\u0696": ("REH WITH DOT BELOW AND DOT WITHIN", "R", "REH"),   # ږ
    "\u0697": ("REH WITH 2 DOTS ABOVE", "R", "REH"),   # ڗ
    "\u0698": ("REH WITH 3 DOTS ABOVE", "R", "REH"),   # ژ
    "\u0699": ("REH WITH 4 DOTS ABOVE", "R", "REH"),   # ڙ
    "\u069A": ("SEEN WITH DOT BELOW AND DOT ABOVE", "D", "SEEN"),   # ښ
    "\u069B": ("SEEN WITH 3 DOTS BELOW", "D", "SEEN"),   # ڛ
    "\u069C": ("SEEN WITH 3 DOTS BELOW AND 3 DOTS ABOVE", "D", "SEEN"),   # ڜ
    "\u069D": ("SAD WITH 2 DOTS BELOW", "D", "SAD"),   # ڝ
    "\u069E": ("SAD WITH 3 DOTS ABOVE", "D", "SAD"),   # ڞ
    "\u069F": ("TAH WITH 3 DOTS ABOVE", "D", "TAH"),   # ڟ
    "\u06A0": ("AIN WITH 3 DOTS ABOVE", "D", "AIN"),   # ڠ
    "\u06A1": ("DOTLESS FEH", "D", "FEH"),   # ڡ
    "\u06A2": ("DOTLESS FEH WITH DOT BELOW", "D", "FEH"),   # ڢ
    "\u06A3": ("FEH WITH DOT BELOW", "D", "FEH"),   # ڣ
    "\u06A4": ("DOTLESS FEH WITH 3 DOTS ABOVE", "D", "FEH"),   # ڤ
    "\u06A5": ("DOTLESS FEH WITH 3 DOTS BELOW", "D", "FEH"),   # ڥ
    "\u06A6": ("DOTLESS FEH WITH 4 DOTS ABOVE", "D", "FEH"),   # ڦ
    "\u06A7": ("DOTLESS QAF WITH DOT ABOVE", "D", "QAF"),   # ڧ
    "\u06A8": ("DOTLESS QAF WITH 3 DOTS ABOVE", "D", "QAF"),   # ڨ
    "\u06A9": ("KEHEH", "D", "GAF"),   # ک
    "\u06AA": ("SWASH KAF", "D", "SWASH KAF"),   # ڪ
    "\u06AB": ("KEHEH WITH ATTACHED RING BELOW", "D", "GAF"),   # ګ
    "\u06AC": ("KAF WITH DOT ABOVE", "D", "KAF"),   # ڬ
    "\u06AD": ("KAF WITH 3 DOTS ABOVE", "D", "KAF"),   # ڭ
    "\u06AE": ("KAF WITH 3 DOTS BELOW", "D", "KAF"),   # ڮ
    "\u06AF": ("GAF", "D", "GAF"),   # گ
    "\u06B0": ("GAF WITH ATTACHED RING BELOW", "D", "GAF"),   # ڰ
    "\u06B1": ("GAF WITH 2 DOTS ABOVE", "D", "GAF"),   # ڱ
    "\u06B2": ("GAF WITH 2 DOTS BELOW", "D", "GAF"),   # ڲ
    "\u06B3": ("GAF WITH VERTICAL 2 DOTS BELOW", "D", "GAF"),   # ڳ
    "\u06B4": ("GAF WITH 3 DOTS ABOVE", "D", "GAF"),   # ڴ
    "\u06B5": ("LAM WITH V ABOVE", "D", "LAM"),   # ڵ
    "\u06B6": ("LAM WITH DOT ABOVE", "D", "LAM"),   # ڶ
    "\u06B7": ("LAM WITH 3 DOTS ABOVE", "D", "LAM"),   # ڷ
    "\u06B8": ("LAM WITH 3 DOTS BELOW", "D", "LAM"),   # ڸ
    "\u06B9": ("NOON WITH DOT BELOW", "D", "NOON"),   # ڹ
    "\u06BA": ("DOTLESS NOON", "D", "NOON"),   # ں
    "\u06BB": ("DOTLESS NOON WITH TAH ABOVE", "D", "NOON"),   # ڻ
    "\u06BC": ("NOON WITH ATTACHED RING BELOW", "D", "NOON"),   # ڼ
    "\u06BD": ("NYA", "D", "NYA"),   # ڽ
    "\u06BE": ("KNOTTED HEH", "D", "KNOTTED HEH"),   # ھ
    "\u06BF": ("HAH WITH 3 DOTS BELOW AND DOT ABOVE", "D", "HAH"),   # ڿ
    "\u06C0": ("DOTLESS TEH MARBUTA WITH HAMZA ABOVE", "R", "TEH MARBUTA"),   # ۀ
    "\u06C1": ("HEH GOAL", "D", "HEH GOAL"),   # ہ
    "\u06C2": ("HEH GOAL WITH HAMZA ABOVE", "D", "HEH GOAL"),   # ۂ
    "\u06C3": ("TEH MARBUTA GOAL", "R", "TEH MARBUTA GOAL"),   # ۃ
    "\u06C4": ("WAW WITH ATTACHED RING WITHIN", "R", "WAW"),   # ۄ
    "\u06C5": ("WAW WITH LOOP", "R", "WAW"),   # ۅ
    "\u06C6": ("WAW WITH V ABOVE", "R", "WAW"),   # ۆ
    "\u06C7": ("WAW WITH COMMA ABOVE", "R", "WAW"),   # ۇ
    "\u06C8": ("WAW WITH ALEF ABOVE", "R", "WAW"),   # ۈ
    "\u06C9": ("WAW WITH INVERTED V ABOVE", "R", "WAW"),   # ۉ
    "\u06CA": ("WAW WITH 2 DOTS ABOVE", "R", "WAW"),   # ۊ
    "\u06CB": ("WAW WITH 3 DOTS ABOVE", "R", "WAW"),   # ۋ
    "\u06CC": ("FARSI YEH", "D", "FARSI YEH"),   # ی
    "\u06CD": ("YEH WITH TAIL", "R", "YEH WITH TAIL"),   # ۍ
    "\u06CE": ("FARSI YEH WITH V ABOVE", "D", "FARSI YEH"),   # ێ
    "\u06CF": ("WAW WITH DOT ABOVE", "R", "WAW"),   # ۏ
    "\u06D0": ("DOTLESS YEH WITH VERTICAL 2 DOTS BELOW", "D", "YEH"),   # ې
    "\u06D1": ("DOTLESS YEH WITH 3 DOTS BELOW", "D", "YEH"),   # ۑ
    "\u06D2": ("YEH BARREE", "R", "YEH BARREE"),   # ے
    "\u06D3": ("YEH BARREE WITH HAMZA ABOVE", "R", "YEH BARREE"),   # ۓ
    "\u06D5": ("DOTLESS TEH MARBUTA", "R", "TEH MARBUTA"),   # ە
    "\u06DD": ("ARABIC END OF AYAH", "U", "No_Joining_Group"),   # ۝
    "\u06EE": ("DAL WITH INVERTED V ABOVE", "R", "DAL"),   # ۮ
    "\u06EF": ("REH WITH INVERTED V ABOVE", "R", "REH"),   # ۯ
    "\u06FA": ("SEEN WITH DOT BELOW AND 3 DOTS ABOVE", "D", "SEEN"),   # ۺ
    "\u06FB": ("SAD WITH DOT BELOW AND DOT ABOVE", "D", "SAD"),   # ۻ
    "\u06FC": ("AIN WITH DOT BELOW AND DOT ABOVE", "D", "AIN"),   # ۼ
    "\u06FF": ("KNOTTED HEH WITH INVERTED V ABOVE", "D", "KNOTTED HEH"),   # ۿ
}

# the base letters whose names carry their marks unsaid (BEH is the dotless beh with a dot below). The
# second stroke of گ is a mark too: Unicode puts ک and گ in one joining group, Glyphs composes گ of ک and
# a «sarkash» component, and seven of eight typefaces draw it apart from the body (the anatomy test).
IMPLIED_MARKS: Dict[str, Tuple[str, ...]] = {
    "BEH": ("DOT BELOW",), "NOON": ("DOT ABOVE",), "FEH": ("DOT ABOVE",), "QAF": ("2 DOTS ABOVE",),
    "YEH": ("2 DOTS BELOW",), "TEH MARBUTA": ("2 DOTS ABOVE",), "TEH MARBUTA GOAL": ("2 DOTS ABOVE",),
    "GAF": ("SARKASH ABOVE",), "NYA": ("3 DOTS ABOVE",),
}
# what in a schematic name is a mark and not part of the skeleton
_PLACES = ("ABOVE", "BELOW", "CENTRE", "CENTER")


def _parse(name: str) -> Tuple[str, Tuple[str, ...]]:
    """(skeleton word, marks) of a schematic name: «DOTLESS BEH WITH 2 DOTS ABOVE» -> ("BEH", ("2 DOTS ABOVE",))."""
    base, _, rest = name.partition(" WITH ")
    high = base.startswith("HIGH HAMZA ")          # ٵ ٶ ٷ ٸ: a letter and the high hamza (Unicode 9.2)
    base = base.replace("HIGH HAMZA ", "")
    dotless = base.startswith("DOTLESS ")
    base = base.replace("DOTLESS ", "")
    marks = tuple(m.strip() for m in rest.replace(" AND ", ",").split(",") if m.strip()) if rest else ()
    if not dotless and not rest:
        marks = IMPLIED_MARKS.get(base, ())
    return base, marks + (("HIGH HAMZA",) if high else ())


def joining(c: str) -> str:
    """D dual-joining, R right-joining, U non-joining, C join-causing (tatweel), T transparent (marks)."""
    if c == "ـ":
        return "C"
    if c in UCD:
        return UCD[c][1]
    if c in MARKS:
        return "T"
    return "U"


def group(c: str) -> str:
    return UCD[c][2] if c in UCD else "No_Joining_Group"


def marks(c: str, form: str = "i") -> Tuple[str, ...]:
    """The marks of `c` in `form`, in Unicode's words. ی (FARSI YEH) has its two dots only when joined
    to the next letter (Unicode 9.2, table 9-11)."""
    if c not in UCD:
        return ()
    m = _parse(UCD[c][0])[1]
    if group(c) == "FARSI YEH" and form in ("i", "F"):
        return tuple(x for x in m if "BELOW" not in x or "DOTS" not in x) if UCD[c][0] != "FARSI YEH" else ()
    if group(c) == "FARSI YEH" and UCD[c][0] == "FARSI YEH":
        return ("2 DOTS BELOW",)
    if group(c) == "NYA":                # Jawi nya: joined, its dots go below, as پ's (Unicode 9.2)
        return ("3 DOTS BELOW",) if form in ("I", "M") else m
    return m


def forms(c: str) -> str:
    """The positional forms a letter takes: i isolated, I initial, M medial, F final."""
    return {"D": "iIMF", "R": "iF", "L": "iI", "C": "iIMF"}.get(joining(c), "i")


# ---------------------------------------------------------------- 2. the skeleton in each form
# Unicode groups letters by the skeleton they share in every form; type designers draw one basic
# shape for a skeleton in a form, and some groups share it in some forms (Glyphs' basic shapes):
# joined to the next letter, ب ن ي ی are one tooth, ف ق one head on its neck; ة is ه with two dots.
TOOTH_WHEN_JOINED = {"BEH", "NOON", "YEH", "FARSI YEH", "AFRICAN NOON", "NYA", "KASHMIRI YEH",
                     "BURUSHASKI YEH BARREE", "THIN YEH"}
FEH_WHEN_JOINED = {"FEH", "QAF", "AFRICAN FEH", "AFRICAN QAF"}
# measured on eight typefaces (the anatomy test): joined, ك is drawn on the body of ک and گ (Unicode keeps
# KAF and GAF apart for their isolated forms, where ك has its own swash). ة alone is drawn on the
# isolated ه in five of the eight, and apart in three (Noto Naskh, Noto Kufi, Reem Kufi): an alternate.
KAF_WHEN_JOINED = {"KAF", "GAF", "SWASH KAF"}
# groups that are another group's shape with marks Unicode keeps apart
SAME_SHAPE = {"FARSI YEH": "YEH", "TEH MARBUTA": "HEH", "TEH MARBUTA GOAL": "HEH GOAL", "NYA": "NOON"}


def skeleton(c: str, form: str) -> str:
    """The basic shape `c` is drawn on in `form` (a Unicode joining group, merged where the shapes are one)."""
    g = group(c)
    if c == "ء":
        return "HAMZA"
    if form in ("I", "M") and g in TOOTH_WHEN_JOINED:
        return "BEH"
    if form in ("I", "M") and g in FEH_WHEN_JOINED:
        return "FEH"
    if form in ("I", "M") and g in KAF_WHEN_JOINED:
        return "KAF"
    return SAME_SHAPE.get(g, g)


def same_body(a: str, b: str, form: str) -> bool:
    """`a` and `b` are drawn on one basic shape in `form`: only their marks tell them apart."""
    return a == b or (a in UCD and b in UCD and skeleton(a, form) == skeleton(b, form))


def siblings(c: str, form: str, among: Optional[Sequence[str]] = None) -> List[str]:
    """The other letters drawn on the same basic shape as `c` in `form`."""
    pool = among if among is not None else UCD
    return [d for d in pool if d != c and d in UCD and form in forms(d) and same_body(c, d, form)]


# the letter each basic shape is named by: the plainest of its letters (ق joined is named ف, ن joined ب)
PLAIN = "ابحدرسصطعفقكلمنهوي" + "کء"


def representative(c: str, form: str, among: Optional[Sequence[str]] = None) -> str:
    """The letter that names the basic shape `c` is drawn on in `form`: «ب» for ن ي ت joined, «ف» for ق
    joined, «ي» for ى ی ئ alone, «ه» for ة, «ا» for أ إ آ; `c` itself if it shares its shape with none."""
    if c not in UCD:
        return c
    for d in PLAIN:
        if (among is None or d in among) and form in forms(d) and same_body(c, d, form):
            return d
    return c


def basic_shapes(letters: Sequence[str]) -> List[Tuple[str, str]]:
    """The (skeleton, form) pairs a typeface for `letters` has to draw: everything else is a mark."""
    out = sorted({(skeleton(c, f), f) for c in letters for f in forms(c) if c in UCD})
    return out


# ---------------------------------------------------------------- 3. the parts, in the established terms
# TypeTogether: «the established terminology in Arabic type anatomy is head, body, eye, tooth, neck,
# cup», with ascender and descender; «some technical terms could be borrowed from Latin» (arm). The
# connection is the stroke on the joining line (Glyphs, alreq); the counter inside a head is its eye.
TERMS: Dict[str, str] = {
    "head": "رأس", "body": "جسم", "eye": "عين", "tooth": "سن", "neck": "عنق", "cup": "كأس",
    "ascender": "صاعد", "descender": "نازل", "arm": "ذراع", "connection": "وصلة",
}
# the parts of each basic shape in each form (the connection is added where the form joins); a cup
# "open right" is the back cup of ح and ع, a descender "returning" the tail of ى that turns back
_P = Dict[str, Tuple[str, ...]]
PARTS: Dict[str, _P] = {
    "ALEF": {"i": ("ascender",), "F": ("ascender",)},
    "BEH": {"i": ("body",), "I": ("tooth",), "M": ("tooth",), "F": ("body",)},
    "NOON": {"i": ("cup",), "F": ("cup",)},
    "YEH": {"i": ("descender returning",), "F": ("descender returning",)},
    "HAH": {"i": ("head", "cup open right"), "I": ("head",), "M": ("head",), "F": ("head", "cup open right")},
    "DAL": {"i": ("head", "body"), "F": ("head", "body")},
    "REH": {"i": ("descender",), "F": ("descender",)},
    "SEEN": {"i": ("tooth", "tooth", "cup"), "I": ("tooth", "tooth", "tooth"), "M": ("tooth", "tooth", "tooth"),
             "F": ("tooth", "tooth", "cup")},
    "SAD": {"i": ("head", "eye", "tooth", "cup"), "I": ("head", "eye", "tooth"), "M": ("head", "eye", "tooth"),
            "F": ("head", "eye", "tooth", "cup")},
    "TAH": {f: ("head", "eye", "ascender") for f in "iIMF"},
    "AIN": {"i": ("head", "cup open right"), "I": ("head",), "M": ("head", "eye"), "F": ("head", "eye", "cup open right")},
    "FEH": {"i": ("head", "eye", "body"), "I": ("head", "eye", "neck"), "M": ("head", "eye", "neck"),
            "F": ("head", "eye", "body")},
    "QAF": {"i": ("head", "eye", "cup"), "F": ("head", "eye", "cup")},
    "KAF": {f: ("ascender", "arm", "body") for f in "iIMF"},
    "GAF": {f: ("ascender", "arm", "arm", "body") for f in "iIMF"},
    "LAM": {"i": ("ascender", "cup"), "I": ("ascender",), "M": ("ascender",), "F": ("ascender", "cup")},
    "MEEM": {"i": ("head", "eye", "descender"), "I": ("head", "eye"), "M": ("head", "eye"), "F": ("head", "eye", "descender")},
    "HEH": {"i": ("head", "eye"), "I": ("head", "eye", "eye"), "M": ("head", "eye", "eye"), "F": ("head", "eye")},
    "WAW": {"i": ("head", "eye", "descender"), "F": ("head", "eye", "descender")},
    "LAM-ALEF": {"i": ("ascender", "ascender", "body"), "F": ("ascender", "ascender", "body")},
    "HAMZA": {"i": ("head", "body")},
}


def parts(c: str, form: str) -> Tuple[str, ...]:
    """The parts `c` is built of in `form`, the connection included where it joins."""
    sk = "LAM-ALEF" if len(c) == 2 and c[0] == "ل" else skeleton(c, form)
    p = PARTS.get(sk, {}).get(form, ())
    if form in ("I", "M"):
        p = p + ("connection out",)
    if form in ("M", "F"):
        p = p + ("connection in",)
    return p


def letters_with(part: str, letters: Sequence[str]) -> List[str]:
    """«letter.form» of every letter form that has `part` (a term, or a term with its qualifier)."""
    return [f"{c}.{f}" for c in letters for f in forms(c) if any(q == part or q.split()[0] == part for q in parts(c, f))]


# ---------------------------------------------------------------- 4. the vertical levels (29LT, TypeTogether)
# name: (Arabic, what sets it). Not x-height: Arabic has several levels, a Kufi four or five.
LEVELS: Dict[str, Tuple[str, str]] = {
    "baseline": ("السطر", "the bottom of the connection"),
    "joining": ("خط الوصل", "the top of the connection (as thick as the stroke)"),
    "tooth": ("ارتفاع السن", "the top of ب's tooth when joined"),
    "loop": ("ارتفاع الرأس", "the top of the heads of ف ق م و"),
    "eye": ("ارتفاع العين", "the top of the counter in a head"),
    "ascender": ("الصاعد", "the top of ا and ل"),
    "descender": ("النازل", "the lowest point of ر و and of the cups"),
    "marks above": ("النقاط فوق", "where dots above sit"),
    "marks below": ("النقاط تحت", "where dots below sit"),
}

# ---------------------------------------------------------------- 5. marks: components hung on anchors (Glyphs)
MARK_ANCHOR: Dict[str, str] = {}
for _m in ("DOT ABOVE", "2 DOTS ABOVE", "3 DOTS ABOVE", "INVERTED 3 DOTS ABOVE", "4 DOTS ABOVE", "VERTICAL 2 DOTS ABOVE",
           "HAMZA ABOVE", "MADDA ABOVE", "SMALL V", "SMALL TAH ABOVE", "TWO DOTS VERTICALLY ABOVE", "WASLA",
           "SARKASH ABOVE", "HIGH HAMZA"):
    MARK_ANCHOR[_m] = "top"
for _m in ("DOT BELOW", "2 DOTS BELOW", "3 DOTS BELOW", "4 DOTS BELOW", "VERTICAL 2 DOTS BELOW", "HAMZA BELOW",
           "SMALL V BELOW", "ATTACHED RING BELOW", "RING"):
    MARK_ANCHOR[_m] = "bottom"


def anchor_of(mark: str) -> str:
    """Where a mark hangs: on the letter's top anchor or its bottom one."""
    return MARK_ANCHOR.get(mark, "bottom" if "BELOW" in mark else "top")


# the vowel and other marks a professional font positions on its letters (mark, mkmk)
MARKS: Dict[str, str] = {
    "ً": "fathatan", "ٌ": "dammatan", "ٍ": "kasratan", "َ": "fatha", "ُ": "damma",
    "ِ": "kasra", "ّ": "shadda", "ْ": "sukun", "ٓ": "maddah above", "ٔ": "hamza above",
    "ٕ": "hamza below", "ٰ": "superscript alef",
}

# ---------------------------------------------------------------- 6. the OpenType features, in order (Microsoft)
FEATURES: List[Tuple[str, str]] = [
    ("ccmp", "compose or decompose (alef + hamza above into أ)"),
    ("isol", "isolated forms"), ("fina", "final forms"), ("medi", "medial forms"), ("init", "initial forms"),
    ("rlig", "obligatory ligatures: لا لأ لإ لآ"),
    ("calt", "contextual alternates: better connections, a letter shaped by its neighbour; joined teeth "
             "of ب ت ث ن ي raised or lowered by their neighbours so that they stay apart (alreq 4.2.2, 4.2.5: "
             "Naskh typefaces such as Amiri and Noto Naskh draw the initial ن and ي unlike the initial ب)"),
    ("liga", "optional ligatures (لم ...)"), ("dlig", "discretionary ligatures"),
    ("curs", "cursive attachment: each exit anchor to the next entry anchor"),
    ("kern", "pair kerning"), ("mark", "marks on their letters"), ("mkmk", "marks on marks"),
]
# what professional Arabic typefaces add (iwantype; TypeTogether's classical alternates)
ALTERNATES: List[Tuple[str, str]] = [
    ("ي بنزول أقل", "a ي with a shorter descender, for an even line (iwantype)"),
    ("نقاط مستديرة", "round dots as an alternate set (iwantype)"),
    ("حروف مرسلة", "extended letters, typed with ءء after the letter (iwantype)"),
    ("س مجموعة، مبسوطة، موقوفة", "compact, extended, halted seen (TypeTogether)"),
    ("لا محققة، ورقية", "the muhaqqaq and the paper-like lam-alef (TypeTogether)"),
    ("رمز الريال", "the riyal sign from ر.س (iwantype)"),
    # where real typefaces draw letters of one basic shape apart (the anatomy test, eight typefaces)
    ("أسنان سياقية", "joined teeth of ب ت ث ن ي raised or lowered by their neighbours (Noto Naskh, Amiri, "
                     "Vazirmatn, Tajawal): calt"),
    ("تاء مربوطة منفصلة", "ة alone drawn apart from the isolated ه (Noto Naskh, Noto Kufi, Reem Kufi)"),
    ("كأس الجيم", "the bowl of ج خ alone or last drawn apart from ح's, to make room for the dot (Noto Kufi, "
                  "Reem Kufi, Kufam)"),
]

# ---------------------------------------------------------------- 7. the character set
ARABIC_CORE = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"
PERSIAN = "پچژکگی"
URDU = "ٹڈڑںھہے"
KURDISH = "ڤۆێڕڵە"
DIGITS = "٠١٢٣٤٥٦٧٨٩" + "۰۱۲۳۴۵۶۷۸۹"
PUNCTUATION = "،؛؟٪٫٬۔«»﴾﴿"
SYMBOLS = "ـ‌‍◌﷼"   # tatweel, ZWNJ, ZWJ, dotted circle, riyal
LAM_ALEFS = ["لا", "لأ", "لإ", "لآ"]


def charset(level: str = "arabic") -> Dict[str, str]:
    """What a typeface covers: «arabic» (Arabic with its digits, punctuation, marks), «extended» (with
    Persian, Urdu and Kurdish)."""
    out = {"letters": ARABIC_CORE + "ٱ", "lam-alef": "".join(LAM_ALEFS), "digits": DIGITS,
           "punctuation": PUNCTUATION, "marks": "".join(MARKS), "symbols": SYMBOLS}
    if level == "extended":
        out["letters"] += PERSIAN + URDU + KURDISH
    return out


# ---------------------------------------------------------------- 8. the rules of a professional typeface
# rule: (Arabic, how it is measured). professional.py measures each on a style.
RULES: List[Tuple[str, str, str]] = [
    ("coverage", "التغطية", "every letter in its forms, لا, the marks, digits and punctuation are drawn"),
    ("levels", "المستويات", "teeth at one height, heads at one, ascenders at one, descenders at one or two"),
    ("stroke", "سمك واحد", "upright strokes one width, level strokes one width, everywhere"),
    ("joins", "الوصل", "every connection on one joining line, as thick as the stroke, overlapping, unbroken"),
    ("kashida", "الكشيدة", "the connection drawn longer, once in a word, at a chosen join"),
    ("lam-alef", "لا", "lam followed by alef is one letter"),
    ("marks", "النقاط", "dots one size, centred on their anchors, clear of letters and of each other"),
    ("counters", "الفراغات", "every eye stays open at the size it is used"),
    ("identity", "هوية الحرف", "each letter reads as itself and not as its neighbours (ف/م، ق/و، ح/ب، ع/ب، ي/ن)"),
    ("spacing", "المسافات", "letters that do not join stand one white apart; words one space apart"),
    ("color", "اللون", "even ink across a line of text"),
]
