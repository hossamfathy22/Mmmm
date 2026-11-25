"""What the engine knows about Arabic letters: how they join, which body each one shares, its dots and
marks, and the few facts of anatomy it uses to read a drawing (Unicode ArabicShaping; standard naming)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

TATWEEL = "ـ"
LAM = "ل"
ALEFS = {"ا": "", "أ": "HamzaAbove", "إ": "HamzaBelow", "آ": "Madda"}

# glyph name, body class, dots (count, above?) , joins left (dual)
LETTERS: Dict[str, Tuple[str, str, int, bool, bool]] = {
    "ا": ("alef", "alef", 0, True, False), "أ": ("alefHamzaAbove", "alef", 0, True, False),
    "إ": ("alefHamzaBelow", "alef", 0, True, False), "آ": ("alefMadda", "alef", 0, True, False),
    "ب": ("beh", "beh", 1, False, True), "ت": ("teh", "beh", 2, True, True),
    "ث": ("theh", "beh", 3, True, True), "پ": ("peh", "beh", 3, False, True),
    "ن": ("noon", "noon", 1, True, True), "ي": ("yeh", "yeh", 2, False, True),
    "ی": ("farsiYeh", "yeh", 0, False, True), "ى": ("alefMaksura", "yeh", 0, False, True),
    "ئ": ("yehHamza", "yeh", 0, False, True),
    "ج": ("jeem", "hah", 1, False, True), "ح": ("hah", "hah", 0, False, True),
    "خ": ("khah", "hah", 1, True, True), "چ": ("tcheh", "hah", 3, False, True),
    "د": ("dal", "dal", 0, True, False), "ذ": ("thal", "dal", 1, True, False),
    "ر": ("reh", "reh", 0, True, False), "ز": ("zain", "reh", 1, True, False),
    "ژ": ("jeh", "reh", 3, True, False),
    "س": ("seen", "seen", 0, True, True), "ش": ("sheen", "seen", 3, True, True),
    "ص": ("sad", "sad", 0, True, True), "ض": ("dad", "sad", 1, True, True),
    "ط": ("tah", "tah", 0, True, True), "ظ": ("zah", "tah", 1, True, True),
    "ع": ("ain", "ain", 0, True, True), "غ": ("ghain", "ain", 1, True, True),
    "ف": ("feh", "feh", 1, True, True), "ق": ("qaf", "qaf", 2, True, True),
    "ك": ("kaf", "kaf", 0, True, True), "ک": ("keheh", "keheh", 0, True, True),
    "گ": ("gaf", "gaf", 0, True, True), "ل": ("lam", "lam", 0, True, True),
    "م": ("meem", "meem", 0, True, True), "ه": ("heh", "heh", 0, True, True),
    "ة": ("tehMarbuta", "heh", 2, True, False),
    "و": ("waw", "waw", 0, True, False), "ؤ": ("wawHamza", "waw", 0, True, False),
    "ء": ("hamza", "hamza", 0, True, False),
}
RIGHT_ONLY = {c for c, v in LETTERS.items() if not v[4]} - {"ء"}
NON_JOINING = {"ء"}
FORMS = ("isol", "init", "medi", "fina")

# Persian yeh has its two dots only where it joins on the left
DOTS_JOINED_ONLY = {"ی": (2, False)}
# the bodies whose joined forms (initial, medial) are the tooth
TOOTH = {"beh", "noon", "yeh"}
# initial/medial forms are a qaf on feh's body
SHARES_JOINED = {"qaf": "feh", "gaf": "keheh"}
# letters a reader expects to go below the baseline when they end a word
DESCENDS = {"reh", "waw", "yeh", "qaf", "ain", "hah"}
# baseline-only stretches inside a joined (initial or medial) letter, as drawn with teeth: س has three
INTERNAL_JOINED = {"seen": 2}


def name(ch: str) -> str:
    return LETTERS[ch][0]


def body(ch: str) -> str:
    return LETTERS[ch][1]


def units(token: str) -> List[str]:
    """Letters of a token; لا (and لأ لإ لآ) is one unit, a tatweel is kept to mark a join."""
    out, i = [], 0
    while i < len(token):
        c = token[i]
        if c == LAM and i + 1 < len(token) and token[i + 1] in ALEFS:
            out.append(token[i:i + 2])
            i += 2
            continue
        out.append(c)
        i += 1
    return out


def can_join_left(u: str) -> bool:
    if u == TATWEEL:
        return True
    if len(u) == 2:                    # lam-alef ends with the alef
        return False
    return u in LETTERS and LETTERS[u][4]


def can_join_right(u: str) -> bool:
    if u == TATWEEL:
        return True
    if len(u) == 2:
        return True
    return u in LETTERS and u not in NON_JOINING


def forms_of(token: str) -> List[Tuple[str, str]]:
    """(unit, form) for each letter in reading order; a tatweel only gives its neighbour a join."""
    us = units(token)
    out = []
    for i, u in enumerate(us):
        if u == TATWEEL:
            continue
        prev = i > 0 and can_join_left(us[i - 1]) and can_join_right(u)
        nxt = i < len(us) - 1 and can_join_left(u) and can_join_right(us[i + 1])
        form = {(False, False): "isol", (False, True): "init", (True, True): "medi", (True, False): "fina"}[(prev, nxt)]
        out.append((u, form))
    return out


def runs_of(token: str) -> List[List[Tuple[str, str]]]:
    """The joined runs of a token, in reading order: each run is drawn as one connected shape."""
    runs, cur = [], []
    for u, f in forms_of(token):
        cur.append((u, f))
        if f in ("isol", "fina"):
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def unit_name(u: str) -> str:
    if len(u) == 2:
        return "lamAlef" + ALEFS[u[1]]
    return name(u)
