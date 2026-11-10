#!/usr/bin/env python3
"""Arabic letterform anatomy as machine-usable knowledge.

Sources studied for this module:
  * Pascal Zoghbi, "Arabic Type Anatomy & Typographic Terms" (29LT, 2015;
    Eye Magazine #90 "from earth to sky") — reference levels and part names,
    read from the article and its six diagrams.
  * Mamoun Sakkal, "Arabic Alphabet Chart (Naskh style)", 2007/2016 —
    writing direction, joining, descenders, pen angle, and rule 8:
    "Similar strokes in different letters should look the same."
  * Calligraphy practice (the user's references): the nuqta as unit of
    measure; proportions governed by the dot, circle and similarity systems.
"""

# --- reference levels (Zoghbi): Arabic has several, unlike Latin's five -------
LEVELS = [
    ("sky1", "سماء ١", "Sky 1 — highest ascender"),
    ("sky2", "سماء ٢", "Sky 2 — ascender (alef, lam)"),
    ("eye1", "عين - حلقة - سن ١", "Tooth/Loop/Eye height 1 (≈ Latin x-height)"),
    ("eye2", "عين - حلقة - سن ٢", "Tooth/Loop/Eye height 2"),
    ("tooth3", "سن ٣", "Tooth height 3"),
    ("tooth4", "سن ٤", "Tooth height 4"),
    ("tooth5", "سن ٥", "Tooth height 5"),
    ("base_thickness", "سمك القاعدة", "top of the baseline stroke"),
    ("base", "قاعدة", "baseline"),
    ("earth1", "أرض ١", "descender 1"),
    ("earth2", "أرض ٢", "descender 2 (deepest tails and bowls)"),
]

# --- parts (Zoghbi diagrams) ---------------------------------------------------
PARTS = {
    "tooth": ("سن", "short vertical stroke rising from the base (beh, teh, seen, noon)"),
    "head": ("رأس", "the top element of a letter (meem, waw, feh, qaf, ain)"),
    "brow": ("حاجب", "upper curved stroke of ain / ghain head"),
    "bowl": ("كاسة", "deep round bowl below the base (noon, qaf, seen, lam)"),
    "basin": ("حوض", "flat wide bowl along the base (beh isolated, kaf)"),
    "belly": ("بطن", "round lower body of jeem/hah/ain, and reh's curve"),
    "tail": ("ذنب", "the descending end stroke (reh, zain, waw, jeem, ain)"),
    "arm": ("ذراع", "diagonal stroke (kaf, the jeem family)"),
    "neck": ("رقبة / عنق", "narrow join between head and body (qaf, feh, waw)"),
    "chin": ("ذقن", "lower part of a closed head (qaf, feh)"),
    "ear": ("أذن", "small pointed counter shape"),
    "eye": ("عين / حلقة", "closed loop / counter (meem, feh, qaf, waw, heh)"),
    "almond": ("لوزة", "almond-shaped counter (sad, tah) — negative space"),
    "link": ("وصلة / زاوية", "joining stroke on the baseline; corner/angle"),
    "dot": ("نقطة", "diacritic dot — also the unit of proportion"),
    "space": ("بياض / فضاء", "whiteness: the negative space inside and around"),
}

# --- letters (Sakkal chart + rasm groups) -------------------------------------
# joins_left: connects to the following letter; descends: goes below the base
# in isolated/final forms (yellow in Sakkal's chart); parts: anatomical
# composition in writing order (right to left, mostly clockwise strokes).
LETTERS = {
    "alef": dict(ch="ا", rasm="alef", joins_left=False, descends=False, parts=["stem"]),
    "beh": dict(ch="ب", rasm="beh", joins_left=True, descends=False, parts=["tooth", "basin", "tooth"], dots=(1, "below")),
    "teh": dict(ch="ت", rasm="beh", joins_left=True, descends=False, parts=["tooth", "basin", "tooth"], dots=(2, "above")),
    "theh": dict(ch="ث", rasm="beh", joins_left=True, descends=False, parts=["tooth", "basin", "tooth"], dots=(3, "above")),
    "jeem": dict(ch="ج", rasm="hah", joins_left=True, descends=True, parts=["head", "arm", "belly", "tail"], dots=(1, "inside")),
    "hah": dict(ch="ح", rasm="hah", joins_left=True, descends=True, parts=["head", "arm", "belly", "tail"]),
    "dal": dict(ch="د", rasm="dal", joins_left=False, descends=False, parts=["head", "arm", "link"]),
    "reh": dict(ch="ر", rasm="reh", joins_left=False, descends=True, parts=["tooth", "tail"]),
    "zain": dict(ch="ز", rasm="reh", joins_left=False, descends=True, parts=["tooth", "tail"], dots=(1, "above")),
    "seen": dict(ch="س", rasm="seen", joins_left=True, descends=True, parts=["tooth", "tooth", "tooth", "bowl"]),
    "sad": dict(ch="ص", rasm="sad", joins_left=True, descends=True, parts=["almond", "tooth", "bowl"]),
    "tah": dict(ch="ط", rasm="tah", joins_left=True, descends=False, parts=["stem", "almond", "link"]),
    "ain": dict(ch="ع", rasm="ain", joins_left=True, descends=True, parts=["brow", "head", "belly", "tail"]),
    "feh": dict(ch="ف", rasm="feh", joins_left=True, descends=False, parts=["head", "neck", "basin"], dots=(1, "above")),
    "qaf": dict(ch="ق", rasm="qaf", joins_left=True, descends=True, parts=["head", "neck", "bowl"], dots=(2, "above")),
    "kaf": dict(ch="ك", rasm="kaf", joins_left=True, descends=False, parts=["stem", "basin"]),
    "lam": dict(ch="ل", rasm="lam", joins_left=True, descends=True, parts=["stem", "bowl"]),
    "meem": dict(ch="م", rasm="meem", joins_left=True, descends=True, parts=["head", "eye", "tail"]),
    "noon": dict(ch="ن", rasm="noon", joins_left=True, descends=True, parts=["tooth", "bowl"], dots=(1, "above")),
    "heh": dict(ch="ه", rasm="heh", joins_left=True, descends=False, parts=["eye"]),
    "waw": dict(ch="و", rasm="waw", joins_left=False, descends=True, parts=["head", "eye", "tail"]),
    "khah": dict(ch="خ", rasm="hah", joins_left=True, descends=True, parts=["head", "arm", "belly", "tail"], dots=(1, "above")),
    "thal": dict(ch="ذ", rasm="dal", joins_left=False, descends=False, parts=["head", "arm", "link"], dots=(1, "above")),
    "sheen": dict(ch="ش", rasm="seen", joins_left=True, descends=True, parts=["tooth", "tooth", "tooth", "bowl"], dots=(3, "above")),
    "dad": dict(ch="ض", rasm="sad", joins_left=True, descends=True, parts=["almond", "tooth", "bowl"], dots=(1, "above")),
    "zah": dict(ch="ظ", rasm="tah", joins_left=True, descends=False, parts=["stem", "almond", "link"], dots=(1, "above")),
    "ghain": dict(ch="غ", rasm="ain", joins_left=True, descends=True, parts=["brow", "head", "belly", "tail"], dots=(1, "above")),
    "alefmaksura": dict(ch="ى", rasm="yeh", joins_left=True, descends=True, parts=["head", "belly", "tail"]),
    "yeh": dict(ch="ي", rasm="yeh", joins_left=True, descends=True, parts=["head", "belly", "tail"], dots=(2, "below")),
}

PEN_ANGLE_NASKH_DEG = 70  # Sakkal, note 6

# Similarity system (Sakkal rule 8 / Zoghbi): which letters may be DERIVED from
# a letter of the uploaded word because they share its strokes. The derivation
# re-proportions the shared strokes; it does not paste them.
SIMILARITY = {
    "beh": ("noon", "same tooth + bowl family; the deep bowl (كاسة) becomes a basin (حوض)"),
    "teh": ("noon", "beh rasm"),
    "theh": ("noon", "beh rasm"),
    "reh": ("zain", "identical rasm (dots only differ)"),
}


def shared_parts(letter, available):
    """Parts of `letter` that already exist in the uploaded letters."""
    have = {p for a in available for p in LETTERS[a]["parts"]}
    return [p for p in LETTERS[letter]["parts"] if p in have]
