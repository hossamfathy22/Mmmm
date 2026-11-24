"""A real reader as the judge: Tesseract's Arabic model reading whole words drawn in a style.

The skeleton reader of the bench (bench.score.read) reads real typefaces' own letters right only 63% of
the time, and the engine's 62.5%: at its own ceiling it no longer tells better from worse. A real OCR
reading the proof's fifteen words does: calibrated on the bench's 53 typeface styles (letters 30
pixels high, the word read as one line, which reads real typefaces best) it reads 68.6% of their words
exactly and 77.5% of their letters; the engine's, drawn in each typeface's measured style, 42.6% and
61.5%. It needs the tesseract program and its Arabic model (apt-get install tesseract-ocr
tesseract-ocr-ara); without them `available()` is False and nothing is judged.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unicodedata
from typing import Dict, List, Optional, Tuple

import numpy as np

# the proof: every joined form of every letter of the core set (41 basic shapes, checked by code), and the
# isolated forms in one line
WORDS = "موسيقى فاطمة مصطلح تعكس الرحمن صديق كهف عشب نجم هلال ظل فلك رقص نبع خط".split()
ISOLATED = "ا ب ح د ر س ص ط ع ف ق ك ل م ن ه و ى لا ء"
X_PX = 30                  # the size the judge reads best (calibrated on real typefaces)
REAL = {"words": 0.686, "letters": 0.775}   # how it reads the real typefaces: the yardstick


def available() -> bool:
    if not shutil.which("tesseract"):
        return False
    out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    return "ara" in (out.stdout + out.stderr).split()


def norm(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", t)
                   if not unicodedata.combining(c) and c not in " ‌‍ـ\n\t")


def lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]


def read(mask: np.ndarray) -> str:
    """What the reader reads in a picture of one word (ink True)."""
    from PIL import Image, ImageOps
    img = ImageOps.expand(Image.fromarray(((~mask) * 255).astype(np.uint8)), border=30, fill=255)
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        out = subprocess.run(["tesseract", f.name, "-", "-l", "ara", "--psm", "7"], capture_output=True, text=True, env=env).stdout
    os.unlink(f.name)
    return norm(out)


def word_mask(word: str, st) -> np.ndarray:
    """The word drawn in the style, upright, its letters X_PX pixels high, marks included."""
    from dataclasses import replace
    from .critic import raster
    from .typeset import typeset
    up = replace(st, slant=0.0)
    m, *_ = raster(typeset(word, up), px_per_stem=X_PX * up.stem / up.x_height, marks=True)
    return m


def judge(st, words: Optional[List[str]] = None) -> Dict:
    """The proof's words in the style, read: the share read exactly, the share of letters right, and
    each word's reading."""
    words = words or WORDS
    got: List[Tuple[str, str]] = [(w, read(word_mask(w, st))) for w in words]
    exact = float(np.mean([w == r for w, r in got]))
    letters = 1.0 - sum(lev(w, r) for w, r in got) / sum(len(w) for w, _ in got)
    return {"words": exact, "letters": letters, "read": got, "real": REAL}
