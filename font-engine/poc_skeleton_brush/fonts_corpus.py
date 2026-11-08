#!/usr/bin/env python3
"""Arabic font corpus measured the calligrapher's way: in nuqtas.

Downloads the open-licensed (SIL OFL) Arabic families from Google Fonts and,
for every font, renders letter forms (isolated / initial / medial / final,
forced with ZWJ and shaped by raqm/HarfBuzz) and measures each letter's BODY
(rasm, without dots):

    width, height, top and bottom (relative to the noon bowl's bottom),
    stroke weight (ink area / centerline length)

all divided by the font's NUQTA = side of the square with the same area as
the noon's dot. That is exactly how calligraphy fixes proportions (alif = n
nuqtas, the head of waw = 2 x 2 nuqtas, ...), only measured automatically
from ~55 typefaces instead of from a teacher's sheet.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

import skeleton_brush as sb

CORPUS = sb.HERE / "corpus"
FAMILIES = [
    "Oi", "Mada", "Kufam", "Vibes", "Blaka", "Rubik", "Amiri", "Mirza", "Cairo", "Cairo Play", "Ruwudu",
    "Thabit", "Marhey", "Changa", "Gulzar", "Fustat", "Rakkas", "Estedad", "Tajawal", "Katibeh", "Beiruti",
    "Handjet", "Almarai", "Lalezar", "Lemonada", "Alyamama", "Jomhuria", "Alkalami", "Parastoo", "Reem Kufi",
    "Reem Kufi Fun", "Aref Ruqaa", "Vazirmatn", "El Messiri", "Readex Pro", "Alexandria", "Markazi Text",
    "Badeen Display", "Baloo Bhaijaan 2", "Playpen Sans Arabic", "Lateef", "Zain", "Scheherazade New",
    "Harmattan", "Qahiri", "Noto Sans Arabic", "Noto Naskh Arabic", "IBM Plex Sans Arabic", "Noto Kufi Arabic",
    "Amiri Quran", "Noto Nastaliq Urdu", "Firjar", "Alan Sans",
]
ZWJ = "‍"
# letter forms: anchors are the forms present in the uploaded word
FORMS = {
    "meem.init": "م" + ZWJ, "zain.fina": ZWJ + "ز", "noon.isol": "ن",
    "beh.isol": "ب", "teh.isol": "ت", "theh.isol": "ث", "alefmaksura.isol": "ى", "reh.isol": "ر",
    "dal.isol": "د", "waw.isol": "و", "lam.isol": "ل", "alef.isol": "ا", "heh.isol": "ه", "ain.isol": "ع",
    "seen.isol": "س", "beh.init": "ب" + ZWJ, "beh.medi": ZWJ + "ب" + ZWJ, "beh.fina": ZWJ + "ب",
    "noon.fina": ZWJ + "ن", "meem.isol": "م", "yeh.isol": "ي", "hah.isol": "ح", "sad.isol": "ص",
}
ANCHORS = ["meem.init", "zain.fina", "noon.isol"]
SIZE = 360


def download():
    CORPUS.mkdir(exist_ok=True)
    got = {}
    for fam in FAMILIES:
        dst = CORPUS / (fam.replace(" ", "") + ".ttf")
        if not dst.exists():
            try:
                css = urllib.request.urlopen(
                    "https://fonts.googleapis.com/css2?family=" + fam.replace(" ", "+"), timeout=30).read().decode()
                url = re.search(r"url\((https://[^)]+\.ttf)\)", css).group(1)
                urllib.request.urlretrieve(url, dst)
            except Exception as ex:  # family not served / renamed
                print("skip", fam, type(ex).__name__)
                continue
        got[fam] = dst
    return got


def render(font_path, text):
    font = ImageFont.truetype(str(font_path), SIZE, layout_engine=ImageFont.Layout.RAQM)
    W = SIZE * 4
    img = Image.new("L", (W, W), 0)
    base = int(W * 0.6)
    ImageDraw.Draw(img).text((W - SIZE, base), text, font=font, fill=255, anchor="rs",
                             direction="rtl", language="ar")
    return np.array(img) > 127, base


def measure_form(font_path, text):
    m, base = render(font_path, text)
    if m.sum() < 50:
        return None
    comps = sb.split_components(m, min_area=20)
    body = max(comps, key=lambda c: c["area"])
    dots = [c for c in comps if c is not body and c["area"] < 0.35 * body["area"]]
    x, y, w, h = body["bbox"]
    skel_len = skeletonize(body["mask"]).sum()
    return {"w": w, "h": h, "top": base - y, "bottom": base - (y + h),
            "weight": body["area"] / max(skel_len, 1),
            "dot_side": float(np.sqrt(np.median([d["area"] for d in dots]))) if dots else None,
            "n_dots": len(dots)}


def measure_font(font_path):
    raw = {}
    for name, text in FORMS.items():
        try:
            raw[name] = measure_form(font_path, text)
        except Exception:
            raw[name] = None
    noon = raw.get("noon.isol")
    if not noon or not noon["dot_side"]:
        return None
    D = noon["dot_side"]           # 1 nuqta
    ref = noon["bottom"]           # vertical reference: bottom of the noon bowl
    out = {"nuqta_px_at_em360": D,
           # where the baseline (قاعدة) sits above the bottom of the noon bowl, in nuqtas
           "baseline_above_noon_bottom": -ref / D}
    for name, r in raw.items():
        if r is None:
            continue
        out[name] = {"w": r["w"] / D, "h": r["h"] / D, "top": (r["top"] - ref) / D,
                     "bottom": (r["bottom"] - ref) / D, "weight": r["weight"] / D,
                     "dot_side": (r["dot_side"] / D) if r["dot_side"] else None, "n_dots": r["n_dots"]}
    return out


def main():
    fonts = download()
    data = {}
    for fam, path in fonts.items():
        m = measure_font(path)
        if m:
            data[fam] = m
    (sb.OUT / "corpus_nuqta_metrics.json").write_text(json.dumps(data, ensure_ascii=False, indent=0))
    print("fonts measured:", len(data), "of", len(fonts))


if __name__ == "__main__":
    main()
