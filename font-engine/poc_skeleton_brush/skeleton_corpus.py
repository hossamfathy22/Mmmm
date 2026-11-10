#!/usr/bin/env python3
"""Pen-path (skeleton) corpus: the single smooth pen path of every letter form
in every OFL Arabic font, normalised so different fonts can be compared and
blended:

  * same drawing rules as for the uploaded word (pen_path.single_walk)
  * resampled to N points by arc length, start = Arabic writing start
  * units = nuqta of that font, origin = (glyph body centre x, baseline y)

Also stored per glyph: mean stroke thickness (ink area / path length) in
nuqtas, so a glyph can be rebuilt as path + constant brush for benchmarks.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import skeleton_brush as sb

N_PTS = 64
ZWJ = "‍"
FORMS = {"meem.init": "م" + ZWJ, "zain.fina": ZWJ + "ز", "noon.isol": "ن",
         "beh.isol": "ب", "dal.isol": "د", "reh.isol": "ر", "alef.isol": "ا", "lam.isol": "ل",
         "alefmaksura.isol": "ى", "seen.isol": "س",
         # every basic body (rasm), for the path-based second eye
         "hah.isol": "ح", "sad.isol": "ص", "tah.isol": "ط", "ain.isol": "ع", "feh.isol": "ف", "qaf.isol": "ق",
         "kaf.isol": "ك", "meem.isol": "م", "heh.isol": "ه", "waw.isol": "و"}
SIZE = 420


def glyph_path(font_path, text):
    import pen_path as pp
    font = ImageFont.truetype(str(font_path), SIZE, layout_engine=ImageFont.Layout.RAQM)
    W = SIZE * 3
    base = int(W * 0.6)
    img = Image.new("L", (W, W), 0)
    ImageDraw.Draw(img).text((W - SIZE, base), text, font=font, fill=255, anchor="rs", direction="rtl", language="ar")
    m = np.array(img) > 127
    comps = sb.split_components(m, min_area=20)
    body = max(comps, key=lambda c: c["area"])
    dots = [c for c in comps if c is not body and c["area"] < 0.2 * body["area"]]
    d = ndimage.distance_transform_edt(body["mask"])
    coords, _ = pp.single_walk(body["mask"], d)
    p = sb.smooth_open(sb.resample(sb.orient_rtl(coords), 1.0), 6.0)
    p = sb.extend_to_tips(p, body["mask"])
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, L[-1], N_PTS)
    P = np.stack([np.interp(t, L, p[:, 0]), np.interp(t, L, p[:, 1])], 1)
    x, y, w, h = body["bbox"]
    return P, (x + w / 2, base), body["area"] / max(L[-1], 1), dots


def font_entry(path):
    try:
        _, _, _, ndots = glyph_path(path, "ن")
        D = float(np.sqrt(np.median([c["area"] for c in ndots]))) if ndots else None
        if not D:
            return None
        out = {"nuqta_px": D}
        for k, t in FORMS.items():
            try:
                P, (cx, base), thick, _ = glyph_path(path, t)
                out[k] = {"path": (((P - [cx, base]) / D) * [1, 1]).round(4).tolist(),
                          "thickness": round(thick / D, 4)}
            except Exception:
                pass
        return out
    except Exception:
        return None


def main():
    fonts = sorted((sb.HERE / "corpus").glob("*.ttf"))
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(zip([f.stem for f in fonts], ex.map(font_entry, fonts)))
    res = {k: v for k, v in res.items() if v}
    (sb.OUT / "skeleton_corpus.json").write_text(json.dumps(res))
    print("fonts:", len(res), "forms per font:", np.mean([len(v) - 1 for v in res.values()]).round(1))


if __name__ == "__main__":
    main()
