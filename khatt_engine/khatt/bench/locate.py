"""python -m khatt.bench.locate [--faces N]

How well is each letter of a logo found? On logos set in real typefaces, where HarfBuzz says which
ink is which letter, the share of the ink given to its own letter:
  - by the engine as it shares the ink out today: its own letters drawn in the measured style and laid
    over the logo (selfcheck._slide), every pixel to the nearest;
  - by match.match: the word set in other typefaces (never the logo's own family) and warped onto it.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

PHRASES = ["قهوة عربية", "رمضان كريم", "حلويات فاخرة", "مطعم السمك", "زهور الربيع", "مخبز الفرح", "عصير طازج",
           "فندق الخليج", "حديقة الورد", "مقهى النخيل", "سفينة", "مستقبل"]
LOGOS = ["reem-kufi-400", "el-messiri-700", "vazirmatn-700", "changa-700", "tajawal-700", "mada-700", "lemonada-700",
         "cairo-700", "almarai-700", "kufam-700", "aref-ruqaa-400", "lalezar-400"]
REFS = ["amiri-400", "noto-naskh-arabic-400", "vazirmatn-400", "cairo-400", "reem-kufi-400", "noto-kufi-arabic-400"]


def engine_owner(ink: np.ndarray, text: str):
    """The engine's way: analyse the picture, set the word in its measured style, lay its letters over
    the logo, every pixel to the nearest letter. Returns a label image in the picture's frame, or None."""
    from scipy import ndimage as ndi
    from PIL import Image
    import io
    from ..pipeline import analyse
    from ..selfcheck import _slide
    from ..style import Style
    from ..typeset import typeset
    im = Image.fromarray((255 - ink * 255).astype(np.uint8)); b = io.BytesIO(); im.save(b, "PNG")
    an = analyse(b.getvalue(), texts=[text], read=False)
    L, k, st = an["lines"][an["src"]], an["k"], an["style"]
    up = Style(**{**st.__dict__, "slant": 0.0})
    lay = typeset(text, up)
    logo = np.asarray(L.B, bool)
    s = 1.0 / k
    base_row = L.row_of_y(0.0)
    xs = [p[0] for sh in lay.shapes for poly in sh for p in poly]
    x_of = float(np.flatnonzero(logo.any(axis=0))[-1]) - max(xs) * s
    slid = _slide(lay, up, logo, s, x_of, base_row, st.pitch * s)
    if not slid:
        return None, L, k
    dist = np.stack([ndi.distance_transform_edt(~m) for _, m, _, _ in slid])
    owner = dist.argmin(axis=0)
    # the letters in the order they were placed are the letters of the text without spaces
    idx = [i for i, c in enumerate(text) if not c.isspace()]
    lab = np.zeros_like(logo, dtype=int)
    for j, (p, *_ ) in enumerate(slid):
        if j < len(idx):
            lab[(owner == j) & logo] = idx[j] + 1
    return lab, L, k


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=len(LOGOS))
    p.add_argument("--no-engine", action="store_true")
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    from ..bench import fonts
    from ..match import match, ref_labels
    files = {f.stem: f for f in fonts.font_files()}
    rows = []
    for i, name in enumerate(LOGOS[:a.n]):
        if name not in files:
            continue
        text = PHRASES[i % len(PHRASES)]
        face = fonts.load(str(files[name]))
        truth, ink = ref_labels(face, text, 120)
        pad = 40
        truth = np.pad(truth, pad); ink = truth > 0
        fam = name.rsplit("-", 1)[0]
        refs = [fonts.load(str(files[r])) for r in REFS if r in files and r.rsplit("-", 1)[0] != fam]
        got = match(ink, text, refs)
        acc_m = float((got["owner"][ink] == truth[ink]).mean()) if got else 0.0
        acc_e = None
        if not a.no_engine:
            try:
                lab, L, k = engine_owner(ink, text)
                # the engine works on its own (deskewed, scaled) frame: compare on its line's ink, with the
                # truth taken to that frame by its rows and columns
                if lab is not None:
                    from PIL import Image
                    H, W = ink.shape
                    tr = np.asarray(Image.fromarray(truth.astype(np.int32)).resize((int(W / k), int(H / k)), Image.NEAREST))
                    t = tr[L.top:L.top + lab.shape[0], :lab.shape[1]]
                    t = np.pad(t, ((0, max(0, lab.shape[0] - t.shape[0])), (0, max(0, lab.shape[1] - t.shape[1]))))[:lab.shape[0], :lab.shape[1]]
                    m = (lab > 0) & (t > 0)
                    acc_e = float((lab[m] == t[m]).mean()) if m.any() else None
            except Exception as e:
                acc_e = None
        rows.append((name, text, acc_e, acc_m, got["face"].name if got else "—"))
        print(f"{name:18s} {text:14s} engine {'—' if acc_e is None else f'{acc_e:.1%}'}   match {acc_m:.1%}  (ref {rows[-1][4]})", flush=True)
    e = [r[2] for r in rows if r[2] is not None]; m = [r[3] for r in rows]
    print(f"\nthe ink given to its own letter: engine {np.mean(e):.1%} on {len(e)}, match {np.mean(m):.1%} on {len(m)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
