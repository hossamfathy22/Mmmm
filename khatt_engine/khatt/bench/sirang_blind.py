"""python -m khatt.bench.sirang_blind SIRANG.png BIDONAQ.png MODE [TABLE_DIR] [--save OUT.png]

The one logo whose answer is known: the designer drew بيدوناق by hand from the logo سیرنگ. The engine,
given سیرنگ, draws بيدوناق, and the drawing is compared with the hand-made one (IoU at the best shift,
the whole word and letter by letter by columns). But the engine's fixed numbers (free_numbers: ~340 of
them in letters.py and geometry.py) are بيدوناق's own, measured on it: the comparison as it stands
reads the answer from the code. MODE takes them away:

  code       Sirang's numbers, as the engine stood (the answer in the code)
  engine     the logo's own numbers, as the engine takes them now (numbers.for_logo)
  random     every number times a factor drawn in [0.7, 1.4] (seed 1..5 averaged): how much the drawing
             leans on them
  nearest    every number the median of what the five typefaces nearest the logo needed (TABLE_DIR)
  nearest+fit  nearest, then the numbers the logo's own letters read fitted to the logo (selfcheck)"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve

from khatt.bench import free_numbers as F


def ink(a: np.ndarray) -> np.ndarray:
    r = np.flatnonzero(a.any(1)); c = np.flatnonzero(a.any(0))
    return a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = fftconvolve(a.astype(float), b[::-1, ::-1].astype(float), mode="full")
    return float((inter / (a.sum() + b.sum() - inter)).max())


def to_height(e: np.ndarray, h: int) -> np.ndarray:
    s = h / e.shape[0]
    return np.asarray(Image.fromarray(e.astype(np.uint8) * 255).resize((max(1, int(e.shape[1] * s)), h))) > 127


def draw(sirang: str, free: dict, fit: bool = False) -> np.ndarray:
    """بيدوناق from سیرنگ as pipeline.run draws it (self-check and taking, the reference, verify)."""
    import io
    from pathlib import Path
    from khatt.pipeline import analyse
    from khatt.verify import verify_and_correct
    from khatt.typeset import typeset
    from khatt.render import to_png
    from khatt.selfcheck import check_and_take
    from khatt.reference import adopt_nearest
    F.FREE.clear(); F.FREE.update(free)
    a = analyse(Path(sirang).read_bytes(), texts=["سیرنگ"])
    st, L, k = a["style"], a["lines"][a["src"]], a["k"]
    if fit:
        free.update(fit_numbers_on_logo(L, st, k, free))
        F.FREE.clear(); F.FREE.update(free)
    check_and_take(L, "سیرنگ", st, k, take=True)
    adopt_nearest(L, "سیرنگ", st, k)
    lay, _, _, _ = verify_and_correct(lambda s: typeset("بيدوناق", s), st)
    im = np.asarray(Image.open(io.BytesIO(to_png(lay, height=400))).convert("L")) < 128
    return ink(im)


def fit_numbers_on_logo(L, st, k, free: dict) -> dict:
    """The numbers the logo's own word reads, fitted to the logo (the self-check's score), from where
    `free` puts them."""
    from khatt.selfcheck import self_check
    F.FREE.clear(); F.FREE.update(free); F.USED.clear()
    from khatt.typeset import typeset
    typeset("سیرنگ", st)
    keys = sorted(F.USED)
    cur = dict(free)

    def score(fr):
        F.FREE.clear(); F.FREE.update(fr)
        r = self_check(L, "سیرنگ", st, k)
        return r["overall"] if r else 0.0
    now = score(cur)
    for steps in ((0.7, 0.85, 1.18, 1.4), (0.93, 1.07)):
        for key in keys:
            base = cur.get(key, 1.0); best = (now, base)
            for m in steps:
                s = score({**cur, key: base * m})
                if s > best[0] + 0.002:
                    best = (s, base * m)
            if best[1] != base:
                cur[key], now = best[1], best[0]
    return {kk: v for kk, v in cur.items() if kk in keys}


def compare(drawn: np.ndarray, hand: np.ndarray) -> float:
    return iou(hand, to_height(drawn, hand.shape[0]))


def main():
    args = sys.argv[1:]
    save = None
    if "--save" in args:
        i = args.index("--save"); save = args[i + 1]; args = args[:i] + args[i + 2:]
    sirang, bidonaq, mode = args[:3]
    table_dir = args[3] if len(args) > 3 else None
    F.free_all()
    hand = ink(np.asarray(Image.open(bidonaq).convert("L")) < 128)
    out = {"mode": mode}
    if mode in ("code", "engine"):
        # code: Sirang's numbers (as before); engine: the logo's own, as the engine now takes them
        if mode == "code":
            os.environ["KHATT_NUMBERS"] = "sirang"
        d = draw(sirang, {}); out["iou"] = compare(d, hand)
    elif mode == "random":
        os.environ["KHATT_NUMBERS"] = "sirang"
        F.FREE.clear(); F.USED.clear()
        draw(sirang, {}); keys = sorted(F.USED)
        vals = []
        for seed in range(1, 6):
            rng = np.random.default_rng(seed)
            d = draw(sirang, {kk: float(np.exp(rng.uniform(np.log(0.7), np.log(1.4)))) for kk in keys})
            vals.append(compare(d, hand))
        out["iou"], out["each"] = float(np.mean(vals)), vals
    else:
        from khatt.bench.hidden_numbers import load_table
        from khatt.bench import fonts
        from khatt.bench.hidden_rules import dna_vec
        from khatt.vector import word_shapes
        table = load_table(table_dir)
        files = {f.stem: f for f in fonts.font_files()}
        # the typefaces nearest the logo: by what was measured on it (stem, hairline, ascender and
        # descender to the x-height, slant), against the same measures of each typeface's word
        from pathlib import Path
        from khatt.pipeline import analyse
        a = analyse(Path(sirang).read_bytes(), texts=["سیرنگ"]); st = a["style"]; X = st.x_height
        me = np.array([st.stem / X, st.hair / X, st.ascender / X, st.descender / X, st.slant])
        from khatt.vector import dna
        from khatt.bench.letter_rules import REF
        V = {}
        for f in table:
            if f in files:
                dd = dna(word_shapes(fonts.load(str(files[f])), REF), REF); XX = dd["x_height"]
                V[f] = np.array([dd.get("stem", 0) / XX, dd.get("hair", 0) / XX, dd.get("ascender", 0) / XX,
                                 dd.get("descender", 0) / XX, dd.get("slant", 0.0)])
        sd = np.std(np.array(list(V.values())), 0) + 1e-6
        near = sorted(V, key=lambda f: np.abs((V[f] - me) / sd).sum())[:5]
        pool = {}
        for f in near:
            for kk, v in table[f].items():
                pool.setdefault(kk, []).append(v)
        free = {kk: float(np.median(v)) for kk, v in pool.items()}
        d = draw(sirang, dict(free), fit=(mode == "nearest+fit"))
        out["iou"], out["nearest"] = compare(d, hand), near
    if save and mode != "random":
        h2 = to_height(d, hand.shape[0]); W = max(hand.shape[1], h2.shape[1])
        pad = lambda m: np.pad(m, ((20, 20), (0, W - m.shape[1])))
        Image.fromarray((~np.vstack([pad(hand), pad(h2)])).astype(np.uint8) * 255).save(save)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
