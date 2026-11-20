"""python -m khatt.bench.hidden_numbers TABLE_DIR FONT WORD OUT.json

Sirang's fixed numbers (the ~340 decimal numbers in the builders of letters.py and geometry.py: where a
curve's control point sits, how deep a bowl turns) against the letters a logo lacks. A word set in a
typeface is the logo (its letters on their vector outlines). Four drawings of the 18 skeletons the word
lacks, each compared with the typeface's own:

  rules     the named rules fitted to the word's letters (bidonaq_fit.fit_word); every number Sirang's
  numbers   and the fixed numbers the word's letters read, fitted to them too: a number in a shared tool
            (bowl_down: ن ل ق س ص ي, tail: ر و, sweep_right: ح ع) reaches the lacking letters that use it
  nearest   and every number the word does not read: the median of what the five typefaces nearest the
            word needed for it, as the engine takes it (numbers.for_logo on data/numbers.json; the
            typeface's own family left out). TABLE_DIR is not read any more (kept for the command line)
  nearest_only  the word's numbers as Sirang's, the rest from the nearest typefaces"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from khatt.bench import free_numbers as F
from khatt.bench import bidonaq_fit as E
from khatt.bench import fonts
from khatt.bench.hidden_rules import dna_vec, NEAREST
from khatt.bench.letter_rules import REF
from khatt.recipes import PARAMS, SKELETON
from khatt.style import Style
from khatt.typeset import typeset
from khatt.vector import word_shapes, dna


def load_table(folder: str) -> dict:
    """typeface -> {number: multiplier}, the median over the letters that fitted it (a number no letter
    moved is 1: Sirang's value served)."""
    out = {}
    for f in glob.glob(os.path.join(folder, "*.json")):
        try:
            r = json.load(open(f))
        except (ValueError, OSError):
            continue
        per: dict = {}
        for c, row in r["letters"].items():
            for key in row.get("keys", []):
                per.setdefault(key, []).append(float(row.get("numbers_fit", {}).get(key, 1.0)))
            if "keys" not in row:                       # older tables: only the numbers that moved
                for key, v in row.get("numbers_fit", {}).items():
                    per.setdefault(key, []).append(float(v))
        out[r["font"]] = {k: float(np.median(v)) for k, v in per.items()}
    return out


def nearest(face, word, table, files, skip_family):
    others = [f for f in table if f.rsplit("-", 1)[0] != skip_family and f in files]
    me = dna_vec(face, word)
    V = {f: dna_vec(fonts.load(str(files[f])), REF) for f in others}
    sd = np.std(np.array(list(V.values())), 0) + 1e-6
    return sorted(others, key=lambda f: np.abs((V[f] - me) / sd).sum())[:NEAREST]


def reads_numbers(word: str, st: Style) -> list:
    F.FREE.clear(); F.USED.clear()
    E.letters_of(word, st, 1.0)
    return sorted(F.USED)


def fit_numbers(st: Style, keys: list, score) -> tuple:
    """Coordinate search over the multipliers of the numbers the word reads, two rounds."""
    free: dict = {}
    F.FREE.clear()
    now = score(st)
    for steps in ((0.6, 0.8, 1.25, 1.6), (0.9, 1.1)):
        for key in keys:
            cur = free.get(key, 1.0)
            best = (now, cur)
            for m in steps:
                F.FREE.clear(); F.FREE.update({**free, key: cur * m})
                try:
                    s = score(st)
                except Exception:
                    continue
                if s > best[0] + 0.002:
                    best = (s, cur * m)
            if best[1] != cur:
                free[key], now = best[1], best[0]
    F.FREE.clear()
    return free, now


def hidden(face, word, st, sc, free: dict) -> dict:
    have = {SKELETON.get(c, c) for c in word} | set(word)
    vals = {}
    F.FREE.clear(); F.FREE.update(free)
    try:
        for c in E.SK18:
            if c in have or SKELETON.get(c, c) in have:
                continue
            tr = E.truth_of(word_shapes(face, c), sc)
            try:
                g = E.letters_of(c, st, sc)
            except Exception:
                vals[c] = 0.0
                continue
            if tr and g:
                vals[c] = E.iou(list(tr.values())[0], list(g.values())[0])
    finally:
        F.FREE.clear()
    return vals


def run(table_dir, name, word):
    F.free_all()
    files = {f.stem: f for f in fonts.font_files()}
    face = fonts.load(str(files[name]))
    sh = word_shapes(face, word); d = dna(sh, word)
    X = d["x_height"]; sc = 70.0 / X
    truth = E.truth_of(sh, sc)
    k = X / Style().x_height
    base = Style(**{**Style().__dict__, **{f: getattr(Style(), f) * k for f in E.LEN}})
    meas = Style(**{**base.__dict__, "stem": d.get("stem", base.stem), "hair": max(1.0, d.get("hair", base.hair)), "x_height": X,
                    "ascender": max(0.2 * X, d.get("ascender", base.ascender)), "descender": max(0.2 * X, d.get("descender", base.descender)),
                    "white": d.get("stem", base.stem)})
    meas.params = {}

    def score(st):
        got = E.letters_of(word, st, sc)
        v = [E.iou(truth[i], got[i]) for i in truth if i in got]
        return float(np.mean(v)) if v else 0.0

    t0 = time.time()
    cache = {}; typeset(word, meas, cache=cache)
    reads = sorted({n for _, deps in cache.values() for n in deps if n in PARAMS})
    st, own_rules = E.fit_word(meas, score, reads)
    keys = reads_numbers(word, st)
    free, own_numbers = fit_numbers(st, keys, score)
    res = {"font": name, "word": word, "own_rules": own_rules, "own_numbers": own_numbers,
           "word_numbers": len(keys), "moved": {k_: round(v, 3) for k_, v in free.items()}}
    H = {"rules": hidden(face, word, st, sc, {}), "numbers": hidden(face, word, st, sc, free)}
    # the nearest typefaces' numbers as the engine takes them for a logo (numbers.for_logo, from the logo's
    # measures), the typeface's own family left out
    from khatt import numbers as N
    from_near = N.for_logo(meas, skip_family=name.rsplit("-", 1)[0])
    probe = Style(**{**st.__dict__, "numbers": dict(from_near)})
    N.prune(probe)                                   # as analyse does, on the rules fitted to the word
    from_near = probe.numbers
    if from_near or N.table():
        H["nearest"] = hidden(face, word, st, sc, {**from_near, **free})
        H["nearest_only"] = hidden(face, word, st, sc, {key: v for key, v in from_near.items() if key not in keys})
        res["nearest"] = N.nearest(N.measures(meas.stem, meas.hair, meas.x_height, meas.ascender, meas.descender, meas.slant),
                                   name.rsplit("-", 1)[0])
    res["hidden"] = {k_: float(np.mean(list(v.values()))) if v else None for k_, v in H.items()}
    res["per_letter"] = {c: [round(H[k_][c], 3) for k_ in H] for c in H["rules"]}
    res["order"] = list(H)
    res["seconds"] = round(time.time() - t0)
    return res


if __name__ == "__main__":
    r = run(*sys.argv[1:4])
    json.dump(r, open(sys.argv[4], "w"), ensure_ascii=False)
    print(f"{r['font']:22s} {r['word']:10s} word: rules {r['own_rules']:.3f} +numbers {r['own_numbers']:.3f} ({r['word_numbers']} read, "
          f"{len(r['moved'])} moved) | lacks: " + ", ".join(f"{k} {v:.3f}" for k, v in r["hidden"].items() if v is not None)
          + f" | {r['seconds']}s", flush=True)
