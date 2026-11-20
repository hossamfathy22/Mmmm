"""python -m khatt.bench.numbers_table FIT_DIR [OUT.json]

Gathers what every typeface needed of Sirang's numbers (free_numbers run on each, with KHATT_FAST=1, into
FIT_DIR) into the table the engine reads (khatt/data/numbers.json): for each typeface its measures
(numbers.measures, on the word REF) and, for each number its letters read, the median factor over those
letters (a letter that read a number and did not move it counts 1). With the fingerprint of every
builder, so that a number whose builder changed afterwards falls back to Sirang's.

To rebuild after a builder changes:
    ls ~/.cache/khatt/fonts | sed s/.woff2// | xargs -P3 -I{} sh -c 'KHATT_FAST=1 python -m khatt.bench.free_numbers {} FIT_DIR/{}.json'
    python -m khatt.bench.numbers_table FIT_DIR"""
from __future__ import annotations

import glob
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from khatt import numbers as N


def per_font(path: str) -> dict:
    r = json.load(open(path))
    per: dict = {}
    for row in r["letters"].values():
        for key in row.get("keys", []):
            per.setdefault(key, []).append(float(row.get("numbers_fit", {}).get(key, 1.0)))
    return r["font"], {k: round(float(np.median(v)), 4) for k, v in per.items()}


def build(fit_dir: str) -> dict:
    from khatt.bench import fonts
    from khatt.bench.letter_rules import REF
    from khatt.vector import dna, word_shapes
    N.install()
    files = {f.stem: f for f in fonts.font_files()}
    out = {"about": "factors on Sirang's shape numbers that each open typeface needed (bench.free_numbers); "
                    "measures: stem, hairline, ascender, descender to the x-height, slant (numbers.measures)",
           "prints": dict(N.PRINTS), "fonts": {}}
    for path in sorted(glob.glob(os.path.join(fit_dir, "*.json"))):
        name, nums = per_font(path)
        if name not in files:
            continue
        d = dna(word_shapes(fonts.load(str(files[name])), REF), REF)
        m = N.measures(d.get("stem", 0.0), d.get("hair", 0.0), d["x_height"], d.get("ascender", 0.0),
                       d.get("descender", 0.0), d.get("slant", 0.0))
        out["fonts"][name] = {"measures": [round(float(v), 4) for v in m],
                              "numbers": {k: round(v, 3) for k, v in nums.items()}}   # 1 kept: read and not moved
    return out


if __name__ == "__main__":
    t = build(sys.argv[1])
    dest = sys.argv[2] if len(sys.argv) > 2 else str(N.TABLE)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(t, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(t['fonts'])} typefaces, {len(t['prints'])} builders -> {dest}")
