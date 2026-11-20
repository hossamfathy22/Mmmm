"""python -m khatt.bench.blind LOGO.png [TEXT] [--json OUT.json]

Does the engine measure a logo, or does it hand back the numbers written in its code? The defaults of
Style are the measures of the hand-made بيدوناق (Sirang's pen: stem 23, x-height 66, slant 14° ...).
Any measure the engine fails to take silently keeps that default, and on Sirang or Bidonaq such a measure
looks right because it is the answer written in.

The logo is measured twice (pipeline.analyse, nothing drawn): once as the engine stands, once blind, with
every number among the defaults replaced by another (POISON). A measure that moves with the defaults was
not read from the picture."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from khatt.style import Style

# Far from Sirang on purpose, so that a value read from here cannot pass for Sirang's.
POISON = {
    "stem": 9.0, "hair": 1.5, "x_height": 140.0, "ascender": 90.0, "descender": 15.0, "slant": -7.0,
    "white": 47.0, "dot": 7.0, "contrast_power": 4.4, "top_angle": 21.0, "top_round": 0.8, "join_in": 0.21,
    "join_out": 0.63, "ring_n": 5.5, "dot_n": 6.0, "dot_rot": 30.0, "tail_reach": 1.7, "overshoot": 3.0,
    "seen_drop": 9.0, "group_gap": 70.0, "reh_corner": 6.0, "join_white": 33.0, "round_gap": 1.6,
    "angular": 0.7, "corner_r": 5.0, "tail_drop": 0.35, "top_join": 0.0, "dash": 30.0,
}

# what the report compares: style fields, as the pipeline turns the measures into drawing rules
FIELDS = ["stem", "hair", "x_height", "ascender", "descender", "slant", "white", "dot", "contrast_power",
          "top_angle", "top_round", "join_in", "join_out", "ring_n", "dot_n", "dot_rot", "tail_reach", "overshoot",
          "seen_drop", "group_gap", "join_white", "reh_corner", "angular", "corner_r", "tail_drop", "top_join",
          "dash", "round_gap", "nib_width", "nib_thick", "nib_angle"]
LENGTHS = {"stem", "hair", "ascender", "descender", "white", "dot", "overshoot", "seen_drop", "group_gap",
           "join_white", "reh_corner", "corner_r", "dash", "nib_width", "nib_thick"}


def poison() -> None:
    """Swap the numbers among Style's defaults for POISON's, in the constructor itself."""
    import inspect
    names = [n for n, p in inspect.signature(Style.__init__).parameters.items() if p.default is not inspect.Parameter.empty]
    vals = list(Style.__init__.__defaults__)
    assert len(names) == len(vals)
    for i, n in enumerate(names):
        if n in POISON:
            vals[i] = type(vals[i])(POISON[n])
    Style.__init__.__defaults__ = tuple(vals)


def measure(path: str, text: str = "", blind: bool = False) -> dict:
    if blind:
        poison()
    from khatt.pipeline import analyse
    a = analyse(Path(path).read_bytes(), texts=[text] if text else None)
    st, L = a["style"], a["lines"][a["src"]]
    out = {f: float(getattr(st, f)) for f in FIELDS}
    out["_line"] = a["src"]
    out["_k"] = a["k"]
    out["_features"] = sorted(a["feats"])
    return out


def compare(path: str, text: str = "") -> dict:
    """Both runs, each in its own process (the defaults are swapped for the whole process)."""
    import subprocess
    got = {}
    for mode in ("seeing", "blind"):
        r = subprocess.run([sys.executable, "-m", "khatt.bench.blind", "--one", mode, path, text],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-2000:])
        got[mode] = json.loads(r.stdout.strip().splitlines()[-1])
    rows = {}
    for f in FIELDS:
        a, b = got["seeing"][f], got["blind"][f]
        same = math.isclose(a, b, rel_tol=0.01, abs_tol=0.01)
        rows[f] = {"seeing": round(a, 3), "blind": round(b, 3), "default": getattr(Style(), f),
                   "from": "الصورة" if same else "الكود"}
    return {"logo": Path(path).name, "text": text, "rows": rows, "features": got["seeing"]["_features"],
            "line": got["seeing"]["_line"]}


def show(res: dict) -> None:
    X = res["rows"]["x_height"]["seeing"]
    print(f"\n{res['logo']}  «{res['text']}»  السطر {res['line'] + 1}  (المعاملات المقيسة: {', '.join(res['features'])})")
    print(f"{'القياس':16s} {'المحرك':>9s} {'÷ارتفاع':>8s} {'أعمى':>9s} {'الافتراضي':>9s}  المصدر")
    for f, r in res["rows"].items():
        ratio = f"{r['seeing'] / X:8.3f}" if f in LENGTHS and X else " " * 8
        print(f"{f:16s} {r['seeing']:9.3f} {ratio} {r['blind']:9.3f} {r['default']:9.3f}  {r['from']}")
    bad = [f for f, r in res["rows"].items() if r["from"] == "الكود"]
    print(f"من الكود لا من الصورة: {len(bad)} من {len(res['rows'])}: {', '.join(bad)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--one":
        print(json.dumps(measure(args[2], args[3] if len(args) > 3 else "", blind=args[1] == "blind")))
        raise SystemExit(0)
    out = None
    if "--json" in args:
        i = args.index("--json"); out = args[i + 1]; args = args[:i] + args[i + 2:]
    res = compare(args[0], args[1] if len(args) > 1 else "")
    show(res)
    if out:
        Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
