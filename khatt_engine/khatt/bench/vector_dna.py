"""python -m khatt.bench.vector_dna [--fonts N] [--out FILE]

How exactly is a logo's style read from its vector drawing, against the engine reading its picture?
Open typefaces are vector drawings whose truth is known: a word set in one is taken as a logo. Its style is
read (vector.dna) from the shapes alone, and by the engine (pipeline.analyse) from a picture of them; both
are compared with the typeface's own measures, taken from glyphs the word does not need:
  x-height  the top of the tooth of ـبـ
  stem      the width of ا half way up
  hairline  the thickness of the join between the teeth of ـببـ
Errors are in hundredths of the x-height. One typeface per process (the memory of one is given back)."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

WORDS = ["حديقة الورد", "مستقبل", "قهوة عربية", "سفينة", "مطعم السمك", "نبيل"]


def truth(face):
    from ..vector import word_shapes, geometry, _runs
    t = {}
    s = [x for x in word_shapes(face, "ـبـ") if x.index == 1]
    t["x_height"] = max(x.bounds()[3] for x in s) if s else None
    a = word_shapes(face, "ا")[0]
    g = geometry(a)
    ym = a.bounds()[3] / 2
    r = _runs(g, a.bounds()[0] - 10, ym, a.bounds()[2] + 10, ym)
    t["stem"] = max(b - c for c, b in r) if r else None
    bb = word_shapes(face, "ـببـ")
    teeth = [x for x in bb if x.index in (1, 2)]
    if len(teeth) == 2:
        xm = 0.5 * (teeth[0].bounds()[0] + teeth[1].bounds()[2]) if teeth[0].bounds()[0] > teeth[1].bounds()[0] else 0.5 * (teeth[1].bounds()[0] + teeth[0].bounds()[2])
        from shapely.ops import unary_union
        G = unary_union([geometry(x) for x in bb if geometry(x) is not None])
        r = [(c, b) for c, b in _runs(G, xm, -2000, xm, 3000) if c < 0.5 * (t["x_height"] or 500)]
        t["hair"] = min(b - c for c, b in r) if r else None
    return t


def one(path: str, out: str) -> None:
    from PIL import Image
    from ..bench import fonts
    from ..vector import word_shapes, dna, raster
    from ..pipeline import analyse
    face = fonts.load(path)
    tr = truth(face)
    rows = []
    for w in WORDS:
        sh = word_shapes(face, w)
        if not sh:
            continue
        d = dna(sh, w)
        # the same word as a picture, 120 pixels to its x-height, for the engine
        scale = 120.0 / (tr["x_height"] or 500)
        m, _ = raster(sh, scale, pad=40)
        buf = io.BytesIO(); Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(buf, "PNG")
        e = {}
        try:
            an = analyse(buf.getvalue(), texts=[w], read=False)
            st = an["style"]
            e = {"x_height": st.x_height / scale, "stem": st.stem / scale, "hair": st.hair / scale, "slant": st.slant}
        except Exception as ex:
            e = {"error": str(ex)[:80]}
        # the same picture traced back to a drawing (trace.py), then read as one
        tr_ = {}
        try:
            from ..match import locate
            from ..pipeline import MATCH_FACES
            from ..trace import trace
            files = {f.stem: f for f in fonts.font_files()}
            faces = [fonts.load(str(files[n])) for n in MATCH_FACES if n in files]
            from ..guard import prepare
            from ..trace import measure
            work, _, pscale, inverted, _ = prepare(buf.getvalue())
            gw = np.asarray(work, float) / 255.0
            grey = gw if inverted else 1.0 - gw
            got = locate(buf.getvalue(), w, faces)
            owner = np.asarray(got["owner"]) if got is not None else None
            if owner is not None and owner.shape != grey.shape:
                owner = None
            dt = measure(grey, owner, w)
            f_ = scale * pscale
            tr_ = {k: (dt.get(k) / f_ if dt.get(k) is not None and k != "slant" else dt.get(k)) for k in ("x_height", "stem", "hair", "slant")}
            tr_["owner"] = owner is not None
        except Exception as ex:
            tr_ = {"error": str(ex)[:80]}
        rows.append({"word": w, "vector": {k: d.get(k) for k in ("x_height", "stem", "hair", "slant", "baseline", "x_from", "stem_from")},
                     "engine": e, "traced": tr_})
    Path(out).write_text(json.dumps({"font": Path(path).stem, "truth": tr, "rows": rows}, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fonts", type=int, default=0)
    p.add_argument("--out", default="vector_dna.json")
    p.add_argument("--one", nargs=2, help=argparse.SUPPRESS)
    a = p.parse_args()
    warnings.filterwarnings("ignore")
    if a.one:
        one(*a.one); return 0
    from ..bench import fonts
    files = [f for f in fonts.font_files() if f.stem.endswith("-400")]
    if a.fonts:
        files = files[:a.fonts]
    res = []
    for f in files:
        dst = Path(a.out).with_suffix(f".{f.stem}.json")
        r = subprocess.run([sys.executable, "-m", "khatt.bench.vector_dna", "--one", str(f), str(dst)],
                           cwd=str(Path(__file__).resolve().parents[2]), capture_output=True, text=True)
        if r.returncode == 0 and dst.exists():
            res.append(json.loads(dst.read_text(encoding="utf-8")))
            print(f.stem, "ok", flush=True)
        else:
            print(f.stem, "failed", (r.stderr.strip().splitlines() or ["?"])[-1][:120], flush=True)
    err = {k: {"vector": [], "engine": [], "traced": []} for k in ("x_height", "stem", "hair")}
    for r in res:
        X = r["truth"].get("x_height")
        if not X:
            continue
        for row in r["rows"]:
            for k in err:
                tv = r["truth"].get(k)
                if tv is None:
                    continue
                for who in ("vector", "engine", "traced"):
                    v = row.get(who, {}).get(k)
                    if v is not None and v == v:
                        err[k][who].append(abs(v - tv) / X * 100)
    print("\\nmedian error, hundredths of the x-height (and the share within 5):")
    for k, d in err.items():
        line = f"  {k:9s}"
        for who in ("vector", "engine", "traced"):
            v = np.array(d[who])
            line += f"   {who}: {np.median(v):5.1f} ({(v <= 5).mean() * 100:3.0f}% within 5, n={len(v)})" if len(v) else f"   {who}: —"
        print(line)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
